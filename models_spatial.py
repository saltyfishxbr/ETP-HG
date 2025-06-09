import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv,HeteroConv,Linear,GINEConv
from torch_geometric.nn.inits import reset

def init_weigths(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias,0)
    elif isinstance(m,HGTConv):
        for weight in [m.query_weights,m.key_weights,m.value_weights]:
            nn.init.xavier_uniform_(weight)
        if hasattr(m,'attn_weights'):
            nn.init.normal_(m.attn_weights,mean=0,std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class HeteroGINE(nn.Module):
    def __init__(self,hidden_channels,metadata,num_layers=1):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GINEConv(
                    nn.Sequential(
                        Linear(hidden_channels, hidden_channels),
                        nn.LayerNorm(hidden_channels),
                        nn.ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        nn.LayerNorm(hidden_channels),
                        nn.ReLU(),
                    ),
                    edge_dim=1
                ) for edge_type in self.edge_types
            })
            self.convs.append(conv)

        self.lin_dict_post = nn.ModuleDict({
            node_type: nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ) for node_type in self.node_types
        })

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        reset(self.lin_dict_post)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        x_dict = {node_type: self.lin_dict_post[node_type](x) for node_type, x in x_dict.items()}

        return x_dict

class HGT(nn.Module):
    def __init__(self,hidden_channels,metadata, num_heads=4, num_layers=1):
        super().__init__()
        self.node_types = metadata[0]

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.lin_dict_post = nn.ModuleDict({
            node_type: nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ) for node_type in self.node_types
        })

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        reset(self.lin_dict_post)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            #x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {node_type: self.lin_dict_post[node_type](x) for node_type, x in x_dict.items()}
        return x_dict

class HeteroGPSConv(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_heads=4, dropout=0.1):
        super().__init__()

        self.gin_conv = HeteroGINE(hidden_channels,metadata, num_layers=1)
        self.hgt_conv = HGT(hidden_channels,metadata, num_heads=num_heads,num_layers=1)
        self.dropout = dropout
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        self.norm1=nn.ModuleDict({
            node_type:nn.LayerNorm(hidden_channels) for node_type in self.node_types
        })
        self.norm2 = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_channels) for node_type in self.node_types
        })

        self.lin_dict_post = nn.ModuleDict({
            node_type: nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            ) for node_type in self.node_types
        })

        self.lin_dict_out=nn.ModuleDict({
            node_type:nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ) for node_type in self.node_types
        })

    def reset_parameters(self):
        self.gin_conv.reset_parameters()
        self.hgt_conv.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        reset(self.lin_dict_post)
        reset(self.lin_dict_out)

    def hetero_add(self, h1, h2):
        h = h1
        for key in h1:
            h[key] = h1[key] + h2[key]
        return h

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # start=time.time()
        h_gine = self.gin_conv(x_dict, edge_index_dict, edge_attr_dict)
        for key in h_gine:
            h_gine[key] = F.dropout(h_gine[key], p=self.dropout, training=self.training)
        h_gine = self.hetero_add(h_gine, x_dict)
        h_gine = {node_type: self.norm1[node_type](x) for node_type, x in h_gine.items()}
        # end_gine = time.time()
        # print('gine: ',end_gine-start)

        h_hgt = self.hgt_conv(x_dict, edge_index_dict)
        for key in h_hgt:
            h_hgt[key] = F.dropout(h_hgt[key], p=self.dropout, training=self.training)
        h_hgt = self.hetero_add(h_hgt, x_dict)
        h_hgt = {node_type: self.norm2[node_type](x) for node_type,x  in h_hgt.items()}
        # end_hgt=time.time()
        # print('hgt: ',end_hgt-end_gine)

        h_hgtgine = self.hetero_add(h_hgt, h_gine)
        h_hgtginemlp={node_type: self.lin_dict_post[node_type](x) for node_type, x in h_hgtgine.items()}
        out=self.hetero_add(h_hgtginemlp,h_hgtgine)
        # end_mlp=time.time()
        # print('mlp: ',end_mlp-end_hgt)

        out={node_type: self.lin_dict_out[node_type](x) for node_type, x in out.items()}
        return out

class GPS(nn.Module):
    def __init__(self, hidden_channels, out_channels,metadata, num_heads, num_layers, dropout=0.1):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.node_types = metadata[0]
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.lin_dict_pre = nn.ModuleDict({
            node_type:nn.Sequential(
                Linear(-1, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            )  for node_type in self.node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroGPSConv(hidden_channels, metadata, num_heads, dropout)
            self.convs.append(conv)

        self.lin_dict_post=nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_channels,hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
                nn.Tanh()
            ) for node_type in self.node_types
        })

    def forward(self, x):
        x_dict=x.x_dict
        edge_index_dict=x.edge_index_dict
        edge_attr_dict=x.edge_label_dict
        edge_index_dict={
            k:v.to(torch.int64) for k,v in edge_index_dict.items()
        }
        x_dict = {node_type: self.lin_dict_pre[node_type](x) for node_type, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        x_dict={node_type: self.lin_dict_post[node_type](x) for node_type, x in x_dict.items()}
        return x_dict

if __name__ == '__main__':
    from torch_geometric.data import HeteroData

    dummy_data = HeteroData()

    dummy_data['small'].x = torch.randn(3, 6)  # (3, 6)
    dummy_data['large'].x = torch.randn(2, 6)  # (2, 6)

    dummy_data['small', 'small'].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    dummy_data['small', 'small'].edge_label = torch.randn(2, 1)  # (2, 1)

    dummy_data['small', 'large'].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    dummy_data['small', 'large'].edge_label = torch.randn(2, 1)  # (2, 1)

    dummy_data['large', 'large'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    dummy_data['large', 'large'].edge_label = torch.randn(1, 1)  # (1, 1)

    dummy_model = GPS(hidden_channels=64, out_channels=8, metadata=dummy_data.metadata(),num_heads=4, num_layers=2,
                      dropout=0.1)

    out = dummy_model(dummy_data)
    print(out)