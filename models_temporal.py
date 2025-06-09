import torch
from torch import nn
from torch.nn import TransformerEncoderLayer,TransformerEncoder,TransformerDecoderLayer,TransformerDecoder

from configs import get_args
#from datasets_NGSIM import *
from datasets_HighD import *
from models_spatial import *

args=get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hist_len=args.hist_len
        self.pred_len=args.pred_len
        self.transformer_hidden=args.transformer_hidden
        self.transformer_head=args.transformer_head
        self.transformer_layers=args.transformer_layers

        encoder_layer=TransformerEncoderLayer(
            d_model=self.transformer_hidden,
            nhead=self.transformer_head,
            dim_feedforward=self.transformer_hidden
        )
        self.encoder=TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.transformer_layers,
        )

        #NGSIM
        # self.rel_PE=self.relative_motion_encoding
        # self.lane_center_proj=nn.Linear(1,self.transformer_hidden//4)
        # self.rel_pe = nn.Linear(3, self.transformer_hidden // 2)
        # self.angle_proj=nn.Linear(1,self.transformer_hidden//4)

        #TGSIM
        self.rel_PE = self.relative_motion_encoding
        # self.dist_proj=nn.Linear(1,self.transformer_hidden//8)
        # self.curve_proj=nn.Linear(1,self.transformer_hidden//8)
        self.rel_pe=nn.Linear(3,self.transformer_hidden//2)
        self.lane_center_proj = nn.Linear(1, self.transformer_hidden // 4)
        self.angle_proj=nn.Linear(1,self.transformer_hidden // 4)

    def relative_motion_encoding(self,pos,speed):
        # pos:[b,s,2],speed:[b,s,1]
        delta_pos = torch.diff(pos, dim=1, prepend=pos[:, 0:1, :])
        motion_feat = torch.cat([delta_pos, speed], dim=-1)
        return self.rel_pe(motion_feat)

    def forward(self,x):
        xy=x[:,:,0:2]
        speed=x[:,:,2].unsqueeze(-1)
        rel_xy=self.rel_PE(xy,speed)

        #NGSIM: x,y,v,lane embedding
        # lane_center_proj = self.lane_center_proj(x[:, :, 3].unsqueeze(-1))
        # angle_proj = self.angle_proj(x[:, :, 4].unsqueeze(-1))
        # src = torch.cat([rel_xy, lane_center_proj,angle_proj], dim=-1)

        # TGSIM: x,y,v,dist,curve,lane embedding
        # dist_proj=self.dist_proj(x[:,:,3].unsqueeze(-1))
        # curve_proj=self.curve_proj(x[:,:,4].unsqueeze(-1))
        lane_center_proj=self.lane_center_proj(x[:,:,3].unsqueeze(-1))
        angle_proj = self.angle_proj(x[:, :, 4].unsqueeze(-1))
        src=torch.cat([rel_xy,lane_center_proj,angle_proj],dim=-1)

        src=src.permute(1,0,2)
        memory=self.encoder(src)
        return memory.permute(1,0,2)

class SpatioTemporalFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.transformer_hidden=args.transformer_hidden
        self.gps_hidden=args.gps_hidden
        self.transformer_head=args.transformer_head
        self.query_proj=nn.Linear(self.transformer_hidden,self.transformer_hidden)
        self.key_proj=nn.Linear(self.gps_hidden,self.transformer_hidden)
        self.multihead_attn=nn.MultiheadAttention(self.transformer_hidden,self.transformer_head,batch_first=True)

    def forward(self,memory,gnn_feats):
        keys=self.key_proj(gnn_feats).unsqueeze(1)#[b,1,hidden_dim]
        keys=keys.repeat(1,memory.size(1),1)#[b,hist_len,hidden_dim]
        fused_feats,_=self.multihead_attn(
            query=self.query_proj(memory),
            key=keys,
            value=keys
        )
        return fused_feats#[b,hist_len,hidden_dim]

class TemporalDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len=args.pred_len
        self.transformer_hidden=args.transformer_hidden
        self.transformer_head=args.transformer_head

        decoder_layer=TransformerDecoderLayer(
            d_model=self.transformer_hidden,
            nhead=self.transformer_head,
            dim_feedforward=self.transformer_hidden
        )

        self.decoder=TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.transformer_head,
        )
        self.pos_embed = nn.Embedding(self.pred_len, self.transformer_hidden)
        self.pred_head=nn.Linear(self.transformer_hidden,2)

    def generate_target_query(self,batch_size):
        position=torch.arange(self.pred_len,device=device)
        target_query = self.pos_embed(position)
        target_query = target_query.unsqueeze(1).repeat(1,batch_size,1)
        return target_query

    def forward(self,memory):
        target_query=self.generate_target_query(memory.shape[0])
        output=self.decoder(target_query,memory.permute(1,0,2))
        pred_points=self.pred_head(output).permute(1,0,2)
        return pred_points


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, args,metadata):
        super().__init__()
        self.traj_encoder=TemporalEncoder(args)
        self.gnn_encoder=GPS(hidden_channels=args.gps_hidden,out_channels=args.transformer_hidden,
                             metadata=metadata,num_heads=args.gps_head,num_layers=args.gps_layers,dropout=args.gps_dropout)
        self.fusion=SpatioTemporalFusion(args)
        self.decoder=TemporalDecoder(args)

    def forward(self,batch):
        x_features=batch[0].float().to(device)
        memory=self.traj_encoder(x_features)
        gnn_feats=get_gps(batch,self.gnn_encoder)
        fused_feats=self.fusion(memory,gnn_feats)
        fut_pred=self.decoder(fused_feats)
        return fut_pred