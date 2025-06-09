import pandas as pd
import numpy as np
from collections import defaultdict
from ordered_set import OrderedSet

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData,Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Graph:
    def __init__(self,traj_df):
        self.traj_df = traj_df

    def get_vehicle_dict(self):
        vehicle_dict=defaultdict(OrderedSet)
        for index, row in self.traj_df.iterrows():
            if index not in vehicle_dict[row['frame']]:
                vehicle_dict[row['frame']].add(index)
        return vehicle_dict

    def load_nodes(self,vehicle_indices):
        num_vehicles=len(vehicle_indices)
        x_small_vehicles = []
        x_large_vehicles = []
        mapping_small_vehicles = {}
        mapping_large_vehicles = {}
        index_small_vehicles = []
        index_large_vehicles = []
        for i in range(num_vehicles):
            x_vehicle=np.zeros((5))
            try:
                x_vehicle[0]=self.traj_df.loc[vehicle_indices[i],'x_norm']
                x_vehicle[1] = self.traj_df.loc[vehicle_indices[i], 'y_norm']
                x_vehicle[2] = self.traj_df.loc[vehicle_indices[i], 'speed_norm']
                # x_vehicle[3] = self.traj_df.loc[vehicle_indices[i], 'distance_to_center']
                # x_vehicle[4] = self.traj_df.loc[vehicle_indices[i], 'curvature']
                x_vehicle[3]=self.traj_df.loc[vehicle_indices[i], 'lane_rel_center']
                x_vehicle[4]=self.traj_df.loc[vehicle_indices[i], 'heading_rad']
            except KeyError as e:
                print(f'name error: {e}')
            except IndexError as e:
                print(f'index error: {e}')
            if self.traj_df.loc[vehicle_indices[i],'width']<6:
                x_small_vehicles.append(x_vehicle)
                mapping_small_vehicles[self.traj_df.loc[vehicle_indices[i],'id']]=len(x_small_vehicles)-1
                index_small_vehicles.append(vehicle_indices[i])
            elif self.traj_df.loc[vehicle_indices[i],'width']>=6:
                x_large_vehicles.append(x_vehicle)
                mapping_large_vehicles[self.traj_df.loc[vehicle_indices[i],'id']] = len(x_large_vehicles)-1
                index_large_vehicles.append(vehicle_indices[i])

        if len(x_small_vehicles)>0:
            x_small_vehicles=torch.from_numpy(np.stack(x_small_vehicles)).float()
        elif len(x_small_vehicles)==0:
            x_small_vehicles=torch.empty(0,5)
        #x_small_vehicles=torch.from_numpy(np.stack(x_small_vehicles)).float()
        if len(x_large_vehicles)>0:
            x_large_vehicles=torch.from_numpy(np.stack(x_large_vehicles)).float()
        elif len(x_large_vehicles)==0:
            x_large_vehicles=torch.empty(0,5)
        #x_large_vehicles=torch.from_numpy(np.stack(x_large_vehicles)).float()
        return [x_small_vehicles,x_large_vehicles,mapping_small_vehicles,mapping_large_vehicles,index_small_vehicles,index_large_vehicles]

    def Euclidean_dst(self,vehicle_index1,vehicle_index2):
        pos1 = np.array([self.traj_df['x'][vehicle_index1], self.traj_df['y'][vehicle_index1]])
        pos2 = np.array([self.traj_df['x'][vehicle_index2], self.traj_df['y'][vehicle_index2]])
        euclidean = np.linalg.norm(pos1 - pos2)
        if euclidean == 0:
            return 1
        else:
            return euclidean

    def load_edges(self,mapping_vehicles1,mapping_vehicles2,vehicle_indices1,vehicle_indices2):
        src_list,dst_list=[],[]
        #edge_attr=[]
        edges_attr=[]
        num_vehicles1,num_vehicles2=len(vehicle_indices1),len(vehicle_indices2)
        for i in range(num_vehicles1):
            d=0
            edge_weights=[]
            for j in range(num_vehicles2):
                if abs(self.traj_df['laneId'][vehicle_indices1[i]]-self.traj_df['laneId'][vehicle_indices2[j]])<=1:
                    euclidean_dst=self.Euclidean_dst(vehicle_indices1[i],vehicle_indices2[j])
                    # 3s法则
                    current_speed = self.traj_df['speed'][vehicle_indices1[i]]
                    vehicle_id1 = self.traj_df['id'][vehicle_indices1[i]]
                    vehicle_id2 = self.traj_df['id'][vehicle_indices2[j]]
                    src = mapping_vehicles1[vehicle_id1]
                    dst = mapping_vehicles2[vehicle_id2]
                    dist_threshold = 3 * current_speed
                    if euclidean_dst < dist_threshold:
                        src_list.append(src)
                        dst_list.append(dst)
                        #edge_weight = 1 / euclidean_dst
                        #edges_attr.append(edge_weight)
                        edge_weight=current_speed/euclidean_dst
                        edge_weights.append(edge_weight)
                        d=d+1
            edge_weights=list(map(lambda x: x * d, edge_weights))
            edges_attr.extend(edge_weights)
        edges_index = torch.tensor([src_list, dst_list], dtype=torch.float)
        edges_attr = torch.tensor(edges_attr, dtype=torch.float).unsqueeze(-1)
        return edges_index, edges_attr

    def build_graph(self):
        vehicle_dict=self.get_vehicle_dict()
        graph_collection={}

        for timestamp,vehicle_indices in vehicle_dict.items():
            nodes=self.load_nodes(vehicle_indices)
            mapping_small_vehicles=nodes[2]
            mapping_large_vehicles=nodes[3]
            index_small_vehicles=nodes[4]
            index_large_vehicles=nodes[5]

            small_small_edges=self.load_edges(mapping_small_vehicles,mapping_small_vehicles,index_small_vehicles,index_small_vehicles)
            small_large_edges=self.load_edges(mapping_small_vehicles,mapping_large_vehicles,index_small_vehicles,index_large_vehicles)
            large_large_edges=self.load_edges(mapping_large_vehicles,mapping_large_vehicles,index_large_vehicles,index_large_vehicles)

            data=HeteroData()
            data['small'].x=torch.tensor(nodes[0],dtype=torch.float)
            data['large'].x=torch.tensor(nodes[1],dtype=torch.float)
            data['small', 'small'].edge_index = small_small_edges[0]
            data['small', 'small'].edge_label = torch.tensor(small_small_edges[1],dtype=torch.float)
            data['small', 'large'].edge_index = small_large_edges[0]
            data['small', 'large'].edge_label = torch.tensor(small_large_edges[1],dtype=torch.float)
            data['large', 'large'].edge_index = large_large_edges[0]
            data['large', 'large'].edge_label = torch.tensor(large_large_edges[1],dtype=torch.float)

            data.vehicle_ids=[mapping_small_vehicles,mapping_large_vehicles]
            graph_collection[timestamp]=data
        return graph_collection

class GenerateDataset(Dataset):
    def __init__(self,traj_df,hist_len,pred_len):
        self.traj_df=traj_df
        self.hist_len=hist_len
        self.pred_len=pred_len

    def graph_dict(self):
        graph=Graph(self.traj_df)
        graph_dict=graph.build_graph()
        return graph_dict

    def generate_data(self):
        graph_dict=self.graph_dict()
        seq_len=self.hist_len+self.pred_len
        data_list=[]
        for id,group_df in self.traj_df.groupby('id'):
            if len(group_df)>=seq_len:
                seq_data=[]
                for i,row in group_df.iterrows():
                    point_graph=graph_dict[row['frame']]
                    point_features=torch.tensor([row['x'],row['y'],row['speed'],row['lane_rel_center'],row['heading_rad']])
                    point_data={
                        'id': row['id'],
                        'features':point_features,
                        'graph':point_graph,
                    }
                    seq_data.append(point_data)
                for i in range(len(seq_data)-seq_len):
                    data=seq_data[i:i+seq_len]
                    data_list.append(data)
        torch.save(data_list, 'processed_data_HighD_5.pt')

class CustomDataset(Dataset):
    def __init__(self,data_path,hist_len,pred_len):
        self.data=torch.load(data_path)
        self.hist_len=hist_len
        self.pred_len=pred_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        hist_data = item[:self.hist_len]
        pred_data = item[self.hist_len:]
        x_hist_features = []
        x_hist_graphs = []
        x_pred_xy = []
        x_hist_ids = []
        for i in range(len(hist_data)):
            x_hist_features.append(torch.tensor(hist_data[i]['features']))
            x_hist_graphs.append(hist_data[i]['graph'])
            x_hist_ids.append(torch.tensor(hist_data[i]['id']))
        for i in range(len(pred_data)):
            pos_x = pred_data[i]['features'][0]
            pos_y = pred_data[i]['features'][1]
            x_pred_xy.append(torch.tensor([pos_x, pos_y]))
        return torch.stack(x_hist_features, dim=0).float(), x_hist_graphs, torch.stack(x_pred_xy,dim=0).float(), torch.stack(x_hist_ids, dim=0).float()

def collate_fn(batch):
    x_hist_features=torch.stack([item[0] for item in batch],dim=0)
    graph_batch=[item[1] for item in batch]
    x_future_xy=torch.stack([item[2] for item in batch],dim=0)
    x_hist_ids=torch.stack([item[3] for item in batch],dim=0)
    return x_hist_features,graph_batch,x_future_xy,x_hist_ids

def get_gps(batch,model):
    graph_batch=batch[1]
    last_graphs=graph_batch[-1].to(device)
    id_batch=batch[3][:,-1]
    batch_list=[]
    last_graphs_list=Batch.to_data_list(last_graphs)
    for i in range(len(last_graphs_list)):
        graph=last_graphs[i]
        id=id_batch[i]
        small_vehicle_ids=graph.vehicle_ids[0]
        large_vehicle_ids=graph.vehicle_ids[1]
        for key in small_vehicle_ids.keys():
            if id==key:
                group=0
                index=small_vehicle_ids[key]
        for key in large_vehicle_ids.keys():
            if id==key:
                group=1
                index=large_vehicle_ids[key]
        out=model(graph)
        if group==0:
            out=out['small'][index]
            batch_list.append(out)
        elif group==1:
            out=out['large'][index]
            batch_list.append(out)
    batch_out=torch.stack(batch_list, dim=0)
    return batch_out

if __name__=='__main__':
    from configs import get_args
    import os
    import torch
    from torch_geometric.loader import DataLoader
    from models_spatial import *
    args = get_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    traj_df=pd.read_csv('vehicle_traj_df_HighD_no_dist_curve.csv')
    data_path='./processed_data_HighD_5.pt'
    if not os.path.exists(data_path):
        GenerateDataset(traj_df,hist_len=args.hist_len,pred_len=args.pred_len).generate_data()
    print('finished dataset generation')

    dataset=CustomDataset(data_path,args.hist_len,args.pred_len)
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True,collate_fn=collate_fn)
    metadata=dataset[0][1][0].metadata()
    model_G = GPS(hidden_channels=args.gps_hidden, out_channels=args.gps_out, metadata=metadata,
                  num_heads=args.gps_head,num_layers=args.gps_layers, dropout=args.gps_dropout).to(device)
    for batch in dataloader:
        print(batch[0].shape)
        batch_out = get_gps(batch, model_G)
        print(batch_out.shape)
        break