import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import torch
from torchsummary import summary

from models_temporal import *
from configs import get_args
from metrics import *
from datasets_HighD import *
from torch_geometric.loader import DataLoader

args=get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def visualize_trajectories(batch,x_true,x_pred,metrics):
    vehicle_ids=batch[3][:,-1]

    fig,axes=plt.subplots(2,3,figsize=(12,120))
    axes=axes.ravel()

    for i in range(6):
        ax=axes[i]
        vehicle_id=vehicle_ids[i]
        length,width=get_dimensions(traj_df,vehicle_id)

        true_traj=x_true[i]
        pred_traj=x_pred[i]
        start_x,start_y=true_traj[0,0].item(),true_traj[0,1].item()
        end_true_x,end_true_y=true_traj[-1,0].item(),true_traj[-1,1].item()
        end_pred_x,end_pred_y=pred_traj[-1,0].item(),pred_traj[-1,1].item()

        ax.plot(true_traj[:,1],true_traj[:,0],'b-',marker='o',markersize=3,label='True Trajectory')
        ax.plot(pred_traj[:,1],pred_traj[:,0],'r--',marker='x',markersize=3,label='Predicted Trajectory')

        rect_start=patches.Rectangle(
            (start_y-length/2,start_x-width/2),
            length,width,
            linewidth=1,edgecolor='g',facecolor='none'
        )
        ax.add_patch(rect_start)
        rect_end_true=patches.Rectangle(
            (end_true_y-length/2,end_true_x-width/2),
            length,width,
            linewidth=1,edgecolor='b',facecolor='none'
        )
        ax.add_patch(rect_end_true)
        rect_end_pred=patches.Rectangle(
            (end_pred_y-length/2,end_pred_x-width/2),
            length,width,
            linewidth=1,edgecolor='r',facecolor='none'
        )
        ax.add_patch(rect_end_pred)
        ax.set_aspect('equal')
        ax.legend()
    fig.suptitle(metrics)
    plt.tight_layout()
    plt.show()

def get_dimensions(traj_df,vehicle_id):
        for index,row in traj_df.iterrows():
            if traj_df.loc[index,'id']==vehicle_id:
                return row['height'],row['width']

traj_path='vehicle_traj_df_HighD_no_dist_curve.csv'
data_path= 'processed_data_HighD_5.pt'
model_path='checkpoint.pt'
traj_df=pd.read_csv(traj_path)
dataset=CustomDataset(data_path,args.hist_len,args.pred_len)
metadata=dataset[0][1][0].metadata()
model=SpatioTemporalTransformer(args,metadata)
model.load_state_dict(torch.load(model_path,map_location=device))
model.to(device)
model.eval()
dataloader=DataLoader(dataset,batch_size=6,shuffle=True)
#rmse1,rmse2,rmse3,rmse4,rmse5,ade,fde=[],[],[],[],[],[],[]
val_batch = next(iter(dataloader))
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# total_params = count_parameters(model)
# print(f"Total trainable parameters: {total_params}")

with torch.no_grad():
    val_x_true_xy = val_batch[2].float().to(device)
    val_x_pred_xy = model(val_batch).to(device)
    val_x_start_xy = val_batch[0][:, -1, 0:2].float().to(device).unsqueeze(1)
    val_x_pred_xy = val_x_pred_xy + val_x_start_xy
    metric=calculate_trajectory_metrics(val_x_pred_xy,val_x_true_xy)
    print(metric)

val_x_true_xy=val_x_true_xy.detach().cpu().numpy()
val_x_pred_xy=val_x_pred_xy.detach().cpu().numpy()
visualize_trajectories(val_batch,val_x_true_xy,val_x_pred_xy,str(metric))