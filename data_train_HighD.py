import os
import time
import numpy as np
import pandas as pd

from datasets_HighD import *
from metrics import *
from configs import get_args
from models_spatial import *
from models_temporal import *

import torch
from torch.utils.data import random_split
from torch.optim.lr_scheduler import OneCycleLR,LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

time=time.localtime()
writer=SummaryWriter(log_dir=f'../runs/HighD/{time}')

args = get_args()

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

traj_path='vehicle_traj_df_HighD_no_dist_curve.csv'
data_path= 'processed_data_HighD_5.pt'
if not os.path.exists(traj_path):
    print('run datasets_xx.py first!')
    exit()

traj_df=pd.read_csv(traj_path)
if not os.path.exists(data_path):
    GenerateDataset(traj_df,hist_len=args.hist_len,pred_len=args.pred_len).generate_data()

dataset=CustomDataset(data_path,args.hist_len,args.pred_len)
def prepare_loaders(full_datset,batch_size,split_ratios=[0.8,0.1,0.1]):
    total=len(full_datset)
    train_len=int(total*split_ratios[0])
    val_len=int(total*split_ratios[1])
    test_len=total-train_len-val_len

    train_dataset,val_dataset,test_dataset=random_split(full_datset,[train_len,val_len,test_len])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    return train_dataloader,val_dataloader,test_dataloader

train_dataloader,val_dataloader,test_dataloader=prepare_loaders(dataset,args.batch_size,split_ratios=[0.8,0.1,0.1])

metadata=dataset[0][1][0].metadata()
model=SpatioTemporalTransformer(args,metadata).to(device)

optimizer=torch.optim.Adam([
    {'params':model.gnn_encoder.parameters(), 'lr':args.gps_lr},
    {'params':model.traj_encoder.parameters(), 'lr':args.transformer_lr},
    {'params':model.fusion.parameters(), 'lr':args.transformer_lr},
    {'params':model.decoder.parameters(), 'lr':args.transformer_lr},
],weight_decay=1e-5)

criterion=nn.MSELoss()
#criterion=nn.HuberLoss(delta=1)
num_epochs=args.epochs

def train_with_interval_validation(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        validate_every,
):
    model.train()
    global_step=0

    for batch_idx, batch in enumerate(train_dataloader):
        x_true_xy = batch[2].float().to(device)
        x_pred_xy=model(batch).to(device)
        x_start_xy = batch[0][:, -1, 0:2].float().to(device).unsqueeze(1)
        x_pred_xy=x_pred_xy+x_start_xy
        loss=criterion(x_pred_xy,x_true_xy)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step+=1
        writer.add_scalar('Train/loss',loss.item(),global_step)

        if batch_idx%validate_every==0:
            model.eval()
            val_batch=next(iter(val_dataloader))
            with torch.no_grad():
                val_x_true_xy = val_batch[2].float().to(device)
                val_x_pred_xy = model(val_batch).to(device)
                val_x_start_xy = val_batch[0][:, -1, 0:2].float().to(device).unsqueeze(1)
                val_x_pred_xy = val_x_pred_xy + val_x_start_xy
                val_loss=criterion(val_x_pred_xy,val_x_true_xy)

            writer.add_scalar('Validation/loss',val_loss.item(),global_step)
            metrics_dict=calculate_trajectory_metrics(val_x_true_xy.cpu(),val_x_pred_xy.cpu())
            for name,value in metrics_dict.items():
                writer.add_scalar(f'Metrics/{name}',value,global_step)

            model.train()
        if global_step%1e2==0:
            torch.save(model.state_dict(),f'../runs/HighD/{global_step}.pt')

for epoch in range(num_epochs):
    train_with_interval_validation(model,train_dataloader,val_dataloader,criterion,optimizer,validate_every=8)

writer.close()