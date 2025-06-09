import numpy as np

from configs import get_args
import torch
from pickle import load

args=get_args()

# def calculate_ade(pred_scaled,y_scaled,scaler_path=args.scaler_path):
#     scaler=load(open(scaler_path,'rb'))
#
#     if isinstance(pred_scaled,torch.Tensor):
#         pred_scaled=pred_scaled.detach().cpu().numpy()
#     if isinstance(y_scaled,torch.Tensor):
#         y_scaled=y_scaled.detach().cpu().numpy()
#
#     batch_size=args.batch_size
#     pred_len=args.pred_len
#
#     pred_fake_features=np.zeros((batch_size*pred_len,2))
#     pred_fake_features[:,:2]=pred_scaled.reshape(-1,2)
#     pred_original=scaler.inverse_transform(pred_fake_features)[:,:2].reshape(batch_size,pred_len,2)
#
#     y_fake_features=np.zeros((batch_size*pred_len,2))
#     y_fake_features[:,:2]=y_scaled.reshape(-1,2)
#     y_original=scaler.inverse_transform(y_fake_features)[:,:2].reshape(batch_size,pred_len,2)
#
#     ade=np.mean(np.linalg.norm(pred_original-y_original,axis=2))
#     return ade

def calculate_ade_orig(x_pred,x_true):
    displacement=torch.norm(x_pred-x_true,p=2,dim=-1)
    ade=torch.mean(displacement)
    return ade

def calculate_trajectory_metrics(x_true,x_pred):
    assert x_true.shape==x_pred.shape
    metrics={}
    time_indices={1:24,2:49,3:74,4:99,5:124}

    #TGSIM
    # scaler=load(open('scaler_minmax_xy.pkl','rb'))
    # batch_size=args.batch_size
    # pred_len=args.pred_len
    # x_pred_scaled = np.zeros((batch_size * pred_len, 2))
    # x_pred_scaled[:, :2] = x_pred.reshape(-1, 2)
    # x_pred = scaler.inverse_transform(x_pred_scaled)[:, :2].reshape(batch_size, pred_len, 2)
    # x_pred=torch.from_numpy(x_pred)
    # x_true_scaled = np.zeros((batch_size * pred_len, 2))
    # x_true_scaled[:, :2] = x_true.reshape(-1, 2)
    # x_true = scaler.inverse_transform(x_true_scaled)[:, :2].reshape(batch_size, pred_len, 2)
    # x_true=torch.from_numpy(x_true)
    for t_sec,idx in time_indices.items():
        pred=x_pred[:,idx,:]
        true=x_true[:,idx,:]

        squared_error=torch.sum((pred-true)**2,dim=1)

        rmse=torch.sqrt(torch.mean(squared_error))
        metrics[f'rmse_{t_sec}s']=rmse.item()

    pointwise_errors=torch.norm(x_true-x_pred,p=2,dim=2)
    ade_per_sample=torch.mean(pointwise_errors,dim=1)
    metrics['ade']=torch.mean(ade_per_sample).item()

    final_errors=torch.norm(x_pred[:,49,:]-x_true[:,49,:],p=2,dim=1)
    metrics['fde']=torch.mean(final_errors).item()

    return metrics