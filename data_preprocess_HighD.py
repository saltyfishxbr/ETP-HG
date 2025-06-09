import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,RobustScaler

traj_path='../Datasets/highD-dataset-v1.0/data/01_tracks.csv'
origin_traj_df=pd.read_csv(traj_path)
origin_traj_df = origin_traj_df[origin_traj_df['laneId'].isin([5,6])]
origin_traj_df.to_csv('one_direction_csv',index=False)

traj_df=pd.read_csv('one_direction_csv')
lane_ids=[5,6]
traj_df = traj_df[traj_df['laneId'].isin([5,6])]

coordinates=traj_df[['x','y']].values
coord_scaler=MinMaxScaler(feature_range=(-1,1)).fit(coordinates)
traj_df[['x_norm','y_norm']]=coord_scaler.transform(coordinates)

traj_df["speed"] = np.sqrt(traj_df["xVelocity"]**2 + traj_df["yVelocity"]**2)
speed_scaler=RobustScaler().fit(traj_df[['speed']])
traj_df['speed_norm']=speed_scaler.transform(traj_df[['speed']])

total_lanes=len(lane_ids)
center_lane=np.median(lane_ids)
traj_df['lane_rel_center']=(traj_df['laneId']-center_lane)/total_lanes

traj_df['heading_rad']=np.arctan2(traj_df['xVelocity'],traj_df['yVelocity'])

traj_df=traj_df.dropna()

traj_df.to_csv('vehicle_traj_df_HighD_no_dist_curve.csv',index=False)