import argparse


def get_args():
    parser=argparse.ArgumentParser(description='Heterogeneous graph enhanced trajectory prediction')

    #add params here
    data_group=parser.add_argument_group('Data')
    # data_group.add_argument('--traj_path',default='./Datasets/TGSIM/I-294/I-294-L1-Run_1_Trajectories.csv')
    # data_group.add_argument('--data_path',default='./processed_data_TGSIM.pt')
    data_group.add_argument('--hist_len',default=75,type=int)
    data_group.add_argument('--pred_len',default=125,type=int)
    data_group.add_argument('--x_feature_dim',default=6,type=int)

    #training params here
    train_group=parser.add_argument_group('Training')
    train_group.add_argument('--batch_size',default=64,type=int)
    train_group.add_argument('--epochs',default=100,type=int)
    train_group.add_argument('--num_workers',default=4,type=int)

    train_group.add_argument('--gps_hidden',default=128,type=int)
    train_group.add_argument('--gps_out',default=128,type=int)
    train_group.add_argument('--gps_head',default=8,type=int)
    train_group.add_argument('--gps_layers',default=4,type=int)
    train_group.add_argument('--gps_dropout',default=0.1,type=float)
    train_group.add_argument('--gps_lr',default=1e-4,type=float)
    train_group.add_argument('--gps_weight_decay',default=1e-5,type=float)
    train_group.add_argument('--gps_clip',default=1.0,type=float)

    train_group.add_argument('--transformer_hidden',default=128,type=int)
    train_group.add_argument('--transformer_head',default=8,type=int)
    train_group.add_argument('--transformer_layers',default=8,type=int)
    train_group.add_argument('--transformer_lr',default=1e-4,type=float)
    train_group.add_argument('--transformer_weight_decay',default=1e-5,type=float)
    train_group.add_argument('--transformer_clip', default=1.0, type=float)

    #testing prams here
    test_group=parser.add_argument_group('Testing')

    args=parser.parse_args()
    return args