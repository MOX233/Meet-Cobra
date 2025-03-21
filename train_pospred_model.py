import os # Configure which GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import random
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair, generate_dft_codebook
from utils.NN_utils import prepare_dataset, PositionPredictionModel, train_pospred_model
from utils.options import args_parser
from utils.sumo_utils import read_trajectoryInfo_timeindex
from utils.mox_utils import setup_seed, get_prepared_dataset


if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)

    # freq = 5.9e9
    # DS_start, DS_end = 500, 700
    freq = 28e9
    DS_start, DS_end = 300, 700
    preprocess_mode = 1
    M_r, N_bs, M_t = 8, 4, 64
    P_t = 1e-1
    P_noise = 1e-14
    n_pilot = 4
    gpu = 7
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device)

    prepared_dataset_filename, data_torch, veh_h_torch, veh_pos_torch, best_beam_pair_index_torch \
        = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, device, P_t, P_noise)
    
    # pretrained_model_path='./models/best_model_acc89.262%.pth'
    pretrained_model_path = None
    result_save_dir = os.path.join('./NN_result',prepared_dataset_filename)
    plt_save_dir = os.path.join(result_save_dir,'plots')
    model_save_dir = os.path.join(result_save_dir,'models')
    if os.path.exists(result_save_dir) == False:
        os.mkdir(result_save_dir)
        os.mkdir(plt_save_dir)
        os.mkdir(model_save_dir)
    num_epochs = 100
    # 运行训练
    model, train_loss_list, train_rmse_list, train_mae_list, val_loss_list, val_rmse_list, val_mae_list = \
        train_pospred_model(num_epochs, device, data_torch, veh_h_torch, veh_pos_torch, best_beam_pair_index_torch, M_t, M_r, pretrained_model_path, model_save_dir)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(train_rmse_list, label='train RMSE')
    plt.plot(val_rmse_list, label='val RMSE')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(plt_save_dir,f'pospred_dimIn{model.feature_input_dim}_valRMSE{min(val_rmse_list):.2f}.png'))
    print(os.path.join(plt_save_dir,f'pospred_dimIn{model.feature_input_dim}_valRMSE{min(val_rmse_list):.2f}.png'))