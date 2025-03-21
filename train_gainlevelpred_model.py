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
from utils.NN_utils import prepare_dataset, BestGainPredictionModel, train_gainpred_model, train_gainlevelpred_model
from utils.options import args_parser
from utils.sumo_utils import read_trajectoryInfo_timeindex
from utils.mox_utils import setup_seed, get_prepared_dataset, get_save_dirs, split_string, save_log
from utils.plot_utils import plot_gainlevelpred

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
    
    result_save_dir, plt_save_dir, model_save_dir, log_save_dir = get_save_dirs(prepared_dataset_filename)
    
    num_epochs = 1
    pretrained_model_path = None
    # 运行训练
    model, train_loss_list, train_acc_list, train_mae_list, train_mse_list, \
    train_bestBS_mae_list, train_bestBS_mse_list, train_bestBS_acc_list, \
    val_loss_list, val_acc_list, val_mae_list, val_mse_list, \
    val_bestBS_mae_list, val_bestBS_mse_list, val_bestBS_acc_list = \
        train_gainlevelpred_model(num_epochs, device, data_torch, veh_h_torch, best_beam_pair_index_torch, M_t, M_r, pretrained_model_path, model_save_dir)
    train_result_name_list = split_string("model, train_loss_list, train_acc_list, train_mae_list, train_mse_list, \
    train_bestBS_mae_list, train_bestBS_mse_list, train_bestBS_acc_list, \
    val_loss_list, val_acc_list, val_mae_list, val_mse_list, \
    val_bestBS_mae_list, val_bestBS_mse_list, val_bestBS_acc_list")
    save_name = f"gainlevelpred_dimIn{model.feature_input_dim}Out{model.num_dBlevel}_valAcc{max(val_acc_list):.2f}%_BBSMAE{min(val_bestBS_mae_list):.2f}"
    
    torch.save(model.state_dict(), os.path.join(model_save_dir, save_name+'.pth'))
    
    log_dict = save_log(locals(), train_result_name_list, os.path.join(log_save_dir,save_name+'.pkl'))
    
    plot_gainlevelpred(os.path.join(plt_save_dir,save_name+'.png'), log_dict)