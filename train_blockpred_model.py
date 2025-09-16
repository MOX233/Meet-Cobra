import os # Configure which GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import random
import torch

from utils.NN_utils import BeamPredictionModel, train_blockpred_model
from utils.options import args_parser
from utils.mox_utils import setup_seed, get_save_dirs, save_NN_results
from utils.plot_utils import plot_record_metrics
from utils.data_utils import get_prepared_dataset, prepare_dataset

if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)
    freq = 28e9
    DS_start, DS_end = 200, 800
    preprocess_mode = 0
    look_ahead_len = 1
    n_pilot = 16
    M_r, N_bs, M_t = 8, 4, 64
    P_t = 1e-1
    P_noise = 1e-14 # -174dBm/Hz * 1.8MHz = 7.165929069962946e-15 W
    gpu = 5
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device)

    prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np \
        = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise, look_ahead_len)
    # data_torch = torch.tensor(data_np) 
    # veh_h_torch = torch.tensor(veh_h_np) 
    data_torch = torch.tensor(data_np[:,0,...]) 
    veh_h_torch = torch.tensor(veh_h_np[:,-1,...]) 
    block_labels = (veh_h_torch[:,0,:,0]==0).long()
    
    num_epochs =  50
    # 运行训练
    print('start training')
    best_model_weights, record_metrics = \
        train_blockpred_model(num_epochs, device, data_torch, block_labels, M_t, M_r, pos_in_data=(preprocess_mode==2))
    save_file_name = f"blockpred_valAcc{max(record_metrics['val_acc'])*100:.2f}%" \
                + time.strftime('_%Y-%m-%d_%H:%M:%S', time.gmtime(time.time() + 8 * 3600))            
    
    save_NN_results(prepared_dataset_filename, save_file_name, best_model_weights, record_metrics)