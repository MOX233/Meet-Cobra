import os # Configure which GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import random
import torch

from utils.NN_utils import BeamPredictionModel, train_gainpred_lstm_model
from utils.options import args_parser
from utils.mox_utils import setup_seed, get_save_dirs, save_NN_results
from utils.plot_utils import plot_record_metrics
from utils.data_utils import get_prepared_dataset, prepare_dataset, augment_dataset
from utils.beam_utils import generate_dft_codebook, beamPairId_to_beamIdPair

if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)
    freq = 28e9
    DS_start, DS_end = 200, 800
    preprocess_mode = 2
    look_ahead_len = 10
    n_pilot = 16
    M_r, N_bs, M_t = 8, 4, 64
    P_t = 1e-1
    P_noise = 1e-14 # -174dBm/Hz * 1.8MHz = 7.165929069962946e-15 W
    gpu = 3
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device)

    prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np \
        = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise, look_ahead_len)
    
    DFT_tx = generate_dft_codebook(M_t)
    DFT_rx = generate_dft_codebook(M_r)
    beamPairId = best_beam_pair_index_np[:,-1,...]
    beamIdPair = beamPairId_to_beamIdPair(beamPairId,M_t,M_r)
    num_sample, _, _, num_bs, _ = veh_h_np.shape
    channel = veh_h_np[:,-1,...]
    g_opt = np.zeros((num_sample,num_bs)).astype(np.float32)
    for veh in range(num_sample):
        for bs in range(num_bs):
            g_opt[veh,bs] = 1/np.sqrt(M_t*M_r)*np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,beamIdPair[veh,bs,0]]).T.conjugate(),DFT_rx[:,beamIdPair[veh,bs,1]]))
            g_opt[veh,bs] = 20 * np.log10(g_opt[veh,bs] + 1e-9) 
    g_opt_normalized = g_opt / 20 + 7
    gain_labels_np = g_opt_normalized[:,np.newaxis,...]
    data_torch, gain_labels, lengths = augment_dataset(
        data_np, gain_labels_np, look_ahead_len, augment_dataset_ratio=2)
    
    num_epochs =  100
    # 运行训练
    print('start training')
    best_model_weights, record_metrics = \
        train_gainpred_lstm_model(num_epochs, device, data_torch, lengths, gain_labels, M_t, M_r, pos_in_data=(preprocess_mode==2))
    save_file_name = f"gainpred_lstm_valMae{min(record_metrics['val_mae']):.2f}dB" \
                + time.strftime('_%Y-%m-%d_%H:%M:%S', time.gmtime(time.time() + 8 * 3600))            
    
    save_NN_results(prepared_dataset_filename, save_file_name, best_model_weights, record_metrics)