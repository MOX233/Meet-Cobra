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
from utils.NN_utils import BeamPredictionModel, train_beampred_model
from utils.options import args_parser
from utils.sumo_utils import read_trajectoryInfo_timeindex
from utils.mox_utils import setup_seed, get_save_dirs, split_string, save_log, np2torch
from utils.plot_utils import plot_beampred
from utils.data_utils import get_prepared_dataset, prepare_dataset

def cal_beamgain(channel, bp_idx, opt_bp_idx, DFT_tx, DFT_rx):
    # channel.shape = (n_car, M_r, n_bs, M_t)
    # opt_bp_idx.shape = bp_idx.shape = (n_car, n_bs, 2)
    # DFT_tx.shape = (M_t, M_t)
    # DFT_rx.shape = (M_r, M_r)
    n_car, M_r, n_bs, M_t = channel.shape
    beamgain_list = []
    beamgain_opt_list = []
    g_list = []
    g_opt_list = []
    ratio_list = []
    # best_beam_pair_index for N_bs BSs
    for veh in range(n_car):
        for bs in range(n_bs):
            g = np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,bp_idx[veh,bs,0]]).T.conjugate(),DFT_rx[:,bp_idx[veh,bs,1]]))
            g_opt = np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,opt_bp_idx[veh,bs,0]]).T.conjugate(),DFT_rx[:,opt_bp_idx[veh,bs,1]]))
            if g_opt==0:
                continue
            beam_gain = g / np.sqrt(M_r*M_t) / np.abs(channel[veh,:,bs,:]).mean()
            beam_gain_opt = g_opt / np.sqrt(M_r*M_t) / np.abs(channel[veh,:,bs,:]).mean()
            beamgain_list.append(beam_gain)
            beamgain_opt_list.append(beam_gain_opt)
            g_list.append(g)
            g_opt_list.append(g_opt)
            ratio_list.append(g/g_opt)
    # return g_list, g_opt_list, ratio_list
    ratio_mean = np.array(ratio_list).mean()
    return ratio_mean, g_list, g_opt_list, ratio_list, beamgain_list, beamgain_opt_list

def cal_beamgain_topK(K, channel, bp_idx, opt_bp_idx, DFT_tx, DFT_rx):
    # channel.shape = (n_car, M_r, n_bs, M_t)
    # opt_bp_idx.shape = (n_car, n_bs, 2)
    # bp_idx.shape = (n_car, n_bs, K, 2)
    # DFT_tx.shape = (M_t, M_t)
    # DFT_rx.shape = (M_r, M_r)
    n_car, M_r, n_bs, M_t = channel.shape
    beamgain_list = []
    beamgain_opt_list = []
    g_list = []
    g_opt_list = []
    ratio_list = []
    # best_beam_pair_index for N_bs BSs
    for veh in range(n_car):
        for bs in range(n_bs):
            g = 0
            for k in range(K):
                g = max(g,np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,bp_idx[veh,bs,k,0]]).T.conjugate(),DFT_rx[:,bp_idx[veh,bs,k,1]])))
            g_opt = np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,opt_bp_idx[veh,bs,0]]).T.conjugate(),DFT_rx[:,opt_bp_idx[veh,bs,1]]))
            if g_opt==0:
                continue
            beam_gain = g / np.sqrt(M_r*M_t) / np.abs(channel[veh,:,bs,:]).mean()
            beam_gain_opt = g_opt / np.sqrt(M_r*M_t) / np.abs(channel[veh,:,bs,:]).mean()
            beamgain_list.append(beam_gain)
            beamgain_opt_list.append(beam_gain_opt)
            g_list.append(g)
            g_opt_list.append(g_opt)
            ratio_list.append(g/g_opt)
    # return g_list, g_opt_list, ratio_list
    ratio_mean = np.array(ratio_list).mean()
    return ratio_mean, g_list, g_opt_list, ratio_list, beamgain_list, beamgain_opt_list

if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(20)

    freq = 28e9
    DS_start, DS_end = 700, 800
    preprocess_mode = 0
    M_r, N_bs, M_t = 8, 4, 64
    P_t = 1e-1
    P_noise = 1e-14
    n_pilot = 16
    gpu = 7
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    saved_model_path = f"/home/ubuntu/niulab/JCBT_AEPHORA/NN_result/300_700_3Dbeam_tx(1,64)_rx(1,8)_freq2.8e+10_Np4_mode{preprocess_mode}/models/beampred_dimIn64_valAcc85.85%.pth"
    
    print('Using device: ', device)

    prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np \
        = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise)
    data_torch = np2torch(data_np,device) 
    veh_h_torch = np2torch(veh_h_np,device) 
    veh_pos_torch = np2torch(veh_pos_np,device) 
    best_beam_pair_index_torch = np2torch(best_beam_pair_index_np,device)
    
    num_beampair = M_t*M_r
    
    # 预处理数据：将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    data = BeamPredictionModel.preprocess_input(None,data_torch)
    # data = torch.concatenate(( (20*torch.log10(torch.abs(data_torch)+1e-12))/20+7, torch.angle(data_torch)),axis=-1)
    labels = best_beam_pair_index_torch.to(device)
    num_bs = labels.shape[-1]
    feature_input_dim = data.shape[-1]
    
    # 初始化模型
    model = BeamPredictionModel(feature_input_dim, num_bs, num_beampair).to(device)
    
    # 划分训练集和验证集（示例比例）
    # dataset = TensorDataset(data, labels)
    
    model.load_state_dict(torch.load(saved_model_path, map_location=device, weights_only=True))

    channel = veh_h_torch.detach().cpu().numpy()
    DFT_tx = generate_dft_codebook(M_t)
    DFT_rx = generate_dft_codebook(M_r)
    beamPairId = best_beam_pair_index_torch.detach().cpu().numpy()
    beamIdPair = beamPairId_to_beamIdPair(beamPairId,M_t,M_r)
    opt_bp_idx = beamIdPair

    bp_idx = beamPairId_to_beamIdPair(model.pred(data).detach().cpu().numpy(),M_t,M_r)

    ratio_mean, g_list, g_opt_list, ratio_list, beamgain_list, beamgain_opt_list = cal_beamgain(channel, bp_idx, opt_bp_idx, DFT_tx, DFT_rx)
    
    # See top1 CDF
    plt.figure()
    plt.hist(np.array(ratio_list),bins=100,cumulative=True,density=True)
    plt.show()
    plt.savefig(f'./plots/test_dimIn{model.feature_input_dim}_top1acc_CDF.png')
    plt.close()
    
    # See beam gain CDF
    plt.figure()
    plt.hist(np.array(beamgain_list),bins=100,cumulative=True,density=True,label='Alg')
    plt.hist(np.array(beamgain_opt_list),bins=100,cumulative=True,density=True,label='Opt')
    plt.legend()
    plt.show()
    plt.savefig(f'./plots/test_dimIn{model.feature_input_dim}_beamgain_CDF.png')
    plt.close()
    
    # See topK acc
    K_list = np.arange(1,5)
    ratio_list_topK = []
    beamgain_list_topK = []
    for i,K in enumerate(K_list):
        print(K)
        bp_idx = beamPairId_to_beamIdPair(model.pred_topK(data,K).detach().cpu().numpy(),M_t,M_r)
        ratio_mean, g_list, g_opt_list, ratio_list, beamgain_list, beamgain_opt_list = cal_beamgain_topK(K,channel, bp_idx, opt_bp_idx, DFT_tx, DFT_rx)
        ratio_list_topK.append(ratio_list)
        beamgain_list_topK.append(beamgain_list)
        reliability = (np.array(ratio_list_topK)>1-1e-9).sum(axis=-1)/len(ratio_list_topK[0])
        plt.figure()
        plt.xlabel('K')
        plt.ylabel('reliability')
        plt.yscale('log')
        plt.plot(K_list[:i+1], 1-reliability, '-*')
        plt.show()
        plt.savefig(f'./plots/test_dimIn{model.feature_input_dim}_topKreliability.png')
        plt.close()

        plt.figure()
        for j in range(i+1):
            plt.hist(np.array(beamgain_list_topK[j]),bins=100,cumulative=True,density=True,label=K_list[j])
            if j > 8:
                break
        plt.hist(np.array(beamgain_opt_list),bins=100,cumulative=True,density=True,label='Opt')
        plt.xlabel('beam gain')
        plt.ylabel('CDF')
        plt.legend()
        plt.show()
        plt.savefig(f'./plots/test_dimIn{model.feature_input_dim}_topKbeamgain_CDF.png')
        plt.close()
        
        
# https://18mh.net/
# https://njavtv.com/cn/parathd-3951
# https://jable.tv/videos/dvdms-917/