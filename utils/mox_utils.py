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
import re

from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair, generate_dft_codebook
from utils.NN_utils import prepare_dataset, BeamPredictionModel, train_beampred_model
from utils.options import args_parser
from utils.sumo_utils import read_trajectoryInfo_timeindex

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def generate_complex_gaussian_vector(shape, scale=1.0, mean=0.0):
    """
    生成服从复高斯分布的多维向量
    
    参数:
        shape (tuple) : 输出向量的形状（如 (n,) 或 (m,n)）
        scale (float) : 标准差缩放因子（默认1.0）
        mean (float)  : 分布的均值（默认0.0）
    
    返回:
        complex_array (ndarray) : 复高斯多维向量
    """
    # 生成独立的高斯分布的实部和虚部
    real_part = np.random.normal(loc=mean, scale=scale/np.sqrt(2), size=shape)
    imag_part = np.random.normal(loc=mean, scale=scale/np.sqrt(2), size=shape)
    
    # 组合为复数形式
    complex_array = real_part + 1j * imag_part
    return complex_array

def get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, device, P_t, P_noise):
    # 加载数据集
    prepared_dataset_filename = f'{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}'
    prepared_dataset_filepath = os.path.join('./prepared_dataset/',prepared_dataset_filename+'.pkl')
    if os.path.exists(prepared_dataset_filepath):
        f_read = open(prepared_dataset_filepath, 'rb')
        prepared_dataset = pickle.load(f_read)
        data_torch, best_beam_pair_index_torch, veh_pos_torch, veh_h_torch = \
            prepared_dataset['data_torch'], prepared_dataset['best_beam_pair_index_torch'], prepared_dataset['veh_pos_torch'], prepared_dataset['veh_h_torch']
        f_read.close()
    else:
        filepath = f'./sionna_result/trajectoryInfo_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
        datasize_upperbound = 1e9
        data_torch, best_beam_pair_index_torch, veh_pos_torch, veh_h_torch = \
            prepare_dataset(filepath,M_t, M_r, N_bs,datasize_upperbound,device,P_t, P_noise, n_pilot, mode=preprocess_mode)
        prepared_dataset = {}
        prepared_dataset['data_torch'] = data_torch
        prepared_dataset['best_beam_pair_index_torch'] = best_beam_pair_index_torch
        prepared_dataset['veh_pos_torch'] = veh_pos_torch
        prepared_dataset['veh_h_torch'] = veh_h_torch
        f_save = open(prepared_dataset_filepath, 'wb')
        pickle.dump(prepared_dataset, f_save)
        f_save.close()
    return prepared_dataset_filename, data_torch, veh_h_torch, veh_pos_torch, best_beam_pair_index_torch

def get_save_dirs(prepared_dataset_filename):
    result_save_dir = os.path.join('./NN_result',prepared_dataset_filename)
    plt_save_dir = os.path.join(result_save_dir,'plots')
    model_save_dir = os.path.join(result_save_dir,'models')
    log_save_dir = os.path.join(result_save_dir,'logs')
    if os.path.exists(result_save_dir) == False:
        os.mkdir(result_save_dir)
        os.mkdir(plt_save_dir)
        os.mkdir(model_save_dir)
        os.mkdir(log_save_dir)
    return result_save_dir, plt_save_dir, model_save_dir, log_save_dir

def split_string(X):
    """
    使用正则表达式分割字符串，处理多个分隔符组合并过滤空字符串
    参数：X (str): 输入的原字符串
    返回：list: 分割后的非空子字符串列表
    """
    # 使用正则表达式匹配任意连续分隔符组合进行分割
    substrs = re.split(r'[,\\\s]+', X)  # 匹配逗号、反斜杠、空白字符中的任意组合
    
    # 过滤掉分割结果中的空字符串
    substr_list = [sub.strip() for sub in substrs if sub.strip()]
    
    return substr_list


def save_log(local_dict, train_result_name_list, log_save_path):
    log_dict = {}
    for train_result_name in train_result_name_list:
        log_dict[train_result_name] = local_dict[train_result_name]
    with open(log_save_path, 'wb') as f:
        pickle.dump(log_dict, f)
    return log_dict