import os # Configure which GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import random
import torch
import re

from utils.plot_utils import plot_record_metrics

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_save_dirs(save_dir_name):
    os.makedirs('./NN_result/', exist_ok=True)
    result_save_dir = os.path.join('./NN_result',save_dir_name)
    plt_save_dir = os.path.join(result_save_dir,'plots')
    model_save_dir = os.path.join(result_save_dir,'models')
    log_save_dir = os.path.join(result_save_dir,'logs')
    if os.path.exists(result_save_dir) == False:
        os.mkdir(result_save_dir)
        os.mkdir(plt_save_dir)
        os.mkdir(model_save_dir)
        os.mkdir(log_save_dir)
    return result_save_dir, plt_save_dir, model_save_dir, log_save_dir

def save_NN_results(prepared_dataset_filename, save_file_name, best_model_weights, record_metrics):
    result_save_dir, plt_save_dir, model_save_dir, log_save_dir = get_save_dirs(prepared_dataset_filename)
    torch.save(best_model_weights, os.path.join(model_save_dir, save_file_name+'.pth'))
    log_dict = {'best_model_weights':best_model_weights, 'record_metrics':record_metrics}
    with open(os.path.join(log_save_dir,save_file_name+'.pkl'), 'wb') as f:
        pickle.dump(log_dict, f)
    plot_record_metrics(record_metrics, plt_save_dir, save_file_name)    


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
        if train_result_name == 'model':
            log_dict[train_result_name] = local_dict[train_result_name].state_dict()
        else:
            log_dict[train_result_name] = local_dict[train_result_name]
    with open(log_save_path, 'wb') as f:
        pickle.dump(log_dict, f)
    return log_dict

def np2torch(x,device):
    return torch.tensor(x).to(device)

def see_model_structure(log_addr):
    with open(log_addr,'rb') as f:
        log = pickle.load(f)
    for k,v in log['model'].items():
        if k.split('.')[-1]=='weight':
            print(k.split('.')[:-1],v.shape[1], '->', v.shape[0])
    return log

def lin2dB(x:np.ndarray, eps=1e-9):
    return 10 * np.log10(x + eps)

def dB2lin(x:np.ndarray):
    return 10 ** (x / 10)

def generate_1Dsamples(split_point_list, spacing_list):
    samples = []
    for i in range(len(split_point_list) - 1):
        start = split_point_list[i]
        end = split_point_list[i + 1]
        samples.append(np.arange(start, end, spacing_list[i]))
    samples.append(np.array([split_point_list[-1]]))  # Add the last point
    return np.concatenate(samples)