import os # Configure which GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import random
import torch
import re

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

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