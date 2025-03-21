#!/usr/bin/env python
import os
import sys
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
import numpy as np

def plot_beampred(save_path, log_dict):
    train_loss_list = log_dict['train_loss_list']
    val_loss_list = log_dict['val_loss_list']
    train_acc_list = log_dict['train_acc_list']
    val_acc_list = log_dict['val_acc_list']
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(train_acc_list, label='train Acc')
    plt.plot(val_acc_list, label='val Acc')
    plt.legend()
    plt.show()
    plt.savefig(save_path)
    
def plot_pospred(save_path, log_dict):
    train_loss_list = log_dict['train_loss_list']
    val_loss_list = log_dict['val_loss_list']
    train_rmse_list = log_dict['train_rmse_list']
    val_rmse_list = log_dict['val_rmse_list']
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
    plt.savefig(save_path)
    
def plot_gainpred(save_path, log_dict):
    train_loss_list = log_dict['train_loss_list']
    val_loss_list = log_dict['val_loss_list']
    train_bestBS_mae_list = log_dict['train_bestBS_mae_list']
    val_bestBS_mae_list = log_dict['val_bestBS_mae_list']
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(train_bestBS_mae_list, label='train bestBS MAE')
    plt.plot(val_bestBS_mae_list, label='val bestBS MAE')
    plt.legend()
    plt.show()
    plt.savefig(save_path)
    
def plot_gainlevelpred(save_path, log_dict):
    train_loss_list = log_dict['train_loss_list']
    val_loss_list = log_dict['val_loss_list']
    train_acc_list = log_dict['train_acc_list']
    val_acc_list = log_dict['val_acc_list']
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(train_acc_list, label='train Acc')
    plt.plot(val_acc_list, label='val Acc')
    plt.legend()
    plt.show()
    plt.savefig(save_path)
    