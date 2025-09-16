#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_record_metrics(record_metrics, plt_save_dir, save_name):
        """Plot the training and validation loss and accuracy."""
        train_record_metrics = dict()
        val_record_metrics = dict()
        for k,v in record_metrics.items():
            if 'train' in k:
                train_record_metrics[k.split('train_')[-1]] = v
            elif 'val' in k:
                val_record_metrics[k.split('val_')[-1]] = v
        num_metrics = len(train_record_metrics)
        plt.figure(figsize=(8, 5*num_metrics))
        for i, (k, v) in enumerate(train_record_metrics.items()):
            plt.subplot(num_metrics, 1, 1+i)
            plt.plot(v, label='train')
            plt.plot(val_record_metrics[k], label='val')
            plt.legend()
            plt.title(k)
            plt.xlabel('epoch')
            plt.ylabel(k)
        # plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(plt_save_dir, save_name+'.png'))
        plt.close()
        
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
    