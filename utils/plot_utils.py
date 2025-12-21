#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import numpy as np

class CyclicList(list):
    def __getitem__(self, i):
        # 自动对长度取模，实现循环索引
        return super().__getitem__(i % len(self))

# 线型列表（linestyle）
plt_linestyle_list = CyclicList([
    '-',    # 0 实线
    '--',   # 1 虚线
    '-.',   # 2 点划线
    ':',    # 3 点线
    '-',    # 4
    '--',   # 5
    '-.',   # 6
    ':',    # 7
    '-',    # 8
    '--',   # 9
])

# 颜色列表（color）——使用 Matplotlib 默认颜色循环，更通用
plt_color_list = CyclicList([
    'C3',   # 3 红
    'C0',   # 0 蓝
    'C1',   # 1 橙
    'C2',   # 2 绿
    'C4',   # 4 紫
    'C5',   # 5 棕
    'C6',   # 6 粉
    'C7',   # 7 灰
    'C8',   # 8 黄绿
    'C9',   # 9 蓝绿
])

# 标记列表（marker）
plt_marker_list = CyclicList([
    'o',    # 0 圆点
    's',    # 1 正方形
    '^',    # 2 三角形（上）
    'D',    # 3 菱形
    'v',    # 4 三角形（下）
    '>',    # 5 三角形（右）
    '<',    # 6 三角形（左）
    'p',    # 7 五边形
    'h',    # 8 六边形
    'x',    # 9 叉号
])


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
    