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

EPS = 1e-9

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

def prepare_dataset(sionna_result_filepath, M_t, M_r, N_bs, datasize_upperbound = 1e15, P_t=1e-1, P_noise=1e-14, n_pilot=16, mode=0):
    DFT_matrix_tx = generate_dft_codebook(M_t)
    DFT_matrix_rx = generate_dft_codebook(M_r)
    with open(sionna_result_filepath, 'rb') as f:
        trajectoryInfo = pickle.load(f)
    data_list = []
    best_beam_pair_index_list = []
    veh_pos_list = []
    veh_h_list = []
    sample_interval = int(M_t/n_pilot)
    for frame in trajectoryInfo.keys():
        print('prepare_dataset: ',frame)
        for veh in trajectoryInfo[frame].keys():
            veh_h = trajectoryInfo[frame][veh]['h']
            veh_pos = trajectoryInfo[frame][veh]['pos']
            # characteristics data
            # trajectoryInfo[frame][veh]['h'].shape  (8, 4, 64)
            
            if mode == 0:
                #data = veh_h.sum(axis=-1).reshape(-1)
                data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
                n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                data = (data + n).astype(np.complex64)
            elif mode == 1:
                data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].reshape(-1)
                n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                data = (data + n).astype(np.complex64)
            else:
                data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
                n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                data = (data + n).astype(np.complex64)
            
            
            data_list.append(data) # shape = (M_t*N_bs,), dtype = np.complex
            
            # best_beam_pair_index for N_bs BSs
            best_beam_pair_index = np.abs(np.matmul(np.matmul(veh_h, DFT_matrix_tx).T.conjugate(),DFT_matrix_rx).transpose([1,0,2]).reshape(N_bs,-1)).argmax(axis=-1)
            best_beam_pair_index_list.append(best_beam_pair_index)

            # vehicle h
            veh_h_list.append(veh_h)
            
            # vehicle position
            veh_pos_list.append(veh_pos)
            if len(veh_pos_list) >= datasize_upperbound:
                break
        if len(veh_pos_list) >= datasize_upperbound:
            break
    
    data_np = np.array(data_list)
    best_beam_pair_index_np = np.array(best_beam_pair_index_list)
    veh_h_np = np.array(veh_h_list)
    veh_pos_np = np.array(veh_pos_list)

    return data_np, best_beam_pair_index_np, veh_pos_np, veh_h_np

# 定义神经网络模型
class BeamPredictionModel(nn.Module):
    def __init__(self, feature_input_dim, num_bs, num_beampair, params_norm=[20,7]):
        super(BeamPredictionModel, self).__init__()
        self.num_bs = num_bs
        self.num_classes = num_beampair  # num_beampair = M_t*M_r
        self.feature_input_dim = feature_input_dim
        self.params_norm = params_norm
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_input_dim, 128),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(64, 128),
            # nn.ReLU()
            nn.GELU(),
        )
        
        # # 共享特征提取层（增强版）
        # self.shared_layers = nn.Sequential(
        #     nn.Linear(feature_input_dim, 256),
        #     nn.BatchNorm1d(256),  # 添加批量归一化
        #     nn.ReLU(),
        #     nn.Dropout(0.3),      # 添加Dropout
            
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
            
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2)       # 最后一层使用较小的Dropout
        # )
        
        # 多任务输出层（每个基站一个输出头）
        # self.output_heads = nn.ModuleList([
        #     nn.Linear(128, self.num_classes) for _ in range(num_bs)
        # ])
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, self.num_classes)
            ) for _ in range(num_bs)
        ])
        
        # # 多任务输出层（添加权重初始化）
        # self.output_heads = nn.ModuleList([
        #     self._create_output_head(128, self.num_classes) for _ in range(num_bs)
        # ])
        
        # # 初始化权重
        # self._initialize_weights()
        
    def _create_output_head(self, in_dim, out_dim):
        """创建带有初始化的输出头"""
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        return layer
    
    def _initialize_weights(self):
        """Xavier初始化所有层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.stack(outputs, dim=-2)

    def pred(self, x):
        outputs = self.forward(x)
        _, predicted = torch.max(outputs, -1)
        return predicted

    def pred_topK(self, x, K=1):
        outputs = self.forward(x)
        _, predicted = torch.topk(outputs, k=K, dim=-1, largest=True)
        return predicted
    
    def preprocess_input(self, x, params_norm=[20,7]):
        assert (x.dtype == torch.complex64) or (x.dtype == torch.complex128)
        # 将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
        amplitude = torch.abs(x)
        dB = 20*torch.log10(amplitude+EPS)
        phase = torch.angle(x)
        if self != None:
            preprocessed = torch.concatenate(((dB/self.params_norm[0]+self.params_norm[1], phase)),axis=-1)
        else:
            preprocessed = torch.concatenate(((dB/params_norm[0]+params_norm[1], phase)),axis=-1)
        return preprocessed
    
class BestGainPredictionModel(nn.Module):
    def __init__(self, feature_input_dim, num_bs, params_norm=[20,7]):
        super(BestGainPredictionModel, self).__init__()
        self.num_bs = num_bs
        self.feature_input_dim = feature_input_dim
        self.params_norm = params_norm
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_input_dim, 256),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(256, 256),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(256, 128),
            # nn.ReLU()
            nn.GELU(),
        )
        
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            ) for _ in range(num_bs)
        ])
        
        # # 多任务输出层（添加权重初始化）
        # self.output_heads = nn.ModuleList([
        #     self._create_output_head(128, self.num_classes) for _ in range(num_bs)
        # ])
        
        # # 初始化权重
        # self._initialize_weights()
        
    def _create_output_head(self, in_dim, out_dim):
        """创建带有初始化的输出头"""
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        return layer
    
    def _initialize_weights(self):
        """Xavier初始化所有层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.concat(outputs, dim=-1)

    def pred(self, x):
        outputs = self.forward(x)
        predicted_bestgain = self.params_norm[0]*( outputs - self.params_norm[1] )
        return predicted_bestgain
    
    def preprocess_input(self, x, params_norm=[20,7]):
        assert (x.dtype == torch.complex64) or (x.dtype == torch.complex128)
        # 将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
        amplitude = torch.abs(x)
        dB = 20*torch.log10(amplitude+EPS)
        phase = torch.angle(x)
        if self != None:
            preprocessed = torch.concatenate(((dB/self.params_norm[0]+self.params_norm[1], phase)),axis=-1)
        else:
            preprocessed = torch.concatenate(((dB/params_norm[0]+params_norm[1], phase)),axis=-1)
        return preprocessed


class PositionPredictionModel(nn.Module):
    def __init__(self, feature_input_dim, num_bs, pos_scale=100, params_norm=[20,7]):
        super(PositionPredictionModel, self).__init__()
        self.num_bs = num_bs
        self.feature_input_dim = feature_input_dim
        self.params_norm = params_norm
        self.pos_scale = pos_scale
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_input_dim, 256),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(256, 256),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(256, 128),
            # nn.ReLU()
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
        # # 多任务输出层（添加权重初始化）
        # self.output_heads = nn.ModuleList([
        #     self._create_output_head(128, self.num_classes) for _ in range(num_bs)
        # ])
        
        # # 初始化权重
        # self._initialize_weights()
        
    def _create_output_head(self, in_dim, out_dim):
        """创建带有初始化的输出头"""
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        return layer
    
    def _initialize_weights(self):
        """Xavier初始化所有层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        outputs = self.shared_layers(x)
        return outputs

    def pred(self, x):
        outputs = self.forward(x)
        predicted_pos = outputs * self.pos_scale
        return predicted_pos
    
    def preprocess_input(self, x, params_norm=[20,7]):
        assert (x.dtype == torch.complex64) or (x.dtype == torch.complex128)
        # 将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
        amplitude = torch.abs(x)
        dB = 20*torch.log10(amplitude+EPS)
        phase = torch.angle(x)
        if self != None:
            preprocessed = torch.concatenate(((dB/self.params_norm[0]+self.params_norm[1], phase)),axis=-1)
        else:
            preprocessed = torch.concatenate(((dB/params_norm[0]+params_norm[1], phase)),axis=-1)
        return preprocessed


class BestGainLevelPredictionModel(nn.Module):
    def __init__(self, feature_input_dim, num_bs, num_dBlevel=64, LB_db=-110, UB_db=-20, params_norm=[20,7]):
        super(BestGainLevelPredictionModel, self).__init__()
        self.num_bs = num_bs
        self.feature_input_dim = feature_input_dim
        self.params_norm = params_norm
        self.num_dBlevel = num_dBlevel
        self.LB_db = LB_db
        self.UB_db = UB_db
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_input_dim, 256),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(256, 256),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(256, 128),
            # nn.ReLU()
            nn.GELU(),
        )
        
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, self.num_dBlevel)
            ) for _ in range(num_bs)
        ])
        
        # # 多任务输出层（添加权重初始化）
        # self.output_heads = nn.ModuleList([
        #     self._create_output_head(128, self.num_classes) for _ in range(num_bs)
        # ])
        
        # # 初始化权重
        # self._initialize_weights()
        
    def _create_output_head(self, in_dim, out_dim):
        """创建带有初始化的输出头"""
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        return layer
    
    def _initialize_weights(self):
        """Xavier初始化所有层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.stack(outputs, dim=-2)

    def pred(self, x):
        outputs = self.forward(x)
        predicted_bestgain = self.params_norm[0]*( outputs - self.params_norm[1] )
        return predicted_bestgain
    
    def preprocess_input(self, x, params_norm=[20,7]):
        assert (x.dtype == torch.complex64) or (x.dtype == torch.complex128)
        # 将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
        amplitude = torch.abs(x)
        dB = 20*torch.log10(amplitude+EPS)
        phase = torch.angle(x)
        if self != None:
            preprocessed = torch.concatenate(((dB/self.params_norm[0]+self.params_norm[1], phase)),axis=-1)
        else:
            preprocessed = torch.concatenate(((dB/params_norm[0]+params_norm[1], phase)),axis=-1)
        return preprocessed
    
    def trans_dB_to_dBLevel(self, dB, num_dBlevel=64, LB_db=-110, UB_db=-20):
        if self != None:
            dBlevel = (dB > self.LB_db) * np.ceil((dB - self.LB_db) / (self.UB_db - self.LB_db) * (self.num_dBlevel-1))
            dBlevel = dBlevel.astype(np.int64)
        else:
            dBlevel = (dB > LB_db) * np.ceil((dB - LB_db) / (UB_db - LB_db) * (num_dBlevel-1))
            dBlevel = dBlevel.astype(np.int64)
        return dBlevel
        
  
# 训练函数
def train_beampred_model(num_epochs, device, data_complex, best_beam_pair_index_torch, M_t, M_r, pretrained_model_path=None, model_save_dir=None):
    num_beampair = M_t*M_r
    
    # 预处理数据：将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    data = BeamPredictionModel.preprocess_input(None,data_complex)
    # data = torch.concatenate(( (20*torch.log10(torch.abs(data_torch)+EPS))/20+7, torch.angle(data_torch)),axis=-1)
    labels = best_beam_pair_index_torch.to(device)
    num_bs = labels.shape[-1]
    feature_input_dim = data.shape[-1]
    
    # 初始化模型
    model = BeamPredictionModel(feature_input_dim, num_bs, num_beampair).to(device)
    
    # 划分训练集和验证集（示例比例）
    dataset = TensorDataset(data, labels)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    
    if pretrained_model_path != None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device, weights_only=True))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_model_state_dict = model.state_dict()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_acc = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)  # [batch_size, num_bs, num_classes]
            loss = criterion(outputs.view(-1, num_beampair), 
                            labels.view(-1))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.view(-1, num_beampair), 1)
            total += labels.numel()
            train_acc += (predicted == labels.view(-1)).sum().item()
        train_loss_list.append(train_loss/len(train_loader))
        train_acc_list.append(100*train_acc/total)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_acc = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, num_beampair), 
                                labels.view(-1))
                val_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs.view(-1, num_beampair), 1)
                total += labels.numel()
                val_acc += (predicted == labels.view(-1)).sum().item()
        val_loss_list.append(val_loss/len(val_loader))
        val_acc_list.append(100*val_acc/total)
        # 调整学习率
        # scheduler.step(val_loss)
        
        # 打印统计信息
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc_list[-1]:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {val_acc_list[-1]:.2f}%")
        
        
        
        # 保存最佳模型
        if val_acc_list[-1]>best_val_acc:
            best_val_acc = val_acc_list[-1]
            best_model_state_dict = model.state_dict()
    # torch.save(best_model_state_dict, model_save_dir+f"/beampred_dimIn{model.feature_input_dim}_valAcc{max(val_acc_list):.2f}%.pth")
    
    
    print("Training complete!")
    return model, train_loss_list, train_acc_list, val_loss_list, val_acc_list


def train_gainpred_model(num_epochs, device, data_complex, veh_h_torch, best_beam_pair_index_torch, M_t, M_r, pretrained_model_path=None, model_save_dir=None):
    num_beampair = M_t*M_r
    
    # 预处理数据：将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    data = BestGainPredictionModel.preprocess_input(None,data_complex)
    # data = torch.concatenate(( (20*torch.log10(torch.abs(data_torch)+EPS))/20+7, torch.angle(data_torch)),axis=-1)
    
    DFT_tx = generate_dft_codebook(M_t)
    DFT_rx = generate_dft_codebook(M_r)
    beamPairId = best_beam_pair_index_torch.detach().cpu().numpy()
    beamIdPair = beamPairId_to_beamIdPair(beamPairId,M_t,M_r)
    
    num_car, _, num_bs, _ = veh_h_torch.shape
    channel = veh_h_torch.detach().cpu().numpy()
    g_opt = np.zeros((num_car,num_bs)).astype(np.float32)
    for veh in range(num_car):
        for bs in range(num_bs):
            g_opt[veh,bs] = np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,beamIdPair[veh,bs,0]]).T.conjugate(),DFT_rx[:,beamIdPair[veh,bs,1]]))
            g_opt[veh,bs] = 20 * np.log10(g_opt[veh,bs]+EPS) 
    g_opt_normalized = g_opt / 20 + 7
    labels = torch.tensor(g_opt_normalized).to(device)
    feature_input_dim = data.shape[-1]

    # 初始化模型
    model = BestGainPredictionModel(feature_input_dim, num_bs).to(device)
    
    # 划分训练集和验证集（示例比例）
    dataset = TensorDataset(data, labels)
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    
    if pretrained_model_path != None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device, weights_only=True))
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_val_loss = float('inf')
    train_loss_list = []
    train_mae_list = []
    train_mse_list = []
    train_bestBS_mae_list = []
    train_bestBS_mse_list = []
    val_loss_list = []
    val_mae_list = []
    val_mse_list = []
    val_bestBS_mae_list = []
    val_bestBS_mse_list = []
    best_model_state_dict = model.state_dict()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_mae = 0
        train_mse = 0
        train_bestBS_mae = 0
        train_bestBS_mse = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_size, num_bs]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            total += labels.numel()
            dB_predicted = (outputs-7)*20
            dB_labels = (labels-7)*20
            train_mae += torch.abs((dB_predicted - dB_labels)).sum().item()
            train_mse += torch.square((dB_predicted - dB_labels)).sum().item()
            dB_bestBS, indices_bestBS = dB_labels.max(axis=-1)
            dB_bestBS_predicted, indices_bestBS_predicted = dB_predicted.max(axis=-1)
            train_bestBS_mae += torch.abs(dB_predicted[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item()
            train_bestBS_mse += torch.square(dB_predicted[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item()
            
            # 计算准确率
            # _, predicted = torch.max(outputs.view(-1, num_beampair), 1)
            # total += labels.numel()
            # train_acc += (predicted == labels.view(-1)).sum().item()
        train_loss_list.append(train_loss/len(train_loader))
        train_mae_list.append(train_mae/total)
        train_mse_list.append(train_mse/total)
        train_bestBS_mae_list.append(train_bestBS_mae/total*model.num_bs)
        train_bestBS_mse_list.append(train_bestBS_mse/total*model.num_bs)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        val_bestBS_mae = 0
        val_bestBS_mse = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                total += labels.numel()
                dB_predicted = (outputs-7)*20
                dB_labels = (labels-7)*20
                val_mae += torch.abs((dB_predicted - dB_labels)).sum().item()
                val_mse += torch.square((dB_predicted - dB_labels)).sum().item()
                dB_bestBS, indices_bestBS = dB_labels.max(axis=-1)
                dB_bestBS_predicted, indices_bestBS_predicted = dB_predicted.max(axis=-1)
                val_bestBS_mae += torch.abs(dB_predicted[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item()
                val_bestBS_mse += torch.square(dB_predicted[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item() 
        val_loss_list.append(val_loss/len(val_loader))
        val_mae_list.append(val_mae/total)
        val_mse_list.append(val_mse/total)
        val_bestBS_mae_list.append(val_bestBS_mae/total*model.num_bs)
        val_bestBS_mse_list.append(val_bestBS_mse/total*model.num_bs)
        
        # 调整学习率
        # scheduler.step(val_loss)
        
        # 打印统计信息
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        """print(f"Train Loss: {train_loss/len(train_loader):.4f} | "
              # f"Train Acc: {train_acc_list[-1]:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              # f"Val Acc: {val_acc_list[-1]:.2f}%"
              )"""
              
        print(f"Loss: {train_loss/len(train_loader):.4f} / {val_loss/len(val_loader):.4f} | "
              f"MAE: {train_mae_list[-1]:.2f} / {val_mae_list[-1]:.2f} | "
              f"MSE: {train_mse_list[-1]:.2f} / {val_mse_list[-1]:.2f} | "
              f"BestBS MAE: {train_bestBS_mae_list[-1]:.2f} / {val_bestBS_mae_list[-1]:.2f} | "
              f"BestBS MSE: {train_bestBS_mse_list[-1]:.2f} / {val_bestBS_mse_list[-1]:.2f} | "
              )
        
        
        
        # 保存最佳模型
        # if val_acc_list[-1]>best_val_acc:
        #     best_val_acc = val_acc_list[-1]
        #     best_model_state_dict = model.state_dict()
    # import ipdb;ipdb.set_trace()
    # torch.save(best_model_state_dict, model_save_dir+f"/gainpred_dimIn{model.feature_input_dim}_BBSMAE{min(train_bestBS_mae_list)}.pth")
    
    
    print("Training complete!")
    return model, \
        train_loss_list, train_mae_list, train_mse_list, \
        train_bestBS_mae_list, train_bestBS_mse_list, \
        val_loss_list, val_mae_list, val_mse_list, \
        val_bestBS_mae_list, val_bestBS_mse_list

def train_pospred_model(num_epochs, device, data_complex, veh_h_torch, veh_pos_torch, best_beam_pair_index_torch, M_t, M_r, pretrained_model_path=None, model_save_dir=None):
    num_beampair = M_t*M_r
    
    # 预处理数据：将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    data = PositionPredictionModel.preprocess_input(None,data_complex)
    # data = torch.concatenate(( (20*torch.log10(torch.abs(data_torch)+EPS))/20+7, torch.angle(data_torch)),axis=-1)
    
    DFT_tx = generate_dft_codebook(M_t)
    DFT_rx = generate_dft_codebook(M_r)
    beamPairId = best_beam_pair_index_torch.detach().cpu().numpy()
    beamIdPair = beamPairId_to_beamIdPair(beamPairId,M_t,M_r)
    
    num_car, _, num_bs, _ = veh_h_torch.shape
    channel = veh_h_torch.detach().cpu().numpy()
    g_opt = np.zeros((num_car,num_bs)).astype(np.float32)
    for veh in range(num_car):
        for bs in range(num_bs):
            g_opt[veh,bs] = np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,beamIdPair[veh,bs,0]]).T.conjugate(),DFT_rx[:,beamIdPair[veh,bs,1]]))
            g_opt[veh,bs] = 20 * np.log10(g_opt[veh,bs]+EPS) 
    feature_input_dim = data.shape[-1]

    # 初始化模型
    model = PositionPredictionModel(feature_input_dim, num_bs).to(device)
    labels = (veh_pos_torch / model.pos_scale).float().to(device)
    
    # 划分训练集和验证集（示例比例）
    dataset = TensorDataset(data, labels)
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    
    if pretrained_model_path != None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device, weights_only=True))
    
    distance_lossfunc = nn.MSELoss()  # 坐标损失
    angle_lossfunc = nn.CosineSimilarity()  # 方向一致性损失（可选）
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_rmse = np.inf
    train_loss_list = []
    train_acc_list = []
    train_mae_list = []
    train_rmse_list = []
    val_loss_list = []
    val_acc_list = []
    val_mae_list = []
    val_rmse_list = []
    best_model_state_dict = model.state_dict()
    
    weight_dist_angle = 0.5
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_rmse = 0
        train_mae = 0
        total = 0
        # if epoch == 20:
        #     import ipdb;ipdb.set_trace()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # import ipdb;ipdb.set_trace()
            outputs = model(inputs)  # [batch_size, num_bs]
            loss = distance_lossfunc(outputs, labels)
            # loss = distance_lossfunc(outputs, labels) - weight_dist_angle*torch.abs(angle_lossfunc(outputs, labels)).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += len(labels)
            train_rmse += torch.square(outputs-labels).sum(-1).sqrt().sum().item()
            train_mae += torch.abs(outputs-labels).sum(-1).sum().item()
            
        train_loss_list.append(train_loss/len(train_loader))
        train_rmse_list.append(train_rmse/total*model.pos_scale)
        train_mae_list.append(train_mae/total*model.pos_scale)
        # train_acc_list.append(100*train_acc/total)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_rmse = 0
        val_mae = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = distance_lossfunc(outputs, labels)
                # loss = distance_lossfunc(outputs, labels) - weight_dist_angle*torch.abs(angle_lossfunc(outputs, labels)).mean()
                val_loss += loss.item()
                total += len(labels)
                val_rmse += torch.square(outputs-labels).sum(-1).sqrt().sum().item()
                val_mae += torch.abs(outputs-labels).sum(-1).sum().item()
        val_loss_list.append(val_loss/len(val_loader))
        val_rmse_list.append(val_rmse/total*model.pos_scale)
        val_mae_list.append(val_mae/total*model.pos_scale)
        
        # 调整学习率
        # scheduler.step(val_loss)
        
        # 打印统计信息
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Loss: {train_loss_list[-1]:.4f} / {val_loss_list[-1]:.4f} | "
              f"RMSE: {train_rmse_list[-1]:.4f} / {val_rmse_list[-1]:.4f} | "
              f"MAE: {train_mae_list[-1]:.4f} / {val_mae_list[-1]:.4f} | "
              )
        
        # 保存最佳模型
        if val_rmse_list[-1]>best_val_rmse:
            best_val_rmse = val_rmse_list[-1]
            best_model_state_dict = model.state_dict()
    # import ipdb;ipdb.set_trace()
    # torch.save(best_model_state_dict, model_save_dir+f"/pospred_dimIn{model.feature_input_dim}_valRMSE{min(val_rmse_list):.2f}%.pth")
    
    
    print("Training complete!")
    return model, train_loss_list, train_rmse_list, train_mae_list, val_loss_list, val_rmse_list, val_mae_list


def train_gainlevelpred_model(num_epochs, device, data_complex, veh_h_torch, best_beam_pair_index_torch, M_t, M_r, pretrained_model_path=None, model_save_dir=None):
    num_beampair = M_t*M_r
    
    # 预处理数据：将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    data = BestGainPredictionModel.preprocess_input(None,data_complex)
    # data = torch.concatenate(( (20*torch.log10(torch.abs(data_torch)+EPS))/20+7, torch.angle(data_torch)),axis=-1)
    
    DFT_tx = generate_dft_codebook(M_t)
    DFT_rx = generate_dft_codebook(M_r)
    beamPairId = best_beam_pair_index_torch.detach().cpu().numpy()
    beamIdPair = beamPairId_to_beamIdPair(beamPairId,M_t,M_r)
    
    num_car, _, num_bs, _ = veh_h_torch.shape
    channel = veh_h_torch.detach().cpu().numpy()
    g_opt = np.zeros((num_car,num_bs)).astype(np.float32)
    for veh in range(num_car):
        for bs in range(num_bs):
            g_opt[veh,bs] = np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,beamIdPair[veh,bs,0]]).T.conjugate(),DFT_rx[:,beamIdPair[veh,bs,1]]))
            g_opt[veh,bs] = 20 * np.log10(g_opt[veh,bs]+EPS) 
    labels_dB = torch.tensor(g_opt).to(device)
    
    feature_input_dim = data.shape[-1]

    LB_db, UB_db = -110, -20
    num_dBlevel = 64
    
    # 初始化模型
    model = BestGainLevelPredictionModel(feature_input_dim, num_bs, num_dBlevel=num_dBlevel, LB_db=LB_db, UB_db=UB_db).to(device)
    dBlevel = BestGainLevelPredictionModel.trans_dB_to_dBLevel(model,g_opt)
    labels_dBlevel = torch.tensor(dBlevel).to(device)
    
    # 划分训练集和验证集（示例比例）
    dataset = TensorDataset(data, labels_dBlevel)
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(20))
    
    dataset2 = TensorDataset(data, labels_dB)
    train_dataset2, val_dataset2 = torch.utils.data.random_split(dataset2, [train_size, val_size], generator=torch.Generator().manual_seed(20))
    
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(20))
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    train_loader2 = DataLoader(train_dataset2, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(20))
    val_loader2 = DataLoader(val_dataset2, batch_size=256, shuffle=False)
    
    if pretrained_model_path != None:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device, weights_only=True))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0
    train_loss_list = []
    train_acc_list = []
    train_mae_list = []
    train_mse_list = []
    train_bestBS_mae_list = []
    train_bestBS_mse_list = []
    train_bestBS_acc_list = []
    val_loss_list = []
    val_acc_list = []
    val_mae_list = []
    val_mse_list = []
    val_bestBS_mae_list = []
    val_bestBS_mse_list = []
    val_bestBS_acc_list = []
    best_model_state_dict = model.state_dict()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_acc = 0
        train_mae = 0
        train_mse = 0
        train_bestBS_mae = 0
        train_bestBS_mse = 0
        train_bestBS_acc = 0
        total = 0
        for (inputs, dBlevel_labels), (inputs2, dB_labels) in zip(train_loader,train_loader2):
        # for inputs, dBlevel_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_size, num_bs]
            loss = criterion(outputs.view(-1, model.num_dBlevel), dBlevel_labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 计算准确率
            _, dBlevel_predicted = torch.max(outputs.view(-1, model.num_dBlevel), 1)
            dB_predicted = (dBlevel_predicted==0)*(-240) + (dBlevel_predicted!=0)*((dBlevel_predicted-0.5)/(model.num_dBlevel-1)*(model.UB_db-model.LB_db)+model.LB_db)
            total += dBlevel_labels.numel()
            train_acc += (dBlevel_predicted == dBlevel_labels.view(-1)).sum().item()
            train_mae += torch.abs((dB_predicted - dB_labels.view(-1))).sum().item()
            train_mse += torch.square((dB_predicted - dB_labels.view(-1))).sum().item()
            
            dB_bestBS, indices_bestBS = dB_labels.max(axis=-1)
            dB_bestBS_predicted, indices_bestBS_predicted = dB_predicted.view_as(dB_labels).max(axis=-1)
            train_bestBS_mae += torch.abs(dB_predicted.view_as(dB_labels)[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item()
            train_bestBS_mse += torch.square(dB_predicted.view_as(dB_labels)[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item()
            train_bestBS_acc += (indices_bestBS_predicted==indices_bestBS).view(-1).sum()
            
        train_loss_list.append(train_loss/len(train_loader))
        train_acc_list.append(100*train_acc/total)
        train_mae_list.append(train_mae/total)
        train_mse_list.append(train_mse/total)
        train_bestBS_mae_list.append(train_bestBS_mae/total*model.num_bs)
        train_bestBS_mse_list.append(train_bestBS_mse/total*model.num_bs)
        train_bestBS_acc_list.append(100*train_bestBS_acc/total*model.num_bs)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_acc = 0
        val_mae = 0
        val_mse = 0
        val_bestBS_mae = 0
        val_bestBS_mse = 0
        val_bestBS_acc = 0
        total = 0
        with torch.no_grad():
            #import ipdb;ipdb.set_trace()
            for (inputs, dBlevel_labels), (inputs2, dB_labels) in zip(val_loader,val_loader2):
                outputs = model(inputs)
                criterion(outputs.view(-1, model.num_dBlevel), dBlevel_labels.view(-1))
                val_loss += loss.item()
                
                # 计算准确率
                _, dBlevel_predicted = torch.max(outputs.view(-1, model.num_dBlevel), 1)
                dB_predicted = (dBlevel_predicted==0)*(-240) + (dBlevel_predicted!=0)*((dBlevel_predicted-0.5)/(model.num_dBlevel-1)*(model.UB_db-model.LB_db)+model.LB_db)
                total += dBlevel_labels.numel()
                val_acc += (dBlevel_predicted == dBlevel_labels.view(-1)).sum().item()
                val_mae += torch.abs((dB_predicted - dB_labels.view(-1))).sum().item()
                val_mse += torch.square((dB_predicted - dB_labels.view(-1))).sum().item()
                
                dB_bestBS, indices_bestBS = dB_labels.max(axis=-1)
                dB_bestBS_predicted, indices_bestBS_predicted = dB_predicted.view_as(dB_labels).max(axis=-1)
                val_bestBS_mae += torch.abs(dB_predicted.view_as(dB_labels)[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item()
                val_bestBS_mse += torch.square(dB_predicted.view_as(dB_labels)[range(len(dB_labels)),indices_bestBS] - dB_labels[range(len(dB_labels)),indices_bestBS]).sum().item()
                val_bestBS_acc += (indices_bestBS_predicted==indices_bestBS).view(-1).sum()
        
        val_loss_list.append(val_loss/len(val_loader))
        val_acc_list.append(100*val_acc/total)
        val_mae_list.append(val_mae/total)
        val_mse_list.append(val_mse/total)
        val_bestBS_mae_list.append(val_bestBS_mae/total*model.num_bs)
        val_bestBS_mse_list.append(val_bestBS_mse/total*model.num_bs)
        val_bestBS_acc_list.append(100*val_bestBS_acc/total*model.num_bs)
        
        # 调整学习率
        # scheduler.step(val_loss)
        
        # 打印统计信息
        print(f"num_dBlevel={num_dBlevel}: Epoch [{epoch+1}/{num_epochs}] ")
        
        """print(f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc_list[-1]:.2f}% | "
              f"Train MAE: {train_mae_list[-1]:.2f} | "
              f"Train MSE: {train_mse_list[-1]:.2f} | "
              f"Train BestBS MAE: {train_bestBS_mae_list[-1]:.2f} | "
              f"Train BestBS MSE: {train_bestBS_mse_list[-1]:.2f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {val_acc_list[-1]:.2f}% | "
              f"Val MAE: {val_mae_list[-1]:.2f} | "
              f"Val MSE: {val_mse_list[-1]:.2f} | "
              f"Val BestBS MAE: {val_bestBS_mae_list[-1]:.2f} | "
              f"Val BestBS MSE: {val_bestBS_mse_list[-1]:.2f} | "
              )"""
        
        print(f"Loss: {train_loss/len(train_loader):.4f} / {val_loss/len(val_loader):.4f} | "
              f"Acc: {train_acc_list[-1]:.2f}% / {val_acc_list[-1]:.2f}% | "
              f"MAE: {train_mae_list[-1]:.2f} / {val_mae_list[-1]:.2f} | "
              f"MSE: {train_mse_list[-1]:.2f} / {val_mse_list[-1]:.2f} | "
              f"BestBS MAE: {train_bestBS_mae_list[-1]:.2f} / {val_bestBS_mae_list[-1]:.2f} | "
              f"BestBS MSE: {train_bestBS_mse_list[-1]:.2f} / {val_bestBS_mse_list[-1]:.2f} | "
              f"BestBS Acc: {train_bestBS_acc_list[-1]:.2f}% / {val_bestBS_acc_list[-1]:.2f}% | "
              )
        
        # 保存最佳模型
        if val_acc_list[-1]>best_val_acc:
            best_val_acc = val_acc_list[-1]
            best_model_state_dict = model.state_dict()
    # torch.save(best_model_state_dict, model_save_dir+f"/gainlevelpred_dimIn{model.feature_input_dim}Out{model.num_dBlevel}_valAcc{max(val_acc_list):.2f}%_BBSMAE{min(val_bestBS_mae_list):.2f}.pth")
    print("Training complete!")
    return model, \
        train_loss_list, train_acc_list, train_mae_list, train_mse_list, \
        train_bestBS_mae_list, train_bestBS_mse_list, train_bestBS_acc_list, \
        val_loss_list, val_acc_list, val_mae_list, val_mse_list, \
        val_bestBS_mae_list, val_bestBS_mse_list, val_bestBS_acc_list
