#!/usr/bin/env python
import os
import abc
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import copy
import random
import collections
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair, generate_dft_codebook
from utils.data_utils import random_truncate_tensor_sequence
import numpy as np

EPS = 1e-9

# 定义神经网络模型
class VectorResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        """
        适用于向量的残差块实现
        结构：Linear -> BN -> ReLU -> Linear -> BN -> Add -> ReLU
        
        参数：
            in_features:  输入特征维度
            out_features: 输出特征维度
        """
        super().__init__()
        
        # 主路径
        self.fc1 = nn.Linear(
            in_features, 
            out_features, 
            bias=False  # 使用BN时无需bias
        )
        self.bn1 = nn.BatchNorm1d(in_features)
        
        self.fc2 = nn.Linear(
            out_features, 
            out_features, 
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 捷径路径（当维度不匹配时启用）
        self.downsample = None
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(
                    in_features, 
                    out_features, 
                    bias=False
                ),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = x  # 原始输入
        
        # 主路径处理
        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        # 捷径路径处理（如果需要）
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # 相加并激活
        out += identity
        
        return out
    
class BasePredictionModel(nn.Module):
    def __init__(self, feature_input_dim, num_bs, params_norm=[20,7]):
        super().__init__()
        self.feature_input_dim = feature_input_dim
        self.num_bs = num_bs
        self.params_norm = params_norm
    
    def _build_LSTM_layers(self, input_dim, hidden_dim, layer_dim, dropout):
        lstm_layers = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout
        )
        return lstm_layers
        
    def _build_shared_layers(self, layer_dim_list=[]):
        shared_layers = nn.Sequential()
        for i in range(len(layer_dim_list)-1):
            # shared_layers.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
            shared_layers.append(VectorResBlock(layer_dim_list[i], layer_dim_list[i+1]))
            shared_layers.append(nn.ReLU())
        return shared_layers

    def _build_output_heads(self, num_head, layer_dim_list):
        output_heads = nn.ModuleList()
        for _ in range(num_head):
            head = nn.Sequential()
            for i in range(len(layer_dim_list)-1):
                # head.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
                head.append(VectorResBlock(layer_dim_list[i], layer_dim_list[i+1]))
                head.append(nn.ReLU())
            head.pop(-1)
            output_heads.append(head)
        return output_heads
    
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

### DNN Models ###

class BeamPredictionModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, num_beampair, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        self.num_class = num_beampair  # num_beampair = M_t*M_r
        
        # 共享特征提取层
        self.shared_layers = self._build_shared_layers(
            layer_dim_list=[feature_input_dim, 128, 64, 32, 64, 128]
        )
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 128, self.num_class]
        )
    
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
    
    def predict(self, x_np, device, K=1):
        x_torch = torch.tensor(x_np).to(device)
        return self.pred_topK(x_torch,K).detach().cpu().numpy()


class BlockPredictionModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        self.num_class = 2
        
        # 共享特征提取层
        self.shared_layers = self._build_shared_layers(
            layer_dim_list=[feature_input_dim, 128, 64, 32, 64, 128]
        )
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 128, self.num_class]
        )
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.stack(outputs, dim=-2)

    def pred(self, x):
        outputs = self.forward(x)
        _, predicted = torch.max(outputs, -1)
        return predicted

    def predict(self, x_np, device):
        x_torch = torch.tensor(x_np).to(device)
        return self.pred(x_torch).detach().cpu().numpy()

    
class BestGainPredictionModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        
        # 共享特征提取层
        self.shared_layers = self._build_shared_layers(
            layer_dim_list=[feature_input_dim, 256, 256, 128]
        )
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 128, 128, 128, 1]
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.concat(outputs, dim=-1)

    def pred(self, x):
        outputs = self.forward(x)
        predicted_bestgain = self.params_norm[0]*( outputs - self.params_norm[1] )
        return predicted_bestgain
    
    def predict(self, x_np, device):
        x_torch = torch.tensor(x_np).to(device)
        return self.pred(x_torch).detach().cpu().numpy()


### LSTM Models ###

class BeamPredictionLSTMModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, num_beampair, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        self.num_class = num_beampair  # num_beampair = M_t*M_r
        
        self.lstm_layers = self._build_LSTM_layers(
            input_dim=feature_input_dim,
            hidden_dim=128,
            layer_dim=1,
            dropout=0
        )
        
        # self.shared_layers = self._build_shared_layers(
        #     layer_dim_list=[128, 64, 32, 64, 128]
        # )
        self.shared_layers = self._build_shared_layers()
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 128, 128, 128, self.num_class]
        )
    
    def forward(self, x, hc=None, lengths=None):
        if lengths is not None:
            assert len(x.shape) == 3 and len(lengths) == x.shape[0]
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (h_c, c_n) =  self.lstm_layers(x, hc)
            pad_packed_out, valid_lengths = pad_packed_sequence(packed_out, batch_first=True)
            lstm_out = pad_packed_out[torch.arange(pad_packed_out.shape[0]), valid_lengths-1, ...]
            shared_features = self.shared_layers(lstm_out)
        else:
            lstm_out, (h_c, c_n) = self.lstm_layers(x, hc)
            shared_features = self.shared_layers(lstm_out[...,-1,:])
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
    
    def predict(self, x_np, device, K=1):
        x_torch = torch.tensor(x_np).to(device)
        return self.pred_topK(x_torch,K).detach().cpu().numpy()
      
        
class BlockPredictionLSTMModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        self.num_class = 2
        
        # 共享特征提取层
        self.lstm_layers = self._build_LSTM_layers(
            input_dim=feature_input_dim,
            hidden_dim=128,
            layer_dim=1,
            dropout=0
        )
        
        # self.shared_layers = self._build_shared_layers(
        #     layer_dim_list=[128, 64, 32, 64, 128]
        # )
        self.shared_layers = self._build_shared_layers()
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 128, 128, 128, self.num_class]
        )
    
    def forward(self, x, hc=None, lengths=None):
        if lengths is not None:
            assert len(x.shape) == 3 and len(lengths) == x.shape[0]
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (h_c, c_n) =  self.lstm_layers(x, hc)
            pad_packed_out, valid_lengths = pad_packed_sequence(packed_out, batch_first=True)
            lstm_out = pad_packed_out[torch.arange(pad_packed_out.shape[0]), valid_lengths-1, ...]
            shared_features = self.shared_layers(lstm_out)
        else:
            lstm_out, (h_c, c_n) = self.lstm_layers(x, hc)
            shared_features = self.shared_layers(lstm_out[...,-1,:])
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.stack(outputs, dim=-2)

    def pred(self, x):
        outputs = self.forward(x)
        _, predicted = torch.max(outputs, -1)
        return predicted

    def predict(self, x_np, device):
        x_torch = torch.tensor(x_np).to(device)
        return self.pred(x_torch).detach().cpu().numpy()
   
     
class BestGainPredictionLSTMModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        
        # 共享特征提取层
        self.lstm_layers = self._build_LSTM_layers(
            input_dim=feature_input_dim,
            hidden_dim=128,
            layer_dim=1,
            dropout=0
        )
        
        # self.shared_layers = self._build_shared_layers(
        #     layer_dim_list=[256, 256, 128]
        # )
        self.shared_layers = self._build_shared_layers()
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 128, 128, 128, 1]
        )
    
    def forward(self, x, hc=None, lengths=None):
        if lengths is not None:
            assert len(x.shape) == 3 and len(lengths) == x.shape[0]
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (h_c, c_n) =  self.lstm_layers(x, hc)
            pad_packed_out, valid_lengths = pad_packed_sequence(packed_out, batch_first=True)
            lstm_out = pad_packed_out[torch.arange(pad_packed_out.shape[0]), valid_lengths-1, ...]
            shared_features = self.shared_layers(lstm_out)
        else:
            lstm_out, (h_c, c_n) = self.lstm_layers(x, hc)
            shared_features = self.shared_layers(lstm_out[...,-1,:])
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.concat(outputs, dim=-1)

    def pred(self, x):
        outputs = self.forward(x)
        predicted_bestgain = self.params_norm[0]*( outputs - self.params_norm[1] )
        return predicted_bestgain
    
    def predict(self, x_np, device):
        x_torch = torch.tensor(x_np).to(device)
        return self.pred(x_torch).detach().cpu().numpy()
       

def preprocess_data(data_complex,pos_in_data):
    # 预处理数据：将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    if pos_in_data:
        data_csi = BasePredictionModel.preprocess_input(None,data_complex[...,:-2])
        data_pos = (data_complex[...,-2:]/100).real
        data = torch.concat([data_csi,data_pos],axis=-1).to(torch.float)
    else:
        data = BasePredictionModel.preprocess_input(None,data_complex)
    return data




























# TODO

class GenericTrainer:
    def __init__(self, config):
        self.device = config['device']
        self.model = config['model_class'](**config['model_args']).to(self.device)
        self.criterion = config['criterion']
        self.optimizer = config['optimizer'](self.model.parameters(), **config['optim_args'])
        self.scheduler = config.get('scheduler', None)
        self.metrics = config['metrics']
        
        # 数据相关配置
        self.train_loader = config['train_loader']
        self.val_loader = config.get('val_loader', None)
        self.various_input_length = config.get('various_input_length', None)
        
        # 回调函数
        self.preprocess_data = config.get('preprocess_data', lambda x: x)
        self.on_epoch_start = config.get('on_epoch_start', lambda *x: None)
        
    def _compute_metrics(self, outputs, labels, phase='train'):
        """动态计算所有注册的指标"""
        results = {}
        for name, func in self.metrics.items():
            results[f"{phase}_{name}"] = func(outputs, labels)
        return results
        
    def run_epoch(self, loader, phase='train'):
        is_train = phase == 'train'
        self.model.train(is_train)
        
        total_loss = 0
        metric_sums = defaultdict(float)
        total_samples = 0
        
        with torch.set_grad_enabled(is_train):
            for inputs, labels in loader:
                if is_train:
                    self.optimizer.zero_grad()
                
                if self.various_input_length is not None:
                    inputs, lengths = inputs 
                    # inputs.shape = (batch_size, seq_len, feature_dim), lengths.shape = (batch_size,), labels.shape = (batch_size, num_bs)
                    # inputs, lengths = random_truncate_tensor_sequence(inputs, lengths)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs, lengths=lengths)
                else:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    
                loss = self.criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                
                # 更新统计量
                total_loss += loss.item() * inputs.size(0)
                metrics = self._compute_metrics(outputs, labels, phase)
                for k, v in metrics.items():
                    metric_sums[k] += v * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_metrics = {f'{phase}_loss': total_loss / total_samples}
        for k, v in metric_sums.items():
            avg_metrics[k] = v / total_samples
        # 计算平均损失和各评价指标
        return avg_metrics
    
    def train(self, num_epochs):
        best_metrics = {}
        best_model_weights = self.model.state_dict()
        record_metrics = defaultdict(list)
        for epoch in range(num_epochs):
            _time = time.time()
            self.on_epoch_start(self, epoch)  # 触发回调
            
            train_metrics = self.run_epoch(self.train_loader, 'train')
            val_metrics = self.run_epoch(self.val_loader, 'val') if self.val_loader else {}
            
            # 合并指标并打印
            epoch_metrics = {**train_metrics, **val_metrics}
            for k, v in epoch_metrics.items():
                record_metrics[k].append(v)
            print(f"Epoch {epoch+1}/{num_epochs}, training time: {time.time()-_time:.2f}s")
            # print(' | '.join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))
            print(' | '.join([f"{k1.split('train_')[-1]}: {v1:.4f}/{v2:.4f}" for (k1,v1),(k2,v2) in zip(train_metrics.items(),val_metrics.items())]))
            
            # 保存最佳模型逻辑
            if 'val_loss' in epoch_metrics and epoch_metrics['val_loss'] < best_metrics.get('val_loss', float('inf')):
                best_metrics = epoch_metrics
                best_model_weights = self.model.state_dict()
        
        return best_model_weights, record_metrics

def collate_fn_lstm(batch): # TODO
    """自定义collate函数，处理变长序列"""
    # import ipdb;ipdb.set_trace()
    data, lengths, labels = zip(*batch)
    return (torch.stack(data), torch.stack(lengths)),  torch.stack(labels)

def create_loaders(dataset, split_ratio=0.7, train_batch=128, val_batch=256, seed=20, collate_fn=None):
    """通用数据集划分和加载器创建"""
    # 固定随机种子保证可复现性
    generator = torch.Generator().manual_seed(seed) if seed else None
    
    # 划分训练/验证集
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )

    # 创建加载器
    train_loader = DataLoader(
        train_set, 
        batch_size=train_batch, 
        shuffle=True, # shuffle=(seed is None),  # 有种子时不shuffle
        generator=generator,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_set, batch_size=val_batch, collate_fn=collate_fn)
    return train_loader, val_loader

def build_trainer(task_config):
    # 数据预处理封装
    def preprocess_wrapper(func):
        return lambda data: func(data, task_config['pos_in_data'])
    
    # 构建数据加载器
    if task_config.get('various_input_length', None) is not None:
        dataset = TensorDataset(
            preprocess_data(task_config['data_complex'], task_config['pos_in_data']),
            task_config['various_input_length'],
            task_config['labels'],
        )
        collate_fn = collate_fn_lstm
    else:
        dataset = TensorDataset(
            preprocess_data(task_config['data_complex'], task_config['pos_in_data']),
            task_config['labels']
        )
        collate_fn = None
    
    train_loader, val_loader = create_loaders(dataset, task_config['split_ratio'], collate_fn=collate_fn)
    
    # 完整配置
    config = {
        'device': task_config['device'],
        'model_class': task_config['model_class'],
        'model_args': task_config.get('model_args', {}),
        'criterion': task_config['loss_func'],
        'optimizer': optim.AdamW,
        'optim_args': {'lr': 1e-3, 'weight_decay': 1e-4},
        'metrics': task_config.get('metrics', {}),
        'train_loader': train_loader,
        'val_loader': val_loader,
        'preprocess_data': preprocess_wrapper(preprocess_data),
        'various_input_length': task_config.get('various_input_length', None),
    }
    
    # 特殊回调示例
    if task_config.get('freeze_layers', False):
        def freeze_callback(trainer, epoch):
            if epoch == task_config['freeze_epoch']:
                for name, param in trainer.model.named_parameters():  # 使用传入的 trainer 实例
                    if "shared" in name:
                        param.requires_grad = False
        config['on_epoch_start'] = freeze_callback
    
    return GenericTrainer(config)


def train_beampred_model(num_epochs, device, data_complex, labels, M_t, M_r, pos_in_data=False):
    config = {
        'device': device,
        'data_complex': data_complex,
        'labels': labels,
        'model_class': BeamPredictionModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_complex.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': labels.shape[-1],
            'num_beampair': M_t * M_r
        },
        'metrics': {
            'acc': lambda o, l: (o.argmax(-1) == l).float().mean().item(),
            'top3_acc': lambda o, l: (o.topk(3, dim=-1).indices == l.unsqueeze(-1)).any(-1).float().mean().item(),
        },
        'loss_func': lambda o, l: nn.CrossEntropyLoss()(o.view(-1,o.size(-1)), l.view(-1)),
        'split_ratio': 0.7,
        'various_input_length': None,
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)


def train_gainpred_model(num_epochs, device, data_complex, labels, M_t, M_r, pos_in_data=False):
    config = {
        'device': device,
        'data_complex': data_complex,
        'labels': labels,
        'model_class': BestGainPredictionModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_complex.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': labels.shape[-1]
        },
        'metrics': {
            'mae': lambda o, l: (o - l).abs().mean().item()*20,
            'mse': lambda o, l: ((o - l) ** 2).mean().item()*400,
            'rmse': lambda o, l: ((o - l) ** 2).mean().sqrt().item()*20,
        },
        'loss_func': nn.MSELoss(),
        'split_ratio': 0.7,
        'various_input_length': None,
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)

def train_blockpred_model(num_epochs, device, data_complex, labels, M_t, M_r, pos_in_data=False):
    config = {
        'device': device,
        'data_complex': data_complex,
        'labels': labels,
        'model_class': BlockPredictionModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_complex.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': labels.shape[-1]
        },
        'metrics': {
            'acc': lambda o, l: (o.argmax(-1) == l).float().mean().item(),
        },
        'loss_func': lambda o, l: nn.CrossEntropyLoss()(o.view(-1,o.size(-1)), l.view(-1)),
        'split_ratio': 0.7,
        'various_input_length': None,
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)

# 序列预测
def train_beampred_lstm_model(num_epochs, device, data_complex, lengths, labels, M_t, M_r, pos_in_data=False):
    config = {
        'device': device,
        'data_complex': data_complex,
        'labels': labels,
        'model_class': BeamPredictionLSTMModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_complex.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': labels.shape[-1],
            'num_beampair': M_t * M_r
        },
        'metrics': {
            'acc': lambda o, l: (o.argmax(-1) == l).float().mean().item(),
            'top3_acc': lambda o, l: (o.topk(3, dim=-1).indices == l.unsqueeze(-1)).any(-1).float().mean().item(),
        },
        'loss_func': lambda o, l: nn.CrossEntropyLoss()(o.view(-1,o.size(-1)), l.view(-1)),
        'split_ratio': 0.7,
        'various_input_length': lengths,
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)

def train_blockpred_lstm_model(num_epochs, device, data_complex, lengths, labels, M_t, M_r, pos_in_data=False):
    config = {
        'device': device,
        'data_complex': data_complex,
        'labels': labels,
        'model_class': BlockPredictionLSTMModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_complex.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': labels.shape[-1]
        },
        'metrics': {
            'acc': lambda o, l: (o.argmax(-1) == l).float().mean().item(),
        },
        'loss_func': lambda o, l: nn.CrossEntropyLoss()(o.view(-1,o.size(-1)), l.view(-1)),
        'split_ratio': 0.7,
        'various_input_length': lengths,
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)

def train_gainpred_lstm_model(num_epochs, device, data_complex, lengths, labels, M_t, M_r, pos_in_data=False):
    config = {
        'device': device,
        'data_complex': data_complex,
        'labels': labels,
        'model_class': BestGainPredictionLSTMModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_complex.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': labels.shape[-1]
        },
        'metrics': {
            'mae': lambda o, l: (o - l).abs().mean().item()*20,
            'mse': lambda o, l: ((o - l) ** 2).mean().item()*400,
            'rmse': lambda o, l: ((o - l) ** 2).mean().sqrt().item()*20,
        },
        'loss_func': nn.MSELoss(),
        'split_ratio': 0.7,
        'various_input_length': lengths,
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)
