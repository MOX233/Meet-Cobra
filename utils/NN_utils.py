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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair, generate_dft_codebook
import numpy as np

EPS = 1e-9

# 定义神经网络模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim):
        super(MLP, self).__init__()
        self.num_fc = len(hidden_dim_list) + 1
        self.dim_list = copy.copy(hidden_dim_list)
        self.dim_list.insert(0, input_dim)
        self.dim_list.append(output_dim)
        self.network = nn.Sequential()
        for i in range(self.num_fc):
            self.network.add_module(
                "fc" + str(i), nn.Linear(self.dim_list[i], self.dim_list[i + 1])
            )
            if i < self.num_fc - 1:
                self.network.add_module("relu" + str(i), nn.ReLU())

    def forward(self, x):
        x = self.network(x)
        return x

class LSTM_Model_abc(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        mlp_hidden_dim_list=[32, 16, 32],
        drop_out=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(LSTM_Model_abc, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.mlp_hidden_dim_list = mlp_hidden_dim_list
        self.drop_out = drop_out
        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_dim,
            hidden_size=hidden_dim,  # rnn hidden unit
            num_layers=layer_dim,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=drop_out,
        ).to(self.device)
        self.mlp = MLP(hidden_dim, mlp_hidden_dim_list, output_dim).to(self.device)
        for name, param in self.rnn.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        """for name, param in self.mlp.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)"""

    def forward(self, x, hc=None):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # c_n shape (n_layers, batch, hidden_size)
        r_out, (h_n, c_n) = self.rnn(
            x, hc
        )  # hc=None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.mlp(r_out[:, -1, :])
        return out

    @abc.abstractmethod
    def predict(self):
        pass

class LSTM_Model_Mobility(LSTM_Model_abc):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        mlp_hidden_dim_list=[32, 16, 32],
        drop_out=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        preprocess_params=None,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            layer_dim,
            output_dim,
            mlp_hidden_dim_list,
            drop_out,
            device,
        )
        if preprocess_params is None:
            preprocess_params = {
                "muX": np.array(
                    [-134.40944, -162.64276, -159.56888, -161.54985, -159.29124],
                    dtype=np.float32,
                ),
                "sigmaX": np.array(
                    [2.761759, 13.521982, 15.282211, 14.245071, 14.93797],
                    dtype=np.float32,
                ),
                "muY": -23.416536,
                "sigmaY": 221.9446,
            }
        self.preprocess_params = preprocess_params

    def load_preprocess_params(self, preprocess_params):
        if preprocess_params is dict or preprocess_params is collections.OrderedDict:
            self.preprocess_params = preprocess_params
        elif preprocess_params is str:
            self.preprocess_params = np.load(
                os.path.join(preprocess_params),
                allow_pickle=True,
            ).item()
        else:
            exit("load failure")

    def predict(self, X):
        self.eval()
        assert type(X) == np.ndarray
        # X = X[np.newaxis,...].astype(np.float32) if X.ndim == 2 else X
        # X = X.astype(np.float32)
        if X.ndim == 3:
            X = X.astype(np.float32)
            X = (
                X - self.preprocess_params["muX"].reshape(1, 1, -1)
            ) / self.preprocess_params["sigmaX"].reshape(1, 1, -1)
            X = torch.from_numpy(X).to(self.device)
            out = self.forward(X).unsqueeze(-2)
            Y = out.detach().cpu().numpy()
            Y = Y * self.preprocess_params["sigmaY"] + self.preprocess_params["muY"]
        elif X.ndim == 2:
            X = X[np.newaxis, ...].astype(np.float32)
            X = (
                X - self.preprocess_params["muX"].reshape(1, 1, -1)
            ) / self.preprocess_params["sigmaX"].reshape(1, 1, -1)
            X = torch.from_numpy(X).to(self.device)
            out = self.forward(X).unsqueeze(-2)
            Y = out.detach().cpu().numpy()
            Y = Y * self.preprocess_params["sigmaY"] + self.preprocess_params["muY"]
            Y = Y.squeeze(0)
        else:
            exit("X.ndim error!")
        return Y

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
        
    def _build_shared_layers(self, layer_dim_list):
        shared_layers = nn.Sequential()
        for i in range(len(layer_dim_list)-1):
            shared_layers.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
            shared_layers.append(nn.GELU())
        return shared_layers

    def _build_output_heads(self, num_head, layer_dim_list):
        output_heads = nn.ModuleList()
        for _ in range(num_head):
            head = nn.Sequential()
            for i in range(len(layer_dim_list)-1):
                head.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
                head.append(nn.GELU())
            head.pop(-1)
            output_heads.append(head)
        return output_heads

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
    
    
class PositionPredictionModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, pos_scale=100, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        self.pos_scale = pos_scale
        
        # 共享特征提取层
        self.shared_layers = self._build_shared_layers(
            layer_dim_list=[feature_input_dim, 256, 256, 128, 128, 64, 64, 32, 2]
        )
        self.shared_layers.pop(-1)
    
    def forward(self, x):
        outputs = self.shared_layers(x)
        return outputs

    def pred(self, x):
        outputs = self.forward(x)
        predicted_pos = outputs * self.pos_scale
        return predicted_pos
    
    def predict(self, x_np, device):
        x_torch = torch.tensor(x_np).to(device)
        return self.pred(x_torch).detach().cpu().numpy()
    

class BestGainLevelPredictionModel(BasePredictionModel):
    def __init__(self, feature_input_dim, num_bs, num_dBlevel=64, LB_db=-110, UB_db=-20, params_norm=[20,7]):
        super().__init__(feature_input_dim, num_bs, params_norm)
        self.num_dBlevel = num_dBlevel
        self.LB_db = LB_db
        self.UB_db = UB_db
        
        # 共享特征提取层
        self.shared_layers = self._build_shared_layers(
            layer_dim_list=[feature_input_dim, 256, 256, 128]
        )
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 128, 128, 128, self.num_dBlevel]
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        outputs = [head(shared_features) for head in self.output_heads]
        return torch.stack(outputs, dim=-2)

    def pred(self, x):
        pass 
        # outputs = self.forward(x)
        # predicted_bestgain = self.params_norm[0]*( outputs - self.params_norm[1] )
        # return predicted_bestgain
    
    def trans_dB_to_dBLevel(self, dB, num_dBlevel=64, LB_db=-110, UB_db=-20):
        if self != None:
            dBlevel = (dB > self.LB_db) * np.ceil((dB - self.LB_db) / (self.UB_db - self.LB_db) * (self.num_dBlevel-1))
            dBlevel = dBlevel.astype(np.int64)
        else:
            dBlevel = (dB > LB_db) * np.ceil((dB - LB_db) / (UB_db - LB_db) * (num_dBlevel-1))
            dBlevel = dBlevel.astype(np.int64)
        return dBlevel


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
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 64, 32, 64, 128, self.num_class]
        )
    
    def forward(self, x, hc=None):
        shared_features, (h_c, c_n) = self.lstm_layers(x, hc)
        outputs = [head(shared_features[...,-1,:]) for head in self.output_heads]
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
        
        # 共享特征提取层
        self.lstm_layers = self._build_LSTM_layers(
            input_dim=feature_input_dim,
            hidden_dim=128,
            layer_dim=1,
            dropout=0
        )
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 64, 32, 64, 128, 2]
        )
    
    def forward(self, x, hc=None):
        shared_features, (h_c, c_n) = self.lstm_layers(x, hc)
        outputs = [head(shared_features[...,-1,:]) for head in self.output_heads]
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
        
        # 多任务输出层（每个基站一个输出头）
        self.output_heads = self._build_output_heads(
            num_head=self.num_bs, layer_dim_list=[128, 64, 64, 32, 1]
        )
    
    def forward(self, x, hc=None):
        shared_features, (h_c, c_n) = self.lstm_layers(x, hc)
        outputs = [head(shared_features[...,-1,:]) for head in self.output_heads]
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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                    
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
            self.on_epoch_start(self, epoch)  # 触发回调
            
            train_metrics = self.run_epoch(self.train_loader, 'train')
            val_metrics = self.run_epoch(self.val_loader, 'val') if self.val_loader else {}
            
            # 合并指标并打印
            epoch_metrics = {**train_metrics, **val_metrics}
            for k, v in epoch_metrics.items():
                record_metrics[k].append(v)
            print(f"Epoch {epoch+1}/{num_epochs}")
            # print(' | '.join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))
            print(' | '.join([f"{k1.split('train_')[-1]}: {v1:.4f}/{v2:.4f}" for (k1,v1),(k2,v2) in zip(train_metrics.items(),val_metrics.items())]))
            
            # 保存最佳模型逻辑
            if 'val_loss' in epoch_metrics and epoch_metrics['val_loss'] < best_metrics.get('val_loss', float('inf')):
                best_metrics = epoch_metrics
                best_model_weights = self.model.state_dict()
        
        return best_model_weights, record_metrics

def create_loaders(dataset, split_ratio=0.7, train_batch=128, val_batch=256, seed=20):
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
        generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=val_batch)
    return train_loader, val_loader

def build_trainer(task_config):
    # 数据预处理封装
    def preprocess_wrapper(func):
        return lambda data: func(data, task_config['pos_in_data'])
    
    # 构建数据加载器
    dataset = TensorDataset(
        preprocess_data(task_config['data_complex'], task_config['pos_in_data']),
        task_config['labels']
    )
    train_loader, val_loader = create_loaders(dataset, task_config['split_ratio'])
    
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
        'preprocess_data': preprocess_wrapper(preprocess_data)
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
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)

def train_pospred_model(num_epochs, device, data_complex, labels, M_t, M_r, pos_in_data=False):
    config = {
        'device': device,
        'data_complex': data_complex,
        'labels': labels,
        'model_class': PositionPredictionModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_complex.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': labels.shape[-1]
        },
        'metrics': {
            'mae': lambda o, l: (o - l).square().sum(-1).sqrt().mean().item()*100,
            'rmse': lambda o, l: (o - l).square().sum(-1).mean().sqrt().item()*100,
        },
        'loss_func': nn.MSELoss(),
        'split_ratio': 0.7,
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)



# 序列预测
def train_beampred_lstm_model(num_epochs, device, data_complex, labels, M_t, M_r, pos_in_data=False):
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
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)

def train_blockpred_lstm_model(num_epochs, device, data_complex, labels, M_t, M_r, pos_in_data=False):
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
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)

def train_gainpred_lstm_model(num_epochs, device, data_complex, labels, M_t, M_r, pos_in_data=False):
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
    }
    trainer = build_trainer(config)
    return trainer.train(num_epochs)
