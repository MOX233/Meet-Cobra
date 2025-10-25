from __future__ import absolute_import
from __future__ import print_function

import os # Configure which GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import sys
import random
import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from utils.NN_utils import BeamPredictionModel, BlockPredictionModel, BestGainPredictionModel
from utils.NN_utils import BeamPredictionLSTMModel, BlockPredictionLSTMModel, BestGainPredictionLSTMModel
from utils.NN_utils import preprocess_data, build_trainer
from utils.options import args_parser
from utils.mox_utils import setup_seed, get_save_dirs, np2torch, save_NN_results
from utils.data_utils import get_prepared_dataset, prepare_dataset
from utils.beam_utils import generate_dft_codebook, beamIdPair_to_beamPairId, beamPairId_to_beamIdPair
from utils.NN_utils import BeamPredictionModel, train_beampred_lstm_model
from utils.options import args_parser
from utils.mox_utils import setup_seed, get_save_dirs, save_NN_results
from utils.plot_utils import plot_record_metrics
from utils.data_utils import get_prepared_dataset, prepare_dataset, augment_dataset
from utils.beam_utils import generate_dft_codebook

N_bs = 4
freq = 28e9
DS_start, DS_end = 800, 950 # test on a different scenario
preprocess_mode = 0
pos_in_data = preprocess_mode==2
look_ahead_len = 50
M_t = 32
M_r = 8
n_pilot = 8
P_t = 1e-1
P_noise = 1e-14 # -174dBm/Hz * 1.8MHz = 7.165929069962946e-15 W
lbd = 1

gpu = 4
device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)

prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np \
        = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise, 
                               look_ahead_len=look_ahead_len, lbd=lbd)    
feature_input_dim = 2 * M_r * n_pilot + 2 * int(preprocess_mode == 2)
num_bs = N_bs
num_beampair = M_r * M_t
data_size_test = int(data_np.shape[0])
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_exp4/lbd{lbd:.2f}_datasize{data_size_test}_"
    + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
os.makedirs(save_path, exist_ok=True)

DFT_tx = generate_dft_codebook(M_t)
DFT_rx = generate_dft_codebook(M_r)

InputLen_list = list(range(1,look_ahead_len+1))
beam_metrics_dict = dict()
gain_metrics_dict = dict()
for InputLen in InputLen_list:
    # import ipdb;ipdb.set_trace()
    beamPairId = best_beam_pair_index_np[:,InputLen,...]
    beamIdPair = beamPairId_to_beamIdPair(beamPairId,M_t,M_r)
    num_sample, _, _, num_bs, _ = veh_h_np.shape
    channel = veh_h_np[:,InputLen,...]
    g_opt = np.zeros((num_sample,num_bs)).astype(np.float32)
    for veh in range(num_sample):
        for bs in range(num_bs):
            g_opt[veh,bs] = 1/np.sqrt(M_t*M_r)*np.abs(np.matmul(np.matmul(channel[veh,:,bs,:], DFT_tx[:,beamIdPair[veh,bs,0]]).T.conjugate(),DFT_rx[:,beamIdPair[veh,bs,1]]))
            g_opt[veh,bs] = 20 * np.log10(g_opt[veh,bs] + 1e-9) 
    g_opt_normalized = g_opt / 20 + 7
    gain_labels_np = g_opt_normalized[:,np.newaxis,...]

    data_torch, best_beam_pair_index_label, lengths = augment_dataset(
        data_np, best_beam_pair_index_np[:,InputLen:InputLen+1,...], look_ahead_len, augment_dataset_ratio=1)
    data_torch, gain_labels, lengths = augment_dataset(
        data_np, gain_labels_np, look_ahead_len, augment_dataset_ratio=1)

    random_sample_indices = random.sample(range(data_torch.shape[0]), data_size_test)
    data_torch_test = data_torch[random_sample_indices,...]
    lengths_test = lengths[random_sample_indices,...]
    best_beam_pair_index_label_test = best_beam_pair_index_label[random_sample_indices,...]
    gain_labels_test = gain_labels[random_sample_indices,...]
    setup_seed(20)
    print(f'start test (InputLen={InputLen})')
    
    config = {
        'device': device,
        'data_complex': data_torch_test,
        'labels': best_beam_pair_index_label_test,
        'model_class': BeamPredictionLSTMModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_torch_test.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': best_beam_pair_index_label_test.shape[-1],
            'num_beampair': M_t * M_r
        },
        'metrics': {
            'acc': lambda o, l: (o.argmax(-1) == l).float().mean().item(),
            'top3_acc': lambda o, l: (o.topk(3, dim=-1).indices == l.unsqueeze(-1)).any(-1).float().mean().item(),
        },
        'loss_func': lambda o, l: nn.CrossEntropyLoss()(o.view(-1,o.size(-1)), l.view(-1)),
        'split_ratio': 0.001,
        'various_input_length': InputLen * torch.ones_like(lengths_test),
    }
    beam_trainer = build_trainer(config)
    beam_trainer.model.load_state_dict(torch.load('./NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models/beampred_lstm_valAcc89.73%_2025-09-19_21:48:48.pth', 
                                          map_location="cpu"))
    beam_trainer.model.eval()
    beam_metrics = beam_trainer.run_epoch(beam_trainer.val_loader, phase='val')
    beam_metrics_dict[InputLen] = beam_metrics
    print(beam_metrics)
    
    config = {
        'device': device,
        'data_complex': data_torch_test,
        'labels': gain_labels_test,
        'model_class': BestGainPredictionLSTMModel,
        'pos_in_data': pos_in_data,
        'model_args': {
            'feature_input_dim': data_torch_test.shape[-1] * 2 - pos_in_data * 2,  # 复数转实数的维度
            'num_bs': gain_labels_test.shape[-1]
        },
        'metrics': {
            'mae': lambda o, l: (o - l).abs().mean().item()*20,
            'mse': lambda o, l: ((o - l) ** 2).mean().item()*400,
            'rmse': lambda o, l: ((o - l) ** 2).mean().sqrt().item()*20,
        },
        'loss_func': nn.MSELoss(),
        'split_ratio': 0.001,
        'various_input_length': InputLen * torch.ones_like(lengths_test),
    }
    gain_trainer = build_trainer(config)
    gain_trainer.model.load_state_dict(torch.load('./NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models/gainpred_lstm_valMae4.07dB_2025-09-25_02:04:34.pth', 
                                          map_location="cpu"))
    gain_trainer.model.eval()
    gain_metrics = gain_trainer.run_epoch(gain_trainer.val_loader, phase='val')
    gain_metrics_dict[InputLen] = gain_metrics
    print(gain_metrics)

for KPI_name in gain_metrics_dict[InputLen_list[0]].keys():
    plt.figure()
    alist = [gain_metrics_dict[InputLen][KPI_name] for InputLen in InputLen_list]
    plt.plot(InputLen_list, alist, marker='o', label=KPI_name)
    plt.legend()
    plt.xlabel("Input Length")
    # plt.xscale("log")
    plt.ylabel(KPI_name)
    plt.savefig(os.path.join(save_path, f"GainPred_{KPI_name}.png"))
    # plt.savefig(os.path.join(save_path, f"{KPI_name}.pdf"))
    plt.close()
    
for KPI_name in beam_metrics_dict[InputLen_list[0]].keys():
    plt.figure()
    alist = [beam_metrics_dict[InputLen][KPI_name] for InputLen in InputLen_list]
    plt.plot(InputLen_list, alist, marker='o', label=KPI_name)
    plt.legend()
    plt.xlabel("Input Length")
    # plt.xscale("log")
    plt.ylabel(KPI_name)
    plt.savefig(os.path.join(save_path, f"BeamPred_{KPI_name}.png"))
    # plt.savefig(os.path.join(save_path, f"{KPI_name}.pdf"))
    plt.close()

# 保存仿真实验结果指标
np.save(os.path.join(save_path, "gain_metrics_dict.npy"), gain_metrics_dict)
np.save(os.path.join(save_path, "beam_metrics_dict.npy"), beam_metrics_dict)