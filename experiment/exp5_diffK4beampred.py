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
DS_start, DS_end = 800, 900 # test on a different scenario
preprocess_mode = 0
pos_in_data = preprocess_mode==2
look_ahead_len = 10
M_t = 32
M_r = 8
n_pilot = 8
P_t = 1e-1
P_noise = 1e-14 # -174dBm/Hz * 1.8MHz = 7.165929069962946e-15 W
lbd = 1

gpu = 1
device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)

num_beampair = M_t * M_r
prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np \
        = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise, 
                               look_ahead_len=look_ahead_len, lbd=lbd)    

data_size_test = int(data_np.shape[0] * 1)
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_exp5/lbd{lbd:.2f}_datasize{data_size_test}_"
    + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
os.makedirs(save_path, exist_ok=True)
# sionna_result_filepath = f'./sionna_result/trajectoryInfo_lbd{args.Lambda:.2f}_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
# data4sim_filepath = f'./data4sim/lbd{lbd:.2f}_{DS_start}_{DS_end}_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}_lookahead{look_ahead_len}.pkl'

data_torch, best_beam_pair_index_label, lengths = augment_dataset(
    data_np, best_beam_pair_index_np, look_ahead_len, augment_dataset_ratio=1)

random_sample_indices = random.sample(range(data_torch.shape[0]), data_size_test)
data_torch_test = data_torch[random_sample_indices,...]
lengths_test = lengths[random_sample_indices,...]
best_beam_pair_index_label_test = best_beam_pair_index_label[random_sample_indices,...]

feature_input_dim = 2 * M_r * n_pilot + 2 * int(preprocess_mode == 2)
num_bs = N_bs
num_beampair = M_r * M_t

InputLen = 10
setup_seed(20)
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
        'topK_acc_list': lambda o, l: np.array([(o.topk(K, dim=-1).indices == l.unsqueeze(-1)).any(-1).float().mean().item() for K in range(1, num_beampair+1)])
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
print(beam_metrics)

# import ipdb;ipdb.set_trace()
plt.figure()
plt.plot(np.arange(1, num_beampair+1), beam_metrics['val_topK_acc_list'])
plt.xlabel("K in Top-K Accuracy")
plt.ylabel("Top-K Accuracy")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(save_path, f"BeamPred_TopK_acc.png"))
plt.close()
# plt.figure()
# alist = [beam_metrics_dict[InputLen][KPI_name] for InputLen in InputLen_list]
# plt.plot(InputLen_list, alist, marker='o', label=KPI_name)
# plt.legend()
# plt.xlabel("Input Length")
# # plt.xscale("log")
# plt.ylabel(KPI_name)
# plt.savefig(os.path.join(save_path, f"BeamPred_{KPI_name}.png"))
# # plt.savefig(os.path.join(save_path, f"{KPI_name}.pdf"))
# plt.close()

# 保存仿真实验结果指标
np.save(os.path.join(save_path, "beam_metrics.npy"), beam_metrics)