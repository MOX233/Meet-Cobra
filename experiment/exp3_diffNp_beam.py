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

sys.path.append(os.getcwd())
from utils.NN_utils import BeamPredictionModel, BlockPredictionModel, BestGainPredictionModel
from utils.NN_utils import BeamPredictionLSTMModel, BlockPredictionLSTMModel, BestGainPredictionLSTMModel
from utils.NN_utils import preprocess_data
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
DS_start, DS_end = 200, 800
preprocess_mode = 0
pos_in_data = preprocess_mode==2
look_ahead_len = 10
M_t = 32
M_r = 8
n_pilot = 32 # fixed
P_t = 1e-1
P_noise = 1e-14 # -174dBm/Hz * 1.8MHz = 7.165929069962946e-15 W
lbd = 1

num_epochs =  100
cut_ratio = 1
gpu = 4
device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)

prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np \
        = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise, 
                               look_ahead_len=look_ahead_len, lbd=lbd)    
data_size_test = int(cut_ratio * data_np.shape[0])
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_exp3/lbd{lbd:.2f}_datasize{data_size_test}_beampred_"
    + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
os.makedirs(save_path, exist_ok=True)
# sionna_result_filepath = f'./sionna_result/trajectoryInfo_lbd{args.Lambda:.2f}_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
# data4sim_filepath = f'./data4sim/lbd{lbd:.2f}_{DS_start}_{DS_end}_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}_lookahead{look_ahead_len}.pkl'


data_torch, best_beam_pair_index_label, lengths = augment_dataset(
    data_np, best_beam_pair_index_np, look_ahead_len, augment_dataset_ratio=2)

random_sample_indices = random.sample(range(data_torch.shape[0]), data_size_test)
data_torch_test = data_torch[random_sample_indices,...]
lengths_test = lengths[random_sample_indices,...]
best_beam_pair_index_label_test = best_beam_pair_index_label[random_sample_indices,...]

record_metrics_list = []
best_model_weights_list = []

Np_list = [1,2,4,8,16,32]
for Np in Np_list:
    setup_seed(20)
    print(f'start training (Np={Np})')
    best_model_weights, record_metrics = \
        train_beampred_lstm_model(num_epochs, device, data_torch_test[:,:,::int(n_pilot/Np)], lengths_test, best_beam_pair_index_label_test, M_t, M_r, pos_in_data=(preprocess_mode==2))
    record_metrics_list.append(record_metrics)
    best_model_weights_list.append(best_model_weights)
    
def get_recored_bestKPIs(record_metrics_list, Np_list=[1,2,4,8,16,32]):
    bestKPI_dict = dict()
    # bestKPIs = np.zeros((len(Np_list), 6)) # Np, best_epoch, best_train_acc, best_val_acc, best_train_top3_acc, best_val_top3_acc
    for Np, record_metrics in zip(Np_list, record_metrics_list):
        bestKPI_dict[Np] = dict()
        best_epoch = np.argmax(record_metrics['val_acc'])
        bestKPI_dict[Np]['best_epoch'] = best_epoch + 1
        bestKPI_dict[Np]['best_train_acc'] = record_metrics['train_acc'][best_epoch]
        bestKPI_dict[Np]['best_val_acc'] = record_metrics['val_acc'][best_epoch]
        bestKPI_dict[Np]['best_train_top3_acc'] = record_metrics['train_top3_acc'][best_epoch]
        bestKPI_dict[Np]['best_val_top3_acc'] = record_metrics['val_top3_acc'][best_epoch]
    return bestKPI_dict

print('Best KPIs (Np, best_epoch, best_train_acc, best_val_acc, best_train_top3_acc, best_val_top3_acc):')
bestKPI_dict = get_recored_bestKPIs(record_metrics_list)

for KPI_name in bestKPI_dict[Np_list[0]].keys():
    plt.figure()
    alist = [bestKPI_dict[Np][KPI_name] for Np in Np_list]
    plt.plot(Np_list, alist, marker='o', label=KPI_name)
    plt.legend()
    plt.xlabel("Pilot number")
    # plt.xscale("log")
    plt.ylabel(KPI_name)
    plt.savefig(os.path.join(save_path, f"{KPI_name}.png"))
    # plt.savefig(os.path.join(save_path, f"{KPI_name}.pdf"))
    plt.close()

# 保存仿真实验结果指标
np.save(os.path.join(save_path, "bestKPI_dict.npy"), bestKPI_dict)
np.save(os.path.join(save_path, "best_model_weights_list.npy"), best_model_weights_list)
np.save(os.path.join(save_path, "record_metrics_list.npy"), record_metrics_list)
# import ipdb;ipdb.set_trace()