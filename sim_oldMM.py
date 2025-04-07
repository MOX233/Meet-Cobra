from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import collections
import torch

sys.argv = [""]
sys.path.append(os.getcwd())
import numpy as np
from utils.sim_utils import run_sim
from utils.options import args_parser
from utils.NN_utils import LSTM_Model_Mobility
from utils.alg_utils import (
    RA_Lyapunov,
    RA_heur_b,
    RA_heur_q,
    RA_heur_qb,
    HO_nearest,
    HO_EE,
    HO_SE,
    HO_SE_loadbalance,
    HO_EE_loadbalance,
)
from utils.sumo_utils import (
    read_trajectoryInfo_carindex,
    read_trajectoryInfo_carindex_matrix,
    read_trajectoryInfo_timeindex,
)

args = args_parser()
args.slots_per_frame = 100
args.beta_macro = 3
args.beta_micro = 4
args.bias_macro = -60
args.bias_micro = -60
args.shd_sigma_macro = 0
args.shd_sigma_micro = 0
args.num_RB_macro = 100
args.num_RB_micro = 100
args.RB_intervel_macro = 0.18 * 1e6
args.RB_intervel_micro = 1.8 * 1e6
args.p_macro = 1
args.p_micro = 0.1
args.data_rate = 0.7 * 1e6
args.trajectoryInfo_path = './sumo_result/trajectory_Lbd0.10.csv'
timeline_dir = read_trajectoryInfo_timeindex(
    args,
    start_time=500,
    end_time=510,
    display_intervel=0.05,
)

args.eta = 1e6


# NN model 参数指定
args.gpu = 3
save_path = "./save/"
hidden_size = 128
num_layers = 2
drop_out = 0
mlp_hidden_dim_list = [64, 64]
args.frame_per_sample = 10
args.frames_per_sample = 10
device = torch.device(
    "cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu"
)
model = LSTM_Model_Mobility(
    input_dim=5,
    hidden_dim=hidden_size,
    layer_dim=num_layers,
    output_dim=2,
    mlp_hidden_dim_list=mlp_hidden_dim_list,
    drop_out=drop_out,
    device=device,
    preprocess_params=None,
)
model_save_path = os.path.join(
    save_path,
    f"model/shd[{args.shd_sigma_macro},{args.shd_sigma_micro}]fps{args.frame_per_sample}/lstm[H{hidden_size}L{num_layers}D{drop_out}],mlp{mlp_hidden_dim_list}.npz",
)
model.load_state_dict(torch.load(model_save_path))


# 给定各个基站的位置
BS0_loc = np.array([0, 0])
BS1_loc = np.array([300, 300])
BS2_loc = np.array([300, -300])
BS3_loc = np.array([-300, 300])
BS4_loc = np.array([-300, -300])
BS_loc_list = [BS0_loc, BS1_loc, BS2_loc, BS3_loc, BS4_loc]
BS_loc_array = np.array(BS_loc_list)
BS_dict = collections.OrderedDict()
for i, loc in enumerate(BS_loc_list):
    BS_dict[i] = loc
print(BS_dict)

(
    energy_record,
    HO_time_record,
    HO_cmd_record,
    violation_prob_record,
    avg_queuelen_record,
) = run_sim(
    args, BS_loc_list, timeline_dir, model, RA_func=RA_heur_q, HO_func=HO_EE_loadbalance
)
