from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import collections
import torch
import ipdb
import pickle

sys.argv = [""]
sys.path.append(os.getcwd())
import numpy as np
from utils.sim_utils import run_sim
from utils.options import args_parser
from utils.NN_utils import LSTM_Model_Mobility, BeamPredictionModel, \
    BestGainLevelPredictionModel, BestGainPredictionModel, PositionPredictionModel
from utils.alg_utils import (
    RA_Lyapunov,
    RA_heur_b,
    RA_heur_q,
    RA_heur_qb,
    RA_heur_fqb_smartRound,
    HO_EE_predG,
    HO_EE_GAP_APX_with_offload_conservative_predG,
    HO_EE_GAP_APX_with_offload_conservative_predG_SINR,
)
from utils.sumo_utils import (
    read_trajectoryInfo_carindex,
    read_trajectoryInfo_carindex_matrix,
    read_trajectoryInfo_timeindex,
)
from utils.mox_utils import setup_seed, get_save_dirs, split_string, save_log, np2torch, lin2dB, dB2lin
from utils.plot_utils import plot_beampred
from utils.data_utils import get_prepared_dataset, generate_complex_gaussian_vector
from utils.beam_utils import generate_dft_codebook, beamPairId_to_beamIdPair

def preprocess_input_np(x, params_norm=[20,7], EPS=1e-9):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    # 将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    amplitude = np.abs(x)
    dB = 20*np.log10(amplitude+EPS)
    phase = np.angle(x)
    preprocessed = np.concatenate(((dB/params_norm[0]+params_norm[1], phase)),axis=-1)
    return preprocessed


# from JCBT_AEPHORA
freq = 28e9
DS_start, DS_end = 700, 800
preprocess_mode = 0
M_r, N_bs, M_t = 8, 4, 64
P_t = 1e-1
P_noise = 1e-14
n_pilot = 16
sample_interval = int(M_t/n_pilot)
gpu = 0
device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)
args = args_parser()
args.from_sionna = True
args.M_t = M_t
args.M_r = M_r
args.slots_per_frame = 100
args.frames_per_sample = 10
args.beta_macro = 2
args.beta_micro = 2
args.bias_macro = 20*np.log10(3e8/freq/4/np.pi) # -61.38493281289306 for freq=28GHz
args.bias_micro = 20*np.log10(3e8/freq/4/np.pi)
args.shd_sigma_macro = 0
args.shd_sigma_micro = 0
args.num_RB_macro = 100
args.num_RB_micro = 100
args.RB_intervel_macro = 0.18 * 1e6
args.RB_intervel_micro = 1.8 * 1e6
args.p_macro = 1
args.p_micro = 0.1
args.data_rate = 10 * 1e6
args.lat_slot_ub = 20
args.trajectoryInfo_path = './sumo_result/trajectory_Lbd0.10.csv'
args.eta = 1e6
args.device = device
args.K = 3 # 每次beam tracking 时选K个最有可能的波束对进行测试

# prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np \
#     = get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise)
sionna_result_filepath = f'./sionna_result/trajectoryInfo_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
data4sim_filepath = f'./data4sim/trajectoryInfo_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'

if os.path.exists(data4sim_filepath):
    with open(data4sim_filepath, 'rb') as f:
        timeline_dir = pickle.load(f)
else:
    with open(sionna_result_filepath, 'rb') as f:
        timeline_dir = pickle.load(f)
        # timeline_dir[700][1.58].keys() == dict_keys(['pos', 'v', 'angle', 'h'])
    # import ipdb;ipdb.set_trace()
    DFT_matrix_tx = generate_dft_codebook(M_t)
    DFT_matrix_rx = generate_dft_codebook(M_r)
    for frame in timeline_dir.keys():
        print(f'prepare simulation data: frame[{frame-DS_start:.1f}/{DS_end-DS_start}]',)
        for veh in timeline_dir[frame].keys():
            veh_h = timeline_dir[frame][veh]['h']
            best_beam_pair_index = np.abs(np.matmul(np.matmul(veh_h, DFT_matrix_tx).T.conjugate(),DFT_matrix_rx).transpose([1,0,2]).reshape(N_bs,-1)).argmax(axis=-1)
            best_beam_index_pair = beamPairId_to_beamIdPair(best_beam_pair_index,M_t,M_r)
            timeline_dir[frame][veh]['best_beam_pair_idx'] = best_beam_pair_index
            timeline_dir[frame][veh]['best_beam_idx_pair'] = best_beam_index_pair
            g_opt = np.zeros((N_bs)).astype(np.float32)
            for bs in range(N_bs):
                g_opt[bs] = 1/np.sqrt(M_r*M_t)*np.abs(np.matmul(np.matmul(veh_h[:,bs,:], DFT_matrix_tx[:,best_beam_index_pair[bs,0]]).T.conjugate(),DFT_matrix_rx[:,best_beam_index_pair[bs,1]]))
                g_opt[bs] = 2 * lin2dB(g_opt[bs])
            timeline_dir[frame][veh]['g_opt_beam'] = g_opt
            timeline_dir[frame][veh]['g_avg'] = 2 * lin2dB(np.abs(veh_h).mean(axis=0).mean(axis=-1))
            veh_CSI = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
            n = generate_complex_gaussian_vector(veh_CSI.shape, scale=np.sqrt(P_noise), mean=0.0)
            veh_CSI = (veh_CSI + n).astype(np.complex64)
            timeline_dir[frame][veh]['CSI'] = veh_CSI
            timeline_dir[frame][veh]['CSI_preprocessed'] = preprocess_input_np(veh_CSI)
        # import ipdb;ipdb.set_trace()
    with open(data4sim_filepath, 'wb') as f:
        pickle.dump(timeline_dir,f)
        

feature_input_dim = 2 * M_r * n_pilot
num_bs = N_bs
num_beampair = M_r * M_t

mobpred_model = None
beampred_model = BeamPredictionModel(feature_input_dim, num_bs, num_beampair).to(device)
beampred_model.load_state_dict(torch.load('/home/ubuntu/niulab/JCBT_AEPHORA/NN_result/300_700_3Dbeam_tx(1,64)_rx(1,8)_freq2.8e+10_Np16_mode0/models/beampred_dimIn256_valAcc91.44%_2025-03-27_13:30:01.pth'))
gainpred_model = BestGainPredictionModel(feature_input_dim, num_bs).to(device)
gainpred_model.load_state_dict(torch.load('/home/ubuntu/niulab/JCBT_AEPHORA/NN_result/300_700_3Dbeam_tx(1,64)_rx(1,8)_freq2.8e+10_Np16_mode0/models/gainpred_dimIn256_BBSMAE2.98_2025-04-02_07:08:25.pth'))
pospred_model = PositionPredictionModel(feature_input_dim, num_bs).to(device)
pospred_model.load_state_dict(torch.load('/home/ubuntu/niulab/JCBT_AEPHORA/NN_result/300_700_3Dbeam_tx(1,64)_rx(1,8)_freq2.8e+10_Np16_mode0/models/pospred_dimIn256_valRMSE18.76_2025-03-27_14:05:24.pth'))
    

# 给定各个基站的位置
# BS0_loc = np.array([0, 0])
BS1_loc = np.array([300, 300])
BS2_loc = np.array([-300, 300])
BS3_loc = np.array([300, -300])
BS4_loc = np.array([-300, -300])
# BS_loc_list = [BS0_loc, BS1_loc, BS2_loc, BS3_loc, BS4_loc]
BS_loc_list = [BS1_loc, BS2_loc, BS3_loc, BS4_loc]
BS_loc_array = np.array(BS_loc_list)
BS_loc_dict = collections.OrderedDict()
for i, loc in enumerate(BS_loc_list):
    BS_loc_dict[i] = loc
print(BS_loc_dict)

(
    energy_record,
    HO_time_record,
    HO_cmd_record,
    violation_prob_record,
    avg_queuelen_record,
    pilot_record,
) = run_sim(
    args, BS_loc_list, timeline_dir, 
    pospred_model, 
    beampred_model,
    gainpred_model,
    RA_func=RA_heur_fqb_smartRound, 
    HO_func=HO_EE_predG,
    #HO_func=HO_EE_GAP_APX_with_offload_conservative_predG,
    prt=True,
    save_pilot=False,
    No_BF=False,
)
