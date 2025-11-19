from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import torch
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import collections
import ipdb
import pickle

sys.path.append(os.getcwd())
from utils.NN_utils import BeamPredictionLSTMModel, BestGainPredictionLSTMModel
from utils.sim_utils import run_sim_withUMa
from utils.options import args_parser
from utils.alg_utils import (
    RA_unlimitRB,
    RA_fqb,
    RA_PF,
    RA_UTO,
    RA_UTPF,
    HO_EE_Greedy,
    HO_EE_GAP_APX_with_offload,
)
from utils.mox_utils import setup_seed, get_save_dirs, split_string, save_log, np2torch, lin2dB, dB2lin, generate_1Dsamples
from utils.data_utils import get_prepared_dataset, generate_complex_gaussian_vector
from utils.plot_utils import plot_beampred
from utils.beam_utils import generate_dft_codebook, beamPairId_to_beamIdPair

def preprocess_input_np(x, params_norm=[20,7], EPS=1e-9):
    assert (x.dtype == np.complex64) or (x.dtype == np.complex128)
    # 将复数类型的信道增益转为实数类型的幅值(dB+normalization)与相位
    amplitude = np.abs(x)
    dB = 20*np.log10(amplitude+EPS)
    phase = np.angle(x)
    preprocessed = np.concatenate(((dB/params_norm[0]+params_norm[1], phase)),axis=-1)
    return preprocessed


if __name__ == "__main__":
    # Urban Macro LoS: PL = 28 + 22*log10(d)+20*log10(f)
    # Urban Micro LoS: PL = 32.4 + 21*log10(d)+20*log10(f)
    N_bs = 4
    freq = 28e9
    DS_start, DS_end = 800, 950 # test on a different scenario
    preprocess_mode = 0
    pos_in_data = preprocess_mode==2
    look_ahead_len = 10
    M_t = 32
    M_r = 8
    n_pilot = 8
    P_t = 1e-1
    P_noise = 1e-14 # -174dBm/Hz * 1.8MHz = 7.165929069962946e-15 W
    lbd = 1
    sample_interval = int(M_t/n_pilot)
    gpu = 3
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print('device: ',device)
    args = args_parser()
    args.from_sionna = True
    args.M_t = M_t
    args.M_r = M_r
    args.slots_per_frame = 100
    args.frames_per_sample = 10
    args.num_RB_macro = 100
    args.num_RB_micro = 100
    args.RB_intervel_macro = 0.36 * 1e6
    args.RB_intervel_micro = 1.8 * 1e6
    args.p_macro = 1
    args.p_micro = 0.2
    args.NF_macro_dB = 5
    args.NF_micro_dB = 10
    args.data_rate = 10 * 1e6
    args.lat_slot_ub = 20
    args.eta = 1e6
    args.device = device
    args.K = 3 # 每次beam tracking 时选K个最有可能的波束对进行测试
    args.Lambda = 1 # 车辆到达率
    args.data_rate = 90e6
    random_factor_range4data_rate_list = np.linspace(0, 1, 11)
    args.trajectoryInfo_path = f'./sumo_data/trajectory_Lbd{args.Lambda:.2f}.csv'
    # 对测试数据集进行截断
    cut_ratio = 1
    cut_end = DS_start + cut_ratio*(DS_end-DS_start)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_exp6/lbd{args.Lambda:.2f}_{DS_start}_{cut_end}_"
        + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./sionna_result', exist_ok=True)
    os.makedirs('./data4sim', exist_ok=True)
    sionna_result_filepath = f'./sionna_result/trajectoryInfo_lbd{args.Lambda:.2f}_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
    data4sim_filepath = f'./data4sim/lbd{args.Lambda:.2f}_{DS_start}_{DS_end}_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}_lookahead{look_ahead_len}.pkl'
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if os.path.exists(data4sim_filepath):
        with open(data4sim_filepath, 'rb') as f:
            timeline_dir = pickle.load(f)
    else:
        with open(sionna_result_filepath, 'rb') as f:
            timeline_dir = pickle.load(f)
        DFT_matrix_tx = generate_dft_codebook(M_t)
        DFT_matrix_rx = generate_dft_codebook(M_r)
        frame_prev = None
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
                # timeline_dir[frame][veh]['CSI'] = veh_CSI
                timeline_dir[frame][veh]['CSI_preprocessed'] = preprocess_input_np(veh_CSI)
                if preprocess_mode == 2:
                    veh_pos = timeline_dir[frame][veh]['pos']
                    timeline_dir[frame][veh]['CSI_preprocessed'] = \
                        np.concatenate((timeline_dir[frame][veh]['CSI_preprocessed'], veh_pos/100), axis=-1)
                if frame_prev is not None and veh in timeline_dir[frame_prev].keys():
                    timeline_dir[frame][veh]['CSI_preprocessed'] = \
                        np.concatenate((timeline_dir[frame_prev][veh]['CSI_preprocessed'], 
                                        timeline_dir[frame][veh]['CSI_preprocessed'].reshape(1, -1)),
                                        axis=0)[-look_ahead_len:,...]
                else:
                    timeline_dir[frame][veh]['CSI_preprocessed'] = timeline_dir[frame][veh]['CSI_preprocessed'].reshape(1, -1)
            frame_prev = frame
        with open(data4sim_filepath, 'wb') as f:
            pickle.dump(timeline_dir,f)
    
    _timeline_dir = collections.OrderedDict()
    for frame,v in timeline_dir.items():
        if frame>=cut_end:
            break
        _timeline_dir[frame] = v
    timeline_dir = _timeline_dir
    
    avg_car_num = sum([len(timeline_dir[frame]) for frame in timeline_dir.keys()]) / len(timeline_dir.keys())
            
    feature_input_dim = 2 * M_r * n_pilot + 2 * int(preprocess_mode == 2)
    num_bs = N_bs
    num_beampair = M_r * M_t

    beampred_model = BeamPredictionLSTMModel(feature_input_dim, num_bs, num_beampair).to(device)
    beampred_model.load_state_dict(torch.load('./NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models/beampred_lstm_valAcc89.73%_2025-09-19_21:48:48.pth'))
    beampred_model.eval()
    gainpred_model = BestGainPredictionLSTMModel(feature_input_dim, num_bs).to(device)
    gainpred_model.load_state_dict(torch.load('./NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models/gainpred_lstm_valMae4.07dB_2025-09-25_02:04:34.pth'))
    gainpred_model.eval()
    pospred_model = None
        
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

    # 给出所需要仿真的方案名和PHO,RA策略
    sim_strategy_dict = collections.OrderedDict()
    
    sim_strategy_dict["MEET-COBRA (PredInfo)"] = {
        "RA": RA_UTO,
        "HO": HO_EE_GAP_APX_with_offload,
        "save_pilot": True,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "NoBF": False,
    }
    
    # sim_strategy_dict["MEET-COBRA (TrueInfo)"] = {
    #     "RA": RA_heur_QPOS, 
    #     "HO": HO_EE_GAP_APX_with_offload_conservative_predG,
    #     "save_pilot": True,
    #     "gainpred_model": None,
    #     "beampred_model": None,
    #     "NoBF": False,
    # }
    
    sim_strategy_dict["MEET-COBRA (NoBF)"] = {
        "RA": RA_UTO, 
        "HO": HO_EE_GAP_APX_with_offload,
        "save_pilot": False,
        "gainpred_model": None,
        "beampred_model": None,
        "NoBF": True,
    }
    
    sim_strategy_dict["GreedyPHO (PredInfo)"] = {
        "RA": RA_UTO, 
        "HO": HO_EE_Greedy,
        "save_pilot": True,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "NoBF": False,
    }
    
    sim_strategy_dict["PropFair (PredInfo)"] = {
        "RA": RA_PF,
        "HO": HO_EE_GAP_APX_with_offload,
        "save_pilot": True,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "NoBF": False,
    }
    
    # sim_strategy_dict["GreedyPHO (TrueInfo)"] = {
    #     "RA": RA_heur_QPOS, 
    #     "HO": HO_EE_predG,
    #     "save_pilot": True,
    #     "gainpred_model": None,
    #     "beampred_model": None,
    #     "NoBF": False,
    # }
    
    # sim_strategy_dict["GreedyPHO (NoBF)"] = {
    #     "RA": RA_heur_QPOS, 
    #     "HO": HO_EE_predG,
    #     "save_pilot": True,
    #     "gainpred_model": None,
    #     "beampred_model": None,
    #     "NoBF": True,
    # }
    
    # sim_strategy_dict["LowerBound"] = {
    #     "RA": RA_unlimitRB,
    #     "HO": HO_EE_predG,
    #     "save_pilot": False,
    #     "gainpred_model": None,
    #     "beampred_model": None,
    #     "NoBF": False,
    # }
    
    sim_result_dict = collections.OrderedDict()
    for strategy_name in sim_strategy_dict.keys():
        sim_result_dict[strategy_name] = {
            "avg_system_power_list": [],
            "HOps_list": [],
            "carnum_under_BS_list": [],
            "vio_prob_list": [],
            "avg_queue_len_list": [],
            "avg_latency_list": [],
            "avg_pilot_list": [],
        }

    # 进行仿真实验
    for data_rate_idx, random_factor_range4data_rate in enumerate(random_factor_range4data_rate_list):
        print(f"random_factor_range4data_rate: {random_factor_range4data_rate:.2f}")
        for strategy_name in sim_strategy_dict.keys():
            print("Strategy: ", strategy_name)
            args.random_factor_range4data_rate = random_factor_range4data_rate
            (
                energy_record,
                HO_time_record,
                HO_cmd_record,
                violation_prob_record,
                avg_queuelen_record,
                pilot_record,
            ) = run_sim_withUMa(
                args, BS_loc_list, timeline_dir, 
                pospred_model, 
                beampred_model=sim_strategy_dict[strategy_name]["beampred_model"],
                gainpred_model=sim_strategy_dict[strategy_name]["gainpred_model"],
                RA_func=sim_strategy_dict[strategy_name]["RA"], 
                HO_func=sim_strategy_dict[strategy_name]["HO"],
                prt=False,
                save_pilot=sim_strategy_dict[strategy_name]["save_pilot"],
                No_BF=sim_strategy_dict[strategy_name]["NoBF"],
            )
            # TODO：HO_cmd_record
            # import ipdb; ipdb.set_trace()
            carnum_under_BS = np.zeros((len(HO_cmd_record.keys())-1,len(BS_loc_list)+1,))
            for frame in range(1, len(HO_cmd_record.keys())):
                for BS_id in HO_cmd_record[frame].values():
                    carnum_under_BS[frame-1, BS_id] += 1
            
            avg_system_power = energy_record.mean() / (
                args.slots_per_frame * args.slot_len
            )
            HOps = HO_time_record[2:].mean() / (args.slots_per_frame * args.slot_len)
            vio_prob = violation_prob_record[2:].mean() * 100
            avg_queue_len = avg_queuelen_record[2:].mean()
            avg_latency = avg_queuelen_record[2:].mean() / args.data_rate * 1000
            avg_pilot = pilot_record[2:].mean()
            sim_result_dict[strategy_name]["avg_system_power_list"].append(avg_system_power)
            sim_result_dict[strategy_name]["HOps_list"].append(HOps)
            sim_result_dict[strategy_name]["vio_prob_list"].append(vio_prob)
            sim_result_dict[strategy_name]["avg_queue_len_list"].append(avg_queue_len)
            sim_result_dict[strategy_name]["avg_latency_list"].append(avg_latency)
            sim_result_dict[strategy_name]["avg_pilot_list"].append(avg_pilot)
            sim_result_dict[strategy_name]["carnum_under_BS_list"].append(carnum_under_BS)
           
            

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                random_factor_range4data_rate_list[: data_rate_idx + 1]/1e6,
                sim_result_dict[strategy_name]["avg_system_power_list"][
                    : data_rate_idx + 1
                ],
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("random_factor_range4data_rate")
        # plt.xscale("log")
        plt.ylabel("Average system power (W)")
        plt.savefig(os.path.join(save_path, "Average system power.png"))
        plt.savefig(os.path.join(save_path, "Average system power.pdf"))
        plt.close()

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                random_factor_range4data_rate_list[: data_rate_idx + 1]/1e6,
                np.array(sim_result_dict[strategy_name]["HOps_list"][: data_rate_idx + 1]) / avg_car_num,
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("random_factor_range4data_rate")
        # plt.xscale("log")
        plt.ylabel("Average HO frequency per vehicle (1/s)")
        plt.savefig(os.path.join(save_path, "Average HO frequency.png"))
        plt.savefig(os.path.join(save_path, "Average HO frequency.pdf"))
        plt.close()

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                random_factor_range4data_rate_list[: data_rate_idx + 1]/1e6,
                sim_result_dict[strategy_name]["vio_prob_list"][: data_rate_idx + 1],
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("random_factor_range4data_rate")
        # plt.xscale("log")
        plt.ylim(0, 100)
        plt.ylabel("Violation probability (%)")
        plt.savefig(os.path.join(save_path, "Violation probability.png"))
        plt.savefig(os.path.join(save_path, "Violation probability.pdf"))
        plt.close()

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                random_factor_range4data_rate_list[: data_rate_idx + 1]/1e6,
                sim_result_dict[strategy_name]["avg_latency_list"][: data_rate_idx + 1],
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("random_factor_range4data_rate")
        # plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Average latency (ms)")
        plt.savefig(os.path.join(save_path, "Average latency.png"))
        plt.savefig(os.path.join(save_path, "Average latency.pdf"))
        plt.close()
        
        plt.figure(figsize=(6, 4*len(BS_loc_list)))
        plt.xlabel("random_factor_range4data_rate")
        plt.ylabel("Average car number under each BS")
        for BS_id in range(len(BS_loc_list)+1):
            plt.subplot(len(BS_loc_list)+1, 1, BS_id + 1)
            for strategy_name in sim_strategy_dict.keys():
                avg_carnum_under_BS_list = np.array(sim_result_dict[strategy_name]["carnum_under_BS_list"][: data_rate_idx + 1]).mean(axis=-2)
                plt.plot(
                    random_factor_range4data_rate_list[: data_rate_idx + 1]/1e6,
                    avg_carnum_under_BS_list[:, BS_id],
                    "*-",
                    label=f"{strategy_name} BS{BS_id}",
                )
            plt.legend()
        plt.savefig(os.path.join(save_path, "Average car number under each BS.png"))
        plt.savefig(os.path.join(save_path, "Average car number under each BS.pdf"))
        plt.close()
        

        # 保存仿真实验设置
        sim_result_dict["args"] = args
        sim_result_dict["random_factor_range4data_rate_list"] = random_factor_range4data_rate_list
        # 保存仿真实验结果指标
        np.save(os.path.join(save_path, "sim_result_dict.npy"), sim_result_dict)
        # 保存仿真实验设置
