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
from utils.sim_utils import *
from utils.alg_utils import *
from utils.options import args_parser
from utils.mox_utils import setup_seed
from utils.plot_utils import plt_color_list, plt_linestyle_list, plt_marker_list


if __name__ == "__main__":
    gpu = 2
    lbd = 1
    cut_ratio = 0.01
    # cut_ratio = 1/5
    data_rate_list = np.linspace(1e6, 35e6, 18)[:]
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_paper_exp1")
    
    args, BS_loc_list, timeline_dir, pospred_model, beampred_model, gainpred_model, inferpred_model, save_path = \
        get_default_sim_params(save_dir, gpu, lbd, cut_ratio)
    
    # 给出所需要仿真的方案名和PHO,RA策略
    sim_strategy_dict = collections.OrderedDict()
    
    sim_strategy_dict["Proposed"] = {
        "RA": RA_OTR_SINR, 
        "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
        "BF": "topKbeam_savePilot",
        "save_pilot": True,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "inferpred_model": inferpred_model,
        "NoBF": False,
        "K_BF": 5,
    }
    
    sim_strategy_dict["Oracle-MC"] = {
        "RA": RA_OTR_SINR, 
        "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
        "BF": "topKbeam_savePilot",
        "save_pilot": True,
        "gainpred_model": None,
        "beampred_model": None,
        "inferpred_model": None,
        "NoBF": False,
        "K_BF": 5,
    }
    
    sim_strategy_dict["Oracle-LP-LB"] = {
        "RA": RA_OTR_SINR, 
        "HO": HO_LowerBound_SINR,
        "BF": "topKbeam_savePilot",
        "save_pilot": True,
        "gainpred_model": None,
        "beampred_model": None,
        "inferpred_model": None,
        "NoBF": False,
        "K_BF": 1,
    }
    
    # Reactive-OBRA
    sim_strategy_dict["Reactive-OBRA"] = {
        "RA": RA_OTR3_SINR, 
        "HO": HO_EE_Greedy_offload,
        "BF": "topKbeam_NoPred",
        "save_pilot": False,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "inferpred_model": inferpred_model,
        "NoBF": False,
        "K_BF": 5,
    }
    
        
    # # EGLA-HO: Energy-Greedy and Load-Aware Handover 
    sim_strategy_dict["w/o GAP-HO"] = {
        "RA": RA_OTR_SINR, 
        "HO": HO_EE_Greedy_offload,
        "BF": "topKbeam_savePilot",
        "save_pilot": True,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "inferpred_model": inferpred_model,
        "NoBF": False,
        "K_BF": 5,
    }
    
    sim_strategy_dict["w/o PET-BF"] = {
        "RA": RA_OTR_SINR, 
        "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
        "BF": "topKbeam_NoPred",
        "save_pilot": False,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "inferpred_model": inferpred_model,
        "NoBF": False,
        "K_BF": 5,
    }
    
    sim_strategy_dict["w/o OTR-RA"] = {
        "RA": RA_OTR3_SINR, 
        "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
        "BF": "topKbeam_savePilot",
        "save_pilot": True,
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "inferpred_model": inferpred_model,
        "NoBF": False,
        "K_BF": 5,
    }
    
    
    sim_result_dict = collections.OrderedDict()
    for i, strategy_name in enumerate(sim_strategy_dict.keys()):
        sim_result_dict[strategy_name] = {
            "avg_system_power_list": [],
            "HOps_list": [],
            "carnum_under_BS_list": [],
            "vio_prob_list": [],
            "avg_queue_len_list": [],
            "avg_latency_list": [],
            "avg_pilot_list": [],
            "queuelen_4eachVeh_record_list": [],
        }

    # 进行仿真实验
    for data_rate_idx, data_rate in enumerate(data_rate_list):
        setup_seed(args.seed)
        print(f"data_rate: {data_rate/1e6:.1f} Mbps")
        _time = time.time()
        for i, strategy_name in enumerate(sim_strategy_dict.keys()):
            print("Strategy: ", strategy_name)
            args.data_rate = data_rate
            if sim_strategy_dict[strategy_name]["HO"] == HO_LowerBound_SINR:
                (
                    energy_record,
                    violation_prob_record,
                ) = run_sim_withUMa_analyzed_lowerbound(
                    args, BS_loc_list, timeline_dir, 
                    pospred_model, 
                    beampred_model=sim_strategy_dict[strategy_name]["beampred_model"],
                    gainpred_model=sim_strategy_dict[strategy_name]["gainpred_model"],
                    inferpred_model=sim_strategy_dict[strategy_name]["inferpred_model"],
                    RA_func=sim_strategy_dict[strategy_name]["RA"], 
                    HO_func=sim_strategy_dict[strategy_name]["HO"],
                    prt=False,
                    No_BF=sim_strategy_dict[strategy_name]["NoBF"],
                    K_BF=sim_strategy_dict[strategy_name]["K_BF"],
                )
                avg_system_power = energy_record.mean() / (
                    args.slots_per_frame * args.slot_len
                )
                vio_prob = violation_prob_record[2:].mean() * 100
                sim_result_dict[strategy_name]["avg_system_power_list"].append(avg_system_power)
                sim_result_dict[strategy_name]["vio_prob_list"].append(vio_prob)
            else:
                (
                    energy_record,
                    HO_time_record,
                    HO_cmd_record,
                    violation_prob_record,
                    avg_queuelen_record,
                    pilot_record,
                    RB_allocated_record,
                    queuelen_4eachVeh_record,
                ) = run_sim_withUMa(
                    args, BS_loc_list, timeline_dir, 
                    pospred_model, 
                    beampred_model=sim_strategy_dict[strategy_name]["beampred_model"],
                    gainpred_model=sim_strategy_dict[strategy_name]["gainpred_model"],
                    inferpred_model=sim_strategy_dict[strategy_name]["inferpred_model"],
                    RA_func=sim_strategy_dict[strategy_name]["RA"], 
                    HO_func=sim_strategy_dict[strategy_name]["HO"],
                    BF_func=sim_strategy_dict[strategy_name]["BF"],
                    prt=False,
                    save_pilot=sim_strategy_dict[strategy_name]["save_pilot"],
                    No_BF=sim_strategy_dict[strategy_name]["NoBF"],
                    K_BF=sim_strategy_dict[strategy_name]["K_BF"],
                )
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
                sim_result_dict[strategy_name]["queuelen_4eachVeh_record_list"].append(queuelen_4eachVeh_record)
        
        # import ipdb;ipdb.set_trace()
           
        print("Elapsed time: ", time.time() - _time)

        plt.figure()
        for i, strategy_name in enumerate(sim_strategy_dict.keys()):
            plt.plot(
                data_rate_list[: data_rate_idx + 1]/1e6,
                sim_result_dict[strategy_name]["avg_system_power_list"][
                    : data_rate_idx + 1
                ],
                linestyle=plt_linestyle_list[0],
                color=plt_color_list[i],
                marker=plt_marker_list[i],
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (Mbps)")
        # plt.xscale("log")
        plt.ylabel("Average system power (W)")
        plt.savefig(os.path.join(save_path, "Average system power.png"))
        plt.savefig(os.path.join(save_path, "Average system power.pdf"))
        plt.close()

        plt.figure()
        avg_car_num = sum([len(timeline_dir[frame]) for frame in timeline_dir.keys()]) / len(timeline_dir.keys())
        for i, strategy_name in enumerate(sim_strategy_dict.keys()):
            if sim_strategy_dict[strategy_name]["HO"] == HO_LowerBound_SINR:
                continue
            plt.plot(
                data_rate_list[: data_rate_idx + 1]/1e6,
                np.array(sim_result_dict[strategy_name]["HOps_list"][: data_rate_idx + 1]) / avg_car_num,
                linestyle=plt_linestyle_list[0],
                color=plt_color_list[i],
                marker=plt_marker_list[i],
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (Mbps)")
        # plt.xscale("log")
        plt.ylabel("Average HO frequency per vehicle (1/s)")
        plt.savefig(os.path.join(save_path, "Average HO frequency.png"))
        plt.savefig(os.path.join(save_path, "Average HO frequency.pdf"))
        plt.close()

        plt.figure()
        for i, strategy_name in enumerate(sim_strategy_dict.keys()):
            plt.plot(
                data_rate_list[: data_rate_idx + 1]/1e6,
                sim_result_dict[strategy_name]["vio_prob_list"][: data_rate_idx + 1],
                linestyle=plt_linestyle_list[0],
                color=plt_color_list[i],
                marker=plt_marker_list[i],
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (Mbps)")
        # plt.xscale("log")
        plt.ylim(0, 100)
        plt.ylabel("Violation probability (%)")
        plt.savefig(os.path.join(save_path, "Violation probability.png"))
        plt.savefig(os.path.join(save_path, "Violation probability.pdf"))
        plt.close()

        plt.figure()
        for i, strategy_name in enumerate(sim_strategy_dict.keys()):
            if sim_strategy_dict[strategy_name]["HO"] == HO_LowerBound_SINR:
                continue
            plt.plot(
                data_rate_list[: data_rate_idx + 1]/1e6,
                sim_result_dict[strategy_name]["avg_latency_list"][: data_rate_idx + 1],
                linestyle=plt_linestyle_list[0],
                color=plt_color_list[i],
                marker=plt_marker_list[i],
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (Mbps)")
        # plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Average latency (ms)")
        plt.savefig(os.path.join(save_path, "Average latency.png"))
        plt.savefig(os.path.join(save_path, "Average latency.pdf"))
        plt.close()
        
        # plot the avg latency of the 90th percentile users using sim_result_dict[strategy_name]["queuelen_4eachVeh_record_list"]
        plt.figure()
        for i, strategy_name in enumerate(sim_strategy_dict.keys()):
            if sim_strategy_dict[strategy_name]["HO"] == HO_LowerBound_SINR:
                continue
            avg_latency_90th_list = []
            for queuelen_4eachVeh_record in sim_result_dict[strategy_name]["queuelen_4eachVeh_record_list"][: data_rate_idx + 1]:
                all_veh_queuelens = []
                for frame_record in queuelen_4eachVeh_record.values():
                    all_veh_queuelens.extend(frame_record.values())
                all_veh_queuelens = np.array(all_veh_queuelens)
                queuelen_90th = np.percentile(all_veh_queuelens, 90)
                latency_90th = queuelen_90th / args.data_rate * 1000
                avg_latency_90th_list.append(latency_90th)
            plt.plot(
                data_rate_list[: data_rate_idx + 1]/1e6,
                avg_latency_90th_list,
                linestyle=plt_linestyle_list[0],
                color=plt_color_list[i],
                marker=plt_marker_list[i],
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (Mbps)")
        # plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Average latency of 90th percentile users (ms)")
        plt.savefig(os.path.join(save_path, "Average latency of 90th percentile users.png"))
        plt.savefig(os.path.join(save_path, "Average latency of 90th percentile users.pdf"))
        plt.close() 
        
        
        # plot the QoS violation probability of the 90th percentile users using sim_result_dict[strategy_name]["queuelen_4eachVeh_record_list"]
        plt.figure()
        for i, strategy_name in enumerate(sim_strategy_dict.keys()):
            if sim_strategy_dict[strategy_name]["HO"] == HO_LowerBound_SINR:
                continue
            vio_prob_90th_list = []
            for data_rate, queuelen_4eachVeh_record in zip(data_rate_list[: data_rate_idx + 1],sim_result_dict[strategy_name]["queuelen_4eachVeh_record_list"][: data_rate_idx + 1]):
                all_veh_queuelens = []
                for frame_record in queuelen_4eachVeh_record.values():
                    all_veh_queuelens.extend(frame_record.values())
                all_veh_queuelens = np.array(all_veh_queuelens)
                latency_90th = np.percentile(all_veh_queuelens, 90) / args.data_rate
                
                latency_threshold = args.lat_slot_ub * args.slot_len
                
                vio_prob_90th = (latency_90th > latency_threshold) * 100
                vio_prob_90th_list.append(vio_prob_90th)
            plt.plot(
                data_rate_list[: data_rate_idx + 1]/1e6,
                vio_prob_90th_list,
                linestyle=plt_linestyle_list[0],
                color=plt_color_list[i],
                marker=plt_marker_list[i],
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (Mbps)")
        # plt.xscale("log")
        plt.ylim(0, 100)
        plt.ylabel("QoS violation probability of 90th percentile users (%)")
        plt.savefig(os.path.join(save_path, "QoS violation probability of 90th percentile users.png"))
        plt.savefig(os.path.join(save_path, "QoS violation probability of 90th percentile users.pdf"))
        plt.close() 
                
        
        plt.figure(figsize=(6, 4*len(BS_loc_list)))
        plt.xlabel("data rate (Mbps)")
        plt.ylabel("Average car number under each BS")
        for BS_id in range(len(BS_loc_list)+1):
            plt.subplot(len(BS_loc_list)+1, 1, BS_id + 1)
            for i, strategy_name in enumerate(sim_strategy_dict.keys()):
                if sim_strategy_dict[strategy_name]["HO"] == HO_LowerBound_SINR:
                    continue
                avg_carnum_under_BS_list = np.array(sim_result_dict[strategy_name]["carnum_under_BS_list"][: data_rate_idx + 1]).mean(axis=-2)
                plt.plot(
                    data_rate_list[: data_rate_idx + 1]/1e6,
                    avg_carnum_under_BS_list[:, BS_id],
                    linestyle=plt_linestyle_list[0],
                    color=plt_color_list[i],
                    marker=plt_marker_list[i],
                    label=f"{strategy_name} BS{BS_id}",
                )
            plt.legend()
        plt.savefig(os.path.join(save_path, "Average car number under each BS.png"))
        plt.savefig(os.path.join(save_path, "Average car number under each BS.pdf"))
        plt.close()
        

        # 保存仿真实验设置
        sim_result_dict["args"] = args
        sim_result_dict["data_rate_list"] = data_rate_list
        # 保存仿真实验结果指标
        np.save(os.path.join(save_path, "sim_result_dict.npy"), sim_result_dict)
        # 保存仿真实验设置
