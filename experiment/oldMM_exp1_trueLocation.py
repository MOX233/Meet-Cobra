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


sys.path.append(os.getcwd())
from utils.sim_utils import run_sim
from utils.options import args_parser
from utils.alg_utils import (
    RA_Lyapunov,
    RA_heur_b,
    RA_heur_q,
    RA_heur_qb,
    RA_unlimitRB,
    HO_nearest,
    HO_EE,
    HO_SE,
    HO_RBE,
    RA_heur_fqb,
    RA_heur_fqb_smartRound,
    HO_SE_loadbalance,
    HO_EE_loadbalance,
    HO_RBE_loadbalance,
    HO_EE_offload,
    HO_EE_GAP_APX,
    HO_EE_GAP_APX_with_offload,
    HO_EE_GAP_APX_with_offload_conservative,
)
from utils.sumo_utils import (
    read_trajectoryInfo_carindex,
    read_trajectoryInfo_carindex_matrix,
    read_trajectoryInfo_timeindex,
)

if __name__ == "__main__":
    # Urban Macro LoS: PL = 28 + 22*log10(d)+20*log10(f)
    # Urban Micro LoS: PL = 32.4 + 21*log10(d)+20*log10(f)
    save_path = (
        "./experiment/exp_results/trueLocation_"
        + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    )
    os.makedirs(save_path, exist_ok=True)
    # data_rate_list = np.logspace(7, 8, 10)
    data_rate_list = np.linspace(1e6, 13e6, 25)
    args = args_parser()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.slots_per_frame = 100
    args.beta_macro = 2.2
    args.beta_micro = 3
    args.bias_macro = -(28 + 20 * np.log10(5.9))
    args.bias_micro = -(32.4 + 20 * np.log10(24))
    shd_sigma = 4
    args.shd_sigma_macro = shd_sigma
    args.shd_sigma_micro = shd_sigma
    args.num_RB_macro = 100
    args.num_RB_micro = 100
    args.RB_intervel_macro = 0.18 * 1e6
    args.RB_intervel_micro = 1.8 * 1e6
    args.p_macro = 1
    args.p_micro = 0.1
    args.lat_slot_ub = 20
    args.trajectoryInfo_path = (
        f"./sumo_result/trajectory_Lbd{0.10:.2f}.csv"
    )
    timeline_dir = read_trajectoryInfo_timeindex(
        args,
        start_time=500,
        end_time=530,
        display_intervel=0.05,
    )
    args.eta = 1e6
    args.frames_per_sample = 10
    args.frame_per_sample = 10
    model = None  # 先不管NN预测，直接用真实坐标

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

    # 给出所需要仿真的方案名和PHO,RA策略
    sim_strategy_dict = collections.OrderedDict()
    """sim_strategy_dict["GAP"] = {"RA": RA_heur_q, "HO": HO_EE_GAP_APX}
    sim_strategy_dict["GAP_offload"] = {
        "RA": RA_heur_q,
        "HO": HO_EE_GAP_APX_with_offload,
    }"""
    # sim_strategy_dict["GAP(fqb_CR)_offload"] = {"RA": RA_heur_fqb_continuerelax,"HO": HO_EE_GAP_APX_with_offload}
    sim_strategy_dict["GAP(fqb)_offload_conservative"] = {
        "RA": RA_heur_fqb_smartRound,
        "HO": HO_EE_GAP_APX_with_offload_conservative,
    }
    # sim_strategy_dict["GAP(fqb)"] = {"RA": RA_heur_fqb, "HO": HO_EE_GAP_APX}
    # sim_strategy_dict["GAP(fqb_CR)"] = {"RA": RA_heur_fqb_continuerelax, "HO": HO_EE_GAP_APX,}
    # sim_strategy_dict["GAP_offload(fqb)"] = {"RA": RA_heur_fqb,"HO": HO_EE_GAP_APX_with_MOX1,}
    sim_strategy_dict["EE"] = {"RA": RA_heur_fqb_smartRound, "HO": HO_EE}
    # sim_strategy_dict["EE_pro"] = {"RA": RA_heur_q, "HO": HO_EE_pro}
    # sim_strategy_dict["SE"] = {"RA": RA_heur_q, "HO": HO_SE}
    # sim_strategy_dict["Nearest"] = {"RA": RA_heur_q, "HO": HO_nearest}
    sim_strategy_dict["RBE"] = {"RA": RA_heur_fqb_smartRound, "HO": HO_RBE}
    sim_strategy_dict["InfRB"] = {"RA": RA_unlimitRB, "HO": HO_EE}
    # sim_strategy_dict["SE_pro"] = {"RA": RA_heur_q, "HO": HO_SE_pro}
    # sim_strategy_dict["EE_offload"] = {"RA": RA_heur_q, "HO": HO_EE_MOX1}
    # sim_strategy_dict["GAPMOX"] = {"RA": RA_heur_q, "HO": HO_EE_GAP_APX_with_MOX1}

    sim_result_dict = collections.OrderedDict()
    for strategy_name in sim_strategy_dict.keys():
        sim_result_dict[strategy_name] = {
            "avg_system_power_list": [],
            "HOps_list": [],
            "vio_prob_list": [],
            "avg_queue_len_list": [],
            "avg_latency_list": [],
        }

    # 进行仿真实验
    for data_rate_idx, data_rate in enumerate(data_rate_list):
        print(f"data_rate: {data_rate} bps")
        for strategy_name in sim_strategy_dict.keys():
            print("Strategy: ", strategy_name)
            args.data_rate = data_rate
            (
                energy_record,
                HO_time_record,
                HO_cmd_record,
                violation_prob_record,
                avg_queuelen_record,
            ) = run_sim(
                args,
                BS_loc_list,
                timeline_dir,
                model,
                RA_func=sim_strategy_dict[strategy_name]["RA"],
                HO_func=sim_strategy_dict[strategy_name]["HO"],
                prt=False,
            )
            avg_system_power = energy_record.mean() / (
                args.slots_per_frame * args.slot_len
            )
            HOps = HO_time_record[2:].mean() / (args.slots_per_frame * args.slot_len)
            vio_prob = violation_prob_record[2:].mean() * 100
            avg_queue_len = avg_queuelen_record[2:].mean()
            avg_latency = avg_queuelen_record[2:].mean() / args.data_rate * 1000
            sim_result_dict[strategy_name]["avg_system_power_list"].append(
                avg_system_power
            )
            sim_result_dict[strategy_name]["HOps_list"].append(HOps)
            sim_result_dict[strategy_name]["vio_prob_list"].append(vio_prob)
            sim_result_dict[strategy_name]["avg_queue_len_list"].append(avg_queue_len)
            sim_result_dict[strategy_name]["avg_latency_list"].append(avg_latency)

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                data_rate_list[: data_rate_idx + 1],
                sim_result_dict[strategy_name]["avg_system_power_list"][
                    : data_rate_idx + 1
                ],
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (bps)")
        # plt.xscale("log")
        plt.ylabel("Average system power (W)")
        plt.savefig(os.path.join(save_path, "exp1 Average system power.png"))
        plt.savefig(os.path.join(save_path, "exp1 Average system power.pdf"))
        plt.close()

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                data_rate_list[: data_rate_idx + 1],
                sim_result_dict[strategy_name]["HOps_list"][: data_rate_idx + 1],
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (bps)")
        # plt.xscale("log")
        plt.ylabel("Average HO frequency (1/s)")
        plt.savefig(os.path.join(save_path, "exp1 Average HO frequency.png"))
        plt.savefig(os.path.join(save_path, "exp1 Average HO frequency.pdf"))
        plt.close()

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                data_rate_list[: data_rate_idx + 1],
                sim_result_dict[strategy_name]["vio_prob_list"][: data_rate_idx + 1],
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (bps)")
        # plt.xscale("log")
        plt.ylim(0, 100)
        plt.ylabel("Violation probability (%)")
        plt.savefig(os.path.join(save_path, "exp1 Violation probability.png"))
        plt.savefig(os.path.join(save_path, "exp1 Violation probability.pdf"))
        plt.close()

        plt.figure()
        for strategy_name in sim_strategy_dict.keys():
            plt.plot(
                data_rate_list[: data_rate_idx + 1],
                sim_result_dict[strategy_name]["avg_latency_list"][: data_rate_idx + 1],
                "*-",
                label=strategy_name,
            )
        plt.legend()
        plt.xlabel("data rate (bps)")
        # plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Average latency (ms)")
        plt.savefig(os.path.join(save_path, "exp1 Average latency.png"))
        plt.savefig(os.path.join(save_path, "exp1 Average latency.pdf"))
        plt.close()

        # 保存仿真实验设置
        sim_result_dict["args"] = args
        sim_result_dict["data_rate_list"] = data_rate_list
        # 保存仿真实验结果指标
        np.save(os.path.join(save_path, "sim_result_dict.npy"), sim_result_dict)
        # 保存仿真实验设置
