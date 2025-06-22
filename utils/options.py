#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import argparse
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        type=str,
        default=1,
        help="Useless. Just for jupyter!",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed which make tests reproducible (default: 1)",
    )
    # data traffic arguments
    parser.add_argument(
        "--data_rate",
        type=float,
        default=1e6,
        help="UE data traffic arrival rate (bps), default (1Mbps)",
    )
    parser.add_argument(
        "--lat_slot_ub",
        type=float,
        default=10,
        help="The upper bound of the delay slot number, default (10)",
    )

    # MIMO arguments
    parser.add_argument(
        "--M_t",
        type=int,
        default=64,
        help="number of transmit antennas (default: 64)",
    )
    parser.add_argument(
        "--M_r",
        type=int,
        default=8,
        help="number of receive antennas (default: 8)",
    )
    parser.add_argument(
        "--N_bs",
        type=int,
        default=4,
        help="number of micro BSs (default: 4)",
    )
    
    # channel arguments
    parser.add_argument(
        "--K_rician",
        type=float,
        default=1e9,
        help="Rician K-factor",
    )
    parser.add_argument(
        "--beta_macro",
        type=float,
        default=2.2,
        help="path loss exponential for the macro cell (5.9GHz)",
    )
    parser.add_argument(
        "--beta_micro",
        type=float,
        default=3,
        help="path loss exponential for the micro cell (24GHz)",
    )
    parser.add_argument(
        "--bias_macro",
        type=float,
        default=-(28 + 20 * np.log10(5.9)),
        help="channel gain bias for the macro cell (5.9GHz)",
    )
    parser.add_argument(
        "--bias_micro",
        type=float,
        default=-(32.4 + 20 * np.log10(24)),
        help="channel gain bias for the micro cell (24GHz)",
    )
    parser.add_argument(
        "--shd_sigma_macro",
        type=float,
        default=0,
        help="shadowing sigma for the macro cell (5.9GHz)",
    )
    parser.add_argument(
        "--shd_sigma_micro",
        type=float,
        default=0,
        help="shadowing sigma for the micro cell (24GHz)",
    )
    parser.add_argument("--slot_len", type=float, default=1e-3, help="slot length (s)")
    parser.add_argument(
        "--RB_intervel_macro",
        type=float,
        default=0.36 * 1e6,
        help="the frequency interval of two successive RBs for a macro BS",
    )
    parser.add_argument(
        "--RB_intervel_micro",
        type=float,
        default=1.8 * 1e6,
        help="the frequency interval of two successive RBs for a micro BS",
    )
    parser.add_argument(
        "--num_RB_macro",
        type=int,
        default=100,
        help="the RB number of the macro cell",
    )
    parser.add_argument(
        "--num_RB_micro",
        type=int,
        default=100,
        help="the RB number of the micro cell",
    )
    parser.add_argument(
        "--p_macro",
        type=float,
        default=1,
        help="the transmission power of the macro cell per RB (W per RB)",
    )
    parser.add_argument(
        "--p_micro",
        type=float,
        default=0.2,
        help="the transmission power of the micro cell per RB (W per RB)",
    )
    parser.add_argument(
        "--N0",
        type=float,
        default=10 ** (-17.4 - 3),
        help="the noise power spectral density (-174 dBm/Hz -> 10**(-17.4-3) W/Hz)",
    )
    
    # other arguments
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")

    # SUMO arguments
    parser.add_argument(
        "--sumo_data_dir",
        type=str,
        default="./sumo_data",
        help="the directory where saves the necessary config files for SUMO running",
    )
    parser.add_argument(
        "--no_sumo_run",
        action="store_true",
        default=False,
        help="run sumo simulation to generate tripinfo.xml",
    )
    parser.add_argument(
        "--trajectoryInfo_path",
        type=str,
        default="./sumo_data/trajectory_Lbd0.10.csv",
        help="the file path where stores the trajectory infomation of cars",
    )
    parser.add_argument(
        "--step_length", type=float, default=0.1, help="sumo sampling interval"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=3600,
        help="number of time steps, which means how many seconds the car flow takes",
    )

    parser.add_argument(
        "--Lambda", type=float, default=0.2, help="arrival rate of car flow"
    )
    parser.add_argument("--accel", type=float, default=3, help="accelerate of car flow")
    parser.add_argument("--decel", type=float, default=6, help="decelerate of car flow")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection",
    )
    parser.add_argument("--carLength", type=float, default=5, help="length of cars")
    parser.add_argument(
        "--minGap",
        type=float,
        default=0.30,
        help="minimum interval between adjacent cars",
    )
    parser.add_argument("--maxSpeed", type=float, default=20, help="maxSpeed for cars")
    parser.add_argument("--speedFactoer_mean", type=float, default=1, help="")
    parser.add_argument("--speedFactoer_dev", type=float, default=0, help="")
    parser.add_argument("--speedFactoer_min", type=float, default=1, help="")
    parser.add_argument("--speedFactoer_max", type=float, default=1, help="")
    
    # sionna arguments
    parser.add_argument(
        "--sionna_start_time", type=float, default=100, help="Sionna simulation start time"
    )
    parser.add_argument(
        "--sionna_end_time", type=float, default=200, help="Sionna simulation end time"
    )
    parser.add_argument(
        "--sionna_result_tmp_dir",
        type=str,
        default="./sionna_result/_tmp",
        help="the directory where saves the tmp files for sionna running",
    )
    parser.add_argument(
        "--h_car",
        type=float,
        default=1.6,
        help="the height of the car",
    )
    parser.add_argument(
        "--h_tx",
        type=float,
        default=35,
        help="the height of the tx",
    )
    parser.add_argument(
        "--h_rx",
        type=float,
        default=3,
        help="the height of the rx",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=28e9,
        help="the frequency of the signal of the micro cell",
    )
    parser.add_argument(
        "--N_t_H",
        type=int,
        default=1,
        help="the row number of the tx antennas",
    )
    parser.add_argument(
        "--N_t_V",
        type=int,
        default=64,
        help="the column number of the tx antennas",
    )
    parser.add_argument(
        "--N_r_H",
        type=int,
        default=1,
        help="the row number of the rx antennas",
    )
    parser.add_argument(
        "--N_r_V",
        type=int,
        default=8,
        help="the column number of the rx antennas",
    )
    parser.add_argument(
        "--antenna_pattern",
        type=str,
        default="iso",
        help="the antenna pattern, which can be 'iso' or 'tr38901'",
    )
    
    args = parser.parse_args()
    return args
