from __future__ import absolute_import
from __future__ import print_function

import os
import sys

sys.argv = [""]
sys.path.append(os.getcwd())
import numpy as np
from utils.options import args_parser
from utils.sumo_utils import sumo_run_with_trajectoryInfo

args = args_parser()
args.Lambda = (
    0.05  # "--Lambda", type=float, default=0.2, help="arrival rate of car flow"
)
args.num_steps = 1000

args.trajectoryInfo_path = f"./sumo_data/trajectory_Lbd{args.Lambda:.2f}.csv"  # the file path where stores the trajectory infomation of cars
sumo_run_with_trajectoryInfo(args)
