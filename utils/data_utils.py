import os
import sys

sys.path.append(os.getcwd())
import numpy as np
from utils.sumo_utils import (
    read_trajectoryInfo_carindex,
    read_trajectoryInfo_carindex_matrix,
    read_trajectoryInfo_timeindex,
)
from utils.channel_utils import (
    generate_random_channel,
    generate_random_channel_onlyiidshd,
    cal_distance_BS_UE,
)


def is_car_under_BS(BS_pos, car_pos, BS_radius=300):
    return np.sqrt(((BS_pos - car_pos) ** 2).sum()) < BS_radius


def cal_car_num_under_BS(BS_pos, cars_pos_list, BS_radius=300):
    num = 0
    for car_pos in cars_pos_list:
        num += is_car_under_BS(BS_pos, car_pos, BS_radius)
    return num


def car_num_under_BS_timeline(BS_pos, timeline_dir, BS_radius=300):
    car_num_dir = {}
    for timeslot, time_record_dir in timeline_dir.items():
        cars_pos_list = [i["pos"] for i in time_record_dir.values()]
        car_num_dir[timeslot] = cal_car_num_under_BS(BS_pos, cars_pos_list, BS_radius)
    return car_num_dir


def first_order_iir_filter(x, alpha):
    """
    一阶 IIR 滤波器
    :param x: 输入信号（数组）
    :param alpha: 滤波器系数 (0 <= alpha <= 1)
    :return: 滤波后的信号（数组）
    """
    y = np.zeros_like(x)
    y[0] = x[0]  # 初始条件

    for n in range(1, len(x)):
        y[n] = alpha * x[n] + (1 - alpha) * y[n - 1]

    return y


def second_order_iir_filter(x, b, a):
    """
    二阶 IIR 滤波器
    :param x: 输入信号（数组）
    :param b: 前向系数 [b0, b1, b2]
    :param a: 反馈系数 [1, a1, a2]（a[0] = 1）
    :return: 滤波后的信号（数组）
    """
    y = np.zeros_like(x)
    for n in range(2, len(x)):
        y[n] = (
            b[0] * x[n]
            + b[1] * x[n - 1]
            + b[2] * x[n - 2]
            - a[1] * y[n - 1]
            - a[2] * y[n - 2]
        )
    return y


def preprocess_data_Zscore(dataX, dataY, muX=-220, muY=0, sigmaX=0.0784, sigmaY=300):
    dataX_prcsd = (dataX - muX.reshape(1, 1, -1)) / sigmaX.reshape(1, 1, -1)
    dataY_prcsd = (dataY - muY) / sigmaY
    return dataX_prcsd, dataY_prcsd


def _prepare_dataset(
    args,
    BS_loc_list,
    frames_per_sample=10,
    slots_per_frame=100,
    shd_sigma=0,
    shd_crr_coeff=0,
    fd_sigma=1,
):
    carline_dir_matrix = read_trajectoryInfo_carindex_matrix(args)
    num_BS = len(BS_loc_list)
    num_sample = 0
    for k, v in carline_dir_matrix.items():
        num_sample += int((len(v) - 1) / frames_per_sample)
    traj_matrix = np.zeros((num_sample, frames_per_sample, 2))  # 记录各样本的车辆轨迹
    dataX = np.zeros((num_sample, frames_per_sample, num_BS)).astype(
        np.float32
    )  # 记录各样本的UE-BS信道增益
    dataY = np.zeros((num_sample, 2)).astype(np.float32)  # 记录各样本的最终车辆位置
    cnt = 0
    for k, v in carline_dir_matrix.items():
        samples_per_veh = int((len(v) - 1) / frames_per_sample)
        for i in range(samples_per_veh):
            traj_matrix[cnt, :, :] = v[
                i * frames_per_sample : (i + 1) * frames_per_sample, 1:3
            ]
            cnt += 1
    for BS_idx, BS_loc in enumerate(BS_loc_list):
        distance_BS_UE = cal_distance_BS_UE(
            traj_matrix, BS_loc
        )  # 计算各样本的UE-BS距离轨迹
        # dataX[:,:,BS_idx],_ = generate_random_channel(distance_BS_UE, shd_sigma=shd_sigma, shd_crr_coeff=shd_crr_coeff, fd_sigma=fd_sigma,num_frame=frames_per_sample, slots_per_frame= slots_per_frame)
        dataX[:, :, BS_idx] = generate_random_channel(
            distance_BS_UE,
            bias=args.bias,
            beta=args.beta,
            shd_sigma=shd_sigma,
            shd_crr_coeff=shd_crr_coeff,
            fd_sigma=fd_sigma,
            num_frame=frames_per_sample,
            slots_per_frame=slots_per_frame,
            eps=1e-12,
        )
    dataY[:, :] = traj_matrix[:, -1, :]
    muX, sigmaX = dataX.mean(), dataX.std()
    muY, sigmaY = dataY.mean(), dataY.std()
    dataX_prcsd, dataY_prcsd = preprocess_data_Zscore(
        dataX, dataY, muX, muY, sigmaX, sigmaY
    )
    return dataX_prcsd, dataY_prcsd, dataX, dataY, muX, sigmaX, muY, sigmaY


def prepare_dataset_onlyiidshd(
    args,
    BS_loc_list,
    frames_per_sample=10,
    shd_sigma_macro=0,
    shd_sigma_micro=0,
    preprocess_params=None,
):
    carline_dir_matrix = read_trajectoryInfo_carindex_matrix(args)
    num_BS = len(BS_loc_list)
    num_sample = 0
    for k, v in carline_dir_matrix.items():
        num_sample += int((len(v) - 1) / frames_per_sample)
    traj_matrix = np.zeros((num_sample, frames_per_sample, 2))  # 记录各样本的车辆轨迹
    dataX = np.zeros((num_sample, frames_per_sample, num_BS)).astype(
        np.float32
    )  # 记录各样本的UE-BS信道增益
    dataY = np.zeros((num_sample, 2)).astype(np.float32)  # 记录各样本的最终车辆位置
    cnt = 0
    for k, v in carline_dir_matrix.items():
        samples_per_veh = int((len(v) - 1) / frames_per_sample)
        for i in range(samples_per_veh):
            traj_matrix[cnt, :, :] = v[
                i * frames_per_sample : (i + 1) * frames_per_sample, 1:3
            ]
            dataY[cnt, :] = v[(i + 1) * frames_per_sample, 1:3]
            cnt += 1
    for BS_idx, BS_loc in enumerate(BS_loc_list):
        distance_BS_UE = cal_distance_BS_UE(
            traj_matrix, BS_loc
        )  # 计算各样本的UE-BS距离轨迹
        if BS_idx == 0:
            dataX[:, :, BS_idx] = generate_random_channel_onlyiidshd(
                distance_BS_UE,
                bias=args.bias_macro,
                beta=args.beta_macro,
                shd_sigma=shd_sigma_macro,
                num_frame=frames_per_sample,
            )
        else:
            dataX[:, :, BS_idx] = generate_random_channel_onlyiidshd(
                distance_BS_UE,
                bias=args.bias_micro,
                beta=args.beta_micro,
                shd_sigma=shd_sigma_micro,
                num_frame=frames_per_sample,
            )
    if preprocess_params == None:
        muX = dataX.mean(axis=0, keepdims=False).mean(
            axis=0, keepdims=False
        )  # (num_BS)
        sigmaX = dataX.reshape(-1, len(BS_loc_list)).std(
            axis=0, keepdims=False
        )  # (num_BS)
        muY, sigmaY = dataY.mean(), dataY.std()
    else:
        muX = preprocess_params["muX"]
        sigmaX = preprocess_params["sigmaX"]
        muY = preprocess_params["muY"]
        sigmaY = preprocess_params["sigmaY"]
    dataX_prcsd, dataY_prcsd = preprocess_data_Zscore(
        dataX, dataY, muX, muY, sigmaX, sigmaY
    )
    return dataX_prcsd, dataY_prcsd, dataX, dataY, muX, sigmaX, muY, sigmaY


def prepare_dataset_multipoint_onlyiidshd(
    args,
    BS_loc_list,
    ioframes_per_sample=10,
    sample_interval=1,
    shd_sigma_macro=0,
    shd_sigma_micro=0,
    preprocess_params=None,
):
    carline_dir_matrix = read_trajectoryInfo_carindex_matrix(args)
    num_BS = len(BS_loc_list)
    num_sample = 0
    # 每个样本包含ioframes_per_sample长的输入和ioframes_per_sample长的输出，因此每个样本包含2*ioframes_per_sample个frame
    for k, v in carline_dir_matrix.items():
        num_sample += max(0, int((len(v) - 2 * ioframes_per_sample) / sample_interval))
    traj_matrix = np.zeros(
        (num_sample, ioframes_per_sample, 2)
    )  # 记录各样本的历史车辆轨迹
    dataX = np.zeros((num_sample, ioframes_per_sample, num_BS)).astype(
        np.float32
    )  # 记录各样本的UE-BS信道增益
    dataY = np.zeros((num_sample, ioframes_per_sample, 2)).astype(
        np.float32
    )  # 记录各样本的未来车辆轨迹
    cnt = 0
    for k, v in carline_dir_matrix.items():
        samples_per_veh = max(
            0, int((len(v) - 2 * ioframes_per_sample) / sample_interval)
        )
        for i in range(samples_per_veh):
            traj_matrix[cnt, :, :] = v[
                i * sample_interval : i * sample_interval + ioframes_per_sample, 1:3
            ]
            dataY[cnt, :, :] = v[
                i * sample_interval
                + ioframes_per_sample : i * sample_interval
                + 2 * ioframes_per_sample,
                1:3,
            ]
            cnt += 1
    for BS_idx, BS_loc in enumerate(BS_loc_list):
        distance_BS_UE = cal_distance_BS_UE(
            traj_matrix, BS_loc
        )  # 计算各样本的UE-BS距离轨迹
        if BS_idx == 0:
            dataX[:, :, BS_idx] = generate_random_channel_onlyiidshd(
                distance_BS_UE,
                bias=args.bias_macro,
                beta=args.beta_macro,
                shd_sigma=shd_sigma_macro,
                num_frame=ioframes_per_sample,
            )
        else:
            dataX[:, :, BS_idx] = generate_random_channel_onlyiidshd(
                distance_BS_UE,
                bias=args.bias_micro,
                beta=args.beta_micro,
                shd_sigma=shd_sigma_micro,
                num_frame=ioframes_per_sample,
            )
    if preprocess_params == None:
        muX = dataX.mean(axis=0, keepdims=False).mean(
            axis=0, keepdims=False
        )  # (num_BS)
        sigmaX = dataX.reshape(-1, len(BS_loc_list)).std(
            axis=0, keepdims=False
        )  # (num_BS)
        muY, sigmaY = dataY.mean(), dataY.std()
    else:
        muX = preprocess_params["muX"]
        sigmaX = preprocess_params["sigmaX"]
        muY = preprocess_params["muY"]
        sigmaY = preprocess_params["sigmaY"]
    dataX_prcsd, dataY_prcsd = preprocess_data_Zscore(
        dataX, dataY, muX, muY, sigmaX, sigmaY
    )
    return dataX_prcsd, dataY_prcsd, dataX, dataY, muX, sigmaX, muY, sigmaY


def prepare_dataset_onlyiidshd_v2(
    args,
    BS_loc_list,
    ifps=10,
    ofps=10,
    sample_interval=1,
    shd_sigma_macro=0,
    shd_sigma_micro=0,
    preprocess_params=None,
):
    carline_dir_matrix = read_trajectoryInfo_carindex_matrix(args)
    num_BS = len(BS_loc_list)
    num_sample = 0
    # 每个样本包含ioframes_per_sample长的输入和ioframes_per_sample长的输出，因此每个样本包含2*ioframes_per_sample个frame
    for k, v in carline_dir_matrix.items():
        num_sample += max(0, int((len(v) - (ifps + ofps)) / sample_interval))
    traj_matrix = np.zeros((num_sample, ifps, 2))  # 记录各样本的历史车辆轨迹
    dataX = np.zeros((num_sample, ifps, num_BS)).astype(
        np.float32
    )  # 记录各样本的UE-BS信道增益
    dataY = np.zeros((num_sample, ofps, 2)).astype(
        np.float32
    )  # 记录各样本的未来车辆轨迹
    cnt = 0
    for k, v in carline_dir_matrix.items():
        samples_per_veh = max(0, int((len(v) - (ifps + ofps)) / sample_interval))
        for i in range(samples_per_veh):
            traj_matrix[cnt, :, :] = v[
                i * sample_interval : i * sample_interval + ifps, 1:3
            ]
            dataY[cnt, :, :] = v[
                i * sample_interval + ifps : i * sample_interval + ifps + ofps,
                1:3,
            ]
            cnt += 1
    distance_BS_UE_list = []
    for BS_idx, BS_loc in enumerate(BS_loc_list):
        distance_BS_UE = cal_distance_BS_UE(
            traj_matrix, BS_loc
        )  # 计算各样本的UE-BS距离轨迹
        if BS_idx == 0:
            dataX[:, :, BS_idx] = generate_random_channel_onlyiidshd(
                distance_BS_UE,
                bias=args.bias_macro,
                beta=args.beta_macro,
                shd_sigma=shd_sigma_macro,
                num_frame=ifps,
            )
        else:
            dataX[:, :, BS_idx] = generate_random_channel_onlyiidshd(
                distance_BS_UE,
                bias=args.bias_micro,
                beta=args.beta_micro,
                shd_sigma=shd_sigma_micro,
                num_frame=ifps,
            )
        distance_BS_UE_list.append(distance_BS_UE)
    distance_BS_UE_matrix = np.stack(distance_BS_UE_list, axis=-1)
    """import ipdb

    ipdb.set_trace()"""
    # distance_BS_UE_matrix.min(axis=-1).mean() -> 93
    if preprocess_params == None:
        muX = dataX.mean(axis=0, keepdims=False).mean(
            axis=0, keepdims=False
        )  # (num_BS)
        sigmaX = dataX.reshape(-1, len(BS_loc_list)).std(
            axis=0, keepdims=False
        )  # (num_BS)
        muY, sigmaY = dataY.mean(), dataY.std()
    else:
        muX = preprocess_params["muX"]
        sigmaX = preprocess_params["sigmaX"]
        muY = preprocess_params["muY"]
        sigmaY = preprocess_params["sigmaY"]
    dataX_prcsd, dataY_prcsd = preprocess_data_Zscore(
        dataX, dataY, muX, muY, sigmaX, sigmaY
    )
    return dataX_prcsd, dataY_prcsd, dataX, dataY, muX, sigmaX, muY, sigmaY
