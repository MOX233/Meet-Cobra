import os
import sys
import pickle
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
from utils.beam_utils import generate_dft_codebook

def generate_complex_gaussian_vector(shape, scale=1.0, mean=0.0):
    """
    生成服从复高斯分布的多维向量
    
    参数:
        shape (tuple) : 输出向量的形状（如 (n,) 或 (m,n)）
        scale (float) : 标准差缩放因子（默认1.0）
        mean (float)  : 分布的均值（默认0.0）
    
    返回:
        complex_array (ndarray) : 复高斯多维向量
    """
    # 生成独立的高斯分布的实部和虚部
    real_part = np.random.normal(loc=mean, scale=scale/np.sqrt(2), size=shape)
    imag_part = np.random.normal(loc=mean, scale=scale/np.sqrt(2), size=shape)
    
    # 组合为复数形式
    complex_array = real_part + 1j * imag_part
    return complex_array

def prepare_dataset(sionna_result_filepath, M_t, M_r, N_bs, datasize_upperbound = 1e15, P_t=1e-1, P_noise=1e-14, n_pilot=16, mode=0, pos_std=1, look_ahead_len=0):
    assert look_ahead_len >= 0
    DFT_matrix_tx = generate_dft_codebook(M_t)
    DFT_matrix_rx = generate_dft_codebook(M_r)
    with open(sionna_result_filepath, 'rb') as f:
        trajectoryInfo = pickle.load(f)
    data_list = []
    best_beam_pair_index_list = []
    veh_pos_list = []
    veh_h_list = []
    sample_interval = int(M_t/n_pilot) if n_pilot>0 else int(1e9)
    
    frame_list = list(trajectoryInfo.keys())
    num_frame = len(frame_list) - look_ahead_len
    # 遍历每一帧
    # trajectoryInfo.keys() 代表每一帧的索引
    # trajectoryInfo[frame].keys() 代表每一帧中所有车辆的索引
    # trajectoryInfo[frame][veh]['h'] 代表每一帧中每辆车的信道矩阵 shape  (8, 4, 64)
    # trajectoryInfo[frame][veh]['pos'] 代表每一帧中每辆车的位置 shape  (2,)
    
    for i in range(num_frame):
        frame, look_ahead_frame = frame_list[i], frame_list[i+look_ahead_len]
        print(f'prepare_dataset {i+1}/{num_frame} : frame {frame}') 
        for veh in trajectoryInfo[frame].keys():
            if look_ahead_len == 0:
                veh_h = trajectoryInfo[frame][veh]['h']
                veh_pos = trajectoryInfo[frame][veh]['pos']
                # characteristics data
                # trajectoryInfo[frame][veh]['h'].shape  (8, 4, 64)
                if mode == 0: # 所有BS同时发射pilot，一共有n_pilot个不同方向
                    #data = veh_h.sum(axis=-1).reshape(-1)
                    data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
                    n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                    data = (data + n).astype(np.complex64)
                elif mode == 1: # 各BS轮流发射pilot，每个BS各有n_pilot个不同方向，一共有n_bs*n_pilot个不同方向
                    data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].reshape(-1)
                    n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                    data = (data + n).astype(np.complex64)
                elif mode == 2: # mode0 的基础上加上了position信息
                    data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
                    n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                    data = (data + n).astype(np.complex64)
                    data = np.concatenate([data,veh_pos+np.random.normal(loc=0, scale=pos_std/np.sqrt(2), size=(2,))])
                    # import ipdb;ipdb.set_trace()
                else:
                    raise ValueError("mode should be 0, 1 or 2")
                # best_beam_pair_index for N_bs BSs
                best_beam_pair_index = np.abs(np.matmul(np.matmul(veh_h, DFT_matrix_tx).T.conjugate(),DFT_matrix_rx).transpose([1,0,2]).reshape(N_bs,-1)).argmax(axis=-1)
            else:
                if veh not in trajectoryInfo[look_ahead_frame].keys():
                    continue
                # 车辆在look_ahead_frame中也存在
                veh_h = np.stack([trajectoryInfo[frame_list[j]][veh]['h'] for j in range(i, i+look_ahead_len+1)], axis=0)
                veh_pos = np.stack([trajectoryInfo[frame_list[j]][veh]['pos'] for j in range(i, i+look_ahead_len+1)], axis=0)
                if mode == 0: # 所有BS同时发射pilot，一共有n_pilot个不同方向
                    #data = veh_h.sum(axis=-1).reshape(-1)
                    data = np.stack([np.sqrt(P_t)*np.matmul(veh_h[j], DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1) for j in range(look_ahead_len+1)], axis=0)
                    # data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
                    n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                    data = (data + n).astype(np.complex64)
                elif mode == 1: # 各BS轮流发射pilot，每个BS各有n_pilot个不同方向，一共有n_bs*n_pilot个不同方向
                    data = np.stack([np.sqrt(P_t)*np.matmul(veh_h[j], DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].reshape(-1) for j in range(look_ahead_len+1)], axis=0)
                    # data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].reshape(-1)
                    n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                    data = (data + n).astype(np.complex64)
                elif mode == 2: # mode0 的基础上加上了position信息
                    data = np.stack([np.sqrt(P_t)*np.matmul(veh_h[j], DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1) for j in range(look_ahead_len+1)], axis=0)
                    # data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
                    n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                    data = (data + n).astype(np.complex64)
                    data = np.concatenate([data,veh_pos+np.random.normal(loc=0, scale=pos_std/np.sqrt(2), size=(look_ahead_len+1,2))], axis=-1)
                    # data = np.concatenate([data,veh_pos+np.random.normal(loc=0, scale=pos_std/np.sqrt(2), size=(2,))])
                    # import ipdb;ipdb.set_trace()
                else:
                    raise ValueError("mode should be 0, 1 or 2")
                # best_beam_pair_index for N_bs BSs
                best_beam_pair_index = np.stack([np.abs(np.matmul(np.matmul(veh_h[j], DFT_matrix_tx).T.conjugate(),DFT_matrix_rx).transpose([1,0,2]).reshape(N_bs,-1)).argmax(axis=-1) for j in range(look_ahead_len+1)], axis=0)
                
            data_list.append(data)
            best_beam_pair_index_list.append(best_beam_pair_index)
            veh_h_list.append(veh_h)
            veh_pos_list.append(veh_pos)
                
            if len(veh_pos_list) >= datasize_upperbound:
                break
        if len(veh_pos_list) >= datasize_upperbound:
            break
    
    data_np = np.array(data_list)
    best_beam_pair_index_np = np.array(best_beam_pair_index_list)
    veh_h_np = np.array(veh_h_list)
    veh_pos_np = np.array(veh_pos_list)

    return data_np, best_beam_pair_index_np, veh_pos_np, veh_h_np

def get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise, look_ahead_len=0, datasize_upperbound = 1e9):
    # 加载数据集
    if look_ahead_len == 0:
        prepared_dataset_filename = f'{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}'
    else:
        prepared_dataset_filename = f'{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}_lookahead{look_ahead_len}'
    prepared_dataset_filepath = os.path.join('./prepared_dataset/',prepared_dataset_filename+'.pkl')
    if os.path.exists(prepared_dataset_filepath):
        f_read = open(prepared_dataset_filepath, 'rb')
        prepared_dataset = pickle.load(f_read)
        data_np, best_beam_pair_index_np, veh_pos_np, veh_h_np = \
            prepared_dataset['data_np'], prepared_dataset['best_beam_pair_index_np'], prepared_dataset['veh_pos_np'], prepared_dataset['veh_h_np']
        f_read.close()
    else:
        filepath = f'./sionna_result/trajectoryInfo_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
        data_np, best_beam_pair_index_np, veh_pos_np, veh_h_np = \
            prepare_dataset(filepath,M_t, M_r, N_bs,datasize_upperbound,P_t, P_noise, n_pilot, mode=preprocess_mode, look_ahead_len=look_ahead_len)
        prepared_dataset = {}
        prepared_dataset['data_np'] = data_np
        prepared_dataset['best_beam_pair_index_np'] = best_beam_pair_index_np
        prepared_dataset['veh_pos_np'] = veh_pos_np
        prepared_dataset['veh_h_np'] = veh_h_np
        f_save = open(prepared_dataset_filepath, 'wb')
        pickle.dump(prepared_dataset, f_save)
        f_save.close()
    return prepared_dataset_filename, data_np, veh_h_np, veh_pos_np, best_beam_pair_index_np


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