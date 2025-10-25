import os
import sys
import pickle
sys.path.append(os.getcwd())
import torch
import random
import shutil
import time
import subprocess
import collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utils.sumo_utils import sumo_run_with_trajectoryInfo
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


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, data_np, label_np, look_ahead_len, augment_dataset_ratio=2.0):
        """
        Args:
            data_np: numpy array [num_samples, look_ahead_len+1, ...]
            label_np: numpy array [num_samples, ...]
            look_ahead_len: int
            augment_dataset_ratio: float >= 1.0
        """
        self.data_np = data_np.astype(np.float32)   # 用 float32 节省内存
        self.label_np = label_np.astype(np.float32)
        self.look_ahead_len = look_ahead_len
        self.augment_dataset_ratio = augment_dataset_ratio
        self.num_origin = data_np.shape[0]
        self.num_total = int(self.num_origin * augment_dataset_ratio)

    def __len__(self):
        return self.num_total

    def __getitem__(self, idx):
        if idx < self.num_origin:  
            # 原始样本
            data = self.data_np[idx, :-1, ...]
            label = self.label_np[idx, -1, ...]
            length = self.look_ahead_len
        else:  
            # 增强样本（随机截断 + 补零）
            sample_idx = np.random.randint(0, self.num_origin)
            valid_len = np.random.randint(1, self.look_ahead_len + 1)
            clipped_data = self.data_np[sample_idx, -valid_len-1:-1, ...]
            padded_data = np.zeros_like(self.data_np[sample_idx, :-1, ...])
            padded_data[:valid_len] = clipped_data
            data = padded_data
            label = self.label_np[sample_idx, -1, ...]
            length = valid_len
        
        return torch.from_numpy(data), torch.from_numpy(label), length


def run_sionna_sim_subprocess(params, timeout):
    """
    执行子进程的线程函数
    :param params: 参数字典
    :param timeout: 子进程超时时间（秒）
    """
    # 构建命令行参数
    cmd = ["python", "generate_data_3Dbeam_subprocess.py"]
    for k, v in params.items():
        cmd.append(f"--{k}={v}")
    
    try:
        # 执行子进程并捕获输出
        result = subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 记录执行结果
        log = f"""
        Task {params['sionna_start_time']:.1f}-{params['sionna_end_time']:.1f} completed!
        """
        print(log)
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Task {params['sionna_start_time']:.1f}-{params['sionna_end_time']:.1f} timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        error_msg = f"""
        ERROR in Task {params['sionna_start_time']:.1f}-{params['sionna_end_time']:.1f}:
        Exit code: {e.returncode}
        Error output: {e.stderr[:5000]}...
        """
        print(error_msg)

def run_sionna_sim(
    args,
    start_time, 
    end_time, 
    subprocess_time=1, 
    max_workers=3, # 最大线程数
    timeout=3600, # 子进程超时时间（秒）
    gpu=7, # GPU编号
    Lambda=0.10, #车辆到达率
    freq=28e9, # 28e9 or 5.9e9
    antenna_pattern="iso", # "iso" or "tr38901"
    N_t_H=1,
    N_t_V=64,
    N_r_H=1,
    N_r_V=8,
    h_car=1.6,
    h_rx=3,
    h_tx=35,
    ):
    sumo_traj_path = f"./sumo_data/trajectory_Lbd{Lambda:.2f}.csv"
    if not os.path.exists(sumo_traj_path):
        print(f"SUMO trajectory file {sumo_traj_path} not found. Generating with SUMO...")
        sumo_run_with_trajectoryInfo(args)
    assert os.path.exists(sumo_traj_path)

    sionna_result_filepath = f"./sionna_result/trajectoryInfo_lbd{Lambda:.2f}_{start_time}_{end_time}_3Dbeam_tx({N_t_H},{N_t_V})_rx({N_r_H},{N_r_V})_freq{freq:.1e}.pkl"
    sionna_result_tmp_dir = f"./sionna_result/_tmp_({start_time:.1f},{end_time:.1f})" + time.strftime("_%Y-%m-%d_%H:%M:%S", time.gmtime(time.time() + 8 * 3600)) 
        
    
    # 设置子进程参数列表
    PARAMETERS = []
    for subprocess_start_time in np.arange(start_time, end_time, subprocess_time):
        subprocess_end_time = min(subprocess_start_time + subprocess_time, end_time)
        PARAMETERS.append({
            "sionna_start_time": subprocess_start_time, 
            "sionna_end_time": subprocess_end_time,
            "sionna_result_tmp_dir": sionna_result_tmp_dir,
            "gpu": gpu,
            "trajectoryInfo_path": sumo_traj_path,
            "freq": freq,
            "antenna_pattern": antenna_pattern,
            "N_t_H": N_t_H,
            "N_t_V": N_t_V,
            "N_r_H": N_r_H,
            "N_r_V": N_r_V,
            "h_car": h_car,
            "h_rx": h_rx,
            "h_tx": h_tx,
            })
    # 多线程执行子进程
    with ThreadPoolExecutor(max_workers) as executor:
        futures = []
        for params in PARAMETERS:
            future = executor.submit(run_sionna_sim_subprocess, params, timeout)
            futures.append(future)
        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 获取执行结果（自动处理异常）
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
    # 合并结果
    missing_files = []
    main_trajectoryInfo = collections.OrderedDict()
    for params in PARAMETERS:
        subprocess_start_time = params["sionna_start_time"]
        subprocess_end_time = params["sionna_end_time"]
        subprocess_result_file = os.path.join(sionna_result_tmp_dir, f"time({subprocess_start_time:.1f},{subprocess_end_time:.1f})_tx({N_t_H},{N_t_V})_rx({N_r_H},{N_r_V})_freq{freq:.1e}.pkl")
        if not os.path.exists(subprocess_result_file):
            missing_files.append(subprocess_result_file)
            print(f"Missing File {subprocess_result_file}.")
            continue
        with open(subprocess_result_file, "rb") as tf:
            sub_trajectoryInfo = pickle.load(tf)
            main_trajectoryInfo.update(sub_trajectoryInfo)
    # 保存结果
    with open(sionna_result_filepath, "wb") as tf:
        pickle.dump(main_trajectoryInfo,tf)
        print("successfully save trajectoryInfo to file ",sionna_result_filepath)
    # 删除临时文件
    if os.path.exists(sionna_result_tmp_dir):
        shutil.rmtree(sionna_result_tmp_dir)
        print(f"Temporary directory {sionna_result_tmp_dir} deleted.")
    else:
        print(f"Temporary directory {sionna_result_tmp_dir} does not exist.")
        


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
                if mode == 0: # 所有BS同时发射pilot，一共有n_pilot次发射
                    #data = veh_h.sum(axis=-1).reshape(-1)
                    data = np.stack([np.sqrt(P_t)*np.matmul(veh_h[j], DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1) for j in range(look_ahead_len+1)], axis=0)
                    # data = np.sqrt(P_t)*np.matmul(veh_h, DFT_matrix_tx)[:,:,:n_pilot*sample_interval:sample_interval].sum(axis=-2).reshape(-1)
                    n = generate_complex_gaussian_vector(data.shape, scale=np.sqrt(P_noise), mean=0.0)
                    data = (data + n).astype(np.complex64)
                elif mode == 1: # 各BS轮流发射pilot，每个BS各有n_pilot个不同方向，一共有n_bs*n_pilot次发射
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

def get_prepared_dataset(preprocess_mode, DS_start, DS_end, M_t, M_r, freq, n_pilot, N_bs, P_t, P_noise, look_ahead_len=0, datasize_upperbound = 1e9, lbd=0.1):
    # 加载数据集
    if look_ahead_len == 0:
        prepared_dataset_filename = f'{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}'
    else:
        prepared_dataset_filename = f'{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}_lookahead{look_ahead_len}'
    os.makedirs('./prepared_dataset/', exist_ok=True)
    prepared_dataset_filepath = os.path.join('./prepared_dataset/',prepared_dataset_filename+'.pkl')
    if os.path.exists(prepared_dataset_filepath):
        f_read = open(prepared_dataset_filepath, 'rb')
        prepared_dataset = pickle.load(f_read)
        data_np, best_beam_pair_index_np, veh_pos_np, veh_h_np = \
            prepared_dataset['data_np'], prepared_dataset['best_beam_pair_index_np'], prepared_dataset['veh_pos_np'], prepared_dataset['veh_h_np']
        f_read.close()
    else:
        filepath = f'./sionna_result/trajectoryInfo_lbd{lbd:.2f}_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
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


def augment_dataset(data_np, label_np, look_ahead_len, augment_dataset_ratio=2.0):
    """数据增强函数
    Args:
        data_np: 原始时序数据 [num_samples, look_ahead_len+1, ...]
        label_np: 标签数据 [num_samples, ...]
        look_ahead_len: 原始序列长度
        augment_dataset_ratio: 增强后数据集与原数据集的比率 (必须>=1)
    Returns:
        data_torch: 增强后的时序数据
        label_torch: 对应的标签
        lengths: 每个样本的有效序列长度
    """
    num_origin = data_np.shape[0]
    num_total = int(num_origin * augment_dataset_ratio)
    num_augment = num_total - num_origin  # 需要新增的样本数

    # 初始数据 (包含原始数据)
    data_torch = [torch.tensor(data_np[:, :-1, ...])]  # 原始时序数据
    label_torch = [torch.tensor(label_np[:, -1, ...])]  # 原始标签数据
    lengths = [look_ahead_len] * num_origin
    
    # 随机增强剩余部分
    for _ in range(num_augment):
        # 随机选择一个原始样本
        sample_idx = np.random.randint(0, num_origin)
        
        # 随机生成截断长度 (1到look_ahead_len)
        valid_len = np.random.randint(1, look_ahead_len + 1)
        
        # 生成截断序列 (后valid_len个时间步)
        clipped_data = data_np[sample_idx, -valid_len-1:-1, ...]
        
        # 后补零
        padded_data = np.zeros_like(data_np[sample_idx, :-1, ...])
        padded_data[:valid_len] = clipped_data
        
        # 存储增强数据
        data_torch.append(torch.from_numpy(padded_data).unsqueeze(0))
        label_torch.append(torch.from_numpy(label_np[sample_idx, -1, ...]).unsqueeze(0))  # 复制对应标签
        lengths.append(valid_len)
    
    # 合并数据
    data_torch = torch.cat(data_torch, dim=0)
    label_torch = torch.cat(label_torch, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    
    return data_torch, label_torch, lengths

def random_truncate_tensor_sequence(inputs, lengths):
    # inputs: [batch_size, max_seq_len, ...]
    # lengths: [batch_size]
    batch_size, max_seq_len = inputs.shape[0], inputs.shape[1]
    truncated_inputs = torch.zeros_like(inputs)
    truncated_lengths = torch.zeros(batch_size, dtype=torch.int64)
    for i in range(batch_size):
        # 随机选择一个截断长度
        valid_len = random.randint(1, lengths[i].item())
        truncated_inputs[i, :valid_len] = inputs[i, -valid_len:]
        truncated_lengths[i] = valid_len
    return truncated_inputs, truncated_lengths

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