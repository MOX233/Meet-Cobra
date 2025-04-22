import os
import shutil
import time
import subprocess
import threading
import collections
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run_subprocess(params, timeout):
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

def main(
    start_time, 
    end_time, 
    subprocess_time=1, 
    max_workers=3, # 最大线程数
    timeout=3600, # 子进程超时时间（秒）
    gpu=7, # GPU编号
    sumo_traj_path="./sumo_data/trajectory_Lbd0.10.csv",
    freq=28e9, # 28e9 or 5.9e9
    antenna_pattern="iso", # "iso" or "tr38901"
    N_t_H=1,
    N_t_V=64,
    N_r_H=1,
    N_r_V=64,
    h_car=1.6,
    h_rx=3,
    h_tx=30,
    ):
    
    sionna_result_file_save_path = f"./sionna_result/trajectoryInfo_{start_time}_{end_time}_3Dbeam_tx({N_t_H},{N_t_V})_rx({N_r_H},{N_r_V})_freq{freq:.1e}.pkl"
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
            future = executor.submit(run_subprocess, params, timeout)
            futures.append(future)
        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 获取执行结果（自动处理异常）
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
    # 合并结果
    main_trajectoryInfo = collections.OrderedDict()
    for params in PARAMETERS:
        subprocess_start_time = params["sionna_start_time"]
        subprocess_end_time = params["sionna_end_time"]
        with open(os.path.join(sionna_result_tmp_dir, f"time({subprocess_start_time:.1f},{subprocess_end_time:.1f})_tx({N_t_H},{N_t_V})_rx({N_r_H},{N_r_V})_freq{freq:.1e}.pkl"), "rb") as tf:
            sub_trajectoryInfo = pickle.load(tf)
            main_trajectoryInfo.update(sub_trajectoryInfo)
    # 保存结果
    with open(sionna_result_file_save_path, "wb") as tf:
        pickle.dump(main_trajectoryInfo,tf)
        print("successfully save trajectoryInfo to file ",sionna_result_file_save_path)
    # 删除临时文件
    if os.path.exists(sionna_result_tmp_dir):
        shutil.rmtree(sionna_result_tmp_dir)
        print(f"Temporary directory {sionna_result_tmp_dir} deleted.")
    else:
        print(f"Temporary directory {sionna_result_tmp_dir} does not exist.")
        
if __name__ == "__main__":
    main(
    start_time=400, 
    end_time=401.5, 
    subprocess_time=1, 
    max_workers=10, # 最大线程数
    timeout=3600, # 子进程超时时间（秒）
    gpu=7, # GPU编号
    sumo_traj_path="./sumo_data/trajectory_Lbd0.10.csv",
    freq=28e9, # 28e9 or 5.9e9
    antenna_pattern="iso", # "iso" or "tr38901"
    N_t_H=1,
    N_t_V=64,
    N_r_H=1,
    N_r_V=64,
    h_car=1.6,
    h_rx=3,
    h_tx=30,
    )
    