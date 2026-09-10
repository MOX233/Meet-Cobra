from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import collections
import ipdb
import pickle
import torch
import copy
import tqdm
sys.argv = [""]
sys.path.append(os.getcwd())
import numpy as np
from utils.options import args_parser
from utils.NN_utils import BeamPredictionLSTMModel, BestGainPredictionLSTMModel
from utils.channel_utils import (
    generate_CSI_oneUE_multiBS_onlyiidshd,
    update_CSI,
    calculate_uma_pathloss_3gpp_38901,
    calculate_channel_gain_from_pathloss,
    get_g_macroBS_dict,
    rician_channel_gain,
)
from utils.ho_utils import interruption_slots
from utils.queue_utils import (
    init_vehset_backlog_queue,
    init4frame_vehset_backlog_queue,
    update4slot_vehset_backlog_queue,
)
from utils.alg_utils import (
    init_vehset_connection,
    update_BS_association_state,
    update_vehset_connection,
    update_measured_g_record_dict,
    measure_gain,
    measure_gain_for_topKbeam,
    measure_gain_for_topKbeam_savePilot,
    measure_gain_NoBeamforming,
    estimate_num_RB_allocated_perBS,
    RA_Lyapunov,
    RA_OTR_SINR,
    HO_EE_Greedy,
    HO_LowerBound_SINR,
)
from utils.mox_utils import setup_seed, get_save_dirs, split_string, save_log, np2torch, lin2dB, dB2lin, generate_1Dsamples

from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair, generate_dft_codebook
from utils.mox_utils import setup_seed, get_save_dirs, split_string, save_log, np2torch, lin2dB, dB2lin, generate_1Dsamples
from utils.data_utils import preprocess_input_np, generate_complex_gaussian_vector
from utils.beam_utils import generate_dft_codebook, beamPairId_to_beamIdPair


def _predict_vehicle_batch(model, CSI_dict, device, K=None, batch_size=512):
    """Run deterministic per-vehicle predictors in length-homogeneous batches.

    The historical CSI sequence is shorter for newly entering vehicles.  Grouping
    records by array shape lets the existing LSTM ``predict`` methods be reused
    without padding or changing their numerical definition.  This helper is only
    used when a caller explicitly enables ``batch_prediction``.
    """
    predictions = collections.OrderedDict()
    grouped_vehicles = collections.OrderedDict()
    for veh, csi in CSI_dict.items():
        grouped_vehicles.setdefault(tuple(csi.shape), []).append(veh)

    with torch.inference_mode():
        for vehicles in grouped_vehicles.values():
            for start in range(0, len(vehicles), batch_size):
                chunk = vehicles[start : start + batch_size]
                inputs = np.stack([CSI_dict[veh] for veh in chunk], axis=0)
                if K is None:
                    outputs = model.predict(inputs, device)
                else:
                    outputs = model.predict(inputs, device, K=K)
                for veh, output in zip(chunk, outputs):
                    predictions[veh] = output
    return predictions


def get_default_sim_params(save_dir, gpu=0, lbd=1, cut_ratio=1, load_predictors=True):
    # Urban Macro LoS: PL = 28 + 22*log10(d)+20*log10(f)
    # Urban Micro LoS: PL = 32.4 + 21*log10(d)+20*log10(f)
    # data_rate_list = np.logspace(7, 8, 10)
    # data_rate_list = np.linspace(10e6, 200e6, 20)
    # data_rate_list = np.linspace(30e6, 50e6, 11)
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
    sample_interval = int(M_t/n_pilot)
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print('device: ',device)
    args = args_parser()
    args.from_sionna = True
    args.M_t = M_t
    args.M_r = M_r
    args.slots_per_frame = 100
    args.frames_per_sample = 10
    args.num_RB_macro = 133
    args.num_RB_micro = 66
    args.RB_intervel_macro = 0.36 * 1e6
    args.RB_intervel_micro = 1.44 * 1e6
    args.p_macro = 1
    args.p_micro = 0.2
    args.NF_macro_dB = 5
    args.NF_micro_dB = 10
    # args.data_rate = 10 * 1e6
    args.random_factor_range4data_rate = 0.
    args.lat_slot_ub = 20
    args.eta = 1e6
    args.device = device
    args.K = 5 # 每次beam tracking 时选K个最有可能的波束对进行测试
    args.Lambda = lbd # 车辆到达率
    args.note = ""
    args.trajectoryInfo_path = f'./sumo_data/trajectory_Lbd{args.Lambda:.2f}.csv'
    # 对测试数据集进行截断
    cut_end = DS_start + cut_ratio*(DS_end-DS_start)
    save_path = os.path.join(save_dir, f"lbd{args.Lambda:.2f}_{DS_start}_{cut_end}_"
        + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + (f"_{args.note}" if args.note != "" else ""))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./sionna_result', exist_ok=True)
    os.makedirs('./data4sim', exist_ok=True)
    sionna_result_filepath = f'./sionna_result/trajectoryInfo_lbd{args.Lambda:.2f}_{DS_start}_{DS_end}_3Dbeam_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}.pkl'
    data4sim_filepath = f'./data4sim/lbd{args.Lambda:.2f}_{DS_start}_{DS_end}_tx(1,{M_t})_rx(1,{M_r})_freq{freq:.1e}_Np{n_pilot}_mode{preprocess_mode}_lookahead{look_ahead_len}.pkl'
    
    setup_seed(args.seed)
    
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

    # Oracle-only diagnostics need neither NN checkpoint loading nor inference.
    if not load_predictors:
        locations = [np.array(p) for p in ((300, 300), (-300, 300),
                                           (300, -300), (-300, -300))]
        return args, locations, timeline_dir, None, None, None, None, save_path
           
    feature_input_dim = 2 * M_r * n_pilot + 2 * int(preprocess_mode == 2)
    num_bs = N_bs
    num_beampair = M_r * M_t

    beampred_model = BeamPredictionLSTMModel(feature_input_dim, num_bs, num_beampair).to(device)
    beampred_model.load_state_dict(torch.load('./NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models/beampred_lstm_valAcc89.73%_2025-09-19_21:48:48.pth'))
    beampred_model.eval()
    gainpred_model = BestGainPredictionLSTMModel(feature_input_dim, num_bs).to(device)
    gainpred_model.load_state_dict(torch.load('./NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models/gainpred_lstm_valMae4.07dB_2025-09-25_02:04:34.pth'))
    gainpred_model.eval()
    inferpred_model = BestGainPredictionLSTMModel(feature_input_dim, num_bs).to(device)
    inferpred_model.load_state_dict(torch.load('./NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models/inferpred_lstm_valMae3.20dB_2025-11-05_01:24:36.pth'))
    inferpred_model.eval()
    pospred_model = None
        
    # 给定各个基站的位置
    # BS0_loc = np.array([0, 0])
    BS1_loc = np.array([300, 300])
    BS2_loc = np.array([-300, 300])
    BS3_loc = np.array([300, -300])
    BS4_loc = np.array([-300, -300])
    # BS_loc_list = [BS0_loc, BS1_loc, BS2_loc, BS3_loc, BS4_loc]
    BS_loc_list = [BS1_loc, BS2_loc, BS3_loc, BS4_loc]
    # BS_loc_dict = collections.OrderedDict()
    # for i, loc in enumerate(BS_loc_list):
    #     BS_loc_dict[i] = loc
    return args, BS_loc_list, timeline_dir, pospred_model, beampred_model, gainpred_model, inferpred_model, save_path


def run_sim_withUMa(
    args,  # 存储仿真参数设置的args对象
    MicroBS_loc_list,  # 存储各基站位置的list
    timeline_dir,  # 对SUMO生成的车流数据（详细版）进行处理后得到的交通车流信息
    pospred_model,  # 基于CSI预测车辆移动性的AI模型
    beampred_model,
    gainpred_model,
    inferpred_model,
    RA_func=RA_OTR_SINR,  # 资源分配算法
    HO_func=HO_EE_Greedy,  # 越区切换算法
    BF_func='topKbeam',  # 波束成形算法
    prt=True,  # 是否在仿真运行时实时打印相关信息
    save_pilot=False, # 是否执行pilot-saved的测量方法
    No_BF=False, # 是否不使用Beamforming
    MacroBS_loc = [0, 0],  # 宏基站位置，默认在原点
    **kwargs,
):
    if No_BF:
        BF_func = 'NoBeamforming'
    elif save_pilot:
        BF_func = 'topKbeam_savePilot'
    else:
        BF_func = BF_func
    # 'topKbeam_NoPred'
    bpID_microBS_dict = collections.OrderedDict()  # 记录各车辆在上一时隙内对各MicroBS测量的最佳波束对ID
    
    K_BF = kwargs.get('K_BF', None) 
    NoPHO = kwargs.get('NoPHO', False) #TODO
    batch_prediction = kwargs.get('batch_prediction', False)
    prediction_batch_size = kwargs.get('prediction_batch_size', 512)
    prediction_cache = kwargs.get('prediction_cache', None)
    measured_gain_gamma = kwargs.get('measured_gain_gamma', 0.1)
    ho_slots = interruption_slots(kwargs.get('ho_interruption_ms', 0.0),
                                  args.slot_len, args.slots_per_frame)
    ho_capacity_correction = kwargs.get('ho_capacity_correction', False)
    traffic_trace = kwargs.get('traffic_trace')
    oracle_ho_cache = kwargs.get('oracle_ho_cache')
    ho_diagnostics = kwargs.get('ho_diagnostics')
    if ho_slots and any(model is not None for model in
                        (pospred_model, beampred_model, gainpred_model, inferpred_model)):
        raise ValueError("HO interruption is currently validated only for Oracle runs")
    K_BF = K_BF if K_BF is not None else args.K
    device = args.device
    DFT_matrix_tx = generate_dft_codebook(args.M_t)
    DFT_matrix_rx = generate_dft_codebook(args.M_r)
    # 加入MacroBS
    BS_loc_list = copy.copy(MicroBS_loc_list)
    BS_loc_list.insert(0, MacroBS_loc)  # 在列表开头插入宏基站位置
    BS_loc_array = np.array(BS_loc_list)
    BS_loc_dict = collections.OrderedDict()
    for i, loc in enumerate(BS_loc_list):
        BS_loc_dict[i] = loc
    frame_list = list(timeline_dir.keys())
    num_frame = len(frame_list) - 1
    frame_prev = frame_list[0]
    veh_set_prev = set(timeline_dir[frame_prev].keys())  # 前一帧的车辆集合
    veh_set_cur = set()  # 当前帧的车辆集合
    Q_dict_prev = (
        collections.OrderedDict()
    )  # 前一帧的车辆业务数据积压队列长度向量 size=(slots_per_frame+1)
    Q_dict_cur = (
        collections.OrderedDict()
    )  # 当前帧的车辆业务数据积压队列长度向量 size=(slots_per_frame+1)
    connection_dict_prev = collections.OrderedDict()  # 前一帧的车辆-基站连接关系
    connection_dict_cur = collections.OrderedDict()  # 当前帧的车辆-基站连接关系
    CSI_dict_prev = collections.OrderedDict()  # 前一帧的车辆CSI 
    CSI_dict_cur = collections.OrderedDict()  # 当前帧的车辆CSI
    # CSI_dict[veh].shape =  (args.frames_per_sample, 2*M_r*N_pilot)
    measured_g_record_dict_prev = collections.OrderedDict()  # 前一帧的车辆历史接入基站的信道增益记录
    measured_g_record_dict_cur = collections.OrderedDict()  # 当前帧的车辆历史接入基站的信道增益记录

    HO_cmd_prev4cur = (
        collections.OrderedDict()
    )  # 前一帧为当前帧做出的HO决策 HO_cmd 包含需要变更连接关系的veh:BS_id
    HO_cmd_cur4next = (
        collections.OrderedDict()
    )  # 当前帧为下一帧做出的HO决策 HO_cmd 包含需要变更连接关系的veh:BS_id
    BS_association_dict, BS_association_num = update_BS_association_state(
        BS_loc_dict, connection_dict_prev
    )  # 各基站关联用户数量
    
    # 仿真中出现的所有用户
    veh_set_all = set()
    for frame in frame_list:
        veh_set_all.update(set(timeline_dir[frame].keys()))
    
    # 初始化各用户的平均业务数据到达率，基于所有车辆的平均业务数据到达率args.data_rate，乘上均匀分布因子作为随机扰动
    veh_data_rate_dict = collections.OrderedDict()
    assert args.random_factor_range4data_rate >= 0 and args.random_factor_range4data_rate <= 1
    for veh in veh_set_all:
        veh_data_rate_dict[veh] = args.data_rate * np.random.uniform(1-args.random_factor_range4data_rate, 1+args.random_factor_range4data_rate)
    if traffic_trace is not None:
        veh_data_rate_dict.update(traffic_trace['rates'])
    # import ipdb;ipdb.set_trace()
    # print("veh_data_rate_dict:", veh_data_rate_dict)  # debug
    
    Q_ub_dict = collections.OrderedDict()  # 各车辆的队列长度上限阈值
    for veh in veh_set_all:
        Q_ub_dict[veh] = args.lat_slot_ub * veh_data_rate_dict[veh] * args.slot_len

    # 仿真输出结果记录
    energy_record = np.zeros((num_frame,))  # 记录每帧能耗
    HO_time_record = np.zeros((num_frame,))  # 记录每帧HO次数
    HO_cmd_record = collections.OrderedDict()  # 记录每帧HO命令
    violation_prob_record = np.zeros((num_frame,))  # 记录每帧的队列长度违规频率
    avg_queuelen_record = np.zeros((num_frame,))  # 记录每帧平均队列长度
    queuelen_4eachVeh_record = collections.OrderedDict()  # 记录每帧各车辆的队列长度
    pilot_record = np.zeros((num_frame,))  # 记录每帧所用pilot数量
    RB_allocated_record = np.zeros((num_frame,len(BS_loc_list)))  # 记录每帧各基站分配的子载波数

    # 初始化车辆业务数据积压队列
    Q_dict_prev = init_vehset_backlog_queue(
        veh_set_prev, Q_ub_dict, Q_th=0.5, slots_per_frame=args.slots_per_frame
    )
    if traffic_trace is not None:
        for veh in veh_set_prev:
            Q_dict_prev[veh][:] = traffic_trace['initial_queues'][frame_prev][veh]

    # 初始化车辆-基站连接关系
    connection_dict_prev = init_vehset_connection(veh_set_prev, BS_loc_array, timeline_dir[frame_prev], macro=True)

    # 初始化车辆-基站信道状态信息(CSI)
    for veh in veh_set_prev:
        # CSI_dict_prev[veh] = timeline_dir[frame_prev][veh]["CSI_preprocessed"][np.newaxis,...].repeat(args.frames_per_sample, axis=0)
        CSI_dict_prev[veh] = timeline_dir[frame_prev][veh]["CSI_preprocessed"]
    # 初始化车辆历史接入基站的信道增益记录
    for veh in veh_set_prev:
        measured_g_record_dict_prev[veh] = np.zeros((len(BS_loc_list),2))  # shape=(num_BS, 2)
        measured_g_record_dict_prev[veh][:,0] = -1 # 车辆与各基站上一次连接的经过时间(帧数)
        measured_g_record_dict_prev[veh][:,1] = -180 # 车辆与各基站上一次连接时的信道增益(dB)
        
    sim_start_time = time.time()
    # 用 tqdm 进行进度条显示
    for x, frame_cur in tqdm.tqdm(
        enumerate(frame_list[1:]),
        total=num_frame,
        desc='Simulating',
        disable=not prt,
    ):
    #for x, frame_cur in tqdm(enumerate(frame_list[1:])):
        veh_set_cur = set(timeline_dir[frame_cur].keys())
        # print("frame: ", frame_cur, " veh num: ", len(veh_set_cur)) #debug
        veh_set_in = veh_set_cur.difference(veh_set_prev)
        veh_set_out = veh_set_prev.difference(veh_set_cur)
        veh_set_remain = veh_set_cur.intersection(veh_set_prev)
        Q_dict_cur = init4frame_vehset_backlog_queue(
            veh_set_remain,
            veh_set_in,
            Q_dict_prev,
            Q_ub_dict,
            Q_th=0.5,
            slots_per_frame=args.slots_per_frame,
        )
        if traffic_trace is not None:
            for veh in veh_set_in:
                Q_dict_cur[veh][0] = traffic_trace['initial_queues'][frame_cur][veh]
        CSI_dict_cur = collections.OrderedDict() #
        for veh in veh_set_cur:
            CSI_dict_cur[veh] = timeline_dir[frame_cur][veh]["CSI_preprocessed"].astype(np.float32)
        # CSI_dict_cur = update_CSI(
        #     args,
        #     veh_set_remain,
        #     veh_set_in,
        #     CSI_dict_prev,
        #     trajectoryInfo=timeline_dir[frame_cur],
        # )
                
        connection_dict_cur, HO_cnt4frame = update_vehset_connection(
            veh_set_remain, veh_set_in, connection_dict_prev, HO_cmd_prev4cur, BS_loc_array, timeline_dir[frame_cur], macro=True
        )
        switched = {veh for veh in veh_set_remain
                    if connection_dict_cur[veh] != connection_dict_prev[veh]}
        if ho_slots or ho_diagnostics is not None:
            assert len(switched) == HO_cnt4frame
        BS_association_dict, BS_association_num = update_BS_association_state(
            BS_loc_dict, connection_dict_cur
        )  # 各基站关联用户数量

        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期的期望位置
        pred_loc_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if pospred_model is None:
                pred_loc_dict[veh] = timeline_dir[frame_cur][veh]["pos"]
            else:
                pred_loc_dict[veh] = pospred_model.predict(CSI_dict_cur[veh],device)
                
        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期与各MicroBS间的最优波束增益     
        pred_gain_opt_beam_dict = collections.OrderedDict()
        if gainpred_model is not None and prediction_cache is not None:
            for veh in veh_set_cur:
                pred_gain_opt_beam_dict[veh] = prediction_cache[frame_cur][veh]["gain"]
        elif gainpred_model is not None and batch_prediction:
            pred_gain_opt_beam_dict.update(
                _predict_vehicle_batch(
                    gainpred_model,
                    CSI_dict_cur,
                    device,
                    batch_size=prediction_batch_size,
                )
            )
        else:
            for veh in veh_set_cur:
                if gainpred_model is None:
                    pred_gain_opt_beam_dict[veh] = timeline_dir[frame_cur][veh]["g_opt_beam"]
                else:
                    pred_gain_opt_beam_dict[veh] = gainpred_model.predict(CSI_dict_cur[veh][np.newaxis,...],device)[0]
        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期与各MicroBS间的最优波束方向
        pred_beamPairId_dict = collections.OrderedDict()
        if beampred_model is not None and prediction_cache is not None:
            for veh in veh_set_cur:
                pred_beamPairId_dict[veh] = prediction_cache[frame_cur][veh]["beam"][:, :K_BF]
        elif beampred_model is not None and batch_prediction:
            pred_beamPairId_dict.update(
                _predict_vehicle_batch(
                    beampred_model,
                    CSI_dict_cur,
                    device,
                    K=K_BF,
                    batch_size=prediction_batch_size,
                )
            )
        else:
            for veh in veh_set_cur:
                if beampred_model is None:
                    pred_beamPairId_dict[veh] = timeline_dir[frame_cur][veh]["best_beam_pair_idx"].reshape(-1,1).repeat(K_BF,axis=-1)
                else:
                    pred_beamPairId_dict[veh] = beampred_model.predict(CSI_dict_cur[veh][np.newaxis,...],device, K=K_BF)[0]
                # pred_beamPairId_dict[veh].shape = (4,K)
        
        pred_g_macroBS_dict = get_g_macroBS_dict(args, pred_loc_dict, MacroBS_loc, fc_ghz=2.8, Gt_macro=0, scenario='los')
        # print('snr_macro_pred', {veh:10*np.log10(dB2lin(pred_g_macroBS_dict[veh]) * args.p_macro / (args.N0 * args.RB_intervel_macro * dB2lin(args.NF_macro_dB))).item() for veh in pred_g_macroBS_dict.keys()})
        
        pred_g_dict = collections.OrderedDict()
        pred_infer_g_dict = collections.OrderedDict() if inferpred_model is not None else None
        pred_infer_micro_dict = None
        if inferpred_model is not None and prediction_cache is not None:
            pred_infer_micro_dict = {
                veh: prediction_cache[frame_cur][veh]["interference"]
                for veh in veh_set_cur
            }
        elif inferpred_model is not None and batch_prediction:
            pred_infer_micro_dict = _predict_vehicle_batch(
                inferpred_model,
                CSI_dict_cur,
                device,
                batch_size=prediction_batch_size,
            )
        for veh in pred_g_macroBS_dict.keys():
            pred_g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], pred_gain_opt_beam_dict[veh]), axis=0)
            if inferpred_model is not None:
                infer_micro = pred_infer_micro_dict[veh] if pred_infer_micro_dict is not None else inferpred_model.predict(CSI_dict_cur[veh][np.newaxis,...],device)[0]
                pred_infer_g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], infer_micro), axis=0)

        # 在每一帧内，让各车对各MicroBS的K个波束对进行测量
        g_microBS_dict, g_microBS_NoBF_dict, bpID_microBS_dict, num_pilot_dict = \
            measure_gain(args, frame_cur, veh_set_cur, timeline_dir, MicroBS_loc_list, pred_beamPairId_dict, pred_gain_opt_beam_dict, \
                DFT_matrix_tx, DFT_matrix_rx, BF_func, bpID_microBS_dict, rician_fading=False, K_BF=K_BF)
        
        g_dict = collections.OrderedDict()
        g_NoBF_dict = collections.OrderedDict()
        for veh in pred_g_macroBS_dict.keys():
            g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], g_microBS_dict[veh]), axis=0)
            g_NoBF_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], g_microBS_NoBF_dict[veh]), axis=0)
        measured_g_record_dict_cur = update_measured_g_record_dict(g_dict, measured_g_record_dict_prev, veh_set_cur, connection_dict_cur, BS_loc_list)    
                      
        def predict_g_from_measured_g_record_dict(measured_g_record_dict, pred_g_dict, veh_set_cur, gamma=0.1):
            for veh in veh_set_cur:
                elapsed_frame = measured_g_record_dict[veh][:,0]
                last_measured_g = measured_g_record_dict[veh][:,1]
                valid_mask = elapsed_frame>=0
                pred_g_dict[veh] = ~valid_mask * pred_g_dict[veh] + \
                    valid_mask * (gamma**(elapsed_frame+1)*last_measured_g + (1-gamma**(elapsed_frame+1)) * pred_g_dict[veh])
            return pred_g_dict
        pred_g_dict = predict_g_from_measured_g_record_dict(
            measured_g_record_dict_cur,
            pred_g_dict,
            veh_set_cur,
            gamma=measured_gain_gamma,
        )
        
        # 基于积压队列、预测位置、BS位置等信息进行PHO决策
        # 将num_RB_allocated_perBS的估计转到在本程序中进行
        est_num_RB_allocated_perBS = estimate_num_RB_allocated_perBS(args, connection_dict_cur, BS_loc_array, veh_set_cur, g_dict, 
                                                                     veh_data_rate_dict, infer_g_dict=pred_infer_g_dict if inferpred_model is not None else g_NoBF_dict)

        # PHO决策
        vio_prob_history = violation_prob_record[:x]
        ho_gain = pred_g_dict
        ho_interference = pred_infer_g_dict if inferpred_model is not None else g_NoBF_dict
        ho_positions, ho_pilots = pred_loc_dict, num_pilot_dict
        if oracle_ho_cache is not None and x + 2 < len(frame_list):
            # True next-frame Oracle for continuing vehicles; departure fallback
            # cannot affect service to a vehicle that is no longer in the scene.
            future = oracle_ho_cache[frame_list[x + 2]]
            ho_gain = {v: future['gain'].get(v, pred_g_dict[v]) for v in veh_set_cur}
            ho_interference = {v: future['interference'].get(v, g_NoBF_dict[v]) for v in veh_set_cur}
            ho_positions = {v: future['positions'].get(v, pred_loc_dict[v]) for v in veh_set_cur}
            ho_pilots = {v: future['pilots'].get(v, num_pilot_dict[v]) for v in veh_set_cur}
        ho_options = {}
        if ho_capacity_correction:
            ho_options = dict(current_connection=connection_dict_cur,
                              ho_capacity_correction=True, ho_interruption_slots=ho_slots)
        HO_cmd_cur4next, pred_num_RB_allocated_perBS = HO_func(
            args, veh_set_cur, Q_dict_cur, veh_data_rate_dict, ho_positions, ho_gain, BS_loc_array,
            infer_g_dict=ho_interference,
            num_pilot_dict=ho_pilots,
            vio_prob_history=vio_prob_history,
            **ho_options,
        )
        
        # 模拟车辆业务数据到达过程 in a frame
        a_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if traffic_trace is None:
                a_dict[veh] = np.random.poisson(
                    veh_data_rate_dict[veh] * args.slot_len,
                    size=(args.slots_per_frame)
                )
            else:
                a_dict[veh] = traffic_trace['arrivals'][frame_cur][veh]
        
        energy4frame = 0  # 统计当前帧的能耗
            
        pilot_slot_record = np.zeros((args.slots_per_frame,))  # 记录当前帧的每个时隙所用pilot数量
        for i in range(0, args.slots_per_frame):
            blocked = switched if i < ho_slots else set()
            
            # # 在每一【时隙】内，让各车对各MicroBS的K个波束对进行测量
            if beampred_model is not None:
                g_microBS_slot_dict, g_microBS_NoBF_slot_dict, bpID_microBS_dict, num_pilot_slot_dict = \
                    measure_gain(args, frame_cur, veh_set_cur, timeline_dir, MicroBS_loc_list, pred_beamPairId_dict, pred_gain_opt_beam_dict, \
                        DFT_matrix_tx, DFT_matrix_rx, BF_func, bpID_microBS_dict, rician_fading=True, K_BF=K_BF)
            else:
                g_microBS_slot_dict = g_microBS_dict
                g_microBS_NoBF_slot_dict = g_microBS_NoBF_dict
                num_pilot_slot_dict = num_pilot_dict
            if blocked:
                # Oracle channel extraction is offline, not physical training.
                # Charge no beam-search attempts to an interrupted vehicle.
                num_pilot_slot_dict = dict(num_pilot_slot_dict)
                for veh in blocked:
                    num_pilot_slot_dict[veh] = np.zeros(len(MicroBS_loc_list))

            g_slot_dict = collections.OrderedDict()
            g_slot_NoBF_dict = collections.OrderedDict()
            
            for veh in pred_g_macroBS_dict.keys():
                g_slot_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], g_microBS_slot_dict[veh]), axis=0)
                g_slot_NoBF_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], g_microBS_NoBF_slot_dict[veh]), axis=0)
            
            # g_slot_dict = g_dict
            # g_slot_NoBF_dict = g_NoBF_dict
            # num_pilot_slot_dict = num_pilot_dict
            
            pilot_slot_record[i] = np.array([num_pilot_slot_dict[veh][connection_dict_cur[veh]-1] if connection_dict_cur[veh]!=0 else 0 
                                        for veh in veh_set_cur]).mean()
            # g_slot_dict = collections.OrderedDict()
            # for veh_id in g_dict.keys():
            #     rician_factor = lin2dB(rician_channel_gain(args.K_rician, size=len(g_dict[veh_id])))
            #     g_slot_dict[veh_id] = g_dict[veh_id] + rician_factor
            
            RA_dict = collections.OrderedDict((veh, 0) for veh in blocked)
            num_RB_allocated_perBS = np.zeros((len(BS_loc_list),), dtype=int) #各基站分配的子载波数
            # 各基站逐时隙进行子载波分配
            for BS_id in range(len(BS_loc_list)):
                BS_RA_dict = RA_func(
                    args,
                    slot_idx=i,
                    BS_id=BS_id,
                    veh_set=([v for v in BS_association_dict[BS_id] if v not in blocked]
                             if blocked else BS_association_dict[BS_id]),
                    veh_data_rate_dict=veh_data_rate_dict,
                    Q_ub_dict=Q_ub_dict,
                    q_dict=Q_dict_cur,
                    a_dict=a_dict,
                    g_dict=g_slot_dict,
                    num_pilot_dict=num_pilot_slot_dict,
                    BS_association_dict=BS_association_dict,
                    infer_g_dict=g_NoBF_dict,
                    est_num_RB_allocated_perBS=est_num_RB_allocated_perBS,
                )
                # if x > 0 and len(BS_RA_dict) > 0 and max(BS_RA_dict.values()) > 50:
                #     print(f'BS_id: {BS_id}, RA_dict: {BS_RA_dict}')
                #     ipdb.set_trace()
                RA_dict.update(BS_RA_dict)
                num_RB_allocated_perBS[BS_id] = sum(BS_RA_dict.values())
                energy4frame += num_RB_allocated_perBS[BS_id] * (args.p_micro if BS_id > 0 else args.p_macro) * args.slot_len
            # print(RA_dict)
            # if RA_dict[2.78]>30:
            #     ipdb.set_trace()
            # 队列更新
            Q_dict_cur = update4slot_vehset_backlog_queue(
                args,
                slot_idx=i,
                RA_dict=RA_dict,
                veh_set=veh_set_cur,
                connection_dict=connection_dict_cur,
                backlog_queue_dict=Q_dict_cur,
                a_dict=a_dict,
                g_dict=g_slot_dict,
                infer_g_dict=g_NoBF_dict,
                num_RB_allocated_perBS=num_RB_allocated_perBS,
                num_pilot_dict=num_pilot_slot_dict,
                sinr_flag=True,
            )
            if ho_diagnostics is not None:
                for veh in blocked:
                    assert RA_dict[veh] == 0
                    assert np.all(num_pilot_slot_dict[veh] == 0)
                    assert Q_dict_cur[veh][i + 1] == Q_dict_cur[veh][i] + a_dict[veh][i]
            # import ipdb;ipdb.set_trace()
            # print(num_pilot_slot_dict)
            # for veh, num_pilot in num_pilot_slot_dict.items():
            #     print(f'veh: {veh}, num_pilot: {num_pilot[connection_dict_cur[veh]-1] if connection_dict_cur[veh]!=0 else None}')
            RB_allocated_record[x,:] += num_RB_allocated_perBS
        
        queuelen_4eachVeh_record[x] = collections.OrderedDict()    
        for veh in veh_set_cur:
            queuelen_4eachVeh_record[x][veh] = Q_dict_cur[veh][1:].copy()
        
        # 统计当前帧所用pilot数量        
        pilot_record[x] = pilot_slot_record.mean()
        # 统计当前帧各基站平均分配的子载波数
        RB_allocated_record[x,:] /= args.slots_per_frame
        
        violation_cnt = sum(
            [(Q_dict_cur[veh][1:] > Q_ub_dict[veh]).sum() for veh in veh_set_cur]
        )  # 队列超上限次数
        judgement_cnt = (
            len(veh_set_cur) * args.slots_per_frame
        )  # 判定队列是否超上限的次数
        # if prt:
        #     print(
        #         "violation_cnt",
        #         violation_cnt,
        #         "judgement_cnt",
        #         judgement_cnt,
        #         "violation_cnt/judgement_cnt",
        #         violation_cnt / judgement_cnt,
        #     )

        energy_record[x] = energy4frame
        HO_time_record[x] = HO_cnt4frame
        violation_prob_record[x] = violation_cnt / judgement_cnt
        avg_queuelen_record[x] = (
            sum([Q_dict_cur[veh][1:].sum() for veh in veh_set_cur])
            / judgement_cnt
        )
        HO_cmd_record[x] = HO_cmd_prev4cur
        if ho_diagnostics is not None:
            ho_diagnostics.append(dict(
                frame=frame_cur, association=dict(connection_dict_cur),
                switched=sorted(switched, key=repr),
                blocked_vehicle_slots=len(switched) * ho_slots,
                active_vehicle_slots=len(veh_set_cur) * args.slots_per_frame,
            ))
        # if prt:
        #     print("\n\nframe: ", x, frame_cur)
        #     print("BS_association_num: ", BS_association_num)
        #     print("energy_record: ", energy_record[x])
        #     print("HO_time_record: ", HO_time_record[x])
        #     print("violation_prob_record: ", violation_prob_record[x])
        #     print("avg_queuelen_record: ", avg_queuelen_record[x])
        #     print("pilot_record: ", pilot_record[x])

        # 记录当前帧各状态，为下一帧做准备
        frame_prev = frame_cur
        veh_set_prev = veh_set_cur
        Q_dict_prev = Q_dict_cur
        connection_dict_prev = connection_dict_cur
        CSI_dict_prev = CSI_dict_cur
        measured_g_record_dict_prev = measured_g_record_dict_cur
        HO_cmd_prev4cur = HO_cmd_cur4next
        
        # 使用tqdm库，通过进度条的形式展示仿真进度，并以时分秒的格式显示已消耗的时间和预计等待的时间
        if prt:
            progress = (x + 1) / num_frame * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            if x + 1 == num_frame:
                print()
            elapsed_time = time.time() - sim_start_time
            estimated_total_time = elapsed_time / (x + 1) * num_frame
            remaining_time = estimated_total_time - elapsed_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            print(f"\rProgress: |{bar}| {progress:.2f}% Elapsed Time: {elapsed_str} Remaining Time: {remaining_str}", end='')
    if prt:
        print("")
    
    return (
        energy_record,  # 存储各帧的系统能耗（单位：J）
        HO_time_record,  # 存储各帧的HO次数
        HO_cmd_record,  # 存储各帧的HO命令
        violation_prob_record,  # 存储各帧的UE业务积压队列长度超阈值的频率
        avg_queuelen_record,  # 存储各帧的UE业务积压队列长度均值
        pilot_record,
        RB_allocated_record,
        queuelen_4eachVeh_record,
    )


def run_sim_withUMa_analyzed_lowerbound(
    args,  # 存储仿真参数设置的args对象
    MicroBS_loc_list,  # 存储各基站位置的list
    timeline_dir,  # 对SUMO生成的车流数据（详细版）进行处理后得到的交通车流信息
    pospred_model,  # 基于CSI预测车辆移动性的AI模型
    beampred_model,
    gainpred_model,
    inferpred_model,
    RA_func=RA_OTR_SINR,  # 资源分配算法
    HO_func=HO_LowerBound_SINR,  # 越区切换算法
    prt=True,  # 是否在仿真运行时实时打印相关信息
    No_BF=False, # 是否不使用Beamforming
    MacroBS_loc = [0, 0],  # 宏基站位置，默认在原点
    **kwargs,
):
    K_BF = kwargs.get('K_BF', None) 
    K_BF = K_BF if K_BF is not None else args.K
    device = args.device
    DFT_matrix_tx = generate_dft_codebook(args.M_t)
    DFT_matrix_rx = generate_dft_codebook(args.M_r)
    # 加入MacroBS
    BS_loc_list = copy.copy(MicroBS_loc_list)
    BS_loc_list.insert(0, MacroBS_loc)  # 在列表开头插入宏基站位置
    BS_loc_array = np.array(BS_loc_list)
    BS_loc_dict = collections.OrderedDict()
    for i, loc in enumerate(BS_loc_list):
        BS_loc_dict[i] = loc
    frame_list = list(timeline_dir.keys())
    num_frame = len(frame_list) - 1
    frame_prev = frame_list[0]
    veh_set_prev = set(timeline_dir[frame_prev].keys())  # 前一帧的车辆集合
    veh_set_cur = set()  # 当前帧的车辆集合
    CSI_dict_prev = collections.OrderedDict()  # 前一帧的车辆CSI 
    CSI_dict_cur = collections.OrderedDict()  # 当前帧的车辆CSI
    # CSI_dict[veh].shape =  (args.frames_per_sample, 2*M_r*N_pilot)
    
    # 仿真中出现的所有用户
    veh_set_all = set()
    for frame in frame_list:
        veh_set_all.update(set(timeline_dir[frame].keys()))
    
    # 初始化各用户的平均业务数据到达率，基于所有车辆的平均业务数据到达率args.data_rate，乘上均匀分布因子作为随机扰动
    veh_data_rate_dict = collections.OrderedDict()
    assert args.random_factor_range4data_rate >= 0 and args.random_factor_range4data_rate <= 1
    for veh in veh_set_all:
        veh_data_rate_dict[veh] = args.data_rate * np.random.uniform(1-args.random_factor_range4data_rate, 1+args.random_factor_range4data_rate)
    # import ipdb;ipdb.set_trace()
    # print("veh_data_rate_dict:", veh_data_rate_dict)  # debug
    
    Q_ub_dict = collections.OrderedDict()  # 各车辆的队列长度上限阈值
    for veh in veh_set_all:
        Q_ub_dict[veh] = args.lat_slot_ub * veh_data_rate_dict[veh] * args.slot_len

    # 仿真输出结果记录
    energy_record = np.zeros((num_frame,))  # 记录每帧能耗
    violation_prob_record = np.zeros((num_frame,))  # 记录每帧的队列长度违规频率


    # 初始化车辆-基站信道状态信息(CSI)
    for veh in veh_set_prev:
        CSI_dict_prev[veh] = timeline_dir[frame_prev][veh]["CSI_preprocessed"]
        
    # 用 tqdm 进行进度条显示
    for x, frame_cur in tqdm.tqdm(
        enumerate(frame_list[1:]),
        total=num_frame,
        desc='Simulating',
        disable=not prt,
    ):
        veh_set_cur = set(timeline_dir[frame_cur].keys())
        # print("frame: ", frame_cur, " veh num: ", len(veh_set_cur)) #debug
        veh_set_in = veh_set_cur.difference(veh_set_prev)
        veh_set_out = veh_set_prev.difference(veh_set_cur)
        veh_set_remain = veh_set_cur.intersection(veh_set_prev)
        CSI_dict_cur = collections.OrderedDict() #
        for veh in veh_set_cur:
            CSI_dict_cur[veh] = timeline_dir[frame_cur][veh]["CSI_preprocessed"].astype(np.float32)

        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期的期望位置
        pred_loc_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if pospred_model is None:
                pred_loc_dict[veh] = timeline_dir[frame_cur][veh]["pos"]
            else:
                pred_loc_dict[veh] = pospred_model.predict(CSI_dict_cur[veh],device)
        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期与各MicroBS间的最优波束增益     
        pred_gain_opt_beam_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if gainpred_model is None:
                pred_gain_opt_beam_dict[veh] = timeline_dir[frame_cur][veh]["g_opt_beam"]
            else:
                pred_gain_opt_beam_dict[veh] = gainpred_model.predict(CSI_dict_cur[veh][np.newaxis,...],device)[0]
                # pred_gain_opt_beam_dict[veh].shape = (4,)
            # if timeline_dir[frame_cur][veh]["g_opt_beam"][pred_gain_opt_beam_dict[veh].argmax()] <= -180:
            #     print('bug!')
                # ipdb.set_trace()
        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期与各MicroBS间的最优波束方向
        pred_beamPairId_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if beampred_model is None:
                pred_beamPairId_dict[veh] = timeline_dir[frame_cur][veh]["best_beam_pair_idx"].reshape(-1,1).repeat(K_BF,axis=-1)
            else:
                pred_beamPairId_dict[veh] = beampred_model.predict(CSI_dict_cur[veh][np.newaxis,...],device, K=K_BF)[0]
                # pred_beamPairId_dict[veh].shape = (4,K)
        
        pred_g_macroBS_dict = get_g_macroBS_dict(args, pred_loc_dict, MacroBS_loc, fc_ghz=2.8, Gt_macro=0, scenario='los')
        # print('snr_macro_pred', {veh:10*np.log10(dB2lin(pred_g_macroBS_dict[veh]) * args.p_macro / (args.N0 * args.RB_intervel_macro * dB2lin(args.NF_macro_dB))).item() for veh in pred_g_macroBS_dict.keys()})
        
        pred_g_dict = collections.OrderedDict()
        pred_infer_g_dict = collections.OrderedDict() if inferpred_model is not None else None
        for veh in pred_g_macroBS_dict.keys():
            pred_g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], pred_gain_opt_beam_dict[veh]), axis=0)
            if inferpred_model is not None:
                pred_infer_g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], inferpred_model.predict(CSI_dict_cur[veh][np.newaxis,...],device)[0]), axis=0)

        # 在每一帧内，让各车对各MicroBS的K个波束对进行测量
        if No_BF:
            g_microBS_dict, g_microBS_NoBF_dict, bpID_microBS_dict, num_pilot_dict = \
                measure_gain_NoBeamforming(args, frame_cur, veh_set_cur, timeline_dir, MicroBS_loc_list, rician_fading=False)
        else:
            g_microBS_dict, g_microBS_NoBF_dict, bpID_microBS_dict, num_pilot_dict = \
                measure_gain_for_topKbeam(args, frame_cur, veh_set_cur, timeline_dir, MicroBS_loc_list, \
                                          pred_beamPairId_dict, DFT_matrix_tx, DFT_matrix_rx, rician_fading=False, K_BF=K_BF)
        g_dict = collections.OrderedDict()
        g_NoBF_dict = collections.OrderedDict()
        for veh in pred_g_macroBS_dict.keys():
            g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], g_microBS_dict[veh]), axis=0)
            g_NoBF_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], g_microBS_NoBF_dict[veh]), axis=0)
        
        # 基于积压队列、预测位置、BS位置等信息进行PHO决策
        vio_prob_history = violation_prob_record[:x]
        HO_cmd, energy4frame, pred_num_RB_allocated_perBS = HO_func(
            args, veh_set_cur, None, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array,
            infer_g_dict=pred_infer_g_dict if inferpred_model is not None else g_NoBF_dict,
            num_pilot_dict=num_pilot_dict,
            vio_prob_history=vio_prob_history,
        )
        violation_prob_record4frame = 1 if HO_cmd is None else 0

        energy_record[x] = energy4frame
        violation_prob_record[x] = violation_prob_record4frame
        # if prt:
        #     print("\n\nframe: ", x, frame_cur)
        #     print("energy_record: ", energy_record[x])
        #     print("violation_prob_record: ", violation_prob_record[x])

        # 记录当前帧各状态，为下一帧做准备
        frame_prev = frame_cur
        veh_set_prev = veh_set_cur
        CSI_dict_prev = CSI_dict_cur
        # End
    
    return (
        energy_record,  # 存储各帧的系统能耗（单位：J）
        violation_prob_record,  # 存储各帧的UE业务积压队列长度超阈值的频率
    )
