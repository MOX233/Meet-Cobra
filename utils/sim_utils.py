from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import collections
import ipdb
import torch
import copy
sys.argv = [""]
sys.path.append(os.getcwd())
import numpy as np
from utils.options import args_parser
from utils.sumo_utils import (
    read_trajectoryInfo_carindex,
    read_trajectoryInfo_carindex_matrix,
    read_trajectoryInfo_timeindex,
)
from utils.channel_utils import (
    generate_CSI_oneUE_multiBS_onlyiidshd,
    update_CSI,
    calculate_uma_pathloss_3gpp_38901,
    calculate_channel_gain_from_pathloss,
    get_g_macroBS_dict
)
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
    measure_gain_for_topKbeam,
    measure_gain_for_topKbeam_savePilot,
    measure_gain_NoBeamforming,
    HO_nearest,
    RA_Lyapunov,
)
from utils.mox_utils import lin2dB
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair, generate_dft_codebook

def run_sim_withUMa(
    args,  # 存储仿真参数设置的args对象
    MicroBS_loc_list,  # 存储各基站位置的list
    timeline_dir,  # 对SUMO生成的车流数据（详细版）进行处理后得到的交通车流信息
    pospred_model,  # 基于CSI预测车辆移动性的AI模型
    beampred_model,
    gainpred_model,
    RA_func=RA_Lyapunov,  # 资源分配算法
    HO_func=HO_nearest,  # 越区切换算法
    prt=True,  # 是否在仿真运行时实时打印相关信息
    save_pilot=False, # 是否执行pilot-saved的测量方法
    No_BF=False, # 是否不使用Beamforming
    MacroBS_loc = [0, 0]  # 宏基站位置，默认在原点
):
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
    measured_g_record_dict_prev = collections.OrderedDict()  # 前一帧的车辆历史接入基站的信道增益记录 #TODO
    measured_g_record_dict_cur = collections.OrderedDict()  # 当前帧的车辆历史接入基站的信道增益记录 #TODO

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
    
    Q_ub_dict = collections.OrderedDict()  # 各车辆的队列长度上限阈值
    for veh in veh_set_all:
        Q_ub_dict[veh] = args.lat_slot_ub * veh_data_rate_dict[veh] * args.slot_len

    # 仿真输出结果记录
    energy_record = np.zeros((num_frame,))  # 记录每帧能耗
    HO_time_record = np.zeros((num_frame,))  # 记录每帧HO次数
    HO_cmd_record = collections.OrderedDict()  # 记录每帧HO命令
    violation_prob_record = np.zeros((num_frame,))  # 记录每帧的队列长度违规频率
    avg_queuelen_record = np.zeros((num_frame,))  # 记录每帧平均队列长度
    pilot_record = np.zeros((num_frame,))  # 记录每帧所用pilot数量

    # 初始化车辆业务数据积压队列
    Q_dict_prev = init_vehset_backlog_queue(
        veh_set_prev, Q_ub_dict, Q_th=0.5, slots_per_frame=args.slots_per_frame
    )

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
        
    for x, frame_cur in enumerate(frame_list[1:]):
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
                pred_beamPairId_dict[veh] = timeline_dir[frame_cur][veh]["best_beam_pair_idx"].reshape(-1,1).repeat(args.K,axis=-1)
            else:
                pred_beamPairId_dict[veh] = beampred_model.predict(CSI_dict_cur[veh][np.newaxis,...],device, K=args.K)[0]
                # pred_beamPairId_dict[veh].shape = (4,K)
        
        pred_g_macroBS_dict = get_g_macroBS_dict(args, pred_loc_dict, MacroBS_loc, fc_ghz=2.8, Gt_macro=17, scenario='los')
        pred_g_dict = collections.OrderedDict()
        for veh in pred_g_macroBS_dict.keys():
            pred_g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], pred_gain_opt_beam_dict[veh]), axis=0)

        
        # 让各车对各MicroBS的K个波束对进行测量
        if No_BF:
            g_microBS_dict, bpID_microBS_dict, num_pilot = measure_gain_NoBeamforming(frame_cur, veh_set_cur, timeline_dir, MicroBS_loc_list)
        elif save_pilot:
            g_microBS_dict, bpID_microBS_dict, num_pilot = measure_gain_for_topKbeam_savePilot(args, frame_cur, veh_set_cur, timeline_dir, MicroBS_loc_list, \
                                                                     pred_beamPairId_dict, pred_gain_opt_beam_dict, DFT_matrix_tx, DFT_matrix_rx)
        else:
            g_microBS_dict, bpID_microBS_dict, num_pilot = measure_gain_for_topKbeam(args, frame_cur, veh_set_cur, timeline_dir, MicroBS_loc_list, \
                                                                         pred_beamPairId_dict, DFT_matrix_tx, DFT_matrix_rx)
        g_dict = collections.OrderedDict()
        for veh in pred_g_macroBS_dict.keys():
            g_dict[veh] = np.concatenate(([pred_g_macroBS_dict[veh]], g_microBS_dict[veh]), axis=0)

        measured_g_record_dict_cur = update_measured_g_record_dict(g_dict, measured_g_record_dict_prev, veh_set_cur, connection_dict_cur, BS_loc_list)    
        
        # def see_micro_macro_diff(veh, fc_ghz=5.9, Gt_macro=0, NF1=5, NF2=10, scenario='nlos'):
        #     PL = calculate_uma_pathloss_3gpp_38901(distance_2d_m=np.linalg.norm(pred_loc_dict[veh]-MacroBS_loc),
        #                                         fc_ghz=fc_ghz,
        #                                         h_ut_m=args.h_car,
        #                                         h_bs_m=args.h_tx,
        #                                         scenario=scenario)
        #     SE_macro = np.log2(1+calculate_channel_gain_from_pathloss(NF1+PL-Gt_macro)*args.p_macro/(args.N0*args.RB_intervel_macro))
        #     SE_micro = np.log2(1+1/16*calculate_channel_gain_from_pathloss(NF2-max(g_microBS_dict[veh]))*args.p_micro/(args.N0*args.RB_intervel_micro))
        #     RBE_macro = args.RB_intervel_macro * SE_macro
        #     RBE_micro = args.RB_intervel_micro * SE_micro
        #     EE_macro = RBE_macro / args.p_macro
        #     EE_micro = RBE_micro / args.p_micro
        #     print(f'SE_macro: {SE_macro:.2f} bps/Hz, SE_micro: {SE_micro:.2f} bps/Hz, '
        #         f'RBE_macro: {RBE_macro/1e6:.2f} Mbps, RBE_micro: {RBE_micro/1e6:.2f} Mbps'
        #         f', EE_macro: {EE_macro/1e3:.2e} bps/mW, EE_micro: {EE_micro/1e3:.2e} bps/mW')
        #     return SE_macro, SE_micro, RBE_macro, RBE_micro    
        # _ = see_micro_macro_diff(veh, fc_ghz=2.8, Gt_macro=17, NF1=5, NF2=10, scenario='los')
        # np.linalg.norm(BS_loc_array-pred_loc_dict[veh],axis=1)
        # import ipdb; ipdb.set_trace()
        
        
        pilot_record[x] += num_pilot 
            # veh7.68 与BS-4的信道可能被阻挡，但是仍然被model预测错为最优基站
        
        # TODO
        #ipdb.set_trace()
        def predict_g_from_measured_g_record_dict(measured_g_record_dict, pred_g_dict, veh_set_cur, gamma=0.9):
            for veh in veh_set_cur:
                elapsed_frame = measured_g_record_dict[veh][:,0]
                last_measured_g = measured_g_record_dict[veh][:,1]
                valid_mask = elapsed_frame>=0
                pred_g_dict[veh] = ~valid_mask * pred_g_dict[veh] + \
                    valid_mask * (gamma**(elapsed_frame+1)*last_measured_g + (1-gamma**(elapsed_frame+1)) * pred_g_dict[veh])
            return pred_g_dict
        pred_g_dict = predict_g_from_measured_g_record_dict(measured_g_record_dict_cur, pred_g_dict, veh_set_cur)
        #ipdb.set_trace()
        
        
        # 基于积压队列、预测位置、BS位置等信息进行PHO决策
        HO_cmd_cur4next = HO_func(
            args, veh_set_cur, Q_dict_cur, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array
        )

        # 模拟车辆业务数据到达过程 in a frame
        a_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            a_dict[veh] = np.random.poisson(
                veh_data_rate_dict[veh] * args.slot_len,
                size=(args.slots_per_frame)
            )
        
        energy4frame = 0  # 统计当前帧的能耗
        for i in range(0, args.slots_per_frame):
            RA_dict = collections.OrderedDict()
            # 各基站逐时隙进行子载波分配
            for BS_id in range(len(BS_loc_list)):
                BS_RA_dict = RA_func(
                    args,
                    slot_idx=i,
                    BS_id=BS_id,
                    veh_set=BS_association_dict[BS_id],
                    veh_data_rate_dict=veh_data_rate_dict,
                    Q_ub_dict=Q_ub_dict,
                    q_dict=Q_dict_cur,
                    a_dict=a_dict,
                    g_dict=g_dict,
                )
                # if x > 0 and len(BS_RA_dict) > 0 and max(BS_RA_dict.values()) > 50:
                #     print(f'BS_id: {BS_id}, RA_dict: {BS_RA_dict}')
                #     ipdb.set_trace()
                RA_dict.update(BS_RA_dict)
                energy4frame += sum(BS_RA_dict.values()) * (args.p_micro if BS_id > 0 else args.p_macro) * args.slot_len
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
                g_dict=g_dict,
            )
        
        violation_cnt = sum(
            [(Q_dict_cur[veh][1:] > Q_ub_dict[veh]).sum() for veh in veh_set_cur]
        )  # 队列超上限次数
        judgement_cnt = (
            len(veh_set_cur) * args.slots_per_frame
        )  # 判定队列是否超上限的次数
        if prt:
            print(
                "violation_cnt",
                violation_cnt,
                "judgement_cnt",
                judgement_cnt,
                "violation_cnt/judgement_cnt",
                violation_cnt / judgement_cnt,
            )
            # print('Q_dict_cur/[veh/]',Q_dict_cur[veh])

        energy_record[x] = energy4frame
        HO_time_record[x] = HO_cnt4frame
        violation_prob_record[x] = violation_cnt / judgement_cnt
        avg_queuelen_record[x] = (
            sum([Q_dict_cur[veh][1:].sum() for veh in veh_set_cur])
            / judgement_cnt
        )
        HO_cmd_record[x] = HO_cmd_prev4cur
        if prt:
            print("\n\nframe: ", x, frame_cur)
            print("BS_association_num: ", BS_association_num)
            print("energy_record: ", energy_record[x])
            print("HO_time_record: ", HO_time_record[x])
            print("violation_prob_record: ", violation_prob_record[x])
            print("avg_queuelen_record: ", avg_queuelen_record[x])
            print("pilot_record: ", pilot_record[x])
            # print('HO_cmd_record: ',HO_cmd_record[x])

        # 记录当前帧各状态，为下一帧做准备
        frame_prev = frame_cur
        veh_set_prev = veh_set_cur
        Q_dict_prev = Q_dict_cur
        connection_dict_prev = connection_dict_cur
        CSI_dict_prev = CSI_dict_cur
        measured_g_record_dict_prev = measured_g_record_dict_cur
        HO_cmd_prev4cur = HO_cmd_cur4next
        # End

    return (
        energy_record,  # 存储各帧的系统能耗（单位：J）
        HO_time_record,  # 存储各帧的HO次数
        HO_cmd_record,  # 存储各帧的HO命令
        violation_prob_record,  # 存储各帧的UE业务积压队列长度超阈值的频率
        avg_queuelen_record,  # 存储各帧的UE业务积压队列长度均值
        pilot_record,
    )
