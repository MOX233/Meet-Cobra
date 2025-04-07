from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import collections
import ipdb
import torch

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
    update_CSI_onlyiidshd,
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
    measure_gain_for_topKbeam,
    measure_gain_for_topKbeam_savePilot,
    measure_gain_NoBeamforming,
    HO_nearest,
    RA_Lyapunov,
)
from utils.mox_utils import lin2dB
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair, generate_dft_codebook

def run_sim(
    args,  # 存储仿真参数设置的args对象
    BS_loc_list,  # 存储各基站位置的list
    timeline_dir,  # 对SUMO生成的车流数据（详细版）进行处理后得到的交通车流信息
    pospred_model,  # 基于CSI预测车辆移动性的AI模型
    beampred_model,
    gainpred_model,
    RA_func=RA_Lyapunov,  # 资源分配算法
    HO_func=HO_nearest,  # 越区切换算法
    prt=True,  # 是否在仿真运行时实时打印相关信息
    save_pilot=False, # 是否执行pilot-saved的测量方法
    No_BF=False, # 是否不使用Beamforming
):
    device = args.device
    DFT_matrix_tx = generate_dft_codebook(args.M_t)
    DFT_matrix_rx = generate_dft_codebook(args.M_r)
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
    CSI_dict_cur = collections.OrderedDict()  # 前一帧的车辆CSI
    # CSI_dict[veh].shape =  (args.frames_per_sample, 2*M_r*N_pilot)
    HO_cmd_prev4cur = (
        collections.OrderedDict()
    )  # 前一帧为当前帧做出的HO决策 HO_cmd 包含需要变更连接关系的veh:BS_id
    HO_cmd_cur4next = (
        collections.OrderedDict()
    )  # 当前帧为下一帧做出的HO决策 HO_cmd 包含需要变更连接关系的veh:BS_id
    BS_association_dict, BS_association_num = update_BS_association_state(
        BS_loc_dict, connection_dict_prev
    )  # 各基站关联用户数量
    Q_ub = args.lat_slot_ub * args.data_rate * args.slot_len  # 队列长度上限阈值

    # 仿真输出结果记录
    energy_record = np.zeros((num_frame,))  # 记录每帧能耗
    HO_time_record = np.zeros((num_frame,))  # 记录每帧HO次数
    HO_cmd_record = collections.OrderedDict()  # 记录每帧HO命令
    violation_prob_record = np.zeros((num_frame,))  # 记录每帧的队列长度违规频率
    avg_queuelen_record = np.zeros((num_frame,))  # 记录每帧平均队列长度
    pilot_record = np.zeros((num_frame,))  # 记录每帧所用pilot数量

    # 初始化车辆业务数据积压队列
    Q_dict_prev = init_vehset_backlog_queue(
        veh_set_prev, Q_th=int(Q_ub / 2), slots_per_frame=args.slots_per_frame
    )

    # 初始化车辆-基站连接关系
    connection_dict_prev = init_vehset_connection(veh_set_prev, BS_loc_array, timeline_dir[frame_prev])

    # 初始化车辆-基站信道状态信息(CSI)
    for veh in veh_set_prev:
        CSI_dict_prev[veh] = timeline_dir[frame_prev][veh]["CSI_preprocessed"][np.newaxis,...].repeat(args.frames_per_sample, axis=0)
        
        
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
            Q_th=int(Q_ub / 2),
            slots_per_frame=args.slots_per_frame,
        )
        CSI_dict_cur = update_CSI_onlyiidshd(
            args,
            veh_set_remain,
            veh_set_in,
            CSI_dict_prev,
            trajectoryInfo=timeline_dir[frame_cur],
        )
                
        connection_dict_cur, HO_cnt4frame = update_vehset_connection(
            veh_set_remain, veh_set_in, connection_dict_prev, HO_cmd_prev4cur, BS_loc_array, timeline_dir[frame_cur]
        )
        BS_association_dict, BS_association_num = update_BS_association_state(
            BS_loc_dict, connection_dict_cur
        )  # 各基站关联用户数量

        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期的期望位置 (TODO)
        pred_loc_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if pospred_model is None:
                pred_loc_dict[veh] = timeline_dir[frame_cur][veh]["pos"]
            else:
                pred_loc_dict[veh] = pospred_model.predict(timeline_dir[frame_cur][veh]['CSI_preprocessed'],device)
                # TODO: eval CSI_dict_cur[-1,:] == timeline_dir[frame_cur][veh]['CSI_preprocessed']
                
        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期的最优波束增益     
        pred_gain_opt_beam_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if gainpred_model is None:
                pred_gain_opt_beam_dict[veh] = timeline_dir[frame_cur][veh]["g_opt_beam"]
            else:
                pred_gain_opt_beam_dict[veh] = gainpred_model.predict(timeline_dir[frame_cur][veh]['CSI_preprocessed'],device)
                # pred_gain_opt_beam_dict[veh].shape = (4,)
            # if timeline_dir[frame_cur][veh]["g_opt_beam"][pred_gain_opt_beam_dict[veh].argmax()] <= -180:
            #     print('bug!')
                # ipdb.set_trace()
        # 基于历史信道状态信息, 通过AI/ML算法预测各车辆在下一PHO周期的最优波束方向
        pred_beamPairId_dict = collections.OrderedDict()
        for veh in veh_set_cur:
            if beampred_model is None:
                pred_beamPairId_dict[veh] = timeline_dir[frame_cur][veh]["best_beam_pair_idx"].reshape(-1,1).repeat(args.K,axis=-1)
            else:
                pred_beamPairId_dict[veh] = beampred_model.predict(timeline_dir[frame_cur][veh]['CSI_preprocessed'],device, K=args.K)
                # pred_beamPairId_dict[veh].shape = (4,K)
        
        # 让各车对各基站的K个波束对进行测量
        if No_BF:
            g_dict, bpID_dict, num_pilot = measure_gain_NoBeamforming(frame_cur, veh_set_cur, timeline_dir, BS_loc_list)
        elif save_pilot:
            g_dict, bpID_dict, num_pilot = measure_gain_for_topKbeam_savePilot(args, frame_cur, veh_set_cur, timeline_dir, BS_loc_list, \
                                                                     pred_beamPairId_dict, pred_gain_opt_beam_dict, DFT_matrix_tx, DFT_matrix_rx)
        else:
            g_dict, bpID_dict, num_pilot = measure_gain_for_topKbeam(args, frame_cur, veh_set_cur, timeline_dir, BS_loc_list, \
                                                                         pred_beamPairId_dict, DFT_matrix_tx, DFT_matrix_rx)
        
        
        # if save_pilot:
        #     g_dict, bpID_dict, num_pilot = measure_gain_for_topKbeam_savePilot(args, frame_cur, veh_set_cur, timeline_dir, BS_loc_list, \
        #                                                              pred_beamPairId_dict, pred_gain_opt_beam_dict, DFT_matrix_tx, DFT_matrix_rx)
        # else:
        #     if No_BF:
        #         g_dict, bpID_dict, num_pilot = measure_gain_NoBeamforming(frame_cur, veh_set_cur, timeline_dir, BS_loc_list)
        #     else:
        #         g_dict, bpID_dict, num_pilot = measure_gain_for_topKbeam(args, frame_cur, veh_set_cur, timeline_dir, BS_loc_list, \
        #                                                                  pred_beamPairId_dict, DFT_matrix_tx, DFT_matrix_rx)
        
        
        # g_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming的信道增益 暂时只实现了K=1的情况
        # bpID_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming的信道增益 暂时只实现了K=1的情况
        # for v, veh in enumerate(veh_set_cur):
        #     g_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        #     bpID_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        #     veh_h = timeline_dir[frame_cur][veh]['h'] # (M_r, N_bs, M_t)
        #     best_beam_index_pair = beamPairId_to_beamIdPair(pred_beamPairId_dict[veh], M_t=args.M_t, M_r=args.M_r) # (N_bs,args.K,2)
        #     for BS_id in range(len(BS_loc_list)):
        #         # TODO: 暂时只实现了K=1的情况 -> 已实现待验证
        #         g_bf = np.zeros((args.K))
        #         for k in range(args.K):
        #             g_bf[k] = 1/np.sqrt(args.M_r*args.M_t) * \
        #                 np.abs(np.matmul(np.matmul(veh_h[:,BS_id,:], DFT_matrix_tx[:,best_beam_index_pair[BS_id,k,0]]).T.conjugate(),DFT_matrix_rx[:,best_beam_index_pair[BS_id,k,1]]))
        #             g_bf[k] = 2 * lin2dB(g_bf[k])
        #         g_dict[veh][BS_id] = g_bf.max()
        #         bpID_dict[veh][BS_id] = pred_beamPairId_dict[veh][BS_id, g_bf.argmax()]
        #         pilot_record[x] += 1
        pilot_record[x] += num_pilot 
            # veh7.68 与BS-4的信道可能被阻挡，但是仍然被model预测错为最优基站
        
        # 基于积压队列、预测位置、BS位置等信息进行PHO决策
        HO_cmd_cur4next = HO_func(
            args, veh_set_cur, Q_dict_cur, pred_loc_dict, g_dict, BS_loc_array
        )

        # 模拟车辆业务数据到达过程 in a frame
        data_arrival_process4frame = np.random.poisson(
            args.data_rate * args.slot_len,
            size=(len(veh_set_cur), args.slots_per_frame),
        )
        """# debug 去掉到达的随机性
        data_arrival_process4frame = (
            args.data_rate
            * args.slot_len
            * np.ones((len(veh_set_cur), args.slots_per_frame))
        )"""
        a_dict = collections.OrderedDict()
        for v, veh in enumerate(veh_set_cur):
            a_dict[veh] = data_arrival_process4frame[v]
        
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
                    q_dict=Q_dict_cur,
                    a_dict=a_dict,
                    g_dict=g_dict,
                )
                RA_dict.update(BS_RA_dict)
                energy4frame += sum(BS_RA_dict.values()) * args.p_micro * args.slot_len
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
            [(Q_dict_cur[veh][1:] > Q_ub).sum() for veh in veh_set_cur]
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
