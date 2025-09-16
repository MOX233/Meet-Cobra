import pulp
import math
import copy
import collections
import numpy as np
from numpy import log2, log10
from scipy.optimize import linprog
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair
from utils.mox_utils import lin2dB

def init_vehset_connection(veh_set, BS_loc_array, vehstate_dict, macro=False):
    connection_dict = collections.OrderedDict()
    for veh in veh_set:
        if macro:
            # connection_dict[veh] = 0
            connection_dict[veh] = np.linalg.norm((BS_loc_array - vehstate_dict[veh]['pos']),ord=2,axis=-1).argmin() # 仍可能导致错误，因为距离最近的bs也有可能被阻挡
        else:
            connection_dict[veh] = vehstate_dict[veh]['g_opt_beam'].argmax() # 但是不现实！需要重新思考如何做connection init
        
    return connection_dict


def update_BS_association_state(BS_dict, connection_dict):
    BS_association_dict = collections.OrderedDict()
    BS_association_num = collections.OrderedDict()
    for k, _ in BS_dict.items():
        BS_association_dict[k] = []
        BS_association_num[k] = 0
    for k, v in connection_dict.items():
        BS_association_dict[v].append(k)
        BS_association_num[v] += 1
    return BS_association_dict, BS_association_num


def update_vehset_connection(veh_set_remain, veh_set_in, connection_dict_prev, HO_cmd, BS_loc_array, vehstate_dict, macro=False):
    HO_cnt = 0
    connection_dict = collections.OrderedDict()
    for veh_id in veh_set_remain:
        connection_dict[veh_id] = connection_dict_prev[veh_id]
    # for veh_id in veh_set_in:
    #     connection_dict[veh_id] = 0
    connection_dict.update(init_vehset_connection(veh_set_in, BS_loc_array, vehstate_dict, macro=macro))
    # HO_cmd 包含需要变更连接关系的veh_id:BS_id
    for veh_id, BS_id in HO_cmd.items():
        if (veh_id in connection_dict.keys()) and (BS_id != connection_dict[veh_id]):
            connection_dict[veh_id] = BS_id
            HO_cnt += 1
    return connection_dict, HO_cnt

def update_measured_g_record_dict(g_dict, measured_g_record_dict_prev, veh_set_cur, connection_dict_cur, BS_loc_list):
    """
    更新车辆历史接入基站的信道增益记录
    g_dict: 当前帧测量的信道增益字典
    measured_g_record_dict_prev: 前一帧的车辆历史接入基站的信道增益记录
    veh_set_cur: 当前帧的车辆集合
    connection_dict_cur: 当前帧的车辆连接基站字典
    BS_loc_list: 基站位置列表
    """
    measured_g_record_dict_cur = collections.OrderedDict()
    for veh in veh_set_cur:
        if veh not in measured_g_record_dict_prev.keys():
            measured_g_record_dict_cur[veh] = np.zeros((len(BS_loc_list),2))
            measured_g_record_dict_cur[veh][:,0] = -1 # 车辆与各基站上一次连接的经过时间(帧数)
            measured_g_record_dict_cur[veh][:,1] = -180 # 车辆与各基站上一次连接时的信道增益(dB)
        else:
            measured_g_record_dict_cur[veh] = copy.copy(measured_g_record_dict_prev[veh])
        
        for BS_id in range(len(BS_loc_list)):
            if BS_id == connection_dict_cur[veh]: # 如果当前基站是车辆的连接基站
                # 更新当前帧测量的信道增益
                measured_g_record_dict_cur[veh][BS_id,0] = 0  # 车辆与当前连接基站的经过时间(帧数)置为0
                measured_g_record_dict_cur[veh][BS_id,1] = g_dict[veh][BS_id]  # 车辆与当前连接基站的信道增益(dB)置为当前值
            else:
                if measured_g_record_dict_cur[veh][BS_id,0] >= 0:
                    measured_g_record_dict_cur[veh][BS_id,0] += 1 # 车辆与非连接基站的经过时间(帧数)加1
    return measured_g_record_dict_cur

def measure_gain_for_topKbeam(args, frame, veh_set, timeline_dir, BS_loc_list, pred_beamPairId_dict, DFT_matrix_tx, DFT_matrix_rx):
    num_pilot = 0
    g_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming的信道增益
    bpID_dict = collections.OrderedDict() # 统计各车与各基站的波束对ID
    for v, veh in enumerate(veh_set):
        g_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        bpID_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        veh_h = timeline_dir[frame][veh]['h'] # (M_r, N_bs, M_t)
        best_beam_index_pair = beamPairId_to_beamIdPair(pred_beamPairId_dict[veh], M_t=args.M_t, M_r=args.M_r) # (N_bs,args.K,2)
        for BS_id in range(len(BS_loc_list)):
            g_bf = np.zeros((args.K))
            for k in range(args.K):
                g_bf[k] = 1/np.sqrt(args.M_r*args.M_t) * \
                    np.abs(np.matmul(np.matmul(veh_h[:,BS_id,:], DFT_matrix_tx[:,best_beam_index_pair[BS_id,k,0]]).T.conjugate(),DFT_matrix_rx[:,best_beam_index_pair[BS_id,k,1]]))
                g_bf[k] = 2 * lin2dB(g_bf[k])
                num_pilot += 1
            g_dict[veh][BS_id] = g_bf.max()
            bpID_dict[veh][BS_id] = pred_beamPairId_dict[veh][BS_id, g_bf.argmax()]
    return g_dict, bpID_dict, num_pilot

def measure_gain_for_topKbeam_savePilot(args, frame, veh_set, timeline_dir, BS_loc_list, pred_beamPairId_dict, pred_gain_opt_beam_dict, DFT_matrix_tx, DFT_matrix_rx, db_err_th=5, db_lb=-100):
    # bug!
    num_pilot = 0
    g_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming的信道增益
    bpID_dict = collections.OrderedDict() # 统计各车与各基站的波束对ID
    for v, veh in enumerate(veh_set):
        g_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        bpID_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        veh_h = timeline_dir[frame][veh]['h'] # (M_r, N_bs, M_t)
        best_beam_index_pair = beamPairId_to_beamIdPair(pred_beamPairId_dict[veh], M_t=args.M_t, M_r=args.M_r) # (N_bs,args.K,2)
        for BS_id in range(len(BS_loc_list)):
            g_bf = 2 * lin2dB(np.zeros((args.K)))
            for k in range(args.K):
                g_bf[k] = 1/np.sqrt(args.M_r*args.M_t) * \
                    np.abs(np.matmul(np.matmul(veh_h[:,BS_id,:], DFT_matrix_tx[:,best_beam_index_pair[BS_id,k,0]]).T.conjugate(),DFT_matrix_rx[:,best_beam_index_pair[BS_id,k,1]]))
                g_bf[k] = 2 * lin2dB(g_bf[k])
                num_pilot += 1
                if (g_bf[k] > pred_gain_opt_beam_dict[veh][BS_id] - db_err_th) and (g_bf[k] > db_lb):
                    break
            g_dict[veh][BS_id] = g_bf.max()
            bpID_dict[veh][BS_id] = pred_beamPairId_dict[veh][BS_id, g_bf.argmax()]
    return g_dict, bpID_dict, num_pilot

def measure_gain_NoBeamforming(frame, veh_set, timeline_dir, BS_loc_list):
    num_pilot = 0
    g_dict = collections.OrderedDict() # 统计各车与各基站的信道增益
    bpID_dict = collections.OrderedDict() # 统计各车与各基站的波束对ID
    for v, veh in enumerate(veh_set):
        g_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        bpID_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        veh_h = timeline_dir[frame][veh]['h'] # (M_r, N_bs, M_t)
        for BS_id in range(len(BS_loc_list)):
            g_dict[veh][BS_id] = 2 * lin2dB(np.abs(veh_h[:,BS_id,:]).max()) 
    return g_dict, bpID_dict, num_pilot

                    
### 250331 ###
def RA_heur_fqb_smartRound(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict):
    RA_dict = collections.OrderedDict()
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])

    b = np.array(
        [delta_f * delta_t * log2(1 + p*10**((G-NF_dB)/10) / N0 / delta_f) for G in g]
    )  # 一个RB能提供的传输量

    alpha = 100
    f_q = np.array([alpha ** (q_dict[veh_id][slot_idx] / Q_ub_dict[veh_id]) - 1 for veh_id in veh_id_list])
    priority = (-f_q * b).argsort()
    resRB = num_RB
    for v in priority:
        if q[v] >= 0.5 * Q_ub_dict[veh_id_list[v]]:
            RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        else:
            RB_alloc = min(int(q[v] / b[v]), resRB)
        # if RB_alloc > 30:
        #     import ipdb;ipdb.set_trace()
        # RB_alloc = min(RB_alloc,10) # TODO
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_PF(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict):
    # Proportional Fair Scheduling
    RA_dict = collections.OrderedDict()
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])

    b = np.array(
        [delta_f * delta_t * log2(1 + p*10**((G-NF_dB)/10) / N0 / delta_f) for G in g]
    )  # 一个RB能提供的传输量

    q_over_Qub = np.array([q_dict[veh_id][slot_idx] / Q_ub_dict[veh_id] for veh_id in veh_id_list])
    priority = (-q_over_Qub * b).argsort()
    resRB = num_RB
    for v in priority:
        if q[v] >= 0.5 * Q_ub_dict[veh_id_list[v]]:
            RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        else:
            RB_alloc = min(int(q[v] / b[v]), resRB)
        # if RB_alloc > 30:
        #     import ipdb;ipdb.set_trace()
        # RB_alloc = min(RB_alloc,10) # TODO
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_QPOS(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict):
    # QoS-Prioritized Opportunistic Scheduling
    RA_dict = collections.OrderedDict()
    if not veh_set:  # 如果没有车辆，则返回空字典
        return RA_dict
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])
    b = np.array(
        [delta_f * delta_t * log2(1 + p*10**((G-NF_dB)/10) / N0 / delta_f) for G in g]
    )  # 一个RB能提供的传输量
    backlog_flag = np.array([q_dict[veh_id][slot_idx] > Q_ub_dict[veh_id]/2 for veh_id in veh_id_list])
    max_b = b.max()
    weight = b/max_b * backlog_flag - (1-b/max_b) * (1-backlog_flag)
    priority = (-weight).argsort()
    resRB = num_RB
    for v in priority:
        if backlog_flag[v]:
            RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        else:
            RB_alloc = min(int(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict

def RA_heur_QPPF(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict):
    # QoS-Prioritized Opportunistic Scheduling
    RA_dict = collections.OrderedDict()
    if not veh_set:  # 如果没有车辆，则返回空字典
        return RA_dict
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])
    b = np.array(
        [delta_f * delta_t * log2(1 + p*10**((G-NF_dB)/10) / N0 / delta_f) for G in g]
    )  # 一个RB能提供的传输量
    backlog_flag = np.array([q_dict[veh_id][slot_idx] > Q_ub_dict[veh_id]/2 for veh_id in veh_id_list])
    max_b = b.max()
    q_over_Qub = np.array([min(q_dict[veh_id][slot_idx] / Q_ub_dict[veh_id],2) for veh_id in veh_id_list])
    weight = q_over_Qub*b/max_b * backlog_flag - (1-b/max_b) * (1-backlog_flag)
    priority = (-weight).argsort()
    
    resRB = num_RB
    for v in priority:
        if backlog_flag[v]:
            RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        else:
            RB_alloc = min(int(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_unlimitRB(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict):
    RA_dict = collections.OrderedDict()
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])
    b = np.array(
        [delta_f * delta_t * log2(1 + p*10**((G-NF_dB)/10) / N0 / delta_f) for G in g]
    )  # 一个RB能提供的传输量
    priority = (-q).argsort()
    for v in priority:
        # 不仅取消了RB上限，还取消了RB是整数的约束
        if q[v] >= Q_ub_dict[veh_id_list[v]] / 2:
            RB_alloc = (q[v] - Q_ub_dict[veh_id_list[v]] / 2) / b[v]
        else:
            RB_alloc = 0
        RA_dict[veh_id_list[v]] = RB_alloc
    # print("BS_id=", BS_id, "RA_dict=", RA_dict)
    return RA_dict


def HO_EE_predG(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array):
    # Energy Efficiency
    HO_cmd = collections.OrderedDict()
    for veh in veh_set_cur:
        pred_veh_loc = pred_loc_dict[veh]
        BS_loc_array - pred_veh_loc
        dist_array = np.linalg.norm(
            BS_loc_array - pred_veh_loc, ord=2, axis=-1, keepdims=False
        )
        EE_array = np.zeros_like(dist_array)
        for BS_id in range(len(BS_loc_array)):
            delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
            p = args.p_micro if BS_id > 0 else args.p_macro
            NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
            G = 10 ** ((pred_g_dict[veh][BS_id]-NF_dB) / 10)
            EE_array[BS_id] = delta_f / p * log2(1 + p * G / args.N0 / delta_f)
        HO_cmd[veh] = EE_array.argmax()
    return HO_cmd


def HO_RBE_predG(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    for veh in veh_set_cur:
        pred_veh_loc = pred_loc_dict[veh]
        BS_loc_array - pred_veh_loc
        dist_array = np.linalg.norm(
            BS_loc_array - pred_veh_loc, ord=2, axis=-1, keepdims=False
        )
        RBE_array = np.zeros_like(dist_array)
        for BS_id in range(len(BS_loc_array)):
            delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
            p = args.p_micro if BS_id > 0 else args.p_macro
            NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
            G = 10 ** ((pred_g_dict[veh][BS_id]-NF_dB) / 10)
            RBE_array[BS_id] = delta_f * log2(1 + p * G / args.N0 / delta_f)
        HO_cmd[veh] = RBE_array.argmax()
    return HO_cmd


def HO_EE_GAP_APX_with_offload_conservative(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    points = np.zeros((len(veh_set_cur), 2))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
    dist_matrix = np.linalg.norm(
        points[:, np.newaxis, :] - BS_loc_array[np.newaxis, :, :],
        ord=2,
        axis=-1,
        keepdims=False,
    )
    power_matrix = np.zeros_like(dist_matrix)
    k_tilde_matrix = np.zeros_like(dist_matrix)
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        G_array = 10 ** (np.array([(pred_g_dict[veh][BS_id]-NF_dB) for veh in veh_set_cur]) / 10)
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        k_tilde_matrix[:, BS_id] = lbd_array / (
            delta_f * np.log2(1 + p * G_array / N0 / delta_f)
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        RB_num_table[BS_id] = args.num_RB_micro
        pm_table[BS_id] = args.p_micro
    T_HO, feasible_flag = HO_GAP_APX_with_offload(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd


def HO_EE_GAP_APX_with_offload_conservative_predG(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    points = np.zeros((len(veh_set_cur), 2))
    pred_G = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
        pred_G[i, :] = pred_g_dict[veh]
    dist_matrix = np.linalg.norm(
        points[:, np.newaxis, :] - BS_loc_array[np.newaxis, :, :],
        ord=2,
        axis=-1,
        keepdims=False,
    ) # (n_car,n_bs)
    power_matrix = np.zeros_like(dist_matrix)
    k_tilde_matrix = np.zeros_like(dist_matrix)
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        G_array = pred_G[:, BS_id]
        G_array = 10 ** ((G_array-NF_dB) / 10)
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        k_tilde_matrix[:, BS_id] = lbd_array / (
            delta_f * np.log2(1 + p * G_array / N0 / delta_f)
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        RB_num_table[BS_id] = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
        pm_table[BS_id] = args.p_micro if BS_id > 0 else args.p_macro
    T_HO, feasible_flag = HO_GAP_APX_with_offload(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd


































#################### others ##########################

def HO_nearest(args, veh_set_cur, backlog_queue_dict, pred_loc_dict, pred_g_dict, BS_loc_array):
    HO_cmd = collections.OrderedDict()
    for veh in veh_set_cur:
        pred_veh_loc = pred_loc_dict[veh]
        BS_loc_array - pred_veh_loc
        dist_array = np.linalg.norm(
            BS_loc_array - pred_veh_loc, ord=2, axis=-1, keepdims=False
        )
        HO_cmd[veh] = dist_array.argmin()
    return HO_cmd


def HO_SE(args, veh_set_cur, backlog_queue_dict, pred_loc_dict, pred_g_dict, BS_loc_array):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    for veh in veh_set_cur:
        pred_veh_loc = pred_loc_dict[veh]
        BS_loc_array - pred_veh_loc
        dist_array = np.linalg.norm(
            BS_loc_array - pred_veh_loc, ord=2, axis=-1, keepdims=False
        )
        SE_array = np.zeros_like(dist_array)
        for BS_id in range(len(BS_loc_array)):
            if BS_id == 0:
                delta_f = args.RB_intervel_macro
                p = args.p_macro
                bias = args.bias_macro
                beta = args.beta_macro
            else:
                delta_f = args.RB_intervel_micro
                p = args.p_micro
                bias = args.bias_micro
                beta = args.beta_micro
            G = 10 ** ((bias - beta * 10 * log10(dist_array[BS_id])) / 10)
            SE_array[BS_id] = log2(1 + p * G / args.N0 / delta_f)
        HO_cmd[veh] = SE_array.argmax()
    return HO_cmd


def HO_EE_loadbalance(
    args,
    veh_set_cur,
    backlog_queue_dict,
    pred_loc_dict,
    pred_g_dict,
    BS_loc_array,
    UEnum_factor=0.9,
):
    # Energy Efficiency
    HO_cmd = collections.OrderedDict()
    BS_UEnum = np.zeros(len(BS_loc_array))
    for veh in veh_set_cur:
        pred_veh_loc = pred_loc_dict[veh]
        BS_loc_array - pred_veh_loc
        dist_array = np.linalg.norm(
            BS_loc_array - pred_veh_loc, ord=2, axis=-1, keepdims=False
        )
        EE_array = np.zeros_like(dist_array)
        for BS_id in range(len(BS_loc_array)):
            if BS_id == 0:
                delta_f = args.RB_intervel_macro
                p = args.p_macro
                bias = args.bias_macro
                beta = args.beta_macro
            else:
                delta_f = args.RB_intervel_micro
                p = args.p_micro
                bias = args.bias_micro
                beta = args.beta_micro
            G = 10 ** ((bias - beta * 10 * log10(dist_array[BS_id])) / 10)
            EE_array[BS_id] = delta_f / p * log2(1 + p * G / args.N0 / delta_f)
        EE_array = EE_array * UEnum_factor**BS_UEnum
        HO_cmd[veh] = EE_array.argmax()
        BS_UEnum[EE_array.argmax()] += 1
    return HO_cmd


def HO_SE_loadbalance(
    args,
    veh_set_cur,
    backlog_queue_dict,
    pred_loc_dict,
    pred_g_dict,
    BS_loc_array,
    UEnum_factor=0.9,
):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    BS_UEnum = np.zeros(len(BS_loc_array))
    for veh in veh_set_cur:
        pred_veh_loc = pred_loc_dict[veh]
        BS_loc_array - pred_veh_loc
        dist_array = np.linalg.norm(
            BS_loc_array - pred_veh_loc, ord=2, axis=-1, keepdims=False
        )
        SE_array = np.zeros_like(dist_array)
        for BS_id in range(len(BS_loc_array)):
            if BS_id == 0:
                delta_f = args.RB_intervel_macro
                p = args.p_macro
                bias = args.bias_macro
                beta = args.beta_macro
            else:
                delta_f = args.RB_intervel_micro
                p = args.p_micro
                bias = args.bias_micro
                beta = args.beta_micro
            G = 10 ** ((bias - beta * 10 * log10(dist_array[BS_id])) / 10)
            SE_array[BS_id] = log2(1 + p * G / args.N0 / delta_f)
        SE_array = SE_array * UEnum_factor**BS_UEnum
        HO_cmd[veh] = SE_array.argmax()
        BS_UEnum[SE_array.argmax()] += 1
    return HO_cmd


def HO_RBE_loadbalance(
    args,
    veh_set_cur,
    backlog_queue_dict,
    pred_loc_dict,
    pred_g_dict,
    BS_loc_array,
    UEnum_factor=0.9,
):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    BS_UEnum = np.zeros(len(BS_loc_array))
    for veh in veh_set_cur:
        pred_veh_loc = pred_loc_dict[veh]
        BS_loc_array - pred_veh_loc
        dist_array = np.linalg.norm(
            BS_loc_array - pred_veh_loc, ord=2, axis=-1, keepdims=False
        )
        RBE_array = np.zeros_like(dist_array)
        for BS_id in range(len(BS_loc_array)):
            if BS_id == 0:
                delta_f = args.RB_intervel_macro
                p = args.p_macro
                bias = args.bias_macro
                beta = args.beta_macro
            else:
                delta_f = args.RB_intervel_micro
                p = args.p_micro
                bias = args.bias_micro
                beta = args.beta_micro
            G = 10 ** ((bias - beta * 10 * log10(dist_array[BS_id])) / 10)
            RBE_array[BS_id] = delta_f * log2(1 + p * G / args.N0 / delta_f)
        RBE_array = RBE_array * UEnum_factor**BS_UEnum
        HO_cmd[veh] = RBE_array.argmax()
        BS_UEnum[RBE_array.argmax()] += 1
    return HO_cmd


def HO_offload(T_KR: np.ndarray, T_TR: np.ndarray, T_PM: np.ndarray):
    # T_KR.shape == (num_BS, num_UE)
    num_BS, num_UE = T_KR.shape
    T_HO = np.zeros((num_BS, num_UE))  # HO decision table
    T_LR = np.zeros((num_BS))  # left RB table
    T_PK = T_KR * T_PM[:, np.newaxis]
    # init HO decision table
    UE2BS = T_PK.argmin(axis=0)
    for ue, bs in enumerate(UE2BS):
        T_HO[bs, ue] = 1
    T_LR = T_TR - (T_KR * T_HO).sum(1)

    feasible_flag = T_LR.min() >= 0

    max_iter_num = 1000
    for _iter in range(max_iter_num):
        if feasible_flag:  # 已经得到可行解
            break
        T_BO = T_LR < 0  # 超重BS表 T_BO.shape == (num_BS,)
        T_UO = T_HO[T_BO].sum(axis=0) == 1  # 超重UE表 T_UO.shape == (num_UE,)
        UE_OL_list = np.where(T_UO)[0]
        priority_list = np.zeros_like(UE_OL_list, dtype=float)
        reHO_list = -np.ones_like(UE_OL_list)
        for i, ue in enumerate(UE_OL_list):
            # 找可达不超重包
            avlb_BS_NOL = np.where((T_LR - T_KR[:, ue]) >= 0)[0]
            if len(avlb_BS_NOL) == 0:
                priority_list[i] = 0
                reHO_list[i] = -1
            else:
                fenzi = (T_KR * T_HO).sum(0)[ue]  # 在当前包里的重量(RB数量)
                fenmu = T_PK[avlb_BS_NOL, ue].min()  # 在可达包集合里的最小重量(能耗)
                priority_list[i] = fenzi / fenmu
                reHO_list[i] = avlb_BS_NOL[T_PK[avlb_BS_NOL, ue].argmin()]
        if (reHO_list != -1).sum() == 0:  # 此时已经无法单步转移任何UE
            break
        else:
            ue_select = UE_OL_list[priority_list.argmax()]
            reHO_select = reHO_list[priority_list.argmax()]
            T_HO[:, ue_select] = 0
            T_HO[reHO_select, ue_select] = 1
            T_LR = T_TR - (T_KR * T_HO).sum(1)
        feasible_flag = T_LR.min() >= 0
    return T_HO, feasible_flag


def HO_MOX2(T_KR: np.ndarray, T_TR: np.ndarray, T_PM: np.ndarray, alpha=2):
    def load_func(A, x, alpha=2):
        M, N = A.shape
        g = (alpha ** ((A * x).sum(axis=-1) - 1)).sum() / M
        return g

    def best_bOg4UE(UE, A, B, X, alpha=2):
        assert X[:, UE].sum() == 0
        g_increment_UE_list = []
        b_increment_UE_list = []
        bOVERg_UE_list = []
        for m in range(len(A)):
            dX = np.zeros_like(X)
            dX[m, UE] = 1
            g_increment_UE_list.append(
                load_func(A, X + dX, alpha) - load_func(A, X, alpha)
            )
            b_increment_UE_list.append((B * dX).reshape(-1).sum())
            bOVERg_UE_list.append(b_increment_UE_list[-1] / g_increment_UE_list[-1])
        bOVERg_UE_array = np.array(bOVERg_UE_list)
        return bOVERg_UE_array.max(), bOVERg_UE_array.argmax()

    num_BS, num_UE = T_KR.shape
    T_PK = T_KR * T_PM[:, np.newaxis]  # record p_m*k （能耗）
    c = T_PK.reshape(-1).max()
    B = 1 - T_PK / c  # profit matrix
    T_HO = np.zeros((num_BS, num_UE))  # HO decision table
    A = T_KR / T_TR[:, np.newaxis]
    for _i in range(num_UE):
        UE_unHO_list = np.sort(
            np.where(T_HO.sum(axis=0) == 0)[0]
        )  # the UE indices which has not been connected
        bOg_list = np.zeros_like(UE_unHO_list, dtype=float)
        UE2BS_list = np.zeros_like(UE_unHO_list)
        for i, ue in enumerate(UE_unHO_list):
            bOg_list[i], UE2BS_list[i] = best_bOg4UE(ue, A, B, T_HO, alpha=alpha)
        UE_selected = UE_unHO_list[bOg_list.argmax()]
        BS_selected = UE2BS_list[bOg_list.argmax()]
        T_HO[BS_selected, UE_selected] = 1
    return T_HO


def HO_EE_offload(
    args,
    veh_set_cur,
    backlog_queue_dict,
    pred_loc_dict,
    pred_g_dict,
    BS_loc_array,
):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    points = np.zeros((len(veh_set_cur), 2))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
    dist_matrix = np.linalg.norm(
        points[:, np.newaxis, :] - BS_loc_array[np.newaxis, :, :],
        ord=2,
        axis=-1,
        keepdims=False,
    )
    power_matrix = np.zeros_like(dist_matrix)
    k_tilde_matrix = np.zeros_like(dist_matrix)
    lbd = args.data_rate
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            delta_f = args.RB_intervel_macro
            p = args.p_macro
            bias = args.bias_macro
            beta = args.beta_macro
            K = args.num_RB_macro
        else:
            delta_f = args.RB_intervel_micro
            p = args.p_micro
            bias = args.bias_micro
            beta = args.beta_micro
            K = args.num_RB_micro
        G_array = 10 ** ((bias - beta * 10 * np.log10(dist_matrix[:, BS_id])) / 10)
        k_tilde_matrix[:, BS_id] = lbd / (
            delta_f * np.log2(1 + p * G_array / N0 / delta_f)
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            RB_num_table[BS_id] = args.num_RB_macro
            pm_table[BS_id] = args.p_macro
        else:
            RB_num_table[BS_id] = args.num_RB_micro
            pm_table[BS_id] = args.p_micro
    T_HO, feasible_flag = HO_offload(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd


def HO_EE_MOX2(
    args,
    veh_set_cur,
    backlog_queue_dict,
    pred_loc_dict,
    pred_g_dict,
    BS_loc_array,
    alpha=100,
):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    points = np.zeros((len(veh_set_cur), 2))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
    dist_matrix = np.linalg.norm(
        points[:, np.newaxis, :] - BS_loc_array[np.newaxis, :, :],
        ord=2,
        axis=-1,
        keepdims=False,
    )
    power_matrix = np.zeros_like(dist_matrix)
    k_tilde_matrix = np.zeros_like(dist_matrix)
    lbd = args.data_rate
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            delta_f = args.RB_intervel_macro
            p = args.p_macro
            bias = args.bias_macro
            beta = args.beta_macro
            K = args.num_RB_macro
        else:
            delta_f = args.RB_intervel_micro
            p = args.p_micro
            bias = args.bias_micro
            beta = args.beta_micro
            K = args.num_RB_micro
        G_array = 10 ** ((bias - beta * 10 * np.log10(dist_matrix[:, BS_id])) / 10)
        k_tilde_matrix[:, BS_id] = lbd / (
            delta_f * np.log2(1 + p * G_array / N0 / delta_f)
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            RB_num_table[BS_id] = args.num_RB_macro
            pm_table[BS_id] = args.p_macro
        else:
            RB_num_table[BS_id] = args.num_RB_micro
            pm_table[BS_id] = args.p_micro
    T_HO = HO_MOX2(
        T_KR=k_tilde_matrix.swapaxes(0, 1),
        T_TR=RB_num_table,
        T_PM=pm_table,
        alpha=alpha,
    )
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd


def RA_Lyapunov(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict):
    eta = args.eta
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    delta_t = args.slot_len
    N0 = args.N0
    veh_num = len(veh_set)
    veh_id_list = list(veh_set)
    q = [q_dict[veh_id][slot_idx] for veh_id in veh_id_list]
    a = [a_dict[veh_id][slot_idx] for veh_id in veh_id_list]
    g = [g_dict[veh_id][BS_id] for veh_id in veh_id_list]

    # 定义问题为最小化问题
    prob = pulp.LpProblem("Minimize_Z", pulp.LpMinimize)
    k = pulp.LpVariable.dict(
        "k", range(veh_num), lowBound=1, upBound=num_RB, cat=pulp.LpInteger
    )
    r = []
    for i in range(veh_num):
        r.append(
            k[i] * delta_f * delta_t * log2(1 + p * 10 ** (g[i] / 10) / N0 / delta_f)
        )
    target = pulp.lpSum([q[i] * (a[i] - r[i]) + eta * p * k[i] for i in range(veh_num)])
    prob += target, "Z"
    prob += pulp.lpSum(k) <= num_RB, "Constraint_RBnum"

    # 求解问题
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 输出结果
    RA_dict = collections.OrderedDict()
    for i, veh_id in enumerate(veh_id_list):
        RA_dict[veh_id] = k[i].value()
    # print(RA_dict)
    return RA_dict


def RA_heur_q(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict):
    RA_dict = collections.OrderedDict()
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])
    b = np.array(
        [delta_f * delta_t * log2(1 + p * 10 ** (G / 10) / N0 / delta_f) for G in g]
    )
    priority = (-q).argsort()
    resRB = num_RB
    for v in priority:
        RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    # print("BS_id=", BS_id, "RA_dict=", RA_dict)
    return RA_dict


def RA_heur_b(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict):
    RA_dict = collections.OrderedDict()
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])
    b = np.array(
        [delta_f * delta_t * log2(1 + p * 10 ** (G / 10) / N0 / delta_f) for G in g]
    )

    priority = (-b).argsort()
    resRB = num_RB
    for v in priority:
        RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_qb(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict):
    RA_dict = collections.OrderedDict()
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])

    b = np.array(
        [delta_f * delta_t * log2(1 + p * 10 ** (G / 10) / N0 / delta_f) for G in g]
    )

    priority = (-q * b).argsort()
    resRB = num_RB
    for v in priority:
        RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_fqb(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict):
    def f(x):
        alpha = 100
        Q_ub = args.lat_slot_ub * args.data_rate * args.slot_len  # 队列长度上限阈值
        y = alpha ** (x / Q_ub) - 1
        return y

    RA_dict = collections.OrderedDict()
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])

    b = np.array(
        [delta_f * delta_t * log2(1 + p * 10 ** (G / 10) / N0 / delta_f) for G in g]
    )  # 一个RB能提供的传输量

    priority = (-f(q) * b).argsort()
    resRB = num_RB
    for v in priority:
        RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict





# 2024.10.6
def solve_gap_lp(c, a, b):
    """
    使用线性规划松弛求解广义指派问题（GAP）的松弛问题。

    参数:
    c: 成本矩阵 (m x n)
    a: 工作量矩阵 (m x n)
    b: 工人容量 (m,)

    返回:
    res: 线性规划求解结果，包括最优解和目标值
    """
    m, n = c.shape

    # 构建目标函数
    c_flat = c.flatten()  # 将 m x n 的成本矩阵展开为向量

    # 构建不等式约束 Ax <= b
    A = np.zeros((m, m * n))
    for i in range(m):
        A[i, i * n : (i + 1) * n] = a[i]
    b_ineq = b

    # 构建等式约束 Ax = 1（每个任务只能分配给一个工人）
    A_eq = np.zeros((n, m * n))
    for j in range(n):
        A_eq[j, j::n] = 1
    b_eq = np.ones(n)

    # 定义变量 x_ij 的上下界 0 <= x_ij <= 1
    bounds = [(0, 1) for _ in range(m * n)]

    # 使用 scipy.optimize.linprog 求解线性规划问题
    res = linprog(
        c_flat, A_ub=A, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if res.success == False:
        return None

    # 将结果转换回矩阵形式
    x = res.x.reshape((m, n))

    return x, res.fun  # 返回松弛解和目标函数值


def build_graphB(x, a):
    # x is the solution of LP problem.
    # a is the burden matrix. It satisfies ax<=1 in our problem
    assert x.shape == a.shape
    m, n = x.shape
    k = np.zeros(m).astype(np.int32)
    for i in range(m):
        k[i] = math.ceil(x[i].sum())
    x_B = np.zeros((k.sum(), n))
    for i in range(m):
        B_v_start = k[:i].sum()
        j_sorted = np.argsort(a[i] * (x[i] > 0))[::-1][: (x[i] > 0).sum()]
        s = 0
        for j in j_sorted:
            if x[i, j] <= 1 - x_B[B_v_start + s, :].sum() + 1e-15:
                x_B[B_v_start + s, j] = x[i, j]
            else:
                x_B[B_v_start + s, j] = 1 - x_B[B_v_start + s, :].sum()
                x_B[B_v_start + s + 1, j] = x[i, j] - x_B[B_v_start + s, j]
                s = s + 1
        if not (s < k[i]) | (k[i] == 0):
            import ipdb

            ipdb.set_trace()
        assert (s < k[i]) | (k[i] == 0)
    return x_B, k


def km_algorithm(cost_matrix):
    # 获取矩阵的行数和列数
    num_rows, num_cols = cost_matrix.shape

    # 如果左右顶点数不相等，扩展矩阵为方阵
    if num_rows != num_cols:
        n = max(num_rows, num_cols)
        # new_cost_matrix = np.zeros((n, n))
        new_cost_matrix = -np.ones((n, n))
        new_cost_matrix[:num_rows, :num_cols] = cost_matrix
        cost_matrix = new_cost_matrix
    else:
        n = num_rows

    # 行顶标和列顶标初始化
    lx = np.max(cost_matrix, axis=1)  # 每行的顶标为行中最大值
    ly = np.zeros(n)  # 列顶标初始为 0

    # 记录右侧节点的匹配情况
    match_y = -np.ones(n, dtype=int)  # 记录右侧节点匹配的左侧节点，-1表示未匹配

    def find_augmenting_path(u, visited_x, visited_y, slack, slack_x):
        visited_x[u] = True
        for v in range(n):
            if visited_y[v]:
                continue
            gap = lx[u] + ly[v] - cost_matrix[u, v]
            # if gap == 0: # bug: 计算机精度误差问题
            if gap <= 1e-10:
                visited_y[v] = True
                if match_y[v] == -1 or find_augmenting_path(
                    match_y[v], visited_x, visited_y, slack, slack_x
                ):
                    match_y[v] = u
                    return True
            else:
                if slack[v] > gap:
                    slack[v] = gap
                    slack_x[v] = u
        return False

    for u in range(n):
        # print(u,'/',n)
        # slack = np.full(n, np.inf)
        slack = np.full(n, 1e9)
        slack_x = np.zeros(n, dtype=int)

        while True:
            visited_x = np.zeros(n, dtype=bool)
            visited_y = np.zeros(n, dtype=bool)

            if find_augmenting_path(u, visited_x, visited_y, slack, slack_x):
                break

            # 更新顶标
            delta = np.min(slack[~visited_y])
            for i in range(n):
                if visited_x[i]:
                    lx[i] -= delta
                if visited_y[i]:
                    ly[i] += delta

    # 计算最大权值和匹配结果
    total_weight = 0
    matching = []
    for v in range(num_cols):  # 只考虑原始矩阵的列数
        if match_y[v] != -1:
            matching.append((match_y[v], v))
            total_weight += cost_matrix[match_y[v], v]

    return matching, total_weight


def alg_GAP_APX(c, a, b):
    m, n = a.shape
    # 求解松弛问题
    lp_result = solve_gap_lp(c, a, b)
    if lp_result is None:
        return None
    else:
        x, total_cost = lp_result
    # 构建 B(x)
    x_B, k_list = build_graphB(x, a)
    # 利用KM 算法进行最大匹配，其中需要对目标函数进行处理
    cost_matrix = np.zeros((k_list.sum(), n))
    s = 0
    for i, k in enumerate(k_list):
        cost_matrix[s : s + k, :] = (x_B[s : s + k, :] > 0) * (2 * c.max() - c[i])
        s += k
    matching, total_weight = km_algorithm(cost_matrix)
    # 得到最后的解
    i_mapping = dict()
    s = 0
    for i, k in enumerate(k_list):
        for s_i in range(k):
            i_mapping[s + s_i] = i
        s += k
    sche_matrix = np.zeros((m, n), dtype=np.int32)
    for m_i, m_j in matching:
        sche_matrix[i_mapping[m_i], m_j] = 1
    return sche_matrix


def alg_GAP_APX_adap(c, a, b, adap_mtp=1.1, debug=False):
    # b_adap = b / 2
    b_adap = b
    while 1:
        alg_GAP_APX_result = alg_GAP_APX(c, a, b_adap)
        if alg_GAP_APX_result is not None:
            break
        else:
            b_adap = b_adap * adap_mtp
    if debug:
        print(b_adap)
    sche_matrix = alg_GAP_APX_result
    return sche_matrix


def HO_GAP_APX(T_KR: np.ndarray, T_TR: np.ndarray, T_PM: np.ndarray):
    # T_KR.shape == (num_BS, num_UE)
    num_BS, num_UE = T_KR.shape
    T_LR = np.zeros((num_BS))  # left RB table
    T_PK = T_KR * T_PM[:, np.newaxis]
    T_HO = alg_GAP_APX_adap(c=T_PK, a=T_KR, b=T_TR, adap_mtp=1.1)
    T_LR = T_TR - (T_KR * T_HO).sum(1)
    feasible_flag = T_LR.min() >= 0
    return T_HO, feasible_flag


def HO_EE_GAP_APX(
    args,
    veh_set_cur,
    backlog_queue_dict,
    pred_loc_dict,
    pred_g_dict,
    BS_loc_array,
):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    points = np.zeros((len(veh_set_cur), 2))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
    dist_matrix = np.linalg.norm(
        points[:, np.newaxis, :] - BS_loc_array[np.newaxis, :, :],
        ord=2,
        axis=-1,
        keepdims=False,
    )
    power_matrix = np.zeros_like(dist_matrix)
    k_tilde_matrix = np.zeros_like(dist_matrix)
    lbd = args.data_rate
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            delta_f = args.RB_intervel_macro
            p = args.p_macro
            bias = args.bias_macro
            beta = args.beta_macro
            K = args.num_RB_macro
        else:
            delta_f = args.RB_intervel_micro
            p = args.p_micro
            bias = args.bias_micro
            beta = args.beta_micro
            K = args.num_RB_micro
        G_array = 10 ** ((bias - beta * 10 * np.log10(dist_matrix[:, BS_id])) / 10)
        k_tilde_matrix[:, BS_id] = lbd / (
            delta_f * np.log2(1 + p * G_array / N0 / delta_f)
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            RB_num_table[BS_id] = args.num_RB_macro
            pm_table[BS_id] = args.p_macro
        else:
            RB_num_table[BS_id] = args.num_RB_micro
            pm_table[BS_id] = args.p_micro
    T_HO, feasible_flag = HO_GAP_APX(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd


def HO_GAP_APX_with_offload(T_KR: np.ndarray, T_TR: np.ndarray, T_PM: np.ndarray):
    # T_KR.shape == (num_BS, num_UE)
    num_BS, num_UE = T_KR.shape
    T_LR = np.zeros((num_BS))  # left RB table
    T_PK = T_KR * T_PM[:, np.newaxis]
    # T_HO = alg_GAP_APX_adap(c=T_PK, a=T_KR, b=T_TR * 2, adap_mtp=1.1)
    T_HO, _ = HO_GAP_APX(T_KR, T_TR, T_PM)
    T_LR = T_TR - (T_KR * T_HO).sum(1)
    feasible_flag = T_LR.min() >= 0

    # print("HO_GAP_APX_with_MOX1, T_LR before iter: ", T_LR)
    max_iter_num = 1000
    for _iter in range(max_iter_num):
        if (_iter > max_iter_num - 10) and (T_LR[0] > 99) and (feasible_flag == False):
            import ipdb

            ipdb.set_trace()
        if feasible_flag:  # 已经得到可行解
            break
        T_BO = T_LR < 0  # 超重BS表 T_BO.shape == (num_BS,)
        T_UO = T_HO[T_BO].sum(axis=0) == 1  # 超重UE表 T_UO.shape == (num_UE,)
        UE_OL_list = np.where(T_UO)[0]
        priority_list = np.zeros_like(UE_OL_list, dtype=float)
        reHO_list = -np.ones_like(UE_OL_list)
        for i, ue in enumerate(UE_OL_list):
            # 找可达不超重包
            avlb_BS_NOL = np.where((T_LR - T_KR[:, ue]) >= 0)[0]
            if len(avlb_BS_NOL) == 0:
                priority_list[i] = 0
                reHO_list[i] = -1
            else:
                fenzi = (T_KR * T_HO).sum(0)[ue]  # 在当前包里的重量(RB数量)
                fenmu = T_PK[avlb_BS_NOL, ue].min()  # 在可达包集合里的最小重量(能耗)
                priority_list[i] = fenzi / fenmu
                reHO_list[i] = avlb_BS_NOL[T_PK[avlb_BS_NOL, ue].argmin()]
        if (reHO_list != -1).sum() == 0:  # 此时已经无法单步转移任何UE
            break
        else:
            ue_select = UE_OL_list[priority_list.argmax()]
            reHO_select = reHO_list[priority_list.argmax()]
            T_HO[:, ue_select] = 0
            T_HO[reHO_select, ue_select] = 1
            T_LR = T_TR - (T_KR * T_HO).sum(1)
        feasible_flag = T_LR.min() >= 0
    # print("HO_GAP_APX_with_offload, T_LR after iter: ", T_LR)
    return T_HO, feasible_flag


def HO_EE_GAP_APX_with_offload(
    args,
    veh_set_cur,
    backlog_queue_dict,
    pred_loc_dict,
    pred_g_dict,
    BS_loc_array,
):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    points = np.zeros((len(veh_set_cur), 2))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
    dist_matrix = np.linalg.norm(
        points[:, np.newaxis, :] - BS_loc_array[np.newaxis, :, :],
        ord=2,
        axis=-1,
        keepdims=False,
    )
    power_matrix = np.zeros_like(dist_matrix)
    k_tilde_matrix = np.zeros_like(dist_matrix)
    lbd = args.data_rate
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            delta_f = args.RB_intervel_macro
            p = args.p_macro
            bias = args.bias_macro
            beta = args.beta_macro
            K = args.num_RB_macro
        else:
            delta_f = args.RB_intervel_micro
            p = args.p_micro
            bias = args.bias_micro
            beta = args.beta_micro
            K = args.num_RB_micro
        G_array = 10 ** ((bias - beta * 10 * np.log10(dist_matrix[:, BS_id])) / 10)
        k_tilde_matrix[:, BS_id] = lbd / (
            delta_f * np.log2(1 + p * G_array / N0 / delta_f)
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        if BS_id == 0:
            RB_num_table[BS_id] = args.num_RB_macro
            pm_table[BS_id] = args.p_macro
        else:
            RB_num_table[BS_id] = args.num_RB_micro
            pm_table[BS_id] = args.p_micro
    T_HO, feasible_flag = HO_GAP_APX_with_offload(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd