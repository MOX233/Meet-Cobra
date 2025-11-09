import pulp
import math
import copy
import collections
import numpy as np
from numpy import log2, log10
from scipy.optimize import linprog
from utils.beam_utils import beamIdPair_to_beamPairId, beamPairId_to_beamIdPair
from utils.mox_utils import lin2dB, dB2lin

min_bf_gain_dB = 20

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
    num_pilot_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming所用的导频数量
    g_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming的信道增益
    g_NoBF_dict = collections.OrderedDict() # 统计各车与各基站间不进行beamforming的信道增益
    bpID_dict = collections.OrderedDict() # 统计各车与各基站的波束对ID
    for v, veh in enumerate(veh_set):
        num_pilot_dict[veh] = np.zeros((len(BS_loc_list)))
        g_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        g_NoBF_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        bpID_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        veh_h = timeline_dir[frame][veh]['h'] # (M_r, N_bs, M_t)
        best_beam_index_pair = beamPairId_to_beamIdPair(pred_beamPairId_dict[veh], M_t=args.M_t, M_r=args.M_r) # (N_bs,args.K,2)
        for BS_id in range(len(BS_loc_list)):
            g_bf = np.zeros((args.K))
            for k in range(args.K):
                g_bf[k] = 1/np.sqrt(args.M_r*args.M_t) * \
                    np.abs(np.matmul(np.matmul(veh_h[:,BS_id,:], DFT_matrix_tx[:,best_beam_index_pair[BS_id,k,0]]).T.conjugate(),DFT_matrix_rx[:,best_beam_index_pair[BS_id,k,1]]))
                g_bf[k] = 2 * lin2dB(g_bf[k])
                num_pilot_dict[veh][BS_id] += 1
            g_dict[veh][BS_id] = g_bf.max()
            g_NoBF_dict[veh][BS_id] = 2 * lin2dB(np.abs(veh_h[:,BS_id,:]).max())
            bpID_dict[veh][BS_id] = pred_beamPairId_dict[veh][BS_id, g_bf.argmax()]
    return g_dict, g_NoBF_dict, bpID_dict, num_pilot_dict

def measure_gain_for_topKbeam_savePilot(args, frame, veh_set, timeline_dir, BS_loc_list, pred_beamPairId_dict, pred_gain_opt_beam_dict, DFT_matrix_tx, DFT_matrix_rx, db_err_th=5, db_lb=-100):
    # bug!
    num_pilot_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming所用的导频数量
    g_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming的信道增益
    g_NoBF_dict = collections.OrderedDict() # 统计各车与各基站间不进行beamforming的信道增益
    bpID_dict = collections.OrderedDict() # 统计各车与各基站的波束对ID
    for v, veh in enumerate(veh_set):
        num_pilot_dict[veh] = np.zeros((len(BS_loc_list)))
        g_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        g_NoBF_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        bpID_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        veh_h = timeline_dir[frame][veh]['h'] # (M_r, N_bs, M_t)
        best_beam_index_pair = beamPairId_to_beamIdPair(pred_beamPairId_dict[veh], M_t=args.M_t, M_r=args.M_r) # (N_bs,args.K,2)
        for BS_id in range(len(BS_loc_list)):
            g_bf = 2 * lin2dB(np.zeros((args.K)))
            for k in range(args.K):
                g_bf[k] = 1/np.sqrt(args.M_r*args.M_t) * \
                    np.abs(np.matmul(np.matmul(veh_h[:,BS_id,:], DFT_matrix_tx[:,best_beam_index_pair[BS_id,k,0]]).T.conjugate(),DFT_matrix_rx[:,best_beam_index_pair[BS_id,k,1]]))
                g_bf[k] = 2 * lin2dB(g_bf[k])
                num_pilot_dict[veh][BS_id] += 1
                if (g_bf[k] > pred_gain_opt_beam_dict[veh][BS_id] - db_err_th) and (g_bf[k] > db_lb):
                    break
            g_dict[veh][BS_id] = g_bf.max()
            g_NoBF_dict[veh][BS_id] = 2 * lin2dB(np.abs(veh_h[:,BS_id,:]).max())
            bpID_dict[veh][BS_id] = pred_beamPairId_dict[veh][BS_id, g_bf.argmax()]
    return g_dict, g_NoBF_dict, bpID_dict, num_pilot_dict

def measure_gain_NoBeamforming(frame, veh_set, timeline_dir, BS_loc_list):
    num_pilot_dict = collections.OrderedDict() # 统计各车与各基站基于pred_beamPairId_dict进行beamforming所用的导频数量
    g_dict = collections.OrderedDict() # 统计各车与各基站的信道增益
    bpID_dict = collections.OrderedDict() # 统计各车与各基站的波束对ID
    for v, veh in enumerate(veh_set):
        num_pilot_dict[veh] = np.zeros((len(BS_loc_list)))
        g_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        bpID_dict[veh] = np.zeros((len(BS_loc_list))) # ()  size = (num_bs)
        veh_h = timeline_dir[frame][veh]['h'] # (M_r, N_bs, M_t)
        for BS_id in range(len(BS_loc_list)):
            num_pilot_dict[veh][BS_id] = 0
            g_dict[veh][BS_id] = 2 * lin2dB(np.abs(veh_h[:,BS_id,:]).max()) 
    g_NoBF_dict = g_dict
    return g_dict, g_NoBF_dict, bpID_dict, num_pilot_dict

def estimate_num_RB_allocated_perBS(args, connection_dict_cur, BS_loc_array, veh_set_cur, g_dict, veh_data_rate_dict, infer_g_dict=None):
    # 估计每个基站在当前帧平均分配出去的RB数量
    T_HO = np.zeros((len(BS_loc_array),len(veh_set_cur)))
    N0 = args.N0
    for i, veh in enumerate(veh_set_cur):
        T_HO[connection_dict_cur[veh], i] = 1
    k_tilde_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    G_dB = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        G_dB[i, :] = g_dict[veh]
    # 迭代至收敛
    _num_RB_allocated_perBS = args.num_RB_micro * np.ones(len(BS_loc_array))
    for _iter in range(10):
        for BS_id in range(len(BS_loc_array)):
            delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
            p = args.p_micro if BS_id > 0 else args.p_macro
            NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
            G_array = dB2lin(G_dB[:, BS_id])
            lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
            if infer_g_dict is not None:
                interference_array = np.array([
                    sum([
                        dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro * (_num_RB_allocated_perBS[other_BS_id]/args.num_RB_micro)
                        for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                    ])
                    for veh in veh_set_cur
                ])  
            else:
                interference_array = np.array([
                    sum([
                        dB2lin(g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro * (_num_RB_allocated_perBS[other_BS_id]/args.num_RB_micro)
                        for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                    ])
                    for veh in veh_set_cur
                ])
            if BS_id == 0:
                interference_array *= 0  # macro BS不考虑微基站的干扰
            k_tilde_matrix[:, BS_id] = lbd_array / (
                delta_f * np.log2(1 + p * G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array)) + 1e-15
            )

        num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
        # print('_num_RB_allocated_perBS',_num_RB_allocated_perBS,'est_num_RB_allocated_perBS:', num_RB_allocated_perBS)
        if np.allclose(_num_RB_allocated_perBS, num_RB_allocated_perBS, atol=1):
            # print('break in iter',_iter)
            break
        _num_RB_allocated_perBS = num_RB_allocated_perBS
    return num_RB_allocated_perBS

                    
### 250331 ###
def RA_heur_fqb_smartRound(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict, num_pilot_dict, **kwargs):
    RA_dict = collections.OrderedDict()
    est_num_RB_allocated_perBS = kwargs.get('est_num_RB_allocated_perBS', None)
    if est_num_RB_allocated_perBS is not None:
        # 逐元素取上界args.num_RB_micro和预测值的较小值
        est_num_RB_allocated_perBS = np.minimum(est_num_RB_allocated_perBS, args.num_RB_micro)
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
    if BS_id == 0:
        BF_pilot_overhead = np.zeros(len(veh_id_list))
    else:
        BF_pilot_overhead = np.array([min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1) for veh_id in veh_id_list])
    b = np.array(
        [(1-BPO) * delta_f * delta_t * log2(1 + p* dB2lin(G-NF_dB) / N0 / delta_f) for G,BPO in zip(g,BF_pilot_overhead)]
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
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_PF(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict, num_pilot_dict, **kwargs):
    # Proportional Fair Scheduling
    RA_dict = collections.OrderedDict()
    est_num_RB_allocated_perBS = kwargs.get('est_num_RB_allocated_perBS', None)
    if est_num_RB_allocated_perBS is not None:
        # 逐元素取上界args.num_RB_micro和预测值的较小值
        est_num_RB_allocated_perBS = np.minimum(est_num_RB_allocated_perBS, args.num_RB_micro)
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
    if BS_id == 0:
        BF_pilot_overhead = np.zeros(len(veh_id_list))
    else:
        BF_pilot_overhead = np.array([min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1) for veh_id in veh_id_list])
    b = np.array(
        [(1-BPO) * delta_f * delta_t * log2(1 + p* dB2lin(G-NF_dB) / N0 / delta_f) for G,BPO in zip(g,BF_pilot_overhead)]
    )  # 一个RB能提供的传输量

    q_over_Qub = np.array([q_dict[veh_id][slot_idx] / Q_ub_dict[veh_id] for veh_id in veh_id_list])
    priority = (-q_over_Qub * b).argsort()
    resRB = num_RB
    for v in priority:
        if q[v] >= 0.5 * Q_ub_dict[veh_id_list[v]]:
            RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        else:
            RB_alloc = min(int(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_QPOS(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict, num_pilot_dict, **kwargs):
    # QoS-Prioritized Opportunistic Scheduling
    RA_dict = collections.OrderedDict()
    est_num_RB_allocated_perBS = kwargs.get('est_num_RB_allocated_perBS', None)
    if est_num_RB_allocated_perBS is not None:
        # 逐元素取上界args.num_RB_micro和预测值的较小值
        est_num_RB_allocated_perBS = np.minimum(est_num_RB_allocated_perBS, args.num_RB_micro)
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
    if BS_id == 0:
        BF_pilot_overhead = np.zeros(len(veh_id_list))
    else:
        BF_pilot_overhead = np.array([min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1) for veh_id in veh_id_list])
    b = np.array(
        [(1-BPO) * delta_f * delta_t * log2(1 + p* dB2lin(G-NF_dB) / N0 / delta_f) for G,BPO in zip(g,BF_pilot_overhead)]
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

def RA_heur_QPOS_SINR(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict, num_pilot_dict, **kwargs):
    # QoS-Prioritized Opportunistic Scheduling
    RA_dict = collections.OrderedDict()
    est_num_RB_allocated_perBS = kwargs.get('est_num_RB_allocated_perBS', None)
    if est_num_RB_allocated_perBS is not None:
        # 逐元素取上界args.num_RB_micro和预测值的较小值
        est_num_RB_allocated_perBS = np.minimum(est_num_RB_allocated_perBS, args.num_RB_micro)
    infer_g_dict = kwargs.get('infer_g_dict', None)
    BS_association_dict = kwargs.get('BS_association_dict', None)
    if not veh_set:  # 如果没有车辆，则返回空字典
        return RA_dict
    num_bs = g_dict[list(veh_set)[0]].shape[0]
    num_RB = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
    delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
    p = args.p_micro if BS_id > 0 else args.p_macro
    NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
    delta_t = args.slot_len
    N0 = args.N0
    veh_id_list = list(veh_set)
    q = np.array([q_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    a = np.array([a_dict[veh_id][slot_idx] for veh_id in veh_id_list])
    if (infer_g_dict is not None) and (est_num_RB_allocated_perBS is not None):
        Interference = np.array([
            sum([
                dB2lin(infer_g_dict[veh_id][other_BS_id]) * (est_num_RB_allocated_perBS[other_BS_id]/args.num_RB_micro) * args.p_micro # 缺了一个Km/K
                for other_BS_id in range(1,num_bs) if other_BS_id != BS_id
            ])
            for veh_id in veh_id_list
        ])  
    else:
        Interference = np.array([
                sum([
                    dB2lin(g_dict[veh_id][other_BS_id] - min_bf_gain_dB) * args.p_micro # 缺了一个Km/K
                    for other_BS_id in range(1,num_bs) if other_BS_id != BS_id
                ])
                for veh_id in veh_id_list
            ])
    if BS_id == 0:
        Interference *= 0  # macro BS不考虑微基站的干扰
            
    g = np.array([g_dict[veh_id][BS_id] for veh_id in veh_id_list])
    
    if BS_id == 0:
        BF_pilot_overhead = np.zeros(len(veh_id_list))
    else:
        BF_pilot_overhead = np.array([min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1) for veh_id in veh_id_list])
    b = np.array(
        [(1-BFO) * delta_f * delta_t * log2(1 + p*dB2lin(G) / (N0*delta_f*dB2lin(NF_dB) + I)) for I,G,BFO in zip(Interference,g,BF_pilot_overhead)]
    )  # 一个RB能提供的传输量
    backlog_flag = np.array([q_dict[veh_id][slot_idx] > Q_ub_dict[veh_id]/2 for veh_id in veh_id_list])
    max_b = b.max()
    weight = b/max_b * backlog_flag - (1-b/max_b) * (1-backlog_flag)
    priority = (-weight).argsort()
    resRB = num_RB
    for v in priority:
        if backlog_flag[v]:
            RB_alloc = min(math.ceil(q[v] / (b[v]+1e-15)), resRB)
        else:
            RB_alloc = min(int(q[v] / (b[v]+1e-15)), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict

def RA_heur_QPPF(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict, num_pilot_dict, **kwargs):
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
    if BS_id == 0:
        BF_pilot_overhead = np.zeros(len(veh_id_list))
    else:
        BF_pilot_overhead = np.array([min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1) for veh_id in veh_id_list])
    b = np.array(
        [(1-BPO) * delta_f * delta_t * log2(1 + p* dB2lin(G-NF_dB) / N0 / delta_f) for G,BPO in zip(g,BF_pilot_overhead)]
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


def RA_unlimitRB(args, slot_idx, BS_id, veh_set, veh_data_rate_dict, Q_ub_dict, q_dict, a_dict, g_dict, num_pilot_dict, **kwargs):
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
    if BS_id == 0:
        BF_pilot_overhead = np.zeros(len(veh_id_list))
    else:
        BF_pilot_overhead = np.array([min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1) for veh_id in veh_id_list])
    b = np.array(
        [(1-BPO) * delta_f * delta_t * log2(1 + p* dB2lin(G-NF_dB) / N0 / delta_f) for G,BPO in zip(g,BF_pilot_overhead)]
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


def HO_EE_predG(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array, **kwargs):
    # Energy Efficiency
    HO_cmd = collections.OrderedDict()
    infer_g_dict = kwargs.get('infer_g_dict', None) 
    num_pilot_dict = kwargs.get('num_pilot_dict', None)
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
            G = dB2lin(pred_g_dict[veh][BS_id] - NF_dB)
            EE_array[BS_id] = delta_f / p * log2(1 + p * G / args.N0 / delta_f)
        HO_cmd[veh] = EE_array.argmax()
    
    # 基于HO_cmd得到T_HO
    T_HO = np.zeros((len(BS_loc_array),len(veh_set_cur)))
    for i, veh in enumerate(veh_set_cur):
        T_HO[HO_cmd[veh], i] = 1
    # 计算k_tilde_matrix和power_matrix
    k_tilde_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    pred_G_dB = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    power_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        pred_G_dB[i, :] = pred_g_dict[veh]
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        k_tilde_matrix[:, BS_id] = lbd_array / (
            delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB)))
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    return HO_cmd, num_RB_allocated_perBS


def HO_EE_GAP_APX_with_offload_conservative_predG(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array, **kwargs):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    infer_g_dict = kwargs.get('infer_g_dict', None) 
    num_pilot_dict = kwargs.get('num_pilot_dict', None)
    points = np.zeros((len(veh_set_cur), 2))
    pred_G_dB = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
        pred_G_dB[i, :] = pred_g_dict[veh]
    power_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    k_tilde_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
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
        
    num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    return HO_cmd, num_RB_allocated_perBS


def HO_EE_GAP_APX_with_offload_conservative_predG_SINR(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array, **kwargs):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    infer_g_dict = kwargs.get('infer_g_dict', None) 
    num_pilot_dict = kwargs.get('num_pilot_dict', None)
    points = np.zeros((len(veh_set_cur), 2))
    pred_G_dB = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
        pred_G_dB[i, :] = pred_g_dict[veh]
    power_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    k_tilde_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        if infer_g_dict is not None:
            interference_array = np.array([
                sum([
                    dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])  
        else:
            interference_array = np.array([
                sum([
                    dB2lin(pred_g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])
        if BS_id == 0:
            interference_array *= 0  # macro BS不考虑微基站的干扰
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
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
    _num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    _T_HO = T_HO
    
    # 在上次迭代的基础上，进行微调
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        if infer_g_dict is not None:
            interference_array = np.array([
                sum([
                    dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro * (_num_RB_allocated_perBS[other_BS_id]/args.num_RB_micro)
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])  
        else:
            interference_array = np.array([
                sum([
                    dB2lin(pred_g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])
        if BS_id == 0:
            interference_array *= 0  # macro BS不考虑微基站的干扰
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
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
    num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd, num_RB_allocated_perBS


def HO_EE_GAP_APX_conservative_predG_SINR(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array, **kwargs):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    infer_g_dict = kwargs.get('infer_g_dict', None) 
    num_pilot_dict = kwargs.get('num_pilot_dict', None)
    points = np.zeros((len(veh_set_cur), 2))
    pred_G_dB = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
        pred_G_dB[i, :] = pred_g_dict[veh]
    power_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    k_tilde_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        if infer_g_dict is not None:
            interference_array = np.array([
                sum([
                    dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])  
        else:
            interference_array = np.array([
                sum([
                    dB2lin(pred_g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])
        if BS_id == 0:
            interference_array *= 0  # macro BS不考虑微基站的干扰
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        RB_num_table[BS_id] = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
        pm_table[BS_id] = args.p_micro if BS_id > 0 else args.p_macro
    T_HO, feasible_flag = HO_GAP_APX(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    _num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    _T_HO = T_HO
    
    # 在上次迭代的基础上，进行微调
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        if infer_g_dict is not None:
            interference_array = np.array([
                sum([
                    dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro * (_num_RB_allocated_perBS[other_BS_id]/args.num_RB_micro)
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])  
        else:
            interference_array = np.array([
                sum([
                    dB2lin(pred_g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])
        if BS_id == 0:
            interference_array *= 0  # macro BS不考虑微基站的干扰
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
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
    num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd, num_RB_allocated_perBS


def HO_EE_GAP_APX_conservative_predG_SINR_Rician(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array, **kwargs):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    g_var = (2*args.K_rician + 1) / (args.K_rician + 1)**2
    g_std = np.sqrt(g_var)
    g_std_factor = 0.2 * g_std  # 99.7%的置信区间
    infer_g_dict = kwargs.get('infer_g_dict', None) 
    num_pilot_dict = kwargs.get('num_pilot_dict', None)
    points = np.zeros((len(veh_set_cur), 2))
    pred_G_dB = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
        pred_G_dB[i, :] = pred_g_dict[veh]
    power_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    k_tilde_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        if infer_g_dict is not None:
            interference_array = np.array([
                sum([
                    dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])  
        else:
            interference_array = np.array([
                sum([
                    dB2lin(pred_g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])
        if BS_id == 0:
            interference_array *= 0  # macro BS不考虑微基站的干扰
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        RB_num_table[BS_id] = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
        pm_table[BS_id] = args.p_micro if BS_id > 0 else args.p_macro
    T_HO, feasible_flag = HO_GAP_APX(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    _num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    _T_HO = T_HO
    
    # 在上次迭代的基础上，进行微调
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        if infer_g_dict is not None:
            interference_array = np.array([
                sum([
                    dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro * (_num_RB_allocated_perBS[other_BS_id]/args.num_RB_micro)
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])  
        else:
            interference_array = np.array([
                sum([
                    dB2lin(pred_g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])
        if BS_id == 0:
            interference_array *= 0  # macro BS不考虑微基站的干扰
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
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
    num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd, num_RB_allocated_perBS


def HO_EE_offload(args, veh_set_cur, backlog_queue_dict, veh_data_rate_dict, pred_loc_dict, pred_g_dict, BS_loc_array, **kwargs):
    # Spectral Efficiency
    HO_cmd = collections.OrderedDict()
    infer_g_dict = kwargs.get('infer_g_dict', None) 
    num_pilot_dict = kwargs.get('num_pilot_dict', None)
    points = np.zeros((len(veh_set_cur), 2))
    pred_G_dB = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    for i, veh in enumerate(veh_set_cur):
        points[i, :] = pred_loc_dict[veh]
        pred_G_dB[i, :] = pred_g_dict[veh]
    power_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    k_tilde_matrix = np.zeros((len(veh_set_cur), len(BS_loc_array)))
    N0 = args.N0
    for BS_id in range(len(BS_loc_array)):
        delta_f = args.RB_intervel_micro if BS_id > 0 else args.RB_intervel_macro
        p = args.p_micro if BS_id > 0 else args.p_macro
        NF_dB = args.NF_micro_dB if BS_id > 0 else args.NF_macro_dB
        pred_G_array = dB2lin(pred_G_dB[:, BS_id])
        lbd_array = np.array([veh_data_rate_dict[veh] for veh in veh_set_cur])
        if infer_g_dict is not None:
            interference_array = np.array([
                sum([
                    dB2lin(infer_g_dict[veh][other_BS_id]) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])  
        else:
            interference_array = np.array([
                sum([
                    dB2lin(pred_g_dict[veh][other_BS_id] - min_bf_gain_dB) * args.p_micro
                    for other_BS_id in range(1,len(BS_loc_array)) if other_BS_id != BS_id
                ])
                for veh in veh_set_cur
            ])
        if BS_id == 0:
            interference_array *= 0  # macro BS不考虑微基站的干扰
        BF_overhad_array = np.zeros(len(veh_set_cur))
        if num_pilot_dict is not None:
            for i, veh in enumerate(veh_set_cur):
                BF_overhad_array[i] = min(num_pilot_dict[veh][BS_id-1] * args.pilot_overhead_factor,1) if BS_id > 0 else 0
        k_tilde_matrix[:, BS_id] = lbd_array / (
            (1-BF_overhad_array) * delta_f * np.log2(1 + p * pred_G_array / (N0 * delta_f * dB2lin(NF_dB) + interference_array))
        )
        power_matrix[:, BS_id] = k_tilde_matrix[:, BS_id] * p
    RB_num_table = np.zeros(len(BS_loc_array))
    pm_table = np.zeros(len(BS_loc_array))
    for BS_id in range(len(BS_loc_array)):
        RB_num_table[BS_id] = args.num_RB_micro if BS_id > 0 else args.num_RB_macro
        pm_table[BS_id] = args.p_micro if BS_id > 0 else args.p_macro
    T_HO, feasible_flag = HO_offload(
        T_KR=k_tilde_matrix.swapaxes(0, 1), T_TR=RB_num_table, T_PM=pm_table
    )
    num_RB_allocated_perBS = (k_tilde_matrix.swapaxes(0, 1)*T_HO).sum(axis=-1)
    for i, veh in enumerate(veh_set_cur):
        HO_cmd[veh] = T_HO[:, i].argmax()
    return HO_cmd, num_RB_allocated_perBS


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






























#################### others ##########################

def RA_Lyapunov(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict, **kwargs):
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
            k[i] * delta_f * delta_t * log2(1 + p * dB2lin(g[i]) / (N0 * delta_f))
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


def RA_heur_q(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict, **kwargs):
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
        [delta_f * delta_t * log2(1 + p * dB2lin(G) / (N0 * delta_f)) for G in g]
    )
    priority = (-q).argsort()
    resRB = num_RB
    for v in priority:
        RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    # print("BS_id=", BS_id, "RA_dict=", RA_dict)
    return RA_dict


def RA_heur_b(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict, **kwargs):
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
        [delta_f * delta_t * log2(1 + p * dB2lin(G) / (N0 * delta_f)) for G in g]
    )

    priority = (-b).argsort()
    resRB = num_RB
    for v in priority:
        RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_qb(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict, **kwargs):
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
        [delta_f * delta_t * log2(1 + p * dB2lin(G) / (N0 * delta_f)) for G in g]
    )

    priority = (-q * b).argsort()
    resRB = num_RB
    for v in priority:
        RB_alloc = min(math.ceil(q[v] / b[v]), resRB)
        RA_dict[veh_id_list[v]] = RB_alloc
        resRB -= RB_alloc
    return RA_dict


def RA_heur_fqb(args, slot_idx, BS_id, veh_set, q_dict, a_dict, g_dict, **kwargs):
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
        [delta_f * delta_t * log2(1 + p * dB2lin(G) / (N0 * delta_f)) for G in g]
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
        G_array = dB2lin(bias - beta * 10 * np.log10(dist_matrix[:, BS_id]))
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
        G_array = dB2lin(bias - beta * 10 * np.log10(dist_matrix[:, BS_id]))
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