import numpy as np
import collections
from numpy import log2, log10
from utils.channel_utils import rician_channel_gain


def init_backlog_queue(Q_th=100, slots_per_frame=100):
    # p.s. slots_per_frame+1 第0元素表示上一帧末时隙的队列长度,第i(i>0)元素表示当前帧第i-1时隙的队列长度
    return np.random.choice(Q_th) * np.ones(slots_per_frame + 1)


def init_vehset_backlog_queue(veh_set, Q_th=100, slots_per_frame=100):
    backlog_queue_dict = collections.OrderedDict()
    for veh in veh_set:
        backlog_queue_dict[veh] = np.random.choice(Q_th) * np.ones(slots_per_frame + 1)
        # p.s. slots_per_frame+1 第0元素表示上一帧末时隙的队列长度,第i(i>0)元素表示当前帧第i-1时隙的队列长度

    return backlog_queue_dict


def init4frame_vehset_backlog_queue(
    veh_set_remain, veh_set_in, Q_dict_prev, Q_th=100, slots_per_frame=100
):
    backlog_queue_dict = collections.OrderedDict()
    for veh in veh_set_remain:
        backlog_queue_dict[veh] = np.zeros(slots_per_frame + 1)
        backlog_queue_dict[veh][0] = Q_dict_prev[veh][-1]
    for veh in veh_set_in:
        backlog_queue_dict[veh] = np.zeros(slots_per_frame + 1)
        backlog_queue_dict[veh][0] = np.random.choice(Q_th)
    return backlog_queue_dict


def update4slot_vehset_backlog_queue(
    args,
    slot_idx,
    RA_dict,
    veh_set,
    connection_dict,
    backlog_queue_dict,
    a_dict,
    g_dict,
):
    delta_t = args.slot_len
    N0 = args.N0
    for veh_id in veh_set:
        BS_id = connection_dict[veh_id]
        if BS_id == 0:
            delta_f = args.RB_intervel_macro
            p = args.p_macro
        else:
            delta_f = args.RB_intervel_micro
            p = args.p_micro
        q = backlog_queue_dict[veh_id][slot_idx]
        a = a_dict[veh_id][slot_idx]
        g = g_dict[veh_id][BS_id] * rician_channel_gain(args.K_rician, size=1)
        k = RA_dict[veh_id]
        r = k * delta_f * delta_t * log2(1 + p * 10 ** (g / 10) / N0 / delta_f)
        
        # print('SNR',10*np.log10(p * 10 ** (g / 10) / N0 / delta_f),'dB')
        # print('SNR_macro',10*np.log10(1600 * p * 10 ** (g / 10) / N0 / delta_f),'dB')
        # print('r (k=1)',f'{(delta_f * delta_t * log2(1 + p * 10 ** (g / 10) / N0 / delta_f)).item():.2e}bps')
        # print('r_macro (k=1)',f'{(0.1 * delta_f * delta_t * log2(1 + 1600 * p * 10 ** (g / 10) / N0 / delta_f)).item():.2e}bps')
        
        # backlog_queue_dict[veh_id][slot_idx + 1] = max(q + a - r, 0)
        # 241022
        backlog_queue_dict[veh_id][slot_idx + 1] = max(q - r, 0) + a
    return backlog_queue_dict
