import numpy as np
import collections
from numpy import log2, log10
from utils.mox_utils import lin2dB, dB2lin


def init_vehset_backlog_queue(veh_set, Q_ub_dict, Q_th=0.5, slots_per_frame=100):
    backlog_queue_dict = collections.OrderedDict()
    for veh in veh_set:
        backlog_queue_dict[veh] = np.random.choice(int(Q_ub_dict[veh]*Q_th)) * np.ones(slots_per_frame + 1)
        # p.s. slots_per_frame+1 第0元素表示上一帧末时隙的队列长度,第i(i>0)元素表示当前帧第i-1时隙的队列长度

    return backlog_queue_dict


def init4frame_vehset_backlog_queue(
    veh_set_remain, veh_set_in, Q_dict_prev, Q_ub_dict, Q_th=0.5, slots_per_frame=100
):
    backlog_queue_dict = collections.OrderedDict()
    for veh in veh_set_remain:
        backlog_queue_dict[veh] = np.zeros(slots_per_frame + 1)
        backlog_queue_dict[veh][0] = Q_dict_prev[veh][-1]
    for veh in veh_set_in:
        backlog_queue_dict[veh] = np.zeros(slots_per_frame + 1)
        backlog_queue_dict[veh][0] = np.random.choice(int(Q_ub_dict[veh]*Q_th))
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
    infer_g_dict,
    num_RB_allocated_perBS,
    num_pilot_dict,
    sinr_flag=True,
):
    delta_t = args.slot_len
    N0 = args.N0
    if sinr_flag:
        # 考虑通信速率由sinr而非snr决定的情况
        for veh_id in veh_set:
            BS_id = connection_dict[veh_id]
            q = backlog_queue_dict[veh_id][slot_idx]
            a = a_dict[veh_id][slot_idx]
            g = g_dict[veh_id][BS_id]
            k = RA_dict[veh_id]
            if BS_id == 0:
                delta_f = args.RB_intervel_macro
                p = args.p_macro
                NF_dB = args.NF_macro_dB
                snr = p * dB2lin(g) / (N0 * delta_f * dB2lin(NF_dB))
                r =  k * delta_f * delta_t * log2(1 + snr)
                # print('SNR',10*np.log10(p * dB2lin(g) / N0 / delta_f),'dB')
                # print('SNR_macro ',10*np.log10(snr),'dB')
                # print('r (k=1)',f'{(delta_f * delta_t * log2(1 + p * dB2lin(g) / N0 / delta_f)).item():.2e}bps')
                # print('r_macro (k=1)',f'{(0.1 * delta_f * delta_t * log2(1 + 1600 * p * dB2lin(g) / N0 / delta_f)).item():.2e}bps')
            else:
                delta_f = args.RB_intervel_micro
                p = args.p_micro
                NF_dB = args.NF_micro_dB
                Interference = 0
                # 计算干扰
                for other_microBS_id in range(1,len(num_RB_allocated_perBS)):
                    if other_microBS_id != BS_id:
                        Interference += num_RB_allocated_perBS[other_microBS_id] / args.num_RB_micro * p * dB2lin(infer_g_dict[veh_id][other_microBS_id])
                sinr = p * dB2lin(g) / (N0 * delta_f * dB2lin(NF_dB) + Interference)
                BF_pilot_overhead = min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1)
                r = (1-BF_pilot_overhead) * k * delta_f * delta_t * log2(1 + sinr)
                # print(f'SNR_micro-{BS_id}',
                #       f'SINR {10*np.log10(sinr).item():.2f} dB',
                #       f'SNR {10*np.log10(p * dB2lin(g) / (N0 * delta_f * dB2lin(NF_dB))).item():.2f} dB',
                #       f'SIR {10*np.log10(p * dB2lin(g) / Interference).item():.2f} dB'
                #     )
                # import ipdb; ipdb.set_trace()
            # 241022
            backlog_queue_dict[veh_id][slot_idx + 1] = max(q - r, 0) + a
    else:
        for veh_id in veh_set:
            BS_id = connection_dict[veh_id]
            q = backlog_queue_dict[veh_id][slot_idx]
            a = a_dict[veh_id][slot_idx]
            g = g_dict[veh_id][BS_id]
            k = RA_dict[veh_id]
            if BS_id == 0:
                delta_f = args.RB_intervel_macro
                p = args.p_macro
                NF_dB = args.NF_macro_dB
                snr = p * dB2lin(g-NF_dB) / (N0 * delta_f)
                r = k * delta_f * delta_t * log2(1 + snr)
            else:
                delta_f = args.RB_intervel_micro
                p = args.p_micro
                NF_dB = args.NF_micro_dB
                snr = p * dB2lin(g-NF_dB) / (N0 * delta_f)
                BF_pilot_overhead = min(num_pilot_dict[veh_id][BS_id-1] * args.pilot_overhead_factor,1)
                r = (1-BF_pilot_overhead) * k * delta_f * delta_t * log2(1 + snr)
            backlog_queue_dict[veh_id][slot_idx + 1] = max(q - r, 0) + a
    return backlog_queue_dict
