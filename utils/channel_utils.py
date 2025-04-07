import os
import sys
import collections

sys.path.append(os.getcwd())
import numpy as np
from utils.mox_utils import lin2dB

def rician_channel_gain(K=1, size=100):
    x = np.random.normal(0, np.sqrt(1 / 2), size)  # Real part of scattered component
    y = np.random.normal(
        0, np.sqrt(1 / 2), size
    )  # Imaginary part of scattered component
    # Combine LOS and NLOS components to form the Rician distribution
    rician_gain = (K + 2 * np.sqrt(K) * x + x**2 + y**2) / (K + 1)
    # Return the squared magnitude, which is the channel gain
    return rician_gain


def cal_distance_BS_UE(pos_array, BS_loc):
    # BS_loc.shape = (2,)
    relative_pos_array = pos_array - BS_loc
    d = np.linalg.norm(relative_pos_array, ord=2, axis=-1, keepdims=False)
    return d


def cal_dist_oneUE_multiBS(UE_loc, BS_loc_array):
    # UE_loc.shape = (2,)    BS_loc_array.shape = (M,2)
    relative_pos_array = BS_loc_array - UE_loc.reshape(1, 2)
    dist_oneUE_multiBS = np.linalg.norm(
        relative_pos_array, ord=2, axis=-1, keepdims=False
    )
    return dist_oneUE_multiBS


def generate_random_channel(
    d,
    bias=-128,
    beta=3.7,
    shd_sigma=0,
    shd_crr_coeff=0.8,
    fd_sigma=0,
    num_frame=1,
    slots_per_frame=100,
    eps=1e-12,
):
    # d is a numpy array containing the historical distances between the UE and the BS
    # output is the historical channel gain (dB)
    assert type(d) == np.ndarray
    assert d.shape[-1] >= num_frame
    d = d[..., :num_frame]
    pathloss = beta * 10 * np.log10(d)
    shd = np.random.normal(loc=0, scale=shd_sigma, size=d.shape)
    # AR(1) self regression process
    if d.ndim == 2:
        for i in range(len(shd)):
            for j in range(1, num_frame):
                shd[i, j] = shd_crr_coeff * shd[i, j - 1] + np.sqrt(
                    1 - shd_crr_coeff**2
                ) * np.random.normal(0, shd_sigma)
    gain_frame = bias - pathloss - shd
    gain_slot = np.zeros((*gain_frame.shape, slots_per_frame))
    gain_slot[:] = gain_frame[..., np.newaxis]
    fd = (
        np.random.rayleigh(scale=fd_sigma, size=gain_slot.shape) + eps
    )  # eps ensures fd>0
    fd = 10 * np.log10(fd)
    gain_slot = gain_slot - fd
    return gain_frame, gain_slot


def generate_random_channel_onlyiidshd(
    d,
    bias=-128,
    beta=3.7,
    shd_sigma=0,
    num_frame=1,
):
    # d is a numpy array containing the historical distances between the UE and the BS
    # output is the historical channel gain (dB)
    assert type(d) == np.ndarray
    assert d.shape[-1] >= num_frame
    d = d[..., :num_frame]
    pathloss = beta * 10 * np.log10(d)
    shd = np.random.normal(loc=0, scale=shd_sigma, size=d.shape)
    gain_frame = bias - pathloss - shd
    return gain_frame


def generate_CSI_oneUE_multiBS_onlyiidshd(
    args,
    UE_loc,
    BS_loc_array,
):
    # d is a numpy array containing the distances between one UE and multiple BSs
    # output is the measured CSI: channel gain (dB)
    dist_oneUE_multiBS = cal_dist_oneUE_multiBS(UE_loc, BS_loc_array)
    assert type(dist_oneUE_multiBS) == np.ndarray
    assert dist_oneUE_multiBS.ndim == 1
    CSI_oneUE_multiBS = np.zeros_like(dist_oneUE_multiBS)
    # macro cell
    pathloss_macro = args.beta_macro * 10 * np.log10(dist_oneUE_multiBS[0])
    shd_macro = np.random.normal(loc=0, scale=args.shd_sigma_macro, size=(1,))
    CSI_oneUE_macroBS = args.bias_macro - pathloss_macro - shd_macro
    CSI_oneUE_multiBS[0] = CSI_oneUE_macroBS

    # micro cell
    pathloss_micro = args.beta_micro * 10 * np.log10(dist_oneUE_multiBS[1:])
    shd_micro = np.random.normal(
        loc=0, scale=args.shd_sigma_micro, size=(len(dist_oneUE_multiBS) - 1,)
    )
    CSI_oneUE_microBS = args.bias_micro - pathloss_micro - shd_micro
    CSI_oneUE_multiBS[1:] = CSI_oneUE_microBS
    return CSI_oneUE_multiBS


def update_CSI_onlyiidshd(
    args,
    veh_set_remain,
    veh_set_in,
    CSI_dict_prev,
    trajectoryInfo,
):
    CSI_dict = collections.OrderedDict() #
    for veh in veh_set_remain:
        CSI_dict[veh] = CSI_dict_prev[veh]
        # trajectoryInfo = timeline_dir[frame_cur]
        CSI_dict[veh] = np.concatenate([CSI_dict[veh], trajectoryInfo[veh]["CSI_preprocessed"].reshape(1, -1)], axis=0)
        CSI_dict[veh] = CSI_dict[veh][-args.frames_per_sample :, :]
    for veh in veh_set_in:
        CSI_dict[veh] = trajectoryInfo[veh]["CSI_preprocessed"][np.newaxis,...].repeat(args.frames_per_sample, axis=0)
    return CSI_dict
