import os
import sys
import collections
import math

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


def update_CSI(
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

def calculate_uma_pathloss_3gpp_38901(
    distance_2d_m: float,
    fc_ghz: float,
    h_bs_m: float,
    h_ut_m: float,
    scenario: str = 'nlos',
    shadow_fading_db: float = 0.0
) -> float:
    """
    Calculates path loss based on 3GPP TR 38.901 UMa scenario.

    Reference: 3GPP TR 38.901 V17.0.0 (2021-12), Table 7.4.1-1.

    Args:
        distance_2d_m (float): 2D horizontal distance between BS and UT in meters.
                               Applicable range: LOS: 10m <= d_2D <= 5km
                                                 NLOS: 10m <= d_2D (implicitly via d_3D, up to 5km)
        fc_ghz (float): Carrier frequency in GHz. Applicable range: 0.5 GHz <= fc <= 100 GHz.
        h_bs_m (float): Base station antenna height in meters. Applicable range: 10m <= h_BS <= 150m.
                        (Refers to effective antenna height h'_BS = h_BS - h_E, where h_E=1m for UMa)
        h_ut_m (float): User terminal antenna height in meters. Applicable range: 1m <= h_UT <= 10m.
                        (Refers to effective antenna height h'_UT = h_UT - h_E, where h_E=1m for UMa)
        scenario (str): 'los' (Line-of-Sight) or 'nlos' (Non-Line-of-Sight). Defaults to 'nlos'.
        shadow_fading_db (float): Shadow fading value in dB. For mean path loss, this is 0.
                                 TR 38.901 defines sigma_SF for UMa LOS and NLOS.
                                 This function allows direct input of a specific shadow fading value.

    Returns:
        float: Path loss in dB. Returns float('inf') if parameters are out of recommended range
               or if calculation is invalid.
    """

    # --- Input validation and model applicability checks (based on TR 38.901 Table 7.4.1-1) ---
    if not (0.5 <= fc_ghz <= 100):
        print(f"警告: 载波频率 {fc_ghz} GHz 超出建议范围 (0.5 GHz 至 100 GHz)。")
    if not (10 <= h_bs_m <= 150):
        print(f"警告: 基站高度 {h_bs_m} m 超出建议范围 (10m 至 150m)。")
    if not (1 <= h_ut_m <= 10):
        print(f"警告: 用户终端高度 {h_ut_m} m 超出建议范围 (1m 至 10m)。")

    if distance_2d_m < 0:
        print("错误: 距离不能为负。")
        return float('inf')
    if fc_ghz <= 0:
        print("错误: 频率必须为正。")
        return float('inf')

    # --- Constants and parameters ---
    c_mps = 299792458.0  # Speed of light (m/s)
    h_E_m = 1.0          # Effective environment height for UMa (m)

    # Calculate effective antenna heights
    h_bs_eff_m = h_bs_m - h_E_m
    h_ut_eff_m = h_ut_m - h_E_m

    # Effective heights must be positive for d_bp calculation
    if h_bs_eff_m <= 0 or h_ut_eff_m <= 0 :
        # print(f"警告: 有效天线高度 (h_bs_eff={h_bs_eff_m}, h_ut_eff={h_ut_eff_m}) 必须为正才能计算断点距离。")
        # d_bp_eff_m will be handled below, but this is a critical check for model validity.
        # If effective heights are not positive, d_bp calculation is problematic.
        pass


    # Calculate breakpoint distance d'_BP (TR 38.901, formula 7.4-2)
    # d'_BP = 4 * h'_BS * h'_UT * f_c / c  (f_c in Hz)
    d_bp_eff_m = float('inf') # Default if not calculable
    if h_bs_eff_m > 0 and h_ut_eff_m > 0 : # fc_ghz > 0 already checked
        d_bp_eff_m = 4 * h_bs_eff_m * h_ut_eff_m * (fc_ghz * 1e9) / c_mps
    else:
        # This condition implies that the standard breakpoint distance formula might not be applicable
        # or that the antenna heights are too low relative to the effective environment height.
        # For UMa, h_E is 1m, so h_bs_m must be > 1m and h_ut_m must be > 1m for positive effective heights.
        # The model constraints are h_bs >=10m and h_ut >=1m, so h_bs_eff will always be positive.
        # h_ut_eff can be zero if h_ut_m = 1m.
        # If h_ut_eff_m is zero, d_bp_eff_m becomes zero.
        # The LOS model formulas use d_2D directly and d_bp_eff_m.
        # If d_bp_eff_m is zero, the condition distance_2d_calc < d_bp_eff_m might behave unexpectedly.
        # Let's ensure d_bp_eff_m is at least a very small positive number if h_ut_eff_m is zero but h_bs_eff_m is positive.
        # Or, more robustly, if d_bp_eff_m calculates to 0, the second LOS formula (d_2D >= d_bp_eff_m) should apply.
        if h_bs_eff_m > 0 and h_ut_eff_m == 0: # UT at effective environment height
            d_bp_eff_m = 0.0 # Breakpoint distance is effectively zero.

    # Calculate 3D distance d_3D
    delta_h_m = abs(h_bs_m - h_ut_m)
    distance_3d_m = math.sqrt(distance_2d_m**2 + delta_h_m**2)
    if distance_3d_m == 0: # Avoid log(0)
        distance_3d_m = 0.01 # Use a very small positive value

    pathloss_db = float('inf')

    # --- Path loss calculation ---
    if scenario.lower() == 'los':
        # Line-of-Sight (LOS) Path Loss (TR 38.901, Table 7.4.1-1)
        # Applicable: 10m <= d_2D <= 5km

        distance_2d_calc = distance_2d_m
        if distance_2d_m < 10:
            print(f"警告: LOS 场景下，二维距离 {distance_2d_m}m 小于模型适用的最小距离 10m。结果可能基于外插。")
            distance_2d_calc = max(distance_2d_m, 0.01) # Use a small positive value to avoid log(0)
        elif distance_2d_m > 5000:
            print(f"警告: LOS 场景下，二维距离 {distance_2d_m}m 大于模型适用的最大距离 5000m。将使用 5000m 进行计算。")
            distance_2d_calc = 5000.0

        if distance_2d_calc == 0: distance_2d_calc = 0.01 # Final check for log

        try:
            if distance_2d_calc < d_bp_eff_m :
                # PL_1 for UMa-LOS
                pathloss_db = 28.0 + 22 * math.log10(distance_2d_calc) + 20 * math.log10(fc_ghz)
            else: # distance_2d_calc >= d_bp_eff_m
                # PL_2 for UMa-LOS
                term_in_log_pl2 = (d_bp_eff_m**2) + (delta_h_m**2)
                if term_in_log_pl2 <= 0: # Should not happen if d_bp_eff_m >= 0
                     # This case usually means d_bp_eff_m is 0 and delta_h_m is 0.
                     # If d_bp_eff_m is 0, then distance_2d_calc >= d_bp_eff_m is always true.
                     # If term_in_log_pl2 is 0, log10 will fail.
                     # Fallback to a very small positive number for the log argument, or reconsider model applicability.
                     # Given d_bp_eff_m can be 0 if h_ut_eff_m is 0.
                     # If d_bp_eff_m = 0 and delta_h_m = 0, then h_ut = h_bs = 1m.
                     # This is outside h_bs range (>=10m).
                     # If d_bp_eff_m > 0, term_in_log_pl2 will be > 0.
                     # If d_bp_eff_m = 0, then term_in_log_pl2 = delta_h_m^2. If delta_h_m is also 0, then it's an issue.
                     # However, h_bs >= 10m and h_ut <=10m, so delta_h_m can be 0 if h_bs = h_ut.
                     # If h_ut_eff_m = 0 (h_ut=1m) and h_bs_eff_m > 0 (h_bs > 1m), then d_bp_eff_m = 0.
                     # Then term_in_log_pl2 = (h_bs_m - h_ut_m)^2. This is >=0.
                     # If h_bs_m = h_ut_m, then term_in_log_pl2 = 0.
                     # For UMa, h_bs is typically much larger than h_ut.
                     # Let's assume term_in_log_pl2 will be positive due to model constraints on h_bs vs h_ut,
                     # or d_bp_eff_m being positive.
                     # If it still becomes zero (e.g. d_bp_eff_m=0 and delta_h_m=0), use a small value for log.
                    print(f"警告: LOS PL2 计算中 log 内的项 ({term_in_log_pl2}) 为零或负。这可能表示参数组合不符合典型模型假设。")
                    pathloss_db = 28.0 + 40 * math.log10(distance_2d_calc) + 20 * math.log10(fc_ghz) # Simpler form without last term
                else:
                    pathloss_db = (28.0 + 40 * math.log10(distance_2d_calc) +
                                   20 * math.log10(fc_ghz) -
                                   9 * math.log10(term_in_log_pl2))
        except ValueError as e:
            print(f"LOS 计算中发生数学错误: {e}。检查输入参数，特别是距离和频率。")
            return float('inf')

    elif scenario.lower() == 'nlos':
        # Non-Line-of-Sight (NLOS) Path Loss (TR 38.901, Table 7.4.1-1, Note 5)
        # PL_UMa,NLOS = max(PL_LOS(d_3D), PL'_UMa,NLOS)
        # Applicable: d_3D >= 10m and d_3D <= 5km (implicitly)

        distance_3d_calc_nlos = distance_3d_m
        if distance_3d_m < 10:
            print(f"警告: NLOS 场景下，三维距离 {distance_3d_m}m 小于模型建议的最小距离 10m。结果可能基于外插。")
            distance_3d_calc_nlos = max(distance_3d_m, 0.01) # Use small positive for log
        elif distance_3d_m > 5000:
            print(f"警告: NLOS 场景下，三维距离 {distance_3d_m}m 大于模型建议的最大距离 5000m。将使用 5000m 进行计算。")
            distance_3d_calc_nlos = 5000.0
        
        if distance_3d_calc_nlos == 0: distance_3d_calc_nlos = 0.01 # Final check

        # 1. Calculate PL_LOS(d_3D) component
        pl_los_component_db = float('inf')
        try:
            if distance_3d_calc_nlos < d_bp_eff_m:
                pl_los_component_db = (28.0 + 22 * math.log10(distance_3d_calc_nlos) +
                                       20 * math.log10(fc_ghz))
            else: # distance_3d_calc_nlos >= d_bp_eff_m
                term_in_log_nlos_los_comp = (d_bp_eff_m**2) + (delta_h_m**2)
                if term_in_log_nlos_los_comp <= 0:
                    print(f"警告: NLOS 中 PL_LOS(d_3D) 部分的 PL2 计算中 log 内的项 ({term_in_log_nlos_los_comp}) 为零或负。")
                    pl_los_component_db = (28.0 + 40 * math.log10(distance_3d_calc_nlos) + 20 * math.log10(fc_ghz))
                else:
                    pl_los_component_db = (28.0 + 40 * math.log10(distance_3d_calc_nlos) +
                                           20 * math.log10(fc_ghz) -
                                           9 * math.log10(term_in_log_nlos_los_comp))
        except ValueError as e:
            print(f"NLOS 中 PL_LOS(d_3D) 计算发生数学错误: {e}")
            # pl_los_component_db remains float('inf')

        # 2. Calculate PL'_UMa,NLOS component
        # PL'_UMa,NLOS = 13.54 + 39.08 * log10(d_3D) + 20 * log10(f_c [GHz]) - 0.6 * (h_UT - 1.5)
        # Note: h_UT is the actual user terminal height here.
        pl_prime_nlos_db = float('inf')
        try:
            pl_prime_nlos_db = (13.54 + 39.08 * math.log10(distance_3d_calc_nlos) +
                                20 * math.log10(fc_ghz) -
                                0.6 * (h_ut_m - 1.5))
        except ValueError as e:
            print(f"NLOS 中 PL'_UMa,NLOS 计算发生数学错误: {e}")
            # pl_prime_nlos_db remains float('inf')

        # 3. PL_UMa,NLOS = max(PL_LOS(d_3D), PL'_UMa,NLOS)
        if pl_los_component_db != float('inf') and pl_prime_nlos_db != float('inf'):
            pathloss_db = max(pl_los_component_db, pl_prime_nlos_db)
        elif pl_los_component_db != float('inf'):
            print("警告: NLOS 计算中 PL'_UMa,NLOS 部分失败，仅使用 PL_LOS(d_3D) 作为路径损耗。")
            pathloss_db = pl_los_component_db
        elif pl_prime_nlos_db != float('inf'):
            print("警告: NLOS 计算中 PL_LOS(d_3D) 部分失败，仅使用 PL'_UMa,NLOS 作为路径损耗。")
            pathloss_db = pl_prime_nlos_db
        else: # Both failed
            print("错误: NLOS 计算中两个组成部分均失败。")
            pathloss_db = float('inf')
    else:
        print(f"错误: 未知场景 '{scenario}'。请使用 'los' 或 'nlos'。")
        return float('inf')

    # Apply shadow fading if pathloss calculation was successful
    if pathloss_db != float('inf'):
        total_pathloss_db = pathloss_db + shadow_fading_db
        return total_pathloss_db
    else:
        return float('inf')

def calculate_channel_gain_from_pathloss(pathloss_db: float) -> float:
    """
    Converts path loss in dB to linear channel gain.
    Channel Gain G = 1 / L, where L is the linear path loss.
    L_linear = 10^(L_dB / 10)

    Args:
        pathloss_db (float): Path loss in dB.

    Returns:
        float: Linear channel gain. Returns 0.0 if pathloss_db is infinite.
    """
    if pathloss_db == float('inf'):
        return 0.0
    try:
        pathloss_linear = 10**(pathloss_db / 10.0)
        if pathloss_linear == 0: # Avoid division by zero if pathloss_linear is extremely high (practically zero)
            return 0.0
        channel_gain_linear = 1.0 / pathloss_linear
        return channel_gain_linear
    except OverflowError:
        # This can happen if pathloss_db is very large, making 10**(pathloss_db / 10.0) too large to represent.
        print(f"警告: 计算线性路径衰落时发生溢出 (pathloss_db = {pathloss_db})。返回增益为0。")
        return 0.0
    
    
def get_g_macroBS_dict(args, loc_dict, MacroBS_loc, fc_ghz=2.8, Gt_macro=17, scenario='los'):
            # 预测宏基站与车辆veh的信道增益
            g_macroBS_dict = collections.OrderedDict()
            for veh, pred_loc in loc_dict.items():
                PL = calculate_uma_pathloss_3gpp_38901(distance_2d_m=np.linalg.norm(pred_loc-MacroBS_loc),
                                                       fc_ghz=fc_ghz,
                                                       h_ut_m=args.h_car,
                                                       h_bs_m=args.h_tx,
                                                       scenario=scenario)
                # g_macroBS_dict[veh] = calculate_channel_gain_from_pathloss(PL-Gt_macro)
                g_macroBS_dict[veh] = Gt_macro-PL
            return g_macroBS_dict
        