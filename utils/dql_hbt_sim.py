"""Exact slot-level evaluation of a frozen DQL-HBT policy."""

from __future__ import annotations

import collections
import dataclasses
import time
from typing import Dict, MutableMapping, Sequence

import numpy as np
import tqdm

from utils.alg_utils import (
    RA_OTR_SINR,
    estimate_num_RB_allocated_perBS,
    update_BS_association_state,
)
from utils.beam_utils import generate_dft_codebook
from utils.channel_utils import rician_channel_gain
from utils.dql_hbt import (
    DQLHBTPolicy,
    HBTLearnerState,
    apply_hbt_action,
    effective_sinr_db,
    make_state_vector,
    should_make_decision,
)
from utils.mox_utils import dB2lin, lin2dB
from utils.pql_ba import (
    fixed_pair_gain_db,
    macro_gain_db,
    no_bf_gain_db,
    sweep_pilots_for_slot,
)
from utils.queue_utils import (
    init4frame_vehset_backlog_queue,
    init_vehset_backlog_queue,
    update4slot_vehset_backlog_queue,
)


@dataclasses.dataclass
class DQLHBTSimulationResult:
    energy_record: np.ndarray
    handover_record: np.ndarray
    beam_switch_record: np.ndarray
    violation_probability_record: np.ndarray
    average_queue_record: np.ndarray
    pilot_record: np.ndarray
    rb_allocated_record: np.ndarray
    queue_per_vehicle_record: MutableMapping
    association_record: MutableMapping
    action_record: MutableMapping
    decision_record: np.ndarray
    skipped_trigger_record: np.ndarray
    tracking_decision_record: np.ndarray
    inference_time_record: np.ndarray
    full_sweep_record: np.ndarray
    local_sweep_record: np.ndarray


def run_sim_dql_hbt(
    args,
    micro_bs_loc_list: Sequence[np.ndarray],
    timeline_dir: MutableMapping,
    policy: DQLHBTPolicy,
    ra_func=RA_OTR_SINR,
    seed: int = 1,
    prt: bool = True,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
    rician_fading: bool = True,
) -> DQLHBTSimulationResult:
    """Evaluate a frozen DQL-HBT policy in the manuscript simulator.

    A command based on frame x observations is applied in frame x+1.  OTR-RA,
    Poisson traffic, Rician fading, interference, and queue updates are then
    executed for all 100 slots of the frame, exactly as for the other methods.
    """

    config = policy.config
    if len(micro_bs_loc_list) != config.num_micro_bs:
        raise ValueError("micro BS count does not match policy")
    np.random.seed(seed)
    dft_tx = generate_dft_codebook(config.num_tx_beams)
    dft_rx = generate_dft_codebook(config.num_rx_beams)
    macro_loc = np.asarray(macro_bs_loc, dtype=float)
    bs_loc_list = [macro_loc] + [np.asarray(x, dtype=float) for x in micro_bs_loc_list]
    bs_loc_array = np.asarray(bs_loc_list)
    bs_loc_dict = collections.OrderedDict((i, x) for i, x in enumerate(bs_loc_list))

    frame_list = list(timeline_dir.keys())
    if len(frame_list) < 2:
        raise ValueError("timeline must contain at least two frames")
    num_frames = len(frame_list) - 1
    frame_prev = frame_list[0]
    veh_set_prev = set(timeline_dir[frame_prev].keys())
    veh_set_all = set()
    for frame in frame_list:
        veh_set_all.update(timeline_dir[frame].keys())

    vehicle_rate = collections.OrderedDict()
    for vehicle in sorted(veh_set_all, key=str):
        vehicle_rate[vehicle] = args.data_rate * np.random.uniform(
            1.0 - args.random_factor_range4data_rate,
            1.0 + args.random_factor_range4data_rate,
        )
    queue_upper_bound = collections.OrderedDict(
        (vehicle, args.lat_slot_ub * rate * args.slot_len)
        for vehicle, rate in vehicle_rate.items()
    )
    queue_prev = init_vehset_backlog_queue(
        veh_set_prev,
        queue_upper_bound,
        Q_th=0.5,
        slots_per_frame=args.slots_per_frame,
    )

    learners: Dict[object, HBTLearnerState] = {}
    for vehicle in veh_set_prev:
        position = np.asarray(timeline_dir[frame_prev][vehicle]["pos"], dtype=float)
        learners[vehicle] = HBTLearnerState(
            action=0,
            rx_beam=None,
            pending_action=None,
            last_position=position.copy(),
            distance_since_event=config.zone_size_m,
            tx_beam=None,
        )

    energy_record = np.zeros(num_frames)
    handover_record = np.zeros(num_frames)
    beam_switch_record = np.zeros(num_frames)
    violation_record = np.zeros(num_frames)
    average_queue_record = np.zeros(num_frames)
    pilot_record = np.zeros(num_frames)
    rb_record = np.zeros((num_frames, len(bs_loc_list)))
    decision_record = np.zeros(num_frames)
    skipped_trigger_record = np.zeros(num_frames)
    tracking_decision_record = np.zeros(num_frames)
    inference_time_record = np.zeros(num_frames)
    full_sweep_record = np.zeros(num_frames)
    local_sweep_record = np.zeros(num_frames)
    queue_per_vehicle = collections.OrderedDict()
    association_record = collections.OrderedDict()
    action_record = collections.OrderedDict()

    sim_start = time.time()
    iterator = enumerate(frame_list[1:])
    iterator = tqdm.tqdm(
        iterator, total=num_frames, desc="DQL-HBT simulation", disable=not prt
    )
    for frame_index, frame_cur in iterator:
        records = timeline_dir[frame_cur]
        veh_set_cur = set(records.keys())
        veh_set_in = veh_set_cur.difference(veh_set_prev)
        veh_set_remain = veh_set_cur.intersection(veh_set_prev)
        for vehicle in veh_set_prev.difference(veh_set_cur):
            learners.pop(vehicle, None)

        queue_cur = init4frame_vehset_backlog_queue(
            veh_set_remain,
            veh_set_in,
            queue_prev,
            queue_upper_bound,
            Q_th=0.5,
            slots_per_frame=args.slots_per_frame,
        )
        for vehicle in veh_set_in:
            position = np.asarray(records[vehicle]["pos"], dtype=float)
            learners[vehicle] = HBTLearnerState(
                action=0,
                rx_beam=None,
                pending_action=None,
                last_position=position.copy(),
                distance_since_event=config.zone_size_m,
                tx_beam=None,
            )

        handovers = 0
        beam_switches = 0
        for vehicle in sorted(veh_set_cur, key=str):
            learner = learners[vehicle]
            pending = learner.pending_action
            learner.pending_action = None
            outcome = apply_hbt_action(
                learner, pending, records[vehicle], config, dft_tx, dft_rx
            )
            handovers += int(outcome.handover)
            beam_switches += int(outcome.beam_switch)
            if outcome.sweep_pilots == config.full_sweep_pilots:
                full_sweep_record[frame_index] += 1
            elif outcome.sweep_pilots > 0:
                local_sweep_record[frame_index] += 1

        connection = collections.OrderedDict(
            (vehicle, int(learners[vehicle].action))
            for vehicle in sorted(veh_set_cur, key=str)
        )
        bs_association, _ = update_BS_association_state(bs_loc_dict, connection)

        gain_frame = collections.OrderedDict()
        inference_gain = collections.OrderedDict()
        selected_gain: Dict[object, float] = {}
        for vehicle in sorted(veh_set_cur, key=str):
            record = records[vehicle]
            learner = learners[vehicle]
            macro_gain = macro_gain_db(args, record["pos"], macro_loc)
            micro_no_bf = no_bf_gain_db(record["h"])
            gains = np.concatenate(([macro_gain], micro_no_bf.copy()))
            serving_bs = connection[vehicle]
            if serving_bs == 0:
                gain = macro_gain
            else:
                if learner.tx_beam is None or learner.rx_beam is None:
                    raise RuntimeError("micro link has no HBT beam pair")
                gain = fixed_pair_gain_db(
                    record["h"],
                    serving_bs - 1,
                    int(learner.tx_beam),
                    int(learner.rx_beam),
                    dft_tx,
                    dft_rx,
                )
            gains[serving_bs] = gain
            gain_frame[vehicle] = gains
            inference_gain[vehicle] = np.concatenate(([macro_gain], micro_no_bf))
            selected_gain[vehicle] = gain

        estimated_rb = estimate_num_RB_allocated_perBS(
            args,
            connection,
            bs_loc_array,
            veh_set_cur,
            gain_frame,
            vehicle_rate,
            infer_g_dict=inference_gain,
        )
        rb_capacities = np.asarray(
            [args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs,
            dtype=float,
        )
        estimated_load = np.clip(estimated_rb / rb_capacities, 0.0, 1.0)

        commands = collections.OrderedDict()
        for vehicle in sorted(veh_set_cur, key=str):
            learner = learners[vehicle]
            position = np.asarray(records[vehicle]["pos"], dtype=float)
            learner.distance_since_event += float(
                np.linalg.norm(position - learner.last_position)
            )
            learner.last_position = position.copy()
            command = None
            if learner.distance_since_event + 1e-9 >= config.zone_size_m:
                serving_bs = connection[vehicle]
                if serving_bs == 0:
                    interference_db = -np.inf
                else:
                    interference_w = sum(
                        dB2lin(inference_gain[vehicle][other_bs])
                        * args.p_micro
                        * estimated_load[other_bs]
                        for other_bs in range(1, config.num_bs)
                        if other_bs != serving_bs
                    )
                    noise_w = (
                        args.N0
                        * args.RB_intervel_micro
                        * dB2lin(args.NF_micro_dB)
                    )
                    interference_db = float(lin2dB(interference_w / noise_w))
                sinr_db = effective_sinr_db(
                    args, serving_bs, selected_gain[vehicle], interference_db
                )
                queue_ratio = float(
                    queue_cur[vehicle][0] / queue_upper_bound[vehicle]
                )
                force_initial = learner.transition_state_vector is None
                if should_make_decision(
                    config, sinr_db, queue_ratio, force_initial=force_initial
                ):
                    state = make_state_vector(
                        config,
                        position,
                        float(records[vehicle].get("angle", 0.0)),
                        float(records[vehicle].get("v", 0.0)),
                        serving_bs,
                        sinr_db,
                        learner.last_dql_action == 0,
                        queue_ratio,
                        estimated_load,
                        interference_db,
                        float(vehicle_rate[vehicle] / 1e6),
                        learner.tx_beam,
                        learner.rx_beam,
                    )
                    inference_start = time.perf_counter()
                    command = policy.select_action(
                        state, serving_bs, explore=False
                    )
                    inference_time_record[frame_index] += (
                        time.perf_counter() - inference_start
                    )
                    learner.pending_action = int(command)
                    # This field is used only to distinguish each vehicle's
                    # first forced decision from subsequent trigger events.
                    learner.transition_state_vector = state
                    decision_record[frame_index] += 1
                    tracking_decision_record[frame_index] += int(command == 0)
                else:
                    skipped_trigger_record[frame_index] += 1
                learner.distance_since_event %= config.zone_size_m
            commands[vehicle] = None if command is None else int(command)

        arrivals = collections.OrderedDict()
        for vehicle in sorted(veh_set_cur, key=str):
            arrivals[vehicle] = np.random.poisson(
                vehicle_rate[vehicle] * args.slot_len,
                size=args.slots_per_frame,
            )

        energy_this_frame = 0.0
        pilot_by_slot = np.zeros(args.slots_per_frame)
        for slot_index in range(args.slots_per_frame):
            gain_slot = collections.OrderedDict()
            pilot_slot = collections.OrderedDict()
            for vehicle in sorted(veh_set_cur, key=str):
                learner = learners[vehicle]
                serving_bs = connection[vehicle]
                slot_gains = gain_frame[vehicle].copy()
                if serving_bs > 0 and rician_fading:
                    channel = records[vehicle]["h"] * np.sqrt(
                        rician_channel_gain(
                            args.K_rician, size=records[vehicle]["h"].shape
                        )
                    )
                    slot_gains[serving_bs] = fixed_pair_gain_db(
                        channel,
                        serving_bs - 1,
                        int(learner.tx_beam),
                        int(learner.rx_beam),
                        dft_tx,
                        dft_rx,
                    )
                gain_slot[vehicle] = slot_gains
                pilots = np.full(
                    config.num_micro_bs, config.tracking_pilots, dtype=float
                )
                if serving_bs > 0 and learner.current_sweep_pilots > 0:
                    pilots[serving_bs - 1] = sweep_pilots_for_slot(
                        learner.current_sweep_pilots,
                        config.tracking_pilots,
                        slot_index,
                        args.pilot_overhead_factor,
                    )
                pilot_slot[vehicle] = pilots

            pilot_values = [
                pilot_slot[vehicle][connection[vehicle] - 1]
                if connection[vehicle] > 0
                else 0.0
                for vehicle in veh_set_cur
            ]
            pilot_by_slot[slot_index] = (
                float(np.mean(pilot_values)) if pilot_values else 0.0
            )

            ra_dict = collections.OrderedDict()
            rb_per_bs = np.zeros(len(bs_loc_list), dtype=int)
            for bs_id in range(len(bs_loc_list)):
                bs_ra = ra_func(
                    args,
                    slot_idx=slot_index,
                    BS_id=bs_id,
                    veh_set=bs_association[bs_id],
                    veh_data_rate_dict=vehicle_rate,
                    Q_ub_dict=queue_upper_bound,
                    q_dict=queue_cur,
                    a_dict=arrivals,
                    g_dict=gain_slot,
                    num_pilot_dict=pilot_slot,
                    BS_association_dict=bs_association,
                    infer_g_dict=inference_gain,
                    est_num_RB_allocated_perBS=estimated_rb,
                )
                ra_dict.update(bs_ra)
                rb_per_bs[bs_id] = sum(bs_ra.values())
                power = args.p_macro if bs_id == 0 else args.p_micro
                energy_this_frame += rb_per_bs[bs_id] * power * args.slot_len

            queue_cur = update4slot_vehset_backlog_queue(
                args,
                slot_idx=slot_index,
                RA_dict=ra_dict,
                veh_set=veh_set_cur,
                connection_dict=connection,
                backlog_queue_dict=queue_cur,
                a_dict=arrivals,
                g_dict=gain_slot,
                infer_g_dict=inference_gain,
                num_RB_allocated_perBS=rb_per_bs,
                num_pilot_dict=pilot_slot,
                sinr_flag=True,
            )
            rb_record[frame_index] += rb_per_bs

        queue_per_vehicle[frame_index] = collections.OrderedDict(
            (vehicle, queue_cur[vehicle][1:].copy())
            for vehicle in sorted(veh_set_cur, key=str)
        )
        judgement_count = len(veh_set_cur) * args.slots_per_frame
        violation_count = sum(
            (queue_cur[vehicle][1:] > queue_upper_bound[vehicle]).sum()
            for vehicle in veh_set_cur
        )
        energy_record[frame_index] = energy_this_frame
        handover_record[frame_index] = handovers
        beam_switch_record[frame_index] = beam_switches
        violation_record[frame_index] = violation_count / judgement_count
        average_queue_record[frame_index] = sum(
            queue_cur[vehicle][1:].sum() for vehicle in veh_set_cur
        ) / judgement_count
        pilot_record[frame_index] = pilot_by_slot.mean()
        rb_record[frame_index] /= args.slots_per_frame
        association_record[frame_index] = connection.copy()
        action_record[frame_index] = commands

        frame_prev = frame_cur
        veh_set_prev = veh_set_cur
        queue_prev = queue_cur

    if prt:
        print("DQL-HBT simulation elapsed: {:.1f} s".format(time.time() - sim_start))
    return DQLHBTSimulationResult(
        energy_record=energy_record,
        handover_record=handover_record,
        beam_switch_record=beam_switch_record,
        violation_probability_record=violation_record,
        average_queue_record=average_queue_record,
        pilot_record=pilot_record,
        rb_allocated_record=rb_record,
        queue_per_vehicle_record=queue_per_vehicle,
        association_record=association_record,
        action_record=action_record,
        decision_record=decision_record,
        skipped_trigger_record=skipped_trigger_record,
        tracking_decision_record=tracking_decision_record,
        inference_time_record=inference_time_record,
        full_sweep_record=full_sweep_record,
        local_sweep_record=local_sweep_record,
    )
