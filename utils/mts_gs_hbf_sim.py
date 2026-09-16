"""Exact slot-level evaluation of MTS-GS-HBF-adapted."""

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
from utils.mts_gs_hbf import (
    MTSCommand,
    MTSGSHBFConfig,
    MTSLinkState,
    apply_mts_command,
    association_commands,
    beam_update_command,
)
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
class MTSGSHBFSimulationResult:
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
    trigger_record: np.ndarray
    skipped_gate_record: np.ndarray
    inference_time_record: np.ndarray
    optimizer_time_record: np.ndarray
    optimizer_failure_record: np.ndarray
    optimizer_overflow_record: np.ndarray
    full_sweep_record: np.ndarray
    local_sweep_record: np.ndarray
    association_epoch_record: np.ndarray
    full_sweep_epoch_record: np.ndarray
    local_tracking_epoch_record: np.ndarray
    proposal_record: np.ndarray
    unassigned_record: np.ndarray


def run_sim_mts_gs_hbf(
    args,
    micro_bs_loc_list: Sequence[np.ndarray],
    timeline_dir: MutableMapping,
    config: MTSGSHBFConfig,
    ra_func=RA_OTR_SINR,
    seed: int = 1,
    prt: bool = True,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
    rician_fading: bool = True,
) -> MTSGSHBFSimulationResult:
    """Run the non-learning three-timescale baseline.

    Association and beam commands use the current frame observation and are
    applied at the beginning of the following frame.  Traffic, fading,
    interference, OTR allocation, queue evolution, and energy accounting are
    identical to the other exact-baseline simulators.
    """

    config.validate()
    if len(micro_bs_loc_list) != config.num_micro_bs:
        raise ValueError("micro BS count does not match MTS-GS-HBF config")
    np.random.seed(seed)
    dft_tx = generate_dft_codebook(config.num_tx_beams)
    dft_rx = generate_dft_codebook(config.num_rx_beams)
    macro_loc = np.asarray(macro_bs_loc, dtype=float)
    bs_loc_list = [macro_loc] + [np.asarray(x, dtype=float) for x in micro_bs_loc_list]
    bs_loc_array = np.asarray(bs_loc_list)
    bs_loc_dict = collections.OrderedDict((i, x) for i, x in enumerate(bs_loc_list))
    rb_capacities = np.asarray(
        [args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs,
        dtype=float,
    )

    frame_list = list(timeline_dir.keys())
    if len(frame_list) < 2:
        raise ValueError("timeline must contain at least two frames")
    num_frames = len(frame_list) - 1
    frame_prev = frame_list[0]
    veh_set_prev = set(timeline_dir[frame_prev])
    veh_set_all = set()
    for frame in frame_list:
        veh_set_all.update(timeline_dir[frame])
    vehicle_rate = collections.OrderedDict(
        (
            vehicle,
            args.data_rate
            * np.random.uniform(
                1.0 - args.random_factor_range4data_rate,
                1.0 + args.random_factor_range4data_rate,
            ),
        )
        for vehicle in sorted(veh_set_all, key=str)
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
    states: Dict[object, MTSLinkState] = {
        vehicle: MTSLinkState() for vehicle in veh_set_prev
    }

    energy_record = np.zeros(num_frames)
    handover_record = np.zeros(num_frames)
    beam_switch_record = np.zeros(num_frames)
    violation_record = np.zeros(num_frames)
    average_queue_record = np.zeros(num_frames)
    pilot_record = np.zeros(num_frames)
    rb_record = np.zeros((num_frames, config.num_bs))
    decision_record = np.zeros(num_frames)
    trigger_record = np.zeros(num_frames)
    skipped_gate_record = np.zeros(num_frames)
    inference_time_record = np.zeros(num_frames)
    optimizer_time_record = np.zeros(num_frames)
    optimizer_failure_record = np.zeros(num_frames)
    optimizer_overflow_record = np.zeros(num_frames)
    full_sweep_record = np.zeros(num_frames)
    local_sweep_record = np.zeros(num_frames)
    association_epoch_record = np.zeros(num_frames)
    full_sweep_epoch_record = np.zeros(num_frames)
    local_tracking_epoch_record = np.zeros(num_frames)
    proposal_record = np.zeros(num_frames)
    unassigned_record = np.zeros(num_frames)
    queue_per_vehicle = collections.OrderedDict()
    association_record = collections.OrderedDict()
    action_record = collections.OrderedDict()

    sim_start = time.time()
    iterator = enumerate(frame_list[1:])
    iterator = tqdm.tqdm(
        iterator, total=num_frames, desc="MTS-GS-HBF simulation", disable=not prt
    )
    for frame_index, frame_cur in iterator:
        records = timeline_dir[frame_cur]
        veh_set_cur = set(records)
        veh_set_in = veh_set_cur.difference(veh_set_prev)
        veh_set_remain = veh_set_cur.intersection(veh_set_prev)
        for departed in veh_set_prev.difference(veh_set_cur):
            states.pop(departed, None)
        queue_cur = init4frame_vehset_backlog_queue(
            veh_set_remain,
            veh_set_in,
            queue_prev,
            queue_upper_bound,
            Q_th=0.5,
            slots_per_frame=args.slots_per_frame,
        )
        for vehicle in veh_set_in:
            states[vehicle] = MTSLinkState()

        for vehicle in sorted(veh_set_cur, key=str):
            command = states[vehicle].pending_command
            states[vehicle].pending_command = None
            outcome = apply_mts_command(states[vehicle], command, config)
            handover_record[frame_index] += int(outcome.handover)
            beam_switch_record[frame_index] += int(outcome.beam_switch)
            if outcome.sweep_pilots == config.full_sweep_pilots:
                full_sweep_record[frame_index] += 1
            elif outcome.sweep_pilots > 0:
                local_sweep_record[frame_index] += 1

        ordered_vehicles = sorted(veh_set_cur, key=str)
        connection = collections.OrderedDict(
            (vehicle, int(states[vehicle].action)) for vehicle in ordered_vehicles
        )
        bs_association, _ = update_BS_association_state(bs_loc_dict, connection)
        gain_frame = collections.OrderedDict()
        inference_gain = collections.OrderedDict()
        for vehicle in ordered_vehicles:
            record = records[vehicle]
            state = states[vehicle]
            macro_gain = macro_gain_db(args, record["pos"], macro_loc)
            no_bf_micro = no_bf_gain_db(record["h"])
            gains = np.concatenate(([macro_gain], no_bf_micro.copy()))
            bs = connection[vehicle]
            if bs == 0:
                selected = macro_gain
            else:
                if state.tx_beam is None or state.rx_beam is None:
                    raise RuntimeError("micro link has no MTS-GS-HBF beam pair")
                selected = fixed_pair_gain_db(
                    record["h"],
                    bs - 1,
                    int(state.tx_beam),
                    int(state.rx_beam),
                    dft_tx,
                    dft_rx,
                )
            gains[bs] = selected
            gain_frame[vehicle] = gains
            inference_gain[vehicle] = np.concatenate(([macro_gain], no_bf_micro))

        estimated_rb = estimate_num_RB_allocated_perBS(
            args,
            connection,
            bs_loc_array,
            veh_set_cur,
            gain_frame,
            vehicle_rate,
            infer_g_dict=inference_gain,
        )
        estimated_load = np.clip(estimated_rb / rb_capacities, 0.0, 1.5)

        commands = collections.OrderedDict((vehicle, None) for vehicle in ordered_vehicles)
        association_due = frame_index % config.association_interval_frames == 0
        full_sweep_due = frame_index % config.full_sweep_interval_frames == 0
        local_tracking_due = frame_index % config.local_tracking_interval_frames == 0
        decision_started = time.perf_counter()
        if association_due:
            association_epoch_record[frame_index] = 1
            queue_at_decision = {
                vehicle: float(queue_cur[vehicle][0]) for vehicle in ordered_vehicles
            }
            selected_commands, matching = association_commands(
                args,
                records,
                states,
                queue_at_decision,
                queue_upper_bound,
                vehicle_rate,
                estimated_load,
                config,
                dft_tx,
                dft_rx,
                macro_loc,
            )
            for vehicle, command in selected_commands.items():
                states[vehicle].pending_command = command
                commands[vehicle] = command
            decision_record[frame_index] = len(selected_commands)
            trigger_record[frame_index] = sum(
                int(command.target_bs != connection[vehicle])
                for vehicle, command in selected_commands.items()
            )
            proposal_record[frame_index] = matching.proposal_count
            unassigned_record[frame_index] = len(matching.unassigned)
            optimizer_overflow_record[frame_index] = np.maximum(
                matching.used_capacity - rb_capacities, 0.0
            ).sum()
        elif full_sweep_due or local_tracking_due:
            if full_sweep_due:
                full_sweep_epoch_record[frame_index] = 1
            else:
                local_tracking_epoch_record[frame_index] = 1
            for vehicle in ordered_vehicles:
                command = beam_update_command(
                    records[vehicle],
                    states[vehicle],
                    config,
                    dft_tx,
                    dft_rx,
                    full_sweep=full_sweep_due,
                )
                if command is None:
                    continue
                states[vehicle].pending_command = command
                commands[vehicle] = command
                decision_record[frame_index] += 1
        if association_due or full_sweep_due or local_tracking_due:
            optimizer_time_record[frame_index] = time.perf_counter() - decision_started

        arrivals = collections.OrderedDict(
            (
                vehicle,
                np.random.poisson(
                    vehicle_rate[vehicle] * args.slot_len,
                    size=args.slots_per_frame,
                ),
            )
            for vehicle in ordered_vehicles
        )
        energy_this_frame = 0.0
        pilot_by_slot = np.zeros(args.slots_per_frame)
        for slot_index in range(args.slots_per_frame):
            gain_slot = collections.OrderedDict()
            pilot_slot = collections.OrderedDict()
            for vehicle in ordered_vehicles:
                state = states[vehicle]
                bs = connection[vehicle]
                slot_gains = gain_frame[vehicle].copy()
                if bs > 0 and rician_fading:
                    channel = records[vehicle]["h"] * np.sqrt(
                        rician_channel_gain(
                            args.K_rician, size=records[vehicle]["h"].shape
                        )
                    )
                    slot_gains[bs] = fixed_pair_gain_db(
                        channel,
                        bs - 1,
                        int(state.tx_beam),
                        int(state.rx_beam),
                        dft_tx,
                        dft_rx,
                    )
                gain_slot[vehicle] = slot_gains
                pilots = np.full(
                    config.num_micro_bs, config.tracking_pilots, dtype=float
                )
                if bs > 0 and state.current_sweep_pilots > 0:
                    pilots[bs - 1] = sweep_pilots_for_slot(
                        state.current_sweep_pilots,
                        config.tracking_pilots,
                        slot_index,
                        args.pilot_overhead_factor,
                    )
                pilot_slot[vehicle] = pilots
            pilot_values = [
                pilot_slot[vehicle][connection[vehicle] - 1]
                if connection[vehicle] > 0
                else 0.0
                for vehicle in ordered_vehicles
            ]
            pilot_by_slot[slot_index] = (
                float(np.mean(pilot_values)) if pilot_values else 0.0
            )

            ra_dict = collections.OrderedDict()
            rb_per_bs = np.zeros(config.num_bs, dtype=int)
            for bs_id in range(config.num_bs):
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
            (vehicle, queue_cur[vehicle][1:].copy()) for vehicle in ordered_vehicles
        )
        judgement_count = len(veh_set_cur) * args.slots_per_frame
        violation_count = sum(
            (queue_cur[vehicle][1:] > queue_upper_bound[vehicle]).sum()
            for vehicle in veh_set_cur
        )
        energy_record[frame_index] = energy_this_frame
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
        print("MTS-GS-HBF simulation elapsed: {:.1f} s".format(time.time() - sim_start))
    return MTSGSHBFSimulationResult(
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
        trigger_record=trigger_record,
        skipped_gate_record=skipped_gate_record,
        inference_time_record=inference_time_record,
        optimizer_time_record=optimizer_time_record,
        optimizer_failure_record=optimizer_failure_record,
        optimizer_overflow_record=optimizer_overflow_record,
        full_sweep_record=full_sweep_record,
        local_sweep_record=local_sweep_record,
        association_epoch_record=association_epoch_record,
        full_sweep_epoch_record=full_sweep_epoch_record,
        local_tracking_epoch_record=local_tracking_epoch_record,
        proposal_record=proposal_record,
        unassigned_record=unassigned_record,
    )

