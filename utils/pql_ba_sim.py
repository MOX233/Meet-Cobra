"""Loaded-network evaluation for a frozen adapted PQL-BA policy."""

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
from utils.pql_ba import (
    PQLBAPolicy,
    _LearnerState,
    _apply_pending_action,
    _learner_gain_db,
    action_serving_bs,
    action_to_link,
    fixed_pair_gain_db,
    macro_gain_db,
    no_bf_gain_db,
    sweep_pilots_for_slot,
)
from utils.mox_utils import dB2lin, lin2dB
from utils.queue_utils import (
    init4frame_vehset_backlog_queue,
    init_vehset_backlog_queue,
    update4slot_vehset_backlog_queue,
)


@dataclasses.dataclass
class PQLBASimulationResult:
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
    known_decision_record: np.ndarray


def run_sim_pql_ba(
    args,
    micro_bs_loc_list: Sequence[np.ndarray],
    timeline_dir: MutableMapping,
    policy: PQLBAPolicy,
    ra_func=RA_OTR_SINR,
    seed: int = 1,
    prt: bool = True,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
    rician_fading: bool = True,
) -> PQLBASimulationResult:
    """Evaluate PQL-BA with the common traffic, interference, queue, and RA model.

    The Q table is never updated here.  A distance-zone event produces a
    command from causally observed state in frame x, and the command is applied
    in frame x+1, matching the manuscript's frame-level re-association timing.
    OTR-RA is then executed independently at every slot, exactly as for the
    proposed scheme.
    """

    config = policy.config
    if len(micro_bs_loc_list) != config.num_micro_bs:
        raise ValueError("micro BS count does not match the PQL-BA policy")
    rng = np.random.default_rng(seed)
    # The common simulator uses the legacy global NumPy RNG for arrivals and
    # Rician factors.  Seed it explicitly so strategies see reproducible draws.
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

    learners: Dict[object, _LearnerState] = {}
    for vehicle in veh_set_prev:
        position = np.asarray(timeline_dir[frame_prev][vehicle]["pos"], dtype=float)
        learners[vehicle] = _LearnerState(
            action=0,
            rx_beam=None,
            pending_action=None,
            last_position=position.copy(),
            distance_since_event=config.zone_size_m,
        )

    energy_record = np.zeros(num_frames)
    handover_record = np.zeros(num_frames)
    beam_switch_record = np.zeros(num_frames)
    violation_record = np.zeros(num_frames)
    average_queue_record = np.zeros(num_frames)
    pilot_record = np.zeros(num_frames)
    rb_record = np.zeros((num_frames, len(bs_loc_list)))
    decision_record = np.zeros(num_frames)
    known_decision_record = np.zeros(num_frames)
    queue_per_vehicle = collections.OrderedDict()
    association_record = collections.OrderedDict()
    action_record = collections.OrderedDict()

    sim_start = time.time()
    iterator = enumerate(frame_list[1:])
    iterator = tqdm.tqdm(iterator, total=num_frames, desc="PQL-BA simulation", disable=not prt)
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
            learners[vehicle] = _LearnerState(
                action=0,
                rx_beam=None,
                pending_action=None,
                last_position=position.copy(),
                distance_since_event=config.zone_size_m,
            )

        receiver_sweep: Dict[object, bool] = {}
        handovers = 0
        beam_switches = 0
        for vehicle in sorted(veh_set_cur, key=str):
            learner = learners[vehicle]
            old_bs = action_serving_bs(learner.action, config)
            decision_applied = learner.pending_action is not None
            changed = _apply_pending_action(
                learner, records[vehicle], config, dft_tx, dft_rx
            )
            new_bs = action_serving_bs(learner.action, config)
            receiver_sweep[vehicle] = decision_applied and new_bs > 0
            if changed:
                beam_switches += 1
                if old_bs != new_bs:
                    handovers += 1

        connection = collections.OrderedDict(
            (vehicle, action_serving_bs(learners[vehicle].action, config))
            for vehicle in sorted(veh_set_cur, key=str)
        )
        bs_association, _ = update_BS_association_state(bs_loc_dict, connection)

        gain_frame = collections.OrderedDict()
        inference_gain = collections.OrderedDict()
        selected_gain = {}
        for vehicle in sorted(veh_set_cur, key=str):
            record = records[vehicle]
            learner = learners[vehicle]
            macro_gain = macro_gain_db(args, record["pos"], macro_loc)
            micro_no_bf = no_bf_gain_db(record["h"])
            gains = np.concatenate(([macro_gain], micro_no_bf.copy()))
            gain = _learner_gain_db(
                args, learner, record, macro_loc, config, dft_tx, dft_rx
            )
            serving_bs = connection[vehicle]
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
            [args.num_RB_macro]
            + [args.num_RB_micro] * config.num_micro_bs,
            dtype=float,
        )
        estimated_load = np.clip(estimated_rb / rb_capacities, 0.0, 1.0)

        # Generate decisions after observing the current serving-link RSSI.
        commands = collections.OrderedDict()
        for vehicle in sorted(veh_set_cur, key=str):
            learner = learners[vehicle]
            current_position = np.asarray(records[vehicle]["pos"], dtype=float)
            learner.distance_since_event += float(
                np.linalg.norm(current_position - learner.last_position)
            )
            learner.last_position = current_position.copy()
            if learner.distance_since_event + 1e-9 >= config.zone_size_m:
                serving_bs = connection[vehicle]
                if serving_bs == 0:
                    interference_db = -np.inf
                else:
                    interference_w = sum(
                        dB2lin(inference_gain[vehicle][other_bs])
                        * args.p_micro
                        * estimated_load[other_bs]
                        for other_bs in range(1, config.num_micro_bs + 1)
                        if other_bs != serving_bs
                    )
                    noise_w = (
                        args.N0
                        * args.RB_intervel_micro
                        * dB2lin(args.NF_micro_dB)
                    )
                    interference_db = float(lin2dB(interference_w / noise_w))
                state = policy.make_state(
                    selected_gain[vehicle],
                    learner.action,
                    float(records[vehicle].get("angle", 0.0)),
                    position=current_position,
                    queue_ratio=float(
                        queue_cur[vehicle][0] / queue_upper_bound[vehicle]
                    ),
                    load_ratio=float(estimated_load[serving_bs]),
                    interference_db=interference_db,
                    traffic_mbps=float(vehicle_rate[vehicle] / 1e6),
                )
                decision_record[frame_index] += 1
                if state in policy.q_table:
                    known_decision_record[frame_index] += 1
                learner.pending_action = policy.select_action(state, rng, explore=False)
                learner.distance_since_event %= config.zone_size_m
            commands[vehicle] = int(
                learner.pending_action if learner.pending_action is not None else learner.action
            )

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
                serving_bs, tx_beam = action_to_link(learner.action, config)
                if config.hierarchical_bs_action:
                    tx_beam = learner.tx_beam
                slot_gains = gain_frame[vehicle].copy()
                if serving_bs > 0 and rician_fading:
                    channel = records[vehicle]["h"] * np.sqrt(
                        rician_channel_gain(args.K_rician, size=records[vehicle]["h"].shape)
                    )
                    slot_gains[serving_bs] = fixed_pair_gain_db(
                        channel,
                        serving_bs - 1,
                        int(tx_beam),
                        int(learner.rx_beam),
                        dft_tx,
                        dft_rx,
                    )
                gain_slot[vehicle] = slot_gains
                pilots = np.full(config.num_micro_bs, config.tracking_pilots, dtype=float)
                if serving_bs > 0 and receiver_sweep[vehicle]:
                    pilots[serving_bs - 1] = sweep_pilots_for_slot(
                        config.receiver_sweep_pilots,
                        config.tracking_pilots,
                        slot_index,
                        args.pilot_overhead_factor,
                    )
                pilot_slot[vehicle] = pilots

            # Match ``run_sim_withUMa``: macro-associated vehicles contribute
            # zero to the network-wide per-vehicle pilot average.
            pilot_values = [
                pilot_slot[vehicle][connection[vehicle] - 1]
                if connection[vehicle] > 0
                else 0.0
                for vehicle in veh_set_cur
            ]
            pilot_by_slot[slot_index] = float(np.mean(pilot_values)) if pilot_values else 0.0

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

        if prt and frame_index + 1 == num_frames:
            elapsed = time.time() - sim_start
            print("PQL-BA simulation elapsed: {:.1f} s".format(elapsed))

    return PQLBASimulationResult(
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
        known_decision_record=known_decision_record,
    )
