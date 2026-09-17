"""Exact slot-level evaluation of a frozen O-MAPPO policy."""

from __future__ import annotations

import collections
import dataclasses
import time
from typing import Dict, MutableMapping, Optional, Sequence

import numpy as np
import tqdm

from utils.alg_utils import (
    RA_OTR_SINR,
    estimate_num_RB_allocated_perBS,
    update_BS_association_state,
)
from utils.beam_utils import generate_dft_codebook
from utils.channel_utils import rician_channel_gain
from utils.dql_hbt import effective_sinr_db
from utils.o_mappo import (
    OMAPPOActionOutcome,
    OMAPPOCommand,
    OMAPPOLearnerState,
    OMAPPPolicy,
    apply_o_mappo_command,
    append_state_sequence,
    candidate_feasibility_context,
    make_global_state,
    make_local_state,
    optimize_triggered_targets,
    record_optimizer_feedback,
    shared_actor_inputs,
    source_gate_allows,
)
from utils.pql_ba import (
    best_beam_pair,
    fixed_pair_gain_db,
    macro_gain_db,
    no_bf_gain_db,
    sweep_pilots_for_slot,
)
from utils.pql_ba_adapted import _capacity_per_rb_bps, _interference_db
from utils.ho_utils import interruption_slots
from utils.queue_utils import (
    init4frame_vehset_backlog_queue,
    init_vehset_backlog_queue,
    update4slot_vehset_backlog_queue,
)


@dataclasses.dataclass
class OMAPPOSimulationResult:
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


def run_sim_o_mappo(
    args,
    micro_bs_loc_list: Sequence[np.ndarray],
    timeline_dir: MutableMapping,
    policy: OMAPPPolicy,
    ra_func=RA_OTR_SINR,
    seed: int = 1,
    prt: bool = True,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
    rician_fading: bool = True,
    optimizer_solver: Optional[str] = "milp",
    traffic_trace=None,
    ho_interruption_ms: float = 0.0,
    paired_fading_seed=None,
    physics_device=None,
    progress_callback=None,
    optimizer_input_hook=None,
) -> OMAPPOSimulationResult:
    """Evaluate O-MAPPO with causal commands and the common exact scheduler.

    ``optimizer_input_hook`` is an opt-in diagnostic interface. It transforms
    only the target optimizer's inputs after the frozen actor has acted;
    the default simulation and actor observations are unchanged.
    """

    config = policy.config
    ho_slots = interruption_slots(ho_interruption_ms, args.slot_len, args.slots_per_frame)
    # Apply the same interruption-aware target capacities to both interfaces.
    config = dataclasses.replace(config, ho_interruption_ms=ho_interruption_ms)
    shared = config.information_mode == "shared_prediction"
    if paired_fading_seed is not None and traffic_trace is None:
        raise ValueError("Paired fading requires independently replayed traffic")
    if physics_device is not None and (not rician_fading or paired_fading_seed is None):
        raise ValueError("GPU physics requires paired Rician fading")
    if len(micro_bs_loc_list) != config.num_micro_bs:
        raise ValueError("micro BS count does not match O-MAPPO policy")
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
    frame_duration = args.slots_per_frame * args.slot_len
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
    if traffic_trace is not None:
        vehicle_rate.update(traffic_trace["rates"])
        queue_upper_bound.update({v: args.lat_slot_ub * r * args.slot_len
                                  for v, r in vehicle_rate.items()})
    queue_prev = init_vehset_backlog_queue(
        veh_set_prev,
        queue_upper_bound,
        Q_th=0.5,
        slots_per_frame=args.slots_per_frame,
    )
    learners: Dict[object, OMAPPOLearnerState] = {}
    if traffic_trace is not None:
        for vehicle in veh_set_prev:
            queue_prev[vehicle][:] = traffic_trace["initial_queues"][frame_prev][vehicle]
    for vehicle in veh_set_prev:
        position = np.asarray(timeline_dir[frame_prev][vehicle]["pos"], dtype=float)
        learners[vehicle] = OMAPPOLearnerState(
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
    queue_per_vehicle = collections.OrderedDict()
    association_record = collections.OrderedDict()
    action_record = collections.OrderedDict()
    previous_throughput_ratio = 0.0
    global_state_history = []

    sim_start = time.time()
    iterator = enumerate(frame_list[1:])
    iterator = tqdm.tqdm(
        iterator, total=num_frames, desc="O-MAPPO simulation", disable=not prt
    )
    for frame_index, frame_cur in iterator:
        records = timeline_dir[frame_cur]
        veh_set_cur = set(records)
        veh_set_in = veh_set_cur.difference(veh_set_prev)
        veh_set_remain = veh_set_cur.intersection(veh_set_prev)
        for departed in veh_set_prev.difference(veh_set_cur):
            learners.pop(departed, None)
        queue_cur = init4frame_vehset_backlog_queue(
            veh_set_remain,
            veh_set_in,
            queue_prev,
            queue_upper_bound,
            Q_th=0.5,
            slots_per_frame=args.slots_per_frame,
        )
        for vehicle in veh_set_in:
            if traffic_trace is not None:
                queue_cur[vehicle][0] = traffic_trace["initial_queues"][frame_cur][vehicle]
            position = np.asarray(records[vehicle]["pos"], dtype=float)
            learners[vehicle] = OMAPPOLearnerState(
                action=0,
                rx_beam=None,
                pending_action=None,
                last_position=position.copy(),
                distance_since_event=config.zone_size_m,
                tx_beam=None,
            )

        outcomes: Dict[object, OMAPPOActionOutcome] = {}
        for vehicle in sorted(veh_set_cur, key=str):
            learner = learners[vehicle]
            command = learner.pending_command
            learner.pending_command = None
            outcome = apply_o_mappo_command(
                learner, command, records[vehicle], config, dft_tx, dft_rx
            )
            outcomes[vehicle] = outcome
            if outcome.sweep_pilots == config.full_sweep_pilots:
                full_sweep_record[frame_index] += 1
            elif outcome.sweep_pilots > 0:
                local_sweep_record[frame_index] += 1
        handover_record[frame_index] = sum(x.handover for x in outcomes.values())
        beam_switch_record[frame_index] = sum(x.beam_switch for x in outcomes.values())

        connection = collections.OrderedDict(
            (vehicle, int(learners[vehicle].action))
            for vehicle in sorted(veh_set_cur, key=str)
        )
        bs_association, user_load_dict = update_BS_association_state(
            bs_loc_dict, connection
        )
        user_load = np.asarray(
            [user_load_dict[index] for index in range(config.num_bs)], dtype=float
        )
        gain_frame = collections.OrderedDict()
        inference_gain = collections.OrderedDict()
        serving_gain: Dict[object, float] = {}
        for vehicle in sorted(veh_set_cur, key=str):
            record = records[vehicle]
            learner = learners[vehicle]
            macro_gain = macro_gain_db(args, record["pos"], macro_loc)
            no_bf_micro = no_bf_gain_db(record["h"])
            gains = np.concatenate(([macro_gain], no_bf_micro.copy()))
            bs = connection[vehicle]
            if bs == 0:
                selected = macro_gain
            else:
                if learner.tx_beam is None or learner.rx_beam is None:
                    raise RuntimeError("micro link has no O-MAPPO beam pair")
                selected = fixed_pair_gain_db(
                    record["h"],
                    bs - 1,
                    int(learner.tx_beam),
                    int(learner.rx_beam),
                    dft_tx,
                    dft_rx,
                )
            gains[bs] = selected
            gain_frame[vehicle] = gains
            inference_gain[vehicle] = np.concatenate(([macro_gain], no_bf_micro))
            serving_gain[vehicle] = selected

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
        if shared:
            # Public, previously realized RB use; not an estimate obtained
            # by reading all current, unobserved channel matrices.
            estimated_load = (rb_record[frame_index - 1] / rb_capacities
                              if frame_index else np.zeros(config.num_bs))
        zone_due: Dict[object, bool] = {}
        for vehicle in sorted(veh_set_cur, key=str):
            learner = learners[vehicle]
            position = np.asarray(records[vehicle]["pos"], dtype=float)
            learner.distance_since_event += float(
                np.linalg.norm(position - learner.last_position)
            )
            learner.last_position = position.copy()
            zone_due[vehicle] = bool(
                learner.distance_since_event + 1e-9 >= config.zone_size_m
            )
            if zone_due[vehicle]:
                learner.distance_since_event %= config.zone_size_m

        current_allocated: Dict[object, float] = {}
        serving_interference: Dict[object, float] = {}
        all_states: Dict[object, np.ndarray] = {}
        all_alternative_sinr: Dict[object, list] = {}
        for vehicle in sorted(veh_set_cur, key=str):
            bs = connection[vehicle]
            interference = _interference_db(
                args, bs, inference_gain[vehicle], estimated_load
            )
            serving_interference[vehicle] = interference
            capacity = _capacity_per_rb_bps(
                args,
                bs,
                serving_gain[vehicle],
                interference,
                config.tracking_pilots if bs > 0 else 0.0,
            )
            backlog = queue_cur[vehicle][0] + vehicle_rate[vehicle] * frame_duration
            requested = backlog / max(capacity * frame_duration, 1e-12)
            current_allocated[vehicle] = min(
                requested, rb_capacities[bs]
            )
            sinr = effective_sinr_db(
                args, bs, serving_gain[vehicle], interference
            )
            alternatives = []
            if zone_due[vehicle] and config.trigger_gate == "source":
                for target in range(config.num_bs):
                    if target == bs:
                        continue
                    if target == 0:
                        gain = inference_gain[vehicle][0]
                        target_interference = -np.inf
                    else:
                        _, _, gain = best_beam_pair(
                            records[vehicle]["h"], target - 1, dft_tx, dft_rx
                        )
                        target_interference = _interference_db(
                            args, target, inference_gain[vehicle], estimated_load
                        )
                    alternatives.append(
                        effective_sinr_db(
                            args, target, gain, target_interference
                        )
                    )
            all_alternative_sinr[vehicle] = alternatives
            state_kwargs = shared_actor_inputs(config, records[vehicle])
            if config.state_variant == "feasibility":
                context = candidate_feasibility_context(
                    args,
                    records[vehicle],
                    learners[vehicle],
                    backlog,
                    estimated_load,
                    config,
                    dft_tx,
                    dft_rx,
                    macro_loc,
                )
                state_kwargs = {
                    "candidate_sinr_db": context[0],
                    "candidate_demand_ratio": context[1],
                    "candidate_residual_ratio": context[2],
                    "candidate_feasibility_margin": context[3],
                    "optimizer_feedback": learners[vehicle].optimizer_feedback,
                }
            all_states[vehicle] = make_local_state(
                config,
                records[vehicle]["pos"],
                float(records[vehicle].get("angle", 0.0)),
                float(records[vehicle].get("v", 0.0)),
                bs,
                sinr,
                float(queue_cur[vehicle][0] / queue_upper_bound[vehicle]),
                float(vehicle_rate[vehicle] / 1e6),
                estimated_load,
                user_load,
                interference,
                learners[vehicle].last_handover,
                previous_throughput_ratio,
                learners[vehicle].previous_rb_fraction,
                learners[vehicle].tx_beam,
                learners[vehicle].rx_beam,
                **state_kwargs,
            )
        # Convert stand-alone RB demands into a per-BS allocation estimate.
        # This prevents the target optimizer from counting every fixed UE as
        # if it alone occupied the full BS bandwidth.
        for bs in range(config.num_bs):
            associated = [x for x in veh_set_cur if connection[x] == bs]
            total_requested = sum(current_allocated[x] for x in associated)
            target_total = min(float(estimated_rb[bs]), rb_capacities[bs])
            scale = target_total / max(total_requested, 1e-12)
            for vehicle in associated:
                current_allocated[vehicle] *= min(scale, 1.0)
        if shared:
            current_allocated = {
                v: learners[v].previous_rb_fraction * rb_capacities[connection[v]]
                for v in veh_set_cur
            }
        ordered_vehicles = sorted(veh_set_cur, key=str)
        global_state = make_global_state(
            np.stack([all_states[x] for x in ordered_vehicles]), len(ordered_vehicles)
        )
        event_vehicles = []
        for vehicle in ordered_vehicles:
            learner = learners[vehicle]
            if not zone_due[vehicle]:
                continue
            bs = connection[vehicle]
            sinr = effective_sinr_db(
                args,
                bs,
                serving_gain[vehicle],
                serving_interference[vehicle],
            )
            if source_gate_allows(
                config, sinr, all_alternative_sinr[vehicle]
            ):
                event_vehicles.append(vehicle)
            else:
                skipped_gate_record[frame_index] += 1

        commands = collections.OrderedDict((vehicle, None) for vehicle in ordered_vehicles)
        if event_vehicles:
            inference_started = time.perf_counter()
            if config.recurrent:
                local_batch = np.stack(
                    [
                        append_state_sequence(
                            learners[x].state_history,
                            all_states[x],
                            config.recurrent_sequence_length,
                        )
                        for x in event_vehicles
                    ]
                )
                policy_global_state = append_state_sequence(
                    global_state_history,
                    global_state,
                    config.recurrent_sequence_length,
                )
            else:
                local_batch = np.stack([all_states[x] for x in event_vehicles])
                policy_global_state = global_state
            actions, _, _ = policy.act(
                local_batch,
                policy_global_state,
                explore=False,
            )
            inference_time_record[frame_index] = time.perf_counter() - inference_started
            triggered = [
                vehicle
                for vehicle, action in zip(event_vehicles, actions)
                if int(action) == 1
            ]
            backlog = {
                vehicle: queue_cur[vehicle][0]
                + vehicle_rate[vehicle] * frame_duration
                for vehicle in veh_set_cur
            }
            optimizer_inputs = dict(records=records, allocated_rb=current_allocated,
                                    load=estimated_load, config=config)
            if optimizer_input_hook is not None:
                optimizer_inputs = optimizer_input_hook(
                    args=args, frame=frame_cur, records=records, learners=learners,
                    backlog=backlog, allocated_rb=current_allocated,
                    load=estimated_load, config=config, dft_tx=dft_tx,
                    dft_rx=dft_rx, macro_loc=macro_loc,
                    legacy_rb=estimated_rb, serving_gain=serving_gain,
                    no_bf_gain=inference_gain,
                )
            optimization = optimize_triggered_targets(
                args,
                optimizer_inputs["records"],
                learners,
                triggered,
                backlog,
                optimizer_inputs["allocated_rb"],
                optimizer_inputs["load"],
                optimizer_inputs["config"],
                dft_tx,
                dft_rx,
                macro_loc,
                solver=optimizer_solver,
            )
            optimizer_time_record[frame_index] = optimization.elapsed_s
            optimizer_failure_record[frame_index] = int(
                not optimization.solver_success
            )
            optimizer_overflow_record[frame_index] = optimization.overflow.sum()
            record_optimizer_feedback(
                args, learners, event_vehicles, actions, optimization
            )
            for vehicle, action in zip(event_vehicles, actions):
                binary = int(action)
                target = (
                    optimization.targets[vehicle]
                    if binary
                    else int(learners[vehicle].action)
                )
                command = OMAPPOCommand(binary, target)
                learners[vehicle].pending_command = command
                commands[vehicle] = command
            decision_record[frame_index] = len(event_vehicles)
            trigger_record[frame_index] = len(triggered)

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
        if traffic_trace is not None:
            arrivals = collections.OrderedDict((v, traffic_trace["arrivals"][frame_cur][v])
                                               for v in ordered_vehicles)
        initial_queue = {vehicle: float(queue_cur[vehicle][0]) for vehicle in ordered_vehicles}
        energy_this_frame = 0.0
        pilot_by_slot = np.zeros(args.slots_per_frame)
        rb_by_vehicle = collections.defaultdict(float)
        gpu_gains = None
        if physics_device is not None:
            from utils.gpu_phy import GPUFramePHY
            physical = GPUFramePHY(args, records, frame_cur, paired_fading_seed, physics_device)
            gpu_gains = physical.fixed_pairs(connection, learners)
            del physical
        for slot_index in range(args.slots_per_frame):
            blocked = {v for v in ordered_vehicles if outcomes[v].handover and slot_index < ho_slots}
            if paired_fading_seed is not None and gpu_gains is None:
                np.random.seed((int(paired_fading_seed) * 1000003 +
                                round(float(frame_cur) * 10) * args.slots_per_frame + slot_index) % 2**32)
            gain_slot = collections.OrderedDict()
            pilot_slot = collections.OrderedDict()
            for vehicle_index, vehicle in enumerate(ordered_vehicles):
                learner = learners[vehicle]
                bs = connection[vehicle]
                slot_gains = gain_frame[vehicle].copy()
                if gpu_gains is not None:
                    if bs > 0:
                        slot_gains[bs] = gpu_gains[slot_index, vehicle_index]
                elif rician_fading and (bs > 0 or paired_fading_seed is not None):
                    channel = records[vehicle]["h"] * np.sqrt(
                        rician_channel_gain(
                            args.K_rician, size=records[vehicle]["h"].shape
                        )
                    )
                    if bs > 0:
                        slot_gains[bs] = fixed_pair_gain_db(
                            channel,
                            bs - 1,
                            int(learner.tx_beam),
                            int(learner.rx_beam),
                            dft_tx,
                            dft_rx,
                        )
                gain_slot[vehicle] = slot_gains
                pilots = np.full(
                    config.num_micro_bs, config.tracking_pilots, dtype=float
                )
                if bs > 0 and learner.current_sweep_pilots > 0:
                    pilots[bs - 1] = sweep_pilots_for_slot(
                        learner.current_sweep_pilots,
                        config.tracking_pilots,
                        max(0, slot_index - (ho_slots if outcomes[vehicle].handover else 0)),
                        args.pilot_overhead_factor,
                    )
                if vehicle in blocked:
                    pilots[:] = 0
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
            ra_dict = collections.OrderedDict((v, 0) for v in blocked)
            rb_per_bs = np.zeros(config.num_bs, dtype=int)
            for bs_id in range(config.num_bs):
                bs_ra = ra_func(
                    args,
                    slot_idx=slot_index,
                    BS_id=bs_id,
                    veh_set=[v for v in bs_association[bs_id] if v not in blocked],
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
                for vehicle, rb in bs_ra.items():
                    rb_by_vehicle[vehicle] += rb
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

        total_offered = 0.0
        total_served = 0.0
        for vehicle in ordered_vehicles:
            offered = float(arrivals[vehicle].sum())
            served = max(
                0.0,
                initial_queue[vehicle] + offered - float(queue_cur[vehicle][-1]),
            )
            total_offered += offered
            total_served += min(served, initial_queue[vehicle] + offered)
            bs = connection[vehicle]
            learners[vehicle].previous_rb_fraction = (
                rb_by_vehicle[vehicle]
                / args.slots_per_frame
                / rb_capacities[bs]
            )
        previous_throughput_ratio = total_served / max(total_offered, 1e-12)
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
        if progress_callback is not None:
            progress_callback(frame_index + 1, num_frames)

    if prt:
        print("O-MAPPO simulation elapsed: {:.1f} s".format(time.time() - sim_start))
    return OMAPPOSimulationResult(
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
    )
