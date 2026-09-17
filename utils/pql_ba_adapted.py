"""Queue/load/interference/energy-aware adaptations of PQL-BA.

The exact slot-level simulator is too expensive to place inside every Q-learning
update.  This module therefore uses a deterministic frame-level fluid surrogate
for training and hyperparameter validation.  It retains the same association,
beam, interference, RB-capacity, pilot-overhead, traffic, and normalized-queue
quantities as the manuscript simulator.  A frozen policy is still evaluated by
``run_sim_pql_ba`` under the exact slot-level OTR-RA model.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from utils.beam_utils import generate_dft_codebook
from utils.mox_utils import dB2lin, lin2dB
from utils.pql_ba import (
    PQLBAConfig,
    PQLBAPolicy,
    _LearnerState,
    _apply_pending_action,
    _learner_gain_db,
    action_serving_bs,
    no_bf_gain_db,
    sweep_pilots_for_slot,
)


@dataclasses.dataclass(frozen=True)
class AdaptedRewardConfig:
    """Dimensionless per-frame multi-objective reward weights."""

    name: str
    reward_offset: float = 10.0
    service_weight: float = 1.0
    queue_weight: float = 0.5
    violation_weight: float = 4.0
    energy_weight: float = 0.0
    load_weight: float = 0.0
    handover_weight: float = 0.0
    beam_switch_weight: float = 0.0
    queue_penalty_cap: float = 10.0
    service_reward_cap: float = 2.0

    def validate(self) -> None:
        values = dataclasses.asdict(self)
        for key, value in values.items():
            if key == "name":
                continue
            if float(value) < 0.0:
                raise ValueError("{} must be nonnegative".format(key))


@dataclasses.dataclass
class _FluidStep:
    queue_end: Dict[object, float]
    served_bits: Dict[object, float]
    user_power_w: Dict[object, float]
    serving_gain_db: Dict[object, float]
    interference_db: Dict[object, float]
    load_ratio: np.ndarray
    connection: Dict[object, int]


def contextual_config(
    zone_size_m: float = 10.0,
    location_bin_size_m: Optional[float] = 100.0,
    epsilon_decay_decisions: float = 1.5e5,
    include_traffic_state: bool = False,
    hierarchical_bs_action: bool = False,
) -> PQLBAConfig:
    """Return the common state design used by PQL-BA-adapted variants."""

    return PQLBAConfig(
        zone_size_m=zone_size_m,
        include_heading=True,
        num_heading_bins=4,
        location_bin_size_m=location_bin_size_m,
        include_queue_state=True,
        include_load_state=True,
        include_interference_state=True,
        include_traffic_state=include_traffic_state,
        hierarchical_bs_action=hierarchical_bs_action,
        receiver_sweep_pilots=(32 * 8 if hierarchical_bs_action else 8),
        epsilon_decay_decisions=epsilon_decay_decisions,
    )


def reward_presets() -> "collections.OrderedDict[str, AdaptedRewardConfig]":
    """Candidate rewards, from QoS-only to progressively energy-aware."""

    return collections.OrderedDict(
        (
            ("qos", AdaptedRewardConfig(name="qos", energy_weight=0.0)),
            (
                "qos_energy_005",
                AdaptedRewardConfig(name="qos_energy_005", energy_weight=0.05),
            ),
            (
                "qos_energy_020",
                AdaptedRewardConfig(name="qos_energy_020", energy_weight=0.20),
            ),
            (
                "qos_energy_050",
                AdaptedRewardConfig(name="qos_energy_050", energy_weight=0.50),
            ),
            (
                "qos_strong",
                AdaptedRewardConfig(
                    name="qos_strong",
                    reward_offset=20.0,
                    queue_weight=1.0,
                    violation_weight=8.0,
                ),
            ),
            (
                "qos_strong_energy_020",
                AdaptedRewardConfig(
                    name="qos_strong_energy_020",
                    reward_offset=25.0,
                    queue_weight=1.0,
                    violation_weight=8.0,
                    energy_weight=0.20,
                ),
            ),
            (
                "delay_strong_energy_010",
                AdaptedRewardConfig(
                    name="delay_strong_energy_010",
                    reward_offset=25.0,
                    queue_weight=2.0,
                    violation_weight=10.0,
                    energy_weight=0.10,
                    queue_penalty_cap=5.0,
                ),
            ),
            (
                "qos_load2_energy_020",
                AdaptedRewardConfig(
                    name="qos_load2_energy_020",
                    reward_offset=15.0,
                    energy_weight=0.20,
                    load_weight=2.0,
                ),
            ),
            (
                "qos_load5_energy_020",
                AdaptedRewardConfig(
                    name="qos_load5_energy_020",
                    reward_offset=20.0,
                    energy_weight=0.20,
                    load_weight=5.0,
                ),
            ),
        )
    )


def _interference_db(
    args,
    serving_bs: int,
    no_bf_gain_db_vector: np.ndarray,
    load_ratio: np.ndarray,
) -> float:
    if serving_bs == 0:
        return -np.inf
    interference_w = sum(
        dB2lin(no_bf_gain_db_vector[other_bs])
        * args.p_micro
        * load_ratio[other_bs]
        for other_bs in range(1, len(load_ratio))
        if other_bs != serving_bs
    )
    noise_w = args.N0 * args.RB_intervel_micro * dB2lin(args.NF_micro_dB)
    return float(lin2dB(interference_w / noise_w))


def _capacity_per_rb_bps(
    args,
    serving_bs: int,
    serving_gain_db: float,
    interference_db: float,
    pilot_average: float,
) -> float:
    if serving_bs == 0:
        bandwidth = args.RB_intervel_macro
        power = args.p_macro
        noise_figure_db = args.NF_macro_dB
        interference_w = 0.0
        useful_fraction = 1.0
    else:
        bandwidth = args.RB_intervel_micro
        power = args.p_micro
        noise_figure_db = args.NF_micro_dB
        noise_w = args.N0 * bandwidth * dB2lin(noise_figure_db)
        interference_w = dB2lin(interference_db) * noise_w
        useful_fraction = max(
            0.0, 1.0 - pilot_average * args.pilot_overhead_factor
        )
    noise_w = args.N0 * bandwidth * dB2lin(noise_figure_db)
    sinr = power * dB2lin(serving_gain_db) / (noise_w + interference_w)
    return float(useful_fraction * bandwidth * np.log2(1.0 + sinr))


def _fluid_allocation(
    args,
    vehicles: Sequence[object],
    connection: Dict[object, int],
    serving_gain: Dict[object, float],
    no_bf_gain: Dict[object, np.ndarray],
    pilot_average: Dict[object, float],
    backlog_bits: Dict[object, float],
    initial_load: np.ndarray,
    frame_duration_s: float,
    iterations: int = 5,
    service_fraction: Optional[Dict[object, float]] = None,
) -> Tuple[Dict[object, float], Dict[object, float], np.ndarray, Dict[object, float]]:
    """Fixed-point fluid approximation of OTR allocation and interference."""

    num_bs = len(initial_load)
    rb_capacities = np.asarray(
        [args.num_RB_macro] + [args.num_RB_micro] * (num_bs - 1), dtype=float
    )
    load = np.clip(np.asarray(initial_load, dtype=float), 0.0, 1.0)
    allocation: Dict[object, float] = {}
    capacity: Dict[object, float] = {}
    interference: Dict[object, float] = {}

    for _ in range(iterations):
        for vehicle in vehicles:
            bs_id = connection[vehicle]
            interference[vehicle] = _interference_db(
                args, bs_id, no_bf_gain[vehicle], load
            )
            capacity[vehicle] = _capacity_per_rb_bps(
                args,
                bs_id,
                serving_gain[vehicle],
                interference[vehicle],
                pilot_average[vehicle],
            )
            if service_fraction is not None:
                capacity[vehicle] *= service_fraction[vehicle]

        allocation = {}
        next_load = np.zeros(num_bs, dtype=float)
        for bs_id in range(num_bs):
            associated = [vehicle for vehicle in vehicles if connection[vehicle] == bs_id]
            associated.sort(key=lambda vehicle: capacity[vehicle], reverse=True)
            remaining = rb_capacities[bs_id]
            for vehicle in associated:
                if capacity[vehicle] <= 1e-9:
                    requested = remaining
                else:
                    requested = backlog_bits[vehicle] / (
                        capacity[vehicle] * frame_duration_s
                    )
                allocated = min(max(requested, 0.0), remaining)
                allocation[vehicle] = allocated
                remaining -= allocated
            next_load[bs_id] = (
                sum(allocation.get(vehicle, 0.0) for vehicle in associated)
                / rb_capacities[bs_id]
            )
        if np.allclose(load, next_load, atol=1e-3, rtol=1e-3):
            load = next_load
            break
        load = next_load

    # Recompute capacities and the allocation once at the converged load.
    for vehicle in vehicles:
        bs_id = connection[vehicle]
        interference[vehicle] = _interference_db(
            args, bs_id, no_bf_gain[vehicle], load
        )
        capacity[vehicle] = _capacity_per_rb_bps(
            args,
            bs_id,
            serving_gain[vehicle],
            interference[vehicle],
            pilot_average[vehicle],
        )
        if service_fraction is not None:
            capacity[vehicle] *= service_fraction[vehicle]
    allocation = {}
    final_load = np.zeros(num_bs, dtype=float)
    for bs_id in range(num_bs):
        associated = [vehicle for vehicle in vehicles if connection[vehicle] == bs_id]
        associated.sort(key=lambda vehicle: capacity[vehicle], reverse=True)
        remaining = rb_capacities[bs_id]
        for vehicle in associated:
            if capacity[vehicle] <= 1e-9:
                requested = remaining
            else:
                requested = backlog_bits[vehicle] / (
                    capacity[vehicle] * frame_duration_s
                )
            allocated = min(max(requested, 0.0), remaining)
            allocation[vehicle] = allocated
            remaining -= allocated
        final_load[bs_id] = (
            sum(allocation.get(vehicle, 0.0) for vehicle in associated)
            / rb_capacities[bs_id]
        )
    return allocation, capacity, final_load, interference


def _fluid_step(
    args,
    records: MutableMapping,
    learners: Dict[object, _LearnerState],
    queue_start: Dict[object, float],
    vehicle_rate: Dict[object, float],
    receiver_sweep: Dict[object, bool],
    previous_load: np.ndarray,
    macro_bs_loc: np.ndarray,
    config: PQLBAConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> _FluidStep:
    vehicles = sorted(records.keys(), key=str)
    frame_duration_s = args.slots_per_frame * args.slot_len
    connection = {
        vehicle: action_serving_bs(learners[vehicle].action, config)
        for vehicle in vehicles
    }
    serving_gain: Dict[object, float] = {}
    no_bf_gain: Dict[object, np.ndarray] = {}
    pilot_average: Dict[object, float] = {}
    backlog_bits: Dict[object, float] = {}
    for vehicle in vehicles:
        serving_bs = connection[vehicle]
        serving_gain[vehicle] = _learner_gain_db(
            args,
            learners[vehicle],
            records[vehicle],
            macro_bs_loc,
            config,
            dft_tx,
            dft_rx,
        )
        no_bf_gain[vehicle] = np.concatenate(
            (
                np.asarray([serving_gain[vehicle] if serving_bs == 0 else -180.0]),
                no_bf_gain_db(records[vehicle]["h"]),
            )
        )
        pilots = 0.0
        if serving_bs > 0:
            pilots = float(config.tracking_pilots)
            if receiver_sweep[vehicle]:
                # Spread a large hierarchical sweep over multiple slots.  The
                # per-slot cap stays strictly below 100% overhead, matching
                # the exact simulator and avoiding zero-capacity RBs.
                overhead_sum = 0.0
                for slot_index in range(args.slots_per_frame):
                    slot_pilots = sweep_pilots_for_slot(
                        config.receiver_sweep_pilots,
                        config.tracking_pilots,
                        slot_index,
                        args.pilot_overhead_factor,
                    )
                    overhead_sum += min(
                        slot_pilots * args.pilot_overhead_factor, 1.0
                    )
                average_overhead = overhead_sum / args.slots_per_frame
                pilots = average_overhead / args.pilot_overhead_factor
        pilot_average[vehicle] = pilots
        backlog_bits[vehicle] = (
            queue_start[vehicle] + vehicle_rate[vehicle] * frame_duration_s
        )

    allocation, capacity, load, interference = _fluid_allocation(
        args,
        vehicles,
        connection,
        serving_gain,
        no_bf_gain,
        pilot_average,
        backlog_bits,
        previous_load,
        frame_duration_s,
    )
    served_bits = {}
    queue_end = {}
    user_power = {}
    for vehicle in vehicles:
        served_bits[vehicle] = min(
            backlog_bits[vehicle],
            allocation[vehicle] * capacity[vehicle] * frame_duration_s,
        )
        queue_end[vehicle] = max(0.0, backlog_bits[vehicle] - served_bits[vehicle])
        power = args.p_macro if connection[vehicle] == 0 else args.p_micro
        user_power[vehicle] = allocation[vehicle] * power
    return _FluidStep(
        queue_end=queue_end,
        served_bits=served_bits,
        user_power_w=user_power,
        serving_gain_db=serving_gain,
        interference_db=interference,
        load_ratio=load,
        connection=connection,
    )


def run_fluid_pql_episode(
    args,
    timeline_dir: MutableMapping,
    policy: PQLBAPolicy,
    reward_config: AdaptedRewardConfig,
    data_rate_mbps: float,
    seed: int = 1,
    learn: bool = False,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
) -> Dict[str, float]:
    """Run one learning or frozen-policy episode in the fluid surrogate."""

    reward_config.validate()
    config = policy.config
    if not (
        config.include_queue_state
        and config.include_load_state
        and config.include_interference_state
    ):
        raise ValueError(
            "the adapted trainer requires queue, load, and interference states"
        )
    rng = np.random.default_rng(seed)
    dft_tx = generate_dft_codebook(config.num_tx_beams)
    dft_rx = generate_dft_codebook(config.num_rx_beams)
    macro_loc = np.asarray(macro_bs_loc, dtype=float)
    frames = list(timeline_dir.keys())
    if len(frames) < 2:
        raise ValueError("timeline must contain at least two frames")
    frame_duration_s = args.slots_per_frame * args.slot_len
    rate_bps = float(data_rate_mbps) * 1e6
    queue_upper_bound = rate_bps * args.lat_slot_ub * args.slot_len
    learners: Dict[object, _LearnerState] = {}
    queues: Dict[object, float] = {}
    previous_load = np.zeros(1 + config.num_micro_bs, dtype=float)

    rewards: List[float] = []
    td_errors: List[float] = []
    power_samples: List[float] = []
    violation_samples: List[float] = []
    delay_samples: List[float] = []
    vehicle_samples: List[float] = []
    handovers = 0
    beam_switches = 0
    decisions = 0
    known_decisions = 0
    updates_start = policy.update_count
    decisions_start = policy.decision_count

    for frame in frames:
        records = timeline_dir[frame]
        present = set(records.keys())
        for departed in set(learners.keys()).difference(present):
            learners.pop(departed, None)
            queues.pop(departed, None)
        for vehicle in sorted(present, key=str):
            if vehicle not in learners:
                position = np.asarray(records[vehicle]["pos"], dtype=float)
                learners[vehicle] = _LearnerState(
                    action=0,
                    rx_beam=None,
                    pending_action=None,
                    last_position=position.copy(),
                    distance_since_event=config.zone_size_m,
                )
                queues[vehicle] = 0.5 * queue_upper_bound

        receiver_sweep: Dict[object, bool] = {}
        changed_by_vehicle: Dict[object, bool] = {}
        handover_by_vehicle: Dict[object, bool] = {}
        for vehicle in sorted(present, key=str):
            learner = learners[vehicle]
            old_bs = action_serving_bs(learner.action, config)
            decision_applied = learner.pending_action is not None
            changed = _apply_pending_action(
                learner, records[vehicle], config, dft_tx, dft_rx
            )
            new_bs = action_serving_bs(learner.action, config)
            receiver_sweep[vehicle] = decision_applied and new_bs > 0
            changed_by_vehicle[vehicle] = changed
            handover_by_vehicle[vehicle] = changed and old_bs != new_bs
            if changed:
                beam_switches += 1
                if old_bs != new_bs:
                    handovers += 1

        vehicle_rate = {vehicle: rate_bps for vehicle in present}
        queue_before_service = queues.copy()
        step = _fluid_step(
            args,
            records,
            learners,
            queues,
            vehicle_rate,
            receiver_sweep,
            previous_load,
            macro_loc,
            config,
            dft_tx,
            dft_rx,
        )
        previous_load = step.load_ratio
        queues = step.queue_end

        frame_power = sum(step.user_power_w.values())
        frame_violations = 0
        frame_delay = 0.0
        for vehicle in present:
            queue_ratio = queues[vehicle] / queue_upper_bound
            frame_violations += int(queue_ratio > 1.0)
            frame_delay += queues[vehicle] / rate_bps
            learner = learners[vehicle]
            if learner.transition_state is not None:
                service_ratio = min(
                    step.served_bits[vehicle] / (rate_bps * frame_duration_s),
                    reward_config.service_reward_cap,
                )
                reward = (
                    reward_config.reward_offset
                    + reward_config.service_weight * service_ratio
                    - reward_config.queue_weight
                    * min(queue_ratio, reward_config.queue_penalty_cap)
                    - reward_config.violation_weight * float(queue_ratio > 1.0)
                    - reward_config.energy_weight * step.user_power_w[vehicle]
                    - reward_config.load_weight
                    * step.load_ratio[step.connection[vehicle]]
                    - reward_config.handover_weight
                    * float(handover_by_vehicle[vehicle])
                    - reward_config.beam_switch_weight
                    * float(changed_by_vehicle[vehicle])
                )
                learner.transition_reward_mbit += reward

        count = max(len(present), 1)
        power_samples.append(frame_power)
        violation_samples.append(frame_violations / count)
        delay_samples.append(frame_delay / count)
        vehicle_samples.append(float(len(present)))

        for vehicle in sorted(present, key=str):
            learner = learners[vehicle]
            current_position = np.asarray(records[vehicle]["pos"], dtype=float)
            learner.distance_since_event += float(
                np.linalg.norm(current_position - learner.last_position)
            )
            learner.last_position = current_position.copy()
            if learner.distance_since_event + 1e-9 < config.zone_size_m:
                continue
            serving_bs = step.connection[vehicle]
            state = policy.make_state(
                step.serving_gain_db[vehicle],
                learner.action,
                float(records[vehicle].get("angle", 0.0)),
                position=current_position,
                # Match the exact simulator: the association command is based
                # on the backlog available before this frame's slot service.
                queue_ratio=queue_before_service[vehicle] / queue_upper_bound,
                load_ratio=step.load_ratio[serving_bs],
                interference_db=step.interference_db[vehicle],
                traffic_mbps=data_rate_mbps,
            )
            if state in policy.q_table:
                known_decisions += 1
            decisions += 1
            if learn and learner.transition_state is not None:
                td_error = policy.update(
                    learner.transition_state,
                    int(learner.transition_action),
                    learner.transition_reward_mbit,
                    state,
                )
                rewards.append(learner.transition_reward_mbit)
                td_errors.append(abs(td_error))
            selected_action = policy.select_action(state, rng, explore=learn)
            learner.pending_action = selected_action
            learner.transition_state = state
            learner.transition_action = selected_action
            learner.transition_reward_mbit = 0.0
            learner.distance_since_event %= config.zone_size_m

    duration_s = len(frames) * frame_duration_s
    average_vehicles = float(np.mean(vehicle_samples))
    return {
        "data_rate_mbps": float(data_rate_mbps),
        "learn": bool(learn),
        "updates": float(policy.update_count - updates_start),
        "exploration_decisions": float(policy.decision_count - decisions_start),
        "decisions": float(decisions),
        "known_state_decision_ratio": float(known_decisions / max(decisions, 1)),
        "mean_event_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_abs_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
        "average_system_power_w": float(np.mean(power_samples)),
        "queue_violation_percent": float(100.0 * np.mean(violation_samples)),
        "average_queueing_proxy_ms": float(1000.0 * np.mean(delay_samples)),
        "handover_per_vehicle_per_s": float(
            handovers / max(duration_s * average_vehicles, 1e-12)
        ),
        "beam_switch_per_vehicle_per_s": float(
            beam_switches / max(duration_s * average_vehicles, 1e-12)
        ),
        "average_vehicle_count": average_vehicles,
        "epsilon": float(policy.epsilon()),
        "q_states": float(len(policy.q_table)),
        "visited_pairs": float(policy.visited_state_action_pairs),
    }


def train_contextual_pql_ba(
    args,
    timeline_dir: MutableMapping,
    config: PQLBAConfig,
    reward_config: AdaptedRewardConfig,
    data_rate_schedule_mbps: Sequence[float],
    epochs: int,
    seed: int = 1,
    verbose: bool = True,
) -> Tuple[PQLBAPolicy, List[Dict[str, float]]]:
    """Train a shared contextual Q table over a schedule of offered loads."""

    if epochs <= 0 or not data_rate_schedule_mbps:
        raise ValueError("epochs and data-rate schedule must be nonempty")
    policy = PQLBAPolicy(config)
    history = []
    for epoch in range(epochs):
        rate = float(data_rate_schedule_mbps[epoch % len(data_rate_schedule_mbps)])
        record = run_fluid_pql_episode(
            args,
            timeline_dir,
            policy,
            reward_config,
            data_rate_mbps=rate,
            seed=seed + epoch,
            learn=True,
        )
        record["epoch"] = float(epoch + 1)
        history.append(record)
        if verbose:
            print(
                "{} epoch {:02d} rate={:g}: reward={:.3f}, power={:.2f} W, "
                "vio={:.2f}%, eps={:.3f}, states={:.0f}".format(
                    reward_config.name,
                    epoch + 1,
                    rate,
                    record["mean_event_reward"],
                    record["average_system_power_w"],
                    record["queue_violation_percent"],
                    record["epsilon"],
                    record["q_states"],
                )
            )
    return policy, history
