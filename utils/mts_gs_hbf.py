"""Non-RL multi-timescale GS association and beamforming baseline.

This module adapts the multi-timescale user-association/hybrid-beamforming
architecture of Heydarian et al. to the MEET-COBRA system model.  Slow-time
UE--BS association is obtained by capacity-aware Gale--Shapley matching,
medium-time analog beamforming uses the common DFT codebooks, and the exact
simulator retains the manuscript's slot-time OTR resource allocator.

The adaptation is deliberately model-driven: there is no training, learned
parameter, replay buffer, or value function.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from utils.dql_hbt import effective_sinr_db, local_track_beam_pair
from utils.pql_ba import best_beam_pair, macro_gain_db, no_bf_gain_db
from utils.pql_ba_adapted import _capacity_per_rb_bps, _interference_db


@dataclasses.dataclass(frozen=True)
class MTSGSHBFConfig:
    """Configuration of the adapted three-timescale controller."""

    name: str = "balanced"
    num_micro_bs: int = 4
    num_tx_beams: int = 32
    num_rx_beams: int = 8
    association_interval_frames: int = 10
    full_sweep_interval_frames: int = 5
    local_tracking_interval_frames: int = 1
    track_tx_radius: int = 1
    track_rx_radius: int = 1
    tracking_pilots: int = 1
    sinr_threshold_db: float = -5.0
    rate_weight: float = 1.0
    load_weight: float = 1.5
    energy_weight: float = 2.0
    handover_penalty: float = 0.25
    handover_hysteresis: float = 0.15
    bs_queue_weight: float = 2.0
    bs_rate_weight: float = 1.0
    bs_demand_weight: float = 0.25
    bs_stay_bonus: float = 0.10
    queue_drain_weight: float = 0.5
    queue_drain_cap_ratio: float = 2.0
    admission_capacity_factor: float = 1.0
    minimum_demand_rb: float = 0.05
    pressure_adaptive: bool = False
    pressure_start_ratio: float = 0.5
    pressure_full_ratio: float = 1.5
    urgent_load_weight: float = 2.0
    urgent_energy_weight: float = 0.5
    urgent_handover_penalty: float = 0.10
    urgent_handover_hysteresis: float = 0.05
    urgent_bs_queue_weight: float = 3.0
    urgent_queue_drain_weight: float = 0.75
    ho_interruption_ms: float = 0.0

    @property
    def num_bs(self) -> int:
        return 1 + self.num_micro_bs

    @property
    def full_sweep_pilots(self) -> int:
        return self.num_tx_beams * self.num_rx_beams

    @property
    def local_sweep_pilots(self) -> int:
        return (2 * self.track_tx_radius + 1) * (
            2 * self.track_rx_radius + 1
        )

    def validate(self) -> None:
        if not np.isfinite(self.ho_interruption_ms) or not 0 <= self.ho_interruption_ms < 100:
            raise ValueError("HO interruption must be within the 100-ms frame")
        if self.num_micro_bs <= 0:
            raise ValueError("num_micro_bs must be positive")
        if self.num_tx_beams <= 0 or self.num_rx_beams <= 0:
            raise ValueError("beam counts must be positive")
        for label in (
            "association_interval_frames",
            "full_sweep_interval_frames",
            "local_tracking_interval_frames",
        ):
            if int(getattr(self, label)) <= 0:
                raise ValueError("{} must be positive".format(label))
        if self.full_sweep_interval_frames > self.association_interval_frames:
            raise ValueError("full sweeps must be at least as frequent as association")
        if self.local_tracking_interval_frames > self.full_sweep_interval_frames:
            raise ValueError("local tracking must be at least as frequent as full sweeps")
        nonnegative = (
            "rate_weight",
            "load_weight",
            "energy_weight",
            "handover_penalty",
            "handover_hysteresis",
            "bs_queue_weight",
            "bs_rate_weight",
            "bs_demand_weight",
            "bs_stay_bonus",
            "queue_drain_weight",
            "queue_drain_cap_ratio",
            "admission_capacity_factor",
            "minimum_demand_rb",
            "pressure_start_ratio",
            "pressure_full_ratio",
            "urgent_load_weight",
            "urgent_energy_weight",
            "urgent_handover_penalty",
            "urgent_handover_hysteresis",
            "urgent_bs_queue_weight",
            "urgent_queue_drain_weight",
        )
        if any(float(getattr(self, x)) < 0.0 for x in nonnegative):
            raise ValueError("MTS-GS-HBF weights must be nonnegative")
        if self.admission_capacity_factor <= 0.0:
            raise ValueError("admission_capacity_factor must be positive")
        if self.pressure_full_ratio <= self.pressure_start_ratio:
            raise ValueError("pressure_full_ratio must exceed pressure_start_ratio")


@dataclasses.dataclass
class MTSLinkState:
    """Causal execution state for one vehicle."""

    action: int = 0
    tx_beam: Optional[int] = None
    rx_beam: Optional[int] = None
    pending_command: Optional["MTSCommand"] = None
    current_sweep_pilots: int = 0


@dataclasses.dataclass(frozen=True)
class MTSCommand:
    target_bs: int
    tx_beam: Optional[int]
    rx_beam: Optional[int]
    sweep_pilots: int
    reason: str


@dataclasses.dataclass(frozen=True)
class MTSActionOutcome:
    handover: bool
    beam_switch: bool
    sweep_pilots: int


@dataclasses.dataclass(frozen=True)
class MTSLinkCandidate:
    vehicle: object
    bs: int
    tx_beam: Optional[int]
    rx_beam: Optional[int]
    gain_db: float
    sinr_db: float
    capacity_per_rb_bps: float
    demand_rb: float
    vehicle_score: float
    bs_score: float


@dataclasses.dataclass(frozen=True)
class GaleShapleyResult:
    assignments: Dict[object, MTSLinkCandidate]
    used_capacity: np.ndarray
    proposal_count: int
    unassigned: Tuple[object, ...]


def apply_mts_command(
    state: MTSLinkState,
    command: Optional[MTSCommand],
    config: MTSGSHBFConfig,
) -> MTSActionOutcome:
    """Apply a command computed from the preceding frame."""

    state.current_sweep_pilots = 0
    if command is None:
        return MTSActionOutcome(False, False, 0)
    if not 0 <= int(command.target_bs) < config.num_bs:
        raise ValueError("invalid target BS")
    if command.target_bs == 0:
        if command.tx_beam is not None or command.rx_beam is not None:
            raise ValueError("macro command must not carry a beam pair")
    else:
        if command.tx_beam is None or command.rx_beam is None:
            raise ValueError("micro command requires a beam pair")
        if not 0 <= int(command.tx_beam) < config.num_tx_beams:
            raise ValueError("invalid TX beam")
        if not 0 <= int(command.rx_beam) < config.num_rx_beams:
            raise ValueError("invalid RX beam")

    old_bs = int(state.action)
    old_pair = (state.tx_beam, state.rx_beam)
    state.action = int(command.target_bs)
    state.tx_beam = None if state.action == 0 else int(command.tx_beam)
    state.rx_beam = None if state.action == 0 else int(command.rx_beam)
    state.current_sweep_pilots = int(command.sweep_pilots)
    new_pair = (state.tx_beam, state.rx_beam)
    return MTSActionOutcome(
        handover=state.action != old_bs,
        beam_switch=state.action != old_bs or new_pair != old_pair,
        sweep_pilots=state.current_sweep_pilots,
    )


def capacity_aware_gale_shapley(
    candidates: MutableMapping[object, Sequence[MTSLinkCandidate]],
    capacities: Sequence[float],
    current_bs: Optional[MutableMapping[object, int]] = None,
    admission_capacity_factor: float = 1.0,
) -> GaleShapleyResult:
    """Many-to-one deferred acceptance with continuous RB demands.

    Vehicles propose according to ``vehicle_score``.  Each BS repeatedly keeps
    the highest ``bs_score`` proposals that fit its RB budget and rejects the
    rest.  A vehicle rejected by every candidate falls back to its current BS;
    this preserves a valid serving link when the offered load exceeds total
    physical capacity.
    """

    rb_capacity = np.asarray(capacities, dtype=float)
    if rb_capacity.ndim != 1 or len(rb_capacity) == 0:
        raise ValueError("capacities must be a non-empty vector")
    if not np.isfinite(rb_capacity).all() or np.any(rb_capacity <= 0.0):
        raise ValueError("capacities must be finite and positive")
    if admission_capacity_factor <= 0.0:
        raise ValueError("admission_capacity_factor must be positive")
    ordered: Dict[object, List[MTSLinkCandidate]] = {}
    by_vehicle_bs: Dict[Tuple[object, int], MTSLinkCandidate] = {}
    for vehicle, links in candidates.items():
        unique = {}
        for link in links:
            if link.vehicle != vehicle:
                raise ValueError("candidate vehicle mismatch")
            if not 0 <= int(link.bs) < len(rb_capacity):
                raise ValueError("candidate BS outside capacity vector")
            existing = unique.get(int(link.bs))
            if existing is None or link.vehicle_score > existing.vehicle_score:
                unique[int(link.bs)] = link
        if not unique:
            raise ValueError("every vehicle needs at least one candidate")
        ordered[vehicle] = sorted(
            unique.values(),
            key=lambda x: (-x.vehicle_score, x.bs),
        )
        for link in unique.values():
            by_vehicle_bs[(vehicle, int(link.bs))] = link

    held: Dict[int, List[MTSLinkCandidate]] = {
        bs: [] for bs in range(len(rb_capacity))
    }
    next_choice = {vehicle: 0 for vehicle in ordered}
    free = sorted(ordered, key=str)
    proposal_count = 0
    exhausted = set()
    while free:
        vehicle = free.pop(0)
        choice_index = next_choice[vehicle]
        if choice_index >= len(ordered[vehicle]):
            exhausted.add(vehicle)
            continue
        proposal = ordered[vehicle][choice_index]
        next_choice[vehicle] += 1
        proposal_count += 1
        bs = int(proposal.bs)
        pool = [x for x in held[bs] if x.vehicle != vehicle] + [proposal]
        pool.sort(key=lambda x: (-x.bs_score, str(x.vehicle)))
        accepted = []
        rejected = []
        budget = rb_capacity[bs] * admission_capacity_factor
        used = 0.0
        for link in pool:
            demand = max(float(link.demand_rb), 0.0)
            # Always permit the first proposal.  This avoids an empty BS when
            # one temporarily backlogged vehicle requests more than its quota.
            if not accepted or used + demand <= budget + 1e-9:
                accepted.append(link)
                used += demand
            else:
                rejected.append(link)
        held[bs] = accepted
        for link in rejected:
            if next_choice[link.vehicle] < len(ordered[link.vehicle]):
                free.append(link.vehicle)
            else:
                exhausted.add(link.vehicle)
        free.sort(key=str)

    assignments = {
        link.vehicle: link for links in held.values() for link in links
    }
    unassigned = sorted(set(ordered).difference(assignments), key=str)
    if current_bs is not None:
        for vehicle in list(unassigned):
            fallback = by_vehicle_bs.get((vehicle, int(current_bs[vehicle])))
            if fallback is None:
                fallback = ordered[vehicle][0]
            assignments[vehicle] = fallback
    used_capacity = np.zeros(len(rb_capacity), dtype=float)
    for link in assignments.values():
        used_capacity[int(link.bs)] += max(float(link.demand_rb), 0.0)
    return GaleShapleyResult(
        assignments=assignments,
        used_capacity=used_capacity,
        proposal_count=proposal_count,
        unassigned=tuple(unassigned),
    )


def build_link_candidates(
    args,
    records: MutableMapping,
    states: MutableMapping[object, MTSLinkState],
    queue_bits: MutableMapping[object, float],
    queue_upper_bound: MutableMapping[object, float],
    vehicle_rate: MutableMapping[object, float],
    estimated_load: Sequence[float],
    config: MTSGSHBFConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
    macro_loc: Sequence[float] = (0.0, 0.0),
) -> Dict[object, List[MTSLinkCandidate]]:
    """Construct one best-beam candidate for every vehicle--BS pair."""

    load = np.asarray(estimated_load, dtype=float)
    if load.shape != (config.num_bs,) or not np.isfinite(load).all():
        raise ValueError("invalid estimated load")
    frame_duration = args.slots_per_frame * args.slot_len
    capacities = np.asarray(
        [args.num_RB_macro]
        + [args.num_RB_micro] * config.num_micro_bs,
        dtype=float,
    )
    macro_position = np.asarray(macro_loc, dtype=float)
    result: Dict[object, List[MTSLinkCandidate]] = {}
    for vehicle in sorted(records, key=str):
        record = records[vehicle]
        current = int(states[vehicle].action)
        no_bf = np.concatenate(([-180.0], no_bf_gain_db(record["h"])))
        queue_ratio = float(queue_bits[vehicle] / max(queue_upper_bound[vehicle], 1e-12))
        if config.pressure_adaptive:
            pressure = float(
                np.clip(
                    (queue_ratio - config.pressure_start_ratio)
                    / (config.pressure_full_ratio - config.pressure_start_ratio),
                    0.0,
                    1.0,
                )
            )
        else:
            pressure = 0.0

        def pressure_blend(normal: float, urgent: float) -> float:
            return float(normal + pressure * (urgent - normal))

        load_weight = pressure_blend(config.load_weight, config.urgent_load_weight)
        energy_weight = pressure_blend(
            config.energy_weight, config.urgent_energy_weight
        )
        handover_penalty = pressure_blend(
            config.handover_penalty, config.urgent_handover_penalty
        )
        handover_hysteresis = pressure_blend(
            config.handover_hysteresis, config.urgent_handover_hysteresis
        )
        bs_queue_weight = pressure_blend(
            config.bs_queue_weight, config.urgent_bs_queue_weight
        )
        queue_drain_weight = pressure_blend(
            config.queue_drain_weight, config.urgent_queue_drain_weight
        )
        drain_bits = queue_drain_weight * min(
            float(queue_bits[vehicle]),
            config.queue_drain_cap_ratio * float(queue_upper_bound[vehicle]),
        )
        target_bits = float(vehicle_rate[vehicle]) * frame_duration + drain_bits
        links = []
        for bs in range(config.num_bs):
            if bs == 0:
                tx_beam = None
                rx_beam = None
                gain = macro_gain_db(args, record["pos"], macro_position)
                interference = -np.inf
                pilot_average = 0.0
                power = args.p_macro
            else:
                tx_beam, rx_beam, gain = best_beam_pair(
                    record["h"], bs - 1, dft_tx, dft_rx
                )
                interference = _interference_db(args, bs, no_bf, load)
                pilot_average = config.full_sweep_pilots / args.slots_per_frame
                power = args.p_micro
            sinr = effective_sinr_db(args, bs, gain, interference)
            capacity_per_rb = _capacity_per_rb_bps(
                args, bs, gain, interference, pilot_average
            )
            demand = target_bits / max(capacity_per_rb * frame_duration, 1e-12)
            demand = float(
                np.clip(demand, config.minimum_demand_rb, capacities[bs])
            )
            predicted_load = float(load[bs] + demand / capacities[bs])
            capacity_mbps = capacity_per_rb / 1e6
            energy_per_mbit = float(power / max(capacity_mbps, 1e-9))
            switching = float(bs != current)
            vehicle_score = float(
                config.rate_weight * math.log1p(max(capacity_mbps, 0.0))
                - load_weight * predicted_load
                - energy_weight * energy_per_mbit
                - handover_penalty * switching
            )
            bs_score = float(
                bs_queue_weight * min(queue_ratio, 10.0)
                + config.bs_rate_weight * math.log1p(max(capacity_mbps, 0.0))
                - config.bs_demand_weight * demand / capacities[bs]
                + config.bs_stay_bonus * float(bs == current)
            )
            if sinr >= config.sinr_threshold_db or bs == current or bs == 0:
                links.append(
                    MTSLinkCandidate(
                        vehicle=vehicle,
                        bs=bs,
                        tx_beam=tx_beam,
                        rx_beam=rx_beam,
                        gain_db=float(gain),
                        sinr_db=float(sinr),
                        capacity_per_rb_bps=float(capacity_per_rb),
                        # Correct admission occupancy only; retain preference
                        # scores and nominal energy cost from the chosen design.
                        demand_rb=(demand / (1.0 - config.ho_interruption_ms / (1000 * frame_duration))
                                   if bs != current else demand),
                        vehicle_score=vehicle_score,
                        bs_score=bs_score,
                    )
                )
        current_link = next((x for x in links if x.bs == current), None)
        if current_link is not None:
            best_other = max(
                (x.vehicle_score for x in links if x.bs != current),
                default=-np.inf,
            )
            if best_other < current_link.vehicle_score + handover_hysteresis:
                links = [
                    dataclasses.replace(
                        x,
                        vehicle_score=(
                            max(x.vehicle_score, best_other + 1e-6)
                            if x.bs == current
                            else x.vehicle_score
                        ),
                    )
                    for x in links
                ]
        result[vehicle] = links
    return result


def association_commands(
    args,
    records: MutableMapping,
    states: MutableMapping[object, MTSLinkState],
    queue_bits: MutableMapping[object, float],
    queue_upper_bound: MutableMapping[object, float],
    vehicle_rate: MutableMapping[object, float],
    estimated_load: Sequence[float],
    config: MTSGSHBFConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
    macro_loc: Sequence[float] = (0.0, 0.0),
) -> Tuple[Dict[object, MTSCommand], GaleShapleyResult]:
    candidates = build_link_candidates(
        args,
        records,
        states,
        queue_bits,
        queue_upper_bound,
        vehicle_rate,
        estimated_load,
        config,
        dft_tx,
        dft_rx,
        macro_loc,
    )
    capacities = [args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs
    matching = capacity_aware_gale_shapley(
        candidates,
        capacities,
        current_bs={vehicle: state.action for vehicle, state in states.items()},
        admission_capacity_factor=config.admission_capacity_factor,
    )
    commands = {}
    for vehicle, candidate in matching.assignments.items():
        commands[vehicle] = MTSCommand(
            target_bs=int(candidate.bs),
            tx_beam=candidate.tx_beam,
            rx_beam=candidate.rx_beam,
            sweep_pilots=(config.full_sweep_pilots if candidate.bs > 0 else 0),
            reason="association",
        )
    return commands, matching


def beam_update_command(
    record: MutableMapping,
    state: MTSLinkState,
    config: MTSGSHBFConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
    full_sweep: bool,
) -> Optional[MTSCommand]:
    """Return a full-codebook or local tracking command for the current BS."""

    bs = int(state.action)
    if bs == 0:
        return None
    if full_sweep or state.tx_beam is None or state.rx_beam is None:
        tx_beam, rx_beam, _ = best_beam_pair(
            record["h"], bs - 1, dft_tx, dft_rx
        )
        pilots = config.full_sweep_pilots
        reason = "full_sweep"
    else:
        tx_beam, rx_beam, _, pilots = local_track_beam_pair(
            record["h"],
            bs - 1,
            int(state.tx_beam),
            int(state.rx_beam),
            dft_tx,
            dft_rx,
            config.track_tx_radius,
            config.track_rx_radius,
        )
        reason = "local_tracking"
    return MTSCommand(
        target_bs=bs,
        tx_beam=tx_beam,
        rx_beam=rx_beam,
        sweep_pilots=int(pilots),
        reason=reason,
    )


def candidate_configs() -> Dict[str, MTSGSHBFConfig]:
    """Small deterministic design set for exploratory parameter screening."""

    return {
        "qos_fast": MTSGSHBFConfig(
            name="qos_fast",
            association_interval_frames=5,
            full_sweep_interval_frames=5,
            local_tracking_interval_frames=1,
            load_weight=2.0,
            energy_weight=0.5,
            handover_penalty=0.10,
            handover_hysteresis=0.05,
            bs_queue_weight=3.0,
            queue_drain_weight=0.75,
        ),
        "balanced": MTSGSHBFConfig(name="balanced"),
        "energy_sticky": MTSGSHBFConfig(
            name="energy_sticky",
            association_interval_frames=10,
            full_sweep_interval_frames=5,
            local_tracking_interval_frames=1,
            load_weight=1.0,
            energy_weight=4.0,
            handover_penalty=0.50,
            handover_hysteresis=0.25,
            bs_queue_weight=2.0,
            queue_drain_weight=0.5,
        ),
        "pressure_adaptive": MTSGSHBFConfig(
            name="pressure_adaptive",
            association_interval_frames=5,
            full_sweep_interval_frames=5,
            local_tracking_interval_frames=1,
            load_weight=1.0,
            energy_weight=4.0,
            handover_penalty=0.50,
            handover_hysteresis=0.25,
            bs_queue_weight=2.0,
            queue_drain_weight=0.5,
            pressure_adaptive=True,
            pressure_start_ratio=0.5,
            pressure_full_ratio=1.5,
        ),
        "pressure_early": MTSGSHBFConfig(
            name="pressure_early",
            association_interval_frames=5,
            full_sweep_interval_frames=5,
            local_tracking_interval_frames=1,
            load_weight=1.0,
            energy_weight=4.0,
            handover_penalty=0.50,
            handover_hysteresis=0.25,
            bs_queue_weight=2.0,
            queue_drain_weight=0.5,
            pressure_adaptive=True,
            pressure_start_ratio=0.25,
            pressure_full_ratio=1.0,
            urgent_load_weight=2.5,
            urgent_energy_weight=0.25,
            urgent_bs_queue_weight=4.0,
            urgent_queue_drain_weight=1.0,
        ),
        "slow_energy": MTSGSHBFConfig(
            name="slow_energy",
            association_interval_frames=20,
            full_sweep_interval_frames=10,
            local_tracking_interval_frames=2,
            load_weight=1.0,
            energy_weight=3.0,
            handover_penalty=0.50,
            handover_hysteresis=0.25,
            bs_queue_weight=2.5,
            queue_drain_weight=0.75,
        ),
    }
