"""Optimization-assisted MAPPO handover baseline for MEET-COBRA.

This module adapts Wang et al., *A novel handover scheme for millimeter wave
network: an approach of integrating reinforcement learning and optimization*,
Digital Communications and Networks, 2024.  The paper's division of labour is
kept explicit: a multi-agent PPO policy emits one binary handover trigger per
vehicle, and a separate optimizer chooses a target BS/beam for the triggered
vehicles.  OTR-RA remains the common final RB scheduler so that the baseline
does not receive a different resource-allocation rule from MEET-COBRA.

The original work has a fixed ten-UE population.  The road trace has a changing
population, so actors share parameters and the centralized critic receives a
permutation-invariant pooled global state.  Frozen policies are evaluated by
``utils.o_mappo_sim`` in the exact slot-level simulator.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import math
import os
import random
import time
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from torch import nn
from torch.distributions import Categorical

from utils.beam_utils import generate_dft_codebook
from utils.dql_hbt import effective_sinr_db, local_track_beam_pair
from utils.hierarchical_beam import hierarchical_beam_pair
from utils.pql_ba import (
    _LearnerState,
    best_beam_pair,
    fixed_pair_gain_db,
    macro_gain_db,
    no_bf_gain_db,
    sweep_pilots_for_slot,
)
from utils.pql_ba_adapted import (
    _capacity_per_rb_bps,
    _fluid_allocation,
    _interference_db,
)


@dataclasses.dataclass(frozen=True)
class OMAPPORewardConfig:
    """Reward definition for one source-like or adapted candidate."""

    name: str
    source_threshold_reward: bool = False
    team_mix: float = 1.0
    service_weight: float = 2.0
    queue_weight: float = 1.0
    violation_weight: float = 5.0
    energy_weight: float = 0.0
    overload_weight: float = 0.0
    cvar_weight: float = 0.0
    cvar_alpha: float = 0.95
    handover_weight: float = 0.05
    sweep_weight: float = 0.02
    reward_offset: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.team_mix <= 1.0:
            raise ValueError("team_mix must be in [0, 1]")
        if not 0.0 <= self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must be in [0, 1)")
        for field in dataclasses.fields(self):
            if field.name in (
                "name",
                "source_threshold_reward",
                "team_mix",
                "cvar_alpha",
            ):
                continue
            if float(getattr(self, field.name)) < 0.0:
                raise ValueError("{} must be nonnegative".format(field.name))


def o_mappo_reward_presets() -> "collections.OrderedDict[str, OMAPPORewardConfig]":
    """Return candidates from a source-derived reward to local adaptations."""

    return collections.OrderedDict(
        (
            (
                "source",
                OMAPPORewardConfig(
                    name="source",
                    source_threshold_reward=True,
                    team_mix=1.0,
                    service_weight=0.0,
                    queue_weight=0.0,
                    violation_weight=0.0,
                    handover_weight=0.05,
                    sweep_weight=0.0,
                ),
            ),
            (
                "qos",
                OMAPPORewardConfig(name="qos", team_mix=0.5),
            ),
            (
                "qos_energy005",
                OMAPPORewardConfig(
                    name="qos_energy005", team_mix=0.5, energy_weight=0.05
                ),
            ),
            (
                "qos_energy020",
                OMAPPORewardConfig(
                    name="qos_energy020", team_mix=0.5, energy_weight=0.20
                ),
            ),
            (
                "qos_energy020_load1",
                OMAPPORewardConfig(
                    name="qos_energy020_load1",
                    team_mix=0.5,
                    energy_weight=0.20,
                    overload_weight=1.0,
                ),
            ),
            (
                "qos_energy020_load1_cvar",
                OMAPPORewardConfig(
                    name="qos_energy020_load1_cvar",
                    team_mix=0.5,
                    energy_weight=0.20,
                    overload_weight=1.0,
                    cvar_weight=2.0,
                    cvar_alpha=0.95,
                ),
            ),
        )
    )


@dataclasses.dataclass
class OMAPPOConfig:
    """System mapping and PPO hyperparameters.

    ``optimizer_variant`` controls only the lower-level target objective:
    ``source`` minimizes instantaneous transmission time; ``load`` also prices
    occupied capacity; and ``load_energy`` adds per-RB transmit energy.
    """

    state_variant: str = "adapted"  # also supports predicted_adapted: original slots, predicted CSI
    information_mode: str = "legacy"  # legacy or shared_prediction
    reported_beam_count: int = 5
    ho_interruption_ms: float = 0.0
    trigger_gate: str = "periodic"  # periodic or source
    optimizer_variant: str = "load_energy"  # source, load, load_energy
    optimizer_solver: str = "greedy"  # greedy or milp
    zone_size_m: float = 10.0
    sinr_threshold_db: float = 2.0
    candidate_count: int = 3
    num_bs: int = 5
    num_micro_bs: int = 4
    num_tx_beams: int = 32
    num_rx_beams: int = 8
    beam_search_variant: str = "exhaustive"  # opt-in hierarchical32 acquisition
    track_tx_radius: int = 1
    track_rx_radius: int = 1
    tracking_pilots: int = 1
    hidden_sizes: Tuple[int, ...] = (64,)
    recurrent: bool = False
    recurrent_hidden_size: int = 128
    recurrent_sequence_length: int = 4
    actor_learning_rate: float = 5.0e-4
    critic_learning_rate: float = 5.0e-4
    discount_factor: float = 0.90
    gae_lambda: float = 0.50
    clip_ratio: float = 0.20
    ppo_epochs: int = 4
    batch_size: int = 256
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    gradient_clip_norm: float = 10.0
    optimizer_overflow_penalty: float = 25.0
    optimizer_load_weight: float = 1.0
    optimizer_energy_weight: float = 0.20
    torch_threads: int = 4

    @property
    def full_sweep_pilots(self) -> int:
        if self.beam_search_variant == "hierarchical32":
            return 32  # 8x2 coarse measurements + 4x4 fine measurements
        return self.num_tx_beams * self.num_rx_beams

    @property
    def tracking_sweep_pilots(self) -> int:
        return (2 * self.track_tx_radius + 1) * (2 * self.track_rx_radius + 1)

    def validate(self) -> None:
        if self.beam_search_variant not in ("exhaustive", "hierarchical32"):
            raise ValueError("invalid beam_search_variant")
        if self.beam_search_variant == "hierarchical32" and (self.num_tx_beams, self.num_rx_beams) != (32, 8):
            raise ValueError("hierarchical32 requires 32 TX and 8 RX beams")
        if self.state_variant not in ("source", "adapted", "feasibility", "pilot", "report", "gain_report", "gain_derived", "predicted_adapted"):
            raise ValueError("invalid state_variant")
        if self.information_mode not in ("legacy", "shared_prediction"):
            raise ValueError("invalid information_mode")
        if self.information_mode == "shared_prediction" and (
            self.state_variant not in ("pilot", "report", "gain_report", "gain_derived", "predicted_adapted") or self.trigger_gate != "periodic"
        ):
            raise ValueError("shared prediction requires a shared CSI state and periodic gate")
        if self.state_variant in ("report", "gain_report", "gain_derived", "predicted_adapted") and self.information_mode != "shared_prediction":
            raise ValueError("report state requires the shared prediction frontend")
        if self.state_variant in ("gain_derived", "predicted_adapted") and self.recurrent:
            raise ValueError("The actor-feature ablation keeps the nonrecurrent architecture")
        if not 1 <= self.reported_beam_count <= self.num_tx_beams * self.num_rx_beams:
            raise ValueError("invalid reported beam count")
        if not 0 <= self.ho_interruption_ms < 100:
            raise ValueError("HO interruption must be within the 100-ms frame")
        if self.trigger_gate not in ("periodic", "source"):
            raise ValueError("trigger_gate must be periodic or source")
        if self.optimizer_variant not in ("source", "load", "load_energy"):
            raise ValueError("invalid optimizer_variant")
        if self.optimizer_solver not in ("greedy", "milp"):
            raise ValueError("optimizer_solver must be greedy or milp")
        if self.num_bs != self.num_micro_bs + 1:
            raise ValueError("num_bs must equal one macro plus micro BSs")
        if not 1 <= self.candidate_count < self.num_bs:
            raise ValueError("candidate_count must be in [1, num_bs-1]")
        if self.zone_size_m <= 0 or self.batch_size <= 0 or self.ppo_epochs <= 0:
            raise ValueError("invalid PPO/system dimensions")
        if self.recurrent_hidden_size <= 0 or self.recurrent_sequence_length <= 0:
            raise ValueError("invalid recurrent dimensions")


@dataclasses.dataclass
class OMAPPOCommand:
    trigger: int
    target_bs: int


@dataclasses.dataclass
class OMAPPOActionOutcome:
    handover: bool
    beam_switch: bool
    sweep_pilots: int


@dataclasses.dataclass
class OMAPPOLearnerState(_LearnerState):
    pending_command: Optional[OMAPPOCommand] = None
    last_trigger: int = 0
    last_handover: bool = False
    current_sweep_pilots: int = 0
    previous_rb_fraction: float = 0.0
    transition_local_state: Optional[np.ndarray] = None
    transition_global_state: Optional[np.ndarray] = None
    transition_action_binary: Optional[int] = None
    transition_log_probability: float = 0.0
    transition_value: float = 0.0
    transition_reward: float = 0.0
    transition_frames: int = 0
    state_history: List[np.ndarray] = dataclasses.field(default_factory=list)
    optimizer_feedback: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )


def apply_o_mappo_command(
    learner: OMAPPOLearnerState,
    command: Optional[OMAPPOCommand],
    vehicle_record: MutableMapping,
    config: OMAPPOConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> OMAPPOActionOutcome:
    """Apply a command one frame after it was selected."""

    learner.current_sweep_pilots = 0
    learner.last_handover = False
    if command is None:
        return OMAPPOActionOutcome(False, False, 0)
    acquire = hierarchical_beam_pair if config.beam_search_variant == "hierarchical32" else best_beam_pair
    old_bs = int(learner.action)
    old_pair = (learner.tx_beam, learner.rx_beam)
    learner.last_trigger = int(command.trigger)
    target = int(command.target_bs)
    if not command.trigger:
        target = old_bs
        if old_bs > 0:
            if learner.tx_beam is None or learner.rx_beam is None:
                learner.tx_beam, learner.rx_beam, _ = acquire(
                    vehicle_record["h"], old_bs - 1, dft_tx, dft_rx
                )
                learner.current_sweep_pilots = config.full_sweep_pilots
            else:
                (
                    learner.tx_beam,
                    learner.rx_beam,
                    _,
                    learner.current_sweep_pilots,
                ) = local_track_beam_pair(
                    vehicle_record["h"],
                    old_bs - 1,
                    int(learner.tx_beam),
                    int(learner.rx_beam),
                    dft_tx,
                    dft_rx,
                    config.track_tx_radius,
                    config.track_rx_radius,
                )
    else:
        if target == old_bs:
            raise ValueError("a triggered O-MAPPO command must change BS")
        learner.action = target
        if target == 0:
            learner.tx_beam = None
            learner.rx_beam = None
        else:
            learner.tx_beam, learner.rx_beam, _ = acquire(
                vehicle_record["h"], target - 1, dft_tx, dft_rx
            )
            learner.current_sweep_pilots = config.full_sweep_pilots
    handover = int(learner.action) != old_bs
    learner.last_handover = handover
    pair = (learner.tx_beam, learner.rx_beam)
    return OMAPPOActionOutcome(
        handover=handover,
        beam_switch=handover or pair != old_pair,
        sweep_pilots=int(learner.current_sweep_pilots),
    )


def source_gate_allows(
    config: OMAPPOConfig,
    serving_sinr_db: float,
    alternative_sinr_db: Sequence[float],
) -> bool:
    """Approximate the source paper's overlap/low-SINR decision context."""

    if config.trigger_gate == "periodic":
        return True
    alternatives = np.asarray(alternative_sinr_db, dtype=float)
    in_overlap = int(np.sum(alternatives >= config.sinr_threshold_db)) >= 2
    return bool(serving_sinr_db < config.sinr_threshold_db or in_overlap)


def state_feature_names(config: OMAPPOConfig) -> List[str]:
    # These first quantities implement Eq. (9)'s previous HO delay, previous
    # system throughput, own bandwidth, and public BS-load information.
    names = ["previous_ho_delay", "system_throughput", "own_rb_fraction"]
    names += ["bs_user_load_{}".format(i) for i in range(config.num_bs)]
    names += ["serving_bs_{}".format(i) for i in range(config.num_bs)]
    if config.state_variant == "source":
        return names
    names = names + [
        "x",
        "y",
        "heading_sin",
        "heading_cos",
        "speed",
        "serving_sinr",
        "queue_ratio",
        "traffic_rate",
    ] + ["bs_rb_load_{}".format(i) for i in range(config.num_bs)] + [
        "interference_to_noise",
        "tx_beam_sin",
        "tx_beam_cos",
        "rx_beam_sin",
        "rx_beam_cos",
    ]
    if config.state_variant in ("pilot", "report", "gain_report", "gain_derived"):
        names = [name for name in names if name not in ("serving_sinr", "interference_to_noise")]
    if config.state_variant == "pilot":
        names += ["superposed_pilot_{}".format(i) for i in range(128)]
    if config.state_variant in ("report", "gain_report", "gain_derived"):
        for bs in range(1, config.num_bs):
            names += [f"reported_desired_gain_{bs}", f"reported_interfering_gain_{bs}"]
            if config.state_variant == "report":
                names += [f"reported_beam_{bs}_rank_{rank + 1}"
                          for rank in range(config.reported_beam_count)]
    if config.state_variant == "gain_derived":
        for group in ("predicted_rb_demand", "predicted_power", "predicted_capacity_margin",
                      "power_saving_vs_stay", "rb_pressure_saving_vs_stay"):
            names += [f"{group}_{bs}" for bs in range(config.num_bs)]
    if config.state_variant == "feasibility":
        names += ["candidate_sinr_{}".format(i) for i in range(config.num_bs)]
        names += ["candidate_demand_{}".format(i) for i in range(config.num_bs)]
        names += ["candidate_residual_{}".format(i) for i in range(config.num_bs)]
        names += ["candidate_margin_{}".format(i) for i in range(config.num_bs)]
        names += [
            "optimizer_total_overflow",
            "optimizer_target_load",
            "optimizer_target_overflow",
            "optimizer_success",
        ]
    return names


def encode_prediction_report(config: OMAPPOConfig, report: MutableMapping) -> np.ndarray:
    """Encode only the CSI payload sent by a vehicle to the macro BS.

    Each micro link contributes two FP32 gains in dB and the ordered top-M_P
    beam-pair indices. Rank is represented by feature position, not by a
    probability vector. Fixed scaling uses no train/test statistics. For the
    paper configuration this is 28 features and 416 bits on the uplink; the
    expanded FP32 actor tensor is not the transmitted representation.
    ``gain_report`` uses exactly the same eight gain entries, without reading
    any predicted beam indices (including during training and critic pooling).
    """
    if report is None:
        raise ValueError("report state requires the vehicle prediction report")
    gain = np.asarray(report["gain"], dtype=np.float32)
    interference = np.asarray(report["interference"], dtype=np.float32)
    if gain.shape != (config.num_micro_bs,) or interference.shape != gain.shape:
        raise ValueError("reported gains have wrong shape")
    if not np.isfinite(gain).all() or not np.isfinite(interference).all():
        raise ValueError("invalid reported gain values")
    # Affine scaling, without clipping, retains all reported gain information.
    gains = np.stack(((gain + 100.0) / 40.0,
                      (interference + 100.0) / 40.0), axis=-1)
    if config.state_variant in ("gain_report", "gain_derived"):
        return gains.ravel().astype(np.float32)
    beam = np.asarray(report["beam"])
    expected = (config.num_micro_bs, config.reported_beam_count)
    if beam.shape != expected:
        raise ValueError("reported ordered beam indices have wrong shape")
    pair_count = config.num_tx_beams * config.num_rx_beams
    if (not np.isfinite(gain).all() or not np.isfinite(interference).all()
            or not np.isfinite(beam).all() or np.any(beam != np.floor(beam))
            or np.any(beam < 0) or np.any(beam >= pair_count)):
        raise ValueError("invalid vehicle prediction report")
    indices = beam.astype(np.float32) / max(pair_count - 1, 1)
    return np.concatenate((gains, indices), axis=1).ravel().astype(np.float32)


def shared_actor_inputs(config: OMAPPOConfig, record: MutableMapping, *, args=None,
                        serving_bs=None, backlog_bits=None, load=None,
                        own_rb_fraction=None, macro_loc=(0.0, 0.0)) -> Dict:
    """Restrict CSI extraction to the configured deployment interface."""
    if config.state_variant == "pilot":
        return {"pilot_observation": record["CSI_preprocessed"][-1]}
    if config.state_variant in ("report", "gain_report", "gain_derived"):
        prediction = record["shared_prediction"]
        keys = ("gain", "interference") if config.state_variant != "report" else ("gain", "interference", "beam")
        report = {key: prediction[key] for key in keys}
        result = {"prediction_report": report}
        if config.state_variant == "gain_derived":
            result["derived_features"] = report_decision_features(args, config, record["pos"],
                report, serving_bs, backlog_bits, load, own_rb_fraction, macro_loc)
        return result
    return {}


def report_decision_features(args, config, position, report, serving_bs, backlog_bits,
                             load, own_rb_fraction, macro_loc=(0.0, 0.0)):
    """Public-report features for the actor; never modify target optimization.

    Gains estimate the best beam pair, including for the serving micro BS;
    they are not measurements of the actually tracked pair. All five BSs
    are represented, in BS order. Positive savings favor leaving the BS.
    """
    if args is None or serving_bs is None or backlog_bits is None or own_rb_fraction is None:
        raise ValueError("Derived actor features require the public decision context")
    encode_prediction_report(dataclasses.replace(config, state_variant="gain_report"), report)
    loads = np.asarray(load, dtype=float)
    if (loads.shape != (config.num_bs,) or not np.isfinite(loads).all()
            or not np.isfinite(backlog_bits) or backlog_bits < 0
            or not np.isfinite(own_rb_fraction) or not 0 <= serving_bs < config.num_bs):
        raise ValueError("Invalid public decision context")
    loads = np.clip(loads, 0.0, 1.0)
    capacities = np.array([args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs)
    powers = np.array([args.p_macro] + [args.p_micro] * config.num_micro_bs)
    desired = np.concatenate(([macro_gain_db(args, position, np.asarray(macro_loc))], report["gain"]))
    interfering = np.concatenate(([-180.0], report["interference"]))
    duration = args.slots_per_frame * args.slot_len
    sweep_average = _cached_report_sweep_average(args.slots_per_frame, args.pilot_overhead_factor,
                                                config.full_sweep_pilots, config.tracking_pilots)
    rate_per_rb = np.array([
        _capacity_per_rb_bps(args, bs, desired[bs], _interference_db(args, bs, interfering, loads),
            0.0 if bs == 0 else config.tracking_pilots if bs == serving_bs else sweep_average)
        for bs in range(config.num_bs)])
    nominal_rb = backlog_bits / np.maximum(rate_per_rb * duration, 1e-12)
    occupancy_rb = nominal_rb.copy()
    switching = np.arange(config.num_bs) != serving_bs
    occupancy_rb[switching] /= 1.0 - config.ho_interruption_ms / (1000.0 * duration)
    demand = occupancy_rb / capacities
    power = nominal_rb * powers  # frame-average cost, not interruption-inflated occupancy
    available = 1.0 - loads
    available[serving_bs] += np.clip(own_rb_fraction, 0.0, 1.0)
    margin = available - demand
    return np.concatenate((np.clip(demand, 0, 2), np.clip(power / 10.0, 0, 5),
        np.clip(margin, -2, 2), np.clip((power[serving_bs] - power) / 10.0, -5, 5),
        np.clip(demand[serving_bs] - demand, -2, 2))).astype(np.float32)


def critic_local_feature_count(config):
    """Keep the 37-feature critic unchanged in the actor-only ablation."""
    if config.state_variant == "gain_derived":
        config = dataclasses.replace(config, state_variant="gain_report")
    return len(state_feature_names(config))


def predicted_actor_link_states(args, config, records, connection, vehicle_rate,
                               macro_loc=(0.0, 0.0)):
    """Reconstruct the original SINR/INR/demand slots from public reports only.

    The mean-rate load estimator follows the legacy ten-round, atol=1-RB
    refinement, without queue, pilot or HO corrections. Physical interfering
    occupancy is capped at one; demand pressure may exceed one. This context
    is exclusively for actor/critic observations, never the target optimizer.
    Best-pair next-frame predictions are not current tracked-pair measurements.
    """
    vehicles = sorted(connection, key=str)
    if not vehicles:
        return {}
    n = len(vehicles)
    capacities = np.array([args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs, dtype=float)
    bandwidth = np.array([args.RB_intervel_macro] + [args.RB_intervel_micro] * config.num_micro_bs)
    power = np.array([args.p_macro] + [args.p_micro] * config.num_micro_bs)
    noise = args.N0 * bandwidth * 10.0 ** (np.array(
        [args.NF_macro_dB] + [args.NF_micro_dB] * config.num_micro_bs) / 10.0)
    associations = np.array([connection[v] for v in vehicles], dtype=int)
    rates = np.array([vehicle_rate[v] for v in vehicles], dtype=float)
    if (np.any(associations < 0) or np.any(associations >= config.num_bs)
            or not np.isfinite(rates).all() or np.any(rates < 0)):
        raise ValueError("Invalid predicted actor association or arrival rate")
    desired = np.empty((n, config.num_bs))
    interfering = np.empty((n, config.num_micro_bs))
    gain_config = dataclasses.replace(config, state_variant="gain_report")
    for i, v in enumerate(vehicles):
        record = records[v]
        report = record["shared_prediction"]
        encode_prediction_report(gain_config, report)  # validate gains, never read beam/CSI
        desired[i, 0] = macro_gain_db(args, record["pos"], np.asarray(macro_loc))
        desired[i, 1:] = report["gain"]
        interfering[i] = report["interference"]
    desired = 10.0 ** (desired / 10.0)
    interfering = 10.0 ** (interfering / 10.0)

    def link_metrics(estimated_rb):
        occupancy = np.clip(estimated_rb / capacities, 0.0, 1.0)
        components = interfering * power[None, 1:] * occupancy[None, 1:]
        interference = np.zeros((n, config.num_bs))
        interference[:, 1:] = np.maximum(components.sum(axis=1)[:, None] - components, 0.0)
        sinr = power[None, :] * desired / (noise[None, :] + interference)
        return sinr, interference / noise[None, :]

    # Preserve the original estimator's initialization and finite iteration limit.
    estimated_rb = np.full(config.num_bs, args.num_RB_micro, dtype=float)
    for _ in range(10):
        sinr, _ = link_metrics(estimated_rb)
        rate_per_rb = bandwidth[None, :] * np.log2(1.0 + sinr)
        demand = rates / (rate_per_rb[np.arange(n), associations] + 2e-10)
        updated = np.bincount(associations, weights=demand, minlength=config.num_bs)
        converged = np.allclose(estimated_rb, updated, atol=1)
        estimated_rb = updated
        if converged:
            break
    sinr, inr = link_metrics(estimated_rb)
    load = np.clip(estimated_rb / capacities, 0.0, 1.5)
    result = {}
    for i, v in enumerate(vehicles):
        bs = associations[i]
        sinr_db = float(10.0 * np.log10(max(sinr[i, bs], 1e-30)))
        inr_db = float(10.0 * np.log10(inr[i, bs])) if inr[i, bs] > 0 else -np.inf
        result[v] = (sinr_db, inr_db, load.copy())
    return result


def make_local_state(
    config: OMAPPOConfig,
    position: Sequence[float],
    heading_deg: float,
    speed_mps: float,
    serving_bs: int,
    serving_sinr_db: float,
    queue_ratio: float,
    traffic_mbps: float,
    rb_load: Sequence[float],
    user_load: Sequence[float],
    interference_db: float,
    previous_handover: bool,
    system_throughput_ratio: float,
    own_rb_fraction: float,
    tx_beam: Optional[int],
    rx_beam: Optional[int],
    candidate_sinr_db: Optional[Sequence[float]] = None,
    candidate_demand_ratio: Optional[Sequence[float]] = None,
    candidate_residual_ratio: Optional[Sequence[float]] = None,
    candidate_feasibility_margin: Optional[Sequence[float]] = None,
    optimizer_feedback: Optional[Sequence[float]] = None,
    pilot_observation: Optional[np.ndarray] = None,
    prediction_report: Optional[MutableMapping] = None,
    derived_features: Optional[np.ndarray] = None,
    predicted_link_state: Optional[Tuple[float, float, np.ndarray]] = None,
) -> np.ndarray:
    if config.state_variant == "predicted_adapted":
        if predicted_link_state is None:
            raise ValueError("Predicted adapted state requires a report-only link context")
        serving_sinr_db, interference_db, rb_load = predicted_link_state
        if not np.isfinite(serving_sinr_db) or not (np.isfinite(interference_db) or interference_db == -np.inf):
            raise ValueError("Invalid predicted link metrics")
    one_hot = np.zeros(config.num_bs, dtype=np.float32)
    one_hot[int(serving_bs)] = 1.0
    values: List[float] = [
        float(previous_handover),
        float(np.clip(system_throughput_ratio, 0.0, 2.0)),
        float(np.clip(own_rb_fraction, 0.0, 2.0)),
    ]
    counts = np.asarray(user_load, dtype=float)
    if counts.shape != (config.num_bs,):
        raise ValueError("user-load vector has wrong shape")
    values.extend(float(x) for x in np.clip(counts / 40.0, 0.0, 2.0))
    values.extend(float(x) for x in one_hot)
    if config.state_variant == "source":
        state = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            raise ValueError("non-finite source state")
        return state

    pos = np.clip(np.asarray(position, dtype=float) / 500.0, -1.5, 1.5)
    heading = math.radians(float(heading_deg) % 360.0)
    values.extend(
        [
            float(pos[0]),
            float(pos[1]),
            math.sin(heading),
            math.cos(heading),
            float(np.clip(speed_mps / 20.0, 0.0, 2.0)),
            float(np.clip((serving_sinr_db - 10.0) / 30.0, -2.0, 2.0)),
            float(np.clip(queue_ratio / 5.0, 0.0, 2.0)),
            float(np.clip(traffic_mbps / 20.0, 0.0, 2.0)),
        ]
    )
    loads = np.asarray(rb_load, dtype=float)
    if loads.shape != (config.num_bs,):
        raise ValueError("RB-load vector has wrong shape")
    values.extend(float(x) for x in np.clip(loads, 0.0, 2.0))
    values.append(
        -2.0
        if not np.isfinite(interference_db)
        else float(np.clip(interference_db / 30.0, -2.0, 2.0))
    )

    def cyclic(index: Optional[int], size: int) -> Tuple[float, float]:
        if index is None:
            return 0.0, 0.0
        angle = 2.0 * math.pi * int(index) / size
        return math.sin(angle), math.cos(angle)

    values.extend(cyclic(tx_beam, config.num_tx_beams))
    values.extend(cyclic(rx_beam, config.num_rx_beams))
    if config.state_variant in ("pilot", "report", "gain_report", "gain_derived"):
        # Drop privileged serving SINR and inferred interference, not merely
        # rename them. Retain public load, queues, mobility and beam indices.
        legacy_names = state_feature_names(dataclasses.replace(config, state_variant="adapted"))
        values = [value for name, value in zip(legacy_names, values)
                  if name not in ("serving_sinr", "interference_to_noise")]
    if config.state_variant == "pilot":
        pilot = np.asarray(pilot_observation, dtype=np.float32)
        if pilot.shape != (128,) or not np.isfinite(pilot).all():
            raise ValueError("pilot state requires the common 128-dimensional observation")
        values.extend(pilot.tolist())
    if config.state_variant in ("report", "gain_report", "gain_derived"):
        values.extend(encode_prediction_report(config, prediction_report).tolist())
    if config.state_variant == "gain_derived":
        derived = np.asarray(derived_features, dtype=np.float32)
        if derived.shape != (5 * config.num_bs,) or not np.isfinite(derived).all():
            raise ValueError("Invalid derived actor features")
        values.extend(derived.tolist())
    if config.state_variant == "feasibility":
        feature_groups = (
            (candidate_sinr_db, "candidate_sinr_db"),
            (candidate_demand_ratio, "candidate_demand_ratio"),
            (candidate_residual_ratio, "candidate_residual_ratio"),
            (candidate_feasibility_margin, "candidate_feasibility_margin"),
        )
        converted = []
        for feature, label in feature_groups:
            if feature is None:
                raise ValueError("{} is required for feasibility state".format(label))
            array = np.asarray(feature, dtype=float)
            if array.shape != (config.num_bs,) or not np.isfinite(array).all():
                raise ValueError("invalid {}".format(label))
            converted.append(array)
        sinr, demand, residual, margin = converted
        values.extend(
            float(x)
            for x in np.clip((sinr - 10.0) / 30.0, -2.0, 2.0)
        )
        values.extend(float(x) for x in np.clip(demand, 0.0, 2.0))
        values.extend(float(x) for x in np.clip(residual, -1.0, 1.0))
        values.extend(float(x) for x in np.clip(margin, -2.0, 2.0))
        feedback = np.asarray(optimizer_feedback, dtype=float)
        if feedback.shape != (4,) or not np.isfinite(feedback).all():
            raise ValueError("invalid optimizer_feedback")
        values.extend(
            (
                float(np.clip(feedback[0], 0.0, 2.0)),
                float(np.clip(feedback[1], 0.0, 2.0)),
                float(np.clip(feedback[2], 0.0, 2.0)),
                float(np.clip(feedback[3], 0.0, 1.0)),
            )
        )
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (len(state_feature_names(config)),) or not np.all(np.isfinite(state)):
        raise ValueError("invalid adapted O-MAPPO state")
    return state


def append_state_sequence(
    history: List[np.ndarray], state: np.ndarray, sequence_length: int
) -> np.ndarray:
    """Append one event observation and return a left-padded GRU sequence."""

    current = np.asarray(state, dtype=np.float32)
    if current.ndim != 1 or not np.isfinite(current).all():
        raise ValueError("state must be a finite vector")
    history.append(current.copy())
    del history[:-sequence_length]
    padding = [np.zeros_like(current) for _ in range(sequence_length - len(history))]
    return np.stack(padding + history).astype(np.float32, copy=False)


def candidate_feasibility_context(
    args,
    record: MutableMapping,
    learner: OMAPPOLearnerState,
    backlog_bits: float,
    load: Sequence[float],
    config: OMAPPOConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-BS SINR, demand, residual capacity, and feasibility margin.

    Alternative micro links include full-sweep overhead; the current micro link
    uses its tracked beam and local-tracking overhead.  These are causal
    current-frame estimates, not future channel predictions.
    """

    loads = np.asarray(load, dtype=float)
    if loads.shape != (config.num_bs,) or not np.isfinite(loads).all():
        raise ValueError("invalid load vector")
    no_bf = np.concatenate(([-180.0], no_bf_gain_db(record["h"])))
    duration = args.slots_per_frame * args.slot_len
    sinr = np.zeros(config.num_bs, dtype=float)
    demand = np.zeros(config.num_bs, dtype=float)
    residual = 1.0 - loads
    current_bs = int(learner.action)
    for bs in range(config.num_bs):
        if bs == 0:
            gain = macro_gain_db(args, record["pos"], np.asarray(macro_bs_loc))
            interference = -np.inf
            pilots = 0.0
            capacity_rb = args.num_RB_macro
        else:
            if (
                bs == current_bs
                and learner.tx_beam is not None
                and learner.rx_beam is not None
            ):
                gain = fixed_pair_gain_db(
                    record["h"],
                    bs - 1,
                    int(learner.tx_beam),
                    int(learner.rx_beam),
                    dft_tx,
                    dft_rx,
                )
                pilots = float(config.tracking_pilots)
            else:
                _, _, gain = best_beam_pair(
                    record["h"], bs - 1, dft_tx, dft_rx
                )
                pilots = average_sweep_pilots(
                    args, config.full_sweep_pilots, config.tracking_pilots
                )
            interference = _interference_db(args, bs, no_bf, loads)
            capacity_rb = args.num_RB_micro
        capacity = _capacity_per_rb_bps(args, bs, gain, interference, pilots)
        required_rb = float(backlog_bits) / max(capacity * duration, 1e-12)
        demand[bs] = required_rb / max(capacity_rb, 1)
        sinr[bs] = effective_sinr_db(args, bs, gain, interference)
    margin = residual - demand
    return sinr, demand, residual, margin


def make_global_state(local_states: np.ndarray, num_vehicles: int, feature_count=None) -> np.ndarray:
    """Pool a changing number of UE observations for the centralized critic."""

    local = np.asarray(local_states, dtype=np.float32)
    if local.ndim != 2 or local.shape[0] == 0:
        raise ValueError("local_states must be a nonempty matrix")
    if feature_count is not None:
        if not 1 <= feature_count <= local.shape[1]:
            raise ValueError("Invalid critic feature count")
        local = local[:, :feature_count]
    pooled = np.concatenate(
        (
            local.mean(axis=0),
            local.min(axis=0),
            local.max(axis=0),
            np.asarray([min(float(num_vehicles) / 150.0, 2.0)], dtype=np.float32),
        )
    )
    return pooled.astype(np.float32, copy=False)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        width = int(input_dim)
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(width, int(hidden)), nn.ReLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, int(output_dim)))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _GRUNet(nn.Module):
    """Fixed-window recurrent encoder followed by a small prediction head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        recurrent_hidden_size: int,
        head_hidden_sizes: Sequence[int],
    ):
        super().__init__()
        self.gru = nn.GRU(
            int(input_dim), int(recurrent_hidden_size), batch_first=True
        )
        self.head = _MLP(
            int(recurrent_hidden_size), int(output_dim), head_hidden_sizes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("GRU input must have shape [batch, time, features]")
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


@dataclasses.dataclass
class PPOTransition:
    vehicle: object
    local_state: np.ndarray
    global_state: np.ndarray
    action: int
    old_log_probability: float
    old_value: float
    reward: float
    next_value: float
    done: bool


class OMAPPOMemory:
    def __init__(self) -> None:
        self.transitions: List[PPOTransition] = []

    def add(self, transition: PPOTransition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)


class OMAPPPolicy:
    """Parameter-shared binary actors with a pooled centralized critic."""

    def __init__(self, config: OMAPPOConfig, seed: int = 1):
        config.validate()
        self.config = config
        self.feature_names = state_feature_names(config)
        self.local_dim = len(self.feature_names)
        self.critic_local_dim = critic_local_feature_count(config)
        self.global_dim = 3 * self.critic_local_dim + 1
        torch.set_num_threads(max(1, int(config.torch_threads)))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        network = _GRUNet if config.recurrent else _MLP
        if config.recurrent:
            self.actor = network(
                self.local_dim,
                2,
                config.recurrent_hidden_size,
                config.hidden_sizes,
            )
            self.critic = network(
                self.global_dim,
                1,
                config.recurrent_hidden_size,
                config.hidden_sizes,
            )
        elif config.state_variant == "gain_derived":
            # Match the from-scratch gain-report initialization exactly on
            # common weights. Zero new input columns preserve initial policy
            # outputs; gradients can learn their contribution immediately.
            base_actor = _MLP(self.critic_local_dim, 2, config.hidden_sizes)
            self.critic = _MLP(self.global_dim, 1, config.hidden_sizes)
            rng_state = torch.random.get_rng_state()
            self.actor = _MLP(self.local_dim, 2, config.hidden_sizes)
            with torch.no_grad():
                for index, (old, new) in enumerate(zip(base_actor.model, self.actor.model)):
                    if isinstance(old, nn.Linear):
                        new.bias.copy_(old.bias)
                        if index == 0:
                            new.weight.zero_()
                            new.weight[:, :self.critic_local_dim].copy_(old.weight)
                        else:
                            new.weight.copy_(old.weight)
            torch.random.set_rng_state(rng_state)
        else:
            self.actor = network(self.local_dim, 2, config.hidden_sizes)
            self.critic = network(self.global_dim, 1, config.hidden_sizes)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )
        self.update_count = 0
        self.decision_count = 0

    def act(
        self,
        local_states: np.ndarray,
        global_state: np.ndarray,
        explore: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_array = np.asarray(local_states, dtype=np.float32)
        global_array = np.asarray(global_state, dtype=np.float32)
        if self.config.recurrent:
            expected_local = (
                local_array.ndim == 3
                and local_array.shape[1:] == (
                    self.config.recurrent_sequence_length,
                    self.local_dim,
                )
            )
            expected_global = global_array.shape == (
                self.config.recurrent_sequence_length,
                self.global_dim,
            )
            if not expected_local or not expected_global:
                raise ValueError("invalid recurrent actor/critic input shapes")
        else:
            if local_array.ndim != 2 or local_array.shape[1] != self.local_dim:
                raise ValueError("invalid MLP actor input shape")
            if global_array.shape != (self.global_dim,):
                raise ValueError("invalid MLP critic input shape")
        local_t = torch.as_tensor(local_array, dtype=torch.float32)
        global_t = torch.as_tensor(global_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(local_t))
            if explore:
                actions_t = distribution.sample()
            else:
                # Argmax deterministically prefers action 0 on exact ties.
                actions_t = distribution.logits.argmax(dim=-1)
            log_prob_t = distribution.log_prob(actions_t)
            value = float(self.critic(global_t).squeeze().cpu())
        self.decision_count += int(local_t.shape[0])
        return (
            actions_t.cpu().numpy().astype(np.int64),
            log_prob_t.cpu().numpy().astype(np.float32),
            np.full(local_t.shape[0], value, dtype=np.float32),
        )

    def value(self, global_state: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0)
            return float(self.critic(tensor).squeeze().cpu())

    def update(self, memory: OMAPPOMemory) -> Dict[str, float]:
        if not memory.transitions:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "transitions": 0.0,
            }
        config = self.config
        transitions = memory.transitions
        by_vehicle: Dict[object, List[int]] = collections.defaultdict(list)
        for index, transition in enumerate(transitions):
            by_vehicle[transition.vehicle].append(index)
        advantages = np.zeros(len(transitions), dtype=np.float32)
        returns = np.zeros(len(transitions), dtype=np.float32)
        for indices in by_vehicle.values():
            gae = 0.0
            for index in reversed(indices):
                item = transitions[index]
                continuation = 0.0 if item.done else 1.0
                delta = (
                    item.reward
                    + config.discount_factor * continuation * item.next_value
                    - item.old_value
                )
                gae = (
                    delta
                    + config.discount_factor
                    * config.gae_lambda
                    * continuation
                    * gae
                )
                advantages[index] = gae
                returns[index] = gae + item.old_value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        local = np.stack([x.local_state for x in transitions]).astype(np.float32)
        global_states = np.stack([x.global_state for x in transitions]).astype(np.float32)
        actions = np.asarray([x.action for x in transitions], dtype=np.int64)
        old_log_prob = np.asarray(
            [x.old_log_probability for x in transitions], dtype=np.float32
        )
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []
        order = np.arange(len(transitions))
        for _ in range(config.ppo_epochs):
            self.rng.shuffle(order)
            for start in range(0, len(order), config.batch_size):
                batch = order[start : start + config.batch_size]
                local_t = torch.as_tensor(local[batch], dtype=torch.float32)
                global_t = torch.as_tensor(global_states[batch], dtype=torch.float32)
                action_t = torch.as_tensor(actions[batch], dtype=torch.int64)
                old_log_t = torch.as_tensor(old_log_prob[batch], dtype=torch.float32)
                advantage_t = torch.as_tensor(advantages[batch], dtype=torch.float32)
                return_t = torch.as_tensor(returns[batch], dtype=torch.float32)

                distribution = Categorical(logits=self.actor(local_t))
                log_prob = distribution.log_prob(action_t)
                ratio = torch.exp(log_prob - old_log_t)
                unclipped = ratio * advantage_t
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                ) * advantage_t
                entropy = distribution.entropy().mean()
                actor_loss = -torch.minimum(unclipped, clipped).mean() - (
                    config.entropy_coefficient * entropy
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), config.gradient_clip_norm
                )
                self.actor_optimizer.step()

                values = self.critic(global_t).squeeze(-1)
                critic_loss = config.value_coefficient * torch.mean(
                    (values - return_t) ** 2
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), config.gradient_clip_norm
                )
                self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
        self.update_count += 1
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "transitions": float(len(transitions)),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "config": dataclasses.asdict(self.config),
                "feature_names": self.feature_names,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self.update_count,
                "decision_count": self.decision_count,
            },
            path,
        )

    @staticmethod
    def load(path: str, seed: int = 1, load_optimizers: bool = False) -> "OMAPPPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config_dict = dict(checkpoint["config"])
        config_dict["hidden_sizes"] = tuple(config_dict["hidden_sizes"])
        policy = OMAPPPolicy(OMAPPOConfig(**config_dict), seed=seed)
        policy.actor.load_state_dict(checkpoint["actor"])
        policy.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            policy.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            policy.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        policy.update_count = int(checkpoint.get("update_count", 0))
        policy.decision_count = int(checkpoint.get("decision_count", 0))
        return policy


def average_sweep_pilots(args, total_sweep_pilots: int, tracking_pilots: int) -> float:
    overhead = 0.0
    for slot in range(args.slots_per_frame):
        pilots = sweep_pilots_for_slot(
            total_sweep_pilots,
            tracking_pilots,
            slot,
            args.pilot_overhead_factor,
        )
        overhead += min(pilots * args.pilot_overhead_factor, 1.0)
    return overhead / args.slots_per_frame / args.pilot_overhead_factor


@functools.lru_cache(maxsize=32)
def _cached_report_sweep_average(slots, factor, total, tracking):
    overhead = sum(min(sweep_pilots_for_slot(total, tracking, slot, factor) * factor, 1.0)
                   for slot in range(slots))
    return overhead / slots / factor


@dataclasses.dataclass
class TargetCandidate:
    bs: int
    tx_beam: Optional[int]
    rx_beam: Optional[int]
    gain_db: float
    required_rb: float
    base_cost: float


@dataclasses.dataclass
class TargetOptimizationResult:
    targets: Dict[object, int]
    estimated_load: np.ndarray
    overflow: np.ndarray
    objective: float
    solver_success: bool
    elapsed_s: float


def record_optimizer_feedback(
    args,
    learners: Dict[object, OMAPPOLearnerState],
    event_vehicles: Sequence[object],
    actions: Sequence[int],
    result: TargetOptimizationResult,
) -> None:
    """Store causal optimizer outcome features for each event vehicle."""

    capacities = np.asarray(
        [args.num_RB_macro] + [args.num_RB_micro] * (len(result.overflow) - 1),
        dtype=float,
    )
    total_overflow = float(result.overflow.sum() / max(capacities.sum(), 1.0))
    for vehicle, action in zip(event_vehicles, actions):
        target = (
            int(result.targets[vehicle])
            if int(action) == 1
            else int(learners[vehicle].action)
        )
        learners[vehicle].optimizer_feedback = np.asarray(
            [
                total_overflow,
                float(result.estimated_load[target]),
                float(result.overflow[target] / max(capacities[target], 1.0)),
                float(result.solver_success),
            ],
            dtype=np.float32,
        )


def _candidate_links(
    args,
    vehicle: object,
    record: MutableMapping,
    current_bs: int,
    backlog_bits: float,
    load: np.ndarray,
    config: OMAPPOConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
    macro_bs_loc: np.ndarray,
) -> List[TargetCandidate]:
    predicted = config.information_mode == "shared_prediction"
    if predicted:
        prediction = record["shared_prediction"]
        no_bf = np.concatenate(([-180.0], prediction["interference"]))
    else:
        no_bf = np.concatenate(([-180.0], no_bf_gain_db(record["h"])))
    frame_duration = args.slots_per_frame * args.slot_len
    sweep_average = average_sweep_pilots(
        args, config.full_sweep_pilots, config.tracking_pilots
    )
    candidates: List[TargetCandidate] = []
    for bs in range(config.num_bs):
        if bs == current_bs:
            continue
        if bs == 0:
            tx = rx = None
            gain = macro_gain_db(args, record["pos"], macro_bs_loc)
            interference = -np.inf
            pilot_average = 0.0
            power = args.p_macro
            rb_capacity = args.num_RB_macro
        else:
            if predicted:
                tx = rx = None  # Target assignment does not choose a beam.
                gain = float(prediction["gain"][bs - 1])
            else:
                tx, rx, gain = best_beam_pair(record["h"], bs - 1, dft_tx, dft_rx)
            interference = _interference_db(args, bs, no_bf, load)
            pilot_average = sweep_average
            power = args.p_micro
            rb_capacity = args.num_RB_micro
        capacity = _capacity_per_rb_bps(
            args, bs, gain, interference, pilot_average
        )
        required = backlog_bits / max(capacity * frame_duration, 1e-12)
        normalized_demand = required / max(rb_capacity, 1)
        cost = normalized_demand
        if config.optimizer_variant in ("load", "load_energy"):
            cost += config.optimizer_load_weight * float(load[bs]) * normalized_demand
        if config.optimizer_variant == "load_energy":
            # Normalize against one macro-RB watt so the coefficient is
            # dimensionless and stable across the two bandwidths.
            cost += config.optimizer_energy_weight * required * power
        # Only capacity occupancy is inflated. Frame-average energy cost is
        # unchanged, as in the common HO-interruption capacity correction.
        if config.ho_interruption_ms:
            required /= 1.0 - config.ho_interruption_ms / (1000.0 * frame_duration)
        candidates.append(
            TargetCandidate(
                bs=bs,
                tx_beam=tx,
                rx_beam=rx,
                gain_db=float(gain),
                required_rb=float(required),
                base_cost=float(cost),
            )
        )
    candidates.sort(key=lambda x: (x.base_cost, x.bs))
    return candidates[: config.candidate_count]


def optimize_triggered_targets(
    args,
    records: MutableMapping,
    learners: Dict[object, OMAPPOLearnerState],
    triggered: Sequence[object],
    backlog_bits: Dict[object, float],
    current_allocated_rb: Dict[object, float],
    previous_load: np.ndarray,
    config: OMAPPOConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
    solver: Optional[str] = None,
) -> TargetOptimizationResult:
    """Assign triggered UEs to three target links under BS capacities.

    A continuous overload slack makes the target problem feasible even when
    the offered load exceeds physical capacity.  Its large objective penalty
    plays the same role as the source method's QoS constraints while allowing
    meaningful overloaded-network tests.
    """

    started = time.perf_counter()
    vehicles = sorted(triggered, key=str)
    capacities = np.asarray(
        [args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs,
        dtype=float,
    )
    fixed = np.zeros(config.num_bs, dtype=float)
    triggered_set = set(vehicles)
    for vehicle, learner in learners.items():
        if vehicle in triggered_set or vehicle not in records:
            continue
        fixed[int(learner.action)] += float(current_allocated_rb.get(vehicle, 0.0))
    residual = capacities - fixed
    candidates = {
        vehicle: _candidate_links(
            args,
            vehicle,
            records[vehicle],
            int(learners[vehicle].action),
            float(backlog_bits[vehicle]),
            np.asarray(previous_load, dtype=float),
            config,
            dft_tx,
            dft_rx,
            np.asarray(macro_bs_loc, dtype=float),
        )
        for vehicle in vehicles
    }
    if not vehicles:
        return TargetOptimizationResult(
            targets={},
            estimated_load=np.clip(fixed / capacities, 0.0, np.inf),
            overflow=np.maximum(fixed - capacities, 0.0),
            objective=0.0,
            solver_success=True,
            elapsed_s=time.perf_counter() - started,
        )

    chosen_solver = solver or config.optimizer_solver
    targets: Dict[object, int] = {}
    objective = 0.0
    assigned = fixed.copy()
    success = True
    if chosen_solver == "greedy":
        # Assign hard-to-place users first, then price the marginal overflow.
        ordering = sorted(
            vehicles,
            key=lambda vehicle: (
                -(candidates[vehicle][1].base_cost - candidates[vehicle][0].base_cost)
                if len(candidates[vehicle]) > 1
                else -1e9,
                str(vehicle),
            ),
        )
        for vehicle in ordering:
            best: Optional[TargetCandidate] = None
            best_augmented = np.inf
            for candidate in candidates[vehicle]:
                before = max(assigned[candidate.bs] - capacities[candidate.bs], 0.0)
                after = max(
                    assigned[candidate.bs]
                    + candidate.required_rb
                    - capacities[candidate.bs],
                    0.0,
                )
                overflow_increment = (after - before) / capacities[candidate.bs]
                augmented = (
                    candidate.base_cost
                    + config.optimizer_overflow_penalty * overflow_increment
                )
                if augmented < best_augmented:
                    best_augmented = augmented
                    best = candidate
            assert best is not None
            targets[vehicle] = best.bs
            assigned[best.bs] += best.required_rb
            objective += best_augmented
    else:
        edges: List[Tuple[object, TargetCandidate]] = [
            (vehicle, candidate)
            for vehicle in vehicles
            for candidate in candidates[vehicle]
        ]
        num_edges = len(edges)
        num_variables = num_edges + config.num_bs
        costs = np.zeros(num_variables, dtype=float)
        costs[:num_edges] = [candidate.base_cost for _, candidate in edges]
        costs[num_edges:] = config.optimizer_overflow_penalty
        integrality = np.zeros(num_variables, dtype=int)
        integrality[:num_edges] = 1
        lower = np.zeros(num_variables, dtype=float)
        upper = np.concatenate((np.ones(num_edges), np.full(config.num_bs, np.inf)))
        matrix = lil_matrix((len(vehicles) + config.num_bs, num_variables))
        lower_constraint = np.full(len(vehicles) + config.num_bs, -np.inf)
        upper_constraint = np.zeros(len(vehicles) + config.num_bs)
        edge_by_vehicle: Dict[object, List[int]] = collections.defaultdict(list)
        for edge_index, (vehicle, candidate) in enumerate(edges):
            edge_by_vehicle[vehicle].append(edge_index)
            row = len(vehicles) + candidate.bs
            matrix[row, edge_index] = candidate.required_rb / capacities[candidate.bs]
        for vehicle_index, vehicle in enumerate(vehicles):
            for edge_index in edge_by_vehicle[vehicle]:
                matrix[vehicle_index, edge_index] = 1.0
            lower_constraint[vehicle_index] = 1.0
            upper_constraint[vehicle_index] = 1.0
        for bs in range(config.num_bs):
            matrix[len(vehicles) + bs, num_edges + bs] = -1.0
            upper_constraint[len(vehicles) + bs] = residual[bs] / capacities[bs]
        result = milp(
            c=costs,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(
                matrix.tocsr(), lower_constraint, upper_constraint
            ),
            options={"time_limit": 2.0, "mip_rel_gap": 1e-3},
        )
        success = bool(result.success and result.x is not None)
        if success:
            objective = float(result.fun)
            for vehicle in vehicles:
                edge_indices = edge_by_vehicle[vehicle]
                edge_index = max(edge_indices, key=lambda x: result.x[x])
                candidate = edges[edge_index][1]
                targets[vehicle] = candidate.bs
                assigned[candidate.bs] += candidate.required_rb
        else:
            # A solver timeout never changes experiment semantics: use the
            # deterministic feasible-with-slack greedy fallback.
            fallback = optimize_triggered_targets(
                args,
                records,
                learners,
                vehicles,
                backlog_bits,
                current_allocated_rb,
                previous_load,
                dataclasses.replace(config, optimizer_solver="greedy"),
                dft_tx,
                dft_rx,
                macro_bs_loc,
                solver="greedy",
            )
            fallback.solver_success = False
            return fallback
    overflow = np.maximum(assigned - capacities, 0.0)
    return TargetOptimizationResult(
        targets=targets,
        estimated_load=assigned / capacities,
        overflow=overflow,
        objective=float(objective),
        solver_success=success,
        elapsed_s=time.perf_counter() - started,
    )


@dataclasses.dataclass
class OMAPPOFluidStep:
    queue_end: Dict[object, float]
    served_bits: Dict[object, float]
    allocated_rb: Dict[object, float]
    user_power_w: Dict[object, float]
    serving_gain_db: Dict[object, float]
    interference_db: Dict[object, float]
    spectral_efficiency: Dict[object, float]
    load_ratio: np.ndarray
    user_load: np.ndarray
    connection: Dict[object, int]


def fluid_o_mappo_step(
    args,
    records: MutableMapping,
    learners: Dict[object, OMAPPOLearnerState],
    queues: Dict[object, float],
    vehicle_rate: Dict[object, float],
    previous_load: np.ndarray,
    macro_bs_loc: np.ndarray,
    config: OMAPPOConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> OMAPPOFluidStep:
    vehicles = sorted(records.keys(), key=str)
    duration = args.slots_per_frame * args.slot_len
    connection = {vehicle: int(learners[vehicle].action) for vehicle in vehicles}
    serving_gain: Dict[object, float] = {}
    no_bf: Dict[object, np.ndarray] = {}
    pilot_average: Dict[object, float] = {}
    backlog: Dict[object, float] = {}
    for vehicle in vehicles:
        bs = connection[vehicle]
        record = records[vehicle]
        macro_gain = macro_gain_db(args, record["pos"], macro_bs_loc)
        if bs == 0:
            gain = macro_gain
        else:
            learner = learners[vehicle]
            if learner.tx_beam is None or learner.rx_beam is None:
                raise RuntimeError("micro link has no O-MAPPO beam pair")
            gain = fixed_pair_gain_db(
                record["h"],
                bs - 1,
                int(learner.tx_beam),
                int(learner.rx_beam),
                dft_tx,
                dft_rx,
            )
        serving_gain[vehicle] = gain
        no_bf[vehicle] = np.concatenate(([macro_gain], no_bf_gain_db(record["h"])))
        pilots = 0.0
        if bs > 0:
            total = learners[vehicle].current_sweep_pilots
            if total > 0:
                pilots = average_sweep_pilots(args, total, config.tracking_pilots)
            else:
                pilots = float(config.tracking_pilots)
        pilot_average[vehicle] = pilots
        backlog[vehicle] = queues[vehicle] + vehicle_rate[vehicle] * duration
    allocation, capacity, load, interference = _fluid_allocation(
        args,
        vehicles,
        connection,
        serving_gain,
        no_bf,
        pilot_average,
        backlog,
        previous_load,
        duration,
        service_fraction=({v: 1.0 - (config.ho_interruption_ms / (1000.0 * duration)
                                   if learners[v].last_handover else 0.0)
                           for v in vehicles} if config.ho_interruption_ms else None),
    )
    queue_end: Dict[object, float] = {}
    served: Dict[object, float] = {}
    user_power: Dict[object, float] = {}
    spectral_efficiency: Dict[object, float] = {}
    for vehicle in vehicles:
        bs = connection[vehicle]
        served[vehicle] = min(
            backlog[vehicle], allocation[vehicle] * capacity[vehicle] * duration
        )
        queue_end[vehicle] = max(0.0, backlog[vehicle] - served[vehicle])
        power = args.p_macro if bs == 0 else args.p_micro
        user_power[vehicle] = allocation[vehicle] * power
        bandwidth = args.RB_intervel_macro if bs == 0 else args.RB_intervel_micro
        spectral_efficiency[vehicle] = capacity[vehicle] / bandwidth
        learners[vehicle].previous_rb_fraction = allocation[vehicle] / (
            args.num_RB_macro if bs == 0 else args.num_RB_micro
        )
    user_load = np.asarray(
        [sum(bs == index for bs in connection.values()) for index in range(config.num_bs)],
        dtype=float,
    )
    return OMAPPOFluidStep(
        queue_end=queue_end,
        served_bits=served,
        allocated_rb=allocation,
        user_power_w=user_power,
        serving_gain_db=serving_gain,
        interference_db=interference,
        spectral_efficiency=spectral_efficiency,
        load_ratio=load,
        user_load=user_load,
        connection=connection,
    )


def _source_team_reward(
    throughput_ratio: float,
    mean_delay_s: float,
    trigger_fraction: float,
    config: OMAPPORewardConfig,
) -> float:
    # Scale the source paper's absolute throughput thresholds by current
    # offered traffic; retain its three-level (10 delta, delta, -delta) form.
    if throughput_ratio >= 0.98 and mean_delay_s <= 0.015:
        reward = 10.0
    elif throughput_ratio >= 0.90 and mean_delay_s <= 0.022:
        reward = 1.0
    else:
        reward = -1.0
    return float(reward - config.handover_weight * trigger_fraction)


def _adapted_rewards(
    reward_config: OMAPPORewardConfig,
    vehicles: Sequence[object],
    step: OMAPPOFluidStep,
    queue_upper_bound: float,
    offered_bits: float,
    outcomes: Dict[object, OMAPPOActionOutcome],
    full_sweep_pilots: int,
) -> Dict[object, float]:
    local: Dict[object, float] = {}
    # OTR caps actual load at one, so a pure ``max(load-1, 0)`` term would be
    # identically zero.  Squared utilization prices congestion before
    # saturation; any numerical/estimated overflow is added explicitly.
    congestion = float(
        np.mean(np.asarray(step.load_ratio, dtype=float) ** 2)
        + np.maximum(step.load_ratio - 1.0, 0.0).sum()
    )
    queue_ratios = np.asarray(
        [
            step.queue_end[vehicle] / max(queue_upper_bound, 1e-12)
            for vehicle in vehicles
        ],
        dtype=float,
    )
    # CVaR is applied to the excess above the queue constraint.  Unlike a
    # mean-queue penalty, it keeps the rare persistently starved vehicles
    # visible in a 100+ vehicle team reward without penalizing healthy queues.
    capped_excess = np.maximum(np.minimum(queue_ratios, 10.0) - 1.0, 0.0)
    tail_count = max(
        1, int(math.ceil((1.0 - reward_config.cvar_alpha) * len(capped_excess)))
    )
    cvar_excess = (
        float(np.mean(np.partition(capped_excess, -tail_count)[-tail_count:]))
        if len(capped_excess)
        else 0.0
    )
    for index, vehicle in enumerate(vehicles):
        queue_ratio = float(queue_ratios[index])
        service_ratio = min(step.served_bits[vehicle] / max(offered_bits, 1e-12), 2.0)
        local[vehicle] = float(
            reward_config.reward_offset
            + reward_config.service_weight * service_ratio
            - reward_config.queue_weight * min(queue_ratio, 10.0)
            - reward_config.violation_weight * float(queue_ratio > 1.0)
            - reward_config.energy_weight * step.user_power_w[vehicle]
            - reward_config.overload_weight * congestion
            - reward_config.handover_weight * float(outcomes[vehicle].handover)
            - reward_config.sweep_weight
            * outcomes[vehicle].sweep_pilots
            / max(full_sweep_pilots, 1)
        )
    team = float(np.mean(list(local.values()))) if local else 0.0
    return {
        vehicle: reward_config.team_mix * team
        + (1.0 - reward_config.team_mix) * local[vehicle]
        - reward_config.cvar_weight * cvar_excess
        for vehicle in vehicles
    }


def _finalize_transition(
    learner: OMAPPOLearnerState,
    vehicle: object,
    memory: OMAPPOMemory,
    next_value: float,
    done: bool,
) -> Optional[float]:
    if learner.transition_local_state is None:
        return None
    reward = learner.transition_reward / max(learner.transition_frames, 1)
    memory.add(
        PPOTransition(
            vehicle=vehicle,
            local_state=learner.transition_local_state,
            global_state=learner.transition_global_state,
            action=int(learner.transition_action_binary),
            old_log_probability=float(learner.transition_log_probability),
            old_value=float(learner.transition_value),
            reward=float(reward),
            next_value=float(next_value),
            done=bool(done),
        )
    )
    return float(reward)


def run_fluid_o_mappo_episode(
    args,
    timeline_dir: MutableMapping,
    policy: OMAPPPolicy,
    reward_config: OMAPPORewardConfig,
    data_rate_mbps: float,
    seed: int = 1,
    learn: bool = False,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
) -> Dict[str, float]:
    """Collect an on-policy episode in the common frame-level surrogate."""

    reward_config.validate()
    config = policy.config
    torch.manual_seed(seed)
    dft_tx = generate_dft_codebook(config.num_tx_beams)
    dft_rx = generate_dft_codebook(config.num_rx_beams)
    macro_loc = np.asarray(macro_bs_loc, dtype=float)
    frames = list(timeline_dir.keys())
    if len(frames) < 2:
        raise ValueError("timeline must contain at least two frames")
    duration = args.slots_per_frame * args.slot_len
    rate = float(data_rate_mbps) * 1e6
    queue_upper_bound = rate * args.lat_slot_ub * args.slot_len
    offered_bits = rate * duration
    learners: Dict[object, OMAPPOLearnerState] = {}
    queues: Dict[object, float] = {}
    previous_load = np.zeros(config.num_bs, dtype=float)
    memory = OMAPPOMemory()
    rewards: List[float] = []
    powers: List[float] = []
    violations: List[float] = []
    delays: List[float] = []
    cvar_excesses: List[float] = []
    max_queue_ratios: List[float] = []
    counts: List[float] = []
    decisions = triggers = handovers = beam_switches = 0
    optimizer_calls = optimizer_failures = 0
    optimizer_time = 0.0
    trigger_counts = np.zeros(2, dtype=np.int64)
    global_state_history: List[np.ndarray] = []

    for frame in frames:
        records = timeline_dir[frame]
        present = set(records)
        departed = set(learners).difference(present)
        for vehicle in departed:
            reward = _finalize_transition(
                learners[vehicle], vehicle, memory, next_value=0.0, done=True
            )
            if reward is not None:
                rewards.append(reward)
            learners.pop(vehicle, None)
            queues.pop(vehicle, None)
        for vehicle in sorted(present, key=str):
            if vehicle not in learners:
                position = np.asarray(records[vehicle]["pos"], dtype=float)
                learners[vehicle] = OMAPPOLearnerState(
                    action=0,
                    rx_beam=None,
                    pending_action=None,
                    last_position=position.copy(),
                    distance_since_event=config.zone_size_m,
                    tx_beam=None,
                )
                queues[vehicle] = 0.5 * queue_upper_bound

        outcomes: Dict[object, OMAPPOActionOutcome] = {}
        for vehicle in sorted(present, key=str):
            learner = learners[vehicle]
            command = learner.pending_command
            learner.pending_command = None
            outcome = apply_o_mappo_command(
                learner, command, records[vehicle], config, dft_tx, dft_rx
            )
            outcomes[vehicle] = outcome
            handovers += int(outcome.handover)
            beam_switches += int(outcome.beam_switch)

        queues_before = queues.copy()
        vehicle_rate = {vehicle: rate for vehicle in present}
        step = fluid_o_mappo_step(
            args,
            records,
            learners,
            queues,
            vehicle_rate,
            previous_load,
            macro_loc,
            config,
            dft_tx,
            dft_rx,
        )
        previous_load = step.load_ratio
        queues = step.queue_end
        throughput_ratio = sum(step.served_bits.values()) / max(
            offered_bits * len(present), 1e-12
        )
        mean_delay = float(
            np.mean([queues[x] / rate for x in present]) if present else 0.0
        )
        trigger_fraction = float(
            np.mean([outcomes[x].handover for x in present]) if present else 0.0
        )
        if reward_config.source_threshold_reward:
            team_reward = _source_team_reward(
                throughput_ratio, mean_delay, trigger_fraction, reward_config
            )
            frame_rewards = {vehicle: team_reward for vehicle in present}
        else:
            frame_rewards = _adapted_rewards(
                reward_config,
                sorted(present, key=str),
                step,
                queue_upper_bound,
                offered_bits,
                outcomes,
                config.full_sweep_pilots,
            )
        for vehicle in present:
            learner = learners[vehicle]
            if learner.transition_local_state is not None:
                learner.transition_reward += frame_rewards[vehicle]
                learner.transition_frames += 1

        count = max(len(present), 1)
        powers.append(sum(step.user_power_w.values()))
        violations.append(
            sum(queues[x] > queue_upper_bound for x in present) / count
        )
        delays.append(sum(queues[x] / rate for x in present) / count)
        queue_ratios = np.asarray(
            [queues[x] / max(queue_upper_bound, 1e-12) for x in present],
            dtype=float,
        )
        tail_count = max(
            1,
            int(
                math.ceil(
                    (1.0 - reward_config.cvar_alpha) * len(queue_ratios)
                )
            ),
        )
        queue_excess = np.maximum(np.minimum(queue_ratios, 10.0) - 1.0, 0.0)
        cvar_excesses.append(
            float(np.mean(np.partition(queue_excess, -tail_count)[-tail_count:]))
        )
        max_queue_ratios.append(float(queue_ratios.max(initial=0.0)))
        counts.append(float(len(present)))

        zone_due: Dict[object, bool] = {}
        for vehicle in sorted(present, key=str):
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

        all_states: Dict[object, np.ndarray] = {}
        all_alt_sinr: Dict[object, List[float]] = {}
        system_user_load = step.user_load
        predicted_states = (predicted_actor_link_states(args, config, records,
            step.connection, vehicle_rate, macro_loc)
            if config.state_variant == "predicted_adapted" else {})
        for vehicle in sorted(present, key=str):
            learner = learners[vehicle]
            bs = int(step.connection[vehicle])
            serving_sinr = effective_sinr_db(
                args,
                bs,
                step.serving_gain_db[vehicle],
                step.interference_db[vehicle],
            )
            record = records[vehicle]
            no_bf = (None if config.information_mode == "shared_prediction" else
                     np.concatenate(([-180.0], no_bf_gain_db(record["h"]))))
            alternatives: List[float] = []
            # The overlap gate is evaluated only at a distance-zone crossing.
            # Periodic adapted candidates do not pay for unused alternative
            # beam searches in every intervening frame.
            if zone_due[vehicle] and config.trigger_gate == "source":
                for target in range(config.num_bs):
                    if target == bs:
                        continue
                    if target == 0:
                        gain = macro_gain_db(args, record["pos"], macro_loc)
                        interference = -np.inf
                    else:
                        _, _, gain = best_beam_pair(
                            record["h"], target - 1, dft_tx, dft_rx
                        )
                        interference = _interference_db(
                            args, target, no_bf, step.load_ratio
                        )
                    alternatives.append(
                        effective_sinr_db(args, target, gain, interference)
                    )
            all_alt_sinr[vehicle] = alternatives
            state_kwargs = shared_actor_inputs(config, record, args=args, serving_bs=bs,
                backlog_bits=queues_before[vehicle] + offered_bits, load=step.load_ratio,
                own_rb_fraction=learner.previous_rb_fraction, macro_loc=macro_loc)
            if config.state_variant == "predicted_adapted":
                state_kwargs["predicted_link_state"] = predicted_states[vehicle]
            if config.state_variant == "feasibility":
                context = candidate_feasibility_context(
                    args,
                    record,
                    learner,
                    queues_before[vehicle] + offered_bits,
                    step.load_ratio,
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
                    "optimizer_feedback": learner.optimizer_feedback,
                }
            all_states[vehicle] = make_local_state(
                config,
                record["pos"],
                float(record.get("angle", 0.0)),
                float(record.get("v", 0.0)),
                bs,
                serving_sinr,
                queues_before[vehicle] / queue_upper_bound,
                data_rate_mbps,
                step.load_ratio,
                system_user_load,
                step.interference_db[vehicle],
                learner.last_handover,
                throughput_ratio,
                learner.previous_rb_fraction,
                learner.tx_beam,
                learner.rx_beam,
                **state_kwargs,
            )
        ordered_present = sorted(present, key=str)
        global_state = make_global_state(
            np.stack([all_states[x] for x in ordered_present]), len(present),
            feature_count=critic_local_feature_count(config)
        )
        event_vehicles: List[object] = []
        for vehicle in ordered_present:
            learner = learners[vehicle]
            if not zone_due[vehicle]:
                continue
            bs = int(step.connection[vehicle])
            serving_sinr = effective_sinr_db(
                args,
                bs,
                step.serving_gain_db[vehicle],
                step.interference_db[vehicle],
            )
            if source_gate_allows(
                config, serving_sinr, all_alt_sinr[vehicle]
            ):
                event_vehicles.append(vehicle)

        if event_vehicles:
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
            actions, log_probabilities, values = policy.act(
                local_batch, policy_global_state, explore=learn
            )
            current_value = float(values[0])
            for vehicle in event_vehicles:
                reward = _finalize_transition(
                    learners[vehicle],
                    vehicle,
                    memory,
                    next_value=current_value,
                    done=False,
                )
                if reward is not None:
                    rewards.append(reward)
            triggered = [
                vehicle
                for vehicle, action in zip(event_vehicles, actions)
                if int(action) == 1
            ]
            backlog = {
                vehicle: queues_before[vehicle] + offered_bits for vehicle in present
            }
            optimization = optimize_triggered_targets(
                args,
                records,
                learners,
                triggered,
                backlog,
                step.allocated_rb,
                step.load_ratio,
                config,
                dft_tx,
                dft_rx,
                macro_loc,
            )
            record_optimizer_feedback(
                args, learners, event_vehicles, actions, optimization
            )
            if triggered:
                optimizer_calls += 1
                optimizer_failures += int(not optimization.solver_success)
                optimizer_time += optimization.elapsed_s
            for index, vehicle in enumerate(event_vehicles):
                action = int(actions[index])
                target = (
                    optimization.targets[vehicle]
                    if action == 1
                    else int(learners[vehicle].action)
                )
                learners[vehicle].pending_command = OMAPPOCommand(action, target)
                learners[vehicle].transition_local_state = local_batch[index].copy()
                learners[vehicle].transition_global_state = (
                    policy_global_state.copy()
                )
                learners[vehicle].transition_action_binary = action
                learners[vehicle].transition_log_probability = float(
                    log_probabilities[index]
                )
                learners[vehicle].transition_value = float(values[index])
                learners[vehicle].transition_reward = 0.0
                learners[vehicle].transition_frames = 0
                trigger_counts[action] += 1
            decisions += len(event_vehicles)
            triggers += len(triggered)

    for vehicle in list(learners):
        reward = _finalize_transition(
            learners[vehicle], vehicle, memory, next_value=0.0, done=True
        )
        if reward is not None:
            rewards.append(reward)
    update = policy.update(memory) if learn else {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "transitions": float(len(memory)),
    }
    episode_s = len(frames) * duration
    mean_count = float(np.mean(counts))
    return {
        "data_rate_mbps": float(data_rate_mbps),
        "learn": bool(learn),
        "decisions": float(decisions),
        "triggers": float(triggers),
        "trigger_ratio": float(triggers / max(decisions, 1)),
        "transitions": float(len(memory)),
        "mean_event_reward": float(np.mean(rewards)) if rewards else 0.0,
        "actor_loss": update["actor_loss"],
        "critic_loss": update["critic_loss"],
        "entropy": update["entropy"],
        "average_system_power_w": float(np.mean(powers)),
        "queue_violation_percent": float(100.0 * np.mean(violations)),
        "average_queueing_proxy_ms": float(1000.0 * np.mean(delays)),
        "queue_cvar_excess_ratio": float(np.mean(cvar_excesses)),
        "maximum_queue_ratio": float(np.max(max_queue_ratios)),
        "handover_per_vehicle_per_s": float(
            handovers / max(episode_s * mean_count, 1e-12)
        ),
        "beam_switch_per_vehicle_per_s": float(
            beam_switches / max(episode_s * mean_count, 1e-12)
        ),
        "average_vehicle_count": mean_count,
        "optimizer_calls": float(optimizer_calls),
        "optimizer_failure_count": float(optimizer_failures),
        "optimizer_ms_per_call": float(
            1000.0 * optimizer_time / max(optimizer_calls, 1)
        ),
        "action_fractions": (
            trigger_counts / max(trigger_counts.sum(), 1)
        ).astype(float).tolist(),
    }


def train_o_mappo(
    args,
    timeline_dir: MutableMapping,
    config: OMAPPOConfig,
    reward_config: OMAPPORewardConfig,
    data_rate_schedule_mbps: Sequence[float],
    epochs: int,
    seed: int = 1,
    verbose: bool = True,
) -> Tuple[OMAPPPolicy, List[Dict[str, float]]]:
    if epochs <= 0 or not data_rate_schedule_mbps:
        raise ValueError("epochs and rate schedule must be nonempty")
    policy = OMAPPPolicy(config, seed=seed)
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        rate = float(data_rate_schedule_mbps[epoch % len(data_rate_schedule_mbps)])
        result = run_fluid_o_mappo_episode(
            args,
            timeline_dir,
            policy,
            reward_config,
            rate,
            seed=seed + epoch,
            learn=True,
        )
        result["epoch"] = float(epoch + 1)
        history.append(result)
        if verbose:
            print(
                "{} epoch {:02d} rate={:g}: reward={:.3f}, actor={:.4f}, "
                "critic={:.4f}, power={:.2f} W, vio={:.2f}%, trigger={:.3f}".format(
                    reward_config.name,
                    epoch + 1,
                    rate,
                    result["mean_event_reward"],
                    result["actor_loss"],
                    result["critic_loss"],
                    result["average_system_power_w"],
                    result["queue_violation_percent"],
                    result["trigger_ratio"],
                )
            )
    return policy, history
