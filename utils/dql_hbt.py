"""Deep-Q-learning handover/beam-tracking baseline for MEET-COBRA.

The implementation adapts Khosravi et al., "Reinforcement Learning-based
Joint Handover and Beam Tracking in Millimeter-wave Networks".  The DQL
agent makes a *high-level* decision: track the serving link, or hand over to
one of the five data-serving BSs.  A handover to a micro BS performs a full
32x8 TX/RX sweep, while tracking searches a small neighbourhood around the
previous beam pair.  Thus beam selection remains a deterministic lower-level
procedure, as in the source method, rather than an output neuron of the DQN.

Training uses the frame-level fluid approximation already audited for the
adapted PQL-BA experiments.  Frozen policies are evaluated separately by the
slot-level simulator in :mod:`utils.dql_hbt_sim`.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import os
import random
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.beam_utils import generate_dft_codebook
from utils.mox_utils import dB2lin, lin2dB
from utils.pql_ba import (
    PQLBAConfig,
    _LearnerState,
    _learner_gain_db,
    action_serving_bs,
    best_beam_pair,
    fixed_pair_gain_db,
    macro_gain_db,
    no_bf_gain_db,
    sweep_pilots_for_slot,
)
from utils.pql_ba_adapted import _fluid_allocation


@dataclasses.dataclass(frozen=True)
class DQLHBTRewardConfig:
    """Dimensionless per-frame reward used between DQL decision epochs."""

    name: str
    reward_offset: float = 0.0
    spectral_efficiency_weight: float = 0.0
    spectral_efficiency_threshold: float = 1.0
    link_outage_weight: float = 0.0
    service_weight: float = 1.0
    queue_weight: float = 0.5
    queue_violation_weight: float = 4.0
    energy_weight: float = 0.0
    load_weight: float = 0.0
    handover_weight: float = 0.0
    sweep_weight: float = 0.0
    queue_penalty_cap: float = 10.0
    service_reward_cap: float = 2.0

    def validate(self) -> None:
        for key, value in dataclasses.asdict(self).items():
            if key != "name" and float(value) < 0.0:
                raise ValueError("{} must be nonnegative".format(key))


def dql_reward_presets() -> "collections.OrderedDict[str, DQLHBTRewardConfig]":
    """Return source-like and increasingly context-aware reward candidates."""

    return collections.OrderedDict(
        (
            (
                "source",
                DQLHBTRewardConfig(
                    name="source",
                    spectral_efficiency_weight=1.0,
                    link_outage_weight=4.0,
                    service_weight=0.0,
                    queue_weight=0.0,
                    queue_violation_weight=0.0,
                ),
            ),
            (
                "qos",
                DQLHBTRewardConfig(
                    name="qos",
                    reward_offset=10.0,
                    service_weight=1.0,
                    queue_weight=0.5,
                    queue_violation_weight=4.0,
                ),
            ),
            (
                "qos_energy_020",
                DQLHBTRewardConfig(
                    name="qos_energy_020",
                    reward_offset=10.0,
                    service_weight=1.0,
                    queue_weight=0.5,
                    queue_violation_weight=4.0,
                    energy_weight=0.20,
                ),
            ),
            (
                "qos_energy_020_load1",
                DQLHBTRewardConfig(
                    name="qos_energy_020_load1",
                    reward_offset=12.0,
                    service_weight=1.0,
                    queue_weight=0.5,
                    queue_violation_weight=4.0,
                    energy_weight=0.20,
                    load_weight=1.0,
                ),
            ),
            (
                "qos_energy_020_ho",
                DQLHBTRewardConfig(
                    name="qos_energy_020_ho",
                    reward_offset=10.0,
                    service_weight=1.0,
                    queue_weight=0.5,
                    queue_violation_weight=4.0,
                    energy_weight=0.20,
                    handover_weight=0.25,
                    sweep_weight=0.10,
                ),
            ),
        )
    )


@dataclasses.dataclass
class DQLHBTConfig:
    """System adaptation and DQN hyperparameters."""

    state_variant: str = "adapted"  # ``source`` or ``adapted``
    decision_trigger: str = "periodic"  # ``threshold`` or ``periodic``
    zone_size_m: float = 10.0
    snr_threshold_db: float = 2.0
    queue_trigger_ratio: Optional[float] = None
    num_bs: int = 5
    num_micro_bs: int = 4
    num_tx_beams: int = 32
    num_rx_beams: int = 8
    track_tx_radius: int = 1
    track_rx_radius: int = 1
    tracking_pilots: int = 1
    hidden_sizes: Tuple[int, ...] = (128, 128)
    learning_rate: float = 3.0e-4
    discount_factor: float = 0.95
    batch_size: int = 256
    replay_capacity: int = 200_000
    replay_warmup: int = 2_000
    train_every_transitions: int = 32
    target_update_steps: int = 250
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_decisions: float = 1.5e5
    double_dqn: bool = True
    reward_clip: float = 20.0
    torch_threads: int = 4

    @property
    def num_actions(self) -> int:
        # Action 0 tracks the current link; action 1+b hands over to BS b.
        return 1 + self.num_bs

    @property
    def full_sweep_pilots(self) -> int:
        return self.num_tx_beams * self.num_rx_beams

    @property
    def tracking_sweep_pilots(self) -> int:
        return (2 * self.track_tx_radius + 1) * (2 * self.track_rx_radius + 1)

    def validate(self) -> None:
        if self.state_variant not in ("source", "adapted"):
            raise ValueError("state_variant must be source or adapted")
        if self.decision_trigger not in ("threshold", "periodic"):
            raise ValueError("decision_trigger must be threshold or periodic")
        if self.num_bs != self.num_micro_bs + 1:
            raise ValueError("num_bs must equal one macro plus num_micro_bs")
        if self.zone_size_m <= 0.0:
            raise ValueError("zone_size_m must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("invalid epsilon range")
        if self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("invalid replay-buffer dimensions")


def hbt_action_target_bs(action: int, config: DQLHBTConfig) -> Optional[int]:
    """Map an HBT action to its target; ``None`` denotes beam tracking."""

    if action < 0 or action >= config.num_actions:
        raise ValueError("invalid HBT action {}".format(action))
    return None if action == 0 else action - 1


def valid_action_mask(serving_bs: int, config: DQLHBTConfig) -> np.ndarray:
    """Return actions valid at a serving BS.

    Tracking is always valid.  A "handover" to the already serving BS is
    masked because it would duplicate tracking while charging full training.
    """

    if not 0 <= int(serving_bs) < config.num_bs:
        raise ValueError("invalid serving BS")
    mask = np.ones(config.num_actions, dtype=np.bool_)
    mask[1 + int(serving_bs)] = False
    return mask


def local_track_beam_pair(
    channel: np.ndarray,
    micro_index: int,
    tx_beam: int,
    rx_beam: int,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
    tx_radius: int = 1,
    rx_radius: int = 1,
) -> Tuple[int, int, float, int]:
    """Select the best pair in a wrapped neighbourhood of the old pair."""

    tx_candidates = sorted(
        {(int(tx_beam) + delta) % dft_tx.shape[1] for delta in range(-tx_radius, tx_radius + 1)}
    )
    rx_candidates = sorted(
        {(int(rx_beam) + delta) % dft_rx.shape[1] for delta in range(-rx_radius, rx_radius + 1)}
    )
    projected = np.matmul(channel[:, micro_index, :], dft_tx[:, tx_candidates])
    amplitudes = np.abs(
        np.matmul(projected.T.conjugate(), dft_rx[:, rx_candidates])
    )
    local_tx, local_rx = np.unravel_index(int(np.argmax(amplitudes)), amplitudes.shape)
    chosen_tx = int(tx_candidates[local_tx])
    chosen_rx = int(rx_candidates[local_rx])
    normalized = amplitudes[local_tx, local_rx] / math.sqrt(
        channel.shape[0] * channel.shape[2]
    )
    return chosen_tx, chosen_rx, float(2.0 * lin2dB(normalized)), int(amplitudes.size)


@dataclasses.dataclass
class HBTLearnerState(_LearnerState):
    """Per-vehicle execution and delayed-transition state."""

    transition_state_vector: Optional[np.ndarray] = None
    transition_dql_action: Optional[int] = None
    transition_reward: float = 0.0
    transition_frames: int = 0
    last_dql_action: int = 0
    current_sweep_pilots: int = 0


@dataclasses.dataclass
class HBTActionOutcome:
    handover: bool
    beam_switch: bool
    sweep_pilots: int


def apply_hbt_action(
    learner: HBTLearnerState,
    dql_action: Optional[int],
    vehicle_record: MutableMapping,
    config: DQLHBTConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> HBTActionOutcome:
    """Apply a causal HBT command at the beginning of the next frame."""

    learner.current_sweep_pilots = 0
    if dql_action is None:
        return HBTActionOutcome(False, False, 0)
    old_bs = int(learner.action)
    old_pair = (learner.tx_beam, learner.rx_beam)
    target = hbt_action_target_bs(int(dql_action), config)
    learner.last_dql_action = int(dql_action)

    if target is None:
        # Track the current BS.  The macro link is omnidirectional in the
        # manuscript model; only a micro link needs a local beam search.
        if old_bs > 0:
            if learner.tx_beam is None or learner.rx_beam is None:
                learner.tx_beam, learner.rx_beam, _ = best_beam_pair(
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
            raise ValueError("handover target duplicates serving BS")
        learner.action = int(target)
        if target == 0:
            learner.tx_beam = None
            learner.rx_beam = None
        else:
            learner.tx_beam, learner.rx_beam, _ = best_beam_pair(
                vehicle_record["h"], target - 1, dft_tx, dft_rx
            )
            learner.current_sweep_pilots = config.full_sweep_pilots

    new_pair = (learner.tx_beam, learner.rx_beam)
    handover = int(learner.action) != old_bs
    beam_switch = handover or new_pair != old_pair
    return HBTActionOutcome(handover, beam_switch, learner.current_sweep_pilots)


def effective_sinr_db(
    args,
    serving_bs: int,
    serving_gain_db: float,
    interference_db: float,
) -> float:
    """Compute per-RB effective SINR from the common system parameters."""

    if serving_bs == 0:
        bandwidth = args.RB_intervel_macro
        power = args.p_macro
        nf_db = args.NF_macro_dB
        interference_w = 0.0
    else:
        bandwidth = args.RB_intervel_micro
        power = args.p_micro
        nf_db = args.NF_micro_dB
        noise_w = args.N0 * bandwidth * dB2lin(nf_db)
        interference_w = 0.0 if not np.isfinite(interference_db) else dB2lin(interference_db) * noise_w
    noise_w = args.N0 * bandwidth * dB2lin(nf_db)
    sinr = power * dB2lin(serving_gain_db) / max(noise_w + interference_w, 1e-30)
    return float(lin2dB(max(sinr, 1e-30)))


def should_make_decision(
    config: DQLHBTConfig,
    sinr_db: float,
    queue_ratio: float,
    force_initial: bool = False,
) -> bool:
    if force_initial or config.decision_trigger == "periodic":
        return True
    if sinr_db < config.snr_threshold_db:
        return True
    return config.queue_trigger_ratio is not None and queue_ratio >= config.queue_trigger_ratio


def state_feature_names(config: DQLHBTConfig) -> List[str]:
    base = ["x", "y"] + ["serving_bs_{}".format(i) for i in range(config.num_bs)]
    base += ["serving_sinr", "tracking_indicator"]
    if config.state_variant == "source":
        return base
    return base + [
        "heading_sin",
        "heading_cos",
        "speed",
        "queue_ratio",
        "traffic_rate",
    ] + ["bs_load_{}".format(i) for i in range(config.num_bs)] + [
        "interference_to_noise",
        "tx_beam_sin",
        "tx_beam_cos",
        "rx_beam_sin",
        "rx_beam_cos",
    ]


def make_state_vector(
    config: DQLHBTConfig,
    position: Sequence[float],
    heading_deg: float,
    speed_mps: float,
    serving_bs: int,
    sinr_db: float,
    tracking_indicator: bool,
    queue_ratio: float,
    load_ratio: Sequence[float],
    interference_db: float,
    traffic_mbps: float,
    tx_beam: Optional[int],
    rx_beam: Optional[int],
) -> np.ndarray:
    """Build a bounded continuous state vector for neural approximation."""

    pos = np.clip(np.asarray(position, dtype=np.float32) / 500.0, -1.5, 1.5)
    one_hot = np.zeros(config.num_bs, dtype=np.float32)
    one_hot[int(serving_bs)] = 1.0
    sinr_norm = np.float32(np.clip((float(sinr_db) - 10.0) / 30.0, -2.0, 2.0))
    values: List[float] = [float(pos[0]), float(pos[1])]
    values.extend(float(x) for x in one_hot)
    values.extend([float(sinr_norm), float(bool(tracking_indicator))])
    if config.state_variant == "source":
        return np.asarray(values, dtype=np.float32)

    heading = math.radians(float(heading_deg) % 360.0)
    values.extend(
        [
            math.sin(heading),
            math.cos(heading),
            float(np.clip(float(speed_mps) / 20.0, 0.0, 2.0)),
            float(np.clip(float(queue_ratio) / 5.0, 0.0, 2.0)),
            float(np.clip(float(traffic_mbps) / 20.0, 0.0, 2.0)),
        ]
    )
    loads = np.asarray(load_ratio, dtype=np.float32)
    if loads.shape != (config.num_bs,):
        raise ValueError("load vector has wrong dimension")
    values.extend(float(x) for x in np.clip(loads, 0.0, 1.5))
    inr_norm = -2.0 if not np.isfinite(interference_db) else np.clip(float(interference_db) / 30.0, -2.0, 2.0)
    values.append(float(inr_norm))

    def cyclic(index: Optional[int], size: int) -> Tuple[float, float]:
        if index is None:
            return 0.0, 0.0
        angle = 2.0 * math.pi * int(index) / size
        return math.sin(angle), math.cos(angle)

    values.extend(cyclic(tx_beam, config.num_tx_beams))
    values.extend(cyclic(rx_beam, config.num_rx_beams))
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (len(state_feature_names(config)),) or not np.all(np.isfinite(state)):
        raise ValueError("invalid DQL state")
    return state


class DuelingQNetwork(nn.Module):
    """Small dueling MLP; output remains a discrete DQL action value."""

    def __init__(self, state_dim: int, num_actions: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        width = state_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(width, int(hidden)), nn.ReLU()])
            width = int(hidden)
        self.encoder = nn.Sequential(*layers)
        self.value = nn.Linear(width, 1)
        self.advantage = nn.Linear(width, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.encoder(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.next_masks: Optional[np.ndarray] = None
        self.position = 0
        self.size = 0

    def initialize_masks(self, num_actions: int) -> None:
        self.next_masks = np.empty((self.capacity, num_actions), dtype=np.bool_)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        if self.next_masks is None:
            self.initialize_masks(len(next_mask))
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.next_masks[idx] = next_mask
        self.position = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        if self.next_masks is None:
            raise RuntimeError("empty replay buffer")
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.next_masks[idx],
        )


class DQLHBTPolicy:
    """Shared DQL policy trained from all vehicles' transitions."""

    def __init__(self, config: DQLHBTConfig, seed: int = 1):
        config.validate()
        self.config = config
        self.feature_names = state_feature_names(config)
        self.state_dim = len(self.feature_names)
        torch.set_num_threads(max(1, int(config.torch_threads)))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = DuelingQNetwork(
            self.state_dim, config.num_actions, config.hidden_sizes
        )
        self.target = DuelingQNetwork(
            self.state_dim, config.num_actions, config.hidden_sizes
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayBuffer(config.replay_capacity, self.state_dim)
        self.decision_count = 0
        self.transition_count = 0
        self.gradient_steps = 0
        self._trained_transition_watermark = 0

    def epsilon(self) -> float:
        decay = max(float(self.config.epsilon_decay_decisions), 1.0)
        return self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * math.exp(-self.decision_count / decay)

    def select_action(
        self,
        state: np.ndarray,
        serving_bs: int,
        explore: bool,
    ) -> int:
        mask = valid_action_mask(serving_bs, self.config)
        candidates = np.flatnonzero(mask)
        if explore:
            epsilon = self.epsilon()
            self.decision_count += 1
            if self.rng.random() < epsilon:
                return int(self.rng.choice(candidates))
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0]
            q_np = q.cpu().numpy()
        q_np[~mask] = -np.inf
        maximum = np.max(q_np)
        ties = np.flatnonzero(np.isclose(q_np, maximum, rtol=1e-6, atol=1e-8))
        # Prefer tracking on an exact tie; otherwise choose reproducibly.
        return 0 if 0 in ties else int(ties[0])

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_serving_bs: int,
        done: bool = False,
    ) -> None:
        clipped = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        self.replay.add(
            state,
            action,
            clipped,
            next_state,
            done,
            valid_action_mask(next_serving_bs, self.config),
        )
        self.transition_count += 1

    def learn_available(self) -> List[float]:
        """Take one gradient step for each elapsed training interval."""

        losses: List[float] = []
        if self.replay.size < max(self.config.replay_warmup, self.config.batch_size):
            return losses
        due = (self.transition_count - self._trained_transition_watermark) // max(
            self.config.train_every_transitions, 1
        )
        # Cap bursts after frames with many simultaneous vehicle decisions.
        due = min(int(due), 8)
        for _ in range(due):
            losses.append(self._gradient_step())
            self._trained_transition_watermark += self.config.train_every_transitions
        return losses

    def _gradient_step(self) -> float:
        states, actions, rewards, next_states, dones, masks = self.replay.sample(
            self.config.batch_size, self.rng
        )
        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.float32)
        masks_t = torch.as_tensor(masks, dtype=torch.bool)

        q = self.online(states_t).gather(1, actions_t[:, None]).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                online_next = self.online(next_states_t).masked_fill(~masks_t, -1e9)
                next_actions = online_next.argmax(dim=1)
                next_q = self.target(next_states_t).gather(
                    1, next_actions[:, None]
                ).squeeze(1)
            else:
                next_q = self.target(next_states_t).masked_fill(~masks_t, -1e9).max(dim=1).values
            target = rewards_t + self.config.discount_factor * (1.0 - dones_t) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def save(self, path: str, include_replay: bool = False) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "config": dataclasses.asdict(self.config),
            "feature_names": self.feature_names,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "decision_count": self.decision_count,
            "transition_count": self.transition_count,
            "gradient_steps": self.gradient_steps,
        }
        if include_replay:
            checkpoint["replay"] = self.replay
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: str, seed: int = 1, load_optimizer: bool = False) -> "DQLHBTPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config_dict = dict(checkpoint["config"])
        config_dict["hidden_sizes"] = tuple(config_dict["hidden_sizes"])
        policy = DQLHBTPolicy(DQLHBTConfig(**config_dict), seed=seed)
        policy.online.load_state_dict(checkpoint["online"])
        policy.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if load_optimizer and "optimizer" in checkpoint:
            policy.optimizer.load_state_dict(checkpoint["optimizer"])
        policy.decision_count = int(checkpoint.get("decision_count", 0))
        policy.transition_count = int(checkpoint.get("transition_count", 0))
        policy.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        return policy


@dataclasses.dataclass
class DQLFluidStep:
    queue_end: Dict[object, float]
    served_bits: Dict[object, float]
    user_power_w: Dict[object, float]
    serving_gain_db: Dict[object, float]
    interference_db: Dict[object, float]
    spectral_efficiency: Dict[object, float]
    load_ratio: np.ndarray
    connection: Dict[object, int]


def _fluid_step_hbt(
    args,
    records: MutableMapping,
    learners: Dict[object, HBTLearnerState],
    queue_start: Dict[object, float],
    vehicle_rate: Dict[object, float],
    previous_load: np.ndarray,
    macro_bs_loc: np.ndarray,
    hbt_config: DQLHBTConfig,
    pql_config: PQLBAConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> DQLFluidStep:
    vehicles = sorted(records.keys(), key=str)
    frame_duration_s = args.slots_per_frame * args.slot_len
    connection = {vehicle: int(learners[vehicle].action) for vehicle in vehicles}
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
            pql_config,
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
            overhead_sum = 0.0
            for slot_index in range(args.slots_per_frame):
                if learners[vehicle].current_sweep_pilots > 0:
                    slot_pilots = sweep_pilots_for_slot(
                        learners[vehicle].current_sweep_pilots,
                        hbt_config.tracking_pilots,
                        slot_index,
                        args.pilot_overhead_factor,
                    )
                else:
                    slot_pilots = hbt_config.tracking_pilots
                overhead_sum += min(slot_pilots * args.pilot_overhead_factor, 1.0)
            pilots = overhead_sum / args.slots_per_frame / args.pilot_overhead_factor
        pilot_average[vehicle] = pilots
        backlog_bits[vehicle] = queue_start[vehicle] + vehicle_rate[vehicle] * frame_duration_s

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
    served_bits: Dict[object, float] = {}
    queue_end: Dict[object, float] = {}
    user_power: Dict[object, float] = {}
    spectral_efficiency: Dict[object, float] = {}
    for vehicle in vehicles:
        bs = connection[vehicle]
        served_bits[vehicle] = min(
            backlog_bits[vehicle], allocation[vehicle] * capacity[vehicle] * frame_duration_s
        )
        queue_end[vehicle] = max(0.0, backlog_bits[vehicle] - served_bits[vehicle])
        user_power[vehicle] = allocation[vehicle] * (args.p_macro if bs == 0 else args.p_micro)
        bandwidth = args.RB_intervel_macro if bs == 0 else args.RB_intervel_micro
        spectral_efficiency[vehicle] = capacity[vehicle] / bandwidth
    return DQLFluidStep(
        queue_end=queue_end,
        served_bits=served_bits,
        user_power_w=user_power,
        serving_gain_db=serving_gain,
        interference_db=interference,
        spectral_efficiency=spectral_efficiency,
        load_ratio=load,
        connection=connection,
    )


def _frame_reward(
    reward_config: DQLHBTRewardConfig,
    spectral_efficiency: float,
    served_bits: float,
    offered_bits: float,
    queue_ratio: float,
    user_power_w: float,
    serving_load: float,
    handover: bool,
    sweep_pilots: int,
    full_sweep_pilots: int,
) -> float:
    service_ratio = min(served_bits / max(offered_bits, 1e-12), reward_config.service_reward_cap)
    normalized_se = min(spectral_efficiency / 4.0, 2.0)
    return float(
        reward_config.reward_offset
        + reward_config.spectral_efficiency_weight * normalized_se
        - reward_config.link_outage_weight
        * float(spectral_efficiency < reward_config.spectral_efficiency_threshold)
        + reward_config.service_weight * service_ratio
        - reward_config.queue_weight * min(queue_ratio, reward_config.queue_penalty_cap)
        - reward_config.queue_violation_weight * float(queue_ratio > 1.0)
        - reward_config.energy_weight * user_power_w
        - reward_config.load_weight * serving_load
        - reward_config.handover_weight * float(handover)
        - reward_config.sweep_weight * sweep_pilots / max(full_sweep_pilots, 1)
    )


def run_fluid_dql_episode(
    args,
    timeline_dir: MutableMapping,
    policy: DQLHBTPolicy,
    reward_config: DQLHBTRewardConfig,
    data_rate_mbps: float,
    seed: int = 1,
    learn: bool = False,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
) -> Dict[str, float]:
    """Train or validate DQL-HBT in the common frame-level fluid model."""

    reward_config.validate()
    config = policy.config
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
    pql_config = PQLBAConfig(
        hierarchical_bs_action=True,
        num_micro_bs=config.num_micro_bs,
        num_tx_beams=config.num_tx_beams,
        num_rx_beams=config.num_rx_beams,
        tracking_pilots=config.tracking_pilots,
    )
    learners: Dict[object, HBTLearnerState] = {}
    queues: Dict[object, float] = {}
    previous_load = np.zeros(config.num_bs, dtype=float)

    reward_samples: List[float] = []
    loss_samples: List[float] = []
    power_samples: List[float] = []
    violation_samples: List[float] = []
    delay_samples: List[float] = []
    vehicle_samples: List[float] = []
    action_counts = np.zeros(config.num_actions, dtype=np.int64)
    handovers = 0
    beam_switches = 0
    decisions = 0
    skipped_events = 0
    transitions_start = policy.transition_count
    gradients_start = policy.gradient_steps

    for frame_index, frame in enumerate(frames):
        records = timeline_dir[frame]
        present = set(records.keys())
        for departed in set(learners).difference(present):
            learners.pop(departed, None)
            queues.pop(departed, None)
        for vehicle in sorted(present, key=str):
            if vehicle not in learners:
                position = np.asarray(records[vehicle]["pos"], dtype=float)
                learners[vehicle] = HBTLearnerState(
                    action=0,
                    rx_beam=None,
                    pending_action=None,
                    last_position=position.copy(),
                    distance_since_event=config.zone_size_m,
                    tx_beam=None,
                )
                queues[vehicle] = 0.5 * queue_upper_bound

        outcomes: Dict[object, HBTActionOutcome] = {}
        for vehicle in sorted(present, key=str):
            learner = learners[vehicle]
            pending = learner.pending_action
            learner.pending_action = None
            outcome = apply_hbt_action(
                learner, pending, records[vehicle], config, dft_tx, dft_rx
            )
            outcomes[vehicle] = outcome
            handovers += int(outcome.handover)
            beam_switches += int(outcome.beam_switch)

        queue_before_service = queues.copy()
        vehicle_rate = {vehicle: rate_bps for vehicle in present}
        step = _fluid_step_hbt(
            args,
            records,
            learners,
            queues,
            vehicle_rate,
            previous_load,
            macro_loc,
            config,
            pql_config,
            dft_tx,
            dft_rx,
        )
        previous_load = step.load_ratio
        queues = step.queue_end

        frame_power = sum(step.user_power_w.values())
        frame_violations = 0
        frame_delay = 0.0
        offered_bits = rate_bps * frame_duration_s
        for vehicle in present:
            queue_ratio_end = queues[vehicle] / queue_upper_bound
            frame_violations += int(queue_ratio_end > 1.0)
            frame_delay += queues[vehicle] / rate_bps
            learner = learners[vehicle]
            if learner.transition_state_vector is not None:
                reward = _frame_reward(
                    reward_config,
                    step.spectral_efficiency[vehicle],
                    step.served_bits[vehicle],
                    offered_bits,
                    queue_ratio_end,
                    step.user_power_w[vehicle],
                    step.load_ratio[step.connection[vehicle]],
                    outcomes[vehicle].handover,
                    outcomes[vehicle].sweep_pilots,
                    config.full_sweep_pilots,
                )
                learner.transition_reward += reward
                learner.transition_frames += 1

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
            serving_bs = int(step.connection[vehicle])
            sinr_db = effective_sinr_db(
                args,
                serving_bs,
                step.serving_gain_db[vehicle],
                step.interference_db[vehicle],
            )
            queue_ratio = queue_before_service[vehicle] / queue_upper_bound
            force_initial = learner.transition_state_vector is None
            if not should_make_decision(config, sinr_db, queue_ratio, force_initial):
                skipped_events += 1
                learner.distance_since_event %= config.zone_size_m
                continue
            state = make_state_vector(
                config,
                current_position,
                float(records[vehicle].get("angle", 0.0)),
                float(records[vehicle].get("v", 0.0)),
                serving_bs,
                sinr_db,
                learner.last_dql_action == 0,
                queue_ratio,
                step.load_ratio,
                step.interference_db[vehicle],
                data_rate_mbps,
                learner.tx_beam,
                learner.rx_beam,
            )
            if learn and learner.transition_state_vector is not None:
                event_reward = learner.transition_reward / max(
                    learner.transition_frames, 1
                )
                policy.observe(
                    learner.transition_state_vector,
                    int(learner.transition_dql_action),
                    event_reward,
                    state,
                    serving_bs,
                )
                reward_samples.append(event_reward)
            selected = policy.select_action(state, serving_bs, explore=learn)
            learner.pending_action = int(selected)
            learner.transition_state_vector = state
            learner.transition_dql_action = int(selected)
            learner.transition_reward = 0.0
            learner.transition_frames = 0
            learner.distance_since_event %= config.zone_size_m
            action_counts[selected] += 1
            decisions += 1

        if learn:
            loss_samples.extend(policy.learn_available())

    duration_s = len(frames) * frame_duration_s
    average_vehicles = float(np.mean(vehicle_samples))
    action_fractions = (action_counts / max(action_counts.sum(), 1)).tolist()
    return {
        "data_rate_mbps": float(data_rate_mbps),
        "learn": bool(learn),
        "decisions": float(decisions),
        "skipped_trigger_events": float(skipped_events),
        "transitions": float(policy.transition_count - transitions_start),
        "gradient_steps": float(policy.gradient_steps - gradients_start),
        "mean_event_reward": float(np.mean(reward_samples)) if reward_samples else 0.0,
        "mean_training_loss": float(np.mean(loss_samples)) if loss_samples else 0.0,
        "average_system_power_w": float(np.mean(power_samples)),
        "queue_violation_percent": float(100.0 * np.mean(violation_samples)),
        "average_queueing_proxy_ms": float(1000.0 * np.mean(delay_samples)),
        "handover_per_vehicle_per_s": float(handovers / max(duration_s * average_vehicles, 1e-12)),
        "beam_switch_per_vehicle_per_s": float(beam_switches / max(duration_s * average_vehicles, 1e-12)),
        "average_vehicle_count": average_vehicles,
        "epsilon": float(policy.epsilon()),
        "replay_size": float(policy.replay.size),
        "action_fractions": action_fractions,
    }


def train_dql_hbt(
    args,
    timeline_dir: MutableMapping,
    config: DQLHBTConfig,
    reward_config: DQLHBTRewardConfig,
    data_rate_schedule_mbps: Sequence[float],
    epochs: int,
    seed: int = 1,
    verbose: bool = True,
) -> Tuple[DQLHBTPolicy, List[Dict[str, float]]]:
    if epochs <= 0 or not data_rate_schedule_mbps:
        raise ValueError("epochs and rate schedule must be nonempty")
    policy = DQLHBTPolicy(config, seed=seed)
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        rate = float(data_rate_schedule_mbps[epoch % len(data_rate_schedule_mbps)])
        result = run_fluid_dql_episode(
            args,
            timeline_dir,
            policy,
            reward_config,
            data_rate_mbps=rate,
            seed=seed + epoch,
            learn=True,
        )
        result["epoch"] = float(epoch + 1)
        history.append(result)
        if verbose:
            print(
                "{} epoch {:02d} rate={:g}: reward={:.3f}, loss={:.4f}, "
                "power={:.2f} W, vio={:.2f}%, eps={:.3f}, replay={:.0f}".format(
                    reward_config.name,
                    epoch + 1,
                    rate,
                    result["mean_event_reward"],
                    result["mean_training_loss"],
                    result["average_system_power_w"],
                    result["queue_violation_percent"],
                    result["epsilon"],
                    result["replay_size"],
                )
            )
    return policy, history
