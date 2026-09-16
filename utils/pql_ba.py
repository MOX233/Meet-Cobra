"""Adaptation of parallel Q-learning beam association (PQL-BA).

The implementation follows Huynh et al., IEEE TCOM 2021: vehicles act as
parallel learners that asynchronously update one global tabular Q function;
decisions are made at distance-zone crossings; the state contains the
quantized RSSI and current beam; and the reward is useful downlink data.

Three adaptations are needed for the MEET-COBRA system model:

* action 0 is the data-serving macro BS (the LTE link in the source paper is
  control-only);
* a micro action selects a micro BS and a transmit beam.  The receiver then
  performs a local receive-codebook sweep, avoiding an intractable table over
  all 4 * 32 * 8 beam pairs while still producing a concrete TX/RX pair;
* an optional heading component resolves the ambiguity introduced by the 2-D,
  bidirectional road topology.  It is disabled for the source-faithful model.

The module deliberately contains no queue or resource-allocation information.
That preserves PQL-BA's throughput-oriented decision rule when it is later
coupled to the same OTR-RA scheduler used by MEET-COBRA.
"""

from __future__ import annotations

import dataclasses
import math
import pickle
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from utils.beam_utils import generate_dft_codebook
from utils.channel_utils import calculate_uma_pathloss_3gpp_38901
from utils.mox_utils import dB2lin, lin2dB


State = Tuple[int, ...]


@dataclasses.dataclass
class PQLBAConfig:
    """Hyperparameters and common-system adaptations for PQL-BA."""

    num_micro_bs: int = 4
    num_tx_beams: int = 32
    num_rx_beams: int = 8
    hierarchical_bs_action: bool = False
    zone_size_m: float = 10.0
    include_heading: bool = False
    num_heading_bins: int = 8
    # None preserves the source-paper state.  A coarse 2-D bin is an optional
    # adaptation for the ambiguity created by a data-serving macro action.
    location_bin_size_m: Optional[float] = None
    location_origin_m: Tuple[float, float] = (-500.0, -500.0)
    include_queue_state: bool = False
    include_load_state: bool = False
    include_interference_state: bool = False
    include_traffic_state: bool = False
    queue_ratio_edges: Tuple[float, ...] = (0.25, 0.75, 1.0, 2.0, 5.0)
    load_ratio_edges: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    interference_db_edges: Tuple[float, ...] = (-10.0, 0.0, 10.0, 20.0, 30.0)
    traffic_mbps_edges: Tuple[float, ...] = (3.0, 9.0, 15.0, 21.0, 27.0, 33.0)
    rssi_edges_dbm: Tuple[float, ...] = (
        -120.0,
        -105.0,
        -95.0,
        -85.0,
        -75.0,
        -65.0,
        -55.0,
    )
    learning_rate: float = 0.10
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_decisions: float = 2.0e5
    receiver_sweep_pilots: int = 8
    tracking_pilots: int = 1
    macro_power_w: float = 1.0
    micro_power_w: float = 0.2
    # The manuscript's common system model excludes HO interruption.  Keeping
    # this at zero therefore compares all schemes under the same assumptions.
    handover_interruption_s: float = 0.0

    @property
    def num_actions(self) -> int:
        if self.hierarchical_bs_action:
            return 1 + self.num_micro_bs
        return 1 + self.num_micro_bs * self.num_tx_beams

    def __setstate__(self, state) -> None:
        """Backfill fields when loading policies saved before an extension."""

        self.__dict__.update(state)
        for field in dataclasses.fields(type(self)):
            if field.name in self.__dict__:
                continue
            if field.default is not dataclasses.MISSING:
                self.__dict__[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                self.__dict__[field.name] = field.default_factory()

    def validate(self) -> None:
        if self.num_micro_bs <= 0 or self.num_tx_beams <= 0 or self.num_rx_beams <= 0:
            raise ValueError("antenna and BS dimensions must be positive")
        if self.zone_size_m <= 0:
            raise ValueError("zone_size_m must be positive")
        if self.location_bin_size_m is not None and self.location_bin_size_m <= 0:
            raise ValueError("location_bin_size_m must be positive when enabled")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= self.discount_factor < 1.0:
            raise ValueError("discount_factor must be in [0, 1)")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")


def action_to_link(action: int, config: PQLBAConfig) -> Tuple[int, Optional[int]]:
    """Return ``(serving_bs, tx_beam)`` for a global action index."""

    if action < 0 or action >= config.num_actions:
        raise ValueError("action {} is outside [0, {})".format(action, config.num_actions))
    if action == 0:
        return 0, None
    if config.hierarchical_bs_action:
        return action, None
    micro_index, tx_beam = divmod(action - 1, config.num_tx_beams)
    return micro_index + 1, tx_beam


def link_to_action(serving_bs: int, tx_beam: Optional[int], config: PQLBAConfig) -> int:
    """Return the global action index for a macro or micro link."""

    if serving_bs == 0:
        return 0
    if not 1 <= serving_bs <= config.num_micro_bs:
        raise ValueError("invalid micro BS index")
    if config.hierarchical_bs_action:
        return int(serving_bs)
    if tx_beam is None or not 0 <= tx_beam < config.num_tx_beams:
        raise ValueError("invalid transmit beam index")
    return 1 + (serving_bs - 1) * config.num_tx_beams + int(tx_beam)


def action_serving_bs(action: int, config: PQLBAConfig) -> int:
    return action_to_link(action, config)[0]


def sweep_pilots_for_slot(
    total_sweep_pilots: int,
    tracking_pilots: int,
    slot_index: int,
    pilot_overhead_factor: float,
) -> int:
    """Spread a sweep while keeping every slot below 100% overhead."""

    max_pilots = max(1, int(math.ceil(1.0 / pilot_overhead_factor) - 1))
    remaining = int(total_sweep_pilots) - int(slot_index) * max_pilots
    if remaining > 0:
        return min(max_pilots, remaining)
    return int(tracking_pilots)


def macro_gain_db(args, position: np.ndarray, macro_bs_loc: np.ndarray) -> float:
    """3GPP UMa LOS gain used by the existing simulator (0 dBi macro gain)."""

    distance = float(np.linalg.norm(np.asarray(position) - np.asarray(macro_bs_loc)))
    pathloss = calculate_uma_pathloss_3gpp_38901(
        distance_2d_m=distance,
        fc_ghz=2.8,
        h_bs_m=args.h_tx,
        h_ut_m=args.h_car,
        scenario="los",
    )
    return -float(pathloss)


def best_rx_for_tx(
    channel: np.ndarray,
    micro_index: int,
    tx_beam: int,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> Tuple[int, float]:
    """Sweep receive beams for one micro-BS/transmit-beam action."""

    projected = np.matmul(channel[:, micro_index, :], dft_tx[:, tx_beam])
    amplitudes = np.abs(np.matmul(projected.T.conjugate(), dft_rx))
    rx_beam = int(np.argmax(amplitudes))
    normalized = amplitudes[rx_beam] / math.sqrt(channel.shape[0] * channel.shape[2])
    return rx_beam, float(2.0 * lin2dB(normalized))


def best_beam_pair(
    channel: np.ndarray,
    micro_index: int,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> Tuple[int, int, float]:
    """Exhaustively select a TX/RX pair for a hierarchical BS action."""

    projected = np.matmul(channel[:, micro_index, :], dft_tx)
    amplitudes = np.abs(np.matmul(projected.T.conjugate(), dft_rx))
    tx_beam, rx_beam = np.unravel_index(int(np.argmax(amplitudes)), amplitudes.shape)
    normalized = amplitudes[tx_beam, rx_beam] / math.sqrt(
        channel.shape[0] * channel.shape[2]
    )
    return int(tx_beam), int(rx_beam), float(2.0 * lin2dB(normalized))


def fixed_pair_gain_db(
    channel: np.ndarray,
    micro_index: int,
    tx_beam: int,
    rx_beam: int,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> float:
    """Gain of one fixed TX/RX DFT beam pair in dB."""

    projected = np.matmul(channel[:, micro_index, :], dft_tx[:, tx_beam])
    amplitude = np.abs(np.matmul(projected.T.conjugate(), dft_rx[:, rx_beam]))
    normalized = amplitude / math.sqrt(channel.shape[0] * channel.shape[2])
    return float(2.0 * lin2dB(normalized))


def no_bf_gain_db(channel: np.ndarray) -> np.ndarray:
    """Match the interference-link gain convention in ``measure_gain``."""

    return np.asarray(
        [2.0 * lin2dB(np.abs(channel[:, m, :]).max()) for m in range(channel.shape[1])],
        dtype=np.float64,
    )


def full_link_rate_bps(args, serving_bs: int, gain_db: float, pilot_count: float = 0.0) -> float:
    """Single-user full-band SNR rate used as the source-paper reward.

    PQL-BA does not model multi-user load or queues.  Its observed rate is
    therefore calculated as the rate available if the selected BS devoted its
    full RB budget to that vehicle.  Loaded-network effects are introduced only
    in the frozen-policy evaluation through the common OTR-RA simulator.
    """

    if serving_bs == 0:
        bandwidth = args.RB_intervel_macro
        power = args.p_macro
        noise_figure_db = args.NF_macro_dB
        num_rb = args.num_RB_macro
        useful_fraction = 1.0
    else:
        bandwidth = args.RB_intervel_micro
        power = args.p_micro
        noise_figure_db = args.NF_micro_dB
        num_rb = args.num_RB_micro
        useful_fraction = max(0.0, 1.0 - pilot_count * args.pilot_overhead_factor)
    snr = power * dB2lin(gain_db) / (
        args.N0 * bandwidth * dB2lin(noise_figure_db)
    )
    return float(useful_fraction * num_rb * bandwidth * np.log2(1.0 + snr))


@dataclasses.dataclass
class _LearnerState:
    action: int
    rx_beam: Optional[int]
    pending_action: Optional[int]
    last_position: np.ndarray
    distance_since_event: float
    tx_beam: Optional[int] = None
    transition_state: Optional[State] = None
    transition_action: Optional[int] = None
    transition_reward_mbit: float = 0.0


class PQLBAPolicy:
    """Sparse global Q table shared by all vehicle learning processes."""

    def __init__(self, config: PQLBAConfig):
        config.validate()
        self.config = config
        self.q_table: Dict[State, np.ndarray] = {}
        self.visit_table: Dict[State, np.ndarray] = {}
        self.decision_count = 0
        self.update_count = 0

    def _row(self, state: State, create: bool = True) -> np.ndarray:
        row = self.q_table.get(state)
        if row is None:
            if not create:
                return np.zeros(self.config.num_actions, dtype=np.float32)
            row = np.zeros(self.config.num_actions, dtype=np.float32)
            self.q_table[state] = row
            self.visit_table[state] = np.zeros(self.config.num_actions, dtype=np.uint32)
        return row

    def epsilon(self) -> float:
        decay = max(float(self.config.epsilon_decay_decisions), 1.0)
        return self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * math.exp(-self.decision_count / decay)

    def make_state(
        self,
        gain_db: float,
        action: int,
        heading_deg: float = 0.0,
        position: Optional[Sequence[float]] = None,
        queue_ratio: Optional[float] = None,
        load_ratio: Optional[float] = None,
        interference_db: Optional[float] = None,
        traffic_mbps: Optional[float] = None,
    ) -> State:
        serving_bs = action_serving_bs(action, self.config)
        power = self.config.macro_power_w if serving_bs == 0 else self.config.micro_power_w
        rssi_dbm = gain_db + 10.0 * math.log10(power * 1000.0)
        rssi_level = int(np.searchsorted(self.config.rssi_edges_dbm, rssi_dbm, side="right"))
        state_components = [rssi_level, int(action)]
        if self.config.include_heading:
            heading = float(heading_deg) % 360.0
            heading_bin = int(
                math.floor(heading / (360.0 / self.config.num_heading_bins))
            ) % self.config.num_heading_bins
            state_components.append(heading_bin)
        if self.config.location_bin_size_m is not None:
            if position is None:
                raise ValueError("position is required for a location-aware state")
            position_array = np.asarray(position, dtype=float)
            origin = np.asarray(self.config.location_origin_m, dtype=float)
            location_bin = np.floor(
                (position_array - origin) / self.config.location_bin_size_m
            ).astype(int)
            state_components.extend([int(location_bin[0]), int(location_bin[1])])
        if self.config.include_queue_state:
            if queue_ratio is None:
                raise ValueError("queue_ratio is required for a queue-aware state")
            state_components.append(
                int(
                    np.searchsorted(
                        self.config.queue_ratio_edges,
                        float(queue_ratio),
                        side="right",
                    )
                )
            )
        if self.config.include_load_state:
            if load_ratio is None:
                raise ValueError("load_ratio is required for a load-aware state")
            state_components.append(
                int(
                    np.searchsorted(
                        self.config.load_ratio_edges,
                        float(load_ratio),
                        side="right",
                    )
                )
            )
        if self.config.include_interference_state:
            if interference_db is None:
                raise ValueError(
                    "interference_db is required for an interference-aware state"
                )
            state_components.append(
                int(
                    np.searchsorted(
                        self.config.interference_db_edges,
                        float(interference_db),
                        side="right",
                    )
                )
            )
        if self.config.include_traffic_state:
            if traffic_mbps is None:
                raise ValueError("traffic_mbps is required for a traffic-aware state")
            state_components.append(
                int(
                    np.searchsorted(
                        self.config.traffic_mbps_edges,
                        float(traffic_mbps),
                        side="right",
                    )
                )
            )
        return tuple(state_components)

    def select_action(
        self,
        state: State,
        rng: np.random.Generator,
        explore: bool = True,
    ) -> int:
        row = self._row(state, create=explore)
        if explore:
            epsilon = self.epsilon()
            self.decision_count += 1
            if rng.random() < epsilon:
                return int(rng.integers(self.config.num_actions))
        if not explore and state in self.visit_table:
            visited_actions = np.flatnonzero(self.visit_table[state] > 0)
        else:
            visited_actions = np.asarray([], dtype=int)
        if not explore and visited_actions.size:
            maximum = float(row[visited_actions].max())
            candidates = visited_actions[
                np.isclose(
                    row[visited_actions], maximum, rtol=1e-7, atol=1e-9
                )
            ]
        else:
            maximum = float(row.max())
            candidates = np.flatnonzero(
                np.isclose(row, maximum, rtol=1e-7, atol=1e-9)
            )
        current_action = int(state[1])
        # During frozen evaluation, an unseen/tied state stays on its current
        # beam.  This avoids arbitrary index-based macro or BS bias.
        if not explore and current_action in candidates:
            return current_action
        return int(rng.choice(candidates))

    def update(self, state: State, action: int, reward: float, next_state: State) -> float:
        row = self._row(state, create=True)
        next_row = self._row(next_state, create=True)
        target = float(reward) + self.config.discount_factor * float(next_row.max())
        td_error = target - float(row[action])
        row[action] += self.config.learning_rate * td_error
        self.visit_table[state][action] += 1
        self.update_count += 1
        return td_error

    @property
    def visited_state_action_pairs(self) -> int:
        return int(sum(np.count_nonzero(row) for row in self.visit_table.values()))

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "PQLBAPolicy":
        with open(path, "rb") as handle:
            policy = pickle.load(handle)
        if not isinstance(policy, PQLBAPolicy):
            raise TypeError("file does not contain a PQLBAPolicy")
        return policy


def _apply_pending_action(
    learner: _LearnerState,
    vehicle_record: MutableMapping,
    config: PQLBAConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> bool:
    """Apply a one-frame-ahead command and return whether the beam changed."""

    if learner.pending_action is None:
        return False
    new_action = int(learner.pending_action)
    learner.pending_action = None
    old_action = learner.action
    old_tx_beam = learner.tx_beam
    old_rx_beam = learner.rx_beam
    learner.action = new_action
    serving_bs, tx_beam = action_to_link(new_action, config)
    if serving_bs == 0:
        learner.tx_beam = None
        learner.rx_beam = None
    elif config.hierarchical_bs_action:
        learner.tx_beam, learner.rx_beam, _ = best_beam_pair(
            vehicle_record["h"], serving_bs - 1, dft_tx, dft_rx
        )
    else:
        learner.tx_beam = int(tx_beam)
        # The global action selects only a transmit beam.  At every SMDP
        # decision epoch the receiver must therefore refresh its local
        # combiner, even when the policy elects to stay on the same TX beam.
        learner.rx_beam, _ = best_rx_for_tx(
            vehicle_record["h"], serving_bs - 1, int(tx_beam), dft_tx, dft_rx
        )
    if not config.hierarchical_bs_action:
        return new_action != old_action
    return (
        new_action != old_action
        or learner.tx_beam != old_tx_beam
        or learner.rx_beam != old_rx_beam
    )


def _learner_gain_db(
    args,
    learner: _LearnerState,
    vehicle_record: MutableMapping,
    macro_bs_loc: np.ndarray,
    config: PQLBAConfig,
    dft_tx: np.ndarray,
    dft_rx: np.ndarray,
) -> float:
    serving_bs, tx_beam = action_to_link(learner.action, config)
    if serving_bs == 0:
        return macro_gain_db(args, vehicle_record["pos"], macro_bs_loc)
    if config.hierarchical_bs_action:
        tx_beam = learner.tx_beam
    if learner.rx_beam is None:
        if config.hierarchical_bs_action:
            learner.tx_beam, learner.rx_beam, gain_db = best_beam_pair(
                vehicle_record["h"], serving_bs - 1, dft_tx, dft_rx
            )
            return gain_db
        learner.rx_beam, gain_db = best_rx_for_tx(
            vehicle_record["h"], serving_bs - 1, int(tx_beam), dft_tx, dft_rx
        )
        return gain_db
    return fixed_pair_gain_db(
        vehicle_record["h"],
        serving_bs - 1,
        int(tx_beam),
        int(learner.rx_beam),
        dft_tx,
        dft_rx,
    )


def train_pql_ba(
    args,
    timeline_dir: MutableMapping,
    config: PQLBAConfig,
    epochs: int = 1,
    seed: int = 1,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
    verbose: bool = True,
) -> Tuple[PQLBAPolicy, List[Dict[str, float]]]:
    """Train one global table from temporally ordered parallel trajectories."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    policy = PQLBAPolicy(config)
    rng = np.random.default_rng(seed)
    dft_tx = generate_dft_codebook(config.num_tx_beams)
    dft_rx = generate_dft_codebook(config.num_rx_beams)
    macro_loc = np.asarray(macro_bs_loc, dtype=float)
    frame_list = list(timeline_dir.keys())
    if len(frame_list) < 2:
        raise ValueError("timeline must contain at least two frames")
    frame_interval_s = float(np.median(np.diff(np.asarray(frame_list, dtype=float))))
    frame_duration_s = args.slots_per_frame * args.slot_len
    if not np.isclose(frame_interval_s, frame_duration_s, rtol=1e-5, atol=1e-8):
        raise ValueError("trace interval and simulated frame duration do not match")

    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        learners: Dict[object, _LearnerState] = {}
        epoch_rewards: List[float] = []
        epoch_td_errors: List[float] = []
        epoch_handover = 0
        epoch_beam_switch = 0
        decisions_at_start = policy.decision_count
        updates_at_start = policy.update_count

        for frame in frame_list:
            frame_records = timeline_dir[frame]
            present = set(frame_records.keys())
            for departed in set(learners.keys()).difference(present):
                del learners[departed]

            for vehicle in sorted(present, key=str):
                record = frame_records[vehicle]
                if vehicle not in learners:
                    learners[vehicle] = _LearnerState(
                        action=0,
                        rx_beam=None,
                        pending_action=None,
                        last_position=np.asarray(record["pos"], dtype=float).copy(),
                        distance_since_event=config.zone_size_m,
                    )
                learner = learners[vehicle]
                old_bs = action_serving_bs(learner.action, config)
                decision_applied = learner.pending_action is not None
                changed = _apply_pending_action(learner, record, config, dft_tx, dft_rx)
                new_bs = action_serving_bs(learner.action, config)
                if changed:
                    epoch_beam_switch += 1
                    if old_bs != new_bs:
                        epoch_handover += 1

                gain_db = _learner_gain_db(
                    args, learner, record, macro_loc, config, dft_tx, dft_rx
                )
                if learner.transition_state is not None:
                    pilot_average = 0.0
                    if new_bs > 0:
                        pilot_average = float(config.tracking_pilots)
                        if decision_applied:
                            pilot_average += (
                                config.receiver_sweep_pilots - config.tracking_pilots
                            ) / args.slots_per_frame
                    reward_rate = full_link_rate_bps(args, new_bs, gain_db, pilot_average)
                    learner.transition_reward_mbit += reward_rate * frame_duration_s / 1e6
                    if (
                        changed
                        and old_bs != new_bs
                        and config.handover_interruption_s > 0.0
                    ):
                        learner.transition_reward_mbit = max(
                            0.0,
                            learner.transition_reward_mbit
                            - reward_rate * config.handover_interruption_s / 1e6,
                        )

                current_position = np.asarray(record["pos"], dtype=float)
                learner.distance_since_event += float(
                    np.linalg.norm(current_position - learner.last_position)
                )
                learner.last_position = current_position.copy()
                if learner.distance_since_event + 1e-9 < config.zone_size_m:
                    continue

                heading = float(record.get("angle", 0.0))
                next_state = policy.make_state(
                    gain_db, learner.action, heading, position=current_position
                )
                if learner.transition_state is not None:
                    td_error = policy.update(
                        learner.transition_state,
                        int(learner.transition_action),
                        learner.transition_reward_mbit,
                        next_state,
                    )
                    epoch_rewards.append(learner.transition_reward_mbit)
                    epoch_td_errors.append(abs(td_error))

                selected_action = policy.select_action(next_state, rng, explore=True)
                learner.pending_action = selected_action
                learner.transition_state = next_state
                learner.transition_action = selected_action
                learner.transition_reward_mbit = 0.0
                learner.distance_since_event = learner.distance_since_event % config.zone_size_m

        record = {
            "epoch": float(epoch + 1),
            "decisions": float(policy.decision_count - decisions_at_start),
            "updates": float(policy.update_count - updates_at_start),
            "mean_reward_mbit": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "mean_abs_td_error": float(np.mean(epoch_td_errors)) if epoch_td_errors else 0.0,
            "epsilon": float(policy.epsilon()),
            "q_states": float(len(policy.q_table)),
            "visited_pairs": float(policy.visited_state_action_pairs),
            "handover_count": float(epoch_handover),
            "beam_switch_count": float(epoch_beam_switch),
        }
        history.append(record)
        if verbose:
            print(
                "PQL-BA epoch {epoch:.0f}: updates={updates:.0f}, reward={reward:.3f} Mbit, "
                "|TD|={td:.3f}, eps={eps:.3f}, states={states:.0f}".format(
                    epoch=record["epoch"],
                    updates=record["updates"],
                    reward=record["mean_reward_mbit"],
                    td=record["mean_abs_td_error"],
                    eps=record["epsilon"],
                    states=record["q_states"],
                )
            )
    return policy, history


def rollout_link_policy(
    args,
    timeline_dir: MutableMapping,
    policy: PQLBAPolicy,
    seed: int = 1,
    macro_bs_loc: Sequence[float] = (0.0, 0.0),
) -> Dict[str, float]:
    """Queue-free greedy rollout for validation and convergence checks."""

    config = policy.config
    rng = np.random.default_rng(seed)
    dft_tx = generate_dft_codebook(config.num_tx_beams)
    dft_rx = generate_dft_codebook(config.num_rx_beams)
    macro_loc = np.asarray(macro_bs_loc, dtype=float)
    frames = list(timeline_dir.keys())
    frame_duration_s = args.slots_per_frame * args.slot_len
    learners: Dict[object, _LearnerState] = {}
    total_mbit = 0.0
    connected_time_s = 0.0
    vehicle_time_s = 0.0
    handovers = 0
    beam_switches = 0
    decisions = 0

    for frame in frames:
        records = timeline_dir[frame]
        present = set(records.keys())
        for departed in set(learners.keys()).difference(present):
            del learners[departed]
        for vehicle in sorted(present, key=str):
            record = records[vehicle]
            if vehicle not in learners:
                learners[vehicle] = _LearnerState(
                    action=0,
                    rx_beam=None,
                    pending_action=None,
                    last_position=np.asarray(record["pos"], dtype=float).copy(),
                    distance_since_event=config.zone_size_m,
                )
            learner = learners[vehicle]
            old_bs = action_serving_bs(learner.action, config)
            decision_applied = learner.pending_action is not None
            changed = _apply_pending_action(learner, record, config, dft_tx, dft_rx)
            new_bs = action_serving_bs(learner.action, config)
            if changed:
                beam_switches += 1
                if old_bs != new_bs:
                    handovers += 1
            gain_db = _learner_gain_db(
                args, learner, record, macro_loc, config, dft_tx, dft_rx
            )
            pilot_average = 0.0
            if new_bs > 0:
                pilot_average = float(config.tracking_pilots)
                if decision_applied:
                    pilot_average += (
                        config.receiver_sweep_pilots - config.tracking_pilots
                    ) / args.slots_per_frame
            total_mbit += (
                full_link_rate_bps(args, new_bs, gain_db, pilot_average)
                * frame_duration_s
                / 1e6
            )
            vehicle_time_s += frame_duration_s
            if gain_db > -150.0:
                connected_time_s += frame_duration_s

            current_position = np.asarray(record["pos"], dtype=float)
            learner.distance_since_event += float(
                np.linalg.norm(current_position - learner.last_position)
            )
            learner.last_position = current_position.copy()
            if learner.distance_since_event + 1e-9 >= config.zone_size_m:
                state = policy.make_state(
                    gain_db,
                    learner.action,
                    float(record.get("angle", 0.0)),
                    position=current_position,
                )
                learner.pending_action = policy.select_action(state, rng, explore=False)
                learner.distance_since_event %= config.zone_size_m
                decisions += 1

    duration_s = max((len(frames) - 1) * frame_duration_s, frame_duration_s)
    return {
        "average_full_band_rate_mbps": total_mbit / max(vehicle_time_s, 1e-12),
        "link_availability": connected_time_s / max(vehicle_time_s, 1e-12),
        "handover_per_second": handovers / duration_s,
        "beam_switch_per_second": beam_switches / duration_s,
        "decisions": float(decisions),
        "vehicle_time_s": vehicle_time_s,
    }
