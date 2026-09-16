"""Focused tests for the O-MAPPO baseline adaptation."""

import os
import tempfile
import unittest

import numpy as np

from experiment.pql_ba_experiment import paper_args
from utils.beam_utils import generate_dft_codebook
from utils.o_mappo import (
    OMAPPOCommand,
    OMAPPOConfig,
    OMAPPOLearnerState,
    OMAPPOMemory,
    OMAPPPolicy,
    PPOTransition,
    apply_o_mappo_command,
    append_state_sequence,
    candidate_feasibility_context,
    make_global_state,
    make_local_state,
    optimize_triggered_targets,
    source_gate_allows,
    state_feature_names,
)


class OMAPPOTest(unittest.TestCase):
    def test_state_and_pooled_global_dimensions(self):
        for variant, expected in (("source", 13), ("adapted", 31)):
            config = OMAPPOConfig(state_variant=variant)
            state = make_local_state(
                config,
                position=(10.0, -20.0),
                heading_deg=45.0,
                speed_mps=10.0,
                serving_bs=0,
                serving_sinr_db=5.0,
                queue_ratio=0.5,
                traffic_mbps=19.0,
                rb_load=np.zeros(5),
                user_load=np.ones(5),
                interference_db=-np.inf,
                previous_handover=False,
                system_throughput_ratio=0.9,
                own_rb_fraction=0.1,
                tx_beam=None,
                rx_beam=None,
            )
            self.assertEqual(len(state), expected)
            self.assertEqual(len(state_feature_names(config)), expected)
            pooled = make_global_state(np.stack((state, state)), 2)
            self.assertEqual(pooled.shape, (3 * expected + 1,))
            self.assertTrue(np.isfinite(pooled).all())

    def test_feasibility_state_and_recurrent_roundtrip(self):
        args = paper_args()
        config = OMAPPOConfig(
            state_variant="feasibility",
            candidate_count=4,
            hidden_sizes=(32,),
            recurrent=True,
            recurrent_hidden_size=32,
            recurrent_sequence_length=4,
            ppo_epochs=2,
            batch_size=4,
        )
        dft_tx = generate_dft_codebook(config.num_tx_beams)
        dft_rx = generate_dft_codebook(config.num_rx_beams)
        rng = np.random.default_rng(11)
        record = {
            "pos": np.asarray([20.0, -30.0]),
            "h": rng.normal(size=(8, 4, 32))
            + 1j * rng.normal(size=(8, 4, 32)),
        }
        learner = OMAPPOLearnerState(
            action=0,
            rx_beam=None,
            pending_action=None,
            last_position=record["pos"].copy(),
            distance_since_event=10.0,
        )
        context = candidate_feasibility_context(
            args,
            record,
            learner,
            backlog_bits=2e6,
            load=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            config=config,
            dft_tx=dft_tx,
            dft_rx=dft_rx,
        )
        state = make_local_state(
            config,
            position=record["pos"],
            heading_deg=45.0,
            speed_mps=10.0,
            serving_bs=0,
            serving_sinr_db=5.0,
            queue_ratio=0.5,
            traffic_mbps=19.0,
            rb_load=np.zeros(5),
            user_load=np.ones(5),
            interference_db=-np.inf,
            previous_handover=False,
            system_throughput_ratio=0.9,
            own_rb_fraction=0.1,
            tx_beam=None,
            rx_beam=None,
            candidate_sinr_db=context[0],
            candidate_demand_ratio=context[1],
            candidate_residual_ratio=context[2],
            candidate_feasibility_margin=context[3],
            optimizer_feedback=np.zeros(4),
        )
        self.assertEqual(state.shape, (55,))
        sequence = append_state_sequence([], state, 4)
        self.assertEqual(sequence.shape, (4, 55))
        global_state = make_global_state(np.stack((state, state)), 2)
        global_sequence = append_state_sequence([], global_state, 4)
        policy = OMAPPPolicy(config, seed=11)
        actions, _, values = policy.act(
            np.stack((sequence, sequence)), global_sequence, explore=False
        )
        self.assertEqual(actions.shape, (2,))
        self.assertTrue(np.isfinite(values).all())

    def test_source_gate(self):
        source = OMAPPOConfig(trigger_gate="source")
        periodic = OMAPPOConfig(trigger_gate="periodic")
        self.assertTrue(source_gate_allows(source, -5.0, [-20.0, -10.0]))
        self.assertTrue(source_gate_allows(source, 10.0, [3.0, 4.0]))
        self.assertFalse(source_gate_allows(source, 10.0, [3.0, -4.0]))
        self.assertTrue(source_gate_allows(periodic, 30.0, []))

    def test_command_tracks_or_changes_bs(self):
        config = OMAPPOConfig()
        dft_tx = generate_dft_codebook(config.num_tx_beams)
        dft_rx = generate_dft_codebook(config.num_rx_beams)
        record = {
            "pos": np.zeros(2),
            "h": np.ones((8, 4, 32), dtype=np.complex128),
        }
        learner = OMAPPOLearnerState(
            action=0,
            rx_beam=None,
            pending_action=None,
            last_position=np.zeros(2),
            distance_since_event=10.0,
        )
        no_ho = apply_o_mappo_command(
            learner, OMAPPOCommand(0, 0), record, config, dft_tx, dft_rx
        )
        self.assertFalse(no_ho.handover)
        handover = apply_o_mappo_command(
            learner, OMAPPOCommand(1, 1), record, config, dft_tx, dft_rx
        )
        self.assertTrue(handover.handover)
        self.assertEqual(learner.action, 1)
        self.assertEqual(handover.sweep_pilots, config.full_sweep_pilots)
        track = apply_o_mappo_command(
            learner, OMAPPOCommand(0, 1), record, config, dft_tx, dft_rx
        )
        self.assertFalse(track.handover)
        self.assertEqual(track.sweep_pilots, config.tracking_sweep_pilots)

    def test_optimizer_excludes_current_bs_with_both_solvers(self):
        args = paper_args()
        config = OMAPPOConfig(candidate_count=3)
        dft_tx = generate_dft_codebook(config.num_tx_beams)
        dft_rx = generate_dft_codebook(config.num_rx_beams)
        rng = np.random.default_rng(3)
        records = {
            "v0": {
                "pos": np.asarray([20.0, 30.0]),
                "h": rng.normal(size=(8, 4, 32))
                + 1j * rng.normal(size=(8, 4, 32)),
            },
            "v1": {
                "pos": np.asarray([-20.0, -30.0]),
                "h": rng.normal(size=(8, 4, 32))
                + 1j * rng.normal(size=(8, 4, 32)),
            },
        }
        learners = {
            vehicle: OMAPPOLearnerState(
                action=0,
                rx_beam=None,
                pending_action=None,
                last_position=record["pos"].copy(),
                distance_since_event=10.0,
            )
            for vehicle, record in records.items()
        }
        for solver in ("greedy", "milp"):
            result = optimize_triggered_targets(
                args,
                records,
                learners,
                list(records),
                {vehicle: 2e6 for vehicle in records},
                {vehicle: 1.0 for vehicle in records},
                np.zeros(5),
                config,
                dft_tx,
                dft_rx,
                solver=solver,
            )
            self.assertEqual(set(result.targets), set(records))
            self.assertTrue(all(target != 0 for target in result.targets.values()))
            self.assertTrue(np.isfinite(result.objective))

    def test_ppo_update_and_checkpoint_roundtrip(self):
        config = OMAPPOConfig(state_variant="source", ppo_epochs=2, batch_size=4)
        policy = OMAPPPolicy(config, seed=4)
        local = np.zeros((2, policy.local_dim), dtype=np.float32)
        global_state = make_global_state(local, 2)
        actions, log_probs, values = policy.act(local, global_state, explore=True)
        memory = OMAPPOMemory()
        for index in range(2):
            memory.add(
                PPOTransition(
                    vehicle="v{}".format(index),
                    local_state=local[index],
                    global_state=global_state,
                    action=int(actions[index]),
                    old_log_probability=float(log_probs[index]),
                    old_value=float(values[index]),
                    reward=1.0,
                    next_value=0.0,
                    done=True,
                )
            )
        result = policy.update(memory)
        self.assertEqual(result["transitions"], 2.0)
        self.assertTrue(np.isfinite(result["actor_loss"]))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.pt")
            policy.save(path)
            loaded = OMAPPPolicy.load(path, seed=4)
            first = policy.act(local, global_state, explore=False)[0]
            second = loaded.act(local, global_state, explore=False)[0]
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
