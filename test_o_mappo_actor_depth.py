"""Actor-only depth changes must preserve old checkpoints and critic design."""
import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.o_mappo import OMAPPOConfig, OMAPPPolicy, _MLP
from experiment.train_o_mappo_hierarchical32 import configuration, initial_policy


class ActorDepthTest(unittest.TestCase):
    def test_defaults_preserve_original_initialization(self):
        policy = OMAPPPolicy(OMAPPOConfig(torch_threads=1), seed=22)
        torch.manual_seed(22)
        actor, critic = _MLP(31, 2, (64,)), _MLP(94, 1, (64,))
        for actual, expected in ((policy.actor, actor), (policy.critic, critic)):
            for k,v in actual.state_dict().items():
                self.assertTrue(torch.equal(v, expected.state_dict()[k]))

    def test_actor_only_depth_and_round_trip(self):
        policy = OMAPPPolicy(OMAPPOConfig(actor_hidden_sizes=(64,64), torch_threads=1),seed=11)
        self.assertEqual(sum(p.numel() for p in policy.actor.parameters()), 6338)
        self.assertEqual(sum(p.numel() for p in policy.critic.parameters()), 6145)
        self.assertEqual(tuple(policy.actor(torch.zeros(3,31)).shape), (3,2))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'policy.pt')
            policy.save(path)
            restored = OMAPPPolicy.load(path, load_optimizers=True)
            a = np.random.default_rng(8).normal(size=(6,31)).astype(np.float32)
            g = np.zeros(94,dtype=np.float32)
            np.testing.assert_array_equal(policy.act(a,g,False)[0],restored.act(a,g,False)[0])
            for k,v in policy.actor.state_dict().items():
                self.assertTrue(torch.equal(v, restored.actor.state_dict()[k]))

    def test_same_initial_critic_and_input_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'initial.pt'
            original = initial_policy(configuration(),22)
            original.save(str(path))
            changed = initial_policy(configuration((64,64)),22,path)
            for k,v in original.critic.state_dict().items():
                self.assertTrue(torch.equal(v,changed.critic.state_dict()[k]))
            self.assertTrue(torch.equal(original.actor.model[0].weight,changed.actor.model[0].weight))
            self.assertTrue(torch.equal(original.actor.model[0].bias,changed.actor.model[0].bias))

    def test_old_checkpoint_without_new_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old.pt'
            original = OMAPPPolicy(OMAPPOConfig(torch_threads=1),seed=33)
            original.save(str(path))
            data = torch.load(path,weights_only=False)
            del data['config']['actor_hidden_sizes']
            torch.save(data,path)
            restored=OMAPPPolicy.load(str(path))
            for k,v in original.actor.state_dict().items():
                self.assertTrue(torch.equal(v,restored.actor.state_dict()[k]))

    def test_invalid_hidden_sizes(self):
        for sizes in ((),(0,),(-1,64),(64.5,)):
            with self.assertRaises(ValueError):
                OMAPPOConfig(actor_hidden_sizes=sizes).validate()


if __name__ == '__main__':
    unittest.main()
