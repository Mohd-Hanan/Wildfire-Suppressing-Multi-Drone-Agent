import unittest
import torch
import numpy as np
import copy
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.rollout import RolloutBuffer, RolloutCollector

class TestRollout(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.env = WildfireEnv()
        self.policy = ActorCritic(device=self.device)
        
    def test_buffer_construction_and_clear(self):
        buffer = RolloutBuffer(128)
        self.assertEqual(len(buffer), 0)
        
        obs, _ = self.env.reset()
        buffer.add(obs, 1, 10.0, False, False, 0.5, -0.2)
        self.assertEqual(len(buffer), 1)
        
        buffer.clear()
        self.assertEqual(len(buffer), 0)
        
    def test_buffer_get_batch(self):
        buffer = RolloutBuffer(2)
        obs, _ = self.env.reset()
        buffer.add(obs, 1, 10.0, False, False, 0.5, -0.2)
        buffer.add(obs, 2, -5.0, True, False, 0.1, -0.4)
        
        spatial, drone, wind, actions, rewards, terminated, truncated, values, log_probs = buffer.get_batch(self.device)
        
        self.assertEqual(spatial.shape, (2, 5, 11, 11))
        self.assertEqual(drone.shape, (2, 6))
        self.assertEqual(wind.shape, (2, 3))
        
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(terminated.shape, (2,))
        self.assertEqual(truncated.shape, (2,))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(log_probs.shape, (2,))
        
        self.assertEqual(spatial.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.long)
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(terminated.dtype, torch.bool)
        
    def test_collector_basic(self):
        collector = RolloutCollector(self.env, self.policy, 32, self.device)
        collector.collect()
        
        self.assertEqual(len(collector.buffer), 32)
        
        _, _, _, actions, rewards, terms, truncs, vals, lps = collector.buffer.get_batch()
        
        self.assertTrue((actions >= 0).all() and (actions < 7).all())
        self.assertTrue(torch.isfinite(rewards).all())
        self.assertTrue(torch.isfinite(vals).all())
        self.assertTrue(torch.isfinite(lps).all())

    def test_episode_boundary(self):
        # max_steps is 500, so a rollout of 550 will cross the boundary
        collector = RolloutCollector(self.env, self.policy, 550, self.device)
        collector.collect()
        
        self.assertEqual(len(collector.buffer), 550)
        
        _, _, _, _, _, terms, truncs, _, _ = collector.buffer.get_batch()
        
        # Should contain exactly one truncation or termination (most likely truncation at step 500)
        total_ends = (terms | truncs).sum().item()
        self.assertGreaterEqual(total_ends, 1)

    def test_policy_immutability(self):
        params_before = [p.clone() for p in self.policy.parameters()]
        
        collector = RolloutCollector(self.env, self.policy, 10, self.device)
        collector.collect()
        
        params_after = [p for p in self.policy.parameters()]
        
        for p_b, p_a in zip(params_before, params_after):
            self.assertTrue(torch.equal(p_b, p_a))

    def test_no_gradients_in_buffer(self):
        collector = RolloutCollector(self.env, self.policy, 10, self.device)
        collector.collect()
        
        _, _, _, _, _, _, _, vals, lps = collector.buffer.get_batch()
        
        self.assertFalse(vals.requires_grad)
        self.assertFalse(lps.requires_grad)

    def test_observation_space(self):
        collector = RolloutCollector(self.env, self.policy, 5, self.device)
        collector.collect()
        
        # Test the raw lists in buffer
        for i in range(len(collector.buffer)):
            obs = {
                "spatial": collector.buffer.spatial[i],
                "drone": collector.buffer.drone[i],
                "wind": collector.buffer.wind[i]
            }
            self.assertTrue(self.env.observation_space.contains(obs))

if __name__ == '__main__':
    unittest.main()
