import unittest
import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.rollout import RolloutCollector

class TestRolloutBootstrap(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.env = WildfireEnv()
        self.policy = ActorCritic(device=self.device)
        
    def test_mid_episode_bootstrap(self):
        collector = RolloutCollector(self.env, self.policy, 5, self.device)
        collector.collect()
        
        self.assertIsNotNone(collector.last_obs)
        self.assertIsNotNone(collector.last_value)
        self.assertTrue(self.env.observation_space.contains(collector.last_obs))
        self.assertTrue(np.isfinite(collector.last_value))
        
    def test_truncation_boundary_bootstrap(self):
        self.env.termination_checker.max_steps = 10
        collector = RolloutCollector(self.env, self.policy, 10, self.device)
        
        collector.collect()
        
        _, _, _, _, _, terms, truncs, _, _ = collector.buffer.get_batch()
        
        self.assertTrue(truncs[-1].item() or terms[-1].item())
        
        self.assertEqual(self.env.step_count, 0)
        self.assertTrue(self.env.observation_space.contains(collector.last_obs))
        self.assertIsNot(collector.last_obs, collector.current_obs)

if __name__ == '__main__':
    unittest.main()
