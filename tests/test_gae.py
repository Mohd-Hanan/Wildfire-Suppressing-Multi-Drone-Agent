import unittest
import torch
from wildfire.rl.gae import compute_gae

class TestGAE(unittest.TestCase):
    def test_single_transition_non_terminal(self):
        # Ordinary transition
        rewards = torch.tensor([1.0])
        values = torch.tensor([2.0])
        terminated = torch.tensor([False])
        truncated = torch.tensor([False])
        last_value = 3.0
        
        gamma = 0.99
        
        adv, ret = compute_gae(rewards, values, terminated, truncated, last_value, gamma=gamma, gae_lambda=0.95)
        
        expected_delta = 1.0 + gamma * 3.0 - 2.0
        self.assertAlmostEqual(adv[0].item(), expected_delta, places=4)
        self.assertAlmostEqual(ret[0].item(), expected_delta + 2.0, places=4)
        
    def test_single_transition_terminated(self):
        # Terminated transition
        rewards = torch.tensor([1.0])
        values = torch.tensor([2.0])
        terminated = torch.tensor([True])
        truncated = torch.tensor([False])
        last_value = 3.0 # Should not be used
        
        adv, ret = compute_gae(rewards, values, terminated, truncated, last_value)
        
        expected_delta = 1.0 - 2.0
        self.assertAlmostEqual(adv[0].item(), expected_delta, places=4)
        
    def test_single_transition_truncated(self):
        # Truncated transition
        rewards = torch.tensor([1.0])
        values = torch.tensor([2.0])
        terminated = torch.tensor([False])
        truncated = torch.tensor([True])
        last_value = 3.0
        
        adv, ret = compute_gae(rewards, values, terminated, truncated, last_value, gamma=0.99)
        expected_delta = 1.0 + 0.99 * 3.0 - 2.0
        self.assertAlmostEqual(adv[0].item(), expected_delta, places=4)

    def test_multi_step_rollout(self):
        # 3 steps
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        terminated = torch.tensor([False, False, True])
        truncated = torch.tensor([False, False, False])
        last_value = 2.0 # Shouldn't be used due to termination
        
        gamma = 0.99
        lam = 0.95
        
        adv, ret = compute_gae(rewards, values, terminated, truncated, last_value, gamma, lam)
        
        delta2 = 1.0 + gamma * 0.0 - 0.5
        adv2 = delta2
        
        delta1 = 1.0 + gamma * 0.5 - 0.5
        adv1 = delta1 + gamma * lam * adv2
        
        delta0 = 1.0 + gamma * 0.5 - 0.5
        adv0 = delta0 + gamma * lam * adv1
        
        self.assertAlmostEqual(adv[2].item(), adv2, places=4)
        self.assertAlmostEqual(adv[1].item(), adv1, places=4)
        self.assertAlmostEqual(adv[0].item(), adv0, places=4)

    def test_numerical_stability(self):
        rewards = torch.zeros(100)
        values = torch.zeros(100)
        terminated = torch.zeros(100, dtype=torch.bool)
        truncated = torch.zeros(100, dtype=torch.bool)
        last_value = 0.0
        
        adv, ret = compute_gae(rewards, values, terminated, truncated, last_value)
        self.assertFalse(torch.isnan(adv).any())
        self.assertFalse(torch.isinf(adv).any())
        self.assertEqual(adv.sum().item(), 0.0)

if __name__ == '__main__':
    unittest.main()
