import unittest
import torch
import numpy as np
from wildfire.rl.policy import ActorCritic
from wildfire.environment.wildfire_env import WildfireEnv

class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.policy = ActorCritic(device=self.device)
        self.env = WildfireEnv()
        
    def test_construction(self):
        self.assertIsInstance(self.policy, ActorCritic)
        
    def test_forward_unbatched_numpy(self):
        obs, _ = self.env.reset()
        logits, value = self.policy(obs)
        
        self.assertEqual(logits.shape, (1, 7))
        self.assertEqual(value.shape, (1,))
        
    def test_forward_batched_tensor(self):
        batch_size = 4
        obs = {
            "spatial": torch.rand(batch_size, 5, 11, 11),
            "drone": torch.rand(batch_size, 6),
            "wind": torch.rand(batch_size, 3)
        }
        logits, value = self.policy(obs)
        
        self.assertEqual(logits.shape, (batch_size, 7))
        self.assertEqual(value.shape, (batch_size,))
        
    def test_action_sampling(self):
        obs, _ = self.env.reset()
        action, log_prob, entropy, value = self.policy.get_action_and_value(obs)
        
        self.assertEqual(action.shape, (1,))
        self.assertTrue(0 <= action.item() < 7)
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(entropy.shape, (1,))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue((entropy >= 0).all())
        
    def test_supplied_action(self):
        obs, _ = self.env.reset()
        action = torch.tensor([5])
        _, log_prob, entropy, value = self.policy.get_action_and_value(obs, action=action)
        
        self.assertEqual(log_prob.shape, (1,))
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_env_integration(self):
        obs, _ = self.env.reset()
        action, _, _, _ = self.policy.get_action_and_value(obs)
        
        act_int = action.item()
        next_obs, reward, terminated, truncated, info = self.env.step(act_int)
        
        self.assertIsInstance(next_obs, dict)
        
    def test_backward_pass(self):
        obs, _ = self.env.reset()
        action, log_prob, entropy, value = self.policy.get_action_and_value(obs)
        
        # Fake loss
        loss = -log_prob.mean() + value.mean()
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in self.policy.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad)

    def test_numerical_stability(self):
        obs, _ = self.env.reset()
        logits, value = self.policy(obs)
        
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())
        self.assertFalse(torch.isnan(value).any())
        self.assertFalse(torch.isinf(value).any())

if __name__ == '__main__':
    unittest.main()
