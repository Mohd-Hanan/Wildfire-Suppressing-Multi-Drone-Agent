import unittest
import torch
import numpy as np
from wildfire.rl.policy import ActorCritic
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.gae import compute_gae

class TestPPOTrainer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.policy = ActorCritic(device=self.device)
        self.trainer = PPOTrainer(
            self.policy, 
            device=self.device,
            learning_rate=1e-3,
            ppo_epochs=2,
            minibatch_size=16
        )

    def generate_dummy_data(self, N):
        spatial = torch.rand(N, 5, 11, 11)
        drone = torch.rand(N, 6)
        wind = torch.rand(N, 3)
        actions = torch.randint(0, 7, (N,))
        old_log_probs = torch.randn(N)
        
        rewards = torch.rand(N)
        values = torch.rand(N)
        terminated = torch.zeros(N, dtype=torch.bool)
        truncated = torch.zeros(N, dtype=torch.bool)
        
        return spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated

    def test_update_changes_parameters(self):
        N = 32
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        params_before = [p.clone() for p in self.policy.parameters()]
        
        metrics = self.trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        
        params_after = list(self.policy.parameters())
        
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        self.assertTrue(changed, "Policy parameters did not change during update.")
        
    def test_update_metrics_finite(self):
        N = 32
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        metrics = self.trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        
        for k, v in metrics.items():
            self.assertTrue(np.isfinite(v), f"Metric {k} is not finite: {v}")
            
    def test_indivisible_minibatch(self):
        N = 35 # Not divisible by 16
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        # Should not crash
        metrics = self.trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        self.assertIn("total_loss", metrics)

if __name__ == '__main__':
    unittest.main()
