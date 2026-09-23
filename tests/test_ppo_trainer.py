import unittest
import torch
import numpy as np
from wildfire.rl.policy import ActorCritic
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.gae import compute_gae
from unittest.mock import patch

class TestPPOTrainer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
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
        drone = torch.rand(N, 9)
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
        
        self.trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        
        params_after = list(self.policy.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        self.assertTrue(changed)

    def test_rollout_immutability(self):
        N = 32
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        spatial_clone = spatial.clone()
        drone_clone = drone.clone()
        wind_clone = wind.clone()
        actions_clone = actions.clone()
        old_log_probs_clone = old_log_probs.clone()
        advantages_clone = advantages.clone()
        returns_clone = returns.clone()
        
        self.trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        
        self.assertTrue(torch.equal(spatial, spatial_clone))
        self.assertTrue(torch.equal(drone, drone_clone))
        self.assertTrue(torch.equal(wind, wind_clone))
        self.assertTrue(torch.equal(actions, actions_clone))
        self.assertTrue(torch.equal(old_log_probs, old_log_probs_clone)) # Also verifies old_log_probs immutability
        self.assertTrue(torch.equal(advantages, advantages_clone))
        self.assertTrue(torch.equal(returns, returns_clone))

    def test_advantage_normalization(self):
        # Deterministic advantage with non-zero variance
        advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        norm_adv = PPOTrainer._normalize_advantages(advantages)
        
        self.assertAlmostEqual(norm_adv.mean().item(), 0.0, places=4)
        # Using unbiased std (ddof=1) which is default for torch.std
        self.assertAlmostEqual(norm_adv.std().item(), 1.0, places=4)

    def test_optimizer_configuration(self):
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Adam)
        for param_group in self.trainer.optimizer.param_groups:
            self.assertEqual(param_group['lr'], 1e-3)

    def test_multiple_ppo_epochs(self):
        # We can intercept optimizer.step() to count how many times it is called
        N = 32
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        trainer_e1 = PPOTrainer(self.policy, ppo_epochs=1, minibatch_size=16)
        with patch.object(trainer_e1.optimizer, 'step') as mock_step:
            trainer_e1.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
            self.assertEqual(mock_step.call_count, 2) # N=32, bs=16 => 2 minibatches * 1 epoch = 2
            
        trainer_e2 = PPOTrainer(self.policy, ppo_epochs=2, minibatch_size=16)
        with patch.object(trainer_e2.optimizer, 'step') as mock_step:
            trainer_e2.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
            self.assertEqual(mock_step.call_count, 4) # N=32, bs=16 => 2 minibatches * 2 epochs = 4

    def test_gradient_clipping(self):
        # We will set a very small max_grad_norm and check if the returned metric respects it
        N = 32
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        # Artificially inflate advantages to produce large gradients
        advantages = advantages * 1000.0
        
        strict_trainer = PPOTrainer(self.policy, max_grad_norm=0.01, ppo_epochs=1, minibatch_size=32)
        
        metrics = strict_trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        
        # The unclipped norm might be large, but the parameter gradients after clipping should have norm <= 0.01 (with tiny tolerance)
        total_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.assertLessEqual(total_norm, 0.01 + 1e-4)

    def test_indivisible_minibatch(self):
        N = 35 # Not divisible by 16
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        metrics = self.trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        self.assertIn("total_loss", metrics)

    def test_metrics_aggregation(self):
        N = 32
        spatial, drone, wind, actions, old_log_probs, rewards, values, terminated, truncated = self.generate_dummy_data(N)
        advantages, returns = compute_gae(rewards, values, terminated, truncated, last_value=0.5)
        
        # Run standard update which internally aggregates and averages over 4 updates
        metrics = self.trainer.update(spatial, drone, wind, actions, old_log_probs, advantages, returns)
        
        for k, v in metrics.items():
            self.assertTrue(np.isfinite(v))

if __name__ == '__main__':
    unittest.main()
