import unittest
import torch
from stage7_15_smoke_test import SeparateActorCritic

class TestActionMasking(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.policy = SeparateActorCritic(device=self.device, drone_type="WATER")
        
        # Create dummy observation
        self.obs = {
            "spatial": torch.zeros((1, 5, 11, 11), dtype=torch.float32, device=self.device),
            "drone": torch.zeros((1, 9), dtype=torch.float32, device=self.device),
            "wind": torch.zeros((1, 3), dtype=torch.float32, device=self.device),
        }

    def test_water_drone_cannot_sample_action_6(self):
        """WATER drone cannot sample action 6 (RETARDANT)."""
        action, log_prob, entropy, value = self.policy.get_action_and_value(self.obs)
        
        # Sample 1000 times to be sure
        for _ in range(1000):
            a, _, _, _ = self.policy.get_action_and_value(self.obs)
            self.assertNotEqual(a.item(), 6, "Action 6 was sampled for WATER drone!")

    def test_valid_actions_remain_available(self):
        """valid actions 0-5 remain available."""
        logits, _ = self.policy(self.obs)
        self.assertNotEqual(logits[0, 0].item(), -1e9)
        self.assertNotEqual(logits[0, 5].item(), -1e9)
        
    def test_masked_action_has_zero_probability(self):
        """masked action has effectively zero probability."""
        logits, _ = self.policy(self.obs)
        self.assertEqual(logits[0, 6].item(), -1e9)
        
        probs = torch.distributions.Categorical(logits=logits).probs
        self.assertEqual(probs[0, 6].item(), 0.0)

    def test_log_prob_and_entropy_calculated_from_masked_distribution(self):
        """log_prob and entropy are calculated from the masked distribution."""
        action = torch.tensor([6])
        logits, _ = self.policy(self.obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        self.assertEqual(dist.log_prob(action).item(), -1e9)
        # Entropy should only account for 6 actions (max ln(6) = 1.7917)
        self.assertLessEqual(dist.entropy().item(), 1.7918)
        
    def test_ppo_old_new_log_prob_calculations_use_same_mask(self):
        """PPO old/new log-prob calculations use the same mask."""
        action = torch.tensor([5])
        
        # Simulating rollout
        _, old_log_prob, _, _ = self.policy.get_action_and_value(self.obs, action=action)
        
        # Simulating update
        _, new_log_prob, _, _ = self.policy.get_action_and_value(self.obs, action=action)
        
        self.assertEqual(old_log_prob.item(), new_log_prob.item())

if __name__ == '__main__':
    unittest.main()
