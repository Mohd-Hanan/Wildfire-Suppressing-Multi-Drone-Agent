import unittest
import torch
import numpy as np
from wildfire.rl.ppo import compute_ppo_loss

class TestPPO(unittest.TestCase):
    def test_ratio_calculation(self):
        old_log_prob = torch.tensor([torch.log(torch.tensor(0.5))])
        new_log_prob = torch.tensor([torch.log(torch.tensor(0.75))])
        
        # dummy other tensors
        adv = torch.tensor([1.0])
        val = torch.tensor([1.0])
        ret = torch.tensor([1.0])
        ent = torch.tensor([0.0])
        
        loss, metrics = compute_ppo_loss(new_log_prob, old_log_prob, adv, val, ret, ent)
        self.assertAlmostEqual(metrics["ratio_mean"], 1.5, places=4)

    def test_no_clipping(self):
        old_log_prob = torch.tensor([torch.log(torch.tensor(0.5))])
        new_log_prob = torch.tensor([torch.log(torch.tensor(0.55))]) # ratio 1.1
        adv = torch.tensor([1.0])
        
        loss, metrics = compute_ppo_loss(new_log_prob, old_log_prob, adv, torch.zeros(1), torch.zeros(1), torch.zeros(1), clip_epsilon=0.2)
        
        self.assertAlmostEqual(metrics["clip_fraction"], 0.0, places=4)
        self.assertAlmostEqual(metrics["policy_loss"], -1.1, places=4) # - (1.1 * 1.0)

    def test_positive_advantage_clipping(self):
        old_log_prob = torch.tensor([torch.log(torch.tensor(0.5))])
        new_log_prob = torch.tensor([torch.log(torch.tensor(0.75))]) # ratio 1.5
        adv = torch.tensor([1.0])
        
        loss, metrics = compute_ppo_loss(new_log_prob, old_log_prob, adv, torch.zeros(1), torch.zeros(1), torch.zeros(1), clip_epsilon=0.2)
        
        self.assertAlmostEqual(metrics["clip_fraction"], 1.0, places=4)
        self.assertAlmostEqual(metrics["policy_loss"], -1.2, places=4) # max clipped at 1.2 * 1.0

    def test_negative_advantage_clipping(self):
        old_log_prob = torch.tensor([torch.log(torch.tensor(0.5))])
        new_log_prob = torch.tensor([torch.log(torch.tensor(0.25))]) # ratio 0.5
        adv = torch.tensor([-1.0])
        
        loss, metrics = compute_ppo_loss(new_log_prob, old_log_prob, adv, torch.zeros(1), torch.zeros(1), torch.zeros(1), clip_epsilon=0.2)
        
        self.assertAlmostEqual(metrics["clip_fraction"], 1.0, places=4)
        self.assertAlmostEqual(metrics["policy_loss"], 0.8, places=4) # clipped surrogate is 0.8 * -1.0 = -0.8 -> loss is 0.8

    def test_value_loss(self):
        val = torch.tensor([0.0, 1.0])
        ret = torch.tensor([2.0, 4.0])
        loss, metrics = compute_ppo_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), val, ret, torch.zeros(2))
        
        self.assertAlmostEqual(metrics["value_loss"], 6.5, places=4)

    def test_total_loss_composition(self):
        new_log = torch.zeros(1, requires_grad=True)
        old_log = torch.zeros(1)
        adv = torch.tensor([1.0])
        val = torch.tensor([0.0], requires_grad=True)
        ret = torch.tensor([2.0])
        ent = torch.tensor([3.0], requires_grad=True)
        
        loss, metrics = compute_ppo_loss(new_log, old_log, adv, val, ret, ent, value_coef=0.5, entropy_coef=0.01)
        
        self.assertAlmostEqual(loss.item(), 0.97, places=4)
        
        loss.backward()
        self.assertIsNotNone(new_log.grad)
        self.assertIsNotNone(val.grad)
        self.assertIsNotNone(ent.grad)

    def test_approx_kl(self):
        old_log_prob = torch.tensor([torch.log(torch.tensor(0.5))])
        new_log_prob = torch.tensor([torch.log(torch.tensor(0.25))]) 
        
        loss, metrics = compute_ppo_loss(new_log_prob, old_log_prob, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        
        self.assertAlmostEqual(metrics["approx_kl"], np.log(2), places=4)

    def test_numerical_stability(self):
        zeros = torch.zeros(100)
        loss, metrics = compute_ppo_loss(zeros, zeros, zeros, zeros, zeros, zeros)
        
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

if __name__ == '__main__':
    unittest.main()
