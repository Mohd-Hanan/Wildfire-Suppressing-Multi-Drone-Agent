import unittest
import torch
import numpy as np
from wildfire.rl.train import train_ppo
import sys
import io

class TestTraining(unittest.TestCase):
    def test_smoke_training(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            policy, metrics, total_steps, ep_stats = train_ppo(
                updates=2,
                rollout_size=16,
                ppo_epochs=1,
                minibatch_size=16,
                seed=42,
                device_name="cpu"
            )
        finally:
            sys.stdout = sys.__stdout__
            
        self.assertEqual(total_steps, 32)
        
        self.assertIn("total_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("gradient_norm", metrics)
        self.assertIn("rollout_reward_mean", metrics)
        self.assertIn("advantage_mean", metrics)
        self.assertIn("return_mean", metrics)
        
        for k, v in metrics.items():
            self.assertTrue(np.isfinite(v), f"Metric {k} is non-finite.")
            
        output = captured_output.getvalue()
        self.assertIn("Training started...", output)
        self.assertIn("Training completed.", output)
        self.assertIn("Update: 2/2", output)

if __name__ == '__main__':
    unittest.main()
