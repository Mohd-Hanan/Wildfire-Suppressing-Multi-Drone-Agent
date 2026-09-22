import unittest
import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.evaluate import evaluate_episode, evaluate_policy

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.env = WildfireEnv()
        self.device = torch.device("cpu")
        self.policy = ActorCritic(device=self.device)
        
    def test_random_evaluation(self):
        res = evaluate_episode(self.env, seed=42)
        
        self.assertIn("reward", res)
        self.assertIn("length", res)
        self.assertIn("burned_cells", res)
        self.assertGreater(res["length"], 0)
        
        # New checks
        self.assertEqual(sum(res["action_counts"].values()), res["length"])
        self.assertIn("initial_battery", res)
        self.assertTrue(np.isfinite(res["final_battery"]))
        
    def test_deterministic_evaluation(self):
        params_before = [p.clone() for p in self.policy.parameters()]
        
        res1 = evaluate_episode(self.env, seed=42, policy=self.policy, device=self.device)
        res2 = evaluate_episode(self.env, seed=42, policy=self.policy, device=self.device)
        
        self.assertEqual(res1["reward"], res2["reward"])
        self.assertEqual(res1["length"], res2["length"])
        self.assertDictEqual(res1["action_counts"], res2["action_counts"])
        
        params_after = list(self.policy.parameters())
        for b, a in zip(params_before, params_after):
            self.assertTrue(torch.equal(b, a))
            
    def test_evaluate_policy_aggregate(self):
        seeds = [42, 43]
        agg = evaluate_policy(self.env, seeds, policy=self.policy, device=self.device)
        
        self.assertEqual(len(agg["episodes"]), 2)
        self.assertIn("mean_reward", agg)
        self.assertIn("action_counts", agg)
        self.assertIn("action_pct", agg)
        
        # Check percentage sums to approx 100
        pct_sum = sum(agg["action_pct"].values())
        self.assertAlmostEqual(pct_sum, 100.0, places=4)

if __name__ == '__main__':
    unittest.main()
