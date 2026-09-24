import unittest
import numpy as np
from wildfire.agents.no_suppression import NoSuppressionAgent
from wildfire.agents.suppression_test_agent import ControlledSuppressionAgent

class TestSuppressionAgents(unittest.TestCase):
    def _make_dummy_obs(self, margin=0.5, payload=1.0, nx=0.5, ny=0.5):
        spatial = np.zeros((5, 11, 11), dtype=np.float32)
        drone = np.array([1.0, payload, margin, 0.5, nx, ny], dtype=np.float32)
        wind = np.zeros(3, dtype=np.float32)
        return {"spatial": spatial, "drone": drone, "wind": wind}

    def test_no_suppression_agent(self):
        agent = NoSuppressionAgent()
        obs = self._make_dummy_obs()
        self.assertEqual(agent.select_action(obs), 0)
        
    def test_controlled_suppression_agent_deploys(self):
        agent = ControlledSuppressionAgent()
        obs = self._make_dummy_obs()
        # Put fire adjacent to center
        obs["spatial"][0, 5, 6] = 0.75 # BURNING
        action = agent.select_action(obs)
        self.assertIn(action, [5, 6])
        
    def test_controlled_suppression_agent_moves(self):
        agent = ControlledSuppressionAgent()
        obs = self._make_dummy_obs()
        # Put fire far away
        obs["spatial"][0, 9, 9] = 0.75 # BURNING
        action = agent.select_action(obs)
        self.assertIn(action, [1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()
