import unittest
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
import gymnasium as gym

class TestWildfireEnv(unittest.TestCase):
    def setUp(self):
        self.env = WildfireEnv()

    def test_environment_construction(self):
        self.assertIsInstance(self.env, gym.Env)

    def test_spaces(self):
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 7)
        
        self.assertIsInstance(self.env.observation_space, gym.spaces.Dict)
        self.assertIn("spatial", self.env.observation_space.spaces)
        self.assertIn("drone", self.env.observation_space.spaces)
        self.assertIn("wind", self.env.observation_space.spaces)

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertIn("spatial", obs)
        self.assertEqual(obs["spatial"].shape, (5, 11, 11))
        self.assertEqual(obs["drone"].shape, (6,))
        self.assertEqual(obs["wind"].shape, (3,))
        self.assertEqual(self.env.step_count, 0)

    def test_step_execution(self):
        obs, info = self.env.reset()
        
        # Valid Stay Action
        next_obs, reward, terminated, truncated, step_info = self.env.step(0)
        
        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(step_info, dict)
        
        self.assertEqual(self.env.step_count, 1)
        
        # Reason should propagate
        self.assertIn('termination_reason', step_info)

    def test_max_step_truncation(self):
        obs, info = self.env.reset()
        self.env.step_count = 499
        
        next_obs, reward, terminated, truncated, step_info = self.env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(step_info['termination_reason'], 'max_steps')

    def test_fire_extinction_precedence(self):
        obs, info = self.env.reset()
        self.env.step_count = 499
        
        # Extinguish fire
        from wildfire.simulation.fire import FireState
        self.env.world.fire_manager.fire_map[:] = FireState.UNBURNED
        
        next_obs, reward, terminated, truncated, step_info = self.env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated) # Extinction wins
        self.assertEqual(step_info['termination_reason'], 'fire_extinguished')

    def test_drone_crash_no_termination(self):
        obs, info = self.env.reset()
        drone = self.env.world.drones[self.env.controlled_drone_idx]
        drone.battery = 1 # Move costs 1
        
        next_obs, reward, terminated, truncated, step_info = self.env.step(1) # Move North
        
        self.assertEqual(drone.battery, 0)
        self.assertFalse(drone.active)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_zero_payload_no_termination(self):
        obs, info = self.env.reset()
        drone = self.env.world.drones[self.env.controlled_drone_idx]
        drone.payload = 1
        drone.x = 10
        drone.y = 10
        drone.battery = 10
        
        next_obs, reward, terminated, truncated, step_info = self.env.step(5) # Drop Water
        
        self.assertEqual(drone.payload, 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

if __name__ == '__main__':
    unittest.main()
