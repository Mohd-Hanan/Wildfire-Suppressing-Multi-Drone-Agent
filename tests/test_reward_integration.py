import unittest
import yaml
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.simulation.fire import FireState

class TestRewardIntegration(unittest.TestCase):
    def setUp(self):
        self.env = WildfireEnv("configs/environment.yaml")
        self.env.reset(seed=42)
        # Clear fire completely
        self.env.world.fire_manager.fire_map[:] = FireState.UNBURNED
        self.drone = self.env.world.drones[self.env.controlled_drone_idx]
        self.drone.battery = 100
        self.drone.payload = 10

    def test_A_successful_water_reward(self):
        self.drone.x, self.drone.y = 10, 10
        self.env.world.fire_manager.ignite(10, 10)
        self.env.world.fire_manager.ignite(11, 10) # to prevent extinction
        self.env.step(0) # init reward
        
        obs, reward, term, trunc, info = self.env.step(5) # WATER
        self.assertEqual(info['newly_suppressed_cells'], 1)
        self.assertEqual(info['suppression_reward'], 10.0)

    def test_B_strict_causal_extinction(self):
        self.drone.x, self.drone.y = 10, 10
        self.env.world.fire_manager.ignite(10, 10)
        self.env.step(0) # init
        
        # Extinguish only cell
        obs, reward, term, trunc, info = self.env.step(5) # WATER
        self.assertEqual(info['extinction_reward'], 50.0)
        
    def test_C_natural_extinction_zero(self):
        self.drone.x, self.drone.y = 10, 10
        self.env.world.fire_manager.ignite(10, 10)
        self.env.step(0) # init
        
        # Force natural burnout
        self.env.world.fire_manager.fire_map[10, 10] = FireState.UNBURNED
        obs, reward, term, trunc, info = self.env.step(0) # STAY
        
        self.assertEqual(info['extinction_reward'], 0.0)
        
    def test_D_multiple_fires_no_extinction(self):
        self.drone.x, self.drone.y = 10, 10
        self.env.world.fire_manager.ignite(10, 10)
        self.env.world.fire_manager.ignite(20, 20)
        self.env.step(0) # init
        
        obs, reward, term, trunc, info = self.env.step(5) # WATER
        self.assertEqual(info['newly_suppressed_cells'], 1)
        self.assertEqual(info['suppression_reward'], 10.0)
        self.assertEqual(info['extinction_reward'], 0.0)

    def test_E_early_suppression_late_extinction(self):
        self.drone.x, self.drone.y = 10, 10
        self.env.world.fire_manager.ignite(10, 10)
        self.env.world.fire_manager.ignite(20, 20)
        self.env.step(0) # init
        
        # Suppress first
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['suppression_reward'], 10.0)
        self.assertEqual(info['extinction_reward'], 0.0)
        
        # Later natural extinction
        self.env.world.fire_manager.fire_map[20, 20] = FireState.UNBURNED
        obs, reward, term, trunc, info = self.env.step(0)
        self.assertEqual(info['extinction_reward'], 0.0)

    def test_F_frozen_target_distance_shaping(self):
        # drone at 10, 10
        self.drone.x, self.drone.y = 10, 10
        self.env.world.fire_manager.ignite(12, 10) # dist 2
        self.env.step(0) # init
        
        # Move drone right (to 11,10)
        # Simultaneously fire spreads and old fire dies (simulate by manually editing fire map just before calculate)
        # Wait, the step() does: 1. Drone moves, 2. Fire steps, 3. Calculate reward
        # Drone moving East (3) should put it at 11, 10.
        # Target was 12, 10. Distance before = 2.
        # Drone new pos = 11, 10. Distance to target (12, 10) = 1.
        # Progress = 2 - 1 = 1. Reward = 0.20
        obs, reward, term, trunc, info = self.env.step(3)
        self.assertAlmostEqual(info['distance_reward'], 0.20)
        
    def test_G_no_reach_farming(self):
        self.drone.x, self.drone.y = 10, 10
        self.env.world.fire_manager.ignite(10, 10)
        self.env.step(0)
        
        obs, reward, term, trunc, info = self.env.step(0)
        self.assertNotIn('reach_fire_reward', info)

if __name__ == '__main__':
    unittest.main()
