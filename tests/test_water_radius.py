import unittest
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.simulation.fire import FireState

class TestWaterRadius(unittest.TestCase):
    def setUp(self):
        self.env = WildfireEnv("configs/environment.yaml")
        self.env.reset()
        # Ensure only 1 WATER drone
        self.env.world.drones = [d for d in self.env.world.drones if d.type.name == "WATER"][:1]
        self.drone = self.env.world.drones[0]
        
        # Clear fire map
        self.env.world.fire_manager.fire_map.fill(FireState.UNBURNED)
        
        # Move drone to center
        self.drone.x = 15
        self.drone.y = 15
        self.drone.payload = 20
        self.drone.battery = 250
        
        # Reset step count
        self.env.step_count = 0
        self.env.step(0)

    def test_1_exact_cell(self):
        self.env.world.fire_manager.fire_map[15, 15] = FireState.BURNING
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 1)
        self.assertEqual(self.env.world.fire_manager.fire_map[15, 15], FireState.BURNED)

    def test_2_one_cell_away(self):
        self.env.world.fire_manager.fire_map[16, 15] = FireState.BURNING
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 1)
        self.assertEqual(self.env.world.fire_manager.fire_map[16, 15], FireState.BURNED)

    def test_3_one_cell_away_other_directions(self):
        for dx, dy in [(-1, 0), (0, -1), (0, 1)]:
            with self.subTest(dx=dx, dy=dy):
                self.env.world.fire_manager.fire_map.fill(FireState.UNBURNED)
                self.drone.payload = 20
                self.env.world.fire_manager.fire_map[15+dx, 15+dy] = FireState.BURNING
                obs, reward, term, trunc, info = self.env.step(5)
                self.assertEqual(info['newly_suppressed_cells'], 1)
                self.assertEqual(self.env.world.fire_manager.fire_map[15+dx, 15+dy], FireState.BURNED)

    def test_4_two_cells_away(self):
        self.env.world.fire_manager.fire_map[17, 15] = FireState.BURNING
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 0)
        self.assertTrue(self.env.world.fire_manager.fire_map[17, 15] in [FireState.BURNING, FireState.SMOLDERING])

    def test_5_far_fire(self):
        self.env.world.fire_manager.fire_map[20, 20] = FireState.BURNING
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 0)
        self.assertTrue(self.env.world.fire_manager.fire_map[20, 20] in [FireState.BURNING, FireState.SMOLDERING])

    def test_6_multiple_active_cells_radius_1(self):
        self.env.world.fire_manager.fire_map[15, 15] = FireState.BURNING
        self.env.world.fire_manager.fire_map[16, 15] = FireState.BURNING
        self.env.world.fire_manager.fire_map[15, 16] = FireState.BURNING
        
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 1)
        
        # Check exactly one was suppressed
        fm = self.env.world.fire_manager.fire_map
        burned_count = np.sum(fm == FireState.BURNED)
        self.assertEqual(burned_count, 1)

    def test_7_no_active_fire(self):
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 0)

    def test_8_payload_mechanics(self):
        self.env.world.fire_manager.fire_map[16, 15] = FireState.BURNING
        old_payload = self.drone.payload
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(self.drone.payload, old_payload - 1)
        self.assertEqual(info['newly_suppressed_cells'], 1)

    def test_9_battery_mechanics(self):
        self.env.world.fire_manager.fire_map[16, 15] = FireState.BURNING
        old_battery = self.drone.battery
        obs, reward, term, trunc, info = self.env.step(5)
        # Expected drop = 1 (if WATER drop costs 1) or whatever action_cost is defined.
        # But wait, action drop cost is 0 in the drone config. It just costs 1 per step.
        # Just ensure it didn't change unexpectedly.
        self.assertEqual(self.drone.battery, old_battery - self.drone.drop_cost)

    def test_10_extinction_reward(self):
        self.env.world.fire_manager.fire_map[16, 15] = FireState.BURNING
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 1)
        self.assertEqual(info['suppression_reward'], 10.0)
        self.assertEqual(info['extinction_reward'], 50.0)

    def test_11_multiple_fire_extinction_reward(self):
        self.env.world.fire_manager.fire_map[16, 15] = FireState.BURNING
        self.env.world.fire_manager.fire_map[20, 20] = FireState.BURNING
        obs, reward, term, trunc, info = self.env.step(5)
        self.assertEqual(info['newly_suppressed_cells'], 1)
        self.assertEqual(info['suppression_reward'], 10.0)
        self.assertEqual(info['extinction_reward'], 0.0)

    def test_12_natural_extinction(self):
        self.env.world.fire_manager.fire_map[20, 20] = FireState.BURNING
        self.env.world.fire_manager.burn_timers[20, 20] = 39 # About to burn out
        # We perform STAY (0)
        obs, reward, term, trunc, info = self.env.step(0)
        self.assertEqual(info['newly_suppressed_cells'], 0)
        if np.sum((self.env.world.fire_manager.fire_map == FireState.BURNING) | (self.env.world.fire_manager.fire_map == FireState.IGNITING)) == 0:
            self.assertEqual(info['extinction_reward'], 0.0)

if __name__ == '__main__':
    unittest.main()
