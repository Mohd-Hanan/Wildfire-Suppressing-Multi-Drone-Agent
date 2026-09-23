import unittest
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.simulation.fire import FireState

class TestSuppressionObservation(unittest.TestCase):

    def test_observation_nearest_fire(self):
        env = WildfireEnv()
        obs, _ = env.reset(seed=42)
        drone = env.world.drones[env.controlled_drone_idx]
        
        # Manually place a fire at (10, 10)
        drone.x, drone.y = 5, 5
        env.world.fire_manager.fire_map[:] = FireState.UNBURNED
        env.world.fire_manager.ignite(10, 10)
        
        # Get observation directly
        from wildfire.environment.observation import ObservationBuilder
        obs_builder = ObservationBuilder()
        obs_dict = obs_builder.get_observation(drone, env.world)
        
        # The drone state should be length 9
        self.assertEqual(len(obs_dict["drone"]), 9)
        
        # fire_dx, fire_dy, fire_distance
        fire_dx = obs_dict["drone"][6]
        fire_dy = obs_dict["drone"][7]
        fire_dist = obs_dict["drone"][8]
        
        self.assertTrue(np.isclose(fire_dx, 5 / 48))
        self.assertTrue(np.isclose(fire_dy, 5 / 48))
        expected_dist = np.sqrt(5**2 + 5**2) / np.sqrt(48**2 + 48**2)
        self.assertTrue(np.isclose(fire_dist, expected_dist))

    def test_observation_no_fire(self):
        env = WildfireEnv()
        env.reset(seed=42)
        drone = env.world.drones[env.controlled_drone_idx]
        
        env.world.fire_manager.fire_map[:] = FireState.UNBURNED
        
        from wildfire.environment.observation import ObservationBuilder
        obs_builder = ObservationBuilder()
        obs_dict = obs_builder.get_observation(drone, env.world)
        
        self.assertTrue(np.isclose(obs_dict["drone"][6], 0.0))
        self.assertTrue(np.isclose(obs_dict["drone"][7], 0.0))
        self.assertTrue(np.isclose(obs_dict["drone"][8], 0.0))

    def test_water_on_active_fire(self):
        env = WildfireEnv()
        env.reset(seed=42)
        drone = env.world.drones[env.controlled_drone_idx]
        
        drone.x, drone.y = 10, 10
        drone.battery = 100
        drone.payload = 5
        
        env.world.fire_manager.fire_map[:] = FireState.UNBURNED
        env.world.fire_manager.ignite(10, 10)
        
        # Advance one step (action 0) so the reward calculator initializes
        env.step(0)
        
        # Execute WATER
        obs, reward, term, trunc, info = env.step(5)
        
        self.assertEqual(env.world.fire_manager.fire_map[10, 10], FireState.BURNED)
        self.assertEqual(env.world.fire_manager.burn_timers[10, 10], 0)
        self.assertEqual(env.world.terrain.moisture[10, 10], 1.0)
        
        self.assertEqual(info['newly_suppressed_cells'], 1)
        self.assertEqual(info['suppression_reward'], 2.0)
        self.assertEqual(info['extinction_reward'], 50.0)
        self.assertTrue(np.isclose(reward, 2.0 + 50.0 - 0.01))

    def test_water_on_non_fire(self):
        env = WildfireEnv()
        env.reset(seed=42)
        drone = env.world.drones[env.controlled_drone_idx]
        
        drone.x, drone.y = 10, 10
        drone.battery = 100
        drone.payload = 5
        
        env.world.fire_manager.fire_map[:] = FireState.UNBURNED
        env.world.fire_manager.ignite(20, 20) # Keep fire alive elsewhere
        
        # Initialize reward
        env.step(0)
        
        # Execute WATER
        obs, reward, term, trunc, info = env.step(5)
        
        self.assertEqual(env.world.fire_manager.fire_map[10, 10], FireState.UNBURNED)
        self.assertEqual(info['newly_suppressed_cells'], 0)
        self.assertEqual(info['suppression_reward'], 0.0)
        self.assertEqual(info['extinction_reward'], 0.0)

    def test_fire_spreading_penalty(self):
        env = WildfireEnv()
        env.reset(seed=42)
        
        # Initialize reward
        env.step(0)
        
        # Manually increment affected cells
        env.reward_calculator.prev_affected_cells -= 5
        
        drone = env.world.drones[env.controlled_drone_idx]
        drone.battery = 100
        
        obs, reward, term, trunc, info = env.step(0)
        
        self.assertEqual(info['new_burned_cells'], 5)
        self.assertEqual(info['damage_penalty'], -0.5)
        self.assertTrue(np.isclose(reward, -0.5 - 0.01))

if __name__ == '__main__':
    unittest.main()
