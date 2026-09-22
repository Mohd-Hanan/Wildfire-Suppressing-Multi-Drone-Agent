import unittest
import numpy as np
from wildfire.simulation.world import World
from wildfire.environment.observation import ObservationBuilder
from wildfire.simulation.fire import FireState

class TestObservation(unittest.TestCase):
    def setUp(self):
        self.world = World(48, 48, seed=42)
        self.obs_builder = ObservationBuilder(window_size=11)
        self.drone = self.world.drones[0]

    def test_observation_generated_successfully(self):
        obs = self.obs_builder.get_observation(self.drone, self.world)
        self.assertIn('spatial', obs)
        self.assertIn('drone', obs)
        self.assertIn('wind', obs)

    def test_spatial_observation_shape(self):
        obs = self.obs_builder.get_observation(self.drone, self.world)
        spatial = obs['spatial']
        # 5 channels, 11x11 window
        self.assertEqual(spatial.shape, (5, 11, 11))
        
    def test_no_nan_or_inf(self):
        obs = self.obs_builder.get_observation(self.drone, self.world)
        self.assertFalse(np.isnan(obs['spatial']).any())
        self.assertFalse(np.isinf(obs['spatial']).any())
        self.assertFalse(np.isnan(obs['drone']).any())
        self.assertFalse(np.isinf(obs['drone']).any())
        self.assertFalse(np.isnan(obs['wind']).any())
        self.assertFalse(np.isinf(obs['wind']).any())

    def test_centered_drone_and_boundary_handling(self):
        # Center of map
        self.drone.x = 24
        self.drone.y = 24
        obs = self.obs_builder.get_observation(self.drone, self.world)
        self.assertEqual(obs['spatial'].shape, (5, 11, 11))
        self.assertEqual(obs['spatial'][3, 5, 5], 1.0) # Self drone at center
        
        # Edge of map (x=0)
        self.drone.x = 0
        self.drone.y = 24
        obs = self.obs_builder.get_observation(self.drone, self.world)
        self.assertEqual(obs['spatial'].shape, (5, 11, 11))
        self.assertEqual(obs['spatial'][3, 5, 5], 1.0) # Center should still be drone
        
        # Corner of map (x=47, y=47)
        self.drone.x = 47
        self.drone.y = 47
        obs = self.obs_builder.get_observation(self.drone, self.world)
        self.assertEqual(obs['spatial'].shape, (5, 11, 11))
        self.assertEqual(obs['spatial'][3, 5, 5], 1.0)
        
    def test_base_footprint_in_observation(self):
        # Base is at 2,2 to 3,3. Put drone at 2,2
        self.drone.x = 2
        self.drone.y = 2
        obs = self.obs_builder.get_observation(self.drone, self.world)
        # Center of window (5,5) corresponds to global (2,2)
        # Therefore global (3,3) should be at window (6,6)
        ch_base = obs['spatial'][4]
        self.assertEqual(ch_base[5, 5], 1.0)
        self.assertEqual(ch_base[6, 5], 1.0)
        self.assertEqual(ch_base[5, 6], 1.0)
        self.assertEqual(ch_base[6, 6], 1.0)

    def test_drone_scalars_normalized(self):
        self.drone.battery = 75
        self.drone.max_battery = 150
        self.drone.payload = 1
        self.drone.max_payload = 5
        
        obs = self.obs_builder.get_observation(self.drone, self.world)
        d_state = obs['drone']
        self.assertAlmostEqual(d_state[0], 0.5) # Battery
        self.assertAlmostEqual(d_state[1], 0.2) # Payload
        
    def test_wind_scalars(self):
        obs = self.obs_builder.get_observation(self.drone, self.world)
        wind = obs['wind']
        self.assertTrue(0.0 <= wind[0] <= 1.0) # Normalized speed
        self.assertTrue(-1.0 <= wind[1] <= 1.0) # Sin
        self.assertTrue(-1.0 <= wind[2] <= 1.0) # Cos
        
    def test_no_state_mutation(self):
        self.drone.x = 10
        self.drone.battery = 100
        original_map = self.world.fire_manager.fire_map.copy()
        
        self.obs_builder.get_observation(self.drone, self.world)
        
        self.assertEqual(self.drone.x, 10)
        self.assertEqual(self.drone.battery, 100)
        np.testing.assert_array_equal(self.world.fire_manager.fire_map, original_map)

    def test_distance_normalization(self):
        # Base is at 2, 2. Map is 48x48.
        # Farthest cell is 47, 47. Max dist = (47-2) + (47-2) = 45 + 45 = 90.
        
        # Test drone at base
        self.drone.x = 2
        self.drone.y = 2
        obs = self.obs_builder.get_observation(self.drone, self.world)
        dist_normalized = obs['drone'][3]
        self.assertEqual(dist_normalized, 0.0)
        
        # Test drone at farthest cell
        self.drone.x = 47
        self.drone.y = 47
        obs = self.obs_builder.get_observation(self.drone, self.world)
        dist_normalized = obs['drone'][3]
        self.assertEqual(dist_normalized, 1.0)
        
        # Test drone somewhere in middle
        self.drone.x = 12
        self.drone.y = 8
        obs = self.obs_builder.get_observation(self.drone, self.world)
        dist_normalized = obs['drone'][3]
        # Dist = |12-2| + |8-2| = 10 + 6 = 16.
        # 16 / 90 = 0.177777...
        self.assertAlmostEqual(dist_normalized, 16.0 / 90.0)

if __name__ == '__main__':
    unittest.main()
