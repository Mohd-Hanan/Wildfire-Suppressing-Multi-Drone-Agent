import unittest
import yaml
from wildfire.simulation.world import World
from wildfire.environment.termination import TerminationChecker
from wildfire.simulation.fire import FireState

class TestTerminationSystem(unittest.TestCase):
    def setUp(self):
        with open('configs/environment.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.term_checker = TerminationChecker(self.config)
        self.world = World(48, 48, seed=42)
        
        # Ensure fire starts for controlled testing
        self.world.fire_manager.fire_map[:] = FireState.UNBURNED
        self.world.fire_manager.fire_map[0, 0] = FireState.BURNING

    def test_active_fire_below_max_steps_continues(self):
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info['termination_reason'])

    def test_fire_completely_extinguished(self):
        # Extinguish all fire
        self.world.fire_manager.fire_map[:] = FireState.UNBURNED
        
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'fire_extinguished')

    def test_maximum_step_limit_reached(self):
        # Fire still burning, but max steps reached
        max_steps = self.term_checker.max_steps
        terminated, truncated, info = self.term_checker.check(self.world, step_count=max_steps)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['termination_reason'], 'max_steps')

    def test_extinction_takes_precedence_over_max_steps(self):
        # Both conditions met simultaneously
        self.world.fire_manager.fire_map[:] = FireState.UNBURNED
        max_steps = self.term_checker.max_steps
        
        terminated, truncated, info = self.term_checker.check(self.world, step_count=max_steps)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'fire_extinguished')

    def test_one_drone_crashes_episode_continues(self):
        self.world.drones[0].battery = 0
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_all_drones_inactive_episode_continues(self):
        for d in self.world.drones:
            d.battery = 0
            
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertFalse(terminated) # Version 1 rule: all drones inactive does NOT automatically terminate
        self.assertFalse(truncated)

    def test_zero_payload_low_battery_episode_continues(self):
        d = self.world.drones[0]
        d.payload = 0
        d.battery = 1
        
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_reset_clears_state(self):
        self.term_checker.reset()
        terminated, truncated, info = self.term_checker.check(self.world, step_count=0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_termination_check_does_not_modify_world(self):
        original_battery = self.world.drones[0].battery
        original_fire = self.world.fire_manager.fire_map.copy()
        
        self.term_checker.check(self.world, step_count=10)
        
        self.assertEqual(self.world.drones[0].battery, original_battery)
        import numpy as np
        np.testing.assert_array_equal(self.world.fire_manager.fire_map, original_fire)

    def test_just_below_max_steps(self):
        max_steps = self.term_checker.max_steps
        terminated, truncated, info = self.term_checker.check(self.world, step_count=max_steps - 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info['termination_reason'])

    def test_exact_max_step_boundary(self):
        max_steps = self.term_checker.max_steps
        terminated, truncated, info = self.term_checker.check(self.world, step_count=max_steps)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['termination_reason'], 'max_steps')

    def test_multiple_drones_inactive_episode_continues(self):
        # 2 inactive, 2 active
        self.world.drones[0].battery = 0
        self.world.drones[1].battery = 0
        self.world.drones[2].battery = 100
        self.world.drones[3].battery = 100
        
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info['termination_reason'])

    def test_battery_zero_away_from_base(self):
        # Base is around 2,2. Move drone to 10,10 and set battery 0
        drone = self.world.drones[0]
        drone.x = 10
        drone.y = 10
        drone.battery = 0
        
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info['termination_reason'])

    def test_termination_reason_fire_extinction(self):
        self.world.fire_manager.fire_map[:] = FireState.UNBURNED
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'fire_extinguished')

    def test_termination_reason_max_steps(self):
        max_steps = self.term_checker.max_steps
        terminated, truncated, info = self.term_checker.check(self.world, step_count=max_steps + 5)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['termination_reason'], 'max_steps')

    def test_burned_only_fire_state_terminates(self):
        # Only BURNED cells remain, no active fire
        self.world.fire_manager.fire_map[:] = FireState.BURNED
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'fire_extinguished')

    def test_unburned_only_fire_state_terminates(self):
        # Only UNBURNED cells remain, no active fire
        self.world.fire_manager.fire_map[:] = FireState.UNBURNED
        terminated, truncated, info = self.term_checker.check(self.world, step_count=10)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'fire_extinguished')

if __name__ == '__main__':
    unittest.main()
