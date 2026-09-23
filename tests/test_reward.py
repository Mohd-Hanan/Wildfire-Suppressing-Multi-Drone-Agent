import unittest
import yaml
from wildfire.simulation.world import World
from wildfire.environment.reward import RewardCalculator
from wildfire.simulation.fire import FireState

class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        with open('configs/environment.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.reward_calc = RewardCalculator(self.config)
        self.world = World(48, 48, seed=42)
        
        # Clear fire for controlled testing
        self.world.fire_manager.fire_map[:] = FireState.UNBURNED

    def test_reset_initializes_state(self):
        self.assertEqual(self.reward_calc.prev_affected_cells, -1)
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(r, 0.0)
        self.assertEqual(self.reward_calc.prev_affected_cells, 0)
        
    def test_first_calculation_does_not_penalize_preexisting_fire(self):
        self.world.fire_manager.fire_map[10:15, 10:15] = FireState.BURNING
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(r, 0.0)
        self.assertEqual(info['damage_penalty'], 0.0)
        self.assertEqual(self.reward_calc.prev_affected_cells, 25)

    def test_new_burned_cell_damage_penalty(self):
        self.reward_calc.calculate(self.world, self.world.drones) # step 0
        
        # One new fire cell
        self.world.fire_manager.fire_map[1, 1] = FireState.IGNITING
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        
        self.assertEqual(info['new_burned_cells'], 1)
        self.assertEqual(info['damage_penalty'], -0.1)
        self.assertEqual(info['step_penalty'], -0.01)
        self.assertAlmostEqual(r, -0.11)

    def test_multiple_new_burned_cells(self):
        self.reward_calc.calculate(self.world, self.world.drones) # step 0
        
        # Five new fire cells
        self.world.fire_manager.fire_map[0:5, 0] = FireState.BURNING
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        
        self.assertEqual(info['new_burned_cells'], 5)
        self.assertEqual(info['damage_penalty'], -0.5)
        self.assertAlmostEqual(r, -0.51)

    def test_no_new_burned_cells(self):
        self.world.fire_manager.fire_map[0, 0] = FireState.BURNING
        self.reward_calc.calculate(self.world, self.world.drones) # step 0
        
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['new_burned_cells'], 0)
        self.assertEqual(info['damage_penalty'], 0.0)
        self.assertAlmostEqual(r, -0.01) # only step penalty

    def test_crash_penalty(self):
        self.reward_calc.calculate(self.world, self.world.drones) # step 0
        
        # Drone 0 crashes
        drone = self.world.drones[0]
        drone.battery = 0
        
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['crash_penalty'], -10.0)
        self.assertAlmostEqual(r, -10.01)
        
        # Subsequent step should not repeatedly penalize
        r2, info2 = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info2['crash_penalty'], 0.0)
        self.assertAlmostEqual(r2, -0.01)

    def test_extinction_reward(self):
        self.world.fire_manager.fire_map[0, 0] = FireState.BURNING
        self.reward_calc.calculate(self.world, self.world.drones) # step 0
        
        # Extinguish fire
        self.world.fire_manager.fire_map[0, 0] = FireState.UNBURNED
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        
        self.assertEqual(info['extinction_reward'], 50.0)
        # Note: new_burned_cells is max(0, -1) = 0
        self.assertAlmostEqual(r, 49.99)
        
        # Subsequent step should not repeatedly reward
        r2, info2 = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info2['extinction_reward'], 0.0)

    def test_hacking_prevention(self):
        self.reward_calc.calculate(self.world, self.world.drones) # step 0
        
        # Moving without fire changing
        drone = self.world.drones[0]
        drone.move(1, 1, 48, 48)
        r, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertAlmostEqual(r, -0.01) # No reward for moving
        
        # Base refill
        drone.battery = drone.max_battery
        r2, info2 = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertAlmostEqual(r2, -0.01) # No reward for refill
        
        # Deploying with no fire changing
        drone.drop()
        r3, info3 = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertAlmostEqual(r3, -0.01) # No reward for dropping
        self.assertEqual(info3['suppression_reward'], 0.0)

if __name__ == '__main__':
    unittest.main()
