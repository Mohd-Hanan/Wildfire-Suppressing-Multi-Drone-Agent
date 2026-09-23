import unittest
from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone, DroneType
from wildfire.environment.reward import RewardCalculator

class DummyConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self['return_to_base'] = {'battery_reserve': 5}
        self['reward'] = {
            'new_burned_cell_penalty': 5.0,
            'effective_suppression_reward': 2.0,
            'step_penalty': 0.01,
            'drone_crash_penalty': 10.0,
            'fire_extinguished_reward': 100.0,
            'boundary_hit_penalty': 0.1
        }
        self['map'] = {'width': 48, 'height': 48}

class TestBatterySafety(unittest.TestCase):
    def setUp(self):
        self.config = DummyConfig()
        self.world = World(48, 48, seed=42)
        # Manually patch config for test purposes
        self.world.config = self.config
        self.world.base_x = 2
        self.world.base_y = 2
        self.drone_config = {
            'max_battery': 100,
            'max_payload': 10,
            'move_cost': 2,
            'drop_cost': 5,
            'drop_payload_cost': 1
        }
        self.drone = Drone(0, DroneType.WATER, 2, 2, self.drone_config)
        self.world.drones = [self.drone]
        self.reward_calc = RewardCalculator(self.config)
        self.reward_calc.calculate(self.world, self.world.drones) # init step 0

    def test_drone_on_base(self):
        self.drone.x = 2
        self.drone.y = 2
        self.assertEqual(self.world.distance_to_base(self.drone), 0)
        self.assertEqual(self.world.minimum_return_battery(self.drone), 0)
        self.assertEqual(self.world.required_battery(self.drone), 5)
        
        self.drone.x = 3
        self.drone.y = 3
        self.assertEqual(self.world.distance_to_base(self.drone), 0)

    def test_drone_one_cell_from_base(self):
        self.drone.x = 4
        self.drone.y = 3
        self.assertEqual(self.world.distance_to_base(self.drone), 1)
        self.assertEqual(self.world.required_battery(self.drone), 1 * 2 + 5)

    def test_drone_10_cells_from_base(self):
        self.drone.x = 13
        self.drone.y = 2
        self.assertEqual(self.world.distance_to_base(self.drone), 10)
        self.assertEqual(self.world.required_battery(self.drone), 10 * 2 + 5)

    def test_drone_at_corner(self):
        self.drone.x = 0
        self.drone.y = 0
        # base cells: (2,2), (3,2), (2,3), (3,3)
        # nearest is (2,2) -> distance is abs(0-2) + abs(0-2) = 4
        self.assertEqual(self.world.distance_to_base(self.drone), 4)

    def test_battery_safely_above_threshold(self):
        self.drone.x = 4
        self.drone.y = 3
        # req = 1*2 + 5 = 7
        self.drone.battery = 10
        margin = self.world.battery_margin(self.drone)
        self.assertEqual(margin, 3)
        
        _, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['battery_safety_penalty'], 0.0)

    def test_battery_exactly_at_threshold(self):
        self.drone.x = 4
        self.drone.y = 3
        self.drone.battery = 7
        margin = self.world.battery_margin(self.drone)
        self.assertEqual(margin, 0)
        
        _, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['battery_safety_penalty'], 0.0)

    def test_battery_below_threshold(self):
        self.drone.x = 4
        self.drone.y = 3
        # req = 7
        self.drone.battery = 5
        margin = self.world.battery_margin(self.drone)
        self.assertEqual(margin, -2)
        
        _, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['battery_safety_penalty'], -1.0) # -0.5 * 2

    def test_battery_severely_below_threshold(self):
        self.drone.x = 4
        self.drone.y = 3
        self.drone.battery = 1
        margin = self.world.battery_margin(self.drone)
        self.assertEqual(margin, -6)
        
        _, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['battery_safety_penalty'], -3.0) # -0.5 * 6

    def test_battery_depletion_crash_penalty(self):
        # Drone initially active (battery > 0)
        self.drone.battery = 2
        _, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['crash_penalty'], 0.0)
        
        # Drone battery depletes
        self.drone.battery = 0
        _, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['crash_penalty'], -10.0)
        
        # Next step, drone still inactive
        _, info = self.reward_calc.calculate(self.world, self.world.drones)
        self.assertEqual(info['crash_penalty'], 0.0)

    def test_returning_to_base_decreases_distance(self):
        self.drone.x = 10
        self.drone.y = 10
        dist1 = self.world.distance_to_base(self.drone)
        self.drone.x -= 1
        self.drone.y -= 1
        dist2 = self.world.distance_to_base(self.drone)
        self.assertTrue(dist2 < dist1)
        self.assertEqual(dist1 - dist2, 2)

if __name__ == '__main__':
    unittest.main()
