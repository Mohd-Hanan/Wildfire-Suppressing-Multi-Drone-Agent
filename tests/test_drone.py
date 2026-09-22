import unittest
from wildfire.simulation.drone import Drone, DroneType

class TestDrone(unittest.TestCase):
    def setUp(self):
        self.drone_config = {
            'water': {
                'max_battery': 150,
                'max_payload': 5,
                'move_cost': 1,
                'drop_cost': 2,
                'drop_payload_cost': 1
            },
            'retardant': {
                'max_battery': 100,
                'max_payload': 1,
                'move_cost': 2,
                'drop_cost': 2,
                'drop_payload_cost': 1
            }
        }

    def test_water_drone_initialization(self):
        drone = Drone(0, DroneType.WATER, 0, 0, self.drone_config['water'])
        self.assertEqual(drone.battery, 150)
        self.assertEqual(drone.payload, 5)

    def test_retardant_drone_initialization(self):
        drone = Drone(1, DroneType.RETARDANT, 0, 0, self.drone_config['retardant'])
        self.assertEqual(drone.battery, 100)
        self.assertEqual(drone.payload, 1)

    def test_water_drone_movement(self):
        drone = Drone(0, DroneType.WATER, 0, 0, self.drone_config['water'])
        drone.move(1, 0, 10, 10)
        self.assertEqual(drone.battery, 149)

    def test_retardant_drone_movement(self):
        drone = Drone(1, DroneType.RETARDANT, 0, 0, self.drone_config['retardant'])
        drone.move(1, 0, 10, 10)
        self.assertEqual(drone.battery, 98)

    def test_water_drone_deployment(self):
        drone = Drone(0, DroneType.WATER, 0, 0, self.drone_config['water'])
        success = drone.drop()
        self.assertTrue(success)
        self.assertEqual(drone.payload, 4)
        self.assertEqual(drone.battery, 148)

    def test_retardant_drone_deployment(self):
        drone = Drone(1, DroneType.RETARDANT, 0, 0, self.drone_config['retardant'])
        success = drone.drop()
        self.assertTrue(success)
        self.assertEqual(drone.payload, 0)
        self.assertEqual(drone.battery, 98)

    def test_deployment_with_zero_payload_fails(self):
        drone = Drone(1, DroneType.RETARDANT, 0, 0, self.drone_config['retardant'])
        drone.payload = 0
        initial_battery = drone.battery
        success = drone.drop()
        self.assertFalse(success)
        self.assertEqual(drone.battery, initial_battery)

if __name__ == '__main__':
    unittest.main()

    def test_drone_inactive_at_zero_battery(self):
        drone = Drone(0, DroneType.WATER, 10, 10, self.drone_config['water'])
        drone.battery = 1
        drone.move(1, 0, 20, 20) # Move costs 1
        self.assertEqual(drone.battery, 0)
        self.assertFalse(drone.active)
        
    def test_inactive_drone_cannot_move(self):
        drone = Drone(0, DroneType.WATER, 10, 10, self.drone_config['water'])
        drone.battery = 0
        drone.move(1, 0, 20, 20)
        self.assertEqual(drone.x, 10)
        self.assertEqual(drone.y, 10)
        
    def test_inactive_drone_cannot_deploy(self):
        drone = Drone(0, DroneType.WATER, 10, 10, self.drone_config['water'])
        drone.battery = 0
        success = drone.drop()
        self.assertFalse(success)
        self.assertEqual(drone.payload, 5) # Payload unchanged
        
    def test_zero_battery_does_not_teleport(self):
        drone = Drone(0, DroneType.WATER, 10, 10, self.drone_config['water'])
        drone.battery = 1
        drone.move(1, 0, 20, 20) # Causes battery to hit 0
        self.assertEqual(drone.x, 11) # Moved once and died
        self.assertEqual(drone.y, 10)
        self.assertFalse(drone.active)

from wildfire.simulation.world import World

class TestWorld(unittest.TestCase):
    def test_base_refill_water(self):
        world = World(48, 48, seed=42)
        drone = world.drones[0] # Water drone
        self.assertEqual(drone.type, DroneType.WATER)
        
        # Partially deplete
        drone.battery = 50
        drone.payload = 2
        
        # Verify it's at base (base_x=2, base_y=2)
        drone.x = 2
        drone.y = 2
        
        self.assertTrue(world.is_at_base(drone))
        
        if world.is_at_base(drone):
            drone.battery = drone.max_battery
            drone.payload = drone.max_payload
            
        self.assertEqual(drone.battery, 150)
        self.assertEqual(drone.payload, 5)

    def test_base_refill_retardant(self):
        world = World(48, 48, seed=42)
        drone = world.drones[3] # Retardant drone
        self.assertEqual(drone.type, DroneType.RETARDANT)
        
        drone.battery = 10
        drone.payload = 0
        drone.x = 3
        drone.y = 2 # (3,2) is within 2x2 footprint of (2,2)
        
        self.assertTrue(world.is_at_base(drone))
        
        if world.is_at_base(drone):
            drone.battery = drone.max_battery
            drone.payload = drone.max_payload
            
        self.assertEqual(drone.battery, 100)
        self.assertEqual(drone.payload, 1)

    def test_distance_and_minimum_battery(self):
        world = World(48, 48, seed=42)
        # Base is at (2, 2)
        
        # Test water drone at (12, 8)
        water_drone = world.drones[0]
        water_drone.x = 12
        water_drone.y = 8
        
        dist = world.distance_to_base(water_drone)
        self.assertEqual(dist, 16)
        self.assertEqual(world.minimum_return_battery(water_drone), 16)
        
        # Test retardant drone at (12, 8)
        ret_drone = world.drones[3]
        ret_drone.x = 12
        ret_drone.y = 8
        
        self.assertEqual(world.distance_to_base(ret_drone), 16)
        self.assertEqual(world.minimum_return_battery(ret_drone), 32)
        
    def test_safe_thresholds_and_margin(self):
        world = World(48, 48, seed=42)
        # Reserve is 5
        water_drone = world.drones[0]
        water_drone.x = 12
        water_drone.y = 8
        water_drone.battery = 25
        
        # Safe return battery for water is 16 + 5 = 21
        # Margin is 25 - 21 = 4
        self.assertEqual(world.battery_margin(water_drone), 4)
        self.assertTrue(world.can_safely_return_to_base(water_drone))
        
        ret_drone = world.drones[3]
        ret_drone.x = 12
        ret_drone.y = 8
        ret_drone.battery = 30
        
        # Safe return battery for retardant is 32 + 5 = 37
        # Margin is 30 - 37 = -7
        self.assertEqual(world.battery_margin(ret_drone), -7)
        self.assertFalse(world.can_safely_return_to_base(ret_drone))
        
    def test_exact_threshold(self):
        world = World(48, 48, seed=42)
        water_drone = world.drones[0]
        water_drone.x = 12
        water_drone.y = 8
        
        # Threshold is 21
        water_drone.battery = 21
        self.assertEqual(world.battery_margin(water_drone), 0)
        self.assertTrue(world.can_safely_return_to_base(water_drone))
        
        # Below threshold
        water_drone.battery = 20
        self.assertEqual(world.battery_margin(water_drone), -1)
        self.assertFalse(world.can_safely_return_to_base(water_drone))
        
    def test_drone_at_base_distance(self):
        world = World(48, 48, seed=42)
        drone = world.drones[0]
        drone.x = 2
        drone.y = 2
        self.assertEqual(world.distance_to_base(drone), 0)
        self.assertEqual(world.minimum_return_battery(drone), 0)
        
    def test_calculations_do_not_modify_state(self):
        world = World(48, 48, seed=42)
        drone = world.drones[0]
        drone.x = 12
        drone.y = 8
        drone.battery = 50
        drone.payload = 3
        
        world.distance_to_base(drone)
        world.minimum_return_battery(drone)
        world.battery_margin(drone)
        world.can_safely_return_to_base(drone)
        
        self.assertEqual(drone.x, 12)
        self.assertEqual(drone.y, 8)
        self.assertEqual(drone.battery, 50)
        self.assertEqual(drone.payload, 3)
