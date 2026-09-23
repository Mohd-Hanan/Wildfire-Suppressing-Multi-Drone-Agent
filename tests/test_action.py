import unittest
from wildfire.simulation.world import World
from wildfire.simulation.drone import DroneType
from wildfire.environment.action import ActionExecutor

class TestActionSystem(unittest.TestCase):
    def setUp(self):
        self.world = World(48, 48, seed=42)
        self.water_drone = self.world.drones[0]
        self.retardant_drone = self.world.drones[3]
        self.executor = ActionExecutor()
        
        # Move them to the center to avoid boundary issues during basic movement
        self.water_drone.x = 24
        self.water_drone.y = 24
        self.water_drone.battery = 150
        self.water_drone.payload = 5
        
        self.retardant_drone.x = 24
        self.retardant_drone.y = 24
        self.retardant_drone.battery = 100
        self.retardant_drone.payload = 1

    def test_stay_action(self):
        self.executor.execute(self.water_drone, self.world, 0)[0]
        self.assertEqual(self.water_drone.x, 24)
        self.assertEqual(self.water_drone.y, 24)
        self.assertEqual(self.water_drone.battery, 150)
        self.assertEqual(self.water_drone.payload, 5)

    def test_north_action(self):
        self.executor.execute(self.water_drone, self.world, 1)[0]
        self.assertEqual(self.water_drone.x, 24)
        self.assertEqual(self.water_drone.y, 23)
        self.assertEqual(self.water_drone.battery, 149)

    def test_south_action(self):
        self.executor.execute(self.water_drone, self.world, 2)[0]
        self.assertEqual(self.water_drone.x, 24)
        self.assertEqual(self.water_drone.y, 25)
        self.assertEqual(self.water_drone.battery, 149)

    def test_east_action(self):
        self.executor.execute(self.water_drone, self.world, 3)[0]
        self.assertEqual(self.water_drone.x, 25)
        self.assertEqual(self.water_drone.y, 24)
        self.assertEqual(self.water_drone.battery, 149)

    def test_west_action(self):
        self.executor.execute(self.water_drone, self.world, 4)[0]
        self.assertEqual(self.water_drone.x, 23)
        self.assertEqual(self.water_drone.y, 24)
        self.assertEqual(self.water_drone.battery, 149)

    def test_water_deployment(self):
        self.executor.execute(self.water_drone, self.world, 5)[0]
        self.assertEqual(self.water_drone.battery, 148)
        self.assertEqual(self.water_drone.payload, 4)
        self.assertEqual(self.world.terrain.moisture[24, 24], 1.0)

    def test_retardant_deployment(self):
        self.executor.execute(self.retardant_drone, self.world, 6)[0]
        self.assertEqual(self.retardant_drone.battery, 98)
        self.assertEqual(self.retardant_drone.payload, 0)
        self.assertEqual(self.world.terrain.fuel[24, 24], 0.0)

    def test_invalid_deployment_type_water(self):
        # Water drone trying to drop retardant
        self.executor.execute(self.water_drone, self.world, 6)[0]
        self.assertEqual(self.water_drone.battery, 150)
        self.assertEqual(self.water_drone.payload, 5)

    def test_invalid_deployment_type_retardant(self):
        # Retardant drone trying to drop water
        self.executor.execute(self.retardant_drone, self.world, 5)[0]
        self.assertEqual(self.retardant_drone.battery, 100)
        self.assertEqual(self.retardant_drone.payload, 1)

    def test_drop_zero_payload(self):
        self.water_drone.payload = 0
        self.executor.execute(self.water_drone, self.world, 5)[0]
        self.assertEqual(self.water_drone.battery, 150) # Unchanged
        
        self.retardant_drone.payload = 0
        self.executor.execute(self.retardant_drone, self.world, 6)[0]
        self.assertEqual(self.retardant_drone.battery, 100) # Unchanged

    def test_drop_insufficient_battery(self):
        self.water_drone.battery = 1
        self.executor.execute(self.water_drone, self.world, 5)[0]
        self.assertEqual(self.water_drone.battery, 1)
        self.assertEqual(self.water_drone.payload, 5)

    def test_inactive_drone_actions(self):
        self.water_drone.battery = 0
        self.executor.execute(self.water_drone, self.world, 1)[0] # Move
        self.assertEqual(self.water_drone.x, 24)
        self.assertEqual(self.water_drone.y, 24)
        
        self.executor.execute(self.water_drone, self.world, 5)[0] # Drop
        self.assertEqual(self.water_drone.payload, 5)
        
        self.assertFalse(self.water_drone.active)

    def test_boundaries(self):
        # North boundary
        self.water_drone.x = 24
        self.water_drone.y = 0
        self.executor.execute(self.water_drone, self.world, 1)[0] # North
        self.assertEqual(self.water_drone.y, 0)
        
        # South boundary
        self.water_drone.y = 47
        self.executor.execute(self.water_drone, self.world, 2)[0] # South
        self.assertEqual(self.water_drone.y, 47)
        
        # West boundary
        self.water_drone.x = 0
        self.water_drone.y = 24
        self.executor.execute(self.water_drone, self.world, 4)[0] # West
        self.assertEqual(self.water_drone.x, 0)
        
        # East boundary
        self.water_drone.x = 47
        self.executor.execute(self.water_drone, self.world, 3)[0] # East
        self.assertEqual(self.water_drone.x, 47)
        
if __name__ == '__main__':
    unittest.main()
