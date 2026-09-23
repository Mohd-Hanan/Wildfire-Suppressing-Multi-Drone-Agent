import unittest
import numpy as np
from wildfire.simulation.fire import FireManager, FireState
from wildfire.simulation.world import World

class TestFireManager(unittest.TestCase):
    def test_configured_spread_rate_and_burning_duration(self):
        # Default config loads
        world = World(48, 48, seed=42)
        fm = world.fire_manager
        
        self.assertEqual(fm.base_spread_rate, 0.04)
        self.assertEqual(fm.base_burning_duration, 40)
        
    def test_fire_lifecycle_transitions(self):
        # We'll use a mocked world or just direct FireManager
        config = {'base_spread_rate': 0.04, 'base_burning_duration': 40}
        fm = FireManager(48, 48, np.random.default_rng(42), config)
        
        # Test IGNITING -> BURNING
        fm.fire_map[10, 10] = FireState.IGNITING
        
        class DummyTerrain:
            def __init__(self):
                self.fuel = np.zeros((48, 48))
                self.moisture = np.zeros((48, 48))
                self.elevation = np.zeros((48, 48))
                
        class DummyWind:
            def __init__(self):
                self.u = 0.0
                self.v = 0.0
                
        terrain = DummyTerrain()
        wind = DummyWind()
        
        fm.step(terrain, wind)
        
        # Now should be burning
        self.assertEqual(fm.fire_map[10, 10], FireState.BURNING)
        self.assertEqual(fm.burn_timers[10, 10], 39) # Should use base_burning_duration
        
        # Advance 40 steps
        for _ in range(40):
            fm.step(terrain, wind)
            
        # Now should be SMOLDERING
        self.assertEqual(fm.fire_map[10, 10], FireState.SMOLDERING)
        self.assertEqual(fm.burn_timers[10, 10], 28) # Hardcoded smolder time
        
        # Advance 30 steps
        for _ in range(30):
            fm.step(terrain, wind)
            
        # Now should be BURNED
        self.assertEqual(fm.fire_map[10, 10], FireState.BURNED)

if __name__ == '__main__':
    unittest.main()
