from .terrain import Terrain
from .wind import Wind
from .fire import FireManager

class World:
    def __init__(self, width: int, height: int, seed: int = None):
        self.width = width
        self.height = height
        self.terrain = Terrain(width, height, seed)
        
        # Randomize wind for every new environment episode
        # Speed: 0.1 to 0.9, Direction: 0 to 360 degrees
        speed = self.terrain.rng.uniform(0.1, 0.9)
        direction = self.terrain.rng.uniform(0.0, 360.0)
        
        self.wind = Wind(speed=speed, direction_degrees=direction)
        
        # Initialize Fire
        self.fire_manager = FireManager(width, height, self.terrain.rng)
        
        # Randomize initial fire location so the agent learns to search!
        start_x = self.terrain.rng.integers(0, width)
        start_y = self.terrain.rng.integers(0, height)
        self.fire_manager.ignite(start_x, start_y)

    def step(self):
        self.fire_manager.step(self.terrain, self.wind)

