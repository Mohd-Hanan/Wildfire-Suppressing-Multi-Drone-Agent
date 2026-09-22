from .terrain import Terrain
from .wind import Wind

class World:
    def __init__(self, width: int, height: int, seed: int = None):
        self.width = width
        self.height = height
        self.terrain = Terrain(width, height, seed)
        
        # Default wind: 60% intensity blowing South-East (135 degrees)
        self.wind = Wind(speed=0.6, direction_degrees=135.0)

