import numpy as np

class Wind:
    def __init__(self, speed: float, direction_degrees: float):
        """
        speed: Wind intensity [0, 1]
        direction_degrees: 0 is North, 90 is East, 180 is South, 270 is West.
        """
        self.speed = np.clip(speed, 0.0, 1.0)
        self.direction = direction_degrees % 360.0
        
        # Convert to U (X) and V (Y) components. 
        # In grid coords: North = (0, -1), East = (1, 0)
        rad = np.radians(self.direction - 90)
        self.u = self.speed * np.cos(rad)
        self.v = self.speed * np.sin(rad)
        
    @property
    def vector(self):
        return np.array([self.u, self.v])

