from enum import Enum
import numpy as np

class DroneType(Enum):
    WATER = 1
    RETARDANT = 2

class Drone:
    def __init__(self, drone_id: int, drone_type: DroneType, start_x: int, start_y: int):
        self.id = drone_id
        self.type = drone_type
        
        self.x = start_x
        self.y = start_y
        
        # Heterogeneous stats
        self.max_battery = 150 if drone_type == DroneType.WATER else 100
        self.max_payload = 5 if drone_type == DroneType.WATER else 1
        
        # Current state
        self.battery = self.max_battery
        self.payload = self.max_payload
        self.active = True # Becomes False if it crashes (battery <= 0)
        
    def move(self, dx: int, dy: int, max_width: int, max_height: int):
        """Moves the drone and consumes battery."""
        if not self.active or self.battery <= 0:
            return
            
        self.x = int(np.clip(self.x + dx, 0, max_width - 1))
        self.y = int(np.clip(self.y + dy, 0, max_height - 1))
        
        # Retardant drones are heavier, so they burn battery faster
        cost = 1 if self.type == DroneType.WATER else 2
        self.battery -= cost
        
        if self.battery <= 0:
            self.active = False
            
    def drop(self):
        """Attempts to drop payload. Returns True if successful."""
        if not self.active or self.payload <= 0 or self.battery <= 0:
            return False
            
        self.payload -= 1
        self.battery -= 2 # Hovering to drop costs extra battery
        
        if self.battery <= 0:
            self.active = False
            
        return True

