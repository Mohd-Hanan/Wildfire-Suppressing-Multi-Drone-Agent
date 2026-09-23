from enum import Enum
import numpy as np

class DroneType(Enum):
    WATER = 1
    RETARDANT = 2

class Drone:
    def __init__(self, drone_id: int, drone_type: DroneType, start_x: int, start_y: int, config: dict):
        self.id = drone_id
        self.type = drone_type
        
        self.x = start_x
        self.y = start_y
        
        self.config = config
        
        # Heterogeneous stats
        self.max_battery = self.config['max_battery']
        self.max_payload = self.config['max_payload']
        self.move_cost = self.config['move_cost']
        self.drop_cost = self.config['drop_cost']
        self.drop_payload_cost = self.config['drop_payload_cost']
        
        # Current state
        self.battery = self.max_battery
        self.payload = self.max_payload
        
    @property
    def active(self):
        """Drone is active as long as it has battery."""
        return self.battery > 0
        
    def move(self, dx: int, dy: int, max_width: int, max_height: int) -> bool:
        """Moves the drone and consumes battery. Returns True if movement was blocked by boundary."""
        if not self.active:
            return False
            
        target_x = self.x + dx
        target_y = self.y + dy
        
        hit_boundary = (target_x < 0 or target_x >= max_width or 
                        target_y < 0 or target_y >= max_height)
            
        self.x = int(np.clip(target_x, 0, max_width - 1))
        self.y = int(np.clip(target_y, 0, max_height - 1))
        
        self.battery -= self.move_cost
        
        # Prevent battery from becoming negative
        if self.battery < 0:
            self.battery = 0
            
        return hit_boundary
            
    def drop(self):
        """Attempts to drop payload. Returns True if successful."""
        if not self.active:
            return False
            
        if self.payload < self.drop_payload_cost:
            return False
            
        if self.battery < self.drop_cost:
            return False
            
        self.payload -= self.drop_payload_cost
        self.battery -= self.drop_cost
            
        return True
