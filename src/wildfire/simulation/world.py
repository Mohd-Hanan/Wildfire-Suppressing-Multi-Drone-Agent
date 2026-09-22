from .terrain import Terrain
from .wind import Wind
from .fire import FireManager
from .drone import Drone, DroneType
import numpy as np

class World:
    def __init__(self, width: int, height: int, seed: int = None):
        self.width = width
        self.height = height
        self.terrain = Terrain(width, height, seed)
        
        speed = self.terrain.rng.uniform(0.1, 0.9)
        direction = self.terrain.rng.uniform(0.0, 360.0)
        self.wind = Wind(speed=speed, direction_degrees=direction)
        
        self.fire_manager = FireManager(width, height, self.terrain.rng)
        start_x = self.terrain.rng.integers(0, width)
        start_y = self.terrain.rng.integers(0, height)
        self.fire_manager.ignite(start_x, start_y)
        
        # Base Station location (Top-Left corner)
        self.base_x = 2
        self.base_y = 2
        
        # Heterogeneous Drone Fleet
        self.drones = [
            Drone(0, DroneType.WATER, self.base_x, self.base_y),
            Drone(1, DroneType.WATER, self.base_x, self.base_y),
            Drone(2, DroneType.WATER, self.base_x, self.base_y),
            Drone(3, DroneType.RETARDANT, self.base_x, self.base_y)
        ]

    def step(self):
        self.fire_manager.step(self.terrain, self.wind)
        
        # Dummy AI (Random Walk) for visual testing until RL is hooked up
        for drone in self.drones:
            if not drone.active:
                continue
                
            # Refill at Base Station
            if drone.x >= self.base_x and drone.x <= self.base_x + 1 and drone.y >= self.base_y and drone.y <= self.base_y + 1:
                drone.battery = drone.max_battery
                drone.payload = drone.max_payload
                
            # Random movement
            dx = self.terrain.rng.integers(-1, 2)
            dy = self.terrain.rng.integers(-1, 2)
            
            target_x = np.clip(drone.x + dx, 0, self.width - 1)
            target_y = np.clip(drone.y + dy, 0, self.height - 1)
            
            # Anti-Collision: Check if another drone is already at the target cell
            collision = False
            for other in self.drones:
                if other != drone and other.active and other.x == target_x and other.y == target_y:
                    collision = True
                    break
                    
            if not collision:
                drone.move(dx, dy, self.width, self.height)
            
            # Randomly test dropping payload
            if self.terrain.rng.random() < 0.05:
                if drone.drop():
                    if drone.type == DroneType.WATER:
                        self.terrain.moisture[drone.x, drone.y] = 1.0 # Max moisture (wetline)
                    else:
                        self.terrain.fuel[drone.x, drone.y] = 0.0 # Remove fuel (fireline)
        
        # Pathfinding back to base if battery is low (Dummy emergency return)
        for drone in self.drones:
            if drone.active and drone.battery < 30:
                dx = np.sign(self.base_x - drone.x)
                dy = np.sign(self.base_y - drone.y)
                
                target_x = np.clip(drone.x + dx, 0, self.width - 1)
                target_y = np.clip(drone.y + dy, 0, self.height - 1)
                
                collision = False
                for other in self.drones:
                    if other != drone and other.active and other.x == target_x and other.y == target_y:
                        collision = True
                        break
                        
                if not collision:
                    drone.move(dx, dy, self.width, self.height)

