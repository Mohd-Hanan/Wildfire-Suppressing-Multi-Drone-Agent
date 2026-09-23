from .terrain import Terrain
from .wind import Wind
from .fire import FireManager
from .drone import Drone, DroneType
import numpy as np
import yaml

class World:
    def __init__(self, width: int, height: int, seed: int = None, config_path: str = "configs/environment.yaml"):
        self.width = width
        self.height = height
        self.terrain = Terrain(width, height, seed)
        
        speed = self.terrain.rng.uniform(0.1, 0.9)
        direction = self.terrain.rng.uniform(0.0, 360.0)
        self.wind = Wind(speed=speed, direction_degrees=direction)
        
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        fire_config = self.config.get('fire', {})
        self.fire_manager = FireManager(width, height, self.terrain.rng, fire_config)
        start_x = self.terrain.rng.integers(0, width)
        start_y = self.terrain.rng.integers(0, height)
        self.fire_manager.ignite(start_x, start_y)
            
        drone_config = self.config['drone']
        
        # Base Station location (Top-Left corner)
        self.base_x = 2
        self.base_y = 2
        
        # Heterogeneous Drone Fleet
        self.drones = [
            Drone(0, DroneType.WATER, self.base_x, self.base_y, drone_config['water']),
            Drone(1, DroneType.WATER, self.base_x, self.base_y, drone_config['water']),
            Drone(2, DroneType.WATER, self.base_x, self.base_y, drone_config['water']),
            Drone(3, DroneType.RETARDANT, self.base_x, self.base_y, drone_config['retardant'])
        ]

    def is_at_base(self, drone: Drone) -> bool:
        """Checks if a drone is within the 2x2 base footprint."""
        return (drone.x >= self.base_x and drone.x <= self.base_x + 1 and 
                drone.y >= self.base_y and drone.y <= self.base_y + 1)

    def distance_to_base(self, drone: Drone) -> int:
        """Calculate minimum Manhattan distance to any valid base cell (2x2 footprint)."""
        base_cells = [
            (self.base_x, self.base_y),
            (self.base_x + 1, self.base_y),
            (self.base_x, self.base_y + 1),
            (self.base_x + 1, self.base_y + 1)
        ]
        return min(abs(drone.x - bx) + abs(drone.y - by) for bx, by in base_cells)

    def minimum_return_battery(self, drone: Drone) -> int:
        """Calculate minimum battery required to return."""
        return self.distance_to_base(drone) * drone.move_cost

    def required_battery(self, drone: Drone) -> int:
        """Calculate total battery required including safety reserve."""
        battery_reserve = self.config['return_to_base'].get('battery_reserve', 5)
        return self.minimum_return_battery(drone) + battery_reserve

    def battery_margin(self, drone: Drone) -> int:
        """Calculate how much battery remains above the safe return threshold."""
        return drone.battery - self.required_battery(drone)

    def can_safely_return_to_base(self, drone: Drone) -> bool:
        """Check if the drone has enough battery to safely return."""
        return self.battery_margin(drone) >= 0

    def process_base_refills(self):
        for drone in self.drones:
            if drone.active and self.is_at_base(drone):
                drone.battery = drone.max_battery
                drone.payload = drone.max_payload

    def step(self):
        self.fire_manager.step(self.terrain, self.wind)
        
        # Dummy AI (Random Walk) for visual testing until RL is hooked up
        for drone in self.drones:
            if not drone.active:
                continue
                
            # Refill at Base Station
            if self.is_at_base(drone):
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
