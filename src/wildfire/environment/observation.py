import numpy as np
from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone
from wildfire.simulation.fire import FireState

class ObservationBuilder:
    def __init__(self, window_size: int = 11):
        self.window_size = window_size
        self.half_window = window_size // 2

    def get_observation(self, drone: Drone, world: World) -> dict:
        """Constructs the local RL observation for the given drone."""
        return {
            'spatial': self._get_spatial(drone, world),
            'drone': self._get_drone_state(drone, world),
            'wind': self._get_wind_state(world)
        }

    def _get_spatial(self, drone: Drone, world: World) -> np.ndarray:
        """
        Builds the 11x11 spatial channels.
        Channel 0: Fire state
        Channel 1: Terrain elevation
        Channel 2: Terrain slope
        Channel 3: Drone occupancy
        Channel 4: Base location
        """
        # Crop bounds
        x_min = drone.x - self.half_window
        x_max = drone.x + self.half_window + 1
        y_min = drone.y - self.half_window
        y_max = drone.y + self.half_window + 1
        
        # Helper to crop and pad an array
        def crop_and_pad(arr: np.ndarray, fill_value=0.0):
            # Calculate pad widths
            pad_x_before = max(0, -x_min)
            pad_x_after = max(0, x_max - world.width)
            pad_y_before = max(0, -y_min)
            pad_y_after = max(0, y_max - world.height)
            
            # Slice indices bounded to the map
            src_x_min = max(0, x_min)
            src_x_max = min(world.width, x_max)
            src_y_min = max(0, y_min)
            src_y_max = min(world.height, y_max)
            
            # Extract
            cropped = arr[src_x_min:src_x_max, src_y_min:src_y_max]
            
            # Pad
            padded = np.pad(cropped, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)), 
                            mode='constant', constant_values=fill_value)
            return padded

        # 1. Fire State Mapping
        # UNBURNED=0.0, IGNITING=0.25, BURNING=0.75, SMOLDERING=0.50, BURNED=1.0
        fire_map = np.zeros((world.width, world.height), dtype=np.float32)
        fire_map[world.fire_manager.fire_map == FireState.IGNITING] = 0.25
        fire_map[world.fire_manager.fire_map == FireState.SMOLDERING] = 0.50
        fire_map[world.fire_manager.fire_map == FireState.BURNING] = 0.75
        fire_map[world.fire_manager.fire_map == FireState.BURNED] = 1.0
        ch_fire = crop_and_pad(fire_map)

        # 2. Elevation
        ch_elev = crop_and_pad(world.terrain.elevation)

        # 3. Slope
        ch_slope = crop_and_pad(world.terrain.slope)

        # 4. Drone Occupancy
        drone_map = np.zeros((world.width, world.height), dtype=np.float32)
        for d in world.drones:
            if not d.active:
                continue
            if d.id == drone.id:
                drone_map[d.x, d.y] = 1.0
            else:
                # Assuming another drone occupies the same space, 1.0 overwrites 0.5
                if drone_map[d.x, d.y] != 1.0:
                    drone_map[d.x, d.y] = 0.5
        ch_drones = crop_and_pad(drone_map)

        # 5. Base Location
        base_map = np.zeros((world.width, world.height), dtype=np.float32)
        base_map[world.base_x:world.base_x+2, world.base_y:world.base_y+2] = 1.0
        ch_base = crop_and_pad(base_map)

        # Note: Suppression state (Channel 6) is pending.

        spatial = np.stack([ch_fire, ch_elev, ch_slope, ch_drones, ch_base], axis=0).astype(np.float32)
        return spatial

    def _get_drone_state(self, drone: Drone, world: World) -> np.ndarray:
        battery = np.clip(drone.battery / drone.max_battery, 0.0, 1.0)
        payload = np.clip(drone.payload / drone.max_payload, 0.0, 1.0)
        
        # Battery margin logic (-1.0 to 1.0 approximately)
        safe_margin = world.battery_margin(drone)
        # Normalize margin using max_battery to keep it stable
        normalized_margin = np.clip(safe_margin / drone.max_battery, -1.0, 1.0)
        
        # Distance to base (normalized by max possible Manhattan distance from base to (width-1, height-1))
        max_distance_to_base = ((world.width - 1 - world.base_x) + 
                                (world.height - 1 - world.base_y))
        dist = np.clip(world.distance_to_base(drone) / max_distance_to_base, 0.0, 1.0)
        
        # Normalized coordinates
        nx = drone.x / (world.width - 1)
        ny = drone.y / (world.height - 1)
        
        # Global Fire Information
        fm = world.fire_manager.fire_map
        active_mask = (fm == FireState.IGNITING) | (fm == FireState.BURNING) | (fm == FireState.SMOLDERING)
        active_coords = np.argwhere(active_mask)
        
        if len(active_coords) == 0:
            fire_dx = 0.0
            fire_dy = 0.0
            fire_distance = 0.0
        else:
            distances = np.sqrt((active_coords[:, 0] - drone.x)**2 + (active_coords[:, 1] - drone.y)**2)
            nearest_idx = np.argmin(distances)
            nearest_x, nearest_y = active_coords[nearest_idx]
            
            dx = nearest_x - drone.x
            dy = nearest_y - drone.y
            
            fire_dx = dx / world.width
            fire_dy = dy / world.height
            max_dist = np.sqrt(world.width**2 + world.height**2)
            fire_distance = distances[nearest_idx] / max_dist
        
        return np.array([battery, payload, normalized_margin, dist, nx, ny, fire_dx, fire_dy, fire_distance], dtype=np.float32)

    def _get_wind_state(self, world: World) -> np.ndarray:
        # Assuming max wind speed around 1.0
        speed = np.clip(world.wind.speed, 0.0, 1.0)
        
        # direction is in degrees
        rads = np.radians(world.wind.direction)
        d_sin = np.sin(rads)
        d_cos = np.cos(rads)
        
        return np.array([speed, d_sin, d_cos], dtype=np.float32)
