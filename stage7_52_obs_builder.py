import numpy as np
from wildfire.environment.observation import ObservationBuilder
from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone
from wildfire.simulation.fire import FireState

class GlobalFireObservationBuilder(ObservationBuilder):
    def _get_spatial(self, drone: Drone, world: World) -> np.ndarray:
        # Call the original to get the first 5 channels
        spatial_5 = super()._get_spatial(drone, world)
        
        # 6. Global Fire Direction Channel
        ch_global_fire = np.zeros((self.window_size, self.window_size), dtype=np.float32)
        fm = world.fire_manager.fire_map
        active_mask = (fm == FireState.IGNITING) | (fm == FireState.BURNING) | (fm == FireState.SMOLDERING)
        active_coords = np.argwhere(active_mask)
        
        if len(active_coords) > 0:
            distances = np.sqrt((active_coords[:, 0] - drone.x)**2 + (active_coords[:, 1] - drone.y)**2)
            nearest_idx = np.argmin(distances)
            fire_x, fire_y = active_coords[nearest_idx]
            
            max_dist = np.sqrt(world.width**2 + world.height**2)
            
            for lx in range(self.window_size):
                for ly in range(self.window_size):
                    wx = drone.x - self.half_window + lx
                    wy = drone.y - self.half_window + ly
                    dist = np.sqrt((fire_x - wx)**2 + (fire_y - wy)**2)
                    ch_global_fire[lx, ly] = 1.0 - (dist / max_dist)
                    
        # Stack all 6
        ch_global_fire = np.expand_dims(ch_global_fire, axis=0)
        spatial_6 = np.concatenate([spatial_5, ch_global_fire], axis=0).astype(np.float32)
        return spatial_6
