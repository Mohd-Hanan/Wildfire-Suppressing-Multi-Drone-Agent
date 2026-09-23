import numpy as np
from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone, DroneType
from wildfire.simulation.fire import FireState

class MockFireManager:
    def __init__(self):
        self.fire_map = np.zeros((48, 48), dtype=int)

class MockWorld:
    def __init__(self):
        self.width = 48
        self.height = 48
        self.fire_manager = MockFireManager()
        
def get_global_fire_channel(drone_x, drone_y, world):
    window_size = 11
    half_window = 5
    ch_global_fire = np.zeros((window_size, window_size), dtype=np.float32)
    fm = world.fire_manager.fire_map
    active_mask = (fm == FireState.IGNITING) | (fm == FireState.BURNING) | (fm == FireState.SMOLDERING)
    active_coords = np.argwhere(active_mask)
    
    if len(active_coords) > 0:
        distances = np.sqrt((active_coords[:, 0] - drone_x)**2 + (active_coords[:, 1] - drone_y)**2)
        nearest_idx = np.argmin(distances)
        fire_x, fire_y = active_coords[nearest_idx]
        
        max_dist = np.sqrt(world.width**2 + world.height**2)
        
        for lx in range(window_size):
            for ly in range(window_size):
                wx = drone_x - half_window + lx
                wy = drone_y - half_window + ly
                dist = np.sqrt((fire_x - wx)**2 + (fire_y - wy)**2)
                ch_global_fire[lx, ly] = 1.0 - (dist / max_dist)
    return ch_global_fire

world = MockWorld()
drone_x = 24
drone_y = 24

cases = {
    "NORTH": (24, 0),
    "SOUTH": (24, 47),
    "EAST": (47, 24),
    "WEST": (0, 24),
    "NE": (47, 0),
    "NW": (0, 0),
    "SE": (47, 47),
    "SW": (0, 47)
}

print("Testing 6th channel gradient...")
for name, (fx, fy) in cases.items():
    world.fire_manager.fire_map.fill(FireState.UNBURNED)
    world.fire_manager.fire_map[fx, fy] = FireState.BURNING
    
    ch = get_global_fire_channel(drone_x, drone_y, world)
    
    # Verify the direction
    # A vector pointing to the highest value from the center
    center_val = ch[5, 5]
    right_val = ch[6, 5]
    left_val = ch[4, 5]
    top_val = ch[5, 4]  # North is negative y
    bottom_val = ch[5, 6] # South is positive y
    
    dx = right_val - left_val
    dy = bottom_val - top_val
    
    # Which direction is max?
    max_idx = np.unravel_index(np.argmax(ch), ch.shape)
    
    print(f"\nFire {name} at {fx, fy}:")
    print(f"  Center value: {center_val:.4f}")
    print(f"  Max value at local coords: {max_idx} (Center is 5,5)")
    print(f"  Gradient dx: {dx:.4f}, dy: {dy:.4f}")
    
