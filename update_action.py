import re

with open('src/wildfire/environment/action.py', 'r') as f:
    content = f.read()

new_water = """        elif action_id == 5:
            if drone.type == DroneType.WATER:
                if drone.drop():
                    world.terrain.moisture[drone.x, drone.y] = 1.0
                    fm = world.fire_manager.fire_map
                    
                    import numpy as np
                    active_mask = (fm == FireState.IGNITING) | (fm == FireState.BURNING) | (fm == FireState.SMOLDERING)
                    active_coords = np.argwhere(active_mask)
                    
                    if len(active_coords) > 0:
                        dists = np.abs(active_coords[:, 0] - drone.x) + np.abs(active_coords[:, 1] - drone.y)
                        nearest_idx = np.argmin(dists)
                        min_dist = dists[nearest_idx]
                        
                        if min_dist <= 1:
                            target_x, target_y = active_coords[nearest_idx]
                            fm[target_x, target_y] = FireState.BURNED
                            world.fire_manager.burn_timers[target_x, target_y] = 0
                            suppressed = 1"""

content = re.sub(r'        elif action_id == 5:.*?suppressed = 1', new_water, content, flags=re.DOTALL)

with open('src/wildfire/environment/action.py', 'w') as f:
    f.write(content)
