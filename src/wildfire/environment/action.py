from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone, DroneType
from wildfire.simulation.fire import FireState
from typing import Tuple

class ActionExecutor:
    """
    Action Mapping:
    0 = Stay
    1 = North (dy = -1)
    2 = South (dy = +1)
    3 = East (dx = +1)
    4 = West (dx = -1)
    5 = Drop Water
    6 = Drop Retardant
    """
    
    @staticmethod
    def execute(drone: Drone, world: World, action_id: int) -> Tuple[bool, int]:
        """Returns (hit_boundary, newly_suppressed_cells)"""
        if not drone.active:
            return False, 0

        hit_boundary = False
        suppressed = 0

        if action_id == 0:
            pass # Stay
        elif action_id == 1:
            hit_boundary = drone.move(0, -1, world.width, world.height)
        elif action_id == 2:
            hit_boundary = drone.move(0, 1, world.width, world.height)
        elif action_id == 3:
            hit_boundary = drone.move(1, 0, world.width, world.height)
        elif action_id == 4:
            hit_boundary = drone.move(-1, 0, world.width, world.height)
        elif action_id == 5:
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
                            suppressed = 1
        elif action_id == 6:
            if drone.type == DroneType.RETARDANT:
                if drone.drop():
                    world.terrain.fuel[drone.x, drone.y] = 0.0
                    
        return hit_boundary, suppressed
