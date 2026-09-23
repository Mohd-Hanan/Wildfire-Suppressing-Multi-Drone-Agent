import numpy as np
from typing import List, Tuple, Dict
from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone
from wildfire.simulation.fire import FireState

class RewardCalculator:
    def __init__(self, config: dict):
        self.reset()

    def reset(self):
        """Clears state from previous episodes."""
        self.prev_affected_cells = -1
        self.prev_drone_active = {}

    def _get_affected_cells(self, world: World) -> int:
        return int(np.sum(world.fire_manager.fire_map != FireState.UNBURNED))

    def _get_active_fire_cells(self, world: World) -> int:
        fm = world.fire_manager.fire_map
        return int(np.sum((fm == FireState.IGNITING) | 
                          (fm == FireState.BURNING) | 
                          (fm == FireState.SMOLDERING)))

    def calculate(
        self, 
        world: World, 
        drones: List[Drone], 
        hit_boundary: bool = False, 
        newly_suppressed_cells: int = 0, 
        last_action: int = 0,
        pre_drone_pos: Tuple[int, int] = None,
        pre_target: Tuple[int, int] = None,
        pre_active_fire_count: int = 0
    ) -> Tuple[float, Dict[str, float]]:
        
        current_affected = self._get_affected_cells(world)
        current_active_fire = self._get_active_fire_cells(world)
        
        # 1. First step initialization
        if self.prev_affected_cells == -1:
            self.prev_affected_cells = current_affected
            for d in drones:
                self.prev_drone_active[d.id] = d.active
            return 0.0, {
                'damage_penalty': 0.0,
                'suppression_reward': 0.0,
                'distance_reward': 0.0,
                'water_penalty': 0.0,
                'step_penalty': 0.0,
                'boundary_penalty': -0.1 if hit_boundary else 0.0,
                'crash_penalty': 0.0,
                'battery_safety_penalty': 0.0,
                'extinction_reward': 0.0,
                'total_reward': 0.0,
                'newly_suppressed_cells': 0,
                'new_burned_cells': 0
            }

        # 2. Damage
        new_burned_cells = max(0, current_affected - self.prev_affected_cells)
        damage_penalty = -0.1 * new_burned_cells

        # 3. Distance progress shaping
        distance_reward = 0.0
        d = drones[0]
        if d.active and pre_drone_pos is not None and pre_target is not None:
            pre_x, pre_y = pre_drone_pos
            tgt_x, tgt_y = pre_target
            
            # Distance before action
            prev_dist = abs(pre_x - tgt_x) + abs(pre_y - tgt_y)
            # Distance after action
            curr_dist = abs(d.x - tgt_x) + abs(d.y - tgt_y)
            
            progress = prev_dist - curr_dist
            distance_reward += 0.20 * progress
            # Note: reach fire reward removed entirely as per instructions

        # 4. Suppression and Water Penalty
        suppression_reward = 0.0
        water_penalty = 0.0
        
        # Calculate current live distance for WATER penalty
        fm = world.fire_manager.fire_map
        active_mask = (fm == FireState.IGNITING) | (fm == FireState.BURNING) | (fm == FireState.SMOLDERING)
        active_coords = np.argwhere(active_mask)
        if len(active_coords) > 0:
            dists = np.abs(active_coords[:, 0] - d.x) + np.abs(active_coords[:, 1] - d.y)
            live_dist = np.min(dists)
        else:
            live_dist = None
            
        if last_action == 5: # WATER
            if newly_suppressed_cells > 0:
                suppression_reward += 10.0 * newly_suppressed_cells
            else:
                # Failed water drop
                if live_dist is not None:
                    if live_dist > 10:
                        water_penalty -= 1.0
                    elif live_dist > 5:
                        water_penalty -= 0.5
                    elif live_dist > 1:
                        water_penalty -= 0.1

        # 5. Penalties
        step_penalty = -0.01
        boundary_penalty = -0.1 if hit_boundary else 0.0
        
        crash_penalty = 0.0
        battery_safety_penalty = 0.0
        
        for drone in drones:
            if self.prev_drone_active.get(drone.id, True) and not drone.active:
                crash_penalty -= 10.0
            self.prev_drone_active[drone.id] = drone.active
            
            if drone.active:
                margin = world.battery_margin(drone)
                if margin < 0:
                    battery_safety_penalty -= 0.05 * abs(margin)

        # 6. Extinction (strictly causal)
        extinction_reward = 0.0
        if current_active_fire == 0 and pre_active_fire_count > 0:
            if newly_suppressed_cells >= pre_active_fire_count:
                extinction_reward = 50.0

        # Update prev state
        self.prev_affected_cells = current_affected

        total_reward = (damage_penalty + 
                        distance_reward +
                        suppression_reward + 
                        water_penalty +
                        step_penalty + 
                        boundary_penalty +
                        crash_penalty + 
                        battery_safety_penalty +
                        extinction_reward)

        info = {
            'damage_penalty': damage_penalty,
            'distance_reward': distance_reward,
            'suppression_reward': suppression_reward,
            'water_penalty': water_penalty,
            'step_penalty': step_penalty,
            'boundary_penalty': boundary_penalty,
            'crash_penalty': crash_penalty,
            'battery_safety_penalty': battery_safety_penalty,
            'extinction_reward': extinction_reward,
            'total_reward': total_reward,
            'newly_suppressed_cells': newly_suppressed_cells,
            'new_burned_cells': new_burned_cells
        }

        return total_reward, info
