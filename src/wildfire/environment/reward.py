import numpy as np
from typing import List, Tuple, Dict
from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone
from wildfire.simulation.fire import FireState

class RewardCalculator:
    def __init__(self, config: dict):
        rc = config.get('reward', {})
        self.new_burned_cell_penalty = float(rc.get('new_burned_cell_penalty', 5.0))
        self.effective_suppression_reward = float(rc.get('effective_suppression_reward', 0.0))
        self.containment_reward = float(rc.get('containment_reward', 0.0))
        self.step_penalty = float(rc.get('step_penalty', 0.01))
        self.drone_crash_penalty = float(rc.get('drone_crash_penalty', 10.0))
        self.fire_extinguished_reward = float(rc.get('fire_extinguished_reward', 100.0))
        self.boundary_hit_penalty = float(rc.get('boundary_hit_penalty', 0.1))

        self.reset()

    def reset(self):
        """Clears state from previous episodes."""
        self.prev_affected_cells = -1
        self.prev_drone_active = {}
        self.extinguished_awarded = False
        self.prev_active_fire_count = -1

    def _get_affected_cells(self, world: World) -> int:
        """Counts cells that are not UNBURNED."""
        return int(np.sum(world.fire_manager.fire_map != FireState.UNBURNED))

    def _get_active_fire_cells(self, world: World) -> int:
        """Counts cells that are IGNITING, BURNING, or SMOLDERING."""
        fm = world.fire_manager.fire_map
        return int(np.sum((fm == FireState.IGNITING) | 
                          (fm == FireState.BURNING) | 
                          (fm == FireState.SMOLDERING)))

    def calculate(self, world: World, drones: List[Drone], hit_boundary: bool = False, newly_suppressed_cells: int = 0) -> Tuple[float, Dict[str, float]]:
        # 1. First-step initialization (no time has passed yet)
        current_affected = self._get_affected_cells(world)
        current_active_fire = self._get_active_fire_cells(world)
        
        if self.prev_affected_cells == -1:
            self.prev_affected_cells = current_affected
            for d in drones:
                self.prev_drone_active[d.id] = d.active
            self.prev_active_fire_count = current_active_fire
            # On step 0, no reward calculation makes sense since no step occurred.
            # But typically this is called after a step. If called before step 1 accidentally, return 0.
            return 0.0, {
                'new_burned_cells': 0, 'damage_penalty': 0.0,
                'newly_suppressed_cells': 0, 'suppression_reward': 0.0,
                'containment_progress': 0.0, 'step_penalty': 0.0,
                'crash_penalty': 0.0, 'extinction_reward': 0.0,
                'boundary_penalty': -self.boundary_hit_penalty if hit_boundary else 0.0,
                'battery_safety_penalty': 0.0,
                'total_reward': 0.0
            }

        # 2. Wildfire Damage
        new_burned_cells = max(0, current_affected - self.prev_affected_cells)
        damage_penalty = -self.new_burned_cell_penalty * new_burned_cells

        # 3. Suppression & Containment
        suppression_reward = self.effective_suppression_reward * newly_suppressed_cells
        containment_progress = 0.0

        # 4. Step penalty
        step_penalty = -self.step_penalty
        
        # 4b. Boundary penalty
        boundary_penalty = -self.boundary_hit_penalty if hit_boundary else 0.0

        # 5. Crash penalty & Battery Safety Penalty
        crash_penalty = 0.0
        battery_safety_penalty = 0.0
        for d in drones:
            if self.prev_drone_active.get(d.id, True) and not d.active:
                # Transitioned to inactive this step
                crash_penalty -= self.drone_crash_penalty
            self.prev_drone_active[d.id] = d.active
            
            # Progressive battery safety penalty
            if d.active:
                margin = world.battery_margin(d)
                if margin < 0:
                    battery_safety_penalty -= 0.5 * abs(margin)

        # 6. Extinction
        extinction_reward = 0.0
        if current_active_fire == 0 and self.prev_active_fire_count > 0 and not self.extinguished_awarded:
            extinction_reward = self.fire_extinguished_reward
            self.extinguished_awarded = True

        # Combine
        total_reward = (damage_penalty + 
                        suppression_reward + 
                        containment_progress + 
                        step_penalty + 
                        boundary_penalty +
                        crash_penalty + 
                        battery_safety_penalty +
                        extinction_reward)

        # Update tracking variables
        self.prev_affected_cells = current_affected
        self.prev_active_fire_count = current_active_fire

        info = {
            'new_burned_cells': new_burned_cells,
            'damage_penalty': damage_penalty,
            'newly_suppressed_cells': newly_suppressed_cells,
            'suppression_reward': suppression_reward,
            'containment_progress': containment_progress,
            'step_penalty': step_penalty,
            'boundary_penalty': boundary_penalty,
            'crash_penalty': crash_penalty,
            'battery_safety_penalty': battery_safety_penalty,
            'extinction_reward': extinction_reward,
            'total_reward': total_reward
        }

        return total_reward, info
