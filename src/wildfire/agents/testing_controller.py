import numpy as np
from typing import Dict
from wildfire.simulation.drone import DroneType
import random

class MultiDroneTestingController:
    def __init__(self, world):
        self.world = world
        
    def get_actions(self) -> Dict[int, int]:
        actions = {}
        fm = self.world.fire_manager.fire_map
        active_fires = np.argwhere((fm == 1) | (fm == 2) | (fm == 3))
        
        planned_positions = {}
        assigned_targets = []
        
        base_cells = [
            (self.world.base_x, self.world.base_y),
            (self.world.base_x + 1, self.world.base_y),
            (self.world.base_x, self.world.base_y + 1),
            (self.world.base_x + 1, self.world.base_y + 1)
        ]
        
        active_drones = [d for d in self.world.drones if d.active]
        
        wind_rad = np.radians(self.world.wind.direction)
        wx = np.cos(wind_rad) * self.world.wind.speed
        wy = np.sin(wind_rad) * self.world.wind.speed
        
        drone_targets = {}
        
        # To make target assignment independent of drone ID processing order,
        # we can shuffle the drones or process them dynamically.
        # But for deterministic testing, we just process them in order.
        
        for drone in active_drones:
            my_base = base_cells[drone.id % 4]
            dist_to_my_base = abs(drone.x - my_base[0]) + abs(drone.y - my_base[1])
            reserve = 20 if drone.type == DroneType.WATER else 10
            battery_needed = dist_to_my_base * drone.move_cost + reserve
            
            must_return = False
            if drone.payload < drone.drop_payload_cost:
                must_return = True
            elif drone.battery <= battery_needed:
                must_return = True
            elif len(active_fires) == 0:
                must_return = True
                
            if self.world.is_at_base(drone) and (drone.payload < drone.drop_payload_cost or drone.battery < drone.max_battery * 0.9):
                must_return = True
                
            if must_return:
                target = my_base
                # simple base block check
                for other in active_drones:
                    if other.id != drone.id and self.world.is_at_base(other):
                        if (other.x, other.y) == target:
                            for bc in base_cells:
                                if bc != (other.x, other.y):
                                    target = bc
                                    break
                drone_targets[drone.id] = target
                assigned_targets.append(target)
                continue
                
            best_score = -float('inf')
            best_target = my_base
            
            fires_to_check = active_fires
            if len(active_fires) > 50:
                indices = np.random.choice(len(active_fires), 50, replace=False)
                fires_to_check = active_fires[indices]
                
            for fx, fy in fires_to_check:
                dist = abs(drone.x - fx) + abs(drone.y - fy)
                score = -dist * 2.0
                
                min_x, max_x = max(0, fx-2), min(self.world.width, fx+3)
                min_y, max_y = max(0, fy-2), min(self.world.height, fy+3)
                intensity = np.sum(fm[min_x:max_x, min_y:max_y] > 0)
                score += intensity * 0.5
                
                dx_wind = fx - drone.x
                dy_wind = fy - drone.y
                wind_alignment = (dx_wind * wx + dy_wind * wy)
                
                if drone.type == DroneType.RETARDANT:
                    score += wind_alignment * 3.0
                    score -= intensity * 0.2
                else:
                    score += wind_alignment * 1.0
                    
                for (tx, ty) in assigned_targets:
                    dist_to_other = abs(tx - fx) + abs(ty - fy)
                    if dist_to_other < 5:
                        score -= (5 - dist_to_other) * 8.0
                        
                if score > best_score:
                    dist_base = abs(fx - my_base[0]) + abs(fy - my_base[1])
                    total_trip_cost = (dist + dist_base) * drone.move_cost + drone.drop_cost + reserve
                    if drone.battery >= total_trip_cost:
                        best_score = score
                        best_target = (fx, fy)
                        
            drone_targets[drone.id] = best_target
            assigned_targets.append(best_target)
            
        def get_dist(d):
            tx, ty = drone_targets[d.id]
            return abs(d.x - tx) + abs(d.y - ty)
            
        active_drones.sort(key=get_dist)
        
        for drone in active_drones:
            target_x, target_y = drone_targets[drone.id]
            drone.current_target = (target_x, target_y)
            
            if self.world.is_at_base(drone) and (drone.payload < drone.drop_payload_cost or drone.battery < drone.max_battery * 0.9):
                actions[drone.id] = 0
                planned_positions[drone.id] = (drone.x, drone.y)
                drone.current_action = 0
                continue
                
            dist = abs(target_x - drone.x) + abs(target_y - drone.y)
            base_target = (target_x, target_y) in base_cells
            
            if dist <= 1 and not base_target:
                actions[drone.id] = 5 if drone.type == DroneType.WATER else 6
                planned_positions[drone.id] = (drone.x, drone.y)
                drone.current_action = actions[drone.id]
            elif dist == 0:
                actions[drone.id] = 0
                planned_positions[drone.id] = (drone.x, drone.y)
                drone.current_action = 0
            else:
                obstacles = set(planned_positions.values())
                for other in active_drones:
                    if other.id != drone.id and other.id not in planned_positions:
                        obstacles.add((other.x, other.y))
                        
                valid_moves = []
                valid_moves.append((dist * 10 + 10.0, 0, drone.x, drone.y)) 
                
                for act, mx, my in [(1, 0, -1), (2, 0, 1), (3, 1, 0), (4, -1, 0)]:
                    nx, ny = drone.x + mx, drone.y + my
                    if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                        if (nx, ny) not in obstacles:
                            euclid = np.sqrt((target_x - nx)**2 + (target_y - ny)**2)
                            noise = random.uniform(0.0, 0.4)
                            score = euclid * 10 + noise
                            valid_moves.append((score, act, nx, ny))
                            
                valid_moves.sort(key=lambda x: x[0])
                best_score, selected_action, nx, ny = valid_moves[0]
                
                actions[drone.id] = selected_action
                planned_positions[drone.id] = (nx, ny)
                drone.current_action = selected_action
                
        return actions
