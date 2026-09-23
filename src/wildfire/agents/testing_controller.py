import numpy as np
from typing import Dict
from collections import deque
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from wildfire.simulation.drone import DroneType

class MultiDroneTestingController:
    def __init__(self, world):
        self.world = world
        
    def get_actions(self) -> Dict[int, int]:
        actions = {}
        fm = self.world.fire_manager.fire_map
        active_fires = np.argwhere((fm == 1) | (fm == 2) | (fm == 3))
        
        planned_positions = {}
        assigned_targets = {}
        
        base_cells = [
            (self.world.base_x, self.world.base_y),
            (self.world.base_x + 1, self.world.base_y),
            (self.world.base_x, self.world.base_y + 1),
            (self.world.base_x + 1, self.world.base_y + 1)
        ]
        
        # 1. Cluster fires
        clusters = []
        if len(active_fires) > 0:
            k = min(4, len(active_fires))
            if k == 1:
                clusters.append({'centroid': active_fires[0], 'size': len(active_fires), 'cells': active_fires})
            else:
                # Add noise to prevent kmeans warnings on identical points
                data = active_fires.astype(float) + np.random.rand(*active_fires.shape) * 0.01
                centroids, labels = kmeans2(data, k, minit='points')
                for i in range(k):
                    c_cells = active_fires[labels == i]
                    if len(c_cells) > 0:
                        clusters.append({'centroid': centroids[i], 'size': len(c_cells), 'cells': c_cells})
        
        # Select one representative cell per cluster
        target_candidates = []
        for c in clusters:
            # Find the actual fire cell closest to centroid
            dists = np.sum(np.abs(c['cells'] - c['centroid']), axis=1)
            best_cell = tuple(c['cells'][np.argmin(dists)])
            target_candidates.append({
                'pos': best_cell,
                'size': c['size']
            })
            
        # 2. Score and Assign Targets Globally
        active_drones = [d for d in self.world.drones if d.active]
        n_drones = len(active_drones)
        
        # We need enough candidates so every drone gets an assignment
        # If there are fewer candidates than drones, duplicate them or use base
        while len(target_candidates) < n_drones and len(target_candidates) > 0:
            # Duplicate the largest cluster
            largest = max(target_candidates, key=lambda x: x['size'])
            target_candidates.append(largest)
            
        if len(target_candidates) == 0:
            # No fires, all candidates are just bases
            for i in range(n_drones):
                target_candidates.append({'pos': base_cells[i % 4], 'size': 0})
        
        cost_matrix = np.zeros((n_drones, len(target_candidates)))
        
        # Evaluate must_return status
        must_return_status = {}
        for d_idx, drone in enumerate(active_drones):
            my_base = base_cells[drone.id % 4]
            dist_to_my_base = abs(drone.x - my_base[0]) + abs(drone.y - my_base[1])
            reserve = 30 if drone.type == DroneType.WATER else 10
            battery_needed_to_return = dist_to_my_base * drone.move_cost + reserve
            
            must_return = False
            if drone.payload < drone.drop_payload_cost:
                must_return = True
            elif drone.battery <= battery_needed_to_return:
                must_return = True
            elif len(active_fires) == 0:
                must_return = True
                
            # Check if at base and needing refill
            if self.world.is_at_base(drone) and (drone.payload < drone.drop_payload_cost or drone.battery < drone.max_battery * 0.9):
                must_return = True # Force it to stay/target base
                
            must_return_status[drone.id] = must_return
            
            for c_idx, cand in enumerate(target_candidates):
                target_pos = cand['pos']
                
                if must_return:
                    # Score base heavily, penalize everything else
                    if target_pos in base_cells:
                        cost_matrix[d_idx, c_idx] = 0
                    else:
                        cost_matrix[d_idx, c_idx] = 1000000
                else:
                    dist_to_target = abs(drone.x - target_pos[0]) + abs(drone.y - target_pos[1])
                    dist_target_to_base = abs(target_pos[0] - my_base[0]) + abs(target_pos[1] - my_base[1])
                    total_cost = (dist_to_target + dist_target_to_base) * drone.move_cost + drone.drop_cost + reserve
                    
                    if total_cost > drone.battery:
                        # Cannot safely reach it
                        cost_matrix[d_idx, c_idx] = 1000000
                    else:
                        # Base cost is distance
                        score = dist_to_target * 10
                        
                        # Retardant prefers larger clusters
                        if drone.type == DroneType.RETARDANT:
                            score -= cand['size'] * 5
                        
                        cost_matrix[d_idx, c_idx] = score

        # Global assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Ensure we have valid base fallbacks if they got blocked by 1000000
        drone_targets = {}
        for idx in range(n_drones):
            d_idx = row_ind[idx]
            c_idx = col_ind[idx]
            drone = active_drones[d_idx]
            
            if cost_matrix[d_idx, c_idx] >= 1000000:
                # Fallback to a base cell
                drone_targets[drone.id] = base_cells[drone.id % 4]
            else:
                drone_targets[drone.id] = target_candidates[c_idx]['pos']
                
        # Handle "blocked base" logic dynamically
        # If multiple drones want a base cell, or it's blocked by a sitting drone
        sitting_drones = [d for d in active_drones if self.world.is_at_base(d) and (d.payload < d.drop_payload_cost or d.battery < d.max_battery * 0.9)]
        blocked_bases = set([(d.x, d.y) for d in sitting_drones])
        
        for drone in active_drones:
            if drone_targets[drone.id] in base_cells:
                # If my assigned base is blocked by a sitter (who is not me), pick another
                if drone_targets[drone.id] in blocked_bases and not ((drone.x, drone.y) == drone_targets[drone.id]):
                    for bc in base_cells:
                        if bc not in blocked_bases:
                            drone_targets[drone.id] = bc
                            break
                            
        # 3. Pathfinding & Movement (resolve concurrently/sequentially with separation)
        for drone in active_drones:
            target_x, target_y = drone_targets[drone.id]
            drone.current_target = (target_x, target_y)
            
            # Refill logic
            if self.world.is_at_base(drone) and (drone.payload < drone.drop_payload_cost or drone.battery < drone.max_battery * 0.9):
                actions[drone.id] = 0
                planned_positions[drone.id] = (drone.x, drone.y)
                drone.current_action = 0
                continue
                
            dx, dy = target_x - drone.x, target_y - drone.y
            dist = abs(dx) + abs(dy)
            
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
                # BFS with soft separation
                obstacles = set(planned_positions.values())
                for other in active_drones:
                    if other.id != drone.id and other.id not in planned_positions:
                        obstacles.add((other.x, other.y))
                        
                queue = deque([(drone.x, drone.y, [])])
                visited = set([(drone.x, drone.y)])
                path = []
                
                # Shuffle the moves to avoid always preferring North/East (adds organic variety)
                import random
                moves = [(1, 0, -1), (2, 0, 1), (3, 1, 0), (4, -1, 0)]
                random.shuffle(moves)
                
                while queue:
                    cx, cy, current_path = queue.popleft()
                    
                    if (cx, cy) == (target_x, target_y):
                        path = current_path
                        break
                        
                    for act, mx, my in moves:
                        nx, ny = cx + mx, cy + my
                        if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                            if (nx, ny) not in obstacles and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny, current_path + [(act, nx, ny)]))
                                
                if path:
                    selected_action = path[0][0]
                    selected_pos = (path[0][1], path[0][2])
                else:
                    selected_action = 0
                    selected_pos = (drone.x, drone.y)
                    
                actions[drone.id] = selected_action
                planned_positions[drone.id] = selected_pos
                drone.current_action = selected_action
                
        return actions
