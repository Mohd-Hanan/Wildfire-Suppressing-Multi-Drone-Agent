import re

with open("src/wildfire/agents/demo_controller.py", "r") as f:
    code = f.read()

replacement = """
            dx, dy = target_x - drone.x, target_y - drone.y
            dist = abs(dx) + abs(dy)
            
            base_target = (target_x, target_y) in base_cells
            if dist <= 1 and not base_target:
                from wildfire.simulation.drone import DroneType
                actions[drone.id] = 5 if drone.type == DroneType.WATER else 6
                planned_positions[drone.id] = (drone.x, drone.y)
            elif dist == 0:
                actions[drone.id] = 0
                planned_positions[drone.id] = (drone.x, drone.y)
            else:
                # BFS to find shortest path to target_x, target_y avoiding planned_positions and sitting drones
                from collections import deque
                
                # Treat other drones that have already planned as obstacles
                obstacles = set(planned_positions.values())
                for other in self.world.drones:
                    if other.id != drone.id and other.active and other.id not in planned_positions:
                        # Also treat drones sitting at base as obstacles if we are not adjacent to target
                        if self.world.is_at_base(other) and (other.payload < other.drop_payload_cost or other.battery < other.max_battery * 0.9):
                            obstacles.add((other.x, other.y))

                queue = deque([(drone.x, drone.y, [])])
                visited = set([(drone.x, drone.y)])
                path = []
                
                while queue:
                    cx, cy, current_path = queue.popleft()
                    
                    if (cx, cy) == (target_x, target_y):
                        path = current_path
                        break
                        
                    # Stop searching if it takes too long (e.g. 50 depth)
                    if len(current_path) > 50:
                        continue
                        
                    for act, mx, my in [(1, 0, -1), (2, 0, 1), (3, 1, 0), (4, -1, 0)]:
                        nx, ny = cx + mx, cy + my
                        if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                            if (nx, ny) not in obstacles and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny, current_path + [(act, nx, ny)]))
                                
                if path:
                    selected_action = path[0][0]
                    selected_pos = (path[0][1], path[0][2])
                else:
                    # Fallback if BFS fails (no path)
                    selected_action = 0
                    selected_pos = (drone.x, drone.y)
                    
                actions[drone.id] = selected_action
                planned_positions[drone.id] = selected_pos
"""

code = re.sub(
    r'            dx, dy = target_x - drone\.x, target_y - drone\.y.*?                planned_positions\[drone\.id\] = selected_pos',
    replacement[1:],
    code,
    flags=re.DOTALL
)

with open("src/wildfire/agents/demo_controller.py", "w") as f:
    f.write(code)
