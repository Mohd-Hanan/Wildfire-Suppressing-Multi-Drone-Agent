import re

with open("src/wildfire/agents/testing_controller.py", "r") as f:
    code = f.read()

replacement = """
                if path:
                    selected_action = path[0][0]
                    selected_pos = (path[0][1], path[0][2])
                else:
                    # Fallback if BFS fails (no path)
                    selected_action = 0
                    selected_pos = (drone.x, drone.y)
                    
                actions[drone.id] = selected_action
                planned_positions[drone.id] = selected_pos
            
            # Inject for renderer
            drone.current_target = (target_x, target_y)
            drone.current_action = actions[drone.id]
"""

code = re.sub(
    r'                if path:.*?                planned_positions\[drone\.id\] = selected_pos',
    replacement[1:],
    code,
    flags=re.DOTALL
)

with open("src/wildfire/agents/testing_controller.py", "w") as f:
    f.write(code)
