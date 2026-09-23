import re

with open("src/wildfire/agents/demo_controller.py", "r") as f:
    code = f.read()

replacement = """
                # Treat other drones that have already planned as obstacles
                obstacles = set(planned_positions.values())
                for other in self.world.drones:
                    if other.id != drone.id and other.active and other.id not in planned_positions:
                        # If the other drone hasn't planned yet, treat its CURRENT position as an obstacle
                        obstacles.add((other.x, other.y))
"""

code = re.sub(
    r'                # Treat other drones that have already planned as obstacles.*?                            obstacles\.add\(\(other\.x, other\.y\)\)',
    replacement[1:],
    code,
    flags=re.DOTALL
)

with open("src/wildfire/agents/demo_controller.py", "w") as f:
    f.write(code)
