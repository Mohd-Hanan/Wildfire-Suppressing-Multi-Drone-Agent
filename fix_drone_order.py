import re

with open("src/wildfire/agents/testing_controller.py", "r") as f:
    code = f.read()

replacement = """
        # Sort drones so RETARDANT drone (type 3 or id 3) gets to pick target first
        from wildfire.simulation.drone import DroneType
        sorted_drones = sorted(self.world.drones, key=lambda d: 0 if d.type == DroneType.RETARDANT else 1)
        for drone in sorted_drones:
"""

code = code.replace(
    '        for drone in self.world.drones:',
    replacement[1:]
)

with open("src/wildfire/agents/testing_controller.py", "w") as f:
    f.write(code)
