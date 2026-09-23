import re

with open("src/wildfire/agents/testing_controller.py", "r") as f:
    code = f.read()

replacement = """
            from wildfire.simulation.drone import DroneType
            reserve = 30 if drone.type == DroneType.WATER else 6
            battery_needed_to_return = dist_to_my_base * drone.move_cost + reserve
"""
code = code.replace(
    'battery_needed_to_return = dist_to_my_base * drone.move_cost + 30 # 30 is safety reserve',
    replacement[1:]
)

replacement2 = """
                    total_cost = (dist_to_target + dist_target_to_base) * drone.move_cost + drone.drop_cost + reserve
"""
code = code.replace(
    'total_cost = (dist_to_target + dist_target_to_base) * drone.move_cost + drone.drop_cost + 30',
    replacement2[1:]
)

with open("src/wildfire/agents/testing_controller.py", "w") as f:
    f.write(code)
