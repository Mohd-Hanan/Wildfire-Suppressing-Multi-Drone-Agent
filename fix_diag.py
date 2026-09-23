with open("stage7_52_eval_diagnostics.py", "r") as f:
    lines = f.readlines()
    
# Replace the MockWorld and drone setup
import re
new_lines = []
skip = False
for line in lines:
    if "class MockWorld" in line:
        skip = True
    if skip and "world.drones = [drone]" in line:
        skip = False
        new_lines.append("from wildfire.environment.wildfire_env import WildfireEnv\n")
        new_lines.append("env = WildfireEnv('configs/environment.yaml')\n")
        new_lines.append("obs, _ = env.reset(seed=42)\n")
        new_lines.append("drone = env.world.drones[0]\n")
        new_lines.append("drone.x, drone.y = 24, 24\n")
        new_lines.append("world = env.world\n")
        continue
    if not skip:
        new_lines.append(line)

with open("stage7_52_eval_diagnostics.py", "w") as f:
    f.writelines(new_lines)
