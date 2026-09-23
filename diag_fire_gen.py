import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv

env = WildfireEnv()
counts = {"NORTH": 0, "SOUTH": 0, "EAST": 0, "WEST": 0}

for i in range(100):
    obs, _ = env.reset(seed=i)
    # The drone starts at base. So drone x, y = base_x, base_y (or very close).
    # We can just look at fire_dx, fire_dy from the observation.
    drone_obs = obs["drone"]
    fire_dx = drone_obs[6]
    fire_dy = drone_obs[7]
    
    if fire_dx > 0: counts["EAST"] += 1
    elif fire_dx < 0: counts["WEST"] += 1
    
    if fire_dy > 0: counts["SOUTH"] += 1
    elif fire_dy < 0: counts["NORTH"] += 1

print(f"Across 100 seeds, nearest fire relative to start position:")
for k, v in counts.items():
    print(f"{k}: {v}")
