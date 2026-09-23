import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv

env = WildfireEnv("configs/environment.yaml")

fdx = []
fdy = []
fdist = []
dirs = {"NORTH": 0, "SOUTH": 0, "EAST": 0, "WEST": 0, "NE": 0, "NW": 0, "SE": 0, "SW": 0, "NEAR": 0}

for _ in range(100):
    obs, _ = env.reset()
    drone = obs['drone']
    dx = drone[6]
    dy = drone[7]
    dist = drone[8]
    
    fdx.append(dx)
    fdy.append(dy)
    fdist.append(dist)
    
    if dist <= (1.5 / np.sqrt(48**2 + 48**2)):
        dirs["NEAR"] += 1
    else:
        if abs(dx) > 1.5 * abs(dy):
            if dx > 0: dirs["EAST"] += 1
            else: dirs["WEST"] += 1
        elif abs(dy) > 1.5 * abs(dx):
            if dy > 0: dirs["SOUTH"] += 1
            else: dirs["NORTH"] += 1
        else:
            if dx > 0 and dy > 0: dirs["SE"] += 1
            elif dx > 0 and dy < 0: dirs["NE"] += 1
            elif dx < 0 and dy > 0: dirs["SW"] += 1
            else: dirs["NW"] += 1

print("\n--- DIAGNOSTIC 14: FIRE DIRECTION DATASET STATISTICS ---")
print(f"fire_dx: mean={np.mean(fdx):.3f}, std={np.std(fdx):.3f}, min={np.min(fdx):.3f}, max={np.max(fdx):.3f}")
print(f"fire_dy: mean={np.mean(fdy):.3f}, std={np.std(fdy):.3f}, min={np.min(fdy):.3f}, max={np.max(fdy):.3f}")
print(f"fire_distance: mean={np.mean(fdist):.3f}, std={np.std(fdist):.3f}, min={np.min(fdist):.3f}, max={np.max(fdist):.3f}")
print("Direction counts:")
for d, c in dirs.items():
    print(f"{d}: {c}")
