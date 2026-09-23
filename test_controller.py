import numpy as np
import yaml
from wildfire.environment.wildfire_env import WildfireEnv

env = WildfireEnv("configs/environment.yaml")
env.reset(seed=3000)
drone = env.world.drones[0]
done = False
step = 0
while not done and step < 500:
    req_bat = env.world.required_battery(drone)
    
    if drone.battery <= req_bat + 1 or drone.payload == 0:
        target_x, target_y = 2, 2
    else:
        burning = (env.world.fire_manager.fire_map == 2)
        if not np.any(burning):
            target_x, target_y = drone.x, drone.y
        else:
            y_idx, x_idx = np.where(burning)
            dists = np.abs(x_idx - drone.x) + np.abs(y_idx - drone.y)
            best = np.argmin(dists)
            target_x, target_y = x_idx[best], y_idx[best]
            
    if drone.x == target_x and drone.y == target_y:
        if (target_x, target_y) != (2,2) and env.world.fire_manager.fire_map[target_x, target_y] == 2:
            action = 5
        else:
            action = 0
    else:
        if np.abs(target_x - drone.x) > np.abs(target_y - drone.y):
            if target_x > drone.x: action = 3
            else: action = 4
        else:
            if target_y > drone.y: action = 2
            else: action = 1
            
    _, _, terminated, truncated, info = env.step(action)
    print(f"Step {step}: pos=({drone.x},{drone.y}) target=({target_x},{target_y}) bat={drone.battery} req={req_bat} action={action} supp={info.get('newly_suppressed', 0)}")
    done = terminated or truncated
    step += 1
    if not drone.active:
        print("Crashed!")
        break
