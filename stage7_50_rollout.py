import torch
import numpy as np
import csv
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_35_final_truth import SeparateActorCriticSymmetric64
import os

# Set up policy
device = torch.device('cpu')
policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER").to(device)
policy.load_state_dict(torch.load("stage7_49_water_radius_20updates.pth", map_location=device, weights_only=True))
policy.eval()

# Set up environment
env = WildfireEnv("configs/environment.yaml")

with open("stage7_50_trajectory_log.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed', 'step', 'drone_x', 'drone_y', 'nearest_fire_x', 'nearest_fire_y', 'fire_dx', 'fire_dy', 'fire_distance', 'action', 'action_prob', 'battery', 'payload', 'reward', 'suppressed', 'fire_in_window'])
    
    for seed in [100, 101, 102, 103, 104]:
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        while not done:
            # Format obs
            spatial = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0)
            drone = torch.tensor(obs['drone'], dtype=torch.float32).unsqueeze(0)
            wind = torch.tensor(obs['wind'], dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, _ = policy({'spatial': spatial, 'drone': drone, 'wind': wind})
                action_logits[0, 6] = -1e9
                probs = torch.softmax(action_logits, dim=-1)[0].numpy()
            
            action = np.random.choice(7, p=probs)
            prob = probs[action]
            
            drone_obj = env.world.drones[0]
            dx = drone[0, 6].item()
            dy = drone[0, 7].item()
            fdist = drone[0, 8].item() * np.sqrt(48**2 + 48**2)
            nx = int(dx * 48 + drone_obj.x)
            ny = int(dy * 48 + drone_obj.y)
            
            in_window = (abs(nx - drone_obj.x) <= 5 and abs(ny - drone_obj.y) <= 5)
            
            next_obs, reward, term, trunc, info = env.step(action)
            
            writer.writerow([seed, step, drone_obj.x, drone_obj.y, nx, ny, dx, dy, fdist, action, prob, drone_obj.battery, drone_obj.payload, reward, info.get('newly_suppressed_cells', 0), in_window])
            
            obs = next_obs
            step += 1
            done = term or trunc
print("Done writing rollout.")
