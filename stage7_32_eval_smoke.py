import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_15_smoke_test import SeparateActorCritic
from collections import Counter

def eval_smoke():
    print("========================================")
    print("QUANTITATIVE EVALUATION: stage7_32_smoke_global_fire_reward.pth")
    print("========================================")
    
    env = WildfireEnv()
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    checkpoint = torch.load("stage7_32_smoke_global_fire_reward.pth", map_location=device, weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    metrics = {
        'rewards': [], 'burned_cells': [], 'suppressed_cells': [], 
        'lengths': [], 'crashes': 0, 'extinctions': 0,
        'base_visits': 0, 'actions': Counter()
    }
    
    # Run 20 evaluation episodes
    for ep in range(1, 21):
        obs, _ = env.reset(seed=ep * 100)
        ep_reward = 0.0
        ep_length = 0
        ep_burned = 0
        ep_suppressed = 0
        drone = env.world.drones[env.controlled_drone_idx]
        was_at_base = True
        
        while True:
            with torch.no_grad():
                obs_t = {
                    "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                    "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                    "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
                }
                logits, _ = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                
            metrics['actions'][action] += 1
            
            # Check base visits
            is_at_base = (env.world.base_x <= drone.x < env.world.base_x + 2 and 
                          env.world.base_y <= drone.y < env.world.base_y + 2)
            if is_at_base and not was_at_base:
                metrics['base_visits'] += 1
            was_at_base = is_at_base
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_burned += info.get('new_burned_cells', 0)
            ep_suppressed += info.get('newly_suppressed_cells', 0)
            ep_length += 1
            
            crashed = not drone.active and not info.get('extinguished', False)
            if terminated or truncated or crashed:
                if crashed:
                    metrics['crashes'] += 1
                if info.get('extinction_reward', 0) > 0:
                    metrics['extinctions'] += 1
                
                metrics['rewards'].append(ep_reward)
                metrics['burned_cells'].append(ep_burned)
                metrics['suppressed_cells'].append(ep_suppressed)
                metrics['lengths'].append(ep_length)
                
                print(f"Ep {ep:2d} | R: {ep_reward:7.1f} | Len: {ep_length:3d} | Burn: {ep_burned:4d} | Supp: {ep_suppressed:2d} | Ext: {info.get('extinction_reward', 0) > 0} | Crash: {crashed}")
                break

    print("\n========================================")
    print("SUMMARY METRICS (20 EPISODES)")
    print("========================================")
    print(f"Mean Reward:        {np.mean(metrics['rewards']):.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"Mean Burned Cells:  {np.mean(metrics['burned_cells']):.2f} ± {np.std(metrics['burned_cells']):.2f}")
    print(f"Mean Suppressed:    {np.mean(metrics['suppressed_cells']):.2f} ± {np.std(metrics['suppressed_cells']):.2f}")
    print(f"Total Successful Suppressions: {sum(metrics['suppressed_cells'])}")
    print(f"Total WATER Actions: {metrics['actions'][5]}")
    print(f"Extinction Rate:    {(metrics['extinctions'] / 20) * 100:.1f}%")
    print(f"Mean Ep Length:     {np.mean(metrics['lengths']):.1f}")
    print(f"Total Crashes:      {metrics['crashes']}")
    print(f"Total Base Visits:  {metrics['base_visits']}")
    
    total_actions = sum(metrics['actions'].values())
    print("\nAction Percentages:")
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    for i in range(7):
        pct = (metrics['actions'][i] / total_actions) * 100
        print(f"{action_names[i]:<10}: {pct:4.1f}%")

if __name__ == "__main__":
    eval_smoke()
