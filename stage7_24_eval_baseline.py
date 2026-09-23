import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_15_smoke_test import SeparateActorCritic

def run_evaluation(checkpoint_path, num_episodes=20):
    print("========================================")
    print(f"QUANTITATIVE EVALUATION: {checkpoint_path}")
    print("========================================")
    
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    env = WildfireEnv()
    
    metrics = {
        'rewards': [],
        'burned_cells': [],
        'extinctions': 0,
        'lengths': [],
        'crashes': 0,
        'water_drops': 0,
        'retardant_drops': 0,
        'action_counts': np.zeros(7)
    }
    
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    for ep in range(num_episodes):
        # Use stochastic sampling for the evaluation to match training and visual watch
        obs, _ = env.reset(seed=5000 + ep)
        
        ep_reward = 0
        ep_length = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            with torch.no_grad():
                obs_t = {
                    "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                    "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                    "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
                }
                logits, _ = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                
            metrics['action_counts'][action] += 1
            if action == 5:
                metrics['water_drops'] += 1
            elif action == 6:
                metrics['retardant_drops'] += 1
                
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            
        drone = env.world.drones[env.controlled_drone_idx]
        crashed = not drone.active and ep_length < 500 and not info.get('extinguished', False)
        if crashed:
            metrics['crashes'] += 1
            
        metrics['rewards'].append(ep_reward)
        metrics['burned_cells'].append(info.get('burned_cells', 0))
        metrics['lengths'].append(ep_length)
        if info.get('extinguished', False):
            metrics['extinctions'] += 1
            
        print(f"Ep {ep+1:2d} | R: {ep_reward:7.1f} | Len: {ep_length:3d} | Burn: {info.get('burned_cells', 0):4d} | Ext: {info.get('extinguished', False)} | Crash: {crashed}")
        
    print("\n========================================")
    print("SUMMARY METRICS (20 EPISODES)")
    print("========================================")
    print(f"Mean Reward:        {np.mean(metrics['rewards']):.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"Mean Burned Cells:  {np.mean(metrics['burned_cells']):.2f} ± {np.std(metrics['burned_cells']):.2f}")
    print(f"Extinction Rate:    {(metrics['extinctions']/num_episodes)*100:.1f}%")
    print(f"Mean Ep Length:     {np.mean(metrics['lengths']):.1f}")
    print(f"Total Crashes:      {metrics['crashes']}")
    print(f"Total Water Drops:  {metrics['water_drops']}")
    print(f"Total Retardant:    {metrics['retardant_drops']}")
    
    print("\nAction Percentages:")
    total_actions = np.sum(metrics['action_counts'])
    for i, count in enumerate(metrics['action_counts']):
        pct = (count / total_actions) * 100
        print(f"{action_names[i]:<10}: {pct:5.1f}%")

if __name__ == "__main__":
    run_evaluation("stage7_single_drone_200_baseline.pth", 20)
