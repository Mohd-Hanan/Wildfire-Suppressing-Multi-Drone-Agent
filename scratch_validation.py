import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv

def validate():
    env = WildfireEnv()
    
    episodes = 20
    total_steps = 0
    terminated_count = 0
    truncated_count = 0
    fire_extinguished_count = 0
    max_steps_count = 0
    
    exceptions = 0
    invalid_obs = 0
    nan_inf_obs = 0
    neg_battery = 0
    neg_payload = 0
    oob_drone = 0
    invalid_terms = 0
    
    min_step_reward = float('inf')
    max_step_reward = float('-inf')
    
    ep_rewards = []
    
    print(f"{'Episode':<10} | {'Seed':<6} | {'Steps':<6} | {'Total Reward':<13} | {'Terminated':<11} | {'Truncated':<10} | {'Termination Reason'}")
    print("-" * 90)

    try:
        for ep in range(episodes):
            seed = 42 + ep
            obs, info = env.reset(seed=seed)
            ep_reward = 0
            ep_steps = 0
            
            while True:
                # Validations
                if not env.observation_space.contains(obs):
                    invalid_obs += 1
                for k in ['spatial', 'drone', 'wind']:
                    if np.isnan(obs[k]).any() or not np.isfinite(obs[k]).all():
                        nan_inf_obs += 1
                        
                drone = env.world.drones[env.controlled_drone_idx]
                if drone.battery < 0: neg_battery += 1
                if drone.payload < 0: neg_payload += 1
                if drone.x < 0 or drone.x >= env.map_width or drone.y < 0 or drone.y >= env.map_height:
                    oob_drone += 1
                    
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                ep_reward += float(reward)
                ep_steps += 1
                
                if reward < min_step_reward: min_step_reward = float(reward)
                if reward > max_step_reward: max_step_reward = float(reward)
                
                if terminated and truncated:
                    invalid_terms += 1
                
                if terminated or truncated:
                    reason = info.get('termination_reason')
                    
                    if terminated:
                        terminated_count += 1
                        if reason == "fire_extinguished":
                            fire_extinguished_count += 1
                        else:
                            invalid_terms += 1
                    if truncated:
                        truncated_count += 1
                        if reason == "max_steps":
                            max_steps_count += 1
                        else:
                            invalid_terms += 1
                    
                    print(f"{ep:<10} | {seed:<6} | {ep_steps:<6} | {ep_reward:<13.2f} | {str(terminated):<11} | {str(truncated):<10} | {reason}")
                    
                    total_steps += ep_steps
                    ep_rewards.append(ep_reward)
                    break
    except Exception as e:
        exceptions += 1
        print("EXCEPTION:", e)
        
    print("\n--- SUMMARY ---")
    print(f"Total steps: {total_steps}")
    print(f"Terminated: {terminated_count}")
    print(f"Truncated: {truncated_count}")
    print(f"Fire extinguished: {fire_extinguished_count}")
    print(f"Max steps: {max_steps_count}")
    print(f"Exceptions: {exceptions}")
    print(f"Invalid obs: {invalid_obs}")
    print(f"NaN/Inf: {nan_inf_obs}")
    print(f"Negative battery: {neg_battery}")
    print(f"Negative payload: {neg_payload}")
    print(f"Out-of-bounds: {oob_drone}")
    print(f"Invalid terms combos: {invalid_terms}")
    
    if ep_rewards:
        print(f"Min ep reward: {min(ep_rewards):.2f}")
        print(f"Max ep reward: {max(ep_rewards):.2f}")
        print(f"Avg ep reward: {np.mean(ep_rewards):.2f}")
    
    print(f"Min step reward: {min_step_reward:.2f}")
    print(f"Max step reward: {max_step_reward:.2f}")

if __name__ == '__main__':
    validate()
