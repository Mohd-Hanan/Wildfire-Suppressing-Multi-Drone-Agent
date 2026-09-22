import numpy as np
import time
from wildfire.environment.wildfire_env import WildfireEnv

def validate():
    env = WildfireEnv()
    
    # 13. Test the full action space deterministically
    print("--- DETERMINISTIC ACTION SMOKE TEST ---")
    env.reset()
    for a in range(7):
        obs, r, t, tr, info = env.step(a)
        assert env.observation_space.contains(obs), f"Action {a} produced invalid obs."
    print("All 7 actions executed successfully and produced valid obs.\n")

    # 12. Reproducibility check
    print("--- REPRODUCIBILITY CHECK ---")
    env1 = WildfireEnv()
    env2 = WildfireEnv()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    # Compare spatial, drone, and wind
    spatial_match = np.array_equal(obs1['spatial'], obs2['spatial'])
    drone_match = np.array_equal(obs1['drone'], obs2['drone'])
    wind_match = np.array_equal(obs1['wind'], obs2['wind'])
    if spatial_match and drone_match and wind_match:
        seed_status = "PASS"
        print("Deterministic seed check: PASS")
    else:
        seed_status = "FAIL"
        print("Deterministic seed check: FAIL")
    print()

    # 1-11, 14. Random Agent Validation
    print("--- RANDOM AGENT VALIDATION ---")
    episodes = 20
    total_steps = 0
    exceptions = 0
    invalid_obs = 0
    nan_inf_obs = 0
    neg_battery = 0
    neg_payload = 0
    oob_drone = 0
    invalid_terms = 0
    
    termination_reasons = {'max_steps': 0, 'fire_extinguished': 0}
    
    episode_lengths = []
    episode_rewards = []
    
    min_step_reward = float('inf')
    max_step_reward = float('-inf')
    
    start_time = time.time()
    
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            ep_reward = 0
            ep_steps = 0
            
            while True:
                # Validate obs
                if not env.observation_space.contains(obs):
                    invalid_obs += 1
                for k in ['spatial', 'drone', 'wind']:
                    if np.isnan(obs[k]).any() or not np.isfinite(obs[k]).all():
                        nan_inf_obs += 1
                        
                # Validate bounds & resources directly from drone
                drone = env.world.drones[env.controlled_drone_idx]
                if drone.battery < 0: neg_battery += 1
                if drone.payload < 0: neg_payload += 1
                if drone.x < 0 or drone.x >= env.map_width or drone.y < 0 or drone.y >= env.map_height:
                    oob_drone += 1
                    
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                ep_reward += float(reward)
                ep_steps += 1
                total_steps += 1
                
                if reward < min_step_reward: min_step_reward = float(reward)
                if reward > max_step_reward: max_step_reward = float(reward)
                
                # Termination sanity
                if terminated and truncated:
                    invalid_terms += 1
                
                if terminated or truncated:
                    if terminated and info.get('termination_reason') != 'fire_extinguished':
                        invalid_terms += 1
                    if truncated and info.get('termination_reason') != 'max_steps':
                        invalid_terms += 1
                    
                    reason = info.get('termination_reason')
                    if reason in termination_reasons:
                        termination_reasons[reason] += 1
                        
                    episode_lengths.append(ep_steps)
                    episode_rewards.append(ep_reward)
                    break
                
                # Infinite loop guard (max_steps is 500, shouldn't hit this)
                if ep_steps > 1000:
                    invalid_terms += 1
                    break
    except Exception as e:
        exceptions += 1
        print("EXCEPTION ENCOUNTERED:", e)
        
    duration = time.time() - start_time
    
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    
    print(f"Validation Duration: {duration:.2f}s")
    print(f"Episodes completed: {len(episode_lengths)}/{episodes}")
    print(f"Total env steps: {total_steps}")
    print(f"Exceptions: {exceptions}")
    print(f"Invalid obs: {invalid_obs}")
    print(f"NaN/Inf obs: {nan_inf_obs}")
    print(f"Negative battery: {neg_battery}")
    print(f"Negative payload: {neg_payload}")
    print(f"Out of bounds drones: {oob_drone}")
    print(f"Invalid termination combos: {invalid_terms}")
    print(f"Termination reasons: {termination_reasons}")
    print(f"Average ep length: {avg_length:.1f}")
    print(f"Average total reward: {avg_reward:.1f}")
    print(f"Min step reward: {min_step_reward:.2f}")
    print(f"Max step reward: {max_step_reward:.2f}")
    
if __name__ == '__main__':
    validate()
