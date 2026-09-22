import torch
import numpy as np
from typing import Dict, List, Optional
from wildfire.environment.wildfire_env import WildfireEnv

def evaluate_episode(env: WildfireEnv, seed: int, policy=None, device=None) -> Dict:
    obs, info = env.reset(seed=seed)
    
    total_reward = 0.0
    steps = 0
    total_burned = 0
    crashes = 0
    
    if policy is not None:
        policy.eval()
        
    terminated = False
    truncated = False
    
    drone = env.world.drones[env.controlled_drone_idx]
    
    # State tracking
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    action_changes = 0
    last_action = -1
    
    initial_battery = drone.battery
    initial_payload = drone.payload
    initial_pos = (drone.x, drone.y)
    
    min_battery = drone.battery
    deploy_attempts = 0
    successful_water = 0
    successful_retardant = 0
    failed_deployments = 0
    base_visits = 0
    boundary_pushes = 0
    
    was_active = drone.active
    crash_cause = None
    
    while not (terminated or truncated):
        if env.world.is_at_base(drone):
            base_visits += 1
            
        if policy is not None:
            with torch.no_grad():
                obs_t = {
                    "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                    "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                    "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
                }
                logits, _ = policy(obs_t)
                action = torch.argmax(logits, dim=-1).item()
        else:
            action = env.action_space.sample()
            
        action_counts[action] += 1
        if last_action != -1 and action != last_action:
            action_changes += 1
        last_action = action
        
        # Track pre-action state to detect failures
        pre_x, pre_y = drone.x, drone.y
        pre_payload = drone.payload
        
        if action in (5, 6):
            deploy_attempts += 1
            
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Determine movement boundary push
        if action in (1, 2, 3, 4):
            if drone.x == pre_x and drone.y == pre_y and drone.active:
                boundary_pushes += 1
                
        # Determine deployment success
        if action == 5:
            if drone.payload < pre_payload:
                successful_water += 1
            else:
                failed_deployments += 1
        elif action == 6:
            if drone.payload < pre_payload:
                successful_retardant += 1
            else:
                failed_deployments += 1
        
        # Crash tracking
        if was_active and not drone.active:
            crashes += 1
            if drone.battery <= 0:
                crash_cause = "Battery depleted"
            else:
                crash_cause = "Unknown inactive transition"
        was_active = drone.active
        
        if drone.battery < min_battery:
            min_battery = drone.battery
            
        total_reward += reward
        steps += 1
        total_burned += step_info.get("new_burned_cells", 0)
        
        obs = next_obs
        
    return {
        "seed": seed,
        "reward": total_reward,
        "length": steps,
        "burned_cells": total_burned,
        "extinguished": step_info.get("termination_reason") == "fire_extinguished",
        "crashes": crashes,
        "crash_cause": crash_cause,
        "termination_reason": step_info.get("termination_reason", "max_steps"),
        
        # Action metrics
        "action_counts": action_counts,
        "action_changes": action_changes,
        
        # Resource metrics
        "initial_battery": initial_battery,
        "initial_payload": initial_payload,
        "initial_pos": initial_pos,
        "final_battery": drone.battery,
        "final_payload": drone.payload,
        "final_pos": (drone.x, drone.y),
        "min_battery": min_battery,
        
        "deploy_attempts": deploy_attempts,
        "successful_water": successful_water,
        "successful_retardant": successful_retardant,
        "failed_deployments": failed_deployments,
        "base_visits": base_visits,
        "boundary_pushes": boundary_pushes,
        "final_active": drone.active
    }

def evaluate_policy(env: WildfireEnv, seeds: List[int], policy=None, device=None) -> Dict:
    results = []
    
    for seed in seeds:
        res = evaluate_episode(env, seed, policy, device)
        results.append(res)
        
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    burned = [r["burned_cells"] for r in results]
    extinguished_count = sum([1 for r in results if r["extinguished"]])
    crashes = sum([r["crashes"] for r in results])
    
    # Aggregate action counts
    total_actions = sum(lengths)
    agg_action_counts = {i: 0 for i in range(7)}
    for r in results:
        for i in range(7):
            agg_action_counts[i] += r["action_counts"][i]
            
    action_pct = {i: (agg_action_counts[i] / max(1, total_actions)) * 100 for i in range(7)}
    
    # Crash causes
    crash_causes = {}
    for r in results:
        if r["crashes"] > 0:
            c = r["crash_cause"]
            crash_causes[c] = crash_causes.get(c, 0) + 1
            
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_burned": float(np.mean(burned)),
        "std_burned": float(np.std(burned)),
        "extinguished_count": int(extinguished_count),
        "extinction_rate": float(extinguished_count / len(seeds)),
        "total_crashes": crashes,
        "crash_causes": crash_causes,
        
        "action_counts": agg_action_counts,
        "action_pct": action_pct,
        "mean_action_changes": float(np.mean([r["action_changes"] for r in results])),
        "most_common_action": max(agg_action_counts.items(), key=lambda x: x[1])[0],
        
        "mean_final_battery": float(np.mean([r["final_battery"] for r in results])),
        "mean_min_battery": float(np.mean([r["min_battery"] for r in results])),
        "mean_final_payload": float(np.mean([r["final_payload"] for r in results])),
        "mean_deploy_attempts": float(np.mean([r["deploy_attempts"] for r in results])),
        "mean_successful_water": float(np.mean([r["successful_water"] for r in results])),
        "mean_successful_retardant": float(np.mean([r["successful_retardant"] for r in results])),
        "mean_failed_deployments": float(np.mean([r["failed_deployments"] for r in results])),
        "mean_base_visits": float(np.mean([r["base_visits"] for r in results])),
        "mean_boundary_pushes": float(np.mean([r["boundary_pushes"] for r in results])),
        
        "episodes": results
    }
