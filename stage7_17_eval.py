import torch
import numpy as np
from collections import defaultdict
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_15_smoke_test import SeparateActorCritic

def evaluate(policy, env, num_episodes=20, deterministic=False):
    device = policy.device
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    metrics = {
        "reward": [],
        "burned_cells": [],
        "extinguished": [],
        "length": [],
        "crashes": [],
        "water_deployments": 0,
        "retardant_deployments": 0,
        "base_visits": 0,
        "actions": {a: 0 for a in range(7)},
        "final_battery": [],
        "final_payload": []
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        terminated = False
        truncated = False
        step = 0
        ep_reward = 0
        
        while not (terminated or truncated):
            controlled_drone = env.world.drones[env.controlled_drone_idx]
            pre_x, pre_y = controlled_drone.x, controlled_drone.y
            pre_bat, pre_pay = controlled_drone.battery, controlled_drone.payload
            
            with torch.no_grad():
                obs_t = {
                    "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                    "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                    "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
                }
                if deterministic:
                    logits, _ = policy(obs_t)
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    action, _, _, _ = policy.get_action_and_value(obs_t)
                    action = action.item()
                    
            metrics["actions"][action] += 1
            if action == 5: metrics["water_deployments"] += 1
            if action == 6: metrics["retardant_deployments"] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            
            post_x, post_y = controlled_drone.x, controlled_drone.y
            post_bat, post_pay = controlled_drone.battery, controlled_drone.payload
            
            # Check for base visits (refill)
            if post_bat > pre_bat or post_pay > pre_pay:
                metrics["base_visits"] += 1
                
        # Episode end stats
        metrics["reward"].append(ep_reward)
        burned = np.sum(env.world.fire_manager.fire_map == 1.0)
        metrics["burned_cells"].append(burned)
        
        reason = info.get('termination_reason', 'max_steps' if truncated else 'unknown')
        if reason == 'fire_extinguished':
            metrics["extinguished"].append(1)
        else:
            metrics["extinguished"].append(0)
            
        metrics["length"].append(step)
        
        final_drone = env.world.drones[env.controlled_drone_idx]
        metrics["final_battery"].append(final_drone.battery)
        metrics["final_payload"].append(final_drone.payload)
        
        if not final_drone.active:
            metrics["crashes"].append(1)
        else:
            metrics["crashes"].append(0)
            
    return metrics

def print_metrics(name, metrics, num_episodes):
    print(f"\n--- {name} ---")
    print(f"Mean Reward: {np.mean(metrics['reward']):.2f} +/- {np.std(metrics['reward']):.2f}")
    print(f"Mean Burned Cells: {np.mean(metrics['burned_cells']):.2f} +/- {np.std(metrics['burned_cells']):.2f}")
    print(f"Extinction Rate: {np.mean(metrics['extinguished'])*100:.1f}%")
    print(f"Mean Episode Length: {np.mean(metrics['length']):.1f}")
    print(f"Crashes: {np.sum(metrics['crashes'])}")
    print(f"Water Deployments: {metrics['water_deployments']}")
    print(f"Retardant Deployments: {metrics['retardant_deployments']}")
    print(f"Base Visits: {metrics['base_visits']}")
    
    total_actions = sum(metrics['actions'].values())
    print("\nAction Percentages:")
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    for a in range(7):
        pct = (metrics['actions'][a] / total_actions) * 100 if total_actions > 0 else 0
        print(f"  {a} ({action_names[a]}): {pct:.2f}%")
        
    print(f"\nFinal Battery (Mean): {np.mean(metrics['final_battery']):.1f}")
    print(f"Final Payload (Mean): {np.mean(metrics['final_payload']):.1f}")


def main():
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    policy.eval()
    
    env = WildfireEnv()
    
    print("Evaluating Stochastic Policy (20 episodes)...")
    stoch_metrics = evaluate(policy, env, num_episodes=20, deterministic=False)
    print_metrics("STOCHASTIC EVALUATION", stoch_metrics, 20)
    
    print("\nEvaluating Deterministic Policy (20 episodes)...")
    det_metrics = evaluate(policy, env, num_episodes=20, deterministic=True)
    print_metrics("DETERMINISTIC EVALUATION", det_metrics, 20)
    
    # Specific verification test
    print("\n--- SPECIFIC VERIFICATION CHECKS ---")
    obs, _ = env.reset(seed=999)
    drone = env.world.drones[env.controlled_drone_idx]
    
    # Force Movement (Action 4 - WEST) so we are outside the base
    pre_x = drone.x
    env.step(4)
    post_x = drone.x
    print(f"4. Movement works (WEST): {'PASS' if post_x < pre_x else 'FAIL'} (x: {pre_x} -> {post_x})")
    
    pre_bat = drone.battery
    pre_pay = drone.payload
    
    # Force WATER (Action 5) OUTSIDE the base
    env.step(5)
    post_bat = drone.battery
    post_pay = drone.payload
    
    print(f"1. WATER deployment decreases payload: {'PASS' if post_pay < pre_pay else 'FAIL'} ({pre_pay} -> {post_pay})")
    print(f"2. WATER deployment decreases battery by drop_cost ({drone.drop_cost}): {'PASS' if post_bat == pre_bat - drone.drop_cost else 'FAIL'} ({pre_bat} -> {post_bat})")
    
    # Force Stay (Action 0) at base to test refill
    obs, _ = env.reset(seed=999) # Back to base
    drone = env.world.drones[env.controlled_drone_idx]
    env.step(5) # Drop water at base to consume resources (which are instantly refilled)
    refill_bat = drone.battery
    refill_pay = drone.payload
    print(f"3. Returning/Staying at base restores resources: {'PASS' if refill_bat == drone.max_battery and refill_pay == drone.max_payload else 'FAIL'} (Bat: {refill_bat}, Pay: {refill_pay})")
    
    print(f"5. WATER drone never selects action 6: {'PASS' if stoch_metrics['actions'][6] == 0 and det_metrics['actions'][6] == 0 else 'FAIL'}")

if __name__ == "__main__":
    main()
