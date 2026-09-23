import torch
import torch.nn as nn
import numpy as np
import time
import pygame
from torch.distributions.categorical import Categorical
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from wildfire.rendering.renderer import Renderer
from stage7_15_smoke_test import SeparateActorCritic, SmokeTestTrainer

def main():
    print("========================================")
    print("STAGE 7.16 — 10 UPDATE TRAINING + VISUAL DIAGNOSTIC")
    print("========================================")
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCritic(device=device)
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    # ---------------------------------------------------------
    # 1. TRAIN FOR 10 UPDATES
    # ---------------------------------------------------------
    print("Training started (10 updates)...")
    for update in range(1, 11):
        policy.eval()
        collector.collect()
        
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
            
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        print(f"Update {update:2d}/10 | Rollout Reward: {np.mean(collector.buffer.rewards):.4f} | Entropy: {train_metrics['entropy']:.4f} | Value Loss: {train_metrics['value_loss']:.1f}")

    print("Training complete! Launching visual evaluation...")
    
    # ---------------------------------------------------------
    # 2. VISUAL EVALUATION WITH DIAGNOSTICS
    # ---------------------------------------------------------
    eval_env = WildfireEnv()
    renderer = Renderer(eval_env.map_width, eval_env.map_height)
    clock = pygame.time.Clock()
    
    policy.eval()
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    for ep in range(3):
        obs, _ = eval_env.reset(seed=1000 + ep)
        terminated = False
        truncated = False
        step = 0
        total_reward = 0
        
        print(f"\nEvaluating Episode {ep+1}...")
        
        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            renderer.render_world(eval_env.world)
            
            with torch.no_grad():
                obs_t = {
                    "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                    "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                    "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
                }
                logits, _ = policy(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                
            controlled_drone = eval_env.world.drones[eval_env.controlled_drone_idx]
            pre_x, pre_y = controlled_drone.x, controlled_drone.y
            pre_bat = controlled_drone.battery
            pre_active = controlled_drone.active
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            post_x, post_y = controlled_drone.x, controlled_drone.y
            post_bat = controlled_drone.battery
            post_active = controlled_drone.active
            
            total_reward += reward
            step += 1
            
            # Print diagnostic for first 5 steps and last step of each episode to avoid spam
            if step <= 5 or terminated or truncated:
                print(f"Step {step:3d} | Action: {action_names[action]:<9} | Pos: ({pre_x:2d}, {pre_y:2d}) -> ({post_x:2d}, {post_y:2d}) | Bat: {pre_bat:3d} -> {post_bat:3d} | Active: {pre_active} -> {post_active}")
                
            clock.tick(15)
            
        reason = info.get('termination_reason', 'unknown')
        if truncated and reason == 'unknown': reason = 'max_steps'
        print(f"Episode {ep+1} finished: Steps = {step}, Total Reward = {total_reward:.1f}")
        print(f"Termination Reason: {reason}")
        pygame.time.delay(1000)
        
    pygame.quit()
    print("\nVisual evaluation finished.")

if __name__ == "__main__":
    main()
