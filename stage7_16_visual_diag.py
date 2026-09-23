import torch
import torch.nn as nn
import numpy as np
import time
import pygame
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from wildfire.rendering.renderer import Renderer
from stage7_15_smoke_test import SeparateActorCritic, SmokeTestTrainer

def main():
    print("========================================")
    print("STAGE 7.16 — DIAGNOSTIC EVALUATION")
    print("========================================")
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCritic(device=device)
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    # Train for 10 updates to replicate user's exact state
    print("Training started (10 updates)...")
    for update in range(1, 11):
        policy.eval()
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)

    print("Training complete! Launching visual evaluation...")
    
    eval_env = WildfireEnv()
    renderer = Renderer(eval_env.map_width, eval_env.map_height)
    clock = pygame.time.Clock()
    policy.eval()
    
    # Monkey-patch renderer to draw drones LARGER so we can definitely see them
    original_draw_drones = renderer._draw_hud # Just a hook, we will draw them manually in our loop
    
    for ep in range(1): # Just 1 episode for diagnosis
        obs, _ = eval_env.reset(seed=1000 + ep)
        terminated = False
        truncated = False
        step = 0
        total_reward = 0
        
        print(f"\nEvaluating Episode {ep+1}...")
        
        while not (terminated or truncated) and step < 20: # Limit to 20 steps for console
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            renderer.render_world(eval_env.world)
            
            # Draw drone manually much larger for visibility
            for d in eval_env.world.drones:
                if d.active:
                    cx = int((d.x + 0.5) * renderer.cell_size)
                    cy = int((d.y + 0.5) * renderer.cell_size)
                    color = (0, 255, 0) if d.id == eval_env.controlled_drone_idx else (200, 0, 0)
                    pygame.draw.circle(renderer.screen, color, (cx, cy), 10) # GIANT RADIUS 10
            pygame.display.flip()
            
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
            
            # Print diagnostic
            action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
            print(f"Step {step+1:3d} | Action: {action_names[action]:<9} | Pos: ({pre_x:2d}, {pre_y:2d}) -> ({post_x:2d}, {post_y:2d}) | Bat: {pre_bat:3d} -> {post_bat:3d} | Active: {pre_active} -> {post_active}")
            
            total_reward += reward
            step += 1
            clock.tick(15)
            
        print(f"Episode {ep+1} finished: Steps = {step}, Total Reward = {total_reward:.1f}")
        reason = info.get('termination_reason', 'max_steps' if truncated else 'unknown')
        print(f"Termination Reason: {reason}")
        
    pygame.quit()
    print("\nDiagnostic evaluation finished.")

if __name__ == "__main__":
    main()
