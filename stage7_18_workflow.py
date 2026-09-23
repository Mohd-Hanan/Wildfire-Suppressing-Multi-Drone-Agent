import torch
import numpy as np
import time
import os
import pygame
from collections import defaultdict
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from wildfire.rendering.renderer import Renderer
from stage7_15_smoke_test import SeparateActorCritic, SmokeTestTrainer
from stage7_17_eval import evaluate, print_metrics

def train_and_save():
    print("========================================")
    print("STEP 1: TRAIN 20 UPDATES")
    print("========================================")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    for update in range(1, 21):
        policy.eval()
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update % 5 == 0 or update == 20:
            print(f"Update {update:2d}/20 | Reward: {np.mean(collector.buffer.rewards):.4f} | Entropy: {train_metrics['entropy']:.4f} | Value Loss: {train_metrics['value_loss']:.1f}")

    # STEP 2: Save Checkpoint
    checkpoint_path = "stage7_18_checkpoint.pth"
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'update': 20,
        'seed': 42
    }, checkpoint_path)
    
    print(f"\n[+] Checkpoint saved to: {os.path.abspath(checkpoint_path)}")
    return checkpoint_path

def evaluate_checkpoint(checkpoint_path):
    print("\n========================================")
    print("STEP 3: EVALUATE CHECKPOINT (20 Episodes)")
    print("========================================")
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    env = WildfireEnv()
    
    print("Evaluating Stochastic Policy (20 episodes)...")
    stoch_metrics = evaluate(policy, env, num_episodes=20, deterministic=False)
    print_metrics("STOCHASTIC EVALUATION", stoch_metrics, 20)
    
    print("\nEvaluating Deterministic Policy (20 episodes)...")
    det_metrics = evaluate(policy, env, num_episodes=20, deterministic=True)
    print_metrics("DETERMINISTIC EVALUATION", det_metrics, 20)
    
def visual_evaluation(checkpoint_path):
    print("\n========================================")
    print("STEP 5: VISUAL EVALUATION (3 Episodes)")
    print("========================================")
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    eval_env = WildfireEnv()
    renderer = Renderer(eval_env.map_width, eval_env.map_height)
    clock = pygame.time.Clock()
    
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    for ep in range(3):
        obs, _ = eval_env.reset(seed=3000 + ep)
        terminated = False
        truncated = False
        step = 0
        total_reward = 0
        
        print(f"\n--- VISUAL EPISODE {ep+1} ---")
        
        while not (terminated or truncated) and step < 500:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            renderer.render_world(eval_env.world)
            pygame.display.flip()
            
            with torch.no_grad():
                obs_t = {
                    "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                    "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                    "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
                }
                logits, _ = policy(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            step += 1
            clock.tick(20)
            
        burned = np.sum(eval_env.world.fire_manager.fire_map == 1.0)
        reason = info.get('termination_reason', 'max_steps' if truncated else 'unknown')
        final_drone = eval_env.world.drones[eval_env.controlled_drone_idx]
        
        print(f"Result: Reward: {total_reward:.1f} | Length: {step} | Burned: {burned} | Reason: {reason}")
        print(f"Final State: pos=({final_drone.x},{final_drone.y}), bat={final_drone.battery}, pay={final_drone.payload}, active={final_drone.active}")
        pygame.time.delay(1000)
        
    pygame.quit()
    print("\nVisual evaluation finished.")

def main():
    checkpoint_path = train_and_save()
    evaluate_checkpoint(checkpoint_path)
    visual_evaluation(checkpoint_path)

if __name__ == "__main__":
    main()
