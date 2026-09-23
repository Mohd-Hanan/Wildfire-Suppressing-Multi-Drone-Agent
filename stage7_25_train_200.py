import torch
import numpy as np
import os
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from stage7_15_smoke_test import SeparateActorCritic, SmokeTestTrainer

def train_200_baseline():
    print("========================================")
    print("STAGE 7.25 — SINGLE-DRONE 200-UPDATE BASELINE")
    print("========================================")
    torch.manual_seed(100)
    np.random.seed(100)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    for update in range(1, 201):
        policy.eval()
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update % 10 == 0 or update == 1:
            print(f"Update {update:3d}/200 | Reward: {np.mean(collector.buffer.rewards):.4f} | Entropy: {train_metrics['entropy']:.4f} | Value Loss: {train_metrics['value_loss']:.1f}")

    # Save Checkpoint
    checkpoint_path = "stage7_single_drone_200_baseline.pth"
    torch.save({
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'update': 200,
        'seed': 100,
        'architecture': 'SeparateActorCritic',
        'drone_type': 'WATER',
        'notes': 'Single-drone baseline for future swarm transfer'
    }, checkpoint_path)
    
    print(f"\n[+] Checkpoint saved to: {os.path.abspath(checkpoint_path)}")
    return checkpoint_path

if __name__ == "__main__":
    train_200_baseline()
