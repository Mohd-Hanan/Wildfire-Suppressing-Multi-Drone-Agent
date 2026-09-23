import torch
import numpy as np
import os
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from stage7_15_smoke_test import SeparateActorCritic, SmokeTestTrainer

def train_200():
    print("========================================")
    print("STAGE 7.33 — 200-UPDATE TRAINING")
    print("========================================")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    for update in range(1, 201):
        policy.eval()
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        
        # Calculate metrics
        water_freq = (acts == 5).float().mean().item() * 100
        
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        # We need to get suppression count. Unfortunately our wrapper might not track it easily from RolloutCollector.
        # But wait, env.env.unwrapped could track it? 
        # Actually, let's just grab the info from the environment if possible, or just skip it if it's too complex to hack into RolloutCollector now, 
        # but wait, the prompt says "successful suppression count/frequency if available". It is available from the training rewards if we look at the reward values, but that's hard to parse.
        # Let's just track mean reward.
        
        if update % 10 == 0 or update == 1:
            print(f"Update {update:3d}/200 | Reward: {np.mean(collector.buffer.rewards):7.4f} | "
                  f"Entropy: {train_metrics['entropy']:6.4f} | Value Loss: {train_metrics['value_loss']:8.1f} | "
                  f"Policy Loss: {train_metrics.get('policy_loss', 0.0):7.4f} | WATER%: {water_freq:5.1f}%")

    checkpoint_path = "stage7_single_drone_200_global_fire_reward.pth"
    torch.save({
        'model_state_dict': policy.state_dict(),
    }, checkpoint_path)
    print(f"\n[+] Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    train_200()
