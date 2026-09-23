import torch
import numpy as np
from stage7_52_network import ActorCritic6Channels
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.rollout import RolloutCollector
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.gae import compute_gae
from stage7_52_obs_builder import GlobalFireObservationBuilder

def train():
    print("========================================")
    print("STAGE 7.52: GLOBAL FIRE CHANNEL - 200 UPDATES")
    print("========================================")
    
    device = torch.device('cpu')
    policy = ActorCritic6Channels(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_52_global_fire_channel_20updates.pth", map_location=device))
    policy.train()
    
    trainer = PPOTrainer(
        policy=policy,
        device=device,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        minibatch_size=64
    )
    
    env = WildfireEnv("configs/environment.yaml")
    # Replace observation builder
    env.obs_builder = GlobalFireObservationBuilder(window_size=11)
    
    collector = RolloutCollector(env, policy, rollout_size=256, device=device)
    
    for update in range(21, 201):
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        
        water_freq = (acts == 5).float().mean().item() * 100
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update % 10 == 0 or update == 21:
            print(f"Update {update:3d}/200 | Reward: {np.mean(collector.buffer.rewards):7.4f} | "
                  f"Value Loss: {train_metrics.get('value_loss', 0.0):7.4f} | "
                  f"Policy Loss: {train_metrics.get('policy_loss', 0.0):7.4f} | "
                  f"Entropy: {train_metrics.get('entropy', 0.0):5.4f} | "
                  f"WATER%: {water_freq:5.1f}%")

    checkpoint_path = "stage7_52_global_fire_channel_200updates.pth"
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"Saved {checkpoint_path}")
    print("Stage 7.52 total updates = 200")
    print("checkpoint saved successfully")
    return policy

if __name__ == "__main__":
    train()
