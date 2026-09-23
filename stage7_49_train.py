import torch
import numpy as np
from stage7_35_final_truth import SeparateActorCriticSymmetric64
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.rollout import RolloutCollector
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.gae import compute_gae

def train_ppo():
    env = WildfireEnv("configs/environment.yaml")
    device = torch.device("cpu")
    
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
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
    
    collector = RolloutCollector(
        env=env,
        policy=policy,
        rollout_size=256,
        device=device
    )
    
    updates = 20
    
    for update in range(updates):
        collector.collect()
        
        spatial, drone_vec, wind, actions, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        last_value = collector.last_value
        
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terms,
            truncated=truncs,
            last_value=last_value,
            gamma=0.99,
            gae_lambda=0.95
        )
        
        metrics = trainer.update(spatial, drone_vec, wind, actions, old_log_probs, advantages, returns)
        print(f"Update {update+1}/{updates} | Value Loss: {metrics['value_loss']:.4f} | Policy Loss: {metrics['policy_loss']:.4f} | Entropy: {metrics['entropy']:.4f}")
        
    torch.save(policy.state_dict(), "stage7_49_water_radius_20updates.pth")
    print("Saved stage7_49_water_radius_20updates.pth")

if __name__ == "__main__":
    train_ppo()
