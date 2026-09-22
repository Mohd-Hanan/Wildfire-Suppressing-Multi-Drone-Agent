import torch
import numpy as np
import gymnasium as gym

from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.ppo_trainer import PPOTrainer

class WildfireStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_burned_cells = []
        self.episode_extinguished = []
        self.episode_crashes = []
        
        self.current_reward = 0.0
        self.current_length = 0
        self.current_burned = 0
        self.current_crashes = 0
        
    def reset(self, **kwargs):
        self.current_reward = 0.0
        self.current_length = 0
        self.current_burned = 0
        self.current_crashes = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_reward += reward
        self.current_length += 1
        self.current_burned += info.get('new_burned_cells', 0)
        
        if info.get('crash_penalty', 0.0) < 0.0:
            self.current_crashes += 1
            
        if terminated or truncated:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_burned_cells.append(self.current_burned)
            self.episode_crashes.append(self.current_crashes)
            
            extinguished = (info.get('termination_reason') == 'fire_extinguished')
            self.episode_extinguished.append(extinguished)
            
        return obs, reward, terminated, truncated, info
        
    def get_and_clear_stats(self):
        stats = {
            'episodes_completed': len(self.episode_rewards),
            'completed_mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else None,
            'completed_mean_length': np.mean(self.episode_lengths) if self.episode_lengths else None,
            'completed_mean_burned': np.mean(self.episode_burned_cells) if self.episode_burned_cells else None,
            'completed_mean_crashes': np.mean(self.episode_crashes) if self.episode_crashes else None,
            'extinguished_count': sum(self.episode_extinguished)
        }
        
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_burned_cells.clear()
        self.episode_extinguished.clear()
        self.episode_crashes.clear()
        
        return stats

def _check_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Tensor {name} contains NaN or Inf.")

def train_ppo(
    updates=20,
    rollout_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    ppo_epochs=4,
    minibatch_size=64,
    seed=42,
    device_name="cpu"
):
    device = torch.device(device_name)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    base_env = WildfireEnv()
    env = WildfireStatsWrapper(base_env)
    
    policy = ActorCritic(device=device)
    collector = RolloutCollector(env, policy, rollout_size, device)
    trainer = PPOTrainer(
        policy, 
        device=device,
        learning_rate=learning_rate,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size
    )
    
    initial_params = [p.clone() for p in policy.parameters()]
    param_changed = False
    
    total_steps = 0
    total_completed_episodes = 0
    total_extinguished = 0
    
    print("Training started...")
    
    for update in range(1, updates + 1):
        collector.collect()
        
        spatial, drone, wind, actions, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        
        _check_finite(rewards, "rewards")
        _check_finite(values, "values")
        _check_finite(old_log_probs, "old_log_probs")
        
        last_value = collector.last_value
        
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terms,
            truncated=truncs,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        
        _check_finite(advantages, "advantages")
        _check_finite(returns, "returns")
            
        metrics = trainer.update(
            spatial, drone, wind, actions, old_log_probs, advantages, returns
        )
        
        if not param_changed:
            current_params = list(policy.parameters())
            for old_p, new_p in zip(initial_params, current_params):
                if not torch.equal(old_p, new_p):
                    param_changed = True
                    break
            if not param_changed:
                raise RuntimeError("Policy parameters did not change after first update!")
                
        for k, v in metrics.items():
            if not np.isfinite(v):
                raise ValueError(f"Metric {k} is non-finite: {v}")
                
        total_steps += rollout_size
        
        ep_stats = env.get_and_clear_stats()
        total_completed_episodes += ep_stats['episodes_completed']
        total_extinguished += ep_stats['extinguished_count']
        
        # Add rollout reward stats to metrics
        metrics['rollout_reward_mean'] = rewards.mean().item()
        metrics['rollout_reward_min'] = rewards.min().item()
        metrics['rollout_reward_max'] = rewards.max().item()
        
        metrics['advantage_mean'] = advantages.mean().item()
        metrics['advantage_std'] = advantages.std().item()
        metrics['advantage_min'] = advantages.min().item()
        metrics['advantage_max'] = advantages.max().item()
        
        metrics['return_mean'] = returns.mean().item()
        metrics['return_std'] = returns.std().item()
        metrics['return_min'] = returns.min().item()
        metrics['return_max'] = returns.max().item()
        
        print(f"========================================")
        print(f"Update: {update}/{updates}")
        print(f"Environment steps: {total_steps}")
        
        print(f"\n--- Rollout Metrics ---")
        print(f"Rollout reward mean: {metrics['rollout_reward_mean']:.4f}")
        print(f"Rollout reward min/max: [{metrics['rollout_reward_min']:.4f}, {metrics['rollout_reward_max']:.4f}]")
        print(f"Advantage mean/std: {metrics['advantage_mean']:.4f} / {metrics['advantage_std']:.4f}")
        print(f"Advantage min/max: [{metrics['advantage_min']:.4f}, {metrics['advantage_max']:.4f}]")
        print(f"Return mean/std: {metrics['return_mean']:.4f} / {metrics['return_std']:.4f}")
        print(f"Return min/max: [{metrics['return_min']:.4f}, {metrics['return_max']:.4f}]")
        
        print(f"\n--- Completed Episode Metrics ---")
        print(f"Episodes completed this rollout: {ep_stats['episodes_completed']}")
        if ep_stats['episodes_completed'] > 0:
            print(f"Completed episode reward mean: {ep_stats['completed_mean_reward']:.2f}")
            print(f"Completed episode length mean: {ep_stats['completed_mean_length']:.1f}")
            print(f"Extinguished count: {ep_stats['extinguished_count']}")
            print(f"Completed mean burned cells: {ep_stats['completed_mean_burned']:.1f}")
            print(f"Completed mean crashes: {ep_stats['completed_mean_crashes']:.1f}")
        else:
            print("No episodes completed this rollout.")
            
        print(f"\n--- Optimization Metrics ---")
        print(f"Policy loss: {metrics.get('policy_loss', 0.0):.4f}")
        print(f"Value loss: {metrics.get('value_loss', 0.0):.4f}")
        print(f"Entropy: {metrics.get('entropy', 0.0):.4f}")
        print(f"Approx KL: {metrics.get('approx_kl', 0.0):.4f}")
        print(f"Clip fraction: {metrics.get('clip_fraction', 0.0):.4f}")
        print(f"Ratio mean: {metrics.get('ratio_mean', 0.0):.4f}")
        print(f"Gradient norm: {metrics.get('gradient_norm', 0.0):.4f}")
        
    print("========================================")
    print("Training completed.")
    print(f"Total environment steps: {total_steps}")
    print(f"Total completed episodes: {total_completed_episodes}")
    print(f"Total extinguished episodes: {total_extinguished}")
    
    return policy, metrics, total_steps, ep_stats

if __name__ == "__main__":
    train_ppo()
