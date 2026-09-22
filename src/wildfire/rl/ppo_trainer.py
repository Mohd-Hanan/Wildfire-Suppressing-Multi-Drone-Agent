import torch
import numpy as np
from wildfire.rl.ppo import compute_ppo_loss

class PPOTrainer:
    def __init__(
        self,
        policy,
        device=torch.device("cpu"),
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        minibatch_size=64
    ):
        self.policy = policy
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    @staticmethod
    def _normalize_advantages(advantages):
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        return (advantages - adv_mean) / (adv_std + 1e-8)
        
    def update(self, spatial, drone, wind, actions, old_log_probs, advantages, returns):
        """
        Performs multiple epochs of PPO optimization over a rollout batch.
        """
        # Advantage normalization
        normalized_advantages = self._normalize_advantages(advantages)
        
        N = len(actions)
        indices = np.arange(N)
        
        agg_metrics = {}
        updates = 0
        
        self.policy.train()
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                
                mb_spatial = spatial[mb_idx]
                mb_drone = drone[mb_idx]
                mb_wind = wind[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = normalized_advantages[mb_idx]
                mb_returns = returns[mb_idx]
                
                mb_obs = {
                    "spatial": mb_spatial,
                    "drone": mb_drone,
                    "wind": mb_wind
                }
                
                # Re-evaluate current policy outputs
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(mb_obs, action=mb_actions)
                
                total_loss, metrics = compute_ppo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    new_values=new_values,
                    returns=mb_returns,
                    entropy=entropy,
                    clip_epsilon=self.clip_epsilon,
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef
                )
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                if not torch.isfinite(total_loss):
                    raise ValueError(f"Loss is not finite: {total_loss}")
                    
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                metrics["gradient_norm"] = grad_norm.item()
                
                self.optimizer.step()
                
                for k, v in metrics.items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                updates += 1
                
        # Average metrics across all minibatches and epochs
        for k in agg_metrics:
            agg_metrics[k] /= max(1, updates)
            
        return agg_metrics
