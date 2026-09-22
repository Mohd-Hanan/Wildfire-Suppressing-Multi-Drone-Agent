import torch
import torch.nn as nn
import numpy as np
import time
from torch.distributions.categorical import Categorical
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper

def get_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    return total_norm.item()

class SeparateActorCritic(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(SeparateActorCritic, self).__init__()
        self.device = device
        
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.actor_mlp = nn.Sequential(nn.Linear(32*11*11 + 9, 64), nn.ReLU())
        self.actor_head = nn.Linear(64, 7)
        
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.critic_mlp = nn.Sequential(nn.Linear(32*11*11 + 9, 64), nn.ReLU())
        self.critic_head = nn.Linear(64, 1)

        self.to(self.device)
        
    def forward(self, obs):
        spatial = obs["spatial"]
        if not isinstance(spatial, torch.Tensor):
            spatial = torch.tensor(spatial, dtype=torch.float32, device=self.device)
            drone = torch.tensor(obs["drone"], dtype=torch.float32, device=self.device)
            wind = torch.tensor(obs["wind"], dtype=torch.float32, device=self.device)
        else:
            drone, wind = obs["drone"], obs["wind"]
            
        if spatial.dim() == 3:
            spatial, drone, wind = spatial.unsqueeze(0), drone.unsqueeze(0), wind.unsqueeze(0)
            
        non_spatial = torch.cat([drone, wind], dim=-1)
        
        a_comb = torch.cat([self.actor_cnn(spatial), non_spatial], dim=-1)
        logits = self.actor_head(self.actor_mlp(a_comb))
        
        c_comb = torch.cat([self.critic_cnn(spatial), non_spatial], dim=-1)
        value = self.critic_head(self.critic_mlp(c_comb)).squeeze(-1)
        
        return logits, value
        
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        probs = Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

class SmokeTestTrainer:
    def __init__(self, policy, device=torch.device("cpu")):
        self.policy = policy
        self.device = device
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.minibatch_size = 64
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    def _normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, spatial, drone, wind, actions, old_log_probs, advantages, returns):
        normalized_advantages = self._normalize_advantages(advantages)
        N = len(actions)
        indices = np.arange(N)
        
        agg_metrics = {}
        updates = 0
        self.policy.train()
        
        actor_params = list(self.policy.actor_cnn.parameters()) + list(self.policy.actor_mlp.parameters()) + list(self.policy.actor_head.parameters())
        critic_params = list(self.policy.critic_cnn.parameters()) + list(self.policy.critic_mlp.parameters()) + list(self.policy.critic_head.parameters())
        
        v_grad_on_actor = 0.0
        p_grad_on_critic = 0.0
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                
                mb_obs = {"spatial": spatial[mb_idx], "drone": drone[mb_idx], "wind": wind[mb_idx]}
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(mb_obs, action=actions[mb_idx])
                
                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                unclipped = ratio * normalized_advantages[mb_idx]
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                clipped = clipped_ratio * normalized_advantages[mb_idx]
                surrogate = torch.min(unclipped, clipped)
                
                policy_loss = -torch.mean(surrogate)
                value_loss = torch.mean((returns[mb_idx] - new_values) ** 2)
                entropy_mean = torch.mean(entropy)
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean
                
                # Check Isolation once per update
                if updates == 0:
                    self.optimizer.zero_grad()
                    (self.value_coef * value_loss).backward(retain_graph=True)
                    v_grad_on_actor = get_grad_norm(actor_params)
                    
                    self.optimizer.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    p_grad_on_critic = get_grad_norm(critic_params)
                    
                self.optimizer.zero_grad()
                total_loss.backward()
                
                pre_clip = get_grad_norm(self.policy.parameters())
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                approx_kl = torch.mean(old_log_probs[mb_idx] - new_log_probs).item()
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()
                
                m = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy_mean.item(),
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "pre_clip": pre_clip
                }
                for k, v in m.items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                updates += 1
                
        for k in agg_metrics:
            agg_metrics[k] /= max(1, updates)
            
        agg_metrics["v_grad_on_actor"] = v_grad_on_actor
        agg_metrics["p_grad_on_critic"] = p_grad_on_critic
        return agg_metrics

def main():
    print("========================================")
    print("STAGE 7.15 — 2 UPDATE SMOKE TEST")
    print("========================================")
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCritic(device=device)
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    start_time = time.time()
    
    total_val_actor = 0.0
    total_pol_critic = 0.0
    
    for update in range(1, 3):
        up_start = time.time()
        
        policy.eval()
        collector.collect()
        
        actions = np.array(collector.buffer.actions)
        counts = [np.sum(actions == a) for a in range(7)]
        pcts = [c/len(actions)*100 for c in counts]
        
        # Calculate Episode stats from wrapper
        rollout_reward_mean = np.mean(collector.buffer.rewards)
        if len(env.episode_rewards) > 0:
            ep_reward = np.mean(env.episode_rewards)
            ep_burned = np.mean(env.episode_burned_cells)
        else:
            ep_reward = 0.0
            ep_burned = 0.0
            
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
            
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        up_time = time.time() - up_start
        
        total_val_actor += train_metrics["v_grad_on_actor"]
        total_pol_critic += train_metrics["p_grad_on_critic"]
        
        print(f"\nUpdate {update}/2")
        print("----------------------------")
        print(f"Steps: 256")
        print(f"Rollout Reward Mean: {rollout_reward_mean:.4f}")
        print(f"Episode Reward Mean: {ep_reward:.4f}")
        print(f"Burned Cells: {ep_burned:.4f}")
        print(f"Entropy: {train_metrics['entropy']:.4f}")
        print(f"Policy Loss: {train_metrics['policy_loss']:.4f}")
        print(f"Value Loss: {train_metrics['value_loss']:.4f}")
        print(f"Approx KL: {train_metrics['approx_kl']:.4f}")
        print(f"Clip Fraction: {train_metrics['clip_fraction']:.4f}")
        print(f"Gradient Norm: {train_metrics['pre_clip']:.4f}")
        print(f"West Action %: {pcts[4]:.1f}%")
        print(f"Water Action %: {pcts[5]:.1f}%")
        print(f"Retardant Action %: {pcts[6]:.1f}%")
        print(f"Elapsed Time: {up_time:.2f}s")
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / 2
    est_200 = avg_time * 200
    
    print("\n========================================")
    print("SMOKE TEST COMPLETE")
    print("========================================")
    print(f"Total Updates: 2")
    print(f"Total Steps: 512")
    print(f"Total Runtime: {total_time:.2f}s")
    print(f"Average Time / Update: {avg_time:.2f}s")
    print(f"Estimated Time for 200 Updates: {est_200/60:.2f} minutes")
    
    print("\nActor/Critic Gradient Isolation:")
    print(f"Value -> Actor: {'PASS' if total_val_actor < 1e-6 else 'FAIL'}")
    print(f"Policy -> Critic: {'PASS' if total_pol_critic < 1e-6 else 'FAIL'}")

if __name__ == "__main__":
    main()
