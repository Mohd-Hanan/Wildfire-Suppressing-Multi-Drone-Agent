import torch
import torch.nn as nn
import numpy as np
import copy
from torch.distributions.categorical import Categorical
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
import torch.nn.functional as F

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
        
        # ACTOR BRANCH
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.actor_mlp = nn.Sequential(
            nn.Linear(32 * 11 * 11 + 9, 64),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(64, 7)
        
        # CRITIC BRANCH
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.critic_mlp = nn.Sequential(
            nn.Linear(32 * 11 * 11 + 9, 64),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(64, 1)

        self.to(self.device)
        
    def _preprocess_obs(self, obs):
        spatial = obs["spatial"]
        drone = obs["drone"]
        wind = obs["wind"]
        
        if not isinstance(spatial, torch.Tensor):
            spatial = torch.tensor(spatial, dtype=torch.float32, device=self.device)
        if not isinstance(drone, torch.Tensor):
            drone = torch.tensor(drone, dtype=torch.float32, device=self.device)
        if not isinstance(wind, torch.Tensor):
            wind = torch.tensor(wind, dtype=torch.float32, device=self.device)
            
        if spatial.dim() == 3:
            spatial = spatial.unsqueeze(0)
            drone = drone.unsqueeze(0)
            wind = wind.unsqueeze(0)
            
        return spatial, drone, wind

    def forward(self, obs):
        spatial, drone, wind = self._preprocess_obs(obs)
        non_spatial = torch.cat([drone, wind], dim=-1)
        
        # Actor
        a_cnn_out = self.actor_cnn(spatial)
        a_comb = torch.cat([a_cnn_out, non_spatial], dim=-1)
        a_feat = self.actor_mlp(a_comb)
        logits = self.actor_head(a_feat)
        
        # Critic
        c_cnn_out = self.critic_cnn(spatial)
        c_comb = torch.cat([c_cnn_out, non_spatial], dim=-1)
        c_feat = self.critic_mlp(c_comb)
        value = self.critic_head(c_feat).squeeze(-1)
        
        return logits, value
        
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


class DiagnosticTrainer:
    def __init__(self, policy, is_separate, device=torch.device("cpu")):
        self.policy = policy
        self.is_separate = is_separate
        self.device = device
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.minibatch_size = 64
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    def _normalize_advantages(self, advantages):
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        return (advantages - adv_mean) / (adv_std + 1e-8)

    def update(self, spatial, drone, wind, actions, old_log_probs, advantages, returns):
        normalized_advantages = self._normalize_advantages(advantages)
        N = len(actions)
        indices = np.arange(N)
        
        agg_metrics = {}
        updates = 0
        self.policy.train()
        
        if self.is_separate:
            actor_params = list(self.policy.actor_cnn.parameters()) + list(self.policy.actor_mlp.parameters()) + list(self.policy.actor_head.parameters())
            critic_params = list(self.policy.critic_cnn.parameters()) + list(self.policy.critic_mlp.parameters()) + list(self.policy.critic_head.parameters())
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                
                mb_obs = {
                    "spatial": spatial[mb_idx],
                    "drone": drone[mb_idx],
                    "wind": wind[mb_idx]
                }
                
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
                
                if self.is_separate:
                    # A. actor/policy-loss gradient norm on ACTOR parameters
                    # D. policy-loss gradient norm on CRITIC parameters
                    self.optimizer.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    p_grad_actor = get_grad_norm(actor_params)
                    p_grad_critic = get_grad_norm(critic_params)
                    
                    # C. value-loss gradient norm on ACTOR parameters
                    # B. value-loss gradient norm on CRITIC parameters
                    self.optimizer.zero_grad()
                    (self.value_coef * value_loss).backward(retain_graph=True)
                    v_grad_actor = get_grad_norm(actor_params)
                    v_grad_critic = get_grad_norm(critic_params)
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    pre_clip = get_grad_norm(self.policy.parameters())
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    post_clip = get_grad_norm(self.policy.parameters())
                    self.optimizer.step()
                    
                    m = {
                        "value_loss": value_loss.item(),
                        "p_grad_actor": p_grad_actor,
                        "v_grad_critic": v_grad_critic,
                        "v_grad_actor": v_grad_actor,
                        "p_grad_critic": p_grad_critic,
                        "pre_clip": pre_clip,
                        "post_clip": post_clip
                    }
                else:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    pre_clip = get_grad_norm(self.policy.parameters())
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    post_clip = get_grad_norm(self.policy.parameters())
                    self.optimizer.step()
                    
                    m = {
                        "value_loss": value_loss.item(),
                        "pre_clip": pre_clip,
                        "post_clip": post_clip
                    }
                    
                for k, v in m.items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                updates += 1
                
        for k in agg_metrics:
            agg_metrics[k] /= max(1, updates)
        return agg_metrics

def run_experiment(name, is_separate):
    print(f"\n==============================================")
    print(f"EXPERIMENT: {name} (Separate: {is_separate})")
    print(f"==============================================")
    
    device = torch.device("cpu")
    env = WildfireStatsWrapper(WildfireEnv())
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if is_separate:
        policy = SeparateActorCritic(device=device)
    else:
        policy = ActorCritic(device=device)
    
    trainer = DiagnosticTrainer(policy=policy, is_separate=is_separate, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    env_eval = WildfireEnv()
    fixed_obs, _ = env_eval.reset(seed=1000)
    fixed_obs_t = {
        "spatial": torch.tensor(fixed_obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
        "drone": torch.tensor(fixed_obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
        "wind": torch.tensor(fixed_obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    
    def eval_fixed_obs():
        policy.eval()
        with torch.no_grad():
            logits, val = policy(fixed_obs_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            return probs.numpy()[0], dist.entropy().item(), torch.argmax(logits, dim=-1).item()
            
    history = {}
    max_pre_clip = 0.0
    final_val_loss = 0.0
    
    final_m = {}

    for update in range(1, 21):
        policy.eval()
        collector.collect()
        
        actions = np.array(collector.buffer.actions)
        counts = [np.sum(actions == a) for a in range(7)]
        stoch_pcts = [c/len(actions)*100 for c in counts]
        
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
            
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        max_pre_clip = max(max_pre_clip, train_metrics['pre_clip'])
        final_val_loss = train_metrics['value_loss']
        final_m = train_metrics
        
        if update in [1, 5, 10, 15, 20]:
            cp, ce, ca = eval_fixed_obs()
            history[update] = {
                'ent': ce,
                'west_pct_stoch': stoch_pcts[4]
            }
            if is_separate:
                print(f"Update {update:2d} | Stoch West: {stoch_pcts[4]:.1f}% | Ent: {ce:.3f} | pActorGrad: {train_metrics['p_grad_actor']:.1f} | vCriticGrad: {train_metrics['v_grad_critic']:.1f} | vActorGrad: {train_metrics['v_grad_actor']:.1f} | pCriticGrad: {train_metrics['p_grad_critic']:.1f}")
            else:
                print(f"Update {update:2d} | Stoch West: {stoch_pcts[4]:.1f}% | Ent: {ce:.3f} | PreClip: {train_metrics['pre_clip']:.1f} | PostClip: {train_metrics['post_clip']:.3f}")

    return history, max_pre_clip, final_val_loss, final_m

def main():
    res_A = run_experiment("A (Shared)", False)
    res_B = run_experiment("B (Separate)", True)
    
    print("\n\n=== COMPACT COMPARISON TABLE ===")
    print(f"{'Experiment':<15} | {'Arch':<10} | {'Ent@5':<7} | {'Ent@10':<7} | {'Ent@20':<7} | {'West@5':<7} | {'West@10':<7} | {'West@20':<7} | {'Max Grad':<12} | {'Final Value Loss':<15}")
    print("-" * 120)
    
    def print_row1(name, arch, r):
        h, mgrad, vloss, _ = r
        print(f"{name:<15} | {arch:<10} | {h[5]['ent']:<7.2f} | {h[10]['ent']:<7.2f} | {h[20]['ent']:<7.2f} | {h[5]['west_pct_stoch']:<7.1f} | {h[10]['west_pct_stoch']:<7.1f} | {h[20]['west_pct_stoch']:<7.1f} | {mgrad:<12.1f} | {vloss:<15.1f}")
        
    print_row1("A (Shared)", "Shared", res_A)
    print_row1("B (Separate)", "Separate", res_B)
    
    print("\n=== ISOLATION TEST (Update 20 Averages, Separate Arch) ===")
    print(f"{'Experiment':<15} | {'Actor Policy Grad':<20} | {'Critic Value Grad':<20} | {'Value Grad on Actor':<20} | {'Policy Grad on Critic':<20}")
    print("-" * 105)
    
    h_b, mgrad_b, vloss_b, fin_b = res_B
    print(f"{'B (Separate)':<15} | {fin_b['p_grad_actor']:<20.1f} | {fin_b['v_grad_critic']:<20.1f} | {fin_b['v_grad_actor']:<20.1f} | {fin_b['p_grad_critic']:<20.1f}")

main()
