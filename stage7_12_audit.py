import torch
import numpy as np
import copy
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from wildfire.rl.ppo import compute_ppo_loss
import torch.nn.functional as F

def get_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    return total_norm.item()

class DiagnosticTrainer:
    def __init__(self, policy, value_coef, device=torch.device("cpu")):
        self.policy = policy
        self.device = device
        self.clip_epsilon = 0.2
        self.value_coef = value_coef
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
                
                # Manual PPO loss calculation to get components
                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                unclipped = ratio * normalized_advantages[mb_idx]
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                clipped = clipped_ratio * normalized_advantages[mb_idx]
                surrogate = torch.min(unclipped, clipped)
                policy_loss = -torch.mean(surrogate)
                value_loss = torch.mean((returns[mb_idx] - new_values) ** 2)
                entropy_mean = torch.mean(entropy)
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean
                
                # 1. Policy Grad
                self.optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                p_grad = get_grad_norm(self.policy.parameters())
                
                # 2. Value Grad
                self.optimizer.zero_grad()
                v_scaled = self.value_coef * value_loss
                v_scaled.backward(retain_graph=True)
                v_grad = get_grad_norm(self.policy.parameters())
                
                # 3. Entropy Grad
                self.optimizer.zero_grad()
                e_scaled = -self.entropy_coef * entropy_mean
                e_scaled.backward(retain_graph=True)
                e_grad = get_grad_norm(self.policy.parameters())
                
                # 4. Total Grad (Pre-clip)
                self.optimizer.zero_grad()
                total_loss.backward()
                pre_clip = get_grad_norm(self.policy.parameters())
                
                # 5. Total Grad (Post-clip)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                post_clip = get_grad_norm(self.policy.parameters())
                
                self.optimizer.step()
                
                m = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy_mean.item(),
                    "p_grad": p_grad,
                    "v_grad": v_grad,
                    "e_grad": e_grad,
                    "pre_clip": pre_clip,
                    "post_clip": post_clip
                }
                for k, v in m.items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                updates += 1
                
        for k in agg_metrics:
            agg_metrics[k] /= max(1, updates)
        return agg_metrics

def run_experiment(name, value_coef, initial_state_dict):
    print(f"\n==============================================")
    print(f"EXPERIMENT: {name} (Value Coef: {value_coef})")
    print(f"==============================================")
    
    device = torch.device("cpu")
    env = WildfireStatsWrapper(WildfireEnv())
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    policy = ActorCritic(device=device)
    policy.load_state_dict(copy.deepcopy(initial_state_dict))
    
    trainer = DiagnosticTrainer(policy=policy, value_coef=value_coef, device=device)
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
    
    avg_p_grad, avg_v_grad, avg_e_grad, avg_pre_clip = 0, 0, 0, 0

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
        
        avg_p_grad = train_metrics['p_grad']
        avg_v_grad = train_metrics['v_grad']
        avg_e_grad = train_metrics['e_grad']
        avg_pre_clip = train_metrics['pre_clip']
        
        if update in [1, 5, 10, 15, 20]:
            cp, ce, ca = eval_fixed_obs()
            history[update] = {
                'ent': ce,
                'west_pct_stoch': stoch_pcts[4]
            }
            print(f"Update {update:2d} | Stoch West: {stoch_pcts[4]:.1f}% | Ent: {ce:.3f} | pGrad: {avg_p_grad:.1f} | vGrad: {avg_v_grad:.1f} | eGrad: {avg_e_grad:.1f} | Pre: {avg_pre_clip:.1f} | Post: {train_metrics['post_clip']:.3f}")

    return history, max_pre_clip, final_val_loss, avg_p_grad, avg_v_grad, avg_e_grad, avg_pre_clip

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    dummy_policy = ActorCritic(device=device)
    initial_sd = copy.deepcopy(dummy_policy.state_dict())
    
    res_A = run_experiment("A (coef=0.5)", 0.5, initial_sd)
    res_B = run_experiment("B (coef=0.1)", 0.1, initial_sd)
    res_C = run_experiment("C (coef=0.01)", 0.01, initial_sd)
    res_D = run_experiment("D (coef=0.0)", 0.0, initial_sd)
    
    print("\n\n=== COMPACT COMPARISON TABLE ===")
    print(f"{'Experiment':<15} | {'vc':<4} | {'Ent@5':<7} | {'Ent@10':<7} | {'Ent@20':<7} | {'West@5':<7} | {'West@10':<7} | {'West@20':<7} | {'Max Grad':<12} | {'Final Value Loss':<15}")
    print("-" * 115)
    
    def print_row1(name, vc, r):
        h, mgrad, vloss, _, _, _, _ = r
        print(f"{name:<15} | {vc:<4} | {h[5]['ent']:<7.2f} | {h[10]['ent']:<7.2f} | {h[20]['ent']:<7.2f} | {h[5]['west_pct_stoch']:<7.1f} | {h[10]['west_pct_stoch']:<7.1f} | {h[20]['west_pct_stoch']:<7.1f} | {mgrad:<12.1f} | {vloss:<15.1f}")
        
    print_row1("A (coef=0.5)", 0.5, res_A)
    print_row1("B (coef=0.1)", 0.1, res_B)
    print_row1("C (coef=0.01)", 0.01, res_C)
    print_row1("D (coef=0.0)", 0.0, res_D)
    
    print("\n=== GRADIENT SOURCE TABLE (Update 20 Averages) ===")
    print(f"{'Experiment':<15} | {'Policy Grad':<12} | {'Value Grad':<12} | {'Entropy Grad':<12} | {'Combined Grad':<15} | {'Dominant Source'}")
    print("-" * 105)
    
    def print_row2(name, r):
        h, mgrad, vloss, pg, vg, eg, cg = r
        dom = "Value" if vg > pg and vg > eg else ("Policy" if pg > eg else "Entropy")
        print(f"{name:<15} | {pg:<12.1f} | {vg:<12.1f} | {eg:<12.1f} | {cg:<15.1f} | {dom}")
        
    print_row2("A (coef=0.5)", res_A)
    print_row2("B (coef=0.1)", res_B)
    print_row2("C (coef=0.01)", res_C)
    print_row2("D (coef=0.0)", res_D)

main()
