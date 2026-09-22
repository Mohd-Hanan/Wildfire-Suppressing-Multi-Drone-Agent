import torch
import numpy as np
import copy
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
import torch.nn.functional as F

def run_experiment(name, scale_factor, initial_state_dict):
    print(f"\n==============================================")
    print(f"EXPERIMENT: {name} (Reward Scale: {scale_factor})")
    print(f"==============================================")
    
    device = torch.device("cpu")
    env = WildfireStatsWrapper(WildfireEnv())
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    policy = ActorCritic(device=device)
    policy.load_state_dict(copy.deepcopy(initial_state_dict))
    
    trainer = PPOTrainer(policy=policy, device=device)
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
    
    # Initial
    cp, ce, ca = eval_fixed_obs()
    history[0] = {'ent': ce, 'west_pct_fixed': cp[4]*100}
    
    max_grad = 0.0
    final_val_loss = 0.0

    for update in range(1, 21):
        policy.eval()
        collector.collect()
        
        actions = np.array(collector.buffer.actions)
        counts = [np.sum(actions == a) for a in range(7)]
        stoch_pcts = [c/len(actions)*100 for c in counts]
        
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        
        # Apply Temporary Scale
        rewards = rewards / scale_factor
        
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value / scale_factor if collector.last_value else 0.0)
            
        policy.train()
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        max_grad = max(max_grad, train_metrics['gradient_norm'])
        final_val_loss = train_metrics['value_loss']
        
        if update in [1, 5, 10, 15, 20]:
            cp, ce, ca = eval_fixed_obs()
            history[update] = {
                'ent': ce,
                'west_pct_fixed': cp[4]*100,
                'west_pct_stoch': stoch_pcts[4]
            }
            print(f"Update {update:2d} | Stoch West: {stoch_pcts[4]:.1f}% | Fixed West: {cp[4]*100:.1f}% | Ent: {ce:.3f} | Val Loss: {train_metrics['value_loss']:.2f} | Grad: {train_metrics['gradient_norm']:.2f}")

    return history, max_grad, final_val_loss

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    dummy_policy = ActorCritic(device=device)
    initial_sd = copy.deepcopy(dummy_policy.state_dict())
    
    res_A, max_grad_A, vloss_A = run_experiment("A (Original)", 1.0, initial_sd)
    res_B, max_grad_B, vloss_B = run_experiment("B (/10)", 10.0, initial_sd)
    res_C, max_grad_C, vloss_C = run_experiment("C (/100)", 100.0, initial_sd)
    
    print("\n\n=== COMPACT COMPARISON TABLE ===")
    print(f"{'Experiment':<15} | {'Ent@5':<7} | {'Ent@10':<7} | {'Ent@20':<7} | {'West@5':<7} | {'West@10':<7} | {'West@20':<7} | {'Max Grad':<12} | {'Final Value Loss':<15}")
    print("-" * 110)
    
    def print_row(name, res, mgrad, vloss):
        e5 = f"{res[5]['ent']:.2f}"
        e10 = f"{res[10]['ent']:.2f}"
        e20 = f"{res[20]['ent']:.2f}"
        w5 = f"{res[5]['west_pct_stoch']:.1f}%"
        w10 = f"{res[10]['west_pct_stoch']:.1f}%"
        w20 = f"{res[20]['west_pct_stoch']:.1f}%"
        print(f"{name:<15} | {e5:<7} | {e10:<7} | {e20:<7} | {w5:<7} | {w10:<7} | {w20:<7} | {mgrad:<12.1f} | {vloss:<15.1f}")
        
    print_row("A (Original)", res_A, max_grad_A, vloss_A)
    print_row("B (/10)", res_B, max_grad_B, vloss_B)
    print_row("C (/100)", res_C, max_grad_C, vloss_C)

main()
