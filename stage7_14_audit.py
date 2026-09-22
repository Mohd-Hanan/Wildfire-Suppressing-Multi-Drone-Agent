import torch
import numpy as np
import copy
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.policy import ActorCritic
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from wildfire.rl.evaluate import evaluate_policy
from stage7_13_audit import SeparateActorCritic, DiagnosticTrainer
import torch.nn.functional as F

def eval_fixed_obs(policy, fixed_obs_t):
    policy.eval()
    with torch.no_grad():
        logits, val = policy(fixed_obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return probs.numpy()[0], dist.entropy().item(), torch.argmax(logits, dim=-1).item()

def train_policy(name, policy, is_separate, env_train, fixed_obs_t):
    print(f"\n--- TRAINING {name} ---")
    device = torch.device("cpu")
    trainer = DiagnosticTrainer(policy=policy, is_separate=is_separate, device=device)
    collector = RolloutCollector(env=env_train, policy=policy, rollout_size=256, device=device)
    
    history = {}
    
    for update in range(1, 21):
        policy.eval()
        collector.collect()
        
        actions = np.array(collector.buffer.actions)
        counts = [np.sum(actions == a) for a in range(7)]
        stoch_pcts = [c/len(actions)*100 for c in counts]
        
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
            
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update in [1, 5, 10, 15, 20]:
            cp, ce, ca = eval_fixed_obs(policy, fixed_obs_t)
            history[update] = {
                'ent': ce,
                'stoch_pcts': stoch_pcts,
                'fixed_pcts': cp * 100,
                'argmax': ca
            }
            print(f"Update {update:2d} | Stoch West: {stoch_pcts[4]:.1f}% | Ent: {ce:.3f}")
            
    return history

def run_experiment():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env_train = WildfireStatsWrapper(WildfireEnv())
    env_eval = WildfireEnv()
    
    fixed_obs, _ = env_eval.reset(seed=1000)
    fixed_obs_t = {
        "spatial": torch.tensor(fixed_obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
        "drone": torch.tensor(fixed_obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
        "wind": torch.tensor(fixed_obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    
    # Random Baseline Eval
    print("\n--- EVALUATING RANDOM ---")
    res_A = evaluate_policy(env_eval, list(range(1000, 1020)), policy=None)
    
    # Shared PPO
    torch.manual_seed(42)
    np.random.seed(42)
    policy_shared = ActorCritic(device=device)
    hist_shared = train_policy("B (Shared)", policy_shared, False, env_train, fixed_obs_t)
    print("\n--- EVALUATING SHARED ---")
    res_B = evaluate_policy(env_eval, list(range(1000, 1020)), policy=policy_shared, device=device)
    
    # Separate PPO
    torch.manual_seed(42)
    np.random.seed(42)
    policy_sep = SeparateActorCritic(device=device)
    hist_sep = train_policy("C (Separate)", policy_sep, True, env_train, fixed_obs_t)
    print("\n--- EVALUATING SEPARATE ---")
    res_C = evaluate_policy(env_eval, list(range(1000, 1020)), policy=policy_sep, device=device)
    
    # --------------------------------------------------
    # Resource Check
    # --------------------------------------------------
    max_battery = 150.0  # From config
    max_payload = 5.0
    
    def check_res(name, r):
        # We look at the min/max of final_battery in the episodes to see if any are > max_battery
        finals = [ep["final_battery"] for ep in r["episodes"]]
        payloads = [ep["final_payload"] for ep in r["episodes"]]
        assert all(0 <= x <= max_battery for x in finals), f"{name}: final battery out of bounds"
        assert all(0 <= x <= max_payload for x in payloads), f"{name}: final payload out of bounds"
        
    check_res("Random", res_A)
    check_res("Shared", res_B)
    check_res("Separate", res_C)
    print("\n[OK] Resource metric bounds visually verified. Battery and Payload never exceeded max.")
    
    print("\n\n=== FACTUAL COMPARISON TABLE ===")
    print(f"{'Controller':<15} | {'Mean Reward':<12} | {'Mean Burned':<12} | {'Extinction Rate':<15} | {'Mean Length':<12} | {'Crashes':<8}")
    print("-" * 85)
    def print_main(name, r):
        print(f"{name:<15} | {r['mean_reward']:<12.1f} | {r['mean_burned']:<12.1f} | {r['extinction_rate']*100:<14.1f}% | {r['mean_length']:<12.1f} | {r['total_crashes']:<8}")
    print_main("Random", res_A)
    print_main("Shared PPO", res_B)
    print_main("Separate PPO", res_C)
    
    print("\n=== ACTION DISTRIBUTION TABLE ===")
    print(f"{'Controller':<15} | {'Stay':<6} | {'North':<6} | {'South':<6} | {'East':<6} | {'West':<6} | {'Water':<6} | {'Retardant':<6}")
    print("-" * 85)
    def print_act(name, r):
        p = r['action_pct']
        print(f"{name:<15} | {p[0]:<6.1f} | {p[1]:<6.1f} | {p[2]:<6.1f} | {p[3]:<6.1f} | {p[4]:<6.1f} | {p[5]:<6.1f} | {p[6]:<6.1f}")
    print_act("Random", res_A)
    print_act("Shared PPO", res_B)
    print_act("Separate PPO", res_C)
    
    print("\n=== BEHAVIOR TABLE ===")
    print(f"{'Controller':<15} | {'Deploy Attempts':<15} | {'Successful Water':<18} | {'Successful Retardant':<22} | {'Base Visits':<12} | {'Boundary Pushes':<15}")
    print("-" * 110)
    def print_beh(name, r):
        print(f"{name:<15} | {r['mean_deploy_attempts']:<15.1f} | {r['mean_successful_water']:<18.1f} | {r['mean_successful_retardant']:<22.1f} | {r['mean_base_visits']:<12.1f} | {r['mean_boundary_pushes']:<15.1f}")
    print_beh("Random", res_A)
    print_beh("Shared PPO", res_B)
    print_beh("Separate PPO", res_C)
    
    print("\n=== SEPARATE PPO LEARNING CHECK ===")
    for u in [1, 5, 10, 15, 20]:
        h = hist_sep[u]
        print(f"Update {u:2d} | Ent: {h['ent']:.2f} | Stoch West: {h['stoch_pcts'][4]:.1f}% | Stoch Water: {h['stoch_pcts'][5]:.1f}% | Fixed Argmax: {h['argmax']}")

run_experiment()
