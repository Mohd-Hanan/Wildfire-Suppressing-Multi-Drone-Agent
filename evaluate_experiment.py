import torch
import sys
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.train import train_ppo
from wildfire.rl.evaluate import evaluate_policy

def main():
    print("Evaluating Random Baseline...")
    env = WildfireEnv()
    seeds = list(range(1000, 1020))
    
    random_agg = evaluate_policy(env, seeds, policy=None)
    
    print("Training PPO for 20 updates...")
    # Suppress training output to keep the log clean
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    policy, _, _, _ = train_ppo(
        updates=20,
        rollout_size=256,
        ppo_epochs=4,
        minibatch_size=64,
        seed=42,
        device_name="cpu"
    )
    sys.stdout = old_stdout
    
    print("Evaluating Trained PPO Baseline...")
    ppo_agg = evaluate_policy(env, seeds, policy=policy, device=torch.device("cpu"))
    
    with open("stage7_8_authoritative_evaluation.log", "w") as f:
        f.write("--- RANDOM BASELINE RESULTS ---\n")
        f.write(f"Mean Reward: {random_agg['mean_reward']:.2f} +/- {random_agg['std_reward']:.2f}\n")
        f.write(f"Mean Length: {random_agg['mean_length']:.2f}\n")
        f.write(f"Mean Burned: {random_agg['mean_burned']:.2f} +/- {random_agg['std_burned']:.2f}\n")
        f.write(f"Extinguished: {random_agg['extinguished_count']}/20 ({random_agg['extinction_rate']*100:.1f}%)\n")
        f.write(f"Total Crashes: {random_agg['total_crashes']}\n")
        f.write(f"Crash Causes: {random_agg['crash_causes']}\n")
        f.write("\nAction Distribution (%):\n")
        for k, v in random_agg['action_pct'].items():
            f.write(f"  Action {k}: {v:.1f}%\n")
        f.write(f"Mean Action Changes: {random_agg['mean_action_changes']:.2f}\n")
        f.write("\nResource Behavior:\n")
        f.write(f"  Final Battery: {random_agg['mean_final_battery']:.1f}\n")
        f.write(f"  Min Battery: {random_agg['mean_min_battery']:.1f}\n")
        f.write(f"  Final Payload: {random_agg['mean_final_payload']:.1f}\n")
        f.write(f"  Deploy Attempts: {random_agg['mean_deploy_attempts']:.1f}\n")
        f.write(f"  Successful Water: {random_agg['mean_successful_water']:.1f}\n")
        f.write(f"  Successful Retardant: {random_agg['mean_successful_retardant']:.1f}\n")
        f.write(f"  Failed Deployments: {random_agg['mean_failed_deployments']:.1f}\n")
        f.write(f"  Base Visits: {random_agg['mean_base_visits']:.1f}\n")
        f.write(f"  Boundary Pushes: {random_agg['mean_boundary_pushes']:.1f}\n")

        f.write("\n--- PPO RESULTS ---\n")
        f.write(f"Mean Reward: {ppo_agg['mean_reward']:.2f} +/- {ppo_agg['std_reward']:.2f}\n")
        f.write(f"Mean Length: {ppo_agg['mean_length']:.2f}\n")
        f.write(f"Mean Burned: {ppo_agg['mean_burned']:.2f} +/- {ppo_agg['std_burned']:.2f}\n")
        f.write(f"Extinguished: {ppo_agg['extinguished_count']}/20 ({ppo_agg['extinction_rate']*100:.1f}%)\n")
        f.write(f"Total Crashes: {ppo_agg['total_crashes']}\n")
        f.write(f"Crash Causes: {ppo_agg['crash_causes']}\n")
        f.write("\nAction Distribution (%):\n")
        for k, v in ppo_agg['action_pct'].items():
            f.write(f"  Action {k}: {v:.1f}%\n")
        f.write(f"Mean Action Changes: {ppo_agg['mean_action_changes']:.2f}\n")
        f.write(f"Most Common Action: {ppo_agg['most_common_action']}\n")
        f.write("\nResource Behavior:\n")
        f.write(f"  Final Battery: {ppo_agg['mean_final_battery']:.1f}\n")
        f.write(f"  Min Battery: {ppo_agg['mean_min_battery']:.1f}\n")
        f.write(f"  Final Payload: {ppo_agg['mean_final_payload']:.1f}\n")
        f.write(f"  Deploy Attempts: {ppo_agg['mean_deploy_attempts']:.1f}\n")
        f.write(f"  Successful Water: {ppo_agg['mean_successful_water']:.1f}\n")
        f.write(f"  Successful Retardant: {ppo_agg['mean_successful_retardant']:.1f}\n")
        f.write(f"  Failed Deployments: {ppo_agg['mean_failed_deployments']:.1f}\n")
        f.write(f"  Base Visits: {ppo_agg['mean_base_visits']:.1f}\n")
        f.write(f"  Boundary Pushes: {ppo_agg['mean_boundary_pushes']:.1f}\n")
        
        f.write("\n--- PAIRED PER-SEED DIFFERENCE (PPO - Random) ---\n")
        f.write(f"{'Seed':<6} | {'Reward Diff':<12} | {'Burned Diff':<12} | {'PPO Ext':<8} | {'Rnd Ext':<8}\n")
        f.write("-" * 60 + "\n")
        for p_ep, r_ep in zip(ppo_agg['episodes'], random_agg['episodes']):
            seed = p_ep['seed']
            r_diff = p_ep['reward'] - r_ep['reward']
            b_diff = p_ep['burned_cells'] - r_ep['burned_cells']
            f.write(f"{seed:<6} | {r_diff:<12.2f} | {b_diff:<12} | {str(p_ep['extinguished']):<8} | {str(r_ep['extinguished']):<8}\n")

if __name__ == "__main__":
    main()
