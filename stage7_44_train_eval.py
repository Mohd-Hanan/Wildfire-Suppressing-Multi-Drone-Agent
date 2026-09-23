import os
import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.ppo_trainer import PPOTrainer
from stage7_35_final_truth import SeparateActorCriticSymmetric64
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae

def train():
    print("========================================")
    print("STAGE 7.44 — CANDIDATE PHYSICS TRAINING")
    print("========================================")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireEnv("configs/environment.yaml")
    
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    trainer = PPOTrainer(policy=policy, device=device, learning_rate=3e-4, ppo_epochs=10, minibatch_size=64)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    for update in range(1, 21):
        policy.eval()
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        
        water_freq = (acts == 5).float().mean().item() * 100
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update % 5 == 0 or update == 1:
            print(f"Update {update:2d}/20 | Reward: {np.mean(collector.buffer.rewards)*256:7.4f} (ep est) | "
                  f"Entropy: {train_metrics['entropy']:6.4f} | Value Loss: {train_metrics['value_loss']:8.1f} | "
                  f"Policy Loss: {train_metrics['policy_loss']:7.4f} | WATER%: {water_freq:5.1f}%")

    torch.save(policy.state_dict(), "stage7_44_candidate_physics_20updates.pth")
    print("Saved stage7_44_candidate_physics_20updates.pth")
    return policy

def evaluate_and_test(policy):
    print("\nStarting 20-episode evaluation...")
    env = WildfireEnv("configs/environment.yaml")
    device = torch.device("cpu")
    policy.eval()
    
    stats = {
        "reward": [],
        "burned": [],
        "suppressed": [],
        "water_total": 0,
        "suppressions_total": 0,
        "natural_ext": 0,
        "suppression_ext": 0,
        "crashes": 0,
        "base_visits": 0,
        "ep_lengths": [],
        "action_counts": {i: 0 for i in range(6)},
        "final_battery": [],
        "final_payload": []
    }
    
    for ep in range(20):
        obs, _ = env.reset(seed=100+ep)
        done = False
        ep_reward = 0
        suppress_count = 0
        steps = 0
        base_visits_ep = 0
        was_at_base = True
        
        while not done and steps < 500:
            with torch.no_grad():
                action_t, _, _, _ = policy.get_action_and_value(obs)
            
            a = action_t.item()
            stats['action_counts'][a] += 1
            if a == 5:
                stats['water_total'] += 1
                
            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            ep_reward += reward
            
            supp = info.get('newly_suppressed_cells', 0)
            suppress_count += supp
                
            drone = env.world.drones[0]
            is_at_base = (drone.x, drone.y) == (2,2)
            if is_at_base and not was_at_base and drone.battery == drone.max_battery:
                base_visits_ep += 1
            was_at_base = is_at_base
            
            steps += 1
            
        drone = env.world.drones[0]
        fm = env.world.fire_manager
        
        crashed = not drone.active
        active_fire = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
        extinguished = active_fire == 0
        
        if crashed: stats['crashes'] += 1
        
        if extinguished:
            total_cells = (fm.fire_map > 1).sum()
            if suppress_count >= total_cells * 0.3 or (suppress_count > 0 and steps < 200):
                stats['suppression_ext'] += 1
            else:
                stats['natural_ext'] += 1
                
        stats['base_visits'] += base_visits_ep
        stats['reward'].append(ep_reward)
        stats['burned'].append((fm.fire_map > 1).sum())
        stats['suppressed'].append(suppress_count)
        stats['suppressions_total'] += suppress_count
        stats['ep_lengths'].append(steps)
        stats['final_battery'].append(drone.battery)
        stats['final_payload'].append(drone.payload)

    print("\nQuantitative Evaluation (20 episodes):")
    print(f"Mean Reward:        {np.mean(stats['reward']):.2f} ± {np.std(stats['reward']):.2f}")
    print(f"Mean Burned Cells:  {np.mean(stats['burned']):.1f} ± {np.std(stats['burned']):.1f}")
    print(f"Mean Suppressed:    {np.mean(stats['suppressed']):.1f} ± {np.std(stats['suppressed']):.1f}")
    print(f"Total Suppressions: {stats['suppressions_total']}")
    
    total_ext = stats['natural_ext'] + stats['suppression_ext']
    print(f"Extinction %:       {(total_ext/20)*100:.1f}%")
    print(f"Natural Ext %:      {(stats['natural_ext']/20)*100:.1f}%")
    print(f"Suppression Ext %:  {(stats['suppression_ext']/20)*100:.1f}%")
    
    print(f"Mean Ep Length:     {np.mean(stats['ep_lengths']):.1f}")
    print(f"Crash Rate:         {stats['crashes']}/20 ({(stats['crashes']/20)*100:.1f}%)")
    print(f"WATER actions:      {stats['water_total']}")
    print(f"Base visits:        {stats['base_visits']}")
    
    total_actions = sum(stats['action_counts'].values())
    print(f"Action dist:        " + ", ".join([f"{k}:{v/total_actions*100:.1f}%" for k, v in stats['action_counts'].items()]))
    print(f"Final Battery:      {np.mean(stats['final_battery']):.1f}")
    print(f"Final Payload:      {np.mean(stats['final_payload']):.1f}")

if __name__ == "__main__":
    policy = train()
    evaluate_and_test(policy)
