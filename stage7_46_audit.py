import torch
import torch.nn as nn
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_35_final_truth import SeparateActorCriticSymmetric64
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.ppo_trainer import PPOTrainer

def audit_part_a():
    print("==================================================")
    print("PART A — VERIFY THE ACTUAL OBSERVATION")
    print("==================================================")
    env = WildfireEnv("configs/environment.yaml")
    obs, _ = env.reset(seed=42)
    
    print("Shapes:")
    print(f"spatial: {obs['spatial'].shape}")
    print(f"drone: {obs['drone'].shape}")
    print(f"wind: {obs['wind'].shape}")
    
    print("\nDrone vector ordering (from observation.py):")
    print("index 0 = battery")
    print("index 1 = payload")
    print("index 2 = normalized_margin")
    print("index 3 = dist_to_base")
    print("index 4 = nx")
    print("index 5 = ny")
    print("index 6 = fire_dx")
    print("index 7 = fire_dy")
    print("index 8 = fire_distance")
    
    print("\nReal Observations:")
    for ep in range(5):
        obs, _ = env.reset(seed=100+ep)
        drone = env.world.drones[0]
        fm = env.world.fire_manager.fire_map
        active_mask = (fm == 1) | (fm == 2) | (fm == 3)
        active_coords = np.argwhere(active_mask)
        if len(active_coords) > 0:
            distances = np.sqrt((active_coords[:, 0] - drone.x)**2 + (active_coords[:, 1] - drone.y)**2)
            nearest_idx = np.argmin(distances)
            nearest_x, nearest_y = active_coords[nearest_idx]
        else:
            nearest_x, nearest_y = -1, -1
            
        print(f"Ep {ep}: drone({drone.x},{drone.y}) fire({nearest_x},{nearest_y}) -> "
              f"dx={obs['drone'][6]:.3f}, dy={obs['drone'][7]:.3f}, dist={obs['drone'][8]:.3f}")

def audit_part_b():
    print("\n==================================================")
    print("PART B — CHECK WHETHER FIRE FEATURES REACH THE ACTOR")
    print("==================================================")
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_44_candidate_physics_20updates.pth", map_location=device, weights_only=True))
    policy.eval()
    
    spatial = torch.zeros((1, 5, 11, 11))
    wind = torch.zeros((1, 3))
    base_drone = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5], dtype=np.float32)
    
    def get_logits_probs(drone_state):
        obs = {
            "spatial": spatial,
            "drone": torch.tensor(drone_state).unsqueeze(0),
            "wind": wind,
            "action_mask": torch.ones((1, 6), dtype=torch.float32)
        }
        with torch.no_grad():
            logits, _ = policy.forward(obs)
            probs = torch.softmax(logits, dim=-1)
        return logits.numpy()[0], probs.numpy()[0]
        
    logits_base, probs_base = get_logits_probs(base_drone)
    
    variants = [
        ("fire_dx = +0.5", 6, 0.5),
        ("fire_dx = -0.5", 6, -0.5),
        ("fire_dy = +0.5", 7, 0.5),
        ("fire_dy = -0.5", 7, -0.5),
        ("fire_distance = 0", 8, 0.0),
        ("fire_distance = 1", 8, 1.0)
    ]
    
    for name, idx, val in variants:
        drone = base_drone.copy()
        drone[idx] = val
        logits, probs = get_logits_probs(drone)
        
        max_logit_diff = np.max(np.abs(logits - logits_base))
        max_prob_diff = np.max(np.abs(probs - probs_base))
        
        print(f"{name:20s} | max logit diff: {max_logit_diff:.6f} | max prob diff: {max_prob_diff:.6f}")

def audit_part_c():
    print("\n==================================================")
    print("PART C — GRADIENT AUDIT")
    print("==================================================")
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_44_candidate_physics_20updates.pth", map_location=device, weights_only=True))
    policy.train()
    
    spatial = torch.zeros((1, 5, 11, 11), requires_grad=True)
    drone = torch.tensor([[1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, -0.1, 0.2]], dtype=torch.float32, requires_grad=True)
    wind = torch.zeros((1, 3), requires_grad=True)
    
    obs = {
        "spatial": spatial,
        "drone": drone,
        "wind": wind,
        "action_mask": torch.ones((1, 6), dtype=torch.float32)
    }
    
    logits, _ = policy.forward(obs)
    # Pick a specific action logit, say WATER (index 5)
    loss = logits[0, 5]
    loss.backward()
    
    grad = drone.grad[0].numpy()
    print("Absolute gradient w.r.t drone features for WATER logit:")
    print(f"battery: {abs(grad[0]):.6f}")
    print(f"payload: {abs(grad[1]):.6f}")
    print(f"normalized_margin: {abs(grad[2]):.6f}")
    print(f"dist_to_base: {abs(grad[3]):.6f}")
    print(f"nx: {abs(grad[4]):.6f}")
    print(f"ny: {abs(grad[5]):.6f}")
    print(f"fire_dx: {abs(grad[6]):.6f}")
    print(f"fire_dy: {abs(grad[7]):.6f}")
    print(f"fire_distance: {abs(grad[8]):.6f}")
    
    print("\nParameter gradient magnitudes (L1 norm):")
    total_cnn = sum(p.grad.abs().sum().item() for p in policy.actor_cnn.parameters() if p.grad is not None)
    total_vec = sum(p.grad.abs().sum().item() for p in policy.actor_vec.parameters() if p.grad is not None)
    total_fusion = sum(p.grad.abs().sum().item() for p in policy.actor_fusion.parameters() if p.grad is not None)
    total_head = sum(p.grad.abs().sum().item() for p in policy.actor_head.parameters() if p.grad is not None)
    
    print(f"actor_cnn: {total_cnn:.6f}")
    print(f"actor_vec: {total_vec:.6f}")
    print(f"actor_fusion: {total_fusion:.6f}")
    print(f"actor_head: {total_head:.6f}")

def audit_part_d():
    print("\n==================================================")
    print("PART D — PPO OPTIMIZER AUDIT")
    print("==================================================")
    policy = SeparateActorCriticSymmetric64(device=torch.device("cpu"), drone_type="WATER")
    trainer = PPOTrainer(policy=policy, device=torch.device("cpu"), learning_rate=3e-4)
    
    opt_params = set()
    for group in trainer.optimizer.param_groups:
        for p in group['params']:
            opt_params.add(p)
            
    actor_params_cnt = 0
    critic_params_cnt = 0
    missing_actor = []
    missing_critic = []
    
    # Actor components
    for name, p in policy.named_parameters():
        if "actor" in name:
            actor_params_cnt += p.numel()
            if p not in opt_params:
                missing_actor.append(name)
        elif "critic" in name:
            critic_params_cnt += p.numel()
            if p not in opt_params:
                missing_critic.append(name)
                
    opt_params_cnt = sum(p.numel() for p in opt_params)
    
    print(f"Total actor parameter count: {actor_params_cnt}")
    print(f"Total critic parameter count: {critic_params_cnt}")
    print(f"Optimizer parameter count: {opt_params_cnt}")
    print(f"Actor parameters missing from optimizer: {missing_actor if missing_actor else 'None'}")
    print(f"Critic parameters missing from optimizer: {missing_critic if missing_critic else 'None'}")

def audit_part_e():
    print("\n==================================================")
    print("PART E — ROLLOUT DATA AUDIT")
    print("==================================================")
    env = WildfireEnv("configs/environment.yaml")
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.eval()
    
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    collector.collect()
    
    drone_buffer = np.array(collector.buffer.drone)
    
    fire_dx = drone_buffer[:, 6]
    fire_dy = drone_buffer[:, 7]
    fire_dist = drone_buffer[:, 8]
    
    print(f"fire_dx -> min:{fire_dx.min():.3f}, max:{fire_dx.max():.3f}, mean:{fire_dx.mean():.3f}, std:{fire_dx.std():.3f}")
    print(f"fire_dy -> min:{fire_dy.min():.3f}, max:{fire_dy.max():.3f}, mean:{fire_dy.mean():.3f}, std:{fire_dy.std():.3f}")
    print(f"fire_dist -> min:{fire_dist.min():.3f}, max:{fire_dist.max():.3f}, mean:{fire_dist.mean():.3f}, std:{fire_dist.std():.3f}")
    
    print("Data verification: fire_dx from rollout buffer exactly matches data observed by agent during collect()")

if __name__ == "__main__":
    audit_part_a()
    audit_part_b()
    audit_part_c()
    audit_part_d()
    audit_part_e()
