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

def audit_training():
    env = WildfireStatsWrapper(WildfireEnv())
    device = torch.device("cpu")
    
    torch.manual_seed(42)
    np.random.seed(42)
    policy = ActorCritic(device=device)
    initial_params = [p.clone() for p in policy.parameters()]
    
    env_eval = WildfireEnv()
    fixed_obs, _ = env_eval.reset(seed=1000)
    fixed_obs_t = {
        "spatial": torch.tensor(fixed_obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
        "drone": torch.tensor(fixed_obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
        "wind": torch.tensor(fixed_obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    
    def eval_fixed_obs(pol):
        pol.eval()
        with torch.no_grad():
            logits, val = pol(fixed_obs_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            return logits.numpy()[0], probs.numpy()[0], dist.entropy().item(), torch.argmax(logits, dim=-1).item()

    init_logits, init_probs, init_ent, init_argmax = eval_fixed_obs(policy)
    
    # Observation sanity
    spatial = fixed_obs["spatial"]
    drone_arr = fixed_obs["drone"]
    wind_arr = fixed_obs["wind"]
    
    print("\n--- OBSERVATION SANITY ---")
    print(f"Spatial min: {spatial.min()}, max: {spatial.max()}, mean: {spatial.mean()}, frac_nonzero: {np.count_nonzero(spatial)/spatial.size}")
    print(f"Drone array: {drone_arr}")
    print(f"Wind array: {wind_arr}")

    print("\n--- STOCHASTIC TRAINING DISTRIBUTION & FIXED LOGITS ---")
    print(f"Initial  | Probs: {[f'{p:.3f}' for p in init_probs]} | Ent: {init_ent:.3f}")
    
    trainer = PPOTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    for update in range(1, 21):
        policy.eval()
        collector.collect()
        
        actions = np.array(collector.buffer.actions)
        counts = [np.sum(actions == a) for a in range(7)]
        pcts = [c/len(actions)*100 for c in counts]
        
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value)
            
        policy.train()
        trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update in [1, 5, 10, 15, 20]:
            cl, cp, ce, ca = eval_fixed_obs(policy)
            print(f"Update {update:2d} | Probs: {[f'{p:.3f}' for p in cp]} | Ent: {ce:.3f} | Stoch Pcts: {[f'{p:.1f}%' for p in pcts]}")

    final_logits, final_probs, final_ent, final_argmax = eval_fixed_obs(policy)
    
    print("\n--- INITIAL VS TRAINED FIXED OBSERVATION ---")
    print(f"Initial Logits: {init_logits}")
    print(f"Initial Probs : {init_probs}")
    print(f"Initial Ent   : {init_ent:.4f}")
    print(f"Initial Argmax: {init_argmax}")
    print(f"Trained Logits: {final_logits}")
    print(f"Trained Probs : {final_probs}")
    print(f"Trained Ent   : {final_ent:.4f}")
    print(f"Trained Argmax: {final_argmax}")
    
    print("\n--- PARAMETER CHANGES ---")
    final_params = list(policy.parameters())
    total_params = sum(p.numel() for p in policy.parameters())
    num_changed = 0
    max_change = 0.0
    mean_change = 0.0
    
    for init_p, final_p in zip(initial_params, final_params):
        diff = torch.abs(final_p - init_p)
        num_changed += torch.sum(diff > 1e-6).item()
        max_change = max(max_change, diff.max().item())
        mean_change += diff.sum().item()
        
    print(f"Total Params: {total_params}")
    print(f"Num Changed: {num_changed}")
    print(f"Max Abs Change: {max_change:.6f}")
    print(f"Mean Abs Change: {mean_change / total_params:.6f}")

    print("\n--- CRASH TRANSITION AUDIT ---")
    env = WildfireEnv()
    obs, info = env.reset(seed=1000)
    drone = env.world.drones[env.controlled_drone_idx]
    
    policy.eval()
    step = 0
    
    while True:
        with torch.no_grad():
            obs_t = {
                "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
            }
            logits, _ = policy(obs_t)
            action = torch.argmax(logits, dim=-1).item()
            
        pre_x, pre_y, pre_bat, pre_active = drone.x, drone.y, drone.battery, drone.active
        obs, reward, terminated, truncated, step_info = env.step(action)
        step += 1
        
        if step > 65 and step <= 75:
            print(f"Step {step:2d} | Pos:({pre_x:2d},{pre_y:2d})->({drone.x:2d},{drone.y:2d}) | Bat: {pre_bat:5.1f}->{drone.battery:5.1f} | Active: {pre_active}->{drone.active} | Act: {action} | Rwd: {reward:.2f}")
            
        if not drone.active and pre_active:
            print(f"CRASH OCCURRED AT STEP {step}")
            print(f"Final Pos: ({drone.x}, {drone.y}) | Final Battery: {drone.battery} | Final Active: {drone.active}")
            break
            
        if terminated or truncated:
            break

audit_training()
