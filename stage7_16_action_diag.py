import torch
import torch.nn.functional as F
import numpy as np
import inspect
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.environment.action import ActionExecutor
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from stage7_15_smoke_test import SeparateActorCritic, SmokeTestTrainer

def main():
    print("========================================")
    print("STAGE 7.16 — ACTION & LOGIT DIAGNOSTIC")
    print("========================================")
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCritic(device=device)
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    # 1. Train for 10 updates
    print("\nTraining started (10 updates)...")
    for update in range(1, 11):
        policy.eval()
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
    print("Training complete.")

    # 2. Print initial observation logits/probs
    eval_env = WildfireEnv()
    obs, _ = eval_env.reset(seed=1000)
    
    policy.eval()
    with torch.no_grad():
        obs_t = {
            "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
            "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
            "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
        }
        logits, _ = policy(obs_t)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        logits = logits.squeeze(0).numpy()
        
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    print("\n--- POLICY LOGITS & PROBABILITIES (At Base (2,2)) ---")
    for i in range(7):
        print(f"Action {i} ({action_names[i]:<9}): Logit = {logits[i]:>8.4f} | Prob = {probs[i]*100:>6.2f}%")
        
    print(f"\nArgmax Action: {np.argmax(probs)} ({action_names[np.argmax(probs)]})")
    
    # 3 & 4. Verify Action Mapping Code
    print("\n--- ACTION MAPPING VERIFICATION ---")
    source = inspect.getsource(ActionExecutor.execute)
    print(source)

    # 5 & 6. Test all 7 actions manually
    print("\n--- MANUAL ACTION TESTING (from (2,2)) ---")
    for action_id in range(7):
        eval_env.reset(seed=1000) # Fresh start at (2,2)
        controlled_drone = eval_env.world.drones[eval_env.controlled_drone_idx]
        pre_x, pre_y = controlled_drone.x, controlled_drone.y
        pre_bat, pre_pay = controlled_drone.battery, controlled_drone.payload
        pre_act = controlled_drone.active
        
        _, reward, _, _, _ = eval_env.step(action_id)
        
        post_x, post_y = controlled_drone.x, controlled_drone.y
        post_bat, post_pay = controlled_drone.battery, controlled_drone.payload
        post_act = controlled_drone.active
        
        print(f"Action {action_id} ({action_names[action_id]:<9}):")
        print(f"  Pos: ({pre_x},{pre_y}) -> ({post_x},{post_y}) | Bat: {pre_bat} -> {post_bat} | Pay: {pre_pay} -> {post_pay} | Act: {pre_act} -> {post_act} | Reward: {reward:.2f}")

    # 7. Action 5 (WATER) and Action 6 (RETARDANT) detailed intermediate state
    print("\n--- DEPLOYMENT VS BASE REFILL INTERMEDIATE STATE ---")
    for action_id in [5, 6]:
        eval_env.reset(seed=1000)
        drone = eval_env.world.drones[eval_env.controlled_drone_idx]
        
        print(f"Action {action_id} ({action_names[action_id]}):")
        print(f"  [Start]       Bat: {drone.battery}, Pay: {drone.payload}")
        
        # Execute deployment ONLY
        eval_env.action_executor.execute(drone, eval_env.world, action_id)
        print(f"  [After Drop]  Bat: {drone.battery}, Pay: {drone.payload}")
        
        # Execute refill ONLY
        eval_env.world.process_base_refills()
        print(f"  [After Fill]  Bat: {drone.battery}, Pay: {drone.payload}")

if __name__ == "__main__":
    main()
