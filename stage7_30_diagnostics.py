import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv

def test_eastern_boundary_rewards():
    print("========================================")
    print("DIAGNOSTICS: EASTERN BOUNDARY REWARDS (Step 2)")
    print("========================================")
    
    env = WildfireEnv()
    
    # Place drone at eastern boundary
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    for action in [0, 1, 2, 3, 4]:
        obs, _ = env.reset(seed=42)
        drone = env.world.drones[env.controlled_drone_idx]
        drone.x = 47
        drone.y = 20
        drone.battery = 100
        
        # Step 1: STAY (initializes reward tracking)
        env.step(0)
        
        # Step 2: The actual action we want to test
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"Action: {action_names[action]:<6} | Pos: ({drone.x},{drone.y}) | Bat: {drone.battery:3d} | Reward: {reward:6.3f} | Boundary Penalty: {info['boundary_penalty']:5.2f}")
        
    print("\n========================================")
    print("DIAGNOSTICS: CRASH DETECTION TRACE")
    print("========================================")
    
    # Run an episode and trace exactly when it becomes inactive
    from stage7_15_smoke_test import SeparateActorCritic
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    checkpoint = torch.load("stage7_single_drone_200_boundary_penalty.pth", weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    obs, _ = env.reset(seed=4000) # Match visual watch ep 1 seed
    drone = env.world.drones[env.controlled_drone_idx]
    
    crashed_step = -1
    crash_reason = ""
    
    for step in range(500):
        with torch.no_grad():
            obs_t = {
                "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
            }
            logits, _ = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            
        was_active = drone.active
        bat_before = drone.battery
        
        obs, reward, term, trunc, info = env.step(action)
        
        if was_active and not drone.active:
            crashed_step = step + 1
            crash_reason = f"Battery depleted from {bat_before} to {drone.battery} trying to execute {action_names[action]}"
            print(f"CRASH DETECTED at Step {crashed_step}")
            print(f"Reason: {crash_reason}")
            print(f"Final Pos: ({drone.x},{drone.y})")
            print(f"Info dict keys related to crash: crash_penalty={info.get('crash_penalty', 0)}")
            break

if __name__ == "__main__":
    test_eastern_boundary_rewards()
