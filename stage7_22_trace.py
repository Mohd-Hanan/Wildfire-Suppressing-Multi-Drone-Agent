import torch
import numpy as np
from stage7_15_smoke_test import SeparateActorCritic
from wildfire.environment.wildfire_env import WildfireEnv

def main():
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    checkpoint = torch.load("stage7_18_checkpoint.pth", weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    env = WildfireEnv()
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    # We will search for a seed that forces the drone to hit the base, 
    # to demonstrate the refill trace perfectly.
    for seed in range(5000, 6000):
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=4001)
        drone = env.world.drones[env.controlled_drone_idx]
        
        trace = []
        refill_occurred = False
        
        for step in range(30):
            with torch.no_grad():
                obs_t = {
                    "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                    "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                    "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
                }
                logits, _ = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                
            x_before, y_before = drone.x, drone.y
            bat_before, pay_before = drone.battery, drone.payload
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            x_after, y_after = drone.x, drone.y
            bat_after, pay_after = drone.battery, drone.payload
            
            is_at_base = env.world.is_at_base(drone)
            refilled = (is_at_base and bat_after == drone.max_battery and bat_before < drone.max_battery)
            
            if refilled:
                refill_occurred = True
                
            act_str = f"{action} ({action_names[action]})"
            pos_b = f"({x_before},{y_before})"
            pos_a = f"({x_after},{y_after})"
            
            trace.append(f"{step:4d} | {act_str:<15} | {pos_b:<10} | {pos_a:<10} | {bat_before:5d} | {bat_after:5d} | {pay_before:5d} | {pay_after:5d} | {str(is_at_base):<5} | {str(refilled):<5}")
            
        if refill_occurred:
            print("========================================")
            print("STAGE 7.22 — REFILL TRACE EXAMPLE")
            print("========================================")
            print(f"{'Step':>4} | {'Action':<15} | {'Pos Before':<10} | {'Pos After':<10} | {'Bat B':>5} | {'Bat A':>5} | {'Pay B':>5} | {'Pay A':>5} | {'Base?':<5} | {'Refill?':<5}")
            print("-" * 115)
            for line in trace:
                print(line)
            return

if __name__ == "__main__":
    main()
