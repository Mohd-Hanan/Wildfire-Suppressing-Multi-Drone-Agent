import torch
import numpy as np
from stage7_15_smoke_test import SeparateActorCritic

device = torch.device("cpu")

policy_200 = SeparateActorCritic(device=device, drone_type="WATER")
policy_200.load_state_dict(torch.load("stage7_single_drone_200_global_fire_reward.pth", map_location=device, weights_only=True)['model_state_dict'])
policy_200.eval()

policy_smoke = SeparateActorCritic(device=device, drone_type="WATER")
policy_smoke.load_state_dict(torch.load("stage7_32_smoke_global_fire_reward.pth", map_location=device, weights_only=True)['model_state_dict'])
policy_smoke.eval()

# Synthetic input
# spatial: 1 x 5 x 11 x 11 (all zeros is fine for testing pure non-spatial bias)
# wind: 1 x 3
# drone: 1 x 9
# Vector: [battery, payload, margin, dist_to_base, nx, ny, fire_dx, fire_dy, fire_distance]

spatial = torch.zeros((1, 5, 11, 11), dtype=torch.float32)
wind = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)

def test_policy(name, policy):
    print(f"\n--- {name} ---")
    scenarios = [
        ("EAST ",  0.8,  0.0),
        ("WEST ", -0.8,  0.0),
        ("SOUTH",  0.0,  0.8),
        ("NORTH",  0.0, -0.8),
        ("SE   ",  0.8,  0.8),
        ("NW   ", -0.8, -0.8)
    ]
    
    action_names = ["STAY ", "NORTH", "SOUTH", "EAST ", "WEST ", "WATER", "RETAR"]
    
    for s_name, dx, dy in scenarios:
        drone = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, dx, dy, 0.8]], dtype=torch.float32)
        obs_t = {"spatial": spatial, "wind": wind, "drone": drone}
        
        with torch.no_grad():
            logits, _ = policy(obs_t)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            
        print(f"Fire {s_name} (dx={dx:+.1f}, dy={dy:+.1f}): " + 
              ", ".join([f"{action_names[i]}={probs[i]*100:4.1f}%" for i in range(6)]))

test_policy("Untrained Smoke (20 updates)", policy_smoke)
test_policy("Trained (200 updates)", policy_200)
