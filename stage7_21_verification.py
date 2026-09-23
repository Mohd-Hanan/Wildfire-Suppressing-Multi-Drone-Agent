import torch
import numpy as np
from stage7_15_smoke_test import SeparateActorCritic
from wildfire.environment.wildfire_env import WildfireEnv

def verify_parameters():
    print("--- 2. PARAMETER VERIFICATION ---")
    device = torch.device("cpu")
    fresh_policy = SeparateActorCritic(device=device, drone_type="WATER")
    fresh_params = {k: v.clone() for k, v in fresh_policy.state_dict().items()}
    
    checkpoint = torch.load("stage7_18_checkpoint.pth", weights_only=True)
    fresh_policy.load_state_dict(checkpoint['model_state_dict'])
    loaded_params = fresh_policy.state_dict()
    
    changed_tensors = 0
    total_diff = 0.0
    max_diff = 0.0
    
    for k in fresh_params.keys():
        diff = torch.abs(fresh_params[k] - loaded_params[k])
        if torch.max(diff) > 1e-6:
            changed_tensors += 1
            total_diff += torch.sum(diff).item()
            max_diff = max(max_diff, torch.max(diff).item())
            
    print(f"Number of parameter tensors changed: {changed_tensors} / {len(fresh_params)}")
    print(f"Total absolute parameter difference: {total_diff:.4f}")
    print(f"Maximum parameter difference: {max_diff:.4f}")
    print(f"Checkpoint identical to fresh initialization? {'YES' if changed_tensors == 0 else 'NO'}")
    return fresh_policy

def verify_distribution(policy):
    print("\n--- 3. DISTRIBUTION VERIFICATION ---")
    env = WildfireEnv()
    obs, _ = env.reset(seed=999)
    
    obs_t = {
        "spatial": torch.tensor(obs["spatial"], dtype=torch.float32).unsqueeze(0),
        "drone": torch.tensor(obs["drone"], dtype=torch.float32).unsqueeze(0),
        "wind": torch.tensor(obs["wind"], dtype=torch.float32).unsqueeze(0),
    }
    
    policy.eval()
    with torch.no_grad():
        logits, _ = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs.squeeze(0).numpy()
        
    print("Network logits:")
    print(logits.squeeze(0).numpy())
    
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    print("\nMasked probabilities:")
    for i in range(7):
        print(f"{action_names[i]:<10}: {probs[i]:.6f}")
        
    print("\nSampling 1000 actions...")
    counts = np.zeros(7)
    with torch.no_grad():
        for _ in range(1000):
            counts[dist.sample().item()] += 1
            
    print("\nEmpirical vs Theoretical:")
    for i in range(7):
        emp = counts[i] / 1000.0
        print(f"{action_names[i]:<10} | Emp: {emp:.4f} | Theo: {probs[i]:.4f}")

def verify_reproducibility(policy):
    print("\n--- 4. REPRODUCIBILITY VERIFICATION ---")
    env = WildfireEnv()
    obs, _ = env.reset(seed=999)
    obs_t = {
        "spatial": torch.tensor(obs["spatial"], dtype=torch.float32).unsqueeze(0),
        "drone": torch.tensor(obs["drone"], dtype=torch.float32).unsqueeze(0),
        "wind": torch.tensor(obs["wind"], dtype=torch.float32).unsqueeze(0),
    }
    
    torch.manual_seed(123)
    seq1 = []
    with torch.no_grad():
        for _ in range(10):
            logits, _ = policy(obs_t)
            seq1.append(torch.distributions.Categorical(logits=logits).sample().item())
            
    torch.manual_seed(123)
    seq2 = []
    with torch.no_grad():
        for _ in range(10):
            logits, _ = policy(obs_t)
            seq2.append(torch.distributions.Categorical(logits=logits).sample().item())
            
    print(f"Sequence 1 (Seed 123): {seq1}")
    print(f"Sequence 2 (Seed 123): {seq2}")
    print(f"Sequences match exactly? {'YES' if seq1 == seq2 else 'NO'}")

def main():
    print("========================================")
    print("STAGE 7.21 — FORMAL POLICY VERIFICATION")
    print("========================================")
    policy = verify_parameters()
    verify_distribution(policy)
    verify_reproducibility(policy)

if __name__ == "__main__":
    main()
