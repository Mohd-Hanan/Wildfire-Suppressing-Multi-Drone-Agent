import torch
import numpy as np
from stage7_15_smoke_test import run_ppo_training

def main():
    print("========================================")
    print("STAGE 7.23 — SINGLE-DRONE 200-UPDATE BASELINE")
    print("========================================")
    
    checkpoint_path = "stage7_single_drone_200_baseline.pth"
    print(f"Training for 200 updates. Model will be saved to: {checkpoint_path}")
    
    # Run training for 200 updates
    # The run_ppo_training function in stage7_15_smoke_test takes num_updates
    # We will import and use it.
    policy = run_ppo_training(num_updates=200)
    
    # Save checkpoint
    torch.save({
        'model_state_dict': policy.state_dict(),
        'num_updates': 200,
        'architecture': 'SeparateActorCritic',
        'drone_type': 'WATER',
        'notes': 'Single-drone baseline for future swarm transfer'
    }, checkpoint_path)
    
    print(f"Training complete. Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
