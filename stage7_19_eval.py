import torch
import numpy as np
from stage7_17_eval import evaluate, print_metrics
from stage7_15_smoke_test import SeparateActorCritic
from wildfire.environment.wildfire_env import WildfireEnv

def main():
    print("========================================")
    print("STAGE 7.19 — AUTHORITATIVE EVALUATION")
    print("========================================")
    
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    
    checkpoint_path = "stage7_18_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    env = WildfireEnv()
    
    # Authoritative Stochastic Eval
    stoch_metrics = evaluate(policy, env, num_episodes=20, deterministic=False)
    print_metrics("STOCHASTIC EVALUATION (Seed 2000-2019)", stoch_metrics, 20)
    
    # Authoritative Deterministic Eval
    det_metrics = evaluate(policy, env, num_episodes=20, deterministic=True)
    print_metrics("DETERMINISTIC EVALUATION (Seed 2000-2019)", det_metrics, 20)

if __name__ == "__main__":
    main()
