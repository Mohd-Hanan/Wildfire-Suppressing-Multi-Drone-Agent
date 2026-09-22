import torch

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
):
    """
    Computes Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: [T] tensor of step rewards
        values: [T] tensor of state values
        terminated: [T] boolean tensor of natural terminations
        truncated: [T] boolean tensor of time-limit truncations
        last_value: scalar float for bootstrapping the very end of the rollout
        gamma: discount factor
        gae_lambda: GAE decay parameter
        
    Returns:
        advantages: [T] tensor of advantages
        returns: [T] tensor of returns (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
            
        # IMPORTANT:
        # terminated=True => bootstrap_mask = 0.0 (no bootstrap)
        # truncated=True => bootstrap_mask = 1.0 (do bootstrap)
        # ordinary mid-episode => bootstrap_mask = 1.0 (do bootstrap)
        bootstrap_mask = 0.0 if terminated[t].item() else 1.0
        
        delta = rewards[t] + gamma * next_value * bootstrap_mask - values[t]
        last_gae = delta + gamma * gae_lambda * bootstrap_mask * last_gae
        
        advantages[t] = last_gae
        
    returns = advantages + values
    return advantages, returns
