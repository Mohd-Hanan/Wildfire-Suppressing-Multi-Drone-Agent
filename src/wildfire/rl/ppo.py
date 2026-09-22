import torch

def compute_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01
):
    # Policy Ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # PPO Clipped Surrogate Objective
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratio * advantages
    surrogate = torch.min(unclipped, clipped)
    policy_loss = -torch.mean(surrogate)
    
    # Value Loss (MSE)
    value_loss = torch.mean((returns - new_values) ** 2)
    
    # Entropy Bonus
    entropy_mean = torch.mean(entropy)
    
    # Total Loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    
    # Metrics
    approx_kl = torch.mean(old_log_probs - new_log_probs)
    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_epsilon).float())
    ratio_mean = torch.mean(ratio)
    
    metrics = {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_mean.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
        "ratio_mean": ratio_mean.item()
    }
    
    return total_loss, metrics
