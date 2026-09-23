from wildfire.rl.train import train_ppo
import torch

def run():
    print("Starting 20-update PPO training...")
    policy, _, _, _ = train_ppo(
        updates=20,
        rollout_size=256,
        ppo_epochs=4,
        minibatch_size=64,
        learning_rate=3e-4,
        device_name="cpu"
    )
    
    torch.save(policy.state_dict(), "stage7_48_reward_causal_20updates.pth")
    print("Saved stage7_48_reward_causal_20updates.pth")

if __name__ == "__main__":
    run()
