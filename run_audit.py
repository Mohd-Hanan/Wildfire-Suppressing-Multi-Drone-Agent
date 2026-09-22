from wildfire.rl.train import train_ppo
import sys

# Redirect stdout to a unique file
with open('diagnostic_training_stage7_7_audit.log', 'w') as f:
    sys.stdout = f
    train_ppo(
        updates=20,
        rollout_size=256,
        ppo_epochs=4,
        minibatch_size=64,
        seed=42,
        device_name="cpu"
    )
    sys.stdout = sys.__stdout__
