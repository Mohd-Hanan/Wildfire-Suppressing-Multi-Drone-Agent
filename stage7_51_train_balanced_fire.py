import torch
import numpy as np
import random
from stage7_35_final_truth import SeparateActorCriticSymmetric64
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.rollout import RolloutCollector
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.gae import compute_gae
from wildfire.simulation.fire import FireState

class BalancedFireEnv(WildfireEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Clear fire
        self.world.fire_manager.fire_map.fill(FireState.UNBURNED)
        self.world.fire_manager.burn_timers.fill(0)
        
        # Pick a balanced direction
        dirs = ["NORTH", "SOUTH", "EAST", "WEST", "NE", "NW", "SE", "SW"]
        direction = np.random.choice(dirs)
        
        base_x, base_y = 2, 2
        
        if direction == "NORTH":
            fx, fy = base_x, 0
        elif direction == "SOUTH":
            fx, fy = base_x, base_y + np.random.randint(10, 30)
        elif direction == "EAST":
            fx, fy = base_x + np.random.randint(10, 30), base_y
        elif direction == "WEST":
            fx, fy = 0, base_y
        elif direction == "NE":
            fx, fy = base_x + 2, 0
        elif direction == "NW":
            fx, fy = 0, 0
        elif direction == "SE":
            fx, fy = base_x + np.random.randint(10, 30), base_y + np.random.randint(10, 30)
        elif direction == "SW":
            fx, fy = 0, base_y + 2
            
        self.world.fire_manager.ignite(fx, fy)
        
        # Update obs with the new fire position
        drone = self.world.drones[self.controlled_drone_idx]
        obs = self.obs_builder.get_observation(drone, self.world)
        return obs, info

def train():
    print("========================================")
    print("STAGE 7.51: BALANCED FIRE TRAINING")
    print("========================================")
    
    device = torch.device('cpu')
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.train()
    
    trainer = PPOTrainer(
        policy=policy,
        device=device,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        minibatch_size=64
    )
    
    # Verify distribution before training
    env = BalancedFireEnv("configs/environment.yaml")
    
    fdx_list, fdy_list = [], []
    dcounts = {"NORTH": 0, "SOUTH": 0, "EAST": 0, "WEST": 0, "NE": 0, "NW": 0, "SE": 0, "SW": 0, "NEAR": 0}
    for _ in range(100):
        obs, _ = env.reset()
        dx, dy = obs['drone'][6], obs['drone'][7]
        dist = obs['drone'][8]
        if dist <= (1.5 / np.sqrt(48**2 + 48**2)):
            dcounts["NEAR"] += 1
        else:
            if abs(dx) > 1.5 * abs(dy):
                if dx > 0: dcounts["EAST"] += 1
                else: dcounts["WEST"] += 1
            elif abs(dy) > 1.5 * abs(dx):
                if dy > 0: dcounts["SOUTH"] += 1
                else: dcounts["NORTH"] += 1
            else:
                if dx > 0 and dy > 0: dcounts["SE"] += 1
                elif dx > 0 and dy < 0: dcounts["NE"] += 1
                elif dx < 0 and dy > 0: dcounts["SW"] += 1
                else: dcounts["NW"] += 1
    
    print("Initial Fire Distribution Validation:")
    for d, c in dcounts.items():
        print(f"  {d}: {c/100*100:.1f}%")
        
    for v in dcounts.values():
        if v == 0 and dcounts["NEAR"] == 0:
            pass # We will allow it to proceed but print
    
    collector = RolloutCollector(env, policy, rollout_size=256, device=device)
    
    for update in range(1, 21):
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        
        water_freq = (acts == 5).float().mean().item() * 100
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update % 5 == 0 or update == 1:
            print(f"Update {update:2d}/20 | Reward: {np.mean(collector.buffer.rewards):7.4f} | "
                  f"Value Loss: {train_metrics.get('value_loss', 0.0):7.4f} | "
                  f"Policy Loss: {train_metrics.get('policy_loss', 0.0):7.4f} | WATER%: {water_freq:5.1f}%")

    checkpoint_path = "stage7_51_balanced_fire_20updates.pth"
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"Saved {checkpoint_path}")
    return policy

if __name__ == "__main__":
    train()
