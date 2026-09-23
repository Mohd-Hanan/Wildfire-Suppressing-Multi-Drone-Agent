import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper
from stage7_15_smoke_test import SmokeTestTrainer

class SeparateActorCriticSymmetric64(nn.Module):
    def __init__(self, device=torch.device("cpu"), drone_type="WATER"):
        super(SeparateActorCriticSymmetric64, self).__init__()
        self.device = device
        self.drone_type = drone_type
        
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*11*11, 64), nn.ReLU()
        )
        # Rename to actor_mlp so the trainer picks it up
        self.actor_mlp = nn.ModuleDict({
            'vec': nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()),
            'fusion': nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        })
        self.actor_head = nn.Linear(64, 7)
        
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*11*11, 64), nn.ReLU()
        )
        self.critic_mlp = nn.ModuleDict({
            'vec': nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()),
            'fusion': nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        })
        self.critic_head = nn.Linear(64, 1)

        self.to(self.device)
        
    def get_action_mask(self, batch_size):
        mask = torch.ones((batch_size, 7), dtype=torch.bool, device=self.device)
        if self.drone_type == "WATER":
            mask[:, 6] = False
        return mask

    def forward(self, obs):
        spatial = obs["spatial"]
        if not isinstance(spatial, torch.Tensor):
            spatial = torch.tensor(spatial, dtype=torch.float32, device=self.device)
            drone = torch.tensor(obs["drone"], dtype=torch.float32, device=self.device)
            wind = torch.tensor(obs["wind"], dtype=torch.float32, device=self.device)
        else:
            drone, wind = obs["drone"], obs["wind"]
            
        if spatial.dim() == 3:
            spatial, drone, wind = spatial.unsqueeze(0), drone.unsqueeze(0), wind.unsqueeze(0)

        vec_input = torch.cat([drone, wind], dim=1)
        
        a_cnn = self.actor_cnn(spatial)
        a_vec = self.actor_mlp['vec'](vec_input)
        a_fused = torch.cat([a_cnn, a_vec], dim=1)
        a_hidden = self.actor_mlp['fusion'](a_fused)
        action_logits = self.actor_head(a_hidden)
        
        mask = self.get_action_mask(action_logits.shape[0])
        action_logits = torch.where(mask, action_logits, torch.tensor(-1e9, device=self.device))
        
        c_cnn = self.critic_cnn(spatial)
        c_vec = self.critic_mlp['vec'](vec_input)
        c_fused = torch.cat([c_cnn, c_vec], dim=1)
        c_hidden = self.critic_mlp['fusion'](c_fused)
        state_value = self.critic_head(c_hidden)
        
        return action_logits, state_value

    def get_action_and_value(self, obs, action=None):
        action_logits, state_value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), state_value


def train():
    print("========================================")
    print("STAGE 7.35 — 20-UPDATE SYMMETRIC TRAINING")
    print("========================================")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    trainer = SmokeTestTrainer(policy=policy, device=device)
    collector = RolloutCollector(env=env, policy=policy, rollout_size=256, device=device)
    
    for update in range(1, 21):
        policy.eval()
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        
        water_freq = (acts == 5).float().mean().item() * 100
        advantages, returns = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, advantages, returns)
        
        if update % 5 == 0 or update == 1:
            print(f"Update {update:2d}/20 | Reward: {np.mean(collector.buffer.rewards):7.4f} | "
                  f"Entropy: {train_metrics['entropy']:6.4f} | Value Loss: {train_metrics['value_loss']:8.1f} | "
                  f"Policy Loss: {train_metrics.get('policy_loss', 0.0):7.4f} | WATER%: {water_freq:5.1f}%")

    checkpoint_path = "stage7_35_symmetric64_20updates.pth"
    torch.save({'model_state_dict': policy.state_dict()}, checkpoint_path)
    print(f"\n[+] Checkpoint saved to: {checkpoint_path}")
    return policy

def evaluate(policy):
    print("\n========================================")
    print("QUANTITATIVE EVALUATION: stage7_35_symmetric64_20updates.pth")
    print("========================================")
    env = WildfireEnv()
    device = torch.device("cpu")
    policy.eval()
    
    metrics = {
        'rewards': [], 'burned_cells': [], 'suppressed_cells': [], 
        'lengths': [], 'crashes': 0, 'extinctions': 0,
        'base_visits': 0, 'actions': Counter()
    }
    
    for ep in range(1, 21):
        obs, _ = env.reset(seed=ep * 100)
        ep_reward, ep_length, ep_burned, ep_suppressed = 0.0, 0, 0, 0
        drone = env.world.drones[env.controlled_drone_idx]
        was_at_base = True
        
        while True:
            with torch.no_grad():
                logits, _ = policy(obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                
            metrics['actions'][action] += 1
            
            is_at_base = (env.world.base_x <= drone.x < env.world.base_x + 2 and 
                          env.world.base_y <= drone.y < env.world.base_y + 2)
            if is_at_base and not was_at_base:
                metrics['base_visits'] += 1
            was_at_base = is_at_base
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_burned += info.get('new_burned_cells', 0)
            ep_suppressed += info.get('newly_suppressed_cells', 0)
            ep_length += 1
            
            crashed = not drone.active and not info.get('extinguished', False)
            if terminated or truncated or crashed:
                if crashed: metrics['crashes'] += 1
                if info.get('extinction_reward', 0) > 0: metrics['extinctions'] += 1
                
                metrics['rewards'].append(ep_reward)
                metrics['burned_cells'].append(ep_burned)
                metrics['suppressed_cells'].append(ep_suppressed)
                metrics['lengths'].append(ep_length)
                break

    print(f"Mean Reward:        {np.mean(metrics['rewards']):.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"Mean Burned Cells:  {np.mean(metrics['burned_cells']):.2f} ± {np.std(metrics['burned_cells']):.2f}")
    print(f"Mean Suppressed:    {np.mean(metrics['suppressed_cells']):.2f} ± {np.std(metrics['suppressed_cells']):.2f}")
    print(f"Total Successful Suppressions: {sum(metrics['suppressed_cells'])}")
    print(f"Total WATER Actions: {metrics['actions'][5]}")
    print(f"Extinction Rate:    {(metrics['extinctions'] / 20) * 100:.1f}%")
    print(f"Mean Ep Length:     {np.mean(metrics['lengths']):.1f}")
    print(f"Total Crashes:      {metrics['crashes']}")
    print(f"Total Base Visits:  {metrics['base_visits']}")
    
    total_actions = sum(metrics['actions'].values())
    print("\nAction Percentages:")
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    for i in range(7):
        pct = (metrics['actions'][i] / total_actions) * 100
        print(f"{action_names[i]:<10}: {pct:4.1f}%")

def sensitivity_test(policy):
    print("\n========================================")
    print("SYNTHETIC FIRE DIRECTIONS (TRAINED 20-UPDATE)")
    print("========================================")
    policy.eval()
    spatial = torch.zeros((1, 5, 11, 11), dtype=torch.float32)
    wind = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    
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
        with torch.no_grad():
            logits, _ = policy({"spatial": spatial, "wind": wind, "drone": drone})
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        print(f"Fire {s_name} (dx={dx:+.1f}, dy={dy:+.1f}): " + 
              ", ".join([f"{action_names[i]}={probs[i]*100:4.1f}%" for i in range(6)]))

if __name__ == "__main__":
    trained_policy = train()
    evaluate(trained_policy)
    sensitivity_test(trained_policy)
