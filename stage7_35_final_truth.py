import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.rollout import RolloutCollector
from wildfire.rl.gae import compute_gae
from wildfire.rl.train import WildfireStatsWrapper

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
        self.actor_vec = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.actor_fusion = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU()
        )
        self.actor_head = nn.Linear(64, 7)
        
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*11*11, 64), nn.ReLU()
        )
        self.critic_vec = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.critic_fusion = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU()
        )
        self.critic_head = nn.Linear(64, 1)
        self.to(self.device)
        
    def get_action_mask(self, batch_size):
        mask = torch.ones((batch_size, 7), dtype=torch.bool, device=self.device)
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
        a_vec = self.actor_vec(vec_input)
        a_hidden = self.actor_fusion(torch.cat([a_cnn, a_vec], dim=1))
        action_logits = self.actor_head(a_hidden)
        action_logits = torch.where(self.get_action_mask(action_logits.shape[0]), action_logits, torch.tensor(-1e9, device=self.device))
        
        c_cnn = self.critic_cnn(spatial)
        c_vec = self.critic_vec(vec_input)
        c_hidden = self.critic_fusion(torch.cat([c_cnn, c_vec], dim=1))
        state_value = self.critic_head(c_hidden)
        
        return action_logits, state_value

    def get_action_and_value(self, obs, action=None):
        action_logits, state_value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), state_value

class BulletproofTrainer:
    def __init__(self, policy, device=torch.device("cpu")):
        self.policy = policy
        self.device = device
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.minibatch_size = 64
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    def _normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, spatial, drone, wind, actions, old_log_probs, advantages, returns):
        normalized_advantages = self._normalize_advantages(advantages)
        N = len(actions)
        indices = np.arange(N)
        agg_metrics = {}
        updates = 0
        self.policy.train()
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                
                mb_spatial = spatial[mb_idx]
                mb_drone = drone[mb_idx]
                mb_wind = wind[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = normalized_advantages[mb_idx]
                mb_returns = returns[mb_idx]

                obs_dict = {"spatial": mb_spatial, "drone": mb_drone, "wind": mb_wind}
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(obs_dict, mb_actions)
                new_values = new_values.view(-1)

                logratio = new_log_probs - mb_old_log_probs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.entropy_coef * entropy_loss + self.value_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                agg_metrics['policy_loss'] = agg_metrics.get('policy_loss', 0) + pg_loss.item()
                agg_metrics['value_loss'] = agg_metrics.get('value_loss', 0) + v_loss.item()
                agg_metrics['entropy'] = agg_metrics.get('entropy', 0) + entropy_loss.item()
                updates += 1

        for k in agg_metrics:
            agg_metrics[k] /= max(1, updates)
            
        return agg_metrics

def train():
    print("========================================")
    print("STAGE 7.35 — BULLETPROOF TRAINING")
    print("========================================")
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    env = WildfireStatsWrapper(WildfireEnv())
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    trainer = BulletproofTrainer(policy=policy, device=device)
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

    torch.save({'model_state_dict': policy.state_dict()}, "stage7_35_symmetric64_20updates.pth")
    return policy

def evaluate_and_test(policy):
    print("\n========================================")
    print("SYNTHETIC FIRE DIRECTIONS (TRAINED)")
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
    trained = train()
    evaluate_and_test(trained)
