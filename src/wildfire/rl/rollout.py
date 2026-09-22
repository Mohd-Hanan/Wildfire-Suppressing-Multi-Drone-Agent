import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, size: int):
        self.size = size
        self.clear()
        
    def add(self, observation, action, reward, terminated, truncated, value, log_prob):
        self.spatial.append(np.copy(observation["spatial"]).astype(np.float32))
        self.drone.append(np.copy(observation["drone"]).astype(np.float32))
        self.wind.append(np.copy(observation["wind"]).astype(np.float32))
        
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        
    def clear(self):
        self.spatial = []
        self.drone = []
        self.wind = []
        
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.values = []
        self.log_probs = []
        
    def __len__(self):
        return len(self.actions)
        
    def get_batch(self, device=torch.device("cpu")):
        spatial = torch.tensor(np.array(self.spatial), dtype=torch.float32, device=device)
        drone = torch.tensor(np.array(self.drone), dtype=torch.float32, device=device)
        wind = torch.tensor(np.array(self.wind), dtype=torch.float32, device=device)
        
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        terminated = torch.tensor(self.terminated, dtype=torch.bool, device=device)
        truncated = torch.tensor(self.truncated, dtype=torch.bool, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        
        return spatial, drone, wind, actions, rewards, terminated, truncated, values, log_probs

class RolloutCollector:
    def __init__(self, env, policy, rollout_size: int, device=torch.device("cpu")):
        self.env = env
        self.policy = policy
        self.rollout_size = rollout_size
        self.device = device
        
        self.buffer = RolloutBuffer(rollout_size)
        self.current_obs, _ = self.env.reset()
        self.last_obs = None
        self.last_value = None
        
    def collect(self):
        self.buffer.clear()
        self.last_obs = None
        self.last_value = None
        
        training_mode_before = self.policy.training
        self.policy.eval()
        
        steps = 0
        with torch.no_grad():
            while steps < self.rollout_size:
                action_tensor, log_prob, entropy, value = self.policy.get_action_and_value(self.current_obs)
                
                action = action_tensor.item()
                val = value.item()
                lp = log_prob.item()
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.buffer.add(
                    observation=self.current_obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    value=val,
                    log_prob=lp
                )
                
                steps += 1
                
                # If this is the final step of the rollout, capture last_obs BEFORE any reset
                if steps == self.rollout_size:
                    self.last_obs = next_obs
                    # Calculate last_value immediately
                    _, val_tensor = self.policy(self.last_obs)
                    self.last_value = val_tensor.item()
                
                if terminated or truncated:
                    self.current_obs, _ = self.env.reset()
                else:
                    self.current_obs = next_obs
                    
        if training_mode_before:
            self.policy.train()
