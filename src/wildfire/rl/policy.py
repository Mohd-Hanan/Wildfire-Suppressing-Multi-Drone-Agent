import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class ActorCritic(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(ActorCritic, self).__init__()
        self.device = device
        
        # Spatial Encoder (CNN)
        # Input: [B, 5, 11, 11]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Output of Flatten: 32 * 11 * 11 = 3872 features
        cnn_out_features = 32 * 11 * 11
        non_spatial_features = 12
        total_features = cnn_out_features + non_spatial_features
        
        # Shared Feature Layer
        self.shared_mlp = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU()
        )
        
        # Actor Head (7 actions)
        self.actor = nn.Linear(64, 7)
        
        # Critic Head (Value)
        self.critic = nn.Linear(64, 1)

        self.to(self.device)
        
    def _preprocess_obs(self, obs):
        spatial = obs["spatial"]
        drone = obs["drone"]
        wind = obs["wind"]
        
        if not isinstance(spatial, torch.Tensor):
            spatial = torch.tensor(spatial, dtype=torch.float32, device=self.device)
        if not isinstance(drone, torch.Tensor):
            drone = torch.tensor(drone, dtype=torch.float32, device=self.device)
        if not isinstance(wind, torch.Tensor):
            wind = torch.tensor(wind, dtype=torch.float32, device=self.device)
            
        if spatial.dim() == 3:
            spatial = spatial.unsqueeze(0)
            drone = drone.unsqueeze(0)
            wind = wind.unsqueeze(0)
            
        return spatial, drone, wind

    def forward(self, obs):
        spatial, drone, wind = self._preprocess_obs(obs)
        
        cnn_features = self.cnn(spatial)
        
        non_spatial = torch.cat([drone, wind], dim=-1)
        combined = torch.cat([cnn_features, non_spatial], dim=-1)
        
        shared_features = self.shared_mlp(combined)
        
        logits = self.actor(shared_features)
        value = self.critic(shared_features).squeeze(-1)
        
        return logits, value
        
    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value
