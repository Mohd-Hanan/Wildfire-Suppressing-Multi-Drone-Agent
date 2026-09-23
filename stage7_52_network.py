import torch
import torch.nn as nn
from stage7_35_final_truth import SeparateActorCriticSymmetric64

class ActorCritic6Channels(SeparateActorCriticSymmetric64):
    def __init__(self, device=torch.device('cpu'), drone_type="WATER"):
        super().__init__(device=device, drone_type=drone_type)
        
        # Override CNNs to take 6 channels
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 64), nn.ReLU()
        )
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 64), nn.ReLU()
        )
        
        self.to(device)
