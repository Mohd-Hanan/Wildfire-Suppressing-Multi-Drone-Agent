import torch
import torch.nn as nn
import numpy as np

class SeparateActorCriticVectorMLP(nn.Module):
    def __init__(self, device=torch.device("cpu"), drone_type="WATER"):
        super(SeparateActorCriticVectorMLP, self).__init__()
        self.device = device
        self.drone_type = drone_type
        
        # Actor Spatial Branch
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Actor Vector Branch
        self.actor_vector_mlp = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        
        # Actor Fusion & Head
        self.actor_fusion = nn.Sequential(
            nn.Linear(32*11*11 + 64, 64), nn.ReLU()
        )
        self.actor_head = nn.Linear(64, 7)
        
        # Critic Spatial Branch
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Critic Vector Branch
        self.critic_vector_mlp = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        
        # Critic Fusion & Head
        self.critic_fusion = nn.Sequential(
            nn.Linear(32*11*11 + 64, 64), nn.ReLU()
        )
        self.critic_head = nn.Linear(64, 1)

        self.to(self.device)
        
    def get_action_mask(self, batch_size):
        mask = torch.ones((batch_size, 7), dtype=torch.bool, device=self.device)
        if self.drone_type == "WATER":
            mask[:, 6] = False # Mask retardant
        elif self.drone_type == "RETARDANT":
            mask[:, 5] = False # Mask water
        return mask

    def forward(self, obs):
        spatial = obs["spatial"]
        drone = obs["drone"]
        wind = obs["wind"]
        
        vec_input = torch.cat([drone, wind], dim=1)
        
        # Actor
        a_cnn_out = self.actor_cnn(spatial)
        a_vec_out = self.actor_vector_mlp(vec_input)
        a_fused = torch.cat([a_cnn_out, a_vec_out], dim=1)
        a_hidden = self.actor_fusion(a_fused)
        action_logits = self.actor_head(a_hidden)
        
        mask = self.get_action_mask(action_logits.shape[0])
        action_logits = torch.where(mask, action_logits, torch.tensor(-1e9, device=self.device))
        
        # Critic
        c_cnn_out = self.critic_cnn(spatial)
        c_vec_out = self.critic_vector_mlp(vec_input)
        c_fused = torch.cat([c_cnn_out, c_vec_out], dim=1)
        c_hidden = self.critic_fusion(c_fused)
        state_value = self.critic_head(c_hidden)
        
        return action_logits, state_value


def run_sensitivity_analysis():
    print("--- SENSITIVITY ANALYSIS (Vector MLP Architecture) ---")
    device = torch.device("cpu")
    torch.manual_seed(42)
    policy = SeparateActorCriticVectorMLP(device=device, drone_type="WATER")
    policy.eval()
    
    spatial = torch.zeros((1, 5, 11, 11), dtype=torch.float32)
    wind = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    
    def get_max_diff(base_drone, mod_drone):
        with torch.no_grad():
            l_base, _ = policy({"spatial": spatial, "wind": wind, "drone": base_drone})
            l_mod, _ = policy({"spatial": spatial, "wind": wind, "drone": mod_drone})
            p_base = torch.softmax(l_base, dim=-1)
            p_mod = torch.softmax(l_mod, dim=-1)
            logit_diff = torch.max(torch.abs(l_base - l_mod)).item()
            prob_diff = torch.max(torch.abs(p_base - p_mod)).item()
            return logit_diff, prob_diff
            
    base = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5]], dtype=torch.float32)
    
    # modify dx only
    mod_dx = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5]], dtype=torch.float32)
    # modify dy only
    mod_dy = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.5]], dtype=torch.float32)
    # modify dist only
    mod_dist = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0]], dtype=torch.float32)
    
    ld, pd = get_max_diff(base, mod_dx)
    print(f"  fire_dx change (0.0 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")
    ld, pd = get_max_diff(base, mod_dy)
    print(f"  fire_dy change (0.0 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")
    ld, pd = get_max_diff(base, mod_dist)
    print(f"  fire_distance (0.5 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")

    print("\n--- SIX DIRECTIONAL VECTORS (Untrained Vector MLP) ---")
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
        obs_t = {"spatial": spatial, "wind": wind, "drone": drone}
        
        with torch.no_grad():
            logits, _ = policy(obs_t)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            
        print(f"Fire {s_name} (dx={dx:+.1f}, dy={dy:+.1f}): " + 
              ", ".join([f"{action_names[i]}={probs[i]*100:4.1f}%" for i in range(6)]))

if __name__ == "__main__":
    run_sensitivity_analysis()

class SeparateActorCriticSymmetric(nn.Module):
    def __init__(self, device=torch.device("cpu"), drone_type="WATER"):
        super(SeparateActorCriticSymmetric, self).__init__()
        self.device = device
        self.drone_type = drone_type
        
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*11*11, 64), nn.ReLU() # Compress CNN to 64
        )
        self.actor_vec = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.actor_head = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(),
            nn.Linear(64, 7)
        )
        self.to(self.device)
        
    def get_action_mask(self, batch_size):
        mask = torch.ones((batch_size, 7), dtype=torch.bool, device=self.device)
        mask[:, 6] = False
        return mask

    def forward(self, obs):
        spatial = obs["spatial"]
        vec_input = torch.cat([obs["drone"], obs["wind"]], dim=1)
        a_cnn = self.actor_cnn(spatial)
        a_vec = self.actor_vec(vec_input)
        a_fused = torch.cat([a_cnn, a_vec], dim=1)
        action_logits = self.actor_head(a_fused)
        mask = self.get_action_mask(action_logits.shape[0])
        action_logits = torch.where(mask, action_logits, torch.tensor(-1e9, device=self.device))
        return action_logits, None

def run_symmetric_test():
    print("\n--- SENSITIVITY ANALYSIS (Symmetric 64+64 Architecture) ---")
    device = torch.device("cpu")
    torch.manual_seed(42)
    policy = SeparateActorCriticSymmetric(device=device, drone_type="WATER")
    policy.eval()
    
    spatial = torch.zeros((1, 5, 11, 11), dtype=torch.float32)
    wind = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    
    def get_max_diff(base_drone, mod_drone):
        with torch.no_grad():
            l_base, _ = policy({"spatial": spatial, "wind": wind, "drone": base_drone})
            l_mod, _ = policy({"spatial": spatial, "wind": wind, "drone": mod_drone})
            p_base = torch.softmax(l_base, dim=-1)
            p_mod = torch.softmax(l_mod, dim=-1)
            logit_diff = torch.max(torch.abs(l_base - l_mod)).item()
            prob_diff = torch.max(torch.abs(p_base - p_mod)).item()
            return logit_diff, prob_diff
            
    base = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5]], dtype=torch.float32)
    mod_dx = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5]], dtype=torch.float32)
    mod_dy = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.5]], dtype=torch.float32)
    mod_dist = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0]], dtype=torch.float32)
    
    ld, pd = get_max_diff(base, mod_dx)
    print(f"  fire_dx change (0.0 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")
    ld, pd = get_max_diff(base, mod_dy)
    print(f"  fire_dy change (0.0 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")
    ld, pd = get_max_diff(base, mod_dist)
    print(f"  fire_distance (0.5 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")

    print("\n--- SIX DIRECTIONAL VECTORS (Untrained Symmetric 64+64) ---")
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
    run_symmetric_test()
