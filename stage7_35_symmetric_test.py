import torch
import torch.nn as nn
import numpy as np

class SeparateActorCriticSymmetric64(nn.Module):
    def __init__(self, device=torch.device("cpu"), drone_type="WATER"):
        super(SeparateActorCriticSymmetric64, self).__init__()
        self.device = device
        self.drone_type = drone_type
        
        # Spatial branch
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*11*11, 64), nn.ReLU()
        )
        
        # Global vector branch
        self.actor_vec = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        
        # Fusion branch
        self.actor_fusion = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU()
        )
        self.actor_head = nn.Linear(64, 7)
        
        # Critic branches (identical structure)
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
            nn.Linear(64 + 64, 64), nn.ReLU()
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
        a_cnn = self.actor_cnn(spatial)
        a_vec = self.actor_vec(vec_input)
        a_fused = torch.cat([a_cnn, a_vec], dim=1)
        a_hidden = self.actor_fusion(a_fused)
        action_logits = self.actor_head(a_hidden)
        
        mask = self.get_action_mask(action_logits.shape[0])
        action_logits = torch.where(mask, action_logits, torch.tensor(-1e9, device=self.device))
        
        # Critic
        c_cnn = self.critic_cnn(spatial)
        c_vec = self.critic_vec(vec_input)
        c_fused = torch.cat([c_cnn, c_vec], dim=1)
        c_hidden = self.critic_fusion(c_fused)
        state_value = self.critic_head(c_hidden)
        
        return action_logits, state_value


def run_tests():
    print("--- UNIT / SHAPE TESTS ---")
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    
    bs = 4
    spatial = torch.zeros((bs, 5, 11, 11), dtype=torch.float32)
    drone = torch.zeros((bs, 9), dtype=torch.float32)
    wind = torch.zeros((bs, 3), dtype=torch.float32)
    
    obs = {"spatial": spatial, "drone": drone, "wind": wind}
    logits, values = policy(obs)
    
    print(f"Logits shape: {logits.shape} (Expected: {bs}, 7)")
    print(f"Values shape: {values.shape} (Expected: {bs}, 1)")
    print("Masking test (WATER): Retardant logit =", logits[0, 6].item(), "(Expected: -1e9)")
    
    print("\n--- GRADIENT / SENSITIVITY DIAGNOSTIC ---")
    # Single batch for gradient analysis
    spatial = torch.rand((1, 5, 11, 11), dtype=torch.float32, requires_grad=True)
    drone = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    wind = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    
    # We want to measure the gradient of each unmasked action logit w.r.t the inputs
    actions = [0, 1, 2, 3, 4, 5]
    for act in actions:
        # Zero gradients
        if spatial.grad is not None: spatial.grad.zero_()
        if drone.grad is not None: drone.grad.zero_()
        
        logits, _ = policy({"spatial": spatial, "drone": drone, "wind": wind})
        logit = logits[0, act]
        logit.backward()
        
        grad_dx = drone.grad[0, 6].item()
        grad_dy = drone.grad[0, 7].item()
        grad_dist = drone.grad[0, 8].item()
        
        # Representative spatial gradient (mean absolute gradient across all pixels)
        grad_spatial_mean = spatial.grad.abs().mean().item()
        grad_spatial_max = spatial.grad.abs().max().item()
        
        print(f"Action {act}:")
        print(f"  fire_dx grad:      {grad_dx:10.6f}")
        print(f"  fire_dy grad:      {grad_dy:10.6f}")
        print(f"  fire_dist grad:    {grad_dist:10.6f}")
        print(f"  Spatial mean abs:  {grad_spatial_mean:10.6f} | max abs: {grad_spatial_max:10.6f}")

    print("\n--- SYNTHETIC FIRE DIRECTIONS BASELINE ---")
    scenarios = [
        ("EAST ",  0.8,  0.0),
        ("WEST ", -0.8,  0.0),
        ("SOUTH",  0.0,  0.8),
        ("NORTH",  0.0, -0.8),
        ("SE   ",  0.8,  0.8),
        ("NW   ", -0.8, -0.8)
    ]
    
    action_names = ["STAY ", "NORTH", "SOUTH", "EAST ", "WEST ", "WATER", "RETAR"]
    policy.eval()
    
    for s_name, dx, dy in scenarios:
        syn_drone = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, dx, dy, 0.8]], dtype=torch.float32)
        with torch.no_grad():
            logits, _ = policy({"spatial": spatial, "drone": syn_drone, "wind": wind})
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        print(f"Fire {s_name} (dx={dx:+.1f}, dy={dy:+.1f}): " + 
              ", ".join([f"{action_names[i]}={probs[i]*100:4.1f}%" for i in range(6)]))

if __name__ == "__main__":
    torch.manual_seed(42) # For reproducible untrained weights
    run_tests()
