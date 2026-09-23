import torch
import numpy as np
from stage7_52_network import ActorCritic6Channels
from stage7_52_obs_builder import GlobalFireObservationBuilder
from wildfire.simulation.world import World
from wildfire.simulation.drone import Drone, DroneType
from wildfire.simulation.fire import FireState
from wildfire.simulation.terrain import Terrain
from wildfire.simulation.fire import FireManager

# Setup
device = torch.device('cpu')
policy = ActorCritic6Channels(device=device, drone_type="WATER").to(device)

try:
    policy.load_state_dict(torch.load("stage7_52_global_fire_channel_200updates.pth", map_location=device, weights_only=True))
    policy.eval()
    print("Loaded Stage 7.52 checkpoint.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit(1)

from wildfire.environment.wildfire_env import WildfireEnv
env = WildfireEnv('configs/environment.yaml')
obs, _ = env.reset(seed=42)
drone = env.world.drones[0]
drone.x, drone.y = 24, 24
world = env.world

obs_builder = GlobalFireObservationBuilder(window_size=11)

cases = {
    "FIRE NORTH": (24, 14),
    "FIRE SOUTH": (24, 34),
    "FIRE EAST": (34, 24),
    "FIRE WEST": (14, 24),
    "FIRE NE": (34, 14),
    "FIRE NW": (14, 14),
    "FIRE SE": (34, 34),
    "FIRE SW": (14, 34)
}

print("\n--- DIAGNOSTIC A: SYNTHETIC DIRECTION RESPONSE ---")

for name, (fx, fy) in cases.items():
    world.fire_manager.fire_map.fill(FireState.UNBURNED)
    world.fire_manager.fire_map[fx, fy] = FireState.BURNING
    
    obs = obs_builder.get_observation(drone, world)
    
    spatial = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0).to(device)
    drone_vec = torch.tensor(obs['drone'], dtype=torch.float32).unsqueeze(0).to(device)
    wind_vec = torch.tensor(obs['wind'], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        action_logits, _ = policy({'spatial': spatial, 'drone': drone_vec, 'wind': wind_vec})
        action_logits[0, 6] = -1e9
        probs = torch.softmax(action_logits, dim=-1)[0].numpy()
        
    print(f"{name}:")
    print(f"  STAY  {probs[0]*100:.1f}%")
    print(f"  NORTH {probs[1]*100:.1f}%")
    print(f"  SOUTH {probs[2]*100:.1f}%")
    print(f"  EAST  {probs[3]*100:.1f}%")
    print(f"  WEST  {probs[4]*100:.1f}%")
    print(f"  WATER {probs[5]*100:.1f}%")

print("\n--- DIAGNOSTIC B: WATER DISTANCE RESPONSE ---")
for dist_cells in [1, 5, 10, 20, 30]:
    world.fire_manager.fire_map.fill(FireState.UNBURNED)
    world.fire_manager.fire_map[24+dist_cells, 24] = FireState.BURNING
    
    obs = obs_builder.get_observation(drone, world)
    spatial = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0).to(device)
    drone_vec = torch.tensor(obs['drone'], dtype=torch.float32).unsqueeze(0).to(device)
    wind_vec = torch.tensor(obs['wind'], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        action_logits, _ = policy({'spatial': spatial, 'drone': drone_vec, 'wind': wind_vec})
        action_logits[0, 6] = -1e9
        probs = torch.softmax(action_logits, dim=-1)[0].numpy()
        
    print(f"distance {dist_cells} -> P(WATER) = {probs[5]*100:.2f}%")
