import torch
import numpy as np
import pygame
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_52_network import ActorCritic6Channels
from stage7_52_obs_builder import GlobalFireObservationBuilder
import time

def run_visual():
    print("Starting visual evaluation for Stage 7.52 (200 updates)...")
    
    # Init Env
    env = WildfireEnv("configs/environment.yaml")
    env.obs_builder = GlobalFireObservationBuilder(window_size=11)
    obs, _ = env.reset(seed=42)
    
    # Setup Pygame Renderer (if available)
    try:
        from wildfire.simulation.renderer import Renderer
        renderer = Renderer(env.world)
    except:
        renderer = None
        
    device = torch.device('cpu')
    policy = ActorCritic6Channels(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_52_global_fire_channel_200updates.pth", map_location=device, weights_only=True))
    policy.eval()

    done = False
    step_count = 0
    total_suppressed = 0

    while not done and step_count < 500:
        if renderer:
            renderer.render()
            time.sleep(0.05)
            
        with torch.no_grad():
            spatial = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0).to(device)
            drone_vec = torch.tensor(obs['drone'], dtype=torch.float32).unsqueeze(0).to(device)
            wind_vec = torch.tensor(obs['wind'], dtype=torch.float32).unsqueeze(0).to(device)
            
            action_logits, _ = policy({'spatial': spatial, 'drone': drone_vec, 'wind': wind_vec})
            action_logits[0, 6] = -1e9  # Mask retardant
            probs = torch.softmax(action_logits, dim=-1)[0].numpy()
            
            # Sample action stochastically (like real evaluation)
            action = np.random.choice(7, p=probs)
            
        drone = env.world.drones[0]
        fm = env.world.fire_manager.fire_map
        active_mask = (fm == 1) | (fm == 2) | (fm == 3)
        active_coords = np.argwhere(active_mask)
        if len(active_coords) > 0:
            dists = np.abs(active_coords[:,0] - drone.x) + np.abs(active_coords[:,1] - drone.y)
            fire_dist = np.min(dists)
        else:
            fire_dist = 0
            
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step_count += 1
        
        suppressed = info.get('newly_suppressed_cells', 0)
        total_suppressed += suppressed
        
        actions = ["STAY", "N", "S", "E", "W", "WATER", "RETARDANT"]
        print(f"Step {step_count:3d} | Act: {actions[action]:5s} | Pos: ({drone.x:2d},{drone.y:2d}) | "
              f"Bat: {drone.battery:3.0f} | Pay: {drone.payload:2d} | DistToFire: {fire_dist:2d} | "
              f"Supp: {suppressed}")

    print(f"Visual Trace Complete. Total Steps: {step_count}. Total Suppressed: {total_suppressed}")

if __name__ == "__main__":
    run_visual()
