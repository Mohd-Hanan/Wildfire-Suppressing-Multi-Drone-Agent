import torch
import pygame
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer
from stage7_35_final_truth import SeparateActorCriticSymmetric64
import time
import yaml

def run_visual():
    print("Starting visual evaluation for Stage 7.44 Candidate Physics...")
    
    env = WildfireEnv("configs/environment.yaml")
    renderer = Renderer(width=env.map_width, height=env.map_height)
    
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    
    # Load the checkpoint
    policy.load_state_dict(torch.load("stage7_44_candidate_physics_20updates.pth", map_location=device, weights_only=True))
    policy.eval() # Use eval mode (no gradients), but Categorical sampling is still active
    
    obs, _ = env.reset(seed=402) # random seed
    done = False
    
    step = 0
    drone = env.world.drones[0]
    
    while not done and step < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        with torch.no_grad():
            action_t, _, _, _ = policy.get_action_and_value(obs)
            
        a = action_t.item()
        
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        
        # Render the state
        renderer.render_world(env.world)
        time.sleep(0.1) # 10 FPS
        
        step += 1
        
    print(f"Visual Trace Complete. Steps: {step}, Active: {drone.active}")
    pygame.quit()

if __name__ == "__main__":
    run_visual()
