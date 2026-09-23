import torch
import pygame
import numpy as np
import sys
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer
from stage7_35_final_truth import SeparateActorCriticSymmetric64
import time

def run_visual():
    print("Starting visual evaluation for Stage 7.48 Reward Causal (20 updates)...")
    
    env = WildfireEnv("configs/environment.yaml")
    renderer = Renderer(width=env.map_width, height=env.map_height)
    
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    
    # Load the checkpoint
    policy.load_state_dict(torch.load("stage7_48_reward_causal_20updates.pth", map_location=device, weights_only=True))
    policy.eval() # Use eval mode (no gradients), but Categorical sampling is still active
    
    obs, _ = env.reset(seed=142) # random seed
    done = False
    
    step = 0
    
    while not done and step < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    pygame.quit()
                    sys.exit(0)
        
        with torch.no_grad():
            spatial = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
            vec = torch.FloatTensor(obs['drone']).unsqueeze(0).to(device)
            wind = torch.FloatTensor(obs['wind']).unsqueeze(0).to(device)
            action, _, _, _ = policy.get_action_and_value({'spatial': spatial, 'drone': vec, 'wind': wind})
            a = action.item()
            
        obs, reward, term, trunc, info = env.step(a)
        done = term or trunc
        step += 1
        
        renderer.render_world(env.world)
        time.sleep(0.05) # ~20 FPS
        
    print(f"Episode finished after {step} steps.")
    print("Close the Pygame window to exit.")
    
    # Wait for user to close window
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    waiting = False
        time.sleep(0.1)
        
    pygame.quit()

if __name__ == "__main__":
    run_visual()
