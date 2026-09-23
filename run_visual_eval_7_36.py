import torch
import pygame
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer
from stage7_35_final_truth import SeparateActorCriticSymmetric64

def run_visual():
    print("Starting visual evaluation for Stage 7.36 Battery Safety...")
    
    env = WildfireEnv("configs/environment.yaml")
    renderer = Renderer(width=env.map_width, height=env.map_height)
    
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_36_battery_20updates.pth", map_location=device, weights_only=True))
    policy.eval()
    
    obs, _ = env.reset(seed=205)
    done = False
    step = 0
    
    clock = pygame.time.Clock()
    
    while not done and step < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        with torch.no_grad():
            action_t, _, _, _ = policy.get_action_and_value(obs)
            
        action = action_t.item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the world
        renderer.render_world(env.world)
        
        drone = env.world.drones[0]
        if not drone.active:
            print(f"*** Drone CRASHED at step {step} ***")
            break
            
        step += 1
        clock.tick(10) # 10 FPS
        
    renderer.close()
    pygame.quit()
    print("Visual evaluation complete.")

if __name__ == "__main__":
    run_visual()
