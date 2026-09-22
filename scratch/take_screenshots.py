import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer

def main():
    os.makedirs('scratch/screenshots', exist_ok=True)
    env = WildfireEnv()
    obs, info = env.reset(seed=46) 
    
    # Initialize pygame explicitly for dummy driver
    pygame.init()
    pygame.display.set_mode((1, 1))
    
    renderer = Renderer(env.map_width, env.map_height, 16)
    
    frames_to_capture = [0, 5, 20, 50, 100, 200]
    
    try:
        for step in range(250):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            renderer.render_world(env.world)
            
            if step in frames_to_capture:
                pygame.image.save(renderer.screen, f"scratch/screenshots/frame_{step}.png")
                
            if terminated or truncated:
                print(f"Ended at {step}")
                pygame.image.save(renderer.screen, f"scratch/screenshots/frame_end.png")
                break
    finally:
        renderer.close()

if __name__ == "__main__":
    main()
