import time
import argparse
import pygame
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell_size", type=int, default=16)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    print(f"Initializing Gymnasium Environment with seed {args.seed}...")
    env = WildfireEnv()
    obs, info = env.reset(seed=args.seed)
    
    # Extract world properties for renderer
    width = env.map_width
    height = env.map_height
    
    renderer = Renderer(width, height, args.cell_size)
    
    print("Rendering... Press Ctrl+C to close.")
    clock = pygame.time.Clock()
    running = True
    
    try:
        while running:
            # Handle pygame events so the window doesn't freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # 1. Agent Step (Random Action)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # If episode ends, pause shortly then reset
            if terminated or truncated:
                print(f"Episode ended (terminated={terminated}, truncated={truncated}). Reason: {info.get('termination_reason')}")
                renderer.render_world(env.world)
                pygame.display.flip()
                time.sleep(1)
                env.reset()
                
            # 2. Render Step
            renderer.render_world(env.world)
            clock.tick(args.fps)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        renderer.close()

if __name__ == "__main__":
    main()
