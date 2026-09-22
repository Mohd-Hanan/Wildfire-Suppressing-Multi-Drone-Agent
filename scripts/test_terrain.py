import time
import argparse
import pygame
from wildfire.simulation.world import World
from wildfire.rendering.renderer import Renderer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell_size", type=int, default=16)
    args = parser.parse_args()

    print(f"Generating world {args.width}x{args.height} with seed {args.seed}...")
    world = World(args.width, args.height, args.seed)
    
    print("World generated successfully.")
    
    renderer = Renderer(args.width, args.height, args.cell_size)
    
    print("Rendering... Press Ctrl+C to close.")
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            # Handle pygame events so the window doesn't freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # 1. Physics Step
            world.step()
            
            # 2. Render Step
            renderer.render_world(world)
            clock.tick(15) # 15 FPS simulation is a good viewing speed
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()

if __name__ == "__main__":
    main()
