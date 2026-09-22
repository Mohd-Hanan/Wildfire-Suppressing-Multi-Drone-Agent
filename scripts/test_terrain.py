import time
import argparse
from src.wildfire.simulation.terrain import Terrain
from src.wildfire.rendering.renderer import Renderer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell_size", type=int, default=16)
    args = parser.parse_args()

    print(f"Generating terrain {args.width}x{args.height} with seed {args.seed}...")
    terrain = Terrain(args.width, args.height, args.seed)
    
    print("Terrain generated successfully.")
    print(f"Elevation min/max: {terrain.elevation.min():.2f}/{terrain.elevation.max():.2f}")
    print(f"Fuel min/max: {terrain.fuel.min():.2f}/{terrain.fuel.max():.2f}")
    
    renderer = Renderer(args.width, args.height, args.cell_size)
    renderer.render_terrain(terrain)
    
    print("Rendering... Press Ctrl+C to close.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()

if __name__ == "__main__":
    main()
