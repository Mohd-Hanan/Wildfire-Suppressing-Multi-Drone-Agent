import yaml
from wildfire.simulation.world import World
with open('configs/environment.yaml', 'r') as f:
    config = yaml.safe_load(f)
world = World(48, 48, seed=42)
print(world.base_x, world.base_y)
