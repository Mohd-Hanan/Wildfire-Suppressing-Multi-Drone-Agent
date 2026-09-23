import yaml
from wildfire.simulation.world import World
with open('configs/environment.yaml', 'r') as f:
    config = yaml.safe_load(f)
world = World(48, 48, seed=42)
drone = world.drones[0]
drone.x = 20
drone.y = 20
drone.battery = 30
print(f"dist: {world.distance_to_base(drone)}")
print(f"req: {world.required_battery(drone)}")
print(f"margin: {world.battery_margin(drone)}")
