from wildfire.environment.wildfire_env import WildfireEnv
env = WildfireEnv("configs/environment.yaml")
env.reset()
env.world.drones[0].x = 10
env.world.drones[0].y = 10
env.world.fire_manager.fire_map[10, 10] = 2 # BURNING
obs, r, t, tr, info = env.step(5) # WATER
print(info['newly_suppressed_cells'])
