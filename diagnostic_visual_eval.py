from wildfire.environment.wildfire_env import WildfireEnv

eval_env = WildfireEnv()
obs, _ = eval_env.reset(seed=1000)

for step in range(5):
    obs, reward, terminated, truncated, _ = eval_env.step(4) # West
    drone = eval_env.world.drones[eval_env.controlled_drone_idx]
    print(f"Step {step+1} | x={drone.x} | battery={drone.battery} | base_x={eval_env.world.base_x} | is_at_base={eval_env.world.is_at_base(drone)}")
