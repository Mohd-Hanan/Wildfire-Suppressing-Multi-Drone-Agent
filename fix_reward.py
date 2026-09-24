import re

with open("stage_testing.py", "r") as f:
    code = f.read()

# Replace print "TESTING EPISODE X/Y"
code = code.replace(
    'print(f"TESTING EPISODE {ep+1}/{episodes}")',
    'print(f"EPISODE {ep+1}")'
)

# Insert RewardCalculator setup
replacement_setup = """
        from wildfire.environment.reward import RewardCalculator
        reward_calculator = RewardCalculator({})
        reward_calculator.reset()
        episode_reward = 0.0
        
        initial_fire_cells = np.sum(world.fire_manager.fire_map > 0)
"""
code = code.replace(
    '        initial_fire_cells = np.sum(world.fire_manager.fire_map > 0)',
    replacement_setup[1:]
)

# Track reward in step loop
# The old code calculates suppressed via hit_boundary, suppressed = executor.execute...
# And then does world.process_base_refills().
# Let's intercept right after executor.execute
replacement_reward = """
                # Use ActionExecutor directly
                pre_active = np.sum((world.fire_manager.fire_map == 1) | (world.fire_manager.fire_map == 2) | (world.fire_manager.fire_map == 3))
                pre_pos = (drone.x, drone.y)
                pre_targ = drone.current_target if hasattr(drone, 'current_target') else (0,0)
                
                hit_boundary, suppressed = executor.execute(drone, world, act)
                total_suppressed += suppressed
                
                rew, _ = reward_calculator.calculate(
                    world, world.drones, hit_boundary, suppressed, act, 
                    pre_pos, pre_targ, pre_active
                )
                episode_reward += rew
"""
code = re.sub(
    r'                # Use ActionExecutor directly.*?                total_suppressed \+= suppressed',
    replacement_reward[1:],
    code,
    flags=re.DOTALL
)

# Print total reward at the end
code = code.replace(
    '        print(f"Fire Extinction Status: {\'SUCCESS\' if extinguished else \'FAILED\'}")',
    '        print(f"Fire Extinction Status: {\'SUCCESS\' if extinguished else \'FAILED\'}")\n        print(f"Total Episode Reward: {episode_reward:.2f}")'
)

with open("stage_testing.py", "w") as f:
    f.write(code)
