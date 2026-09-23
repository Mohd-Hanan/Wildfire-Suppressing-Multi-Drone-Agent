import re

with open('src/wildfire/environment/wildfire_env.py', 'r') as f:
    content = f.read()

new_step = """    def step(self, action: int):
        \"\"\"Executes one step in the environment.\"\"\"
        drone = self.world.drones[self.controlled_drone_idx]
        
        # PRE-ACTION CAPTURE
        pre_drone_x, pre_drone_y = drone.x, drone.y
        
        fm = self.world.fire_manager.fire_map
        from wildfire.simulation.fire import FireState
        import numpy as np
        active_mask = (fm == FireState.IGNITING) | (fm == FireState.BURNING) | (fm == FireState.SMOLDERING)
        active_coords = np.argwhere(active_mask)
        pre_active_fire_count = len(active_coords)
        
        if pre_active_fire_count > 0:
            dists = np.abs(active_coords[:, 0] - pre_drone_x) + np.abs(active_coords[:, 1] - pre_drone_y)
            nearest_idx = np.argmin(dists)
            pre_target = tuple(active_coords[nearest_idx])
        else:
            pre_target = None
        
        # A. Execute action
        hit_boundary, suppressed_cells = self.action_executor.execute(drone, self.world, action)
        
        # B. Advance the wildfire/world simulation by one timestep
        self.world.fire_manager.step(self.world.terrain, self.world.wind)
        self.world.process_base_refills()
        
        # C. Increment step_count
        self.step_count += 1
        
        # D. Calculate reward
        reward, reward_info = self.reward_calculator.calculate(
            self.world, 
            self.world.drones, 
            hit_boundary, 
            suppressed_cells, 
            action,
            pre_drone_pos=(pre_drone_x, pre_drone_y),
            pre_target=pre_target,
            pre_active_fire_count=pre_active_fire_count
        )
        
        # E. Check termination
        terminated, truncated, term_info = self.termination_checker.check(self.world, self.step_count)
        
        # F. Build next observation
        obs = self.obs_builder.get_observation(drone, self.world)
        
        # G. Build info dict
        info = {}
        info.update(reward_info)
        info.update(term_info)
        
        return obs, float(reward), terminated, truncated, info"""

content = re.sub(r'    def step\(self, action: int\):.*?(?=\n    def render|\Z)', new_step + '\n', content, flags=re.DOTALL)

with open('src/wildfire/environment/wildfire_env.py', 'w') as f:
    f.write(content)
