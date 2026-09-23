import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv

def run_diagnostic():
    spreads = [0.08, 0.06, 0.04, 0.028]
    
    for spread in spreads:
        initial_active_list = []
        max_active_list = []
        final_active_list = []
        total_burned_list = []
        natural_extinction_times = []
        is_spreading_when_dead_list = []
        lifetime_ignited_cells = []
        new_cells_per_tick = []
        sustained_list = []
        
        for i in range(20):
            env = WildfireEnv("configs/environment.yaml")
            env.reset(seed=4000+i)
            env.world.fire_manager.base_spread_rate = spread
            
            # Disable drone intervention
                
            fm = env.world.fire_manager
            
            initial_active = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
            
            max_active = initial_active
            total_new_ignitions = 0
            
            done = False
            step = 0
            
            while not done and step < 500:
                # Step the environment with dummy action 0
                _, _, terminated, truncated, _ = env.step(0)
                done = terminated or truncated
                
                active = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
                if active > max_active:
                    max_active = active
                    
                new_ignitions = np.sum(fm.fire_map == 1)
                total_new_ignitions += new_ignitions
                
                step += 1
                
            final_active = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
            
            initial_active_list.append(initial_active)
            max_active_list.append(max_active)
            final_active_list.append(final_active)
            
            total_burned = np.sum(fm.fire_map > 0)
            total_burned_list.append(total_burned)
            
            if final_active == 0:
                natural_extinction_times.append(step)
                is_spreading = False
            else:
                is_spreading = np.any(fm.fire_map == 2) # Still has burning cells
            
            is_spreading_when_dead_list.append(is_spreading)
            
            new_cells_per_tick.append(total_new_ignitions / step)
            
            # A fire is sustained if it reaches max steps (500) and is still spreading
            if step >= 500 and is_spreading:
                sustained_list.append(True)
            else:
                sustained_list.append(False)
                
        # Since active states are hardcoded: IGNITING(1)+BURNING(20)+SMOLDERING(30)
        # Average lifetime of an ignited cell is exactly 51 ticks, unless it hits the end of episode.
        
        print(f"### Spread Rate: {spread}")
        print(f"- Initial active cells (mean): {np.mean(initial_active_list):.1f}")
        print(f"- Maximum active cells reached (mean): {np.mean(max_active_list):.1f}")
        print(f"- Final active cells (mean): {np.mean(final_active_list):.1f}")
        print(f"- Total burned cells (mean): {np.mean(total_burned_list):.1f}")
        print(f"- Time until natural extinction (mean): {np.mean(natural_extinction_times) if natural_extinction_times else '>500':.1f}")
        print(f"- Is still spreading when it dies: False (By definition, natural extinction means active=0)")
        print(f"- Sustained growth regime (>500 steps): {np.sum(sustained_list)}/20 episodes")
        print(f"- Average lifetime of an ignited cell: 51 ticks (Hardcoded: 1 Igniting + 20 Burning + 30 Smoldering)")
        print(f"- Average new cells ignited per tick: {np.mean(new_cells_per_tick):.2f}\n")

if __name__ == "__main__":
    run_diagnostic()
