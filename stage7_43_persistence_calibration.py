import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv

def run_fire_only_diagnostic():
    configs = [(0.08, 20), (0.04, 40)]
    
    print("### FIRE-ONLY DIAGNOSTIC")
    
    for spread, dur in configs:
        max_active_list = []
        total_burned_list = []
        extinction_times = []
        new_cells_per_tick = []
        sustained = 0
        final_active_list = []
        
        for i in range(20):
            env = WildfireEnv("configs/environment.yaml")
            env.reset(seed=4000+i)
            env.world.fire_manager.base_spread_rate = spread
            env.world.fire_manager.base_burning_duration = dur
            
            for drone in env.world.drones:
                pass
                
            fm = env.world.fire_manager
            
            max_active = 0
            total_new_ignitions = 0
            step = 0
            
            while step < 500:
                _, _, terminated, truncated, _ = env.step(0)
                
                active = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
                if active > max_active: max_active = active
                    
                total_new_ignitions += np.sum(fm.fire_map == 1)
                
                step += 1
                if active == 0 or terminated or truncated:
                    break
                    
            final_active = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
            final_active_list.append(final_active)
            max_active_list.append(max_active)
            total_burned_list.append(np.sum(fm.fire_map > 0))
            new_cells_per_tick.append(total_new_ignitions / max(1, step))
            
            if final_active == 0:
                extinction_times.append(step)
            elif step >= 500 and np.sum(fm.fire_map == 2) > 0:
                sustained += 1
                
        ext_rate = (len(extinction_times) / 20) * 100
        mean_ext_time = np.mean(extinction_times) if extinction_times else float('nan')
        
        print(f"\nConfiguration: base_spread_rate={spread}, burning_duration={dur}")
        print(f"- Initial active cells: 1")
        print(f"- Maximum active cells reached: {np.mean(max_active_list):.1f}")
        print(f"- Final active cells at termination: {np.mean(final_active_list):.1f}")
        print(f"- Total burned cells: {np.mean(total_burned_list):.1f}")
        print(f"- Natural extinction rate: {ext_rate:.0f}%")
        print(f"- Mean extinction time: {mean_ext_time:.1f}")
        print(f"- Mean new cells ignited per tick: {np.mean(new_cells_per_tick):.2f}")
        print(f"- Episodes surviving to 500 steps: {sustained}/20")
        print(f"- Maintains propagation rather than dying naturally: {'Yes' if sustained > 10 else 'No'}")


def run_drone_calibration():
    configs = [(0.08, 20), (0.04, 40)]
    
    print("\n### DRONE CALIBRATION CONTROLLER")
    
    for spread, dur in configs:
        crash_count = 0
        suppressions_list = []
        total_supp = 0
        natural_extinctions = 0
        suppression_extinctions = 0
        burned_list = []
        active_at_term = []
        payload_depletions = 0
        base_returns_list = []
        ep_lengths = []
        
        for i in range(20):
            env = WildfireEnv("configs/environment.yaml")
            env.reset(seed=4000+i)
            env.world.fire_manager.base_spread_rate = spread
            env.world.fire_manager.base_burning_duration = dur
            
            drone = env.world.drones[0]
            fm = env.world.fire_manager
            
            step = 0
            ep_suppressed = 0
            base_returns_this_ep = 0
            depleted_payload_this_ep = False
            was_at_base = True
            
            while step < 500:
                req_bat = env.world.required_battery(drone)
                
                if drone.battery <= req_bat + 2 or drone.payload == 0:
                    target_x, target_y = 2, 2
                    if drone.payload == 0: depleted_payload_this_ep = True
                else:
                    burning = (fm.fire_map == 2)
                    if not np.any(burning):
                        target_x, target_y = drone.x, drone.y
                    else:
                        x_idx, y_idx = np.where(burning)
                        dists = np.abs(x_idx - drone.x) + np.abs(y_idx - drone.y)
                        best = np.argmin(dists)
                        target_x, target_y = x_idx[best], y_idx[best]
                
                if drone.x == target_x and drone.y == target_y:
                    if (target_x, target_y) != (2,2) and fm.fire_map[target_x, target_y] == 2:
                        action = 5
                    else:
                        action = 0
                else:
                    if np.abs(target_x - drone.x) > np.abs(target_y - drone.y):
                        if target_x > drone.x: action = 3
                        else: action = 4
                    else:
                        if target_y > drone.y: action = 2
                        else: action = 1
                        
                _, _, terminated, truncated, info = env.step(action)
                
                supp = info.get('newly_suppressed_cells', 0)
                ep_suppressed += supp
                total_supp += supp
                
                is_at_base = (drone.x, drone.y) == (2,2)
                if is_at_base and not was_at_base and drone.battery == drone.max_battery:
                    base_returns_this_ep += 1
                was_at_base = is_at_base
                
                step += 1
                
                active = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
                if active == 0 or not drone.active or terminated or truncated:
                    break
                    
            if not drone.active: crash_count += 1
            if depleted_payload_this_ep: payload_depletions += 1
            
            active = np.sum((fm.fire_map == 1) | (fm.fire_map == 2) | (fm.fire_map == 3))
            active_at_term.append(active)
            
            if active == 0:
                total_cells = (fm.fire_map > 1).sum()
                # Determine if it's natural or suppressed
                # If we suppressed a substantial portion of the cells (e.g., >30%) or if it died very fast
                if ep_suppressed >= total_cells * 0.3 or (ep_suppressed > 0 and step < 200):
                    suppression_extinctions += 1
                else:
                    natural_extinctions += 1
                    
            burned_list.append((fm.fire_map > 1).sum())
            suppressions_list.append(ep_suppressed)
            base_returns_list.append(base_returns_this_ep)
            ep_lengths.append(step)
            
        print(f"\nConfiguration: base_spread_rate={spread}, burning_duration={dur}")
        print(f"- Total successful suppressions: {total_supp}")
        print(f"- Mean successful suppressions: {np.mean(suppressions_list):.1f}")
        print(f"- Mean burned-area metric: {np.mean(burned_list):.1f}")
        print(f"- Extinction percentage: {((natural_extinctions + suppression_extinctions)/20)*100:.0f}%")
        print(f"- Natural extinction count: {natural_extinctions}")
        print(f"- Suppression-driven extinction count: {suppression_extinctions}")
        print(f"- Payload depletion: {(payload_depletions/20)*100:.0f}%")
        print(f"- Crash percentage: {(crash_count/20)*100:.0f}%")
        print(f"- Mean episode length: {np.mean(ep_lengths):.1f}")
        print(f"- Mean base returns: {np.mean(base_returns_list):.1f}")
        print(f"- Active fire at termination (mean): {np.mean(active_at_term):.1f}")


if __name__ == "__main__":
    run_fire_only_diagnostic()
    # Reset active state issue
    # Wait, the previous run didn't set pass properly because we commented it out and used env.step(0). 
    # Yes, env.step(0) is perfectly fine for drone inactive.
    run_drone_calibration()
