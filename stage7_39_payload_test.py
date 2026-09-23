import numpy as np
import yaml
from wildfire.environment.wildfire_env import WildfireEnv

def run_payload_test():
    env = WildfireEnv("configs/environment.yaml")
    
    returns_count = 0
    crash_count = 0
    suppressions_list = []
    extinctions = 0
    burned_list = []
    
    payload_depletions = 0
    total_drops_attempted = 0
    total_drops_successful = 0
    base_returns_list = []
    ep_lengths = []
    
    # Run same 20 seeds
    for i in range(20):
        env.reset(seed=4000+i)
        drone = env.world.drones[0]
        # battery is already 150 from config
        # payload is already 20 from config
        
        done = False
        step = 0
        ep_suppressed = 0
        base_returns_this_ep = 0
        
        depleted_payload_this_ep = False
        was_at_base = True
        
        while not done and step < 500:
            req_bat = env.world.required_battery(drone)
            
            # Check for return to base condition
            if drone.battery <= req_bat + 2 or drone.payload == 0:
                target_x, target_y = 2, 2
                if drone.payload == 0:
                    depleted_payload_this_ep = True
            else:
                burning = (env.world.fire_manager.fire_map == 2)
                if not np.any(burning):
                    target_x, target_y = drone.x, drone.y
                else:
                    x_idx, y_idx = np.where(burning)
                    dists = np.abs(x_idx - drone.x) + np.abs(y_idx - drone.y)
                    best = np.argmin(dists)
                    target_x, target_y = x_idx[best], y_idx[best]
            
            if drone.x == target_x and drone.y == target_y:
                if (target_x, target_y) != (2,2) and env.world.fire_manager.fire_map[target_x, target_y] == 2:
                    action = 5 # water
                    total_drops_attempted += 1
                else:
                    action = 0 # hover
            else:
                if np.abs(target_x - drone.x) > np.abs(target_y - drone.y):
                    if target_x > drone.x: action = 3
                    else: action = 4
                else:
                    if target_y > drone.y: action = 2
                    else: action = 1
                    
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            supp = info.get('newly_suppressed_cells', 0)
            ep_suppressed += supp
            total_drops_successful += supp
            
            # Count base returns
            is_at_base = (drone.x, drone.y) == (2,2)
            if is_at_base and not was_at_base and drone.battery == drone.max_battery:
                base_returns_this_ep += 1
                returns_count += 1
            was_at_base = is_at_base
            
            step += 1
            if not drone.active:
                break
                
        if not drone.active:
            crash_count += 1
        if depleted_payload_this_ep:
            payload_depletions += 1
            
        burning = (env.world.fire_manager.fire_map == 2)
        igniting = (env.world.fire_manager.fire_map == 1)
        smoldering = (env.world.fire_manager.fire_map == 3)
        
        if not (np.any(burning) or np.any(igniting) or np.any(smoldering)):
            extinctions += 1
            
        burned_list.append((env.world.fire_manager.fire_map > 1).sum())
        suppressions_list.append(ep_suppressed)
        base_returns_list.append(base_returns_this_ep)
        ep_lengths.append(step)
        
    print("### Payload-20 Calibration Results\n")
    print(f"Payload capacity: {env.world.drones[0].max_payload}")
    print(f"Battery capacity: {env.world.drones[0].max_battery}")
    print(f"Return percentage: {(returns_count / (20 * max(1, np.mean(base_returns_list)))) * 100:.0f}% (Counted multiple per episode)")
    print(f"Crash percentage: {(crash_count / 20) * 100:.0f}%")
    print(f"Total successful suppressions: {total_drops_successful}")
    print(f"Mean successful suppressions: {np.mean(suppressions_list):.1f}")
    print(f"Extinction percentage: {(extinctions / 20) * 100:.0f}%")
    print(f"Burned-area metric (mean): {np.mean(burned_list):.1f}")
    print(f"Payload depletion (episodes): {(payload_depletions / 20) * 100:.0f}%")
    print(f"Mean base returns per episode: {np.mean(base_returns_list):.1f}")
    print(f"Average episode length: {np.mean(ep_lengths):.1f}")
    
if __name__ == "__main__":
    run_payload_test()
