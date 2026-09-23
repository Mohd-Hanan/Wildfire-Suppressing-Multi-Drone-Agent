import numpy as np
import yaml
from wildfire.environment.wildfire_env import WildfireEnv

def run_sweep():
    env = WildfireEnv("configs/environment.yaml")
    
    batteries_to_test = [150, 200, 250, 300, 400]
    
    results = {}
    
    for bat in batteries_to_test:
        returns_count = 0
        crash_count = 0
        suppressions_list = []
        extinctions = 0
        burned_list = []
        
        payload_depletions = 0
        total_drops_attempted = 0
        total_drops_successful = 0
        
        for i in range(20):
            env.reset(seed=4000+i)
            drone = env.world.drones[0]
            drone.max_battery = bat
            drone.battery = bat
            
            done = False
            step = 0
            ep_suppressed = 0
            
            returned_to_base_this_ep = False
            depleted_payload_this_ep = False
            
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
                
                # Choose action
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
                
                if action == 5 and supp > 0:
                    # just to verify
                    pass
                
                if (drone.x, drone.y) == (2,2) and step > 0 and drone.battery == drone.max_battery:
                    returned_to_base_this_ep = True
                
                step += 1
                if not drone.active:
                    break
                    
            if not drone.active:
                crash_count += 1
            if returned_to_base_this_ep:
                returns_count += 1
            if depleted_payload_this_ep:
                payload_depletions += 1
                
            burning = (env.world.fire_manager.fire_map == 2)
            igniting = (env.world.fire_manager.fire_map == 1)
            smoldering = (env.world.fire_manager.fire_map == 3)
            
            if not (np.any(burning) or np.any(igniting) or np.any(smoldering)):
                extinctions += 1
                
            burned_list.append((env.world.fire_manager.fire_map > 1).sum()) # count burning, smoldering, burned
            suppressions_list.append(ep_suppressed)
            
        results[bat] = {
            'returns': returns_count,
            'crashes': crash_count,
            'suppressions': np.mean(suppressions_list),
            'extinctions': extinctions,
            'burned': np.mean(burned_list),
            'payload_depletions': payload_depletions,
            'total_drops': total_drops_attempted,
            'successful_drops': total_drops_successful
        }
        
    print("### Battery Sweep\n")
    print("| Battery | Return % | Crash % | Suppressions | Extinction % | Burned |")
    print("|---:|---:|---:|---:|---:|---:|")
    
    for bat in batteries_to_test:
        r = results[bat]
        ret_pct = (r['returns'] / 20) * 100
        crash_pct = (r['crashes'] / 20) * 100
        ext_pct = (r['extinctions'] / 20) * 100
        print(f"| {bat} | {ret_pct:.0f}% | {crash_pct:.0f}% | {r['suppressions']:.1f} | {ext_pct:.0f}% | {r['burned']:.1f} |")
        
    print("\n### Payload Analysis\n")
    print("- Current payload: 5")
    
    # We aggregate payload stats from the highest battery test (where it can actually drop payload)
    best_bat = 400
    r_best = results[best_bat]
    # Mean drops required: this is a bit abstract, let's say mean successful drops.
    # We will just print the data
    
    print(f"- Mean drops attempted (at 400 bat): {r_best['total_drops']/20:.1f}")
    print(f"- Mean successful suppressions (at 400 bat): {r_best['suppressions']:.1f}")
    print(f"- Payload depletion rate (at 400 bat): {(r_best['payload_depletions']/20)*100:.0f}%")
    print("- Proposed range, if necessary: We will wait to see if suppressions fix the fire.")
    
if __name__ == "__main__":
    run_sweep()
