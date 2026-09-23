import numpy as np
import yaml
from wildfire.environment.wildfire_env import WildfireEnv

def run_sweep():
    env = WildfireEnv("configs/environment.yaml")
    
    spreads = [0.08, 0.06, 0.04, 0.028]
    
    results = {}
    
    for spread in spreads:
        returns_count = 0
        crash_count = 0
        suppressions_list = []
        extinctions = 0
        natural_extinctions = 0
        suppression_extinctions = 0
        burned_list = []
        active_at_term = []
        fire_growth_rate = []
        
        payload_depletions = 0
        total_drops_successful = 0
        base_returns_list = []
        ep_lengths = []
        
        for i in range(20):
            env.reset(seed=4000+i)
            env.world.fire_manager.base_spread_rate = spread
            drone = env.world.drones[0]
            
            done = False
            step = 0
            ep_suppressed = 0
            base_returns_this_ep = 0
            
            depleted_payload_this_ep = False
            was_at_base = True
            
            initial_burned = (env.world.fire_manager.fire_map > 1).sum()
            
            while not done and step < 500:
                req_bat = env.world.required_battery(drone)
                
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
                done = terminated or truncated
                
                supp = info.get('newly_suppressed_cells', 0)
                ep_suppressed += supp
                total_drops_successful += supp
                
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
                
            fm = env.world.fire_manager.fire_map
            active = np.sum((fm == 1) | (fm == 2) | (fm == 3))
            active_at_term.append(active)
            
            if active == 0:
                extinctions += 1
                if step >= 200 and ep_suppressed < 50:
                    natural_extinctions += 1
                else:
                    # If it extinguished fast with fewer steps or we suppressed all of it
                    total_cells = (fm > 1).sum()
                    if ep_suppressed >= total_cells * 0.5:
                        suppression_extinctions += 1
                    else:
                        natural_extinctions += 1
                        
            final_burned = (fm > 1).sum()
            burned_list.append(final_burned)
            suppressions_list.append(ep_suppressed)
            base_returns_list.append(base_returns_this_ep)
            ep_lengths.append(step)
            
            # Growth rate (cells/tick)
            growth = (final_burned - initial_burned) / max(1, step)
            fire_growth_rate.append(growth)
            
        results[spread] = {
            'growth': np.mean(fire_growth_rate),
            'total_supp': total_drops_successful,
            'mean_supp': np.mean(suppressions_list),
            'ext': extinctions,
            'natural_ext': natural_extinctions,
            'supp_ext': suppression_extinctions,
            'burned': np.mean(burned_list),
            'active_term': np.mean(active_at_term),
            'payload_dep': payload_depletions,
            'crashes': crash_count,
            'ep_length': np.mean(ep_lengths),
            'base_ret': np.mean(base_returns_list)
        }
        
    print("### Fire Spread Calibration Sweep\n")
    for s in spreads:
        r = results[s]
        print(f"#### Spread Rate: {s}")
        print(f"- Mean fire growth rate: {r['growth']:.2f} cells/tick")
        print(f"- Total successful suppressions: {r['total_supp']}")
        print(f"- Mean successful suppressions: {r['mean_supp']:.1f}")
        print(f"- Extinction percentage: {(r['ext'] / 20) * 100:.0f}% (Natural: {r['natural_ext']}, Suppressed: {r['supp_ext']})")
        print(f"- Mean burned-area metric: {r['burned']:.1f}")
        print(f"- Active-fire cells at termination: {r['active_term']:.1f}")
        print(f"- Payload depletion percentage: {(r['payload_dep'] / 20) * 100:.0f}%")
        print(f"- Crash percentage: {(r['crashes'] / 20) * 100:.0f}%")
        print(f"- Mean episode length: {r['ep_length']:.1f}")
        print(f"- Mean base returns: {r['base_ret']:.1f}\n")

if __name__ == "__main__":
    run_sweep()
