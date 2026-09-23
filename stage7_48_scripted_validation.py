import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv

def navigate_to(drone, target_x, target_y):
    if drone.x < target_x: return 3 # East
    if drone.x > target_x: return 4 # West
    if drone.y < target_y: return 2 # South
    if drone.y > target_y: return 1 # North
    return 0

def run_scripted():
    env = WildfireEnv("configs/environment.yaml")
    
    stats = {
        'total': 0.0,
        'suppression': 0.0,
        'distance': 0.0,
        'damage': 0.0,
        'battery': 0.0,
        'step': 0.0,
        'extinction': 0.0
    }
    
    supp_ext = 0
    nat_ext = 0
    crash_cnt = 0
    succ_supp_cnt = 0
    water_attempts = 0
    
    water_dists = []
    
    burned_areas = []
    episode_lengths = []
    
    for ep in range(20):
        obs, _ = env.reset(seed=100+ep)
        done = False
        steps = 0
        ep_supp_count = 0
        
        while not done and steps < 500:
            fm = env.world.fire_manager.fire_map
            active_mask = (fm == 1) | (fm == 2) | (fm == 3)
            active_coords = np.argwhere(active_mask)
            drone = env.world.drones[0]
            
            # Record dist before action
            if len(active_coords) > 0:
                dists = np.abs(active_coords[:,0]-drone.x) + np.abs(active_coords[:,1]-drone.y)
                current_fire_dist = np.min(dists)
            else:
                current_fire_dist = None
            
            # Very simple greedy controller
            a = 0
            if not drone.active:
                pass
            elif env.world.battery_margin(drone) <= 1 and not env.world.is_at_base(drone):
                bx = env.world.base_x
                by = env.world.base_y
                a = navigate_to(drone, bx, by)
            elif drone.payload == 0 and not env.world.is_at_base(drone):
                bx = env.world.base_x
                by = env.world.base_y
                a = navigate_to(drone, bx, by)
            elif len(active_coords) > 0:
                nearest = active_coords[np.argmin(dists)]
                if drone.x == nearest[0] and drone.y == nearest[1]:
                    a = 5 # WATER
                else:
                    a = navigate_to(drone, nearest[0], nearest[1])
            else:
                a = 0
                
            if a == 5:
                water_attempts += 1
                if current_fire_dist is not None:
                    water_dists.append(current_fire_dist)
                
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            steps += 1
            
            stats['total'] += r
            stats['suppression'] += info.get('suppression_reward', 0.0)
            stats['distance'] += info.get('distance_reward', 0.0)
            stats['damage'] += info.get('damage_penalty', 0.0)
            stats['battery'] += info.get('battery_safety_penalty', 0.0)
            stats['step'] += info.get('step_penalty', 0.0)
            stats['extinction'] += info.get('extinction_reward', 0.0)
            
            supp = info.get('newly_suppressed_cells', 0)
            ep_supp_count += supp
            succ_supp_cnt += supp
            
        drone = env.world.drones[0]
        if not drone.active:
            crash_cnt += 1
            
        active_fire = np.sum((fm == 1) | (fm == 2) | (fm == 3))
        if active_fire == 0:
            if ep_supp_count > 0:
                supp_ext += 1
            else:
                nat_ext += 1
                
        burned = np.sum(fm != 0)
        burned_areas.append(burned)
        episode_lengths.append(steps)
        
    print("==================================================")
    print("SCRIPTED VALIDATION RESULTS (20 EPISODES)")
    print("==================================================")
    print(f"Mean total reward:       {stats['total']/20:.2f}")
    print(f"Mean suppression reward: {stats['suppression']/20:.2f}")
    print(f"Mean distance reward:    {stats['distance']/20:.2f}")
    print(f"Mean damage penalty:     {stats['damage']/20:.2f}")
    print(f"Mean battery penalty:    {stats['battery']/20:.2f}")
    print(f"Mean step penalty:       {stats['step']/20:.2f}")
    print(f"Mean extinction reward:  {stats['extinction']/20:.2f}")
    print(f"Successful suppressions: {succ_supp_cnt}")
    print(f"WATER attempts:          {water_attempts}")
    print(f"WATER success rate:      {(succ_supp_cnt/water_attempts*100) if water_attempts else 0:.1f}%")
    print(f"Mean distance when WATER:{np.mean(water_dists) if water_dists else 0:.2f}")
    
    water_dists = np.array(water_dists)
    if len(water_dists) > 0:
        print(f"% WATER within 1 cell:   {np.mean(water_dists <= 1)*100:.1f}%")
        print(f"% WATER within 5 cells:  {np.mean(water_dists <= 5)*100:.1f}%")
        print(f"% WATER beyond 5 cells:  {np.mean(water_dists > 5)*100:.1f}%")
        print(f"% WATER beyond 10 cells: {np.mean(water_dists > 10)*100:.1f}%")
        
    print(f"Strict supp extinctions: {supp_ext}")
    print(f"Natural extinctions:     {nat_ext}")
    print(f"Crashes:                 {crash_cnt}")
    print(f"Mean episode length:     {np.mean(episode_lengths):.1f}")
    print(f"Mean burned area:        {np.mean(burned_areas):.2f}")

if __name__ == "__main__":
    run_scripted()
