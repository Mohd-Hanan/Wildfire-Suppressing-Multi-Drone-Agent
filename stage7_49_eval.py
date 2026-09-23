import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_35_final_truth import SeparateActorCriticSymmetric64

def evaluate():
    env = WildfireEnv("configs/environment.yaml")
    device = torch.device("cpu")
    
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_49_water_radius_20updates.pth", map_location=device))
    policy.eval()
    
    ep_rewards = []
    ep_lengths = []
    ep_burned_areas = []
    
    total_water = 0
    total_water_success = 0
    water_dists = []
    
    strict_ext = 0
    nat_ext = 0
    crash_count = 0
    base_visits = 0
    
    action_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    
    for ep in range(20):
        obs, _ = env.reset(seed=100+ep)
        done = False
        steps = 0
        ep_r = 0
        ep_supp_cnt = 0
        
        while not done:
            with torch.no_grad():
                spatial = torch.FloatTensor(obs['spatial']).unsqueeze(0).to(device)
                vec = torch.FloatTensor(obs['drone']).unsqueeze(0).to(device)
                wind = torch.FloatTensor(obs['wind']).unsqueeze(0).to(device)
                action, _, _, _ = policy.get_action_and_value({'spatial': spatial, 'drone': vec, 'wind': wind})
                a = action.item()
                
            action_counts[a] += 1
            
            fm = env.world.fire_manager.fire_map
            active_mask = (fm == 1) | (fm == 2) | (fm == 3)
            active_coords = np.argwhere(active_mask)
            drone = env.world.drones[0]
            
            if env.world.is_at_base(drone) and a == 0:
                base_visits += 1
                
            current_fire_dist = None
            if len(active_coords) > 0:
                dists = np.abs(active_coords[:,0]-drone.x) + np.abs(active_coords[:,1]-drone.y)
                current_fire_dist = np.min(dists)
                
            if a == 5:
                total_water += 1
                if current_fire_dist is not None:
                    water_dists.append(current_fire_dist)
                    
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ep_r += r
            steps += 1
            
            supp = info.get('newly_suppressed_cells', 0)
            ep_supp_cnt += supp
            total_water_success += supp
            
        drone = env.world.drones[0]
        if not drone.active:
            crash_count += 1
            
        active_fire = np.sum((fm == 1) | (fm == 2) | (fm == 3))
        if active_fire == 0:
            if ep_supp_cnt > 0:
                strict_ext += 1
            else:
                nat_ext += 1
                
        ep_rewards.append(ep_r)
        ep_lengths.append(steps)
        ep_burned_areas.append(np.sum(fm != 0))
        
    print("==================================================")
    print("20-EPISODE STOCHASTIC EVALUATION")
    print("==================================================")
    print(f"Mean reward:              {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"Mean burned area:         {np.mean(ep_burned_areas):.2f} ± {np.std(ep_burned_areas):.2f}")
    print(f"Total / Mean suppressions:{total_water_success} / {total_water_success/20:.2f}")
    print(f"WATER action count:       {total_water}")
    print(f"WATER success rate:       {(total_water_success/total_water*100) if total_water else 0:.1f}%")
    print(f"Mean dist when WATER:     {np.mean(water_dists) if water_dists else 0:.2f}")
    
    wd = np.array(water_dists)
    if len(wd) > 0:
        print(f"% WATER within 1 cell:    {np.mean(wd <= 1)*100:.1f}%")
        print(f"% WATER within 5 cells:   {np.mean(wd <= 5)*100:.1f}%")
        print(f"% WATER beyond 10 cells:  {np.mean(wd > 10)*100:.1f}%")
        
    print(f"Extinction %:             {(strict_ext+nat_ext)/20*100:.1f}%")
    print(f"Strict supp extinction %: {strict_ext/20*100:.1f}%")
    print(f"Natural extinction %:     {nat_ext/20*100:.1f}%")
    print(f"Crash rate:               {crash_count/20*100:.1f}% ({crash_count})")
    print(f"Mean episode length:      {np.mean(ep_lengths):.1f}")
    print(f"Base visits:              {base_visits}")
    
    total_actions = sum(action_counts.values())
    print("\nAction Distribution:")
    for a in range(7):
        print(f"Action {a}: {action_counts[a]} ({action_counts[a]/total_actions*100:.1f}%)")

if __name__ == "__main__":
    evaluate()
