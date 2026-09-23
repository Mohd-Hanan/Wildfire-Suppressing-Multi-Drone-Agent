import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_35_final_truth import SeparateActorCriticSymmetric64

def run_audit(policy=None, name="Random"):
    print(f"\n==================================================")
    print(f"AUDITING POLICY: {name}")
    print(f"==================================================")
    
    env = WildfireEnv("configs/environment.yaml")
    
    stats = {
        'total': 0.0,
        'damage': 0.0,
        'suppression': 0.0,
        'step': 0.0,
        'boundary': 0.0,
        'crash': 0.0,
        'battery': 0.0,
        'extinction': 0.0
    }
    
    ep_rewards = []
    
    succ_supp_count = 0
    water_count = 0
    natural_ext = 0
    supp_ext = 0
    crash_count = 0
    
    water_dists = []
    all_dists = []
    
    for ep in range(20):
        obs, _ = env.reset(seed=100+ep)
        done = False
        steps = 0
        ep_total = 0.0
        ep_supp = 0
        
        while not done and steps < 500:
            fm = env.world.fire_manager.fire_map
            active_mask = (fm == 1) | (fm == 2) | (fm == 3)
            active_coords = np.argwhere(active_mask)
            drone = env.world.drones[0]
            
            if len(active_coords) > 0:
                dists = np.abs(active_coords[:,0]-drone.x) + np.abs(active_coords[:,1]-drone.y)
                f_dist = np.min(dists)
            else:
                f_dist = 0
                
            if policy is None:
                a = env.action_space.sample()
                # Bias random to 30% WATER to have some stats
                if np.random.rand() < 0.3:
                    a = 5
            else:
                with torch.no_grad():
                    action_t, _, _, _ = policy.get_action_and_value(obs)
                a = action_t.item()
                
            all_dists.append(f_dist)
            if a == 5:
                water_count += 1
                water_dists.append(f_dist)
                
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            
            ep_total += r
            stats['total'] += r
            stats['damage'] += info.get('damage_penalty', 0.0)
            stats['suppression'] += info.get('suppression_reward', 0.0)
            stats['step'] += info.get('step_penalty', 0.0)
            stats['boundary'] += info.get('boundary_penalty', 0.0)
            stats['crash'] += info.get('crash_penalty', 0.0)
            stats['battery'] += info.get('battery_safety_penalty', 0.0)
            stats['extinction'] += info.get('extinction_reward', 0.0)
            
            supp = info.get('newly_suppressed_cells', 0)
            ep_supp += supp
            succ_supp_count += supp
            
            steps += 1
            
        drone = env.world.drones[0]
        if not drone.active:
            crash_count += 1
            
        active_fire = np.sum((fm == 1) | (fm == 2) | (fm == 3))
        if active_fire == 0:
            if ep_supp > 0:
                supp_ext += 1
            else:
                natural_ext += 1
                
        ep_rewards.append(ep_total)
        
    print("\nREWARD COMPONENTS (TOTAL 20 EPS):")
    print(f"Damage:      {stats['damage']:.2f}")
    print(f"Suppression: {stats['suppression']:.2f}")
    print(f"Step:        {stats['step']:.2f}")
    print(f"Boundary:    {stats['boundary']:.2f}")
    print(f"Crash:       {stats['crash']:.2f}")
    print(f"Battery:     {stats['battery']:.2f}")
    print(f"Extinction:  {stats['extinction']:.2f}")
    print(f"TOTAL:       {stats['total']:.2f}")
    
    total_steps = len(all_dists)
    print("\nPER-EPISODE / PER-STEP MEANS:")
    print(f"Total Reward   - Ep Mean: {np.mean(ep_rewards):.2f}, Min: {np.min(ep_rewards):.2f}, Max: {np.max(ep_rewards):.2f}")
    print(f"Damage         - Ep: {stats['damage']/20:.2f}, Step: {stats['damage']/total_steps:.4f}")
    print(f"Suppression    - Ep: {stats['suppression']/20:.2f}, Step: {stats['suppression']/total_steps:.4f}")
    print(f"Battery Penalty- Ep: {stats['battery']/20:.2f}, Step: {stats['battery']/total_steps:.4f}")
    print(f"Crash Penalty  - Ep: {stats['crash']/20:.2f}, Step: {stats['crash']/total_steps:.4f}")
    
    abs_tot = (abs(stats['damage']) + abs(stats['suppression']) + abs(stats['step']) + 
               abs(stats['boundary']) + abs(stats['crash']) + abs(stats['battery']) + abs(stats['extinction']))
    
    print("\nABSOLUTE COMPONENT RATIOS:")
    if abs_tot > 0:
        print(f"Damage:      {abs(stats['damage'])/abs_tot*100:.1f}%")
        print(f"Suppression: {abs(stats['suppression'])/abs_tot*100:.1f}%")
        print(f"Battery:     {abs(stats['battery'])/abs_tot*100:.1f}%")
        print(f"Step:        {abs(stats['step'])/abs_tot*100:.1f}%")
        print(f"Crash:       {abs(stats['crash'])/abs_tot*100:.1f}%")
        print(f"Extinction:  {abs(stats['extinction'])/abs_tot*100:.1f}%")
        
    print("\nBEHAVIOR STATS:")
    print(f"WATER actions: {water_count}")
    print(f"Successful suppressions: {succ_supp_count}")
    print(f"WATER success rate: {(succ_supp_count/water_count*100) if water_count else 0:.1f}%")
    print(f"Mean fire distance during WATER: {np.mean(water_dists) if water_dists else 0:.1f}")
    print(f"Mean fire distance overall: {np.mean(all_dists) if all_dists else 0:.1f}")
    print(f"Natural extinctions: {natural_ext}")
    print(f"Suppression extinctions: {supp_ext}")
    print(f"Crashes: {crash_count}")

if __name__ == "__main__":
    run_audit(policy=None, name="Random Baseline")
    
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_44_candidate_physics_20updates.pth", map_location=device, weights_only=True))
    policy.eval()
    
    run_audit(policy=policy, name="Stage 7.44 Policy")
