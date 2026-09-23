import torch
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_35_final_truth import SeparateActorCriticSymmetric64

def run_diagnostic():
    print("========================================")
    print("STAGE 7.45 — BEHAVIOR DIAGNOSTIC")
    print("========================================")
    
    device = torch.device("cpu")
    env = WildfireEnv("configs/environment.yaml")
    
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_44_candidate_physics_20updates.pth", map_location=device, weights_only=True))
    policy.eval()
    
    water_logs = []
    
    all_dist_to_fire = []
    term_dist_to_fire = []
    reached_active_fire = False
    
    print("\nRunning 20 evaluation episodes...")
    for ep in range(20):
        obs, _ = env.reset(seed=100+ep)
        done = False
        steps = 0
        
        while not done and steps < 500:
            fm = env.world.fire_manager.fire_map
            active_mask = (fm == 1) | (fm == 2) | (fm == 3)
            active_coords = np.argwhere(active_mask)
            
            drone = env.world.drones[0]
            
            if len(active_coords) > 0:
                distances = np.abs(active_coords[:, 0] - drone.x) + np.abs(active_coords[:, 1] - drone.y)
                nearest_idx = np.argmin(distances)
                nearest_x, nearest_y = active_coords[nearest_idx]
                dist = distances[nearest_idx]
                dx = nearest_x - drone.x
                dy = nearest_y - drone.y
                on_fire = dist == 0
                if on_fire:
                    reached_active_fire = True
            else:
                dist = None
                dx = 0
                dy = 0
                nearest_x, nearest_y = -1, -1
                on_fire = False
                
            if dist is not None:
                all_dist_to_fire.append(dist)
                
            with torch.no_grad():
                action_t, _, _, _ = policy.get_action_and_value(obs)
            
            a = action_t.item()
            
            if a == 5:
                # Log this WATER action
                log_entry = {
                    'ep': ep,
                    'step': steps,
                    'drone_x': drone.x,
                    'drone_y': drone.y,
                    'fire_x': nearest_x,
                    'fire_y': nearest_y,
                    'dx': dx,
                    'dy': dy,
                    'dist_before': dist,
                    'on_fire': on_fire,
                    'battery': drone.battery,
                    'payload': drone.payload
                }
                
                obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                
                suppressed = info.get('newly_suppressed_cells', 0) > 0
                log_entry['suppressed'] = suppressed
                
                # dist after
                fm_after = env.world.fire_manager.fire_map
                active_mask_after = (fm_after == 1) | (fm_after == 2) | (fm_after == 3)
                active_coords_after = np.argwhere(active_mask_after)
                
                if len(active_coords_after) > 0:
                    distances_after = np.abs(active_coords_after[:, 0] - drone.x) + np.abs(active_coords_after[:, 1] - drone.y)
                    dist_after = np.min(distances_after)
                else:
                    dist_after = None
                log_entry['dist_after'] = dist_after
                
                water_logs.append(log_entry)
            else:
                obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                
            steps += 1
            
        if dist is not None:
            term_dist_to_fire.append(dist)
            
    print("\n--- WATER ACTION STATISTICS ---")
    total_water = len(water_logs)
    successful_water = sum(1 for l in water_logs if l['suppressed'])
    failed_water = total_water - successful_water
    success_rate = (successful_water / total_water * 100) if total_water > 0 else 0.0
    
    print(f"Total WATER actions: {total_water}")
    print(f"Successful WATER actions: {successful_water}")
    print(f"Failed WATER actions: {failed_water}")
    print(f"WATER success rate: {success_rate:.1f}%")
    
    dists_water = [l['dist_before'] for l in water_logs if l['dist_before'] is not None]
    if dists_water:
        print(f"Mean distance-to-fire when WATER selected: {np.mean(dists_water):.1f}")
        print(f"Median distance-to-fire when WATER selected: {np.median(dists_water):.1f}")
        print(f"Minimum distance-to-fire when WATER selected: {np.min(dists_water):.1f}")
        
        le1 = sum(1 for d in dists_water if d <= 1) / len(dists_water) * 100
        gt5 = sum(1 for d in dists_water if d > 5) / len(dists_water) * 100
        gt10 = sum(1 for d in dists_water if d > 10) / len(dists_water) * 100
        
        print(f"WATER actions with dist <= 1: {le1:.1f}%")
        print(f"WATER actions with dist > 5:  {gt5:.1f}%")
        print(f"WATER actions with dist > 10: {gt10:.1f}%")
        
    print("\n--- ALL ACTION STATISTICS ---")
    if all_dist_to_fire:
        print(f"Mean distance-to-fire: {np.mean(all_dist_to_fire):.1f}")
        print(f"Minimum distance-to-fire: {np.min(all_dist_to_fire):.1f}")
    if term_dist_to_fire:
        print(f"Distance-to-fire at termination (mean): {np.mean(term_dist_to_fire):.1f}")
    print(f"Drone ever reached active fire cell: {reached_active_fire}")

def run_policy_diagnostic():
    print("\n========================================")
    print("POLICY DIRECTION DIAGNOSTIC")
    print("========================================")
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    policy.load_state_dict(torch.load("stage7_44_candidate_physics_20updates.pth", map_location=device, weights_only=True))
    policy.eval()
    
    # 0: battery, 1: payload, 2: margin, 3: dist_base, 4: nx, 5: ny, 6: f_dx, 7: f_dy, 8: f_dist
    base_drone = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5], dtype=np.float32)
    spatial = torch.zeros((1, 5, 11, 11), dtype=torch.float32)
    wind = torch.zeros((1, 3), dtype=torch.float32)
    
    directions = {
        "NORTH (dy=-0.5)": (0.0, -0.5),
        "SOUTH (dy=0.5)": (0.0, 0.5),
        "EAST (dx=0.5)": (0.5, 0.0),
        "WEST (dx=-0.5)": (-0.5, 0.0),
        "NE (dx=0.5, dy=-0.5)": (0.5, -0.5),
        "NW (dx=-0.5, dy=-0.5)": (-0.5, -0.5),
        "SE (dx=0.5, dy=0.5)": (0.5, 0.5),
        "SW (dx=-0.5, dy=0.5)": (-0.5, 0.5)
    }
    
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER"]
    
    for name, (dx, dy) in directions.items():
        drone_state = base_drone.copy()
        drone_state[6] = dx
        drone_state[7] = dy
        
        obs = {
            "spatial": spatial,
            "drone": torch.tensor(drone_state).unsqueeze(0),
            "wind": wind,
            "action_mask": torch.ones((1, 6), dtype=torch.float32)
        }
        
        with torch.no_grad():
            logits, _ = policy.forward(obs)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            
        print(f"\nFire Direction: {name}")
        for i, act_name in enumerate(action_names):
            print(f"  {act_name:5s}: {probs[i]*100:5.1f}%")

if __name__ == "__main__":
    run_diagnostic()
    run_policy_diagnostic()
