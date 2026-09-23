import numpy as np
import torch
from wildfire.environment.wildfire_env import WildfireEnv
from stage7_15_smoke_test import SeparateActorCritic

def run_distribution_analysis():
    print("--- 1. DISTRIBUTION ANALYSIS (1000 Resets) ---")
    env = WildfireEnv()
    quadrants = {"NE": 0, "SE": 0, "NW": 0, "SW": 0, "AXIS": 0}
    all_dx, all_dy, all_dist = [], [], []
    
    for i in range(1000):
        obs, _ = env.reset(seed=10000 + i)
        drone_obs = obs["drone"]
        dx, dy, dist = drone_obs[6], drone_obs[7], drone_obs[8]
        
        all_dx.append(dx)
        all_dy.append(dy)
        all_dist.append(dist)
        
        if dx > 0 and dy < 0: quadrants["NE"] += 1
        elif dx > 0 and dy > 0: quadrants["SE"] += 1
        elif dx < 0 and dy < 0: quadrants["NW"] += 1
        elif dx < 0 and dy > 0: quadrants["SW"] += 1
        else: quadrants["AXIS"] += 1
        
    print(f"Quadrants: {quadrants}")
    print(f"fire_dx: mean={np.mean(all_dx):.3f}, std={np.std(all_dx):.3f}, min={np.min(all_dx):.3f}, max={np.max(all_dx):.3f}")
    print(f"fire_dy: mean={np.mean(all_dy):.3f}, std={np.std(all_dy):.3f}, min={np.min(all_dy):.3f}, max={np.max(all_dy):.3f}")
    print(f"fire_dist: mean={np.mean(all_dist):.3f}, std={np.std(all_dist):.3f}, min={np.min(all_dist):.3f}, max={np.max(all_dist):.3f}")

def check_episode_dynamics():
    print("\n--- 2. DYNAMICS ANALYSIS ---")
    # Simulate one random episode to see if quadrant changes significantly.
    env = WildfireEnv()
    obs, _ = env.reset(seed=12345)
    
    quadrants_visited = set()
    for _ in range(500):
        dx, dy = obs["drone"][6], obs["drone"][7]
        if dx > 0 and dy < 0: quadrants_visited.add("NE")
        elif dx > 0 and dy > 0: quadrants_visited.add("SE")
        elif dx < 0 and dy < 0: quadrants_visited.add("NW")
        elif dx < 0 and dy > 0: quadrants_visited.add("SW")
        
        # Take a random action
        action = np.random.randint(0, 7)
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc: break
        
    print(f"Quadrants visited by a random-walk drone in one episode: {quadrants_visited}")

def run_sensitivity_analysis():
    print("\n--- 3. SENSITIVITY ANALYSIS ---")
    device = torch.device("cpu")
    policy_smoke = SeparateActorCritic(device=device, drone_type="WATER")
    policy_smoke.load_state_dict(torch.load("stage7_32_smoke_global_fire_reward.pth", map_location=device, weights_only=True)['model_state_dict'])
    policy_smoke.eval()

    policy_200 = SeparateActorCritic(device=device, drone_type="WATER")
    policy_200.load_state_dict(torch.load("stage7_single_drone_200_global_fire_reward.pth", map_location=device, weights_only=True)['model_state_dict'])
    policy_200.eval()
    
    spatial = torch.zeros((1, 5, 11, 11), dtype=torch.float32)
    wind = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
    
    def get_max_diff(policy, base_drone, mod_drone):
        with torch.no_grad():
            l_base, _ = policy({"spatial": spatial, "wind": wind, "drone": base_drone})
            l_mod, _ = policy({"spatial": spatial, "wind": wind, "drone": mod_drone})
            p_base = torch.softmax(l_base, dim=-1)
            p_mod = torch.softmax(l_mod, dim=-1)
            logit_diff = torch.max(torch.abs(l_base - l_mod)).item()
            prob_diff = torch.max(torch.abs(p_base - p_mod)).item()
            return logit_diff, prob_diff
            
    base = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5]], dtype=torch.float32)
    
    # modify dx only
    mod_dx = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5]], dtype=torch.float32)
    # modify dy only
    mod_dy = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.5]], dtype=torch.float32)
    # modify dist only
    mod_dist = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0]], dtype=torch.float32)
    
    for name, pol in [("Smoke (20)", policy_smoke), ("Trained (200)", policy_200)]:
        print(f"Policy: {name}")
        ld, pd = get_max_diff(pol, base, mod_dx)
        print(f"  fire_dx change (0.0 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")
        ld, pd = get_max_diff(pol, base, mod_dy)
        print(f"  fire_dy change (0.0 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")
        ld, pd = get_max_diff(pol, base, mod_dist)
        print(f"  fire_distance (0.5 -> 1.0): Max logit diff = {ld:.5f}, Max prob diff = {pd*100:.2f}%")

if __name__ == "__main__":
    run_distribution_analysis()
    check_episode_dynamics()
    run_sensitivity_analysis()
