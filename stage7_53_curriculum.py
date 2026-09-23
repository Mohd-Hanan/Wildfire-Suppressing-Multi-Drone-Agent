import torch
import numpy as np
import time
from stage7_52_network import ActorCritic6Channels
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.rollout import RolloutCollector
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.gae import compute_gae
from stage7_52_obs_builder import GlobalFireObservationBuilder

class CurriculumWildfireEnv(WildfireEnv):
    def __init__(self, config_path, fire_dist_min=None, fire_dist_max=None):
        super().__init__(config_path)
        self.fire_dist_min = fire_dist_min
        self.fire_dist_max = fire_dist_max

    def set_distance(self, dmin, dmax):
        self.fire_dist_min = dmin
        self.fire_dist_max = dmax

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        
        if self.fire_dist_max is not None:
            self.world.fire_manager.fire_map[:] = 0
            self.world.fire_manager.burn_timers[:] = 0
            drone = self.world.drones[0]
            
            valid_positions = []
            for dx in range(-self.fire_dist_max, self.fire_dist_max + 1):
                for dy in range(-self.fire_dist_max, self.fire_dist_max + 1):
                    dist = abs(dx) + abs(dy)
                    if self.fire_dist_min <= dist <= self.fire_dist_max:
                        fx, fy = drone.x + dx, drone.y + dy
                        if 0 <= fx < self.world.width and 0 <= fy < self.world.height:
                            if self.world.terrain.fuel[fx, fy] > 0:
                                valid_positions.append((fx, fy))
            
            if len(valid_positions) > 0:
                idx = np.random.randint(len(valid_positions))
                fx, fy = valid_positions[idx]
                self.world.fire_manager.ignite(fx, fy)
            else:
                self.world.fire_manager.ignite_fires(1)
        
        obs = self.obs_builder.get_observation(drone, self.world)
        return obs, info

def evaluate_stage(policy, env, episodes=20, device=torch.device('cpu')):
    policy.eval()
    
    total_reward = []
    total_suppressions = []
    water_attempts = 0
    water_successes = 0
    steps_closer = 0
    steps_farther = 0
    steps_same = 0
    crashes = 0
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0
        ep_supp = 0
        
        drone = env.world.drones[0]
        fm = env.world.fire_manager.fire_map
        active_mask = (fm == 1) | (fm == 2) | (fm == 3)
        active_coords = np.argwhere(active_mask)
        prev_dist = np.min(np.abs(active_coords[:,0] - drone.x) + np.abs(active_coords[:,1] - drone.y)) if len(active_coords)>0 else None

        step_count = 0
        while not done and step_count < 500:
            step_count += 1
            with torch.no_grad():
                spatial = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0).to(device)
                drone_vec = torch.tensor(obs['drone'], dtype=torch.float32).unsqueeze(0).to(device)
                wind_vec = torch.tensor(obs['wind'], dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = policy({'spatial': spatial, 'drone': drone_vec, 'wind': wind_vec})
                logits[0, 6] = -1e9
                probs = torch.softmax(logits, dim=-1)[0].numpy()
                action = np.random.choice(7, p=probs)
                
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += reward
            ep_supp += info.get('newly_suppressed_cells', 0)
            
            if action == 5:
                water_attempts += 1
                if info.get('newly_suppressed_cells', 0) > 0:
                    water_successes += 1
            
            fm = env.world.fire_manager.fire_map
            active_mask = (fm == 1) | (fm == 2) | (fm == 3)
            active_coords = np.argwhere(active_mask)
            if len(active_coords) > 0:
                current_dist = np.min(np.abs(active_coords[:,0] - drone.x) + np.abs(active_coords[:,1] - drone.y))
                if prev_dist is not None:
                    if current_dist < prev_dist: steps_closer += 1
                    elif current_dist > prev_dist: steps_farther += 1
                    else: steps_same += 1
                prev_dist = current_dist
                
            if info.get('crash', False):
                crashes += 1
                
        total_reward.append(ep_r)
        total_suppressions.append(ep_supp)
        
    policy.train()
    
    tot_steps = steps_closer + steps_farther + steps_same
    return {
        'mean_reward': np.mean(total_reward),
        'mean_suppressions': np.mean(total_suppressions),
        'total_suppressions': np.sum(total_suppressions),
        'water_attempts': water_attempts,
        'water_success_rate': (water_successes / max(1, water_attempts)) * 100,
        'closer_pct': (steps_closer / max(1, tot_steps)) * 100,
        'farther_pct': (steps_farther / max(1, tot_steps)) * 100,
        'crash_rate': (crashes / episodes) * 100
    }

def main():
    device = torch.device('cpu')
    print("STAGE 7.53: DISTANCE CURRICULUM")
    
    policy = ActorCritic6Channels(device=device, drone_type="WATER")
    trainer = PPOTrainer(
        policy=policy, device=device, learning_rate=3e-4, clip_epsilon=0.2,
        value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
        ppo_epochs=4, minibatch_size=64
    )
    
    env = CurriculumWildfireEnv("configs/environment.yaml")
    env.obs_builder = GlobalFireObservationBuilder(window_size=11)
    
    eval_env = CurriculumWildfireEnv("configs/environment.yaml")
    eval_env.obs_builder = GlobalFireObservationBuilder(window_size=11)
    
    stages = [
        ("03", 2, 3),
        ("07", 5, 7),
        ("12", 10, 12),
        ("20", 15, 20),
        ("30", 25, 30),
        ("Normal", None, None)
    ]
    
    for stage_name, dmin, dmax in stages:
        print(f"\n=============================================")
        print(f"CURRICULUM STAGE: {stage_name} (Dist: {dmin}-{dmax})")
        print(f"=============================================")
        
        env.set_distance(dmin, dmax)
        eval_env.set_distance(dmin, dmax)
        collector = RolloutCollector(env, policy, rollout_size=256, device=device)
        
        stage_learned = False
        updates_done = 0
        
        for block in range(5): # 5 blocks of 20 updates = 100 max
            for u in range(20):
                collector.collect()
                spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
                adv, ret = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
                trainer.update(spatial, drone, wind, acts, old_log_probs, adv, ret)
                updates_done += 1
                
            # Evaluate after 20 updates
            metrics = evaluate_stage(policy, eval_env, episodes=10) # 10 eps to test quickly
            print(f"Updates {updates_done}/100 | Reward: {metrics['mean_reward']:.2f} | "
                  f"Supp: {metrics['mean_suppressions']:.2f} | "
                  f"WATER Success: {metrics['water_success_rate']:.1f}% | "
                  f"Closer: {metrics['closer_pct']:.1f}% | Farther: {metrics['farther_pct']:.1f}%")
                  
            if metrics['water_success_rate'] >= 30.0 and metrics['closer_pct'] > metrics['farther_pct'] and metrics['total_suppressions'] > 0:
                print(f">>> STAGE LEARNED after {updates_done} updates!")
                stage_learned = True
                break
                
        if not stage_learned:
            print(f">>> FAILED to learn stage after 100 updates. Stopping curriculum.")
            break
            
        if stage_name != "Normal":
            chk = f"stage7_53_curriculum_{stage_name}.pth"
            torch.save(policy.state_dict(), chk)
            print(f"Saved {chk}")

if __name__ == "__main__":
    main()
