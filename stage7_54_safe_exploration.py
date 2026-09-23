import torch
import numpy as np
from stage7_52_network import ActorCritic6Channels
from wildfire.rl.ppo_trainer import PPOTrainer
from wildfire.rl.rollout import RolloutCollector
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.gae import compute_gae
from stage7_52_obs_builder import GlobalFireObservationBuilder

class SafeExplorationEnv(WildfireEnv):
    def __init__(self, config_path):
        super().__init__(config_path)

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)
        
        # Center drone
        drone = self.world.drones[0]
        drone.x = 24
        drone.y = 24
        
        # Clear fires
        self.world.fire_manager.fire_map[:] = 0
        self.world.fire_manager.burn_timers[:] = 0
        
        # Spawn fire 2-3 cells away
        valid_positions = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                dist = abs(dx) + abs(dy)
                if 2 <= dist <= 3:
                    fx, fy = drone.x + dx, drone.y + dy
                    if 0 <= fx < self.world.width and 0 <= fy < self.world.height:
                        if self.world.terrain.fuel[fx, fy] > 0:
                            valid_positions.append((fx, fy))
        
        if len(valid_positions) > 0:
            idx = np.random.randint(len(valid_positions))
            fx, fy = valid_positions[idx]
            self.world.fire_manager.ignite(fx, fy)
        else:
            self.world.fire_manager.ignite(drone.x + 2, drone.y)
            
        obs = self.obs_builder.get_observation(drone, self.world)
        return obs, info

def evaluate_diagnostic(policy, env, device):
    print("\n--- DIAGNOSTIC: DIRECTION SENSITIVITY ---")
    policy.eval()
    
    # NORTH, SOUTH, EAST, WEST
    directions = {
        'NORTH': (0, -2),
        'SOUTH': (0, 2),
        'EAST': (2, 0),
        'WEST': (-2, 0)
    }
    
    actions_labels = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    for name, (dx, dy) in directions.items():
        obs, _ = env.reset()
        drone = env.world.drones[0]
        drone.x = 24
        drone.y = 24
        env.world.fire_manager.fire_map[:] = 0
        env.world.fire_manager.burn_timers[:] = 0
        env.world.fire_manager.ignite(drone.x + dx, drone.y + dy)
        
        obs = env.obs_builder.get_observation(drone, env.world)
        
        with torch.no_grad():
            spatial = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0).to(device)
            drone_vec = torch.tensor(obs['drone'], dtype=torch.float32).unsqueeze(0).to(device)
            wind_vec = torch.tensor(obs['wind'], dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = policy({'spatial': spatial, 'drone': drone_vec, 'wind': wind_vec})
            logits[0, 6] = -1e9
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            
        print(f"FIRE {name}:")
        for i, act in enumerate(actions_labels):
            if i != 6:
                print(f"  {act:<5s} {probs[i]*100:4.1f}%")

def evaluate_stochastic(policy, env, device, episodes=20):
    print("\n--- 20-EPISODE STOCHASTIC EVALUATION ---")
    policy.eval()
    
    total_reward = []
    total_suppressions = []
    water_attempts = 0
    water_successes = 0
    steps_closer = 0
    steps_farther = 0
    steps_same = 0
    crashes = 0
    ep_lengths = []
    action_counts = np.zeros(7)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0
        ep_supp = 0
        
        drone = env.world.drones[0]
        fm = env.world.fire_manager.fire_map
        active_coords = np.argwhere((fm == 1) | (fm == 2) | (fm == 3))
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
                
            action_counts[action] += 1
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += reward
            ep_supp += info.get('newly_suppressed_cells', 0)
            
            if action == 5:
                water_attempts += 1
                if info.get('newly_suppressed_cells', 0) > 0:
                    water_successes += 1
            
            fm = env.world.fire_manager.fire_map
            active_coords = np.argwhere((fm == 1) | (fm == 2) | (fm == 3))
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
        ep_lengths.append(step_count)
        
    tot_steps = steps_closer + steps_farther + steps_same
    print(f"Mean reward:              {np.mean(total_reward):.2f}")
    print(f"Mean suppressions:        {np.mean(total_suppressions):.2f}")
    print(f"Total suppressions:       {np.sum(total_suppressions)}")
    print(f"WATER attempts:           {water_attempts}")
    print(f"WATER success rate:       {(water_successes/max(1,water_attempts)*100):.1f}%")
    print(f"Steps moving closer:      {(steps_closer/max(1,tot_steps)*100):.1f}%")
    print(f"Steps moving farther:     {(steps_farther/max(1,tot_steps)*100):.1f}%")
    print(f"Steps same distance:      {(steps_same/max(1,tot_steps)*100):.1f}%")
    print(f"Crash rate:               {(crashes/episodes)*100:.1f}%")
    print(f"Mean episode length:      {np.mean(ep_lengths):.1f}")
    
    print("\nAction Distribution:")
    total_acts = np.sum(action_counts)
    actions_labels = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    for i, act in enumerate(actions_labels):
        if i != 6:
            print(f"Action {i} ({act}): {action_counts[i]/max(1,total_acts)*100:.1f}%")

def main():
    device = torch.device('cpu')
    print("STAGE 7.54: SAFE EXPLORATION / BOUNDARY-NEUTRAL DIAGNOSTIC")
    
    policy = ActorCritic6Channels(device=device, drone_type="WATER")
    # FRESH INITIALIZATION - DO NOT LOAD ANY PREVIOUS CHECKPOINT!
    policy.train()
    
    trainer = PPOTrainer(
        policy=policy, device=device, learning_rate=3e-4, clip_epsilon=0.2,
        value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
        ppo_epochs=4, minibatch_size=64
    )
    
    env = SafeExplorationEnv("configs/environment.yaml")
    env.obs_builder = GlobalFireObservationBuilder(window_size=11)
    
    collector = RolloutCollector(env, policy, rollout_size=256, device=device)
    
    for update in range(1, 21):
        collector.collect()
        spatial, drone, wind, acts, rewards, terms, truncs, values, old_log_probs = collector.buffer.get_batch(device)
        adv, ret = compute_gae(rewards, values, terms, truncs, collector.last_value if collector.last_value else 0.0)
        train_metrics = trainer.update(spatial, drone, wind, acts, old_log_probs, adv, ret)
        
        print(f"Update {update:2d}/20 | Reward: {np.mean(collector.buffer.rewards):7.4f} | "
              f"Entropy: {train_metrics.get('entropy', 0.0):5.4f} | "
              f"WATER%: {(acts == 5).float().mean().item()*100:5.1f}%")
              
    chk = "stage7_54_safe_exploration_20updates.pth"
    torch.save(policy.state_dict(), chk)
    print(f"Saved {chk}")
    
    eval_env = SafeExplorationEnv("configs/environment.yaml")
    eval_env.obs_builder = GlobalFireObservationBuilder(window_size=11)
    
    evaluate_diagnostic(policy, eval_env, device)
    evaluate_stochastic(policy, eval_env, device, episodes=20)

if __name__ == "__main__":
    main()
