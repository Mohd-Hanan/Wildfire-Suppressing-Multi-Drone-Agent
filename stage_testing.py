import argparse
import time
import numpy as np
import pygame
import os

from wildfire.simulation.world import World
from wildfire.simulation.drone import DroneType
from stage7_52_network import ActorCritic6Channels
from stage7_52_obs_builder import GlobalFireObservationBuilder
import torch
from torch.distributions import Categorical
from wildfire.environment.action import ActionExecutor

def run_testing(headless=False, episodes=5, seed=None):
    if not headless:
        pygame.init()
        from wildfire.rendering.renderer import Renderer
        renderer = Renderer(width=48, height=48, cell_size=16)
        
    device = torch.device('cpu')
    model_path = "stage7_52_global_fire_channel_200updates.pth"
    model = ActorCritic6Channels(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    obs_builder = GlobalFireObservationBuilder(window_size=11)
    
    if seed is not None:
        np.random.seed(seed)
        
    for ep in range(episodes):
        print(f"\n==========================================")
        print(f"EPISODE {ep+1}")
        
        ep_seed = seed + ep if seed is not None else None
        world = World(48, 48, seed=ep_seed)
        
        # Randomize initial drone positions within a 5x5 region around base
        rng = world.terrain.rng
        for d in world.drones:
            d.x = np.clip(world.base_x + rng.integers(-2, 3), 0, 47)
            d.y = np.clip(world.base_y + rng.integers(-2, 3), 0, 47)
            
        # Ignite 1-3 fires to randomize fire size/intensity
        num_fires = rng.integers(1, 4)
        for _ in range(num_fires):
            fx = rng.integers(5, 43)
            fy = rng.integers(5, 43)
            world.fire_manager.ignite(fx, fy)
            
        executor = ActionExecutor()
        
        from wildfire.environment.reward import RewardCalculator
        reward_calculator = RewardCalculator({})
        reward_calculator.reset()
        episode_reward = 0.0
        
        initial_fire_cells = np.sum(world.fire_manager.fire_map > 0)

        print(f"Initial Fire Cells: {initial_fire_cells}")
        
        step_count = 0
        total_suppressed = 0
        water_deps = 0
        retardant_deps = 0
        crashes = 0
        collisions = 0
        action_counts = {0:{}, 1:{}, 2:{}, 3:{}}
        unique_targets = {0:set(), 1:set(), 2:set(), 3:set()}
        target_changes = {0:0, 1:0, 2:0, 3:0}
        prev_targets = {0:None, 1:None, 2:None, 3:None}
        history = {0:[], 1:[], 2:[], 3:[]}

        
        while step_count < 1000:
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                        
            actions = {}
            for d in world.drones:
                if not d.active:
                    continue
                obs = obs_builder.get_observation(d, world)
                spatial = torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0)
                drone_vec = torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0)
                wind_vec = torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    logits, _ = model({"spatial": spatial, "drone": drone_vec, "wind": wind_vec})
                    # Use stochastic policy
                    dist = Categorical(logits=logits)
                    act = dist.sample().item()
                    actions[d.id] = act
            
            step_log = f"Step {step_count:4d} | "
            for d in world.drones:
                act = actions.get(d.id, 0)
                act_name = ['STAY', 'N', 'S', 'E', 'W', 'WATER', 'RETARD'][act]
                step_log += f"D{d.id} ({d.x:2d},{d.y:2d}) Bat:{int(d.battery):3d} {act_name:6s} | "
            print(step_log)

            
            for drone in world.drones:
                if not drone.active:
                    continue
                act = actions.get(drone.id, 0)
                action_counts[drone.id][act] = action_counts[drone.id].get(act, 0) + 1
                if step_count <= 20:
                    history[drone.id].append((drone.x, drone.y, act))
                if hasattr(drone, "current_target"):
                    curr = drone.current_target
                    unique_targets[drone.id].add(curr)
                    if prev_targets[drone.id] is not None and prev_targets[drone.id] != curr:
                        target_changes[drone.id] += 1
                    prev_targets[drone.id] = curr

                
                # Use ActionExecutor directly
                pre_active = np.sum((world.fire_manager.fire_map == 1) | (world.fire_manager.fire_map == 2) | (world.fire_manager.fire_map == 3))
                pre_pos = (drone.x, drone.y)
                pre_targ = drone.current_target if hasattr(drone, 'current_target') else (0,0)
                
                hit_boundary, suppressed = executor.execute(drone, world, act)
                total_suppressed += suppressed
                
                rew, _ = reward_calculator.calculate(
                    world, world.drones, hit_boundary, suppressed, act, 
                    pre_pos, pre_targ, pre_active
                )
                episode_reward += rew

                
                if act == 5: water_deps += 1
                if act == 6: retardant_deps += 1
                        
                if drone.battery <= 0 and not world.is_at_base(drone):
                    print(f"CRASH: Drone {drone.id} at ({drone.x},{drone.y}) battery={drone.battery}, payload={drone.payload}")
                    crashes += 1
            
            # Check collisions
            positions = set()
            for drone in world.drones:
                if drone.active:
                    if (drone.x, drone.y) in positions:
                        collisions += 1
                    positions.add((drone.x, drone.y))
            
            # Step environment
            world.fire_manager.step(world.terrain, world.wind)
            world.process_base_refills()
            
            if not headless:
                renderer.render_world(world)
                time.sleep(0.05)
                
            active_fires = np.sum((world.fire_manager.fire_map == 1) | (world.fire_manager.fire_map == 2) | (world.fire_manager.fire_map == 3))
            
            drones_safe = all((not d.active or world.is_at_base(d)) for d in world.drones)
            
            if active_fires == 0 and drones_safe:
                print(f"Fire extinguished and all drones returned safely at step {step_count}!")
                break
                
            step_count += 1
            
        extinguished = (np.sum((world.fire_manager.fire_map == 1) | (world.fire_manager.fire_map == 2) | (world.fire_manager.fire_map == 3)) == 0)
        
        print(f"Episode Length: {step_count}")
        print(f"Final Active Fire Cells: {np.sum((world.fire_manager.fire_map == 1) | (world.fire_manager.fire_map == 2) | (world.fire_manager.fire_map == 3))}")
        print(f"Total Suppressed Cells: {total_suppressed}")
        print(f"WATER Deployments: {water_deps}")
        print(f"RETARDANT Deployments: {retardant_deps}")
        print(f"Drone Collisions: {collisions}")
        print(f"Drone Crashes: {crashes}")
        print(f"Number of drones remaining: {4 - crashes}")
        print(f"Fire Extinction Status: {'SUCCESS' if extinguished else 'FAILED'}")
        print(f"Total Episode Reward: {episode_reward:.2f}")
        
        for drone_id in range(4):
            print(f"Drone {drone_id} Actions: {action_counts[drone_id]}")
            print(f"Drone {drone_id} Unique Targets: {len(unique_targets[drone_id])}")
            print(f"Drone {drone_id} Target Changes: {target_changes[drone_id]}")




    if not headless:
        renderer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without Pygame")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Reproducible random seed")
    args = parser.parse_args()
    
    os.environ['CONTROLLER_MODE'] = "RL_DEMO"
    run_testing(headless=args.headless, episodes=args.episodes, seed=args.seed)
