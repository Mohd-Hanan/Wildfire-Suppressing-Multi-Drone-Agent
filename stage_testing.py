import argparse
import time
import numpy as np
import pygame
import os

from wildfire.simulation.world import World
from wildfire.simulation.drone import DroneType
from wildfire.agents.testing_controller import MultiDroneTestingController
from wildfire.environment.action import ActionExecutor

def run_testing(headless=False, episodes=5):
    if not headless:
        pygame.init()
        from wildfire.rendering.renderer import Renderer
        renderer = Renderer(width=48, height=48, cell_size=16)
        
    for ep in range(episodes):
        print(f"\n==========================================")
        print(f"EPISODE {ep+1}")
        
        world = World(48, 48)
        world.drones[0].x, world.drones[0].y = world.base_x, world.base_y
        world.drones[1].x, world.drones[1].y = world.base_x + 1, world.base_y
        world.drones[2].x, world.drones[2].y = world.base_x, world.base_y + 1
        world.drones[3].x, world.drones[3].y = world.base_x + 1, world.base_y + 1
        controller = MultiDroneTestingController(world)
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
                        
            actions = controller.get_actions()
            
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
                

                pre_payload = drone.payload
                hit_boundary, suppressed = executor.execute(drone, world, act)
                post_payload = drone.payload
                
                # If payload decreased, a successful drop occurred
                if post_payload < pre_payload and not headless and hasattr(renderer, 'deployment_effects'):
                    import time
                    dtype = "WATER" if drone.type == DroneType.WATER else "RETARDANT"
                    duration = 0.8 if dtype == "WATER" else 1.0
                    renderer.deployment_effects.append({
                        "x": drone.x,
                        "y": drone.y,
                        "type": dtype,
                        "time": time.time(),
                        "duration": duration
                    })

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
    args = parser.parse_args()
    
    os.environ['CONTROLLER_MODE'] = "DEMO"
    run_testing(headless=args.headless, episodes=args.episodes)
