import torch
import pygame
import numpy as np
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer
from stage7_35_final_truth import SeparateActorCriticSymmetric64
import time

def run_visual():
    print("Starting visual evaluation for Stage 7.36 Battery Safety...")
    
    # Force single drone
    import yaml
    with open("configs/environment.yaml", 'r') as f:
        config = yaml.safe_load(f)
    config['drones']['num_water_drones'] = 1
    config['drones']['num_retardant_drones'] = 0
    config['reward']['drone_crash_penalty'] = 10.0
    config['return_to_base'] = {'battery_reserve': 5}
    
    with open("configs/temp_visual_config.yaml", 'w') as f:
        yaml.dump(config, f)
        
    env = WildfireEnv("configs/temp_visual_config.yaml")
    renderer = Renderer(width=env.map_width, height=env.map_height)
    
    device = torch.device("cpu")
    policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER")
    
    # Load the checkpoint
    policy.load_state_dict(torch.load("stage7_36_battery_20updates.pth", map_location=device, weights_only=True))
    policy.eval() # Use eval mode (no gradients), but Categorical sampling is still active
    
    obs, _ = env.reset(seed=402) # random seed
    done = False
    step = 0
    
    clock = pygame.time.Clock()
    
    action_map = {0: "HOVER", 1: "NORTH", 2: "SOUTH", 3: "EAST", 4: "WEST", 5: "WATER", 6: "RETARDANT"}
    
    print("\n--- VISUAL TRACE LOG ---")
    while not done and step < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        with torch.no_grad():
            # get_action_and_value samples stochastically
            action_t, _, _, _ = policy.get_action_and_value(obs)
            
        action = action_t.item()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the world
        renderer.render_world(env.world)
        
        drone = env.world.drones[0]
        
        if step % 10 == 0 or not drone.active:
            margin = env.world.battery_margin(drone)
            req_bat = env.world.required_battery(drone)
            dist_base = env.world.distance_to_base(drone.x, drone.y)
            
            # Fire vector is in observation!
            # observation is 9-dimensional: [battery, payload, margin, dist_to_base, nx, ny, fire_dx, fire_dy, fire_distance]
            # Since obs['drone'] has shape (9,) or (1,9), let's extract:
            drone_obs = obs['drone']
            if len(drone_obs.shape) > 1:
                drone_obs = drone_obs.squeeze(0)
                
            fire_dx = drone_obs[6].item()
            fire_dy = drone_obs[7].item()
            fire_dist = drone_obs[8].item()
            
            print(f"\nStep {step:03d}:")
            print(f"Action:                  {action} ({action_map.get(action, 'UNKNOWN')})")
            print(f"Position:                ({drone.x}, {drone.y})")
            print(f"Battery:                 {drone.battery} / {drone.max_battery}")
            print(f"Required return battery: {req_bat}")
            print(f"Battery margin:          {margin}")
            print(f"Distance to base:        {dist_base}")
            print(f"Fire dx (normalized):    {fire_dx:.2f}")
            print(f"Fire dy (normalized):    {fire_dy:.2f}")
            print(f"Fire dist (normalized):  {fire_dist:.2f}")
            print(f"Payload:                 {drone.payload} / {drone.max_payload}")
            
        if not drone.active:
            print(f"\n*** Drone CRASHED at step {step} ***")
            break
            
        step += 1
        clock.tick(5) # Slow down to 5 FPS so user can watch easily
        
    renderer.close()
    pygame.quit()
    print("\nVisual evaluation finished.")

if __name__ == "__main__":
    run_visual()
