import torch
import pygame
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer
from stage7_15_smoke_test import SeparateActorCritic

def main():
    print("========================================")
    print("STAGE 7.20 — RENDERER COORDINATE DIAGNOSTIC")
    print("========================================")
    
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    
    checkpoint_path = "stage7_18_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    eval_env = WildfireEnv()
    renderer = Renderer(eval_env.map_width, eval_env.map_height)
    clock = pygame.time.Clock()
    
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    print(f"Map Width: {eval_env.map_width}, Map Height: {eval_env.map_height}")
    print(f"Cell Size: {renderer.cell_size}")
    
    # 1. Identify controlled drone idx
    controlled_idx = eval_env.controlled_drone_idx
    print(f"Controlled Drone Index: {controlled_idx}")
    
    # 2. Verify eval_env.world object identity
    print(f"eval_env.world object ID: {id(eval_env.world)}")
    
    obs, _ = eval_env.reset(seed=4000)
    terminated = False
    truncated = False
    step = 0
    
    print("\nStarting Pygame. Running 1 Episode...")
    print(f"{'Step':>5} | {'Action':<10} | {'ID'} | {'World Pos':<10} | {'Screen Pos':<12} | {'Bat':>4} | {'Pay':>3}")
    print("-" * 75)
    
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                
        # 3. Verify same drones object is passed to renderer
        # renderer.render_world accesses world.drones
        renderer.render_world(eval_env.world)
        
        # 4. Add temporary diagnostic marker for Drone 0
        drone = eval_env.world.drones[controlled_idx]
        if drone.active:
            cx = int((drone.x + 0.5) * renderer.cell_size)
            cy = int((drone.y + 0.5) * renderer.cell_size)
            # Draw a bright green hollow box around the drone to highlight it
            pygame.draw.rect(renderer.screen, (0, 255, 0), (cx-10, cy-10, 20, 20), 2)
            
            # 5. Draw a line from the base to the drone to show how far it moved
            base_cx = int((eval_env.world.base_x + 1) * renderer.cell_size)
            base_cy = int((eval_env.world.base_y + 1) * renderer.cell_size)
            pygame.draw.line(renderer.screen, (255, 255, 0), (base_cx, base_cy), (cx, cy), 1)
        
        pygame.display.flip()
        
        with torch.no_grad():
            obs_t = {
                "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
            }
            logits, _ = policy(obs_t)
            action = torch.argmax(logits, dim=-1).item()
            
        pos_str = f"({drone.x},{drone.y})"
        screen_pos = f"({cx},{cy})" if drone.active else "CRASHED"
        
        if step % 10 == 0 or step < 5:
            act_str = f"{action} ({action_names[action]})"
            print(f"{step:5d} | {act_str:<10} | {drone.id:2d} | {pos_str:<10} | {screen_pos:<12} | {drone.battery:4d} | {drone.payload:3d}")
            
        obs, reward, terminated, truncated, info = eval_env.step(action)
        step += 1
        clock.tick(10) # Slower so user can watch the first 3 steps
        
    print("-" * 75)
    print("Episode ended. Close the Pygame window to return.")
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
        clock.tick(15)
        
    pygame.quit()

if __name__ == "__main__":
    main()
