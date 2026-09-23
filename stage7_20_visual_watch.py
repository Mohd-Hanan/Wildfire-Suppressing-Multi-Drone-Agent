import torch
import pygame
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rendering.renderer import Renderer
from stage7_15_smoke_test import SeparateActorCritic

def main():
    print("========================================")
    print("STAGE 7.20 — STOCHASTIC VISUAL EVALUATION")
    print("========================================")
    
    device = torch.device("cpu")
    policy = SeparateActorCritic(device=device, drone_type="WATER")
    
    checkpoint_path = "stage7_single_drone_200_boundary_penalty.pth"
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    eval_env = WildfireEnv()
    renderer = Renderer(eval_env.map_width, eval_env.map_height)
    clock = pygame.time.Clock()
    
    action_names = ["STAY", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "RETARDANT"]
    
    print("\nStarting Pygame. Press ESC or close the window to exit early.")
    
    # Run only ONE episode
    ep = 1
    # Use fixed seed for environment generation to maintain consistency, 
    # but NO manual_seed for PyTorch to allow stochastic action sampling
    obs, _ = eval_env.reset(seed=4000 + ep)
    terminated = False
    truncated = False
    step = 0
    total_reward = 0
    
    print(f"\n--- VISUAL EPISODE {ep} ---")
    print(f"{'Step':>5} | {'Action':<15} | {'Position':<10} | {'Bat':>4} | {'Pay':>3}")
    print("-" * 55)
    
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return
                
        renderer.render_world(eval_env.world)
        pygame.display.flip()
        
        with torch.no_grad():
            obs_t = {
                "spatial": torch.tensor(obs["spatial"], dtype=torch.float32, device=device).unsqueeze(0),
                "drone": torch.tensor(obs["drone"], dtype=torch.float32, device=device).unsqueeze(0),
                "wind": torch.tensor(obs["wind"], dtype=torch.float32, device=device).unsqueeze(0),
            }
            logits, _ = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            
        drone = eval_env.world.drones[eval_env.controlled_drone_idx]
        pos_str = f"({drone.x},{drone.y})"
        
        # Print diagnostic every 10 steps (and step 0)
        if step % 10 == 0:
            act_str = f"{action} ({action_names[action]})"
            print(f"{step:5d} | {act_str:<15} | {pos_str:<10} | {drone.battery:4d} | {drone.payload:3d}")
            
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        step += 1
        clock.tick(15)
        
    print("-" * 55)
    print(f"Episode ended. Reward: {total_reward:.1f}, Steps: {step}")
    print("Waiting... Close window to end program.")
    
    # Keep window open until user closes it
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                waiting = False
        clock.tick(15)
        
    pygame.quit()
    print("\nVisual inspection finished.")

if __name__ == "__main__":
    main()
