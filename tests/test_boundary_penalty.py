import pytest
import yaml
from wildfire.environment.wildfire_env import WildfireEnv

def test_boundary_penalty():
    env = WildfireEnv()
    obs, info = env.reset(seed=42)
    drone = env.world.drones[env.controlled_drone_idx]
    
    # Reset position to a safe middle ground
    drone.x = 24
    drone.y = 24
    
    # Test valid movement -> no boundary penalty
    obs, reward, terminated, truncated, info = env.step(1) # NORTH
    assert info['boundary_penalty'] == 0.0, f"Expected 0.0 penalty for valid move, got {info['boundary_penalty']}"
    assert drone.y == 23
    
    # Test North from y=0
    drone.y = 0
    obs, reward, terminated, truncated, info = env.step(1) # NORTH
    assert info['boundary_penalty'] == -0.1, f"Expected -0.1 penalty for blocked North, got {info['boundary_penalty']}"
    assert drone.y == 0
    
    # Test South from y=47
    drone.y = 47
    obs, reward, terminated, truncated, info = env.step(2) # SOUTH
    assert info['boundary_penalty'] == -0.1, f"Expected -0.1 penalty for blocked South, got {info['boundary_penalty']}"
    assert drone.y == 47
    
    # Test West from x=0
    drone.x = 0
    obs, reward, terminated, truncated, info = env.step(4) # WEST
    assert info['boundary_penalty'] == -0.1, f"Expected -0.1 penalty for blocked West, got {info['boundary_penalty']}"
    assert drone.x == 0
    
    # Test East from x=47
    drone.x = 47
    obs, reward, terminated, truncated, info = env.step(3) # EAST
    assert info['boundary_penalty'] == -0.1, f"Expected -0.1 penalty for blocked East, got {info['boundary_penalty']}"
    assert drone.x == 47
    
    # Test STAY does not receive penalty
    obs, reward, terminated, truncated, info = env.step(0) # STAY
    assert info['boundary_penalty'] == 0.0, f"Expected 0.0 penalty for STAY, got {info['boundary_penalty']}"

def test_battery_behavior():
    env = WildfireEnv()
    obs, info = env.reset(seed=42)
    drone = env.world.drones[env.controlled_drone_idx]
    
    # Test battery decreases correctly on valid move
    drone.x, drone.y = 24, 24
    initial_battery = drone.battery
    env.step(1) # NORTH
    assert drone.battery == initial_battery - env.world.config['drone']['water']['move_cost']
    
    # Test battery decreases correctly on blocked move
    drone.y = 0
    initial_battery = drone.battery
    env.step(1) # NORTH
    assert drone.battery == initial_battery - env.world.config['drone']['water']['move_cost']

if __name__ == "__main__":
    pytest.main(["tests/test_boundary_penalty.py", "-v"])
