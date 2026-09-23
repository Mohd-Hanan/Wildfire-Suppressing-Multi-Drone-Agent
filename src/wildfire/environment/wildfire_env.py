import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from wildfire.simulation.world import World
from wildfire.environment.observation import ObservationBuilder
from wildfire.environment.action import ActionExecutor
from wildfire.environment.reward import RewardCalculator
from wildfire.environment.termination import TerminationChecker

class WildfireEnv(gym.Env):
    """
    Gymnasium-compatible environment orchestrating the Wildfire Simulator.
    """
    
    def __init__(self, config_path: str = "configs/environment.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        env_config = self.config.get('environment', {})
        self.map_width = int(env_config.get('width', 48))
        self.map_height = int(env_config.get('height', 48))
        
        # Core components
        self.obs_builder = ObservationBuilder(window_size=11)
        self.action_executor = ActionExecutor()
        self.reward_calculator = RewardCalculator(self.config)
        self.termination_checker = TerminationChecker(self.config)
        
        # Action space: Discrete(7)
        self.action_space = spaces.Discrete(7)
        
        # Observation space matching exactly the existing observation system
        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(low=0.0, high=1.0, shape=(5, 11, 11), dtype=np.float32),
            "drone": spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32),
            "wind": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        })
        
        self.world = None
        self.step_count = 0
        self.controlled_drone_idx = 0 
        
    def reset(self, *, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.world = World(self.map_width, self.map_height, seed=seed)
        
        self.reward_calculator.reset()
        self.termination_checker.reset()
        
        drone = self.world.drones[self.controlled_drone_idx]
        obs = self.obs_builder.get_observation(drone, self.world)
        
        return obs, {}
        
    def step(self, action: int):
        """Executes one step in the environment."""
        drone = self.world.drones[self.controlled_drone_idx]
        
        # A. Execute action
        hit_boundary, suppressed_cells = self.action_executor.execute(drone, self.world, action)
        
        # B. Advance the wildfire/world simulation by one timestep
        # We step the fire physics directly, skipping the dummy random-walk AI in world.step()
        self.world.fire_manager.step(self.world.terrain, self.world.wind)
        # Process base refills for all drones
        self.world.process_base_refills()
        
        # C. Increment step_count
        self.step_count += 1
        
        # D. Calculate reward
        reward, reward_info = self.reward_calculator.calculate(self.world, self.world.drones, hit_boundary, suppressed_cells)
        
        # E. Check termination
        terminated, truncated, term_info = self.termination_checker.check(self.world, self.step_count)
        
        # F. Build next observation
        obs = self.obs_builder.get_observation(drone, self.world)
        
        # G. Build info dict
        info = {}
        info.update(reward_info)
        info.update(term_info)
        
        return obs, float(reward), terminated, truncated, info
