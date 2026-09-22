import numpy as np
from typing import Tuple, Dict
from wildfire.simulation.world import World
from wildfire.simulation.fire import FireState

class TerminationChecker:
    def __init__(self, config: dict):
        env_config = config.get('environment', {})
        self.max_steps = int(env_config.get('max_steps', 500))

    def reset(self):
        """Clears state from previous episodes."""
        pass # Currently, termination checks are entirely stateless based on world and step_count

    def check(self, world: World, step_count: int) -> Tuple[bool, bool, Dict[str, str]]:
        """
        Evaluates the terminal state of the simulation.
        Returns: (terminated, truncated, info)
        """
        # 1. Check natural termination (fire extinguished)
        fm = world.fire_manager.fire_map
        active_fire = np.sum((fm == FireState.IGNITING) | 
                             (fm == FireState.BURNING) | 
                             (fm == FireState.SMOLDERING))
        
        if active_fire == 0:
            return True, False, {"termination_reason": "fire_extinguished"}
            
        # 2. Check truncation (max steps reached)
        if step_count >= self.max_steps:
            return False, True, {"termination_reason": "max_steps"}
            
        # 3. Episode continues
        return False, False, {"termination_reason": None}
