import numpy as np
from typing import Dict, Any, Optional
from wildfire.agents.base import Agent

class ControlledSuppressionAgent(Agent):
    """
    An evaluation-only agent designed to explicitly deploy suppression 
    mechanics to test whether they have a measurable effect on the simulation.
    It returns to base to refill, targets fire, and explicitly deploys 
    on unburned cells ADJACENT to fire (or on the fire) to maximize the 
    chance of mechanical interaction.
    """
    def reset(self, seed: Optional[int] = None):
        pass

    def _move_towards(self, tx: int, ty: int, cx: int = 5, cy: int = 5) -> int:
        dx = tx - cx
        dy = ty - cy
        if dx == 0 and dy == 0: return 0
        if abs(dx) > abs(dy): return 3 if dx > 0 else 4
        elif abs(dy) > abs(dx): return 2 if dy > 0 else 1
        else: return 3 if dx > 0 else 4

    def select_action(self, obs: Dict[str, Any], deterministic: bool = True) -> int:
        spatial = obs['spatial']
        drone = obs['drone']
        
        payload = drone[1]
        margin = drone[2]
        nx = drone[4]
        ny = drone[5]
        
        ch_fire = spatial[0]
        ch_base = spatial[4]
        center = 5
        
        # 1. Return to Base logic
        if margin < 0.1 or payload == 0.0:
            base_idx = np.argwhere(ch_base == 1.0)
            if len(base_idx) > 0:
                return self._move_towards(base_idx[0][0], base_idx[0][1], center, center)
            else:
                return 4 if nx > ny else 1 # Move towards top-left
                
        # 2. Deploy if near active fire
        fire_mask = (ch_fire == 0.25) | (ch_fire == 0.50) | (ch_fire == 0.75)
        
        # If we are literally on fire, OR if fire is adjacent, just drop to test mechanics!
        # Check 3x3 area around center for fire
        local_fire = fire_mask[center-1:center+2, center-1:center+2]
        if np.any(local_fire):
            h = int(np.sum(spatial) * 1000)
            return 5 if (h % 2 == 0) else 6
            
        # 3. Fire Targeting
        fire_idx = np.argwhere(fire_mask)
        if len(fire_idx) > 0:
            distances = np.abs(fire_idx[:, 0] - center) + np.abs(fire_idx[:, 1] - center)
            target = fire_idx[np.argmin(distances)]
            return self._move_towards(target[0], target[1], center, center)
            
        # 4. Fallback Exploration
        return 3 if (int(nx * 100) + int(ny * 100)) % 2 == 0 else 2
