from typing import Dict, Any, Optional
from wildfire.agents.base import Agent

class NoSuppressionAgent(Agent):
    """
    A control baseline that never deploys suppression resources.
    It simply stays in place (Action 0) to serve as a pure environmental baseline.
    """
    def reset(self, seed: Optional[int] = None):
        pass

    def select_action(self, obs: Dict[str, Any], deterministic: bool = True) -> int:
        return 0 # Stay
