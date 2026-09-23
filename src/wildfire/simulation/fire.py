import numpy as np

class FireState:
    UNBURNED = 0
    IGNITING = 1
    BURNING = 2
    SMOLDERING = 3
    BURNED = 4

class FireManager:
    def __init__(self, width: int, height: int, rng: np.random.Generator = None, config: dict = None):
        config = config or {}
        self.base_spread_rate = config.get('base_spread_rate', 0.08)
        self.base_burning_duration = config.get('base_burning_duration', 20)
        self.width = width
        self.height = height
        self.rng = rng if rng is not None else np.random.default_rng()
        
        self.fire_map = np.full((width, height), FireState.UNBURNED, dtype=np.int8)
        self.burn_timers = np.zeros((width, height), dtype=np.int16)
        
    def ignite(self, x, y):
        """Forces a specific cell to ignite immediately."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.fire_map[x, y] = FireState.BURNING
            self.burn_timers[x, y] = self.base_burning_duration # Stays burning for 20 ticks
            
    def step(self, terrain, wind):
        """Progresses the cellular automata simulation by one tick."""
        
        # 1. State Transitions (Lifecycle of the fire)
        # IGNITING -> BURNING
        igniting = (self.fire_map == FireState.IGNITING)
        self.fire_map[igniting] = FireState.BURNING
        self.burn_timers[igniting] = self.base_burning_duration # Base burn time
        
        # BURNING -> SMOLDERING
        burning = (self.fire_map == FireState.BURNING)
        self.burn_timers[burning] -= 1
        done_burning = burning & (self.burn_timers <= 0)
        self.fire_map[done_burning] = FireState.SMOLDERING
        self.burn_timers[done_burning] = 30 # Smolder time
        
        # SMOLDERING -> BURNED
        smoldering = (self.fire_map == FireState.SMOLDERING)
        self.burn_timers[smoldering] -= 1
        done_smoldering = smoldering & (self.burn_timers <= 0)
        self.fire_map[done_smoldering] = FireState.BURNED
        
        # 2. Fire Spread (Rothermel Physics)
        # Accumulate the probability that each unburned cell catches fire this tick
        ignition_prob = np.zeros((self.width, self.height), dtype=np.float32)
        burning_mask = (self.fire_map == FireState.BURNING)
        
        # 8 adjacent neighbors: (dx, dy)
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        
        for dx, dy in directions:
            # Shift the burning mask to see which cells have a burning neighbor at (-dx, -dy)
            shifted_burning = np.roll(np.roll(burning_mask, dx, axis=0), dy, axis=1)
            
            # Mask out the wrap-around artifacts
            if dx == 1: shifted_burning[0, :] = False
            elif dx == -1: shifted_burning[-1, :] = False
            if dy == 1: shifted_burning[:, 0] = False
            elif dy == -1: shifted_burning[:, -1] = False
            
            # We only care about spreading TO cells that are currently UNBURNED
            target_mask = shifted_burning & (self.fire_map == FireState.UNBURNED)
            
            if not np.any(target_mask):
                continue
                
            # A) Fuel and Moisture Factor
            fuel = terrain.fuel[target_mask]
            moisture = terrain.moisture[target_mask]
            base_p = self.base_spread_rate * (fuel + 0.1) * (1.0 - moisture * 0.7)
            
            # B) Slope Factor (Fire travels much faster uphill)
            # Get the elevation of the burning neighbor that is spreading the fire
            neighbor_elev = np.roll(np.roll(terrain.elevation, dx, axis=0), dy, axis=1)[target_mask]
            target_elev = terrain.elevation[target_mask]
            elev_diff = target_elev - neighbor_elev
            slope_factor = np.exp(elev_diff * 12.0)
            
            # C) Wind Factor
            dist = np.sqrt(dx**2 + dy**2)
            dir_x = dx / dist
            dir_y = dy / dist
            
            wind_dot = (wind.u * dir_x) + (wind.v * dir_y)
            wind_factor = np.exp(wind_dot * 3.5) 
            
            dist_factor = 1.0 / dist
            
            # Final probability of ignition
            prob = base_p * slope_factor * wind_factor * dist_factor
            ignition_prob[target_mask] += prob
            
        # 3. Apply Ignition
        rolls = self.rng.uniform(0, 1, (self.width, self.height))
        ignite_mask = (self.fire_map == FireState.UNBURNED) & (rolls < ignition_prob)
        self.fire_map[ignite_mask] = FireState.IGNITING

