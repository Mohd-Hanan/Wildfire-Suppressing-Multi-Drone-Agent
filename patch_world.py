import re

with open('src/wildfire/simulation/world.py', 'r') as f:
    content = f.read()

# Load config earlier
content = content.replace(
"""        self.fire_manager = FireManager(width, height, self.terrain.rng)
        start_x = self.terrain.rng.integers(0, width)
        start_y = self.terrain.rng.integers(0, height)
        self.fire_manager.ignite(start_x, start_y)
        
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)""",
"""        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        fire_config = self.config.get('fire', {})
        self.fire_manager = FireManager(width, height, self.terrain.rng, fire_config)
        start_x = self.terrain.rng.integers(0, width)
        start_y = self.terrain.rng.integers(0, height)
        self.fire_manager.ignite(start_x, start_y)"""
)

with open('src/wildfire/simulation/world.py', 'w') as f:
    f.write(content)
