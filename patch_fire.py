import re

with open('src/wildfire/simulation/fire.py', 'r') as f:
    content = f.read()

content = content.replace(
"""    def __init__(self, width: int, height: int, rng: np.random.Generator = None):
        self.base_spread_rate = 0.08
        self.base_burning_duration = 20""",
"""    def __init__(self, width: int, height: int, rng: np.random.Generator = None, config: dict = None):
        config = config or {}
        self.base_spread_rate = config.get('base_spread_rate', 0.08)
        self.base_burning_duration = config.get('base_burning_duration', 20)"""
)

with open('src/wildfire/simulation/fire.py', 'w') as f:
    f.write(content)
