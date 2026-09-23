import re

with open("test_battery_safety.py", "r") as f:
    content = f.read()

replacement = """    def setUp(self):
        self.config = DummyConfig()
        self.world = World(48, 48, seed=42)
        # Manually patch config for test purposes
        self.world.config = self.config
        self.world.base_x = 2
        self.world.base_y = 2"""

content = re.sub(
    r'    def setUp\(self\):\n        self\.config = DummyConfig\(\)\n        self\.world = World\(48, 48, self\.config\)\n        self\.world\.base_x = 2\n        self\.world\.base_y = 2',
    replacement,
    content,
    flags=re.DOTALL
)

with open("test_battery_safety.py", "w") as f:
    f.write(content)
