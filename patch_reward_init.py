import re

with open("src/wildfire/environment/reward.py", "r") as f:
    content = f.read()

replacement = """            return 0.0, {
                'new_burned_cells': 0, 'damage_penalty': 0.0,
                'newly_suppressed_cells': 0, 'suppression_reward': 0.0,
                'containment_progress': 0.0, 'step_penalty': 0.0,
                'crash_penalty': 0.0, 'extinction_reward': 0.0,
                'boundary_penalty': -self.boundary_hit_penalty if hit_boundary else 0.0,
                'battery_safety_penalty': 0.0,
                'total_reward': 0.0
            }"""

content = re.sub(
    r'            return 0\.0, {\n                \'new_burned_cells\': 0, \'damage_penalty\': 0\.0,\n                \'newly_suppressed_cells\': 0, \'suppression_reward\': 0\.0,\n                \'containment_progress\': 0\.0, \'step_penalty\': 0\.0,\n                \'crash_penalty\': 0\.0, \'extinction_reward\': 0\.0,\n                \'boundary_penalty\': -self\.boundary_hit_penalty if hit_boundary else 0\.0,\n                \'total_reward\': 0\.0\n            }',
    replacement,
    content,
    flags=re.DOTALL
)

with open("src/wildfire/environment/reward.py", "w") as f:
    f.write(content)
