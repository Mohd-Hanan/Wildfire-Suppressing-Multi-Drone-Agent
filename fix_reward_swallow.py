import os

with open('src/wildfire/environment/reward.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "'boundary_penalty': 0.0," in line:
        new_lines.append("                'boundary_penalty': -self.boundary_hit_penalty if hit_boundary else 0.0,\n")
    else:
        new_lines.append(line)

with open('src/wildfire/environment/reward.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed reward swallow.")
