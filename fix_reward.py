with open('src/wildfire/environment/reward.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "if last_action == 5:" in line:
        # replace the next block
        pass
