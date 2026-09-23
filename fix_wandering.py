import re

with open("src/wildfire/agents/demo_controller.py", "r") as f:
    code = f.read()

# Replace the wandering logic
code = re.sub(
    r'                # Try moving closer to base if nothing else works\n                if base_target:\n                    # just randomly add all moves if we get stuck\n                    possible_moves\.extend\(\[\(1, \(0, -1\)\), \(2, \(0, 1\)\), \(3, \(1, 0\)\), \(4, \(-1, 0\)\)\]\)',
    '                # If returning to base and blocked, just STAY to save battery and wait',
    code
)

with open("src/wildfire/agents/demo_controller.py", "w") as f:
    f.write(code)
