import os
import re

# 1. test_rollout.py
with open('tests/test_rollout.py', 'r') as f:
    text = f.read()
text = text.replace('(2, 6)', '(2, 9)')
with open('tests/test_rollout.py', 'w') as f:
    f.write(text)

# 2. test_wildfire_env.py
with open('tests/test_wildfire_env.py', 'r') as f:
    text = f.read()
text = text.replace('(6,)', '(9,)')
with open('tests/test_wildfire_env.py', 'w') as f:
    f.write(text)

# 3. test_reward.py
with open('tests/test_reward.py', 'r') as f:
    text = f.read()
text = text.replace("-5.0", "-0.1")
text = text.replace("-25.0", "-0.5")
text = text.replace("100.0", "50.0")
with open('tests/test_reward.py', 'w') as f:
    f.write(text)

# 4. test_action.py
with open('tests/test_action.py', 'r') as f:
    text = f.read()
text = re.sub(r'self\.executor\.execute\((.*?)\)', r'self.executor.execute(\1)[0]', text)
with open('tests/test_action.py', 'w') as f:
    f.write(text)

print("Tests updated.")
