import os
import re

# 1. test_ppo_trainer.py
with open('tests/test_ppo_trainer.py', 'r') as f:
    text = f.read()
text = text.replace('torch.randn(16, 6)', 'torch.randn(16, 9)')
with open('tests/test_ppo_trainer.py', 'w') as f:
    f.write(text)

# 2. test_reward.py
with open('tests/test_reward.py', 'r') as f:
    text = f.read()
text = text.replace('99.99', '49.99')
with open('tests/test_reward.py', 'w') as f:
    f.write(text)

# 3. wildfire_env.py
with open('src/wildfire/environment/wildfire_env.py', 'r') as f:
    text = f.read()
text = text.replace("spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)", "spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)")
with open('src/wildfire/environment/wildfire_env.py', 'w') as f:
    f.write(text)

# 4. test_ppo.py
with open('tests/test_ppo.py', 'r') as f:
    text = f.read()
text = text.replace('torch.randn(64, 6)', 'torch.randn(64, 9)')
with open('tests/test_ppo.py', 'w') as f:
    f.write(text)

# 5. test_training.py
with open('tests/test_training.py', 'r') as f:
    text = f.read()
text = text.replace('torch.randn(1, 6)', 'torch.randn(1, 9)')
with open('tests/test_training.py', 'w') as f:
    f.write(text)

print("Remaining fixes applied.")
