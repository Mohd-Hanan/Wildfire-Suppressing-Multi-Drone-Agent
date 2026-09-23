with open("stage7_36_train_eval.py", "r") as f:
    text = f.read()

import re
text = re.sub(
    r'    trainer = PPOTrainer\(.*?minibatch_size=64\n    \)',
    '    trainer = PPOTrainer(policy, device=device, learning_rate=3e-4, ppo_epochs=10, minibatch_size=64)',
    text,
    flags=re.DOTALL
)

with open("stage7_36_train_eval.py", "w") as f:
    f.write(text)
