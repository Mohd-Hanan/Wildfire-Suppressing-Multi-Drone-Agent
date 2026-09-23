import os

with open('configs/environment.yaml', 'r') as f:
    text = f.read()

# Update environment.yaml values
text = text.replace('new_burned_cell_penalty: 5.0', 'new_burned_cell_penalty: 0.1')
text = text.replace('effective_suppression_reward: 0.0', 'effective_suppression_reward: 2.0')
text = text.replace('fire_extinguished_reward: 100.0', 'fire_extinguished_reward: 50.0')

with open('configs/environment.yaml', 'w') as f:
    f.write(text)

with open('configs/training.yaml', 'r') as f:
    text2 = f.read()
text2 = text2.replace('drone_dim: 6', 'drone_dim: 9')
with open('configs/training.yaml', 'w') as f:
    f.write(text2)

with open('configs/evaluation.yaml', 'r') as f:
    text3 = f.read()
text3 = text3.replace('drone_dim: 6', 'drone_dim: 9')
with open('configs/evaluation.yaml', 'w') as f:
    f.write(text3)

print("Configs updated.")
