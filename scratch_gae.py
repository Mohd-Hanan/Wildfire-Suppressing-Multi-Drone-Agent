import torch
from wildfire.rl.gae import compute_gae

rewards = torch.tensor([1.0, -0.5, 2.0, 10.0])
values = torch.tensor([0.5, 0.2, 1.5, 8.0])
terminated = torch.tensor([False, False, False, True])
last_value = 5.0 # Shouldn't be used since the last step is a terminal state

adv, ret = compute_gae(rewards, values, terminated, last_value)

print("Rewards:", rewards.tolist())
print("Values:", values.tolist())
print("Terminated:", terminated.tolist())
print("Last Value:", last_value)
print("Advantages:", adv.tolist())
print("Returns:", ret.tolist())
