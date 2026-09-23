import torch
from torch.distributions.categorical import Categorical

logits = torch.tensor([[1.0, 2.0, -1e9]])
dist = Categorical(logits=logits)

print("Probs:", dist.probs)
print("Logits:", dist.logits)
print("Entropy:", dist.entropy())

action = torch.tensor([2])
print("Log Prob of 2:", dist.log_prob(action))

action = torch.tensor([1])
print("Log Prob of 1:", dist.log_prob(action))
