import re

with open('stage7_45_behavior_diagnostic.py', 'r') as f:
    content = f.read()

content = content.replace(
"""        with torch.no_grad():
            _, _, _, probs = policy.get_action_and_value(obs)
            probs = probs.squeeze(0).numpy()""",
"""        with torch.no_grad():
            logits, _ = policy.forward(obs)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()"""
)

with open('stage7_45_behavior_diagnostic.py', 'w') as f:
    f.write(content)
