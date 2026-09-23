import re

with open("stage7_35_train_eval.py", "r") as f:
    content = f.read()

replacement = """    def get_action_and_value(self, obs, action=None):
        obs_t = {k: torch.tensor(v, dtype=torch.float32, device=self.device) if not isinstance(v, torch.Tensor) else v for k,v in obs.items()}
        if obs_t["spatial"].dim() == 3: # Handle unbatched
            obs_t = {k: v.unsqueeze(0) for k,v in obs_t.items()}
            
        action_logits, state_value = self.forward(obs_t)
        probs = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), state_value"""

content = re.sub(
    r'    def get_action_and_value\(self, obs, action=None\):.*?return action, probs\.log_prob\(action\), probs\.entropy\(\), state_value',
    replacement,
    content,
    flags=re.DOTALL
)

with open("stage7_35_train_eval.py", "w") as f:
    f.write(content)
