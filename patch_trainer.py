import re

with open("stage7_35_train_eval.py", "r") as f:
    content = f.read()

replacement = """class SymmetricTrainer(SmokeTestTrainer):
    def update(self, spatial, drone, wind, acts, old_log_probs, advantages, returns):
        # We need to rewrite the actor and critic parameter lists to match the new architecture
        actor_params = (list(self.policy.actor_cnn.parameters()) + 
                        list(self.policy.actor_vec.parameters()) + 
                        list(self.policy.actor_fusion.parameters()) + 
                        list(self.policy.actor_head.parameters()))
        critic_params = (list(self.policy.critic_cnn.parameters()) + 
                         list(self.policy.critic_vec.parameters()) + 
                         list(self.policy.critic_fusion.parameters()) + 
                         list(self.policy.critic_head.parameters()))
        
        # Manually construct optimizers since the base init created them with wrong params
        if not hasattr(self, 'fixed_optimizers'):
            self.actor_optimizer = torch.optim.Adam(actor_params, lr=3e-4)
            self.critic_optimizer = torch.optim.Adam(critic_params, lr=1e-3)
            self.fixed_optimizers = True
            
        return super().update(spatial, drone, wind, acts, old_log_probs, advantages, returns)

def train():"""

content = re.sub(
    r'def train\(\):',
    replacement,
    content
)
content = content.replace("trainer = SmokeTestTrainer(policy=policy, device=device)", "trainer = SymmetricTrainer(policy=policy, device=device)")

with open("stage7_35_train_eval.py", "w") as f:
    f.write(content)
