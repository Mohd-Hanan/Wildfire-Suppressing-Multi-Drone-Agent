import re

with open("src/wildfire/environment/reward.py", "r") as f:
    content = f.read()

replacement1 = """        # 5. Crash penalty & Battery Safety Penalty
        crash_penalty = 0.0
        battery_safety_penalty = 0.0
        for d in drones:
            if self.prev_drone_active.get(d.id, True) and not d.active:
                # Transitioned to inactive this step
                crash_penalty -= self.drone_crash_penalty
            self.prev_drone_active[d.id] = d.active
            
            # Progressive battery safety penalty
            if d.active:
                margin = world.battery_margin(d)
                if margin < 0:
                    battery_safety_penalty -= 0.5 * abs(margin)"""

content = re.sub(
    r'        # 5\. Crash penalty\n        crash_penalty = 0\.0\n        for d in drones:\n            if self\.prev_drone_active\.get\(d\.id, True\) and not d\.active:\n                # Transitioned to inactive this step\n                crash_penalty -= self\.drone_crash_penalty\n            self\.prev_drone_active\[d\.id\] = d\.active',
    replacement1,
    content,
    flags=re.DOTALL
)

replacement2 = """        # Combine
        total_reward = (damage_penalty + 
                        suppression_reward + 
                        containment_progress + 
                        step_penalty + 
                        boundary_penalty +
                        crash_penalty + 
                        battery_safety_penalty +
                        extinction_reward)

        # Update tracking variables
        self.prev_affected_cells = current_affected
        self.prev_active_fire_count = current_active_fire

        info = {
            'new_burned_cells': new_burned_cells,
            'damage_penalty': damage_penalty,
            'newly_suppressed_cells': newly_suppressed_cells,
            'suppression_reward': suppression_reward,
            'containment_progress': containment_progress,
            'step_penalty': step_penalty,
            'boundary_penalty': boundary_penalty,
            'crash_penalty': crash_penalty,
            'battery_safety_penalty': battery_safety_penalty,
            'extinction_reward': extinction_reward,
            'total_reward': total_reward
        }"""

content = re.sub(
    r'        # Combine\n        total_reward = \(damage_penalty \+ \n                        suppression_reward \+ \n                        containment_progress \+ \n                        step_penalty \+ \n                        boundary_penalty \+\n                        crash_penalty \+ \n                        extinction_reward\)\n\n        # Update tracking variables\n        self\.prev_affected_cells = current_affected\n        self\.prev_active_fire_count = current_active_fire\n\n        info = {\n            \'new_burned_cells\': new_burned_cells,\n            \'damage_penalty\': damage_penalty,\n            \'newly_suppressed_cells\': newly_suppressed_cells,\n            \'suppression_reward\': suppression_reward,\n            \'containment_progress\': containment_progress,\n            \'step_penalty\': step_penalty,\n            \'boundary_penalty\': boundary_penalty,\n            \'crash_penalty\': crash_penalty,\n            \'extinction_reward\': extinction_reward,\n            \'total_reward\': total_reward\n        }',
    replacement2,
    content,
    flags=re.DOTALL
)

with open("src/wildfire/environment/reward.py", "w") as f:
    f.write(content)
