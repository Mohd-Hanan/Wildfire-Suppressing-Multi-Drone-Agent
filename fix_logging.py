import re

with open("stage_testing.py", "r") as f:
    code = f.read()

# Remove the 'First 20 steps' print at the end
code = re.sub(r'            if episode <= 3:\n                print\(f"Drone \{drone_id\} First 20 steps.*?history\[drone_id\]\}\"\)', '', code, flags=re.DOTALL)
# The variable is actually 'if (ep + 1) <= 3:' now
code = re.sub(r'            if \(ep \+ 1\) <= 3:\n                print\(f"Drone \{drone_id\} First 20 steps.*?history\[drone_id\]\}\"\)', '', code, flags=re.DOTALL)

# Add step-by-step logging inside the while loop
# We'll put it right after actions = controller.get_actions()
logging_code = """            actions = controller.get_actions()
            
            step_log = f"Step {step_count:4d} | "
            for d in world.drones:
                act = actions.get(d.id, 0)
                act_name = ['STAY', 'N', 'S', 'E', 'W', 'WATER', 'RETARD'][act]
                step_log += f"D{d.id} ({d.x:2d},{d.y:2d}) Bat:{int(d.battery):3d} {act_name:6s} | "
            print(step_log)
"""

code = code.replace("            actions = controller.get_actions()", logging_code)

with open("stage_testing.py", "w") as f:
    f.write(code)
