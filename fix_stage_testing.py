import re

with open("stage_testing.py", "r") as f:
    code = f.read()

replacement1 = """
        collisions = 0
        action_counts = {0:{}, 1:{}, 2:{}, 3:{}}
        unique_targets = {0:set(), 1:set(), 2:set(), 3:set()}
        target_changes = {0:0, 1:0, 2:0, 3:0}
        prev_targets = {0:None, 1:None, 2:None, 3:None}
        history = {0:[], 1:[], 2:[], 3:[]}
"""
code = re.sub(
    r'        collisions = 0.*?        unique_targets = \{0:set\(\), 1:set\(\), 2:set\(\), 3:set\(\)\}',
    replacement1[1:],
    code,
    flags=re.DOTALL
)

replacement2 = """
                act = actions.get(drone.id, 0)
                action_counts[drone.id][act] = action_counts[drone.id].get(act, 0) + 1
                if step_count <= 20:
                    history[drone.id].append((drone.x, drone.y, act))
                if hasattr(drone, "current_target"):
                    curr = drone.current_target
                    unique_targets[drone.id].add(curr)
                    if prev_targets[drone.id] is not None and prev_targets[drone.id] != curr:
                        target_changes[drone.id] += 1
                    prev_targets[drone.id] = curr
"""
code = re.sub(
    r'                act = actions\.get\(drone\.id, 0\).*?                    unique_targets\[drone\.id\]\.add\(drone\.current_target\)',
    replacement2[1:],
    code,
    flags=re.DOTALL
)

replacement3 = """
        for drone_id in range(4):
            print(f"Drone {drone_id} Actions: {action_counts[drone_id]}")
            print(f"Drone {drone_id} Unique Targets: {len(unique_targets[drone_id])}")
            print(f"Drone {drone_id} Target Changes: {target_changes[drone_id]}")
            if episode <= 3:
                print(f"Drone {drone_id} First 20 steps (x, y, act): {history[drone_id]}")
"""
code = re.sub(
    r'        for drone_id in range\(4\):.*?            print\(f"Drone \{drone_id\} Unique Targets: \{len\(unique_targets\[drone_id\]\)\}"\)',
    replacement3[1:],
    code,
    flags=re.DOTALL
)

code = code.replace("default=2", "default=10")

with open("stage_testing.py", "w") as f:
    f.write(code)
