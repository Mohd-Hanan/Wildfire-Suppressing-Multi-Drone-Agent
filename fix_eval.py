import re

with open("stage7_52_eval.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "ep_r = 0" in line:
        new_lines.append(line)
        new_lines.append("        steps_closer = 0\n")
        new_lines.append("        steps_farther = 0\n")
        new_lines.append("        steps_same = 0\n")
        new_lines.append("        prev_dist = None\n")
        new_lines.append("        init_dist = None\n")
        continue
        
    if "current_fire_dist = np.min(dists)" in line:
        new_lines.append(line)
        new_lines.append("                if prev_dist is not None:\n")
        new_lines.append("                    if current_fire_dist < prev_dist: steps_closer += 1\n")
        new_lines.append("                    elif current_fire_dist > prev_dist: steps_farther += 1\n")
        new_lines.append("                    else: steps_same += 1\n")
        new_lines.append("                prev_dist = current_fire_dist\n")
        new_lines.append("                if init_dist is None: init_dist = current_fire_dist\n")
        continue
        
    if "ep_burned_areas.append(np.sum(fm != 0))" in line:
        new_lines.append(line)
        new_lines.append("        if not hasattr(env, 'total_closer'): env.total_closer = 0\n")
        new_lines.append("        if not hasattr(env, 'total_farther'): env.total_farther = 0\n")
        new_lines.append("        if not hasattr(env, 'total_same'): env.total_same = 0\n")
        new_lines.append("        if not hasattr(env, 'init_dists'): env.init_dists = []\n")
        new_lines.append("        if not hasattr(env, 'final_dists'): env.final_dists = []\n")
        new_lines.append("        env.total_closer += steps_closer\n")
        new_lines.append("        env.total_farther += steps_farther\n")
        new_lines.append("        env.total_same += steps_same\n")
        new_lines.append("        if init_dist is not None: env.init_dists.append(init_dist)\n")
        new_lines.append("        if prev_dist is not None: env.final_dists.append(prev_dist)\n")
        continue

    if "print(f\"Mean episode length:" in line:
        new_lines.append(line)
        new_lines.append("    tot = env.total_closer + env.total_farther + env.total_same\n")
        new_lines.append("    print(f\"Steps moving closer:      {env.total_closer/tot*100:.1f}%\")\n")
        new_lines.append("    print(f\"Steps moving farther:     {env.total_farther/tot*100:.1f}%\")\n")
        new_lines.append("    print(f\"Steps same distance:      {env.total_same/tot*100:.1f}%\")\n")
        new_lines.append("    print(f\"Initial fire distance:    {np.mean(env.init_dists):.2f}\")\n")
        new_lines.append("    print(f\"Final fire distance:      {np.mean(env.final_dists):.2f}\")\n")
        continue

    new_lines.append(line)

with open("stage7_52_eval.py", "w") as f:
    f.writelines(new_lines)

