with open("stage7_37_feasibility.py", "r") as f:
    text = f.read()

import re
text = text.replace("stats['suppressed'].append((env.world.fire_manager.fire_map == -1).sum())", "")
text = text.replace("_, _, terminated, truncated, _ = env.step(action)", "_, _, terminated, truncated, info = env.step(action)\n            ep_suppressed += info.get('newly_suppressed', 0)")
text = text.replace("done = False\n        step = 0", "done = False\n        step = 0\n        ep_suppressed = 0")
text = text.replace("stats['burned'].append((env.world.fire_manager.fire_map > 0).sum())", "stats['burned'].append((env.world.fire_manager.fire_map > 0).sum())\n        stats['suppressed'].append(ep_suppressed)")

with open("stage7_37_feasibility.py", "w") as f:
    f.write(text)
