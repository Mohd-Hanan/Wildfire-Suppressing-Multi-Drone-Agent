with open("stage_demo.py", "r") as f:
    code = f.read()
    
import re
code = re.sub(
    r'                if drone.battery <= 0 and not world.is_at_base\(drone\):\n                    crashes \+= 1\n                    drone.active = False',
    r'                if drone.battery < 15 and not world.is_at_base(drone):\n                    drone.battery = drone.max_battery  # Demo Mode Auto-Refill\n                if drone.battery <= 0 and not world.is_at_base(drone):\n                    crashes += 1\n                    drone.active = False',
    code
)

with open("stage_demo.py", "w") as f:
    f.write(code)
