with open("stage_demo.py", "r") as f:
    code = f.read()

code = code.replace(
    'if drone.battery <= 0 and not world.is_at_base(drone):',
    'if drone.battery <= 0 and not world.is_at_base(drone):\n                    print(f"CRASH: Drone {drone.id} at ({drone.x},{drone.y}) battery={drone.battery}, payload={drone.payload}")'
)

with open("stage_demo.py", "w") as f:
    f.write(code)
