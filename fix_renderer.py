import re

with open("src/wildfire/rendering/renderer.py", "r") as f:
    code = f.read()

replacement = """
        for drone in world.drones:
            if drone.active:
                target_str = getattr(drone, 'current_target', 'None')
                action_id = getattr(drone, 'current_action', 0)
                action_str = ['STAY', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'WATER', 'RETARDANT'][action_id]
                stats.append(f"  {drone.type.name[:3]}-{drone.id}: Bat:{drone.battery} Pay:{drone.payload} Target:{target_str} Act:{action_str}")
            else:
                stats.append(f"  {drone.type.name[:3]}-{drone.id}: CRASHED")
"""

# Now we strictly replace just that small loop in `_draw_hud`
old_code = """        for drone in world.drones:
            if drone.active:
                # Removed missing emojis, replaced with text
                stats.append(f"  {drone.type.name[:3]}-{drone.id}: Bat:{drone.battery} Pay:{drone.payload}")
            else:
                stats.append(f"  {drone.type.name[:3]}-{drone.id}: CRASHED")"""

code = code.replace(old_code, replacement[1:])
code = code.replace('"COMMAND CENTER"', '"TESTING MODE"')

with open("src/wildfire/rendering/renderer.py", "w") as f:
    f.write(code)
