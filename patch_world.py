import re

with open("src/wildfire/simulation/world.py", "r") as f:
    content = f.read()

replacement = """    def distance_to_base(self, drone: Drone) -> int:
        \"\"\"Calculate minimum Manhattan distance to any valid base cell (2x2 footprint).\"\"\"
        base_cells = [
            (self.base_x, self.base_y),
            (self.base_x + 1, self.base_y),
            (self.base_x, self.base_y + 1),
            (self.base_x + 1, self.base_y + 1)
        ]
        return min(abs(drone.x - bx) + abs(drone.y - by) for bx, by in base_cells)

    def minimum_return_battery(self, drone: Drone) -> int:
        \"\"\"Calculate minimum battery required to return.\"\"\"
        return self.distance_to_base(drone) * drone.move_cost

    def required_battery(self, drone: Drone) -> int:
        \"\"\"Calculate total battery required including safety reserve.\"\"\"
        battery_reserve = self.config['return_to_base'].get('battery_reserve', 5)
        return self.minimum_return_battery(drone) + battery_reserve

    def battery_margin(self, drone: Drone) -> int:
        \"\"\"Calculate how much battery remains above the safe return threshold.\"\"\"
        return drone.battery - self.required_battery(drone)

    def can_safely_return_to_base(self, drone: Drone) -> bool:
        \"\"\"Check if the drone has enough battery to safely return.\"\"\"
        return self.battery_margin(drone) >= 0"""

content = re.sub(
    r'    def distance_to_base\(self, drone: Drone\) -> int:.*?return drone\.battery >= safe_return_battery',
    replacement,
    content,
    flags=re.DOTALL
)

with open("src/wildfire/simulation/world.py", "w") as f:
    f.write(content)
