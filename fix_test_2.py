with open("tests/test_water_radius.py", "r") as f:
    content = f.read()

content = content.replace("10+dx, 10+dy", "15+dx, 15+dy")
content = content.replace("self.assertEqual(self.drone.battery, old_battery - 1)", "self.assertEqual(self.drone.battery, old_battery - self.drone.drop_cost)")

with open("tests/test_water_radius.py", "w") as f:
    f.write(content)
