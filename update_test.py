with open("tests/test_water_radius.py", "r") as f:
    content = f.read()

content = content.replace("self.env.step_count = 0", "self.env.step_count = 0\n        self.env.step(0)")

with open("tests/test_water_radius.py", "w") as f:
    f.write(content)
