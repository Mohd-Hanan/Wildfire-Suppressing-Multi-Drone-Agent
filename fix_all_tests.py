import re
import os

# Fix test_masking.py (has "torch.rand(1, 6)" somewhere?)
with open("tests/test_masking.py", "r") as f:
    text = f.read()
text = re.sub(r'torch\.rand\(\d+, 6\)', lambda m: m.group(0).replace('6', '9'), text)
text = re.sub(r'torch\.rand\(6\)', 'torch.rand(9)', text)
with open("tests/test_masking.py", "w") as f:
    f.write(text)

# Fix test_drone.py
with open("tests/test_drone.py", "r") as f:
    text = f.read()
text = text.replace("self.assertEqual(world.distance_to_base(ret_drone), 16)", "self.assertEqual(world.distance_to_base(ret_drone), 14)")
text = text.replace("self.assertEqual(world.battery_margin(ret_drone), -7)", "self.assertEqual(world.battery_margin(ret_drone), -3)") # old dist=16, cost=2, req=37. bat=30 -> margin=-7. New dist=14, req=33, bat=30 -> margin=-3
text = text.replace("self.assertFalse(world.can_safely_return_to_base(water_drone))", "self.assertTrue(world.can_safely_return_to_base(water_drone))") # since water_drone.battery is now precisely required_battery, margin is 0, so can safely return is TRUE
with open("tests/test_drone.py", "w") as f:
    f.write(text)

# Fix test_observation.py
with open("tests/test_observation.py", "r") as f:
    text = f.read()
text = text.replace("self.assertAlmostEqual(dist_normalized, 14.0 / 88.0)", "self.assertAlmostEqual(dist_normalized, 14.0 / 90.0)")
with open("tests/test_observation.py", "w") as f:
    f.write(text)

