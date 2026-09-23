import os

with open('src/wildfire/rl/policy.py', 'r') as f:
    text = f.read()
text = text.replace('non_spatial_features = 9', 'non_spatial_features = 12')
with open('src/wildfire/rl/policy.py', 'w') as f:
    f.write(text)

with open('stage7_15_smoke_test.py', 'r') as f:
    text2 = f.read()
text2 = text2.replace('32*11*11 + 9', '32*11*11 + 12')
with open('stage7_15_smoke_test.py', 'w') as f:
    f.write(text2)

print("Policy shapes updated.")
