with open("stage7_50_diagnostics.py", "r") as f:
    content = f.read()

content = content.replace("_, _, action_logits, _ = pol({'spatial': spatial, 'drone': d, 'wind': wind})", "action_logits, _ = pol({'spatial': spatial, 'drone': d, 'wind': wind})")

with open("stage7_50_diagnostics.py", "w") as f:
    f.write(content)
