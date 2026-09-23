import re

with open("src/wildfire/rendering/renderer.py", "r") as f:
    code = f.read()

# 1. Increase hud_width
code = code.replace("self.hud_width = 320", "self.hud_width = 380")

# 2. Adjust vertical spacing slightly
code = code.replace("y += 10", "y += 4")
code = code.replace("y += 5", "y += 2")
code = code.replace("y += font.get_height() + 2", "y += font.get_height()")
code = code.replace("y += self.font_mono.get_height() + 2", "y += self.font_mono.get_height()")

# 3. Remove COORDINATION
replacement = """
"""
code = re.sub(
    r'        y \+= 15.*?        render_text\(f"  Live Reassessment", self\.font_body, \(100, 255, 100\)\)',
    replacement,
    code,
    flags=re.DOTALL
)

with open("src/wildfire/rendering/renderer.py", "w") as f:
    f.write(code)
