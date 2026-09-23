import re

with open("stage7_36_train_eval.py", "r") as f:
    text = f.read()

text = re.sub(
    r'    config = load_config\("configs/environment.yaml"\).*?env = WildfireEnv\(config, render_mode=None\)',
    '    env = WildfireEnv("configs/environment.yaml", render_mode=None)',
    text,
    flags=re.DOTALL
)

text = re.sub(
    r'    config = load_config\("configs/environment.yaml"\).*?env = WildfireEnv\(config, render_mode="human"\)',
    '    env = WildfireEnv("configs/environment.yaml", render_mode="human")',
    text,
    flags=re.DOTALL
)

with open("stage7_36_train_eval.py", "w") as f:
    f.write(text)
