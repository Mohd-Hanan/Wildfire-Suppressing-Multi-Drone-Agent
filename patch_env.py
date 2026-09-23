with open("stage7_36_train_eval.py", "r") as f:
    text = f.read()

text = text.replace('env = WildfireEnv("configs/environment.yaml", render_mode=None)', 'env = WildfireEnv("configs/environment.yaml")')
text = text.replace('env = WildfireEnv("configs/environment.yaml", render_mode="human")', 'env = WildfireEnv("configs/environment.yaml")')

with open("stage7_36_train_eval.py", "w") as f:
    f.write(text)
