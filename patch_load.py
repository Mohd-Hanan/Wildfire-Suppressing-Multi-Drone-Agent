import re

with open("stage7_36_train_eval.py", "r") as f:
    text = f.read()

text = text.replace(
    "from stage7_35_final_truth import SeparateActorCriticSymmetric64, load_config",
    "from stage7_35_final_truth import SeparateActorCriticSymmetric64\nimport yaml\ndef load_config(path):\n    with open(path, 'r') as f:\n        return yaml.safe_load(f)"
)

with open("stage7_36_train_eval.py", "w") as f:
    f.write(text)
