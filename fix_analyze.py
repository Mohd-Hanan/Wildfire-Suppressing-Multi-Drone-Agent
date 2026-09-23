with open("stage7_50_analyze.py", "r") as f:
    content = f.read()
content = content.replace("records.append({k: float(v) for k, v in row.items()})", "records.append({k: (1.0 if v == 'True' else (0.0 if v == 'False' else float(v))) for k, v in row.items()})")
with open("stage7_50_analyze.py", "w") as f:
    f.write(content)
