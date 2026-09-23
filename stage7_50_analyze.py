import csv
import numpy as np

records = []
with open("stage7_50_trajectory_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append({k: (1.0 if v == 'True' else (0.0 if v == 'False' else float(v))) for k, v in row.items()})

print(f"Total steps: {len(records)}")

# Diagnostic 5: Action by Fire Direction
dirs = {"NORTH": [], "SOUTH": [], "EAST": [], "WEST": [], "NEAR": []}
for r in records:
    dx, dy = r['fire_dx'], r['fire_dy']
    fdist = r['fire_distance']
    if fdist <= 1.5:
        dirs["NEAR"].append(r)
    else:
        if abs(dx) > abs(dy):
            if dx > 0: dirs["EAST"].append(r)
            else: dirs["WEST"].append(r)
        else:
            if dy > 0: dirs["SOUTH"].append(r)
            else: dirs["NORTH"].append(r)

print("\n--- DIAGNOSTIC 5: ACTION DISTRIBUTION BY FIRE DIRECTION ---")
for d, rs in dirs.items():
    if not rs: continue
    acts = [r['action'] for r in rs]
    counts = {i: acts.count(i) for i in range(6)}
    tot = len(acts)
    print(f"{d} ({tot}): STAY {counts[0]/tot*100:.1f}%, NORTH {counts[1]/tot*100:.1f}%, SOUTH {counts[2]/tot*100:.1f}%, EAST {counts[3]/tot*100:.1f}%, WEST {counts[4]/tot*100:.1f}%, WATER {counts[5]/tot*100:.1f}%")

# Diagnostic 6: Action by Fire Distance
bins = {"0-1": [], "2-5": [], "6-10": [], "11-20": [], "21-30": [], ">30": []}
for r in records:
    fdist = r['fire_distance']
    if fdist <= 1.5: bins["0-1"].append(r)
    elif fdist <= 5.5: bins["2-5"].append(r)
    elif fdist <= 10.5: bins["6-10"].append(r)
    elif fdist <= 20.5: bins["11-20"].append(r)
    elif fdist <= 30.5: bins["21-30"].append(r)
    else: bins[">30"].append(r)

print("\n--- DIAGNOSTIC 6: ACTION BY FIRE DISTANCE ---")
for b, rs in bins.items():
    if not rs: continue
    acts = [r['action'] for r in rs]
    counts = {i: acts.count(i) for i in range(6)}
    tot = len(acts)
    print(f"Dist {b} ({tot}): WATER {counts[5]/tot*100:.1f}% | N {counts[1]/tot*100:.1f}%, S {counts[2]/tot*100:.1f}%, E {counts[3]/tot*100:.1f}%, W {counts[4]/tot*100:.1f}%")

# Diagnostic 9: Trajectory Progress
print("\n--- DIAGNOSTIC 9: TRAJECTORY PROGRESS ---")
closer = farther = same = 0
movement_deltas = {1: [], 2: [], 3: [], 4: []}
for i in range(1, len(records)):
    if records[i]['seed'] == records[i-1]['seed']:
        delta = records[i]['fire_distance'] - records[i-1]['fire_distance']
        if delta < -0.1: closer += 1
        elif delta > 0.1: farther += 1
        else: same += 1
        
        act = records[i-1]['action']
        if act in [1, 2, 3, 4]:
            movement_deltas[act].append(delta)

tot_trans = closer + farther + same
print(f"Steps closer: {closer/tot_trans*100:.1f}%")
print(f"Steps farther: {farther/tot_trans*100:.1f}%")
print(f"Steps same: {same/tot_trans*100:.1f}%")
names = {1: "NORTH", 2: "SOUTH", 3: "EAST", 4: "WEST"}
for act in [1,2,3,4]:
    print(f"Mean distance change for {names[act]}: {np.mean(movement_deltas[act]):.2f}")

# Diagnostic 10: Window vs Vector
in_win = sum(r['fire_in_window'] for r in records)
out_win = len(records) - in_win
print(f"\n--- DIAGNOSTIC 10: FIRE IN SPATIAL WINDOW ---")
print(f"Inside 11x11 window: {in_win/len(records)*100:.1f}%")
print(f"Outside 11x11 window: {out_win/len(records)*100:.1f}%")

# Diagnostic 12: Action Collapse
print("\n--- DIAGNOSTIC 12: ACTION COLLAPSE ---")
acts = [r['action'] for r in records]
counts = {i: acts.count(i) for i in range(6)}
tot = len(acts)
print(f"STAY {counts[0]/tot*100:.1f}%")
print(f"NORTH {counts[1]/tot*100:.1f}%")
print(f"SOUTH {counts[2]/tot*100:.1f}%")
print(f"EAST {counts[3]/tot*100:.1f}%")
print(f"WEST {counts[4]/tot*100:.1f}%")
print(f"WATER {counts[5]/tot*100:.1f}%")

# Diagnostic 13: Normalization ranges
fdx = [r['fire_dx'] for r in records]
fdy = [r['fire_dy'] for r in records]
fdist_norm = [r['fire_distance']/np.sqrt(48**2+48**2) for r in records]
print("\n--- DIAGNOSTIC 13: NORMALIZATION RANGES ---")
print(f"fire_dx: min={min(fdx):.3f}, max={max(fdx):.3f}")
print(f"fire_dy: min={min(fdy):.3f}, max={max(fdy):.3f}")
print(f"fire_distance (norm): min={min(fdist_norm):.3f}, max={max(fdist_norm):.3f}")
