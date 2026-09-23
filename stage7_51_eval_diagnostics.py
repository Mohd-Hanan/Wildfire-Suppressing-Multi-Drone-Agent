import torch
import numpy as np
from stage7_35_final_truth import SeparateActorCriticSymmetric64

# --- Constants & Setup ---
CHECKPOINT_PATH = "stage7_51_balanced_fire_20updates.pth"

# Load policy
device = torch.device('cpu')
policy = SeparateActorCriticSymmetric64(device=device, drone_type="WATER").to(device)

try:
    policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    policy.eval()
    print("Loaded Stage 7.51 checkpoint.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit(1)

# Diagnostic A & B: Synthetic states
spatial = torch.zeros(1, 5, 11, 11, dtype=torch.float32)
wind = torch.zeros(1, 3, dtype=torch.float32)
base_drone = torch.zeros(1, 9, dtype=torch.float32)
base_drone[0, 0] = 1.0 # battery
base_drone[0, 1] = 1.0 # payload
base_drone[0, 4] = 0.5 # nx
base_drone[0, 5] = 0.5 # ny

def get_probs(fire_dx, fire_dy, fire_distance, pol=policy):
    d = base_drone.clone()
    d[0, 6] = fire_dx
    d[0, 7] = fire_dy
    d[0, 8] = fire_distance
    
    with torch.no_grad():
        action_logits, _ = pol({'spatial': spatial, 'drone': d, 'wind': wind})
    
    # Mask retardant
    action_logits[0, 6] = -1e9
    probs = torch.softmax(action_logits, dim=-1)[0].numpy()
    return action_logits[0].numpy(), probs

# Map distance=10 pixels in a 48x48 map to normalized
max_dist = np.sqrt(48**2 + 48**2)
d10 = 10 / max_dist
dx10 = 10 / 48
d1 = 1 / max_dist
dx1 = 1 / 48
d30 = 30 / max_dist
dx30 = 30 / 48

print("\n--- DIAGNOSTIC A: SYNTHETIC DIRECTION RESPONSE ---")
cases = {
    "FIRE NORTH": (0.0, -dx10, d10),
    "FIRE SOUTH": (0.0, dx10, d10),
    "FIRE EAST": (dx10, 0.0, d10),
    "FIRE WEST": (-dx10, 0.0, d10),
    "FIRE NE": (dx10, -dx10, d10*1.414),
    "FIRE NW": (-dx10, -dx10, d10*1.414),
    "FIRE SE": (dx10, dx10, d10*1.414),
    "FIRE SW": (-dx10, dx10, d10*1.414),
}

for name, (fdx, fdy, fdist) in cases.items():
    logits, probs = get_probs(fdx, fdy, fdist)
    print(f"{name}:")
    print(f"  STAY  {probs[0]*100:.1f}%")
    print(f"  NORTH {probs[1]*100:.1f}%")
    print(f"  SOUTH {probs[2]*100:.1f}%")
    print(f"  EAST  {probs[3]*100:.1f}%")
    print(f"  WEST  {probs[4]*100:.1f}%")
    print(f"  WATER {probs[5]*100:.1f}%")

print("\n--- DIAGNOSTIC C: FEATURE SENSITIVITY ---")
base_logits, base_probs = get_probs(0.0, 0.0, d10)

def measure_sensitivity(feature_idx, val_range, desc):
    max_prob_change = 0.0
    for v in val_range:
        if feature_idx == 6:
            l, p = get_probs(v, 0.0, d10)
        elif feature_idx == 7:
            l, p = get_probs(0.0, v, d10)
        elif feature_idx == 8:
            l, p = get_probs(0.0, 0.0, v)
        prob_change = np.max(np.abs(p[:6] - base_probs[:6]))
        if prob_change > max_prob_change: max_prob_change = prob_change
    print(f"{desc}: prob sensitivity={max_prob_change*100:.2f}%")

measure_sensitivity(6, np.linspace(-1, 1, 100), "fire_dx")
measure_sensitivity(7, np.linspace(-1, 1, 100), "fire_dy")
measure_sensitivity(8, np.linspace(0, 1, 100), "fire_distance")

print("\n--- DIAGNOSTIC B: WATER DISTANCE RESPONSE ---")
for dist_cells in [1, 5, 10, 20, 30]:
    dist_norm = dist_cells / max_dist
    dx_norm = dist_cells / 48
    _, probs = get_probs(dx_norm, 0, dist_norm)
    print(f"distance {dist_cells} -> P(WATER) = {probs[5]*100:.2f}%")
