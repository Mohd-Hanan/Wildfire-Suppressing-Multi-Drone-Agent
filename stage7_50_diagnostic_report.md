# Stage 7.50 Diagnostic Report

## 1. Executive Finding
The PPO policy has **almost entirely ignored the global fire direction vector (`fire_dx`, `fire_dy`)**, experiencing a form of partial sensory collapse on these features. The policy selects its actions with near-identical probabilities (e.g. `P(WATER) ~ 14.0%`) regardless of the fire direction or distance. The underlying reason appears to be a severe **training data distribution imbalance** combined with the short 20-update training time: because the drone always starts at the base `(2, 2)` and the fire spawns randomly across the `48x48` map, the initial fire direction is overwhelmingly South-East. Consequently, the agent rarely encounters negative `fire_dx` or `fire_dy` vectors unless it manages to fly past the fire, which it currently does not do effectively. 

## 2. Observation Audit
Inspecting `src/wildfire/environment/observation.py` and `src/wildfire/environment/action.py`:
- `fire_dx` = `(nearest_x - drone.x) / world.width`
- `fire_dy` = `(nearest_y - drone.y) / world.height`
- `fire_distance` = `Euclidean distance / max_dist`
- The `nearest_x`, `nearest_y` are chosen using Euclidean distance among active fire cells (IGNITING, BURNING, SMOLDERING).
- The vector elements are placed at indices 6, 7, and 8 of the 9-element drone vector (the actual model input vector is 12 elements: 9 drone + 3 wind).

## 3. Coordinate Convention
- **x-axis**: Horizontal, increasing East.
- **y-axis**: Vertical, increasing South (Top-Left origin).
- **NORTH**: `dy = -1`
- **SOUTH**: `dy = +1`
- **EAST**: `dx = +1`
- **WEST**: `dx = -1`
- `fire_dx` is calculated as `nearest_x - drone_x`. Positive `fire_dx` means the fire is EAST.
- `fire_dy` is calculated as `nearest_y - drone_y`. Positive `fire_dy` means the fire is SOUTH.
- The coordinate convention matches perfectly between the simulator, the action executor, and the observation builder.

## 4. Synthetic Direction Response
A synthetic observation with the drone at `(24, 24)` and the fire at distance 10 in various directions yields:
- **FIRE NORTH**: STAY 16.3%, NORTH 16.7%, SOUTH 19.4%, EAST 17.6%, WEST 16.1%, WATER 14.0%
- **FIRE SOUTH**: STAY 16.4%, NORTH 16.7%, SOUTH 19.4%, EAST 17.5%, WEST 16.0%, WATER 14.0%
- **FIRE EAST**: STAY 16.3%, NORTH 16.6%, SOUTH 19.4%, EAST 17.6%, WEST 16.0%, WATER 14.0%
- **FIRE WEST**: STAY 16.3%, NORTH 16.7%, SOUTH 19.4%, EAST 17.5%, WEST 16.0%, WATER 14.0%

The policy distribution is virtually static regardless of the fire direction vector.

## 5. Feature Sensitivity
Sweeping each feature individually from -1 to 1 (or 0 to 1 for distance):
- `fire_dx`: max logit sensitivity = 0.0163, max prob sensitivity = 0.0022 (0.22%)
- `fire_dy`: max logit sensitivity = 0.0108, max prob sensitivity = 0.0017 (0.17%)
- `fire_distance`: max logit sensitivity = 0.0124, max prob sensitivity = 0.0022 (0.22%)

The policy is entirely insensitive to the global fire vector.

## 6. WATER Distance Response
Evaluating the probability of the WATER action when the fire is precisely EAST, at varying distances:
- distance 1 -> P(WATER) = 14.02%
- distance 5 -> P(WATER) = 14.02%
- distance 10 -> P(WATER) = 14.02%
- distance 20 -> P(WATER) = 14.01%
- distance 30 -> P(WATER) = 14.01%

The policy has not learned to correlate WATER timing with fire distance.

## 7. Real Trajectory Analysis
Analyzing 5 deterministically seeded evaluation episodes (1624 steps):
- Mean distance change when taking NORTH: +0.37 (moving away)
- Mean distance change when taking SOUTH: -0.44 (moving closer)
- Mean distance change when taking EAST: -0.08 (moving slightly closer)
- Mean distance change when taking WEST: +0.14 (moving away)
- 22.9% of steps moved closer to the fire, while 19.2% moved farther. 57.9% of steps (STAY, WATER, or boundaries) resulted in no distance change.

## 8. Action Distribution by Fire Direction
From the real trajectory logs:
- **Fire SOUTH (704 steps)**: SOUTH 16.8%, NORTH 14.9%, EAST 19.6%
- **Fire EAST (569 steps)**: EAST 18.1%, WEST 16.0%, SOUTH 21.4%
- **Fire WEST (351 steps)**: WEST 14.0%, EAST 16.5%, SOUTH 19.9%
- **Fire NORTH**: 0 steps recorded.
There is a slight natural drift, but no strong correlation between the actual fire direction and the chosen action. The drone is largely executing a random walk biased slightly South/East.

## 9. Action Distribution by Fire Distance
- Dist 2-5 (15 steps): WATER 33.3%
- Dist 6-10 (157 steps): WATER 16.6%
- Dist 11-20 (276 steps): WATER 13.4%
- Dist 21-30 (590 steps): WATER 12.7%
- Dist >30 (586 steps): WATER 15.2%
The WATER action is heavily over-selected at long distances.

## 10. Local vs Global Fire Visibility
- **Inside 11x11 window**: 2.3% of steps
- **Outside 11x11 window**: 97.7% of steps
The fire is outside the drone's local spatial observation 97.7% of the time. The global fire vector is theoretically the *only* way the drone can navigate to the fire for the vast majority of the episode.

## 11. Trained vs Fresh Policy
An identical but completely untrained network (`Fresh`) evaluates similarly:
- Fresh FIRE EAST: EAST=15.7%, WEST=16.6%
- Fresh FIRE WEST: EAST=15.7%, WEST=16.6%
The trained policy (20 updates) has not moved significantly away from the random initialization distribution regarding these inputs.

## 12. Observation Normalization
From the real trajectories:
- `fire_dx`: min=-0.479, max=0.688
- `fire_dy`: min=-0.188, max=0.938
- `fire_distance`: min=0.061, max=0.781
The values are properly normalized within [-1, 1], but notice that `fire_dy` is heavily skewed positive.

## 13. Fire Direction Dataset Statistics
Evaluating 100 fresh environment resets:
- `fire_dx`: mean=0.422 (min=-0.042, max=0.938)
- `fire_dy`: mean=0.477 (min=-0.042, max=0.938)
- **Direction counts**: SOUTH: 40, EAST: 31, SE: 29. (NORTH: 0, WEST: 0, NW: 0, SW: 0, NE: 0).
Because the drone always starts at the base `(2, 2)` and the fire spawns randomly in `[0, 48)`, the initial fire vector is almost exclusively South-East.

## 14. Stage 7.48 vs Stage 7.49
The behavior in Stage 7.49 is largely identical to Stage 7.48. In both stages, the policy failed to learn the global fire vector and relied on a biased random walk to occasionally stumble near the fire. The difference in success (2.7%) is simply a byproduct of the random walk occasionally landing exactly on the fire.

## 15. Primary Diagnosis
**A. Fire vector ignored by policy** (with strong support for **B. PPO has insufficient training** and **F. Observation distribution issue**).

## 16. Evidence
- **Diagnostic 3**: Probabilities shift by a maximum of 0.22% across the entire possible range of the global fire vector.
- **Diagnostic 10**: The drone cannot see the fire in its 11x11 spatial window 97.7% of the time, meaning the ignored global vector is its only navigational aid.
- **Diagnostic 14**: 100% of episode initializations place the fire South, East, or South-East of the drone, depriving the network of diverse directional training examples.

## 17. Possible Next Experiment
- **Randomize the Drone Start Location**: Spawning the drone at random coordinates rather than fixed at `(2, 2)` would immediately balance the `fire_dx` and `fire_dy` distributions across all 8 cardinal/ordinal directions, forcing the policy to learn the relationship.
- **Longer Training**: 20 updates (5120 steps) is very short for a network attempting to fuse a CNN and a vector branch. Training for 100-200 updates may be required for the vector gradients to penetrate.

## 18. What Must NOT Be Changed Yet
Do NOT implement any fixes. Do NOT modify the architecture, reward, or action space. Await user review of this diagnostic report.
