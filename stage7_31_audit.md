# Learning Signal Audit Report

### 1. Reward Formula Components
- **Valid Movement**: -0.01 step penalty
- **Blocked Movement**: -0.11 (-0.01 step + -0.1 boundary)
- **STAY**: -0.01 step penalty
- **Water Deployment (successful)**: -0.01 step penalty (suppression reward is hardcoded to 0.0!)
- **Water Deployment (unsuccessful)**: -0.01 step penalty
- **Battery Depletion / Crash**: -10.0 crash penalty
- **Return to Base / Base Refill**: 0.0 reward 
- **Fire Spreading**: -5.0 per newly burned cell
- **Fire Extinction**: +100.0 (requires 0 active fire cells)

### 2. Is suppression rewarded?
**NO.** `effective_suppression_reward` is set to 0.0 in the configuration, and `newly_suppressed_cells` is literally hardcoded to `0` inside `RewardCalculator.calculate(Line 62)`. The drone receives **absolutely zero** positive reinforcement for dropping water on the fire.

### 3. Can the drone see the fire?
**NO.** The observation builder uses an 11x11 spatial window centered on the drone (5 cells in each direction). 
The drone spawns at the base `(2,2)`. The fire spawns far away (e.g., `(30,30)`). 
For the first 100+ steps, the drone's fire observation channel is entirely zeroes. It is functionally blind and has absolutely no information indicating where the fire is located or which direction to fly.

### 4. Can the drone learn to find the fire?
**NO.** There is no "distance-to-fire" reward or heuristic guiding it. While it is flying blindly in an empty field, the fire is spreading miles away. Every step, the drone receives a massive penalty (e.g. -15.0) because the fire burned 3 new cells. Moving NORTH produces -15.01. Moving SOUTH produces -15.01. The drone receives no differential signal indicating it is getting closer to the problem.

### 5. Scale Imbalance
Over a 500-step episode:
- **Fire Spreading**: ~1500 cells burned = **-7500.0**
- **Step Penalty**: -0.01 * 500 = **-5.0**
- **Crash Penalty**: **-10.0**
- **Suppression**: **0.0**

The massive, noisy fire spreading penalty completely dwarfs everything else. The drone's individual actions (which cost -0.01 or -0.1) are drowned out by the noise of the fire spreading.

### Conclusion
The policy currently has **no visibility of the fire** and **no reward for suppressing it**. 
The EAST/SOUTH biases were the only mathematically rational response: since the drone is blind and its actions don't affect the massive negative reward, it simply picks a random cardinal direction and flies in a straight line until its battery runs out.

We must modify the observation space to provide a global fire vector, and we must fix the reward function to actually reward suppression before attempting any more training.
