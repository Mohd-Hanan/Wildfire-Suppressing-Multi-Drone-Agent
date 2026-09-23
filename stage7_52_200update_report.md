# Stage 7.52 — 200 Update Training

## 1. Training configuration
The `ActorCritic6Channels` policy loaded from the 20-update checkpoint was trained for an additional 180 PPO updates (bringing the total to 200). The hyperparameters, rewards, architecture, and observation spaces were identical to the previous step.

## 2. Training progress
During training, the policy initially increased its `STAY` and boundary-avoidance behavior, causing crash-driven value loss to drop. Entropy fell slightly but remained healthy (`~1.5` - `1.6`). The `WATER` action probability gradually declined to roughly `2%` as the model learned that randomly dropping water was ineffective. However, mean rollout reward remained consistently negative throughout the 200 updates (`-0.4` to `-0.8`).

## 3. Fire-direction diagnostic
**NOT IMPROVED**. The synthetic response diagnostic evaluated after 200 updates confirmed that the network *still* entirely ignores the 6th spatial channel:
- **FIRE EAST**: P(EAST) = 21.3%, P(WEST) = 9.2%
- **FIRE WEST**: P(EAST) = 21.2%, P(WEST) = 9.1%
- **FIRE NORTH**: P(NORTH) = 15.0%, P(SOUTH) = 28.1%
- **FIRE SOUTH**: P(NORTH) = 15.0%, P(SOUTH) = 28.5%

Action probabilities are driven completely by a learned static bias (favoring `STAY` and `SOUTH` to avoid the North-West boundary near the starting base) rather than the dynamically moving fire gradient.

## 4. WATER distance diagnostic
**NOT IMPROVED**.
- distance 1 -> P(WATER) = 2.74%
- distance 5 -> P(WATER) = 2.68%
- distance 10 -> P(WATER) = 2.46%
- distance 20 -> P(WATER) = 2.20%

## 5. Real trajectory analysis
**NOT IMPROVED**. Over 20 evaluation episodes:
- **Steps moving closer**: 26.7%
- **Steps moving farther**: 22.7%
- **Steps same distance**: 50.6%
- **Initial fire distance**: 44.95 cells
- **Final fire distance**: 34.10 cells

The drone merely drifts slowly away from the base, cutting the initial distance slightly, but not actively navigating the 30+ cells required to reach the active fire. 

## 6. 20-episode evaluation
- **Mean reward**: -186.88 ± 114.90
- **Mean burned area**: 476.90 ± 405.15
- **Total suppressions**: 10 (0.50 per episode)
- **WATER success rate**: 6.4%
- **Mean WATER distance**: 20.03 cells
- **WATER >10 cells**: 68.2%
- **Strict supp extinction**: 5.0%
- **Natural extinction**: 50.0%
- **Crash rate**: 70.0%

## 7. Comparison with previous stages
- **Fire-direction response**: Invariant (identical to Stage 7.51 and Stage 7.52-20).
- **Mean fire distance**: Identical (~34 cells).
- **Movement toward fire**: No active homing behavior.
- **WATER success**: Increased slightly (6.4% from 1.4%) only because total WATER attempts dropped drastically (from 796 to 157).
- **Successful suppressions**: Invariant (~10).
- **Reward/Crashes**: Crash rate increased back to 70% during evaluation stochastic sampling.

## 8. Visual behavior
Skipped per instructions ("ONLY run the Pygame visual evaluation if the quantitative diagnostics show meaningful fire-directed behavior.")

## 9. Objective 1 status
**NOT ACHIEVED**. The drone completely fails to move towards the fire. It is stuck in a local minimum where it optimizes for surviving map boundaries rather than following the distance-shaping reward gradient.

## 10. Recommended next step
The hypothesis from Stage 7.47 regarding credit assignment remains the most likely culprit. The shaping reward (`+0.2` for moving closer) is simply too small or too noisy given the length of the map (`48x48`) and the short rollout horizon (`256`).

If the drone takes a random walk, the net shaping reward is exactly `0`, but the variance is massive due to boundary crashes (`-10`). The optimizer is dropping the subtle spatial gradients in favor of strong static biases (`SOUTH`/`EAST`/`STAY`) that immediately prevent crashes near the `(2,2)` base.

To solve this, we must either:
1. **Curriculum learning on distance**: Spawn the fire directly adjacent to the drone (distance 1-3) so it immediately discovers the massive `+10` suppression reward, and gradually expand the spawn radius over updates.
2. **Pathfinding baseline**: Stop trying to force PPO to learn long-range A* navigation from raw pixels, and instead provide a higher-level action space (e.g., `MOVE_TOWARD_FIRE`) or provide the pre-computed A* waypoint as the observation.
