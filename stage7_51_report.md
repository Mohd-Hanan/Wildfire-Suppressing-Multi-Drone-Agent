# Stage 7.51 Balanced Fire Training

## 1. Training distribution
To expose the agent to all major fire directions while preserving map validity, the `BalancedFireEnv` overridden `reset()` method spawned fires at variable distances directly in the 8 cardinal/ordinal directions relative to the `(2,2)` base. 

Verification of 100 resets showed a highly balanced initial distribution:
- NORTH: 9.0%
- SOUTH: 21.0%
- EAST: 12.0%
- WEST: 6.0%
- NE: 11.0%
- NW: 8.0%
- SE: 11.0%
- SW: 22.0%

## 2. Training configuration
A fresh `SeparateActorCriticSymmetric64` policy was initialized and trained for exactly 20 PPO updates using the existing training hyperparameters (LR=3e-4, 4 epochs, minibatch=64). No changes were made to architecture, rewards, observations, or environment mechanics.

## 3. Directional policy response
Testing the trained policy on a synthetic observation with the fire placed at distance 10 in various directions yielded:
- **FIRE EAST**: P(EAST) = 19.2%, P(WEST) = 16.7%
- **FIRE WEST**: P(EAST) = 19.3%, P(WEST) = 16.7%
- **FIRE NORTH**: P(NORTH) = 13.3%, P(SOUTH) = 19.0%
- **FIRE SOUTH**: P(NORTH) = 13.4%, P(SOUTH) = 19.0%

**NOT IMPROVED**: The policy continues to completely ignore the global fire direction vector. P(EAST) remains at ~19.2% and P(NORTH) remains at ~13.3% regardless of where the fire actually is.

## 4. Feature sensitivity
- `fire_dx` probability sensitivity: **0.65%**
- `fire_dy` probability sensitivity: **0.74%**
- `fire_distance` probability sensitivity: **0.36%**

**NOT IMPROVED**: While slightly higher than the 0.22% observed in Stage 7.50, the sensitivity remains virtually zero.

## 5. WATER distance response
- distance 1 -> P(WATER) = 11.40%
- distance 5 -> P(WATER) = 11.37%
- distance 10 -> P(WATER) = 11.33%
- distance 20 -> P(WATER) = 11.23%
- distance 30 -> P(WATER) = 11.11%

**NOT IMPROVED**: The agent randomly drops WATER at ~11.3% frequency regardless of fire proximity.

## 6. 20-episode evaluation
Quantitative evaluation on the *normal* stochastic evaluation environment:
- **Mean reward**: -130.78 ± 73.76
- **Mean burned area**: 474.35 ± 403.80
- **Total suppressions**: 11 (0.55 per episode)
- **WATER success rate**: 1.4%
- **Mean dist when WATER**: 27.11 cells
- **WATER >10 cells**: 84.4%
- **Strict supp extinction**: 5.0%
- **Crash rate**: 75.0%

## 7. Stage 7.49 vs Stage 7.51
- **Fire-direction sensitivity**: Remained at zero (0.22% vs 0.74%).
- **WATER distance**: Worsened slightly (81.7% >10 cells vs 84.4% >10 cells).
- **WATER success rate**: Worsened slightly (2.7% vs 1.4%).
- **Successful suppressions**: Worsened slightly (27 vs 11).
- **Strict extinction**: Worsened slightly (10% vs 5%).
- **Crashes**: Worsened (50% vs 75%).
- **Reward**: Worsened (-104.68 vs -130.78).

The performance difference is largely noise; both policies exhibit identical fundamental behavior (a biased random walk ignoring global vector data).

## 8. Visual behavior
Skipped per instructions ("If the quantitative results show meaningful directional learning: run stage7_51_visual_trace.py").

## 9. Objective 1 status
**NOT ACHIEVED**. The drone is fundamentally unable to navigate to the fire because the PPO network fails to integrate the global vector observation into its decision-making.

## 10. Recommended next step
**Architecture Bottleneck**: The policy network `SeparateActorCriticSymmetric64` fuses a 64-channel CNN output (from the 5x11x11 spatial map) with a 64-dim vector representation (from the 12-dim drone/wind vector). Because the spatial map contains the rich, immediately dense local fire/terrain data, its gradients likely dominate the fusion layer. The sparse 3-element global vector `(fire_dx, fire_dy, fire_distance)` is either being washed out during backpropagation or requires substantially more than 20 updates for the optimizer to assign it meaningful weight.

**Proposed Interventions**:
1. **Extended Training (Fastest test)**: Train for 100-200 updates on the balanced curriculum to see if the vector gradients eventually penetrate the fusion layer once the CNN features stabilize.
2. **Global Map Representation**: Instead of passing `fire_dx/dy` as a tiny vector, project the global fire position into an additional 11x11 spatial channel (e.g. a 2D coordinate gradient pointing towards the fire) so the CNN natively processes direction.
3. **Architecture Adjustment**: Increase the dimensionality of the vector representation, add LayerNorm, or introduce a skip-connection to prevent the 64-dim spatial embedding from dominating the actor head.
