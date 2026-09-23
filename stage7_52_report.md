# Stage 7.52 Global Fire Spatial Channel

## Observation change
A 6th observation channel was added to encode a global spatial gradient pointing toward the nearest active fire. The channel encodes `1.0 - (Euclidean distance to fire / Max world distance)`. This effectively paints a gradient over the drone's 11x11 local view that points directly at the fire, regardless of how far away the fire is.

## Sixth-channel validation
**VERIFIED**. A synthetic deterministic unit test confirmed the correct operation of the spatial channel. The highest channel value consistently mapped to the exact map edge that faced the fire, creating a perfectly aligned directional gradient (e.g. `(5,0)` for North, `(10,5)` for East). 

## Training configuration
A fresh `ActorCritic6Channels` policy was initialized and trained for exactly 20 updates on the standard environment using the existing PPO hyperparameters. The vector head was left unchanged, meaning the model technically had both the 6th spatial channel and the original 3 scalar features available.

## Directional action response
**NOT IMPROVED**. Using synthetic evaluation, the probability of selecting directional movement was still completely invariant to the position of the fire:
- **FIRE EAST**: P(EAST) = 17.7%, P(WEST) = 21.8%
- **FIRE WEST**: P(EAST) = 17.8%, P(WEST) = 21.8%
- **FIRE NORTH**: P(NORTH) = 22.1%, P(SOUTH) = 14.4%
- **FIRE SOUTH**: P(NORTH) = 22.1%, P(SOUTH) = 14.3%

## WATER distance response
**NOT IMPROVED**.
- distance 1 -> P(WATER) = 7.15%
- distance 10 -> P(WATER) = 7.33%
- distance 20 -> P(WATER) = 7.57%
The network still lacks any correlation between fire distance and WATER drops.

## 20-episode evaluation
Quantitative evaluation on the stochastic evaluation environment:
- **Mean reward**: -85.16 ± 49.69
- **Mean burned area**: 478.05 ± 406.99
- **Total suppressions**: 4 (0.20 per episode)
- **WATER success rate**: 0.8%
- **Mean dist when WATER**: 35.38 cells
- **WATER >10 cells**: 89.8%
- **Strict supp extinction**: 0.0%
- **Crash rate**: 10.0%

## Stage 7.51 vs Stage 7.52
- **Directional Sensitivity**: Both stages failed completely to learn any meaningful directional correlation.
- **Suppression Performance**: Worsened (4 suppressions vs 11).
- **Crash Rate**: Improved significantly (10% vs 75%), which is typical of early PPO training as it learns to avoid map boundaries, but it achieved this without moving toward the fire.
- **WATER accuracy**: Worsened (0.8% vs 1.4%).

## Visual behavior
Skipped per instructions ("Only run Pygame if quantitative diagnostics show that movement probabilities now respond to fire direction").

## Objective 1 status
**NOT ACHIEVED**. The drone continues to behave as a random-walking agent that occasionally drifts and drops water aimlessly.

## Next step
The fundamental issue remains exactly what was hypothesized in Stage 7.51: **20 PPO updates (5,120 steps) is severely insufficient for a randomly initialized convolutional neural network to learn a mapping from spatial gradients to explicit navigation**. 

In typical deep reinforcement learning, teaching a CNN to extract a directional signal from a 2D feature map and navigate towards a goal takes hundreds of updates. In the first 20 updates, PPO's gradient steps are entirely dominated by learning to avoid the massive crash penalties (-10 reward), which it successfully achieved in this run (crashes dropped to 10%). It simply has not had enough training time to optimize the much more subtle distance-shaping reward that would teach it to follow the gradient.

**Recommended Next Step**: Stop changing the architecture and immediately allow the policy to train for **200 updates**. 
