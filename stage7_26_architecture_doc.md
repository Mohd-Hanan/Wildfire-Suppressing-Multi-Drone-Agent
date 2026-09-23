# Stage 7.25 Single-Drone Baseline Architecture

This document describes the state of the PPO single-drone baseline after the 200-update training run.

## Checkpoint Information
* **Path**: `stage7_single_drone_200_baseline.pth`
* **Network**: `SeparateActorCritic` (Stage 7.15 implementation)
* **Status**: Trained for 200 updates with stable gradient scaling (no collapse).
* **Role**: Foundational single-drone weights for future multi-agent fine-tuning/transfer learning.
* **Preservation Status**: The 20-update checkpoint (`stage7_18_checkpoint.pth`) was left intact and untouched.

## Network Architecture
**Actor Network**:
* 1D CNN over spatial grid (moisture, fuel, elevation).
* FFN over global drone/wind state.
* Shared representations.
* Separate Actor and Critic bodies (not shared weights).

**Action Masking**:
* Dynamic masking using `logits.masked_fill_`.
* Action 6 (RETARDANT) statically masked for WATER drone.

## Simulation/Environment Status
* **Fire Physics**: Unmodified (Cellular Automata).
* **Resource Physics**: Unmodified. Drone moves deduct 1 battery. Refilling requires returning to `(2,2)` 2x2 footprint base.
* **Drone Type**: PPO exclusively controls Drone 0 (WATER). Drones 1, 2, and 3 are present in the simulation but idle.

## Quantitative Results (200 Updates)
* **Mean Reward**: -4146.97 ± 2043.95
* **Mean Burned Cells**: 827.95
* **Extinction Rate**: 0.0%
* **Mean Episode Length**: 331.7 steps
* **Total Crashes**: 18
* **Total Water Drops**: 955

### Note for Swarm Expansion
This checkpoint expects a specific observation dimension. When expanding to 4-drone multi-agent control later, the `Actor` body may need its input layer resized, or it may need to be applied iteratively to each drone. 
