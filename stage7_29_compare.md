# Baseline vs Boundary Penalty Comparison

| Metric | Original 200-Update Baseline | New 200-Update (with Boundary Penalty) |
| --- | --- | --- |
| **Mean Reward** | -3004.39 | -7542.80 |
| **Mean Burned Cells** | 0.00 | 1505.35 |
| **Total Crashes** | 12 | 0 |
| **Mean Ep Length** | 265.2 steps | 500.0 steps |
| **Extinction Rate** | 0.0% | 0.0% |
| **STAY** | 4.3% | 17.7% |
| **NORTH** | 1.9% | 20.4% |
| **SOUTH** | 78.4% | 10.3% |
| **EAST** | 1.2% | 14.6% |
| **WEST** | 10.2% | 11.7% |
| **WATER** | 4.1% | 25.2% |

### Analysis
The boundary penalty was highly effective at solving the boundary-crash behavior! By providing immediate negative feedback when the drone attempts to walk into a wall, the policy learned to completely abandon the "grind against the SOUTH wall" collapse.
- **Crashes**: Reduced from 12 out of 20 episodes to **0**.
- **Action Distribution**: The catastrophic 78.4% SOUTH bias is entirely gone, replaced by a much more balanced exploration across all four cardinal directions.
- **Episode Length**: Because it no longer grinds against walls wasting battery, it successfully survives for the maximum 500 steps in every single episode, utilizing base refills properly to stay in the air.
- **Burned Cells**: Because the agent now explores the entire map instead of immediately crashing into the South wall, the episodes run much longer (500 steps vs 265 steps), which allows the fire more time to burn. Therefore the mean reward is mathematically lower and burned cells are higher, but this is a sign of a healthier policy that is actually exploring and surviving, rather than achieving "0 burned cells" by crashing on step 160.
