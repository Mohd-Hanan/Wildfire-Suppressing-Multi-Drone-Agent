# Stage 7.53 Distance Curriculum

## Stage 1 — 2–3 cells
**NOT LEARNED**
- **Updates**: 100
- **Reward**: -29.01 (final evaluation)
- **Suppression**: Dropped from 4.40 at Update 20 to 0.30 at Update 100.
- **WATER success**: Maxed at 10.7%.
- **Distance**: 2-3 cells.
- **Closer %**: 13.2%
- **Farther %**: 19.4%
- **Crashes**: Policy learned to avoid boundaries by minimizing movement (`Same distance` = 67.4%).

The policy failed to reliably suppress the fire even when it was spawned directly adjacent to the drone's starting base. Instead of learning to move 2 steps and drop WATER for a `+10` reward, the policy learned that moving randomly results in boundary crashes (`-10`). To minimize this massive negative variance, the drone converged on a conservative "stay put" policy.

## Stage 2 — 5–7 cells
**UNKNOWN** (Skipped due to Stage 1 failure)

## Stage 3 — 10–12 cells
**UNKNOWN** (Skipped due to Stage 1 failure)

## Stage 4 — 15–20 cells
**UNKNOWN** (Skipped due to Stage 1 failure)

## Stage 5 — 25–30 cells
**UNKNOWN** (Skipped due to Stage 1 failure)

## Stage 6 — Normal distribution
**UNKNOWN** (Skipped due to Stage 1 failure)

## Objective 1 Status
**NOT ACHIEVED**. The drone completely fails to demonstrate fire-directed suppression. 

### Core Conclusion
The learning dynamics are fundamentally broken by the `-10` crash penalty dominating the gradients. Even when the fire is 2 cells away and easily extinguishable for `+10` reward, the network learns to freeze in place (STAY) because random exploration has a high probability of hitting the map boundary. The `+10` reward is sparse and hidden behind a multi-step sequence (Move → Move → Water), while the `-10` crash penalty is dense (one wrong step near the base edge triggers it instantly). 

PPO will not learn to navigate to the fire without a fundamentally different pathfinding approach (e.g., A* waypoints or a `MOVE_TOWARD_FIRE` high-level action) or removing the crash penalty entirely during early training.
