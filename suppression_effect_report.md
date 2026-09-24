# Controlled Suppression-Effect Experiment Report

## 1. Objective
To determine whether drone deployment/suppression actions (Water/Retardant) actually have a measurable effect on wildfire dynamics within the current simulator environment.

## 2. Experimental Setup
We evaluated four agents over an identical set of initial environment seeds. All agents used the exact same `WildfireEnv` simulator, observation logic, action mapping, and termination semantics without any modifications. The resulting episode metrics were collected side-by-side to detect any deviation from the pure baseline control.

## 3. Agents Tested
- **NoSuppressionAgent (Control)**: A deterministic baseline that simply stays at the base (Action 0), deliberately avoiding all resource deployment and movement.
- **RandomAgent**: An agent selecting uniform random actions across the entire 7-action space.
- **HeuristicAgent**: An explicitly programmed controller that targets active fire cells (`IGNITING`, `BURNING`, `SMOLDERING`) and attempts direct-attack suppression.
- **ControlledSuppressionAgent**: A purpose-built testing agent that seeks out active fire, but explicitly deploys resources in the local unburned footprint adjacent to or directly on the fire, designed specifically to trigger the environment's suppression mechanics.

## 4. Seeds Used
`100, 101, 102, 103, 104, 105, 106, 107, 108, 109` (10 total episodes)

## 5. Metrics Collected
- Total Episode Reward
- Total Burned Cells
- Fire Extinction Status
- Episode Length (Steps)
- Total Crashes
- Final Battery & Payload
- Number of Successful Water Deployments

## 6. Per-Seed Burned Cells Comparison

| Seed | No Suppression | Random | Heuristic | Controlled Supp. |
|------|----------------|--------|-----------|------------------|
| 100  | 290            | 290    | 290       | 290              |
| 101  | 3              | 3      | 3         | 3                |
| 102  | 1928           | 1926   | 1928      | 1926             |
| 103  | 196            | 196    | 196       | 196              |
| 104  | 504            | 504    | 504       | 504              |
| 105  | 817            | 817    | 817       | 817              |
| 106  | 608            | 605    | 608       | 608              |
| 107  | 1785           | 1779   | 1785      | 1785             |
| 108  | 1137           | 1137   | 1137      | 1137             |
| 109  | 533            | 533    | 533       | 533              |

*(Note: Extinction state was identical for all agents across all 10 seeds: 9 extinguishments, 1 max-step truncation).*

## 7. Summary Comparison

| Metric | No Suppression | Random | Heuristic | Controlled Supp. |
|--------|----------------|--------|-----------|------------------|
| Mean Burned Cells | 780.1 | 779.0 | 780.1 | 779.9 |
| Extinction Rate | 90% | 90% | 90% | 90% |
| Mean Length | 310.7 | 308.1 | 310.7 | 310.7 |
| Mean Successful Drops | 0.0 | 11.2 | 9.2 | 8.6 |

## 8. Measurable Fire-State Changes
Under the tested seeds, deployment actions **did produce** a measurable, albeit extremely microscopic, change in burned-cell outcomes.
- In Seed 102, Random and Controlled Suppression reduced the burn count by exactly 2 cells (from 1928 to 1926).
- In Seed 106, Random reduced the burn count by 3 cells.
- In Seed 107, Random reduced the burn count by 6 cells.

However, the `HeuristicAgent`, despite successfully dropping water 9.2 times on average, produced **zero** deviation from the `NoSuppression` control baseline across all 10 seeds.

## 9. Important Limitations
The existing environment simulator (`src/wildfire/simulation/fire.py`) processes Cellular Automata (CA) physics using a spread probability metric (`base_p`) that relies on `moisture`. However, this `base_p` is calculated exclusively for `UNBURNED` cells. 
The simulator currently does not support extinguishing an actively burning cell (i.e. if a cell is `IGNITING` or `BURNING`, increasing its moisture has zero physical effect on stopping it). Because the `HeuristicAgent` exclusively targets and deploys resources on actively burning cells, its actions mechanically cannot change the fire outcome. Drops must land precisely on unburned fuel ahead of the fire line to achieve the minor reductions seen in Random and Controlled agents.

## 10. Conclusion
Deploying suppression resources technically possesses mechanical causality in the environment simulator, but only when targeting `UNBURNED` cells. Direct-attack suppression on actively burning fires yields identically zero mechanical effect compared to doing nothing. The 90% extinction rate observed is a function of the fire burning itself out naturally within the closed grid, not the result of active drone suppression.
