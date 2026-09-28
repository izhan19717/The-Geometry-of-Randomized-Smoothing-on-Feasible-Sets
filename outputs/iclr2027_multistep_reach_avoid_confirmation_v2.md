# ICLR 2027 multi-step reach--avoid controller benchmark
This is a frozen-split discrete-time waypoint benchmark. The controller is a standard one-step receding-horizon goal-directed controller, not a learned policy. The smoothed variable is its **action center**. No observation-space or end-to-end policy certificate is claimed.
Each sampled next-state waypoint is checked exactly against the current box-minus-rectangle support. Collision does not include the swept segment between waypoints.
## Confirmation results
| method | success (95% layout CI) | return (95% CI) | path length | final distance | actual proposals | rejection-equivalent proposals | runtime ms |
|:--|--:|--:|--:|--:|--:|--:|--:|
| native_controller | 1.000 [1.000, 1.000] | 2.994 [2.993, 2.996] | 7.006 | 0.000 | 0.00 | 0.00 | 0.17 |
| moving_conditioned_gaussian | 0.932 [0.916, 0.947] | 1.694 [1.517, 1.852] | 7.186 | 0.218 | 10.18 | 10.20 | 0.19 |
| fixed_vertical_collision_repair | 0.908 [0.892, 0.924] | 1.436 [1.247, 1.617] | 7.198 | 0.224 | 7.00 | 7.00 | 0.19 |
| anchor_frozen_corridor_projection | 0.948 [0.934, 0.962] | 1.895 [1.727, 2.046] | 7.171 | 0.209 | 7.00 | 7.00 | 0.52 |
| fixed_uniform_conditioned_coarse4 | 0.700 [0.681, 0.719] | -1.394 [-1.610, -1.173] | 7.648 | 0.374 | 0.00 | 69695564113168441344.00 | 0.27 |
| anchor_frozen_conditioned_coarse4 | 0.938 [0.926, 0.949] | 1.756 [1.626, 1.875] | 7.190 | 0.217 | 0.00 | 40.79 | 0.50 |
| anchor_frozen_conditioned_refined8 | 0.931 [0.918, 0.942] | 1.675 [1.540, 1.804] | 7.192 | 0.220 | 0.00 | 81.78 | 0.54 |

### Per-state certificate audit
| method | theorem-backed? | states attacked | layouts with any attack (Wilson 95%) | median max full-KL ratio | max full-KL ratio |
|:--|:--:|--:|--:|--:|--:|
| moving_conditioned_gaussian | no | 448 (100.0%) | 100.0% [94.3, 100.0] | 1.086 | 1.499 |
| fixed_vertical_collision_repair | yes | 448 (0.0%) | 0.0% [0.0, 5.7] | -- | -- |
| anchor_frozen_corridor_projection | yes | 448 (0.0%) | 0.0% [0.0, 5.7] | -- | -- |
| fixed_uniform_conditioned_coarse4 | yes | 448 (0.0%) | 0.0% [0.0, 5.7] | 0.631 | 0.755 |
| anchor_frozen_conditioned_coarse4 | yes | 448 (0.0%) | 0.0% [0.0, 5.7] | 0.856 | 0.998 |
| anchor_frozen_conditioned_refined8 | yes | 448 (0.0%) | 0.0% [0.0, 5.7] | 0.520 | 0.961 |
| convex_box_conditioned_control | yes | 448 (0.0%) | 0.0% [0.0, 5.7] | 1.000 | 1.000 |

## Interpretation boundary
The moving-law Gaussian radius is deliberately audited although it is not valid: occupancy weights move with the action center. The five fixed alternatives are sound from data processing or the convex fixed-weight theorem; a zero attack-found rate is not used as proof. The convex-box row is a negative-control audit and is not deployed as an obstacle-avoidance controller.

The anchor-occupancy coarse4 and refined8 laws both exactly partition the same feasible support and match whole-set conditioning at their anchor. Their away-from-anchor laws differ, so their utility/cost comparison is a partition-sensitivity result rather than a claim that one partition is canonical.

Wall-clock timing is secondary and implementation-specific. Primitive random-variate counts, inverse-CDF calls, actual moving rejection draws, and exact rejection-equivalent expectations are reported separately.

Deterministic result SHA-256 excluding wall clock: `1918fbb42e35876a8af58e77f3d147307c76e52b73d8ec3e658647cfb1aa26c8`.
