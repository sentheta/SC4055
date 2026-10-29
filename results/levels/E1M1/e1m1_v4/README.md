# E1M1 / e1m1_v4

Training: full-game behavior cloning, 6000 samples, selected epoch 6, auto_weapon=None, max_steps=None.

| Epoch | Loss | Held-out accuracy | Validation completion |
|---:|---:|---:|---:|
| 1 | 2.0185 | 0.701 |  |
| 2 | 1.2147 | 0.785 |  |
| 3 | 0.8026 | 0.806 |  |
| 4 | 0.5609 | 0.889 |  |
| 5 | 0.4245 | 0.889 |  |
| 6 | 0.3395 | 0.903 | 1.0 |
| 7 | 0.2872 | 0.924 |  |
| 8 | 0.2418 | 0.896 |  |
| 9 | 0.2097 | 0.938 |  |
| 10 | 0.1785 | 0.931 |  |
| 11 | 0.1602 | 0.910 |  |
| 12 | 0.1429 | 0.938 | 1.0 |

| Backend | Episodes | Completion | Deaths | Mean kills | Mean steps | Inference s |
|---|---:|---:|---:|---:|---:|---:|
| torch | 50 | 0.98 | 0.02 | 4.8 | 517 | 49 |
| pennylane | 50 | 0.98 | 0.02 | 4.8 | 517 | 178 |

Identical episodes (outcome, steps, kills): 50/50.

GIFs: qnn_E1M1.gif

## Notes

- Evaluated at commit 13f5585 (before the E1M2 planner work), seeds 800-849,
  1500 steps, without --auto-weapon. Later planner commits keep the E1M1
  teacher at 20/20; this checkpoint was not re-evaluated with them.
