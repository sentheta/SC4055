# E1M3 / e1m3_v1

Training: full-game behavior cloning, 200000 samples, selected epoch 6, auto_weapon=True, max_steps=8000.

| Epoch | Loss | Held-out accuracy | Validation completion |
|---:|---:|---:|---:|
| 1 | 0.4914 | 0.938 |  |
| 2 | 0.2472 | 0.979 |  |
| 3 | 0.1981 | 0.973 |  |
| 4 | 0.1705 | 0.979 |  |
| 5 | 0.1479 | 0.986 |  |
| 6 | 0.1318 | 0.986 | 0.4 |
| 7 | 0.1170 | 0.993 |  |
| 8 | 0.1061 | 0.979 |  |
| 9 | 0.0935 | 0.993 |  |
| 10 | 0.0844 | 0.986 |  |
| 11 | 0.0790 | 1.000 |  |
| 12 | 0.0739 | 1.000 | 0.4 |

| Backend | Episodes | Completion | Deaths | Mean kills | Mean steps | Inference s |
|---|---:|---:|---:|---:|---:|---:|
| torch | 20 | 0.50 | 0.35 | 27.4 | 4579 | 163 |
| pennylane | 20 | 0.50 | 0.35 | 27.4 | 4579 | 523 |

Identical episodes (outcome, steps, kills): 20/20.

GIFs: qnn_E1M3_seed804_death.gif, qnn_E1M3_seed815_completed.gif
