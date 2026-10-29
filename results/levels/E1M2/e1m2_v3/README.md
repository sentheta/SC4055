# E1M2 / e1m2_v3

Training: full-game behavior cloning, 100000 samples, selected epoch 12, auto_weapon=True, max_steps=8000.

| Epoch | Loss | Held-out accuracy | Validation completion |
|---:|---:|---:|---:|
| 1 | 0.6560 | 0.929 |  |
| 2 | 0.2699 | 0.936 |  |
| 3 | 0.2131 | 0.955 |  |
| 4 | 0.1814 | 0.970 |  |
| 5 | 0.1595 | 0.965 |  |
| 6 | 0.1412 | 0.972 | 0.4 |
| 7 | 0.1233 | 0.971 |  |
| 8 | 0.1095 | 0.973 |  |
| 9 | 0.1002 | 0.968 |  |
| 10 | 0.0930 | 0.974 |  |
| 11 | 0.0843 | 0.973 |  |
| 12 | 0.0770 | 0.976 | 0.8 |

| Backend | Episodes | Completion | Deaths | Mean kills | Mean steps | Inference s |
|---|---:|---:|---:|---:|---:|---:|
| torch | 20 | 0.95 | 0.00 | 51.3 | 4703 | 408 |
| pennylane | 20 | 0.95 | 0.00 | 51.3 | 4703 | 1143 |

Identical episodes (outcome, steps, kills): 20/20.

GIFs: qnn_E1M2_seed800.gif, qnn_E1M2_seed805.gif

## Notes

- Planner/teacher code: commit 4c679fd (health threshold 90). The student
  follows the planner's waypoint features, so evaluate with this planner.
- Wider check with the same code (parallel evaluator, seeds 820-869):
  47/50 completed, 2 deaths, 1 timeout. Combined with the 20 seeds above:
  66/70 (94%).
- Further 100 seeds (870-969, `eval_extra_seeds870-969.txt`): 87/100
  (8 deaths, 5 timeouts). **All 170 seeds: 153/170 (90.0%).**
- `pennylane_parity.txt`: circuit and logit agreement on 2000 recorded states.
- Reproduce: `python -m corridor_qnn full-eval --policy quantum --map E1M2
  --auto-weapon --max-steps 8000 --episodes 20 --seed 800 --checkpoint
  results/levels/E1M2/e1m2_v3/quantum.pt --output-dir <new dir>`, and the same
  arguments through `python scripts/pennylane-check.py eval CHECKPOINT ...`.
