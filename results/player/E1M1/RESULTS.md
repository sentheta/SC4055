# E1M1 player-view student

Seeds 900-939, 3000 steps. Quantum model `model.pt`: 36/40. Classical model `classical.pt` trained from scratch on the same data: 33/40. Quantum model with gain zeroed: 34/40. These differences are within noise, so they show no quantum advantage.

PennyLane on seeds 900-919: 17/20, the same as torch. Logits differ by at most 4e-6 over 50 real inputs, with all 50 actions matching.

Larger E1M1 check on new seeds 1000-1079:
- model: eval E1M1 completed 74 / 80 | deaths 1 | mean cells 63 | keys 0 06:42
- classical: eval E1M1 completed 64 / 80 | deaths 5 | mean cells 62 | keys 0 06:45
- PennyLane on seeds 900-939: 36/40 (1 death), the same as torch.
