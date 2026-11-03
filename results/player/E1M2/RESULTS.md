# E1M2 player-view student: not solved

`model.pt` is round 36 of `e1m2_v`: RGB frames, GRU memory, and a frozen pretrained enemy detector (`enemy_detector.pt`, also stored inside the model).

Seeds 900-919, 8000 steps: 0/20 completed, 14 deaths, mean 111 map cells covered (the teacher covers about 450), a key in 10 runs. A low-rate fine-tune gave 0/20 with 17 deaths. `gifs/seed900.gif` shows a typical attempt. See `docs/LEVEL_PROGRESS.md` and section 6 of `notebooks/project_summary.ipynb` for the diagnosis.
