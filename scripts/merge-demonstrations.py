"""Aggregate full-game demonstration datasets for DAgger.

  python scripts/merge-demonstrations.py OUT.npz IN1.npz IN2.npz ...

Episode ids are offset so they stay unique; the last input's last episode
becomes the held-out validation episode of `full-train --dataset`.
"""
import sys
import numpy as np

out, paths = sys.argv[1], sys.argv[2:]
parts = {k: [] for k in ('images', 'features', 'labels', 'episode_ids')}
offset = 0
for path in paths:
    data = np.load(path, allow_pickle=False)
    for key in parts:
        parts[key].append(data[key] + (offset if key == 'episode_ids' else 0))
    offset = int(parts['episode_ids'][-1].max()) + 1
    print(path, len(data['labels']), 'samples')
np.savez_compressed(out, **{k: np.concatenate(v) for k, v in parts.items()})
print('merged', sum(len(x) for x in parts['labels']), 'samples,', offset, 'episodes ->', out)
