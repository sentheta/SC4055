"""player-diagnose.py CHECKPOINT DATA.npz: per-action recall and prediction counts against teacher labels."""
import sys
import numpy as np
import torch
from corridor_qnn.full_actions import ACTION_NAMES
from corridor_qnn.player import load_player, stack_indices
model = load_player(sys.argv[1]).eval()
d = np.load(sys.argv[2])
ep, im, f, y = d['episode_ids'], d['images'], d['features'], d['labels']
idx = np.random.default_rng(0).choice(len(y), min(6000, len(y)), replace=False)
pred = []
with torch.no_grad():
    for k in range(0, len(idx), 256):
        i = idx[k:k+256]
        pred.append(model(torch.from_numpy(im[stack_indices(ep, i)]), torch.from_numpy(f[i])).argmax(-1).numpy())
pred = np.concatenate(pred); t = y[idx]
print('accuracy', round((pred == t).mean(), 3))
for a, name in enumerate(ACTION_NAMES):
    if (t == a).any() or (pred == a).any():
        print(f'{name:28s} teacher {(t == a).sum():5d} predicted {(pred == a).sum():5d} recall {(pred[t == a] == a).mean() if (t == a).any() else 0:.2f}')
