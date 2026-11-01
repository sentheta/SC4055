"""Behaviour cloning for the player-view student.

  player-train.py OUT.pt EPOCHS KIND INIT|none DATA.npz [DATA.npz ...]   (env LR, THREADS, HIDDEN, AUX=weight of the training-only auxiliary loss)
"""
import json
import os
import sys
import time
import numpy as np
import torch
from corridor_qnn.full_actions import ACTIONS
from corridor_qnn.player import AUX_INDEX, PlayerModel, load_player, save_player, stack_indices

out, epochs, kind, init, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:]
torch.manual_seed(0); rng = np.random.default_rng(0)
torch.set_num_threads(int(os.environ.get('THREADS', 8)))
images, features, labels, episodes, aux = [], [], [], [], []; offset = 0
for path in paths:
    data = np.load(path)
    images.append(data['images']); features.append(data['features']); labels.append(data['labels'])
    # Older datasets have no auxiliary targets: NaN rows are masked out of the loss.
    aux.append(data['aux'] if 'aux' in data else np.full((len(data['labels']), len(AUX_INDEX)), np.nan, np.float32))
    episodes.append(data['episode_ids']+offset); offset = episodes[-1].max()+1
images, features, labels, episodes, aux = (np.concatenate(v) for v in (images, features, labels, episodes, aux))
aux_weight = float(os.environ.get('AUX', 0))
model = load_player(init) if init != 'none' else PlayerModel(kind, hidden=int(os.environ.get('HIDDEN', 256)), aux=aux_weight > 0)
if aux_weight > 0 and model.aux_head is None:
    model.aux_head = torch.nn.Linear(64, len(AUX_INDEX))
held = episodes == episodes.max()
train = np.flatnonzero(~held); val = np.flatnonzero(held)
counts = np.bincount(labels[train], minlength=len(ACTIONS))
weights = torch.tensor(np.where(counts > 0, np.sqrt(counts.max()/np.maximum(counts, 1)), 0), dtype=torch.float32)
lr = float(os.environ.get('LR', 1e-3))
opt = torch.optim.Adam(model.parameters(), lr=lr)
steps = epochs*(len(train)//256+1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.1)
history = []; started = time.time()


def batch(idx):
    return (torch.from_numpy(images[stack_indices(episodes, idx)]), torch.from_numpy(features[idx]),
            torch.from_numpy(labels[idx]).long())


for epoch in range(1, epochs+1):
    model.train(); total = 0.
    order = rng.permutation(train)
    for k in range(0, len(order), 256):
        x, f, y = batch(order[k:k+256])
        if aux_weight > 0:
            logits, predicted = model(x, f, return_aux=True)
            target = torch.from_numpy(aux[order[k:k+256]]); mask = ~target.isnan()
            extra = ((predicted-target.nan_to_num())[mask]**2).mean() if mask.any() else 0.
        else:
            logits, extra = model(x, f), 0.
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)+aux_weight*extra
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
        if sched.last_epoch < steps-1:
            sched.step()
        total += loss.item()*len(y)
    model.eval(); correct = 0
    with torch.no_grad():
        for k in range(0, len(val), 512):
            x, f, y = batch(val[k:k+512]); correct += (model(x, f).argmax(-1) == y).sum().item()
    row = dict(epoch=epoch, loss=total/len(train), episode_heldout_accuracy=correct/max(len(val), 1))
    history.append(row); print('EPOCH', json.dumps(row), round(time.time()-started), flush=True)
save_player(out, model, dict(datasets=paths, samples=len(labels), history=history))
json.dump(history, open(out.rsplit('.', 1)[0]+'_history.json', 'w'))
