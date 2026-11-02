"""Behaviour cloning for the player-view student.

  player-train.py OUT.pt EPOCHS KIND INIT|none DATA.npz [DATA.npz ...]   (env LR, THREADS, HIDDEN, AUX=weight of the training-only auxiliary loss, MEMORY=sequence length for a GRU student)
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
# Filled in place: colour frames are large and a list-then-concatenate would double peak memory.
sizes = [len(np.load(p)['labels']) for p in paths]; total = sum(sizes); first = np.load(paths[0])
images = np.empty((total, *first['images'].shape[1:]), np.uint8); features = np.empty((total, first['features'].shape[1]), np.float32)
labels = np.empty(total, np.int64); episodes = np.empty(total, np.int64); aux = np.full((total, len(AUX_INDEX)), np.nan, np.float32)
at = offset = 0
for path, n in zip(paths, sizes):
    data = np.load(path)
    images[at:at+n] = data['images']; features[at:at+n] = data['features']; labels[at:at+n] = data['labels']
    if 'aux' in data:
        aux[at:at+n] = data['aux']  # older datasets stay NaN and are masked out of the loss
    episodes[at:at+n] = data['episode_ids']+offset; offset = episodes[at+n-1]+1; at += n
aux_weight = float(os.environ.get('AUX', 0))
model = load_player(init) if init != 'none' else PlayerModel(kind, hidden=int(os.environ.get('HIDDEN', 256)), aux=aux_weight > 0, rgb=images.ndim == 4)
if aux_weight > 0 and model.aux_head is None:
    model.aux_head = torch.nn.Linear(64, len(AUX_INDEX))
seq = int(os.environ.get('MEMORY', 0))
if seq and model.gru is None:
    model.gru = torch.nn.GRU(48, 48, batch_first=True)
held = episodes == episodes.max()
train = np.flatnonzero(~held); val = np.flatnonzero(held)
counts = np.bincount(labels[train], minlength=len(ACTIONS))
weights = torch.tensor(np.where(counts > 0, np.sqrt(counts.max()/np.maximum(counts, 1)), 0), dtype=torch.float32)
lr = float(os.environ.get('LR', 1e-3))
opt = torch.optim.Adam(model.parameters(), lr=lr)
per = 12 if seq else 256  # batch of 12 sequences, or of 256 frames
unit = max(seq, 1)
steps = epochs*(len(train)//(per*unit)+1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.1)
history = []; started = time.time()


def batch(idx):
    return (torch.from_numpy(images[stack_indices(episodes, idx)]), torch.from_numpy(features[idx]),
            torch.from_numpy(labels[idx]).long())


for epoch in range(1, epochs+1):
    model.train(); total = 0.
    order = rng.permutation(train)
    if seq:
        # Windows that stay inside one episode.
        order = order[(order+seq <= len(episodes)) & (episodes[np.minimum(order+seq-1, len(episodes)-1)] == episodes[order])][:len(train)//seq]
    for k in range(0, len(order), per):
        chosen = order[k:k+per]
        chosen = (chosen[:, None]+np.arange(seq)).ravel() if seq else chosen
        x, f, y = batch(chosen)
        if aux_weight > 0:
            logits, predicted = model(x, f, return_aux=True, seq=unit)
            target = torch.from_numpy(aux[chosen]); mask = ~target.isnan()
            extra = ((predicted-target.nan_to_num())[mask]**2).mean() if mask.any() else 0.
        else:
            logits, extra = model(x, f, seq=unit), 0.
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)+aux_weight*extra
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
        if sched.last_epoch < steps-1:
            sched.step()
        total += loss.item()*len(y)
    model.eval(); correct = 0
    with torch.no_grad():
        for k in range(0, len(val), 512):
            x, f, y = batch(val[k:k+512])
            # The held-out episode runs as one sequence in chunks, keeping memory between them.
            logits = model(x, f, seq=len(y), hidden=None if k == 0 else model.last_hidden) if seq else model(x, f)
            correct += (logits.argmax(-1) == y).sum().item()
    row = dict(epoch=epoch, loss=total/len(train), episode_heldout_accuracy=correct/max(len(val), 1))
    history.append(row); print('EPOCH', json.dumps(row), round(time.time()-started), flush=True)
save_player(out, model, dict(datasets=paths, samples=len(labels), history=history))
json.dump(history, open(out.rsplit('.', 1)[0]+'_history.json', 'w'))
