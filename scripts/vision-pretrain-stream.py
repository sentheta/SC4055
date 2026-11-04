"""vision-pretrain-stream.py OUT.pt EPOCHS INIT|none VAL.npz DATA.npz...: train the enemy detector over many datasets, one file in memory at a time.

Same detector and targets as vision-pretrain.py; VAL.npz is held out and scored after each epoch.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from corridor_qnn.player import PRIVILEGED, _cnn, stack_indices, FRAMES
torch.set_num_threads(14); torch.manual_seed(0); rng = np.random.default_rng(0)
out, epochs, init, val_path, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:]
vi, si, ci, di = (PRIVILEGED.index(n) for n in ('enemy_visible', 'enemy_sin', 'enemy_cos', 'enemy_distance'))
vision, head = _cnn(FRAMES*3), torch.nn.Linear(64, 4)
if init != 'none':
    w = torch.load(init); vision.load_state_dict(w['vision']); head.load_state_dict(w['head'])
params = [*vision.parameters(), *head.parameters()]
opt = torch.optim.Adam(params, 5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs*len(paths))


def run(X, A, E, i):
    x = torch.from_numpy(X[stack_indices(E, i)]).permute(0, 1, 4, 2, 3).flatten(1, 2).float()/255
    return head(vision(x)), torch.from_numpy(A[i][:, [vi, si, ci, di]])


def score():
    d = np.load(val_path); X, A, E = d['images'], d['aux'], d['episode_ids']; va = np.arange(len(X))
    vision.eval(); P, Y = [], []
    with torch.no_grad():
        for k in range(0, len(va), 256):
            p, y = run(X, A, E, va[k:k+256]); P.append(p); Y.append(y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); v = Y[:, 0] > .5; hit = P[:, 0] > 0
    side = (np.sign(P[v, 1]) == np.sign(Y[v, 1]))[np.abs(Y[v, 1]) > .1].mean()
    return f'recall {hit[v].mean():.3f} precision {(v & hit).sum()/max(hit.sum(), 1):.3f} correct side {side:.3f}'


print('start', score(), flush=True)
for epoch in range(epochs):
    for path in rng.permutation(paths):
        d = np.load(path); X, A, E = d['images'], d['aux'], d['episode_ids']; vision.train()
        o = rng.permutation(len(X))
        for k in range(0, len(o), 128):
            p, y = run(X, A, E, o[k:k+128]); vis = y[:, 0] > .5
            loss = F.binary_cross_entropy_with_logits(p[:, 0], vis.float())
            if vis.any():
                loss = loss+((p[vis, 1:]-y[vis, 1:])**2).mean()*4
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); del X, A, E, d
    print(f'epoch {epoch+1}', score(), flush=True)
    torch.save(dict(vision=vision.state_dict(), head=head.state_dict()), out)
