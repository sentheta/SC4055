"""vision-pretrain.py OUT.pt EPOCHS DATA.npz...: train the student's CNN alone to locate enemies, on DAgger frames with auxiliary targets.

Prints held-out enemy detection and bearing error each epoch; saves the vision weights for a student to start from.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from corridor_qnn.player import PRIVILEGED, PlayerModel, stack_indices
torch.set_num_threads(6); torch.manual_seed(0); rng = np.random.default_rng(0)
out, epochs, paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
X, A, E = [], [], []; off = 0
for p in paths:
    d = np.load(p); X.append(d['images']); A.append(d['aux']); E.append(d['episode_ids']+off); off = E[-1].max()+1
X, A, E = np.concatenate(X), np.concatenate(A), np.concatenate(E)
vi, si, ci, di = (PRIVILEGED.index(n) for n in ('enemy_visible', 'enemy_sin', 'enemy_cos', 'enemy_distance'))
val = E >= E.max()-1; tr, va = np.flatnonzero(~val), np.flatnonzero(val)
model = PlayerModel(rgb=True, aux=True); vision = model.vision
head = torch.nn.Linear(64, 4)  # visible logit, bearing sin, bearing cos, distance
opt = torch.optim.Adam([*vision.parameters(), *head.parameters()], 1e-3)
print('frames', len(X), 'visible share', round((A[:, vi] > .5).mean(), 3), 'bearing sin std', round(A[A[:, vi] > .5, si].std(), 3), flush=True)


def run(i):
    x = torch.from_numpy(X[stack_indices(E, i)]).permute(0, 1, 4, 2, 3).flatten(1, 2).float()/255
    return head(vision(x)), torch.from_numpy(A[i][:, [vi, si, ci, di]])


for epoch in range(epochs):
    vision.train(); o = rng.permutation(tr)
    for k in range(0, len(o), 128):
        p, y = run(o[k:k+128]); vis = y[:, 0] > .5
        loss = F.binary_cross_entropy_with_logits(p[:, 0], vis.float())
        if vis.any():
            loss = loss+((p[vis, 1:]-y[vis, 1:])**2).mean()*4
        opt.zero_grad(); loss.backward(); opt.step()
    vision.eval(); P, Y = [], []
    with torch.no_grad():
        for k in range(0, len(va), 256):
            p, y = run(va[k:k+256]); P.append(p); Y.append(y)
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy(); v = Y[:, 0] > .5; hit = P[:, 0] > 0
    side = (np.sign(P[v, 1]) == np.sign(Y[v, 1]))[np.abs(Y[v, 1]) > .1].mean()
    print(f'epoch {epoch+1} recall {hit[v].mean():.3f} precision {(v & hit).sum()/max(hit.sum(), 1):.3f} sin_rmse {np.sqrt(((P[v, 1]-Y[v, 1])**2).mean()):.3f} correct side {side:.3f}', flush=True)
    torch.save(vision.state_dict(), out)
