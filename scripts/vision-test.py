"""vision-test.py DATA.npz: can a CNN detect enemies from one frame? Compares input formats on held-out episodes."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from corridor_qnn.player import PRIVILEGED
torch.set_num_threads(14)
d = np.load(sys.argv[1]); X = d['images']; A = d['aux']; ep = d['episode']
vi, ci, si, di = (PRIVILEGED.index(n) for n in ('enemy_visible', 'enemy_cos', 'enemy_sin', 'enemy_distance'))
Y = A[:, [vi, si, di]]; val = ep >= ep.max()-1; tr = np.flatnonzero(~val); va = np.flatnonzero(val)
print('train', len(tr), 'val', len(va), 'visible share', (Y[va, 0] > .5).mean().round(3))


def prep(x, h, w, rgb):
    x = torch.from_numpy(x).float().permute(0, 3, 1, 2)/255
    if not rgb:
        x = x.mean(1, keepdim=True)
    return F.interpolate(x, size=(h, w), mode='area')


def net(ch, h, w, deep):
    layers = [nn.Conv2d(ch, 16, 5, 3), nn.ReLU(), nn.Conv2d(16, 32, 3, 2), nn.ReLU(), nn.Conv2d(32, 32, 3, 2), nn.ReLU()] if not deep else \
        [nn.Conv2d(ch, 24, 5, 2, 2), nn.ReLU(), nn.Conv2d(24, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 48, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(48, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveMaxPool2d((3, 4))]
    f = nn.Sequential(*layers, nn.Flatten())
    n = f(torch.zeros(1, ch, h, w)).shape[1]
    return nn.Sequential(f, nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 3))


for label, h, w, rgb, deep in (('64x96 gray old', 64, 96, False, False), ('64x96 rgb old', 64, 96, True, False), ('64x96 rgb deep', 64, 96, True, True), ('96x128 rgb deep', 96, 128, True, True), ('120x160 rgb deep', 120, 160, True, True)):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    m = net(3 if rgb else 1, h, w, deep); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for epoch in range(6):
        o = rng.permutation(tr)
        for k in range(0, len(o), 128):
            i = np.sort(o[k:k+128]); p = m(prep(X[i], h, w, rgb)); y = torch.from_numpy(Y[i])
            vis = y[:, 0] > .5
            loss = F.binary_cross_entropy_with_logits(p[:, 0], y[:, 0].clamp(0, 1))
            if vis.any():
                loss = loss+((p[vis, 1:]-y[vis, 1:])**2).mean()*4
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); P = []
    with torch.no_grad():
        for k in range(0, len(va), 256):
            P.append(m(prep(X[va[k:k+256]], h, w, rgb)))
    P = torch.cat(P).numpy(); y = Y[va]; v = y[:, 0] > .5; hit = P[:, 0] > 0
    prec = (v & hit).sum()/max(hit.sum(), 1)
    print(f'{label:18s} recall {hit[v].mean():.3f} precision {prec:.3f} sin_rmse {np.sqrt(((P[v, 1]-y[v, 1])**2).mean()):.3f} (std {y[v, 1].std():.3f}) params {sum(q.numel() for q in m.parameters())}', flush=True)
