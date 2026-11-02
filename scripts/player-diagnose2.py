"""player-diagnose2.py CHECKPOINT MAP STEPS SEED...: fresh student-driven episodes, teacher labels; recall by enemy visibility, aux error, position drift."""
import sys
import numpy as np
import torch
from corridor_qnn.full_environment import FullDoom, BASE_FEATURES
from corridor_qnn.full_teacher import teacher_action
from corridor_qnn.player import AUX_INDEX, PRIVILEGED, PlayerView, load_player
ckpt, name, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = load_player(ckpt); torch.set_num_threads(4)
rows, aux_p, aux_t, drift = [], [], [], []
for seed in map(int, sys.argv[4:]):
    env = FullDoom(map=name, max_steps=steps, auto_weapon=True, navigation='improved2')
    view = PlayerView(env); view.reset(); obs = env.reset(seed=seed); start = None; t = 0
    while True:
        view.vector = view.update(obs)
        with torch.no_grad():
            logits, a = model(torch.from_numpy(view.stack()).unsqueeze(0), torch.from_numpy(view.vector).unsqueeze(0), return_aux=True)
        action = int(logits.argmax()); label = teacher_action(obs.features)
        visible = obs.features[BASE_FEATURES.index('enemy_visible')] > .5
        rows.append((label, action, visible)); aux_p.append(a[0].numpy()); aux_t.append(obs.features[AUX_INDEX])
        if t == 0:
            print('state attrs', [k for k in dir(env) if 'pos' in k.lower() or k in ('game', 'state')])
        obs, _, done, info = env.step(action); t += 1
        if done:
            break
    print(seed, info['outcome'], t, 'cells', info['coverage_cells'], flush=True); env.close()
rows = np.array(rows); ap, at = np.array(aux_p), np.array(aux_t)
for v in (0, 1):
    m = rows[:, 2] == v
    print('enemy_visible', v, 'n', m.sum(), 'agreement', round((rows[m, 0] == rows[m, 1]).mean(), 3))
for i, n in enumerate(PRIVILEGED[:13]):
    print(f'{n:22s} target std {at[:, i].std():.3f} rmse {np.sqrt(((ap[:, i]-at[:, i])**2).mean()):.3f}')
vis = at[:, PRIVILEGED.index('enemy_visible')] > .5
print('visible detection: recall', round((ap[vis, PRIVILEGED.index('enemy_visible')] > .5).mean(), 3), 'false alarm', round((ap[~vis, PRIVILEGED.index('enemy_visible')] > .5).mean(), 3))
