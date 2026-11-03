"""player-deaths.py CHECKPOINT MAP STEPS SEED...: the student's last 40 steps before each death against the teacher's labels."""
import sys
from collections import Counter
from multiprocessing import Pool
import numpy as np


def run(a):
    ckpt, name, steps, seed = a
    import torch
    torch.set_num_threads(1); torch.manual_seed(seed)
    from corridor_qnn.full_actions import ACTION_NAMES
    from corridor_qnn.full_environment import FullDoom, BASE_FEATURES
    from corridor_qnn.full_teacher import teacher_action
    from corridor_qnn.player import PlayerView, load_player, act
    model = load_player(ckpt); env = FullDoom(map=name, max_steps=steps, auto_weapon=True, navigation='improved2')
    view = PlayerView(env, model.rgb); view.reset(); obs = env.reset(seed=seed); trace = []
    while True:
        view.vector = view.update(obs); f = obs.features
        s, t = act(model, view), teacher_action(f)
        trace.append((ACTION_NAMES[t], ACTION_NAMES[s], f[BASE_FEATURES.index('enemy_visible')] > .5, round(float(f[0])*100)))
        obs, _, done, info = env.step(s)
        if done:
            break
    env.close()
    return seed, info['outcome'], info['steps'], trace[-40:]


if __name__ == '__main__':
    ckpt, name, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(4) as p:
        rows = p.map(run, [(ckpt, name, steps, int(s)) for s in sys.argv[4:]], chunksize=1)
    pairs = Counter(); agree = vis = n = 0
    for seed, outcome, t, trace in rows:
        print(seed, outcome, t, 'health', [h for *_, h in trace[::8]])
        if outcome != 'death':
            continue
        for teacher, student, visible, _ in trace:
            n += 1; agree += teacher == student; vis += visible
            if teacher != student:
                pairs[(teacher, student)] += 1
    print(f'last 40 steps before deaths: agreement {agree/max(n, 1):.2f}, enemy visible {vis/max(n, 1):.2f}')
    for (teacher, student), c in pairs.most_common(10):
        print(f'  teacher {teacher:24s} student {student:24s} {c}')
