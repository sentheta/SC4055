"""Parallel evaluation: parallel-eval.py CHECKPOINT MAP STEPS AUTO(0/1) SEED...

Env: NAV=improved|improved2, NO_QUANTUM=1 zeroes the quantum gain (ablation).
Prints one line per episode and a final completion count.
"""
import os
import sys
from multiprocessing import Pool


def run(args):
    ckpt, name, steps, auto, seed = args
    import torch
    torch.set_num_threads(1)
    from corridor_qnn.full_environment import FullDoom
    from corridor_qnn.full_models import load_full
    from corridor_qnn.training import infer
    model = load_full(ckpt)
    if os.environ.get('NO_QUANTUM') == '1':
        with torch.no_grad():
            model.quantum_gain.zero_()
    env = FullDoom(map=name, max_steps=steps, auto_weapon=auto, navigation=os.environ.get('NAV', 'improved'))
    nav = env.navigator
    obs = env.reset(seed=seed)
    keys = {}
    while True:
        for c in nav.held:
            keys.setdefault(c, env.steps)
        obs, _, done, info = env.step(infer(model, obs))
        if done:
            break
    env.close()
    return seed, info['outcome'], info['steps'], info['kills'], keys


if __name__ == '__main__':
    ckpt, name, steps, auto = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == '1'
    seeds = [int(s) for s in sys.argv[5:]]
    with Pool(min(10, len(seeds))) as pool:
        rows = pool.map(run, [(ckpt, name, steps, auto, s) for s in seeds])
    for row in rows:
        print(*row)
    print('completed', sum(r[1] == 'completed' for r in rows), '/', len(rows),
          '| deaths', sum(r[1] == 'death' for r in rows))
