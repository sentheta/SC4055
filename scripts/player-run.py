"""Parallel episodes with the player-view student.

  player-run.py collect OUT.npz MAP STEPS BETA CHECKPOINT|none SEED...   DAgger data: teacher labels every decision;
                                                                         it also acts with probability BETA.
  player-run.py eval CHECKPOINT MAP STEPS SEED...                        student acts alone.

Env: JOBS (default 10), TEMP>0 samples eval actions, RGB=1 records colour frames, COLLECT_LIMIT stops collection early, NO_QUANTUM=1 zeroes the quantum gain, PENNYLANE=1 runs the circuit in PennyLane.
"""
import os
import sys
from multiprocessing import Pool
import numpy as np


def run(args):
    mode, ckpt, name, steps, beta, seed = args
    import torch
    torch.set_num_threads(1); torch.manual_seed(seed)
    from corridor_qnn.full_environment import FullDoom
    from corridor_qnn.full_teacher import teacher_action
    from corridor_qnn.player import AUX_INDEX, PlayerView, load_player, act
    model = load_player(ckpt) if ckpt != 'none' else None
    if model is not None and os.environ.get('NO_QUANTUM') == '1':
        with torch.no_grad():
            model.quantum_gain.zero_()
    if model is not None and os.environ.get('PENNYLANE') == '1':
        import importlib.util
        spec = importlib.util.spec_from_file_location('plcheck', os.path.join(os.path.dirname(__file__), 'pennylane-check.py'))
        pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
        model = pl.to_pennylane(model)
    env = FullDoom(map=name, max_steps=steps, auto_weapon=True, navigation='improved2')
    rgb = model.rgb if model is not None else os.environ.get('RGB') == '1'
    view = PlayerView(env, rgb); view.reset()
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    images, features, labels, aux = [], [], [], []
    while True:
        view.vector = view.update(obs)
        if mode == 'collect':
            label = teacher_action(obs.features)
            images.append(view.current); features.append(view.vector); labels.append(label); aux.append(obs.features[AUX_INDEX])
            action = label if model is None or rng.random() < beta else act(model, view)
        else:
            action = act(model, view, float(os.environ.get('TEMP', 0)))
        obs, _, done, info = env.step(action)
        if done or (mode == 'collect' and len(labels) >= int(os.environ.get('COLLECT_LIMIT', 10**9))):
            break
    env.close()
    row = (seed, info['outcome'], info['steps'], int(info['kills']), info['coverage_cells'], sorted(view.env.navigator.held))
    return row, (np.stack(images), np.stack(features), np.asarray(labels), np.stack(aux, dtype=np.float32)) if mode == 'collect' else None


def main():
    mode = sys.argv[1]
    jobs = int(os.environ.get('JOBS', 10))
    if mode == 'collect':
        out, name, steps, beta, ckpt = sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
        seeds = [int(s) for s in sys.argv[7:]]
    else:
        ckpt, name, steps, out, beta = sys.argv[2], sys.argv[3], int(sys.argv[4]), None, 0.
        seeds = [int(s) for s in sys.argv[5:]]
    with Pool(min(jobs, len(seeds))) as pool:
        results = pool.map(run, [(mode, ckpt, name, steps, beta, s) for s in seeds], chunksize=1)
    rows = [r[0] for r in results]
    if os.environ.get('VERBOSE'):
        for row in rows:
            print(*row)
    print(mode, name, 'completed', sum(r[1] == 'completed' for r in rows), '/', len(rows), '| deaths',
          sum(r[1] == 'death' for r in rows), '| mean cells', round(np.mean([r[4] for r in rows])), '| keys',
          sum(bool(r[5]) for r in rows), flush=True)
    if mode == 'collect':
        np.savez_compressed(out, images=np.concatenate([r[1][0] for r in results]),
                            features=np.concatenate([r[1][1] for r in results]),
                            labels=np.concatenate([r[1][2] for r in results]),
                            aux=np.concatenate([r[1][3] for r in results]),
                            episode_ids=np.concatenate([np.full(len(r[1][2]), i) for i, r in enumerate(results)]))


if __name__ == '__main__':
    main()
