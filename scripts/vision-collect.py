"""vision-collect.py OUT.npz MAP STEPS CHECKPOINT SEED...: raw 120x160 RGB frames with enemy targets, half student half teacher driven."""
import sys
from multiprocessing import Pool
import numpy as np
from PIL import Image


def run(a):
    name, steps, ckpt, seed = a
    import torch
    torch.set_num_threads(1)
    from corridor_qnn.full_environment import FullDoom
    from corridor_qnn.full_teacher import teacher_action
    from corridor_qnn.player import AUX_INDEX, PlayerView, load_player, act
    model = load_player(ckpt); rng = np.random.default_rng(seed)
    env = FullDoom(map=name, max_steps=steps, auto_weapon=True, navigation='improved2')
    view = PlayerView(env); view.reset(); obs = env.reset(seed=seed); F, Y = [], []; t = 0
    while True:
        view.vector = view.update(obs)
        if t % 3 == 0:
            F.append(np.asarray(Image.fromarray(env.game.get_state().screen_buffer).resize((160, 120), Image.BILINEAR)))
            Y.append(obs.features[AUX_INDEX])
        action = teacher_action(obs.features) if rng.random() < .5 else act(model, view)
        obs, _, done, _ = env.step(action); t += 1
        if done:
            break
    env.close()
    return np.stack(F), np.stack(Y)


if __name__ == '__main__':
    out, name, steps, ckpt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    with Pool(8) as p:
        r = p.map(run, [(name, steps, ckpt, int(s)) for s in sys.argv[5:]], chunksize=1)
    np.savez(out, images=np.concatenate([x[0] for x in r]), aux=np.concatenate([x[1] for x in r]),
             episode=np.concatenate([np.full(len(x[0]), i) for i, x in enumerate(r)]))
    print('frames', sum(len(x[0]) for x in r))
