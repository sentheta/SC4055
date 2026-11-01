"""player-gif.py CHECKPOINT MAP STEPS OUT.gif SEED: record one player-view episode as a GIF, printing its outcome."""
import sys
import numpy as np
from PIL import Image
from corridor_qnn.full_environment import FullDoom
from corridor_qnn.player import PlayerView, load_player, act
ckpt, name, steps, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
model = load_player(ckpt)
env = FullDoom(map=name, max_steps=steps, auto_weapon=True, navigation='improved2')
view = PlayerView(env); view.reset(); obs = env.reset(seed=seed); frames = []; t = 0
while True:
    view.vector = view.update(obs)
    if t % 4 == 0:
        frames.append(Image.fromarray(np.asarray(obs.pixels).squeeze().astype(np.uint8)).resize((288, 192)).convert('P'))
    obs, _, done, info = env.step(act(model, view)); t += 1
    if done:
        break
env.close()
frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0)
print(seed, info['outcome'], info['steps'], len(frames))
