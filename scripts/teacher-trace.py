"""Trace one teacher episode: teacher-trace.py MAP STEPS SEED [EVERY] (env NAV=improved|improved2)."""
import os
import sys
from corridor_qnn.full_environment import FullDoom
from corridor_qnn.full_teacher import teacher_action

name, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 80
env = FullDoom(map=name, max_steps=steps, auto_weapon=True, navigation=os.environ.get('NAV', 'improved'))
nav = env.navigator
obs = env.reset(seed=seed)
i = 0
while True:
    if i % every == 0:
        pos = [round(v) for v in obs.context['position']]
        ctx = {k: v for k, v in obs.context.items() if k in ('health', 'armor', 'ammo', 'kills')}
        active = [(t.get('special'), t.get('sector'), sorted(t['targets']), t['pending'], t['until']) for t in nav.triggers if t.get('pending') is not None]
        print(i, pos, nav.mode, ctx, 'required', sorted(nav.required_cache.get((frozenset(nav.held), tuple(nav.dynamic_state)), [])), 'pending', active[:3], flush=True)
    obs, _, done, info = env.step(teacher_action(obs.features))
    i += 1
    if done:
        break
print(info)
env.close()
