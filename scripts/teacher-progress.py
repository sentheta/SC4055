"""Teacher progress across seeds: teacher-progress.py MAP STEPS SEED... (env NAV=improved|improved2)."""
import os
import sys
from multiprocessing import Pool


def run(args):
    name, steps, seed = args
    from corridor_qnn.full_environment import FullDoom
    from corridor_qnn.full_teacher import teacher_action
    env = FullDoom(map=name, max_steps=steps, auto_weapon=True, navigation=os.environ.get('NAV', 'improved'))
    nav = env.navigator
    obs = env.reset(seed=seed)
    modes = {}
    while True:
        modes[nav.mode] = modes.get(nav.mode, 0)+1
        position = [round(v) for v in obs.context['position']]
        obs, _, done, info = env.step(teacher_action(obs.features))
        if done:
            break
    env.close()
    return seed, info['outcome'], info['steps'], int(info['kills']), int(info['health']), position, sorted(nav.held), modes


if __name__ == '__main__':
    name, steps = sys.argv[1], int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3:]]
    with Pool(min(8, len(seeds))) as pool:
        rows = pool.map(run, [(name, steps, s) for s in seeds])
    for row in rows:
        print(*row)
    print('completed', sum(r[1] == 'completed' for r in rows), '/', len(rows), '| deaths', sum(r[1] == 'death' for r in rows))
