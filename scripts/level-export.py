"""Copy a level run's deliverables into the tracked results tree.

  python scripts/level-export.py MAP RUN_DIR TORCH_EVAL_JSON [PENNYLANE_EVAL_JSON] [GIF ...]

Writes results/levels/<MAP>/<run>/ with the checkpoint, the training history
(loss, held-out accuracy, validation gameplay), both evaluation summaries,
GIFs and a README.md table comparing the two circuit backends.
"""
import json
import shutil
import sys
from pathlib import Path

name, run = sys.argv[1], Path(sys.argv[2])
torch_eval = Path(sys.argv[3])
rest = [Path(p) for p in sys.argv[4:]]
pennylane_eval = next((p for p in rest if p.suffix == '.json'), None)
gifs = [p for p in rest if p.suffix == '.gif']

out = Path('results/levels') / name / run.name
out.mkdir(parents=True, exist_ok=True)
shutil.copy2(run / 'quantum.pt', out / 'quantum.pt')
training = next(run.glob('training_quantum_*.json'))
shutil.copy2(training, out / 'training.json')
shutil.copy2(torch_eval, out / 'eval_torch.json')
if pennylane_eval:
    shutil.copy2(pennylane_eval, out / 'eval_pennylane.json')
for gif in gifs:
    shutil.copy2(gif, out / gif.name)

meta = json.loads(training.read_text())
evals = {'torch': json.loads(torch_eval.read_text())}
if pennylane_eval:
    evals['pennylane'] = json.loads(pennylane_eval.read_text())
lines = [f'# {name} / {run.name}', '',
         f"Training: {meta.get('training')}, {meta.get('samples')} samples, "
         f"selected epoch {meta.get('selected_epoch')}, auto_weapon={meta.get('auto_weapon')}, "
         f"max_steps={meta.get('max_steps')}.", '',
         '| Epoch | Loss | Held-out accuracy | Validation completion |', '|---:|---:|---:|---:|']
for row in meta['history']:
    gameplay = row.get('gameplay', {}).get('completion_rate', '')
    lines.append(f"| {row['epoch']} | {row['loss']:.4f} | {row['episode_heldout_accuracy']:.3f} | {gameplay} |")
lines += ['', '| Backend | Episodes | Completion | Deaths | Mean kills | Mean steps | Inference s |',
          '|---|---:|---:|---:|---:|---:|---:|']
for backend, data in evals.items():
    s = data['summary']
    lines.append(f"| {backend} | {s['episodes']} | {s['completion_rate']:.2f} | {s['death_rate']:.2f} | "
                 f"{s['mean_kills']:.1f} | {s['mean_steps']:.0f} | {s['inference_seconds']:.0f} |")
if pennylane_eval:
    a, b = evals['torch']['episodes'], evals['pennylane']['episodes']
    same = sum(x['outcome'] == y['outcome'] and x['steps'] == y['steps'] and x['kills'] == y['kills']
               for x, y in zip(a, b))
    lines += ['', f'Identical episodes (outcome, steps, kills): {same}/{min(len(a), len(b))}.']
lines += ['', 'GIFs: ' + (', '.join(g.name for g in gifs) or 'none')]
(out / 'README.md').write_text('\n'.join(lines) + '\n')
print('exported', out)
