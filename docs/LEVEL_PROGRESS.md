# Level progress log (Freedoom Phase 1, E1Mx)

Goal: the hybrid QNN student completes each Freedoom E1 map in at least 90% of
evaluation episodes, starting with E1M2, then E1M3 onwards. Each checkpoint is
evaluated with the torch state-vector simulator and re-run with the circuit
executed by PennyLane as verification. Artifacts for each kept run live in
`results/levels/<MAP>/<run>/`: weights, training history (loss, held-out
accuracy, validation gameplay), evaluation JSON for both backends and GIFs.

All results use skill 1, frame skip 4, privileged planner features, and the
bundled `freedoom1.wad` (not id Software's Doom). Completion is the engine's
native map-exit event only.

## Method (unchanged from E1M1)

1. A rule-based teacher follows waypoints from a privileged IWAD planner
   (`corridor_qnn/full_navigation.py`) and fights with simple combat rules
   (`corridor_qnn/full_teacher.py`).
2. The 6-qubit, 2-layer hybrid (`FullActorCritic`, 70k parameters, 24 circuit
   parameters) is initialised from `pretrained/quantum_skill1_baseline.pt`
   and trained by behaviour cloning on teacher-labelled decisions
   (`full-train`), checkpoint selected by short validation gameplay.
3. Evaluation: `full-eval` (torch) and `scripts/pennylane-check.py eval`
   (PennyLane `default.qubit`, same weights), compared episode by episode.

The student sees the planner's waypoint, so planner fixes help a trained
student without retraining. The time feature is `steps / max_steps`, so
evaluation must use the training step limit.

## E1M1 (done before this log)

QNN `e1m1_v4`: 49/50 completions (seeds 800-849, 1500 steps). PennyLane
re-run: 50/50 episodes identical. Fixes that mattered: USE no longer reverses
opening doors; the teacher skips distant/ammo-less fights and switches only to
weapons with ammo.

## E1M2

### Planner and teacher work (commits 8895e7c and later)

E1M2 needs three keys, several lifts, switch-lowered platforms, remote doors
and monster closets. The original planner treated every moving sector as a
static wall. Added, in order of discovery:

| Problem observed | Fix |
|---|---|
| Teacher pushes on the red-key door forever | Key-locked doors blocked until the key is touched |
| Lifts/platforms look like walls | Moving sectors use live engine heights |
| Exit sealed; nothing to do | Switches, walk lines, teleporters become subgoals |
| Wrong triggers chosen, back-and-forth | One-step lookahead: open a trigger's targets, score keys/exit/new ground |
| Rides lift down again | Ride only when boarded from below, stand inside trigger ring |
| Stuck on a raised box | One-way ledge drops (static and live-height) |
| Lamps/barrels block corridors | Solid decorations rasterized |
| Red key on a 32-unit ledge | Items touched within body height: key goal radius 56 |
| Ambush deaths | Health detours scaled by damage; back off firing in melee range |
| Student keeps the pistol | `--auto-weapon`: environment holds the shotgun when it has shells |
| A missed switch press counted as done forever | Trigger only counts once a target sector moves; else retried |

### Results so far

| Run | Teacher (16 seeds) | QNN student | Notes |
|---|---|---|---|
| e1m2_v1 | 12/16 | 3/10 | 50k decisions, 8000 steps |
| e1m2_v2 (+auto-weapon) | 12/16 | 1/10 | deaths later, many timeouts |
| e1m2_v2 + trigger verification (no retrain) | 12/16, 0 timeouts | 7/10; 11/20 | 1 death in 20 |

PennyLane check of e1m2_v1: 10/10 episodes identical, circuit within 6e-7.

### Student failure analysis (22:40)

Replaying e1m2_v2 on failing seeds: in 5/6 timeouts the student agreed with
the teacher 95-100% while the *planner* looped; most were fixed by trigger
verification. The remaining stalls are absorbing states: the student stands
still pressing ATTACK for thousands of steps (inputs no longer change, so the
output does not either) while the teacher's label is "turn". Pure behaviour
cloning never sees these states, so the fix is DAgger.

### Teacher combat experiments (32 seeds, 800-831, 8000 steps)

Baseline teacher: 20/32 complete, 11 deaths. Deaths cluster in two fixed
encounters: the room behind door 266 (before the red key, 4) and the
approach to the yellow key (after the red key, 6). Most are close fights with
demons while holding the pistol (shells spent).

| Variant | Completed | Deaths | Kept? |
|---|---|---|---|
| Baseline | 20 | 11 | yes |
| Melee back-off at 200 instead of 128 units (16 seeds) | 9/16 | 7 | no |
| A: strafe-fire when aimed | 17 | 12 | no |
| B: back off firing when health < 40 | 13 | 17 | no |
| A+B | ~8 | 22 | no |
| C: shotgun only within 300 units | 18 | 12 | no |
| D: heal to 90 before pushing on, long detours < 70 | 21 | 9 | no (noise) |
| Ammo seeking (16 seeds) | 7/16 | 8 | no: detours walk into fights |
| Always run (`cl_run`, then SPEED on moves) | 0 | 0 | no: turning breaks |

Hand-tuned combat rules have plateaued around 60-65% teacher completion.
The 16-seed estimate of 75% was optimistic; 32 seeds are needed to compare.

### Open issues (22:36, 29 Oct)

- 3/20 student seeds stall after the blue key with ~20 kills (planner loop).
- 5/20 have the red key but run out of time (health/trigger detours).
- Teacher dies in 4/16 (ambushes).
- Evaluating at a longer step limit changes behaviour (time feature).
