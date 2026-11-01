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

### DAgger round 1 and two hidden bugs (23:00-00:30)

`e1m2_d1` (DAgger: 50k student-driven + 50k teacher decisions, retrained)
scored 0/20 although its in-training validation said 2/5. Tracing showed the
student turning left/right forever on the spot, and the teacher agreeing 99%:

1. **Turn oscillation in the teacher.** One turn action rotates up to ~28
   degrees, but the waypoint dead zone was +/-0.16 rad (9 degrees). When a
   waypoint fell between two reachable headings the teacher swung back and
   forth forever. Dead zone widened to +/-0.25 rad (more than half a step).
   This was also the cause of most remaining *teacher* timeouts.
2. **Empty planner path without a goal point** could leave the waypoint on
   the player. The planner now falls back to an exploration goal.

Teacher after the turn fix: **26/32** (was 20/32), no timeouts; E1M1 and
MAP01 still 20/20. Because the student copied the oscillation, every
student must be retrained with the fixed teacher (`e1m2_v3`, 100k
decisions).

Lesson: a 3-5 episode in-training validation is not a reliable checkpoint
signal on long levels; always evaluate saved checkpoints on 20+ seeds.

### Teacher reaches 94% (01:13)

With the turn fix in place, the earlier "heal before pushing on" variant
was re-tested and is now decisive: health goals start below 90 (was 70),
with unlimited detours below 70.

| Teacher | Seeds 800-831 | Seeds 832-863 | Total |
|---|---|---|---|
| Health threshold 70 | 26/32 | 23/32 | 49/64 (77%) |
| Health threshold 90 | 30/32 | 30/32 | **60/64 (94%)** |

Remaining teacher failures: 3 deaths, 1 timeout in 64. Students are
retrained with this teacher (`e1m2_v4`).

### E1M2 target reached: QNN `e1m2_v3` 94% (01:55)

`e1m2_v3`: 100k decisions from the turn-fixed teacher (21 completed, 3
deaths, 1 cut-off episode), corridor initialisation, 12 epochs, epoch 12
selected. Evaluated with the current planner (health threshold 90), which
the student follows through its waypoint features:

| Seeds | Completed | Deaths | Timeouts |
|---|---|---|---|
| 800-819 | 19/20 | 0 | 1 |
| 820-869 | 47/50 | 2 | 1 |
| **Total** | **66/70 (94%)** | 2 | 2 |

PennyLane parity: circuit outputs within 6.6e-7, argmax action identical on
2000 recorded states. Full torch and PennyLane gameplay re-runs and
artifacts: `results/levels/E1M2/e1m2_v3/`.

Survey of later maps with this teacher (4 seeds each, 8000 steps): E1M4,
E1M5, E1M6 and E1M8 end in deaths (combat-heavy, 20-60 kills), E1M7 and
E1M9 time out with ~2-9 kills (navigation). Combat is the common bottleneck
after E1M2.

### DAgger round 2: `e1m2_d2` (03:45-04:20)

A further 100 seeds put `e1m2_v3` at 87/100, so 153/170 (90.0%) overall:
just at the target. One DAgger round to add margin:

- 4 parallel student-driven collections from `e1m2_v3` (seeds 1000-4000,
  12.5k decisions each, teacher labels), merged with v3's 100k teacher
  decisions (150k samples, 36 episodes).
- Fine-tuned from `e1m2_v3` (lr 2e-4, 3 epochs) instead of retraining.

| Seeds 870-969 | Completed | Deaths |
|---|---|---|
| e1m2_v3 | 87/100 | 8 |
| e1m2_d2 | **91/100** | 2 |

On seeds 800-869 `e1m2_d2` scores 61/70 (v3: 66/70), so overall **152/170
(89.4%)** against v3's 153/170 (90.0%): no measurable gain. Its torch and
PennyLane re-runs (seeds 800-819) agree exactly (17/20 each, 20/20 episodes
identical). `e1m2_v3` remains the exported E1M2 model.

### Does the circuit matter? (23:45, 30 Oct)

Ablation of `e1m2_v3` with the quantum gain set to zero, seeds 800-869:
**64/70, against 66/70 with the circuit on.** No measurable contribution:
the classical residual path makes essentially all decisions. The 90% is a
result for the hybrid system, not evidence that the circuit helps.

Literature consulted: trainable-encoding re-uploading (arXiv 2606.25598),
component evaluation of quantum RL (2511.17112), classical surrogates of
quantum policies (2609.16266). New opt-in options: `--reupload`, `--zz`
(all ZZ correlators), `--no-residual` (policy from measurements only).
All compared on v3's 100k decisions, same initialisation, 12 epochs, seeds
800-869:

| Model | Completed |
| --- | --- |
| `e1m2_v3` (baseline, quantum on) | 66/70 |
| `e1m2_v3`, quantum gain 0 | 64/70 |
| classical, retrained | 62/70 |
| quantum, re-uploading, 4 layers | 61/70 |
| quantum, re-uploading, 2 layers | 59/70 |
| classical, ZZ, no residual (control) | 58/70 |
| quantum, re-uploading + ZZ, no residual | 53/70 |
| quantum, re-uploading, no residual | 50/70 |

No new circuit beat the baseline. These are single training runs, so gaps of a
few episodes are within noise, including the baseline's lead over the retrained
models. The one clear signal is that removing the residual hurt the quantum
models (50-53/70 against 59-61/70 with it). The classical no-residual control
reached 58/70, so the circuit-only policy trails its classical equivalent.

### E1M3: why every seed died (01:50, 31 Oct)

Correction: the north nukage room is on the route, not off it, as noted earlier.
The real order, read from the WAD:

1. Lift 91 (switches at (-288, 2456) and (-288, 2336)) climbs out of the nukage
   onto ledge sector 92.
2. Walk lines at x=480 lower lifts 138/139, giving access to the east side.
3. Switch (1052, 2612), tag 9, lowers pedestal sector 117 under the blue key at
   (432, 2608).
4. Blue door (-1136, 104) opens the west region: switch (-2076, 2424) raises
   sector 146.
5. Switch (-88, 896) lowers door 236; exit at (284, 1483).

Faults found by tracing, all fixed behind `improved2`:

- Missed presses counted as presses: a sector that has not moved 8 calls after
  the press now means a retry.
- Momentum carried the player off small lifts: `HOLD_INTERACTION` makes the
  teacher brake and stand still, with a 60-call limit for lifts that stay down.
- Path smoothing cut across pits: lookahead may not cross damaging cells, and
  routes pay extra beside them.
- Charging enemies and running beside pits walked the player off ledges:
  `LEDGE_INTERACTION` stops the charge and caps speed.
- Backward planning chased the exit while the key was missing, and its cache
  ignored the player's region: missing keys and their pedestals now come first.
- A far medkit outranked the key: long health detours now wait for HP below 30.

Teacher, seeds 800-807: still 0/8 completed, but deaths fell from 8 to 4 and
some seeds now reach the east side. No key pickup yet.

## E1M3 (started 00:20, 30 Oct)

One blue key, 42 triggers, 15 lifts, 15 damaging sectors (no radiation
suit). The first exit is only reachable across nukage.

| Change | Teacher result (6-8 seeds) |
|---|---|
| E1M2 planner as is | dies at ~330 steps in a nukage pit by the start |
| Avoid damaging floors (blocked) | survives 600-2300 steps, still 0/6 |
| Weighted crossing (cost 10), no drops into damaging pits, never a destination | 0/8; ~95% of damage is taken on nukage in the large north room |

Further E1M3 work, gated behind the opt-in `--navigation improved2` so that
the E1M2 checkpoint keeps the exact waypoints it was trained on (un-gated,
these changes dropped `e1m2_v3` from 19/20 to 15/20):

| Change (improved2) | Finding |
|---|---|
| Damage split per death | 90-130 of ~130 damage taken on nukage, mostly in trigger/explore modes |
| Squared-up switch presses (aim through the wall within 72 units) | presses on the pit-lift switches now register |
| Step off nukage when idle; explore without crossing it | idle damage reduced |
| Plan from the nearest free cell when the clearance raster blocks the player's own cell | removed a "no path, stand in nukage" state in a narrow channel |
| Retry escape triggers when trapped on nukage; switches for the same lift share a cooldown | the second switch no longer sends the lift back up |
| Board a lowered lift right after pressing its switch | 2/8 escape the pit and reach the east area |

E1M3 status at 03:20: teacher 0/8 completions. The pit (a lift inside a
nukage pool) is now escapable; the survivors then explore the east area for
thousands of steps without finding the mechanism that opens the way to the
exit. No student was trained for E1M3, since the method needs a teacher
that completes the level.

## Summary (03:20, 30 Oct)

| Map | Teacher | QNN student | Torch vs PennyLane | Artifacts |
|---|---|---|---|---|
| E1M1 | 20/20 | 49/50 (`e1m1_v4`) | 50/50 episodes identical | `results/levels/E1M1/e1m1_v4/` |
| E1M2 | 60/64 (94%) | **153/170 (90.0%)** (`e1m2_v3`; 66/70 on seeds 800-869, 87/100 on 870-969) | 20/20 episodes identical | `results/levels/E1M2/e1m2_v3/` |
| E1M3 | 0/8 | not trained | - | - |
| E1M4-E1M9 | 0/4 each (survey) | not trained | - | - |

What moved E1M2 from 3/10 to 94%, in order of impact:

1. Teacher turn oscillation fix (waypoint dead zone wider than half a turn).
2. Heal-to-90 before pushing on (fixed ambushes).
3. Trigger verification: a press counts only when a target sector moves.
4. Automatic shotgun selection (removes a rare action label).
5. 100k teacher decisions instead of 50k.

What did not help: hand-tuned combat variants (strafe-fire, retreat when
hurt, shell conservation, ammo seeking, always-run), and DAgger round 1 on
a teacher that still oscillated.

Main lessons:

- Most "student failures" were planner or teacher bugs. The student follows
  the planner's waypoints, so planner fixes can raise a trained student's
  completion without retraining, and planner changes can also silently
  break a trained student; new planner behaviour must be opt-in.
- 3-5 validation episodes cannot select checkpoints on long levels; saved
  checkpoints were evaluated on 20-70 seeds.
- Evaluation must use the training step limit: the time feature is
  `steps / max_steps`.
- The bottleneck beyond E1M2 is combat on combat-heavy maps and map-specific
  mechanisms (E1M3's nukage pit and east area).

### E1M3 results (04:35, 31 Oct): first QNN, below the 90% target

Further teacher faults fixed after the list above: gaps sealed by wall inflation
(a channel is now carved across short walkable lines), untagged stair sectors
(now followed with Doom's stair-building rule), spent one-shot switches, distant
lifts (a direct run after the press), health detours while standing in nukage,
and charging enemies while riding a lift. The level finishes through the exit at
(-2196, 2264), behind the lift called by switch (-1856, 2472). That corrects
steps 4-5 of the route above: switch (-88, 896) is not needed on this route, and
sector 146 was not seen to move in the run traced.

| Agent | Step limit | Seeds | Completed |
| --- | --- | --- | --- |
| Teacher | 8000 | 800-839 | 24/40 (60%) |
| Teacher | 16000 | 800-839 | 32/40 (80%) |
| QNN `e1m3_v1` | 8000 | 800-839 | 18/40 (45%) |
| QNN `e1m3_v1`, torch | 8000 | 800-819 | 10/20 |
| QNN `e1m3_v1`, PennyLane | 8000 | 800-819 | 10/20, same outcome and steps per seed |
| QNN `e1m3_v2` (later teacher, 16000 limit) | 16000 | 800-839 | 17/40 (43%) |

Parity on `e1m3_v1`: 100% action agreement over 2000 game states.

`e1m3_v1` is exported to `results/levels/E1M3/e1m3_v1/`. It was trained before
the lift-fight teacher rule. The teacher's 75% on seeds 800-815 did not hold up
on 40 seeds.

Rejected, each measured on seeds 800-815 or 800-807: doubling cooldowns for
repeatable triggers (1/16) and speed-based braking (3/16, against 9/16 without).

Regressions noticed: `scripts/pennylane-check.py` crashed for every model from the
`--zz` change until now, because unused ZZ measurements were queued. Fixed; results
from before that change stand, and `e1m2_v3` parity was rerun and passes. A stray
`cat > /dev/null` also hung one test for about 90 minutes.

Not checked: the `improved2` changes have not been re-run on E1M1 or E1M2. Models
on `improved` navigation are unaffected by the planner gates, but the shared teacher
rules only fire on the `improved2` signals, and no test covers them.

Open: the teacher misses the target even at 16000 steps, and the student loses
about 15 to 37 points against it. E1M4 onward has not been started.

### Final checks (31 Oct)

Raw output, seeds as run:

```
E1M1 improved2 teacher completed 8 / 8 | deaths 0
e1m3_v1 gain0 completed 0 / 40 | deaths 6
e1m2_v3 (improved navigation, seeds 800-819): completed 19 / 20 | deaths 0
E1M2 improved2 teacher (seeds 800-807): completed 8 / 8 | deaths 0
E1M4 improved2 teacher (seeds 800-807): completed 0 / 8 | deaths 8
```

The gain-0 line is `e1m3_v1` on seeds 800-839, to compare with 18/40 with the circuit on. These checks replace the "not checked" note above for the maps listed. E1M4 dies with 15-53 kills, half of the seeds holding the yellow key; no planner work has been done on it.

The E1M3 gain-0 result was cross-checked with `full-eval --quantum-disabled` on
seeds 801, 805, 807, 811 and 815. All five complete with the circuit on and none
completes with it off, so the two ablation routes agree. This model depends on its
circuit, unlike `e1m2_v3` (64/70 against 66/70), with a similar learned gain (2.25
against 2.23). It does not show a quantum advantage: removing a trained component
says nothing about what a classical model trained on the same data would reach.
That comparison has not been run on E1M3 and is the next experiment to do.

## Player-view student (no privileged inputs)

The student no longer receives planner, engine-label or depth features. Inputs: 3 stacked frames, HUD values, its own motion, dead-reckoned position from the start, recent displacement, USE count, cell revisits and last action. Two caveats: the key flags are read from `navigator.held`, which is equivalent to the HUD, and the map is not given, so it must be learned from experience on that level. The privileged teacher still labels every decision (DAgger).

E1M1 (`outputs/p/e1m1_a`): 0/20 in rounds 0-2, then 7, 14, 16, 13, 12, 19, 18, 15, 18, 16 of 20 in rounds 3-12 on seeds 800-819. These round scores are noisy. Fine-tuning on all data (15 epochs, LR 2e-4) gave 36/40 on seeds 900-939, which were not used to compare rounds; m12 got 32/40 and m8, the best round on 800-819, got 21/40.

E1M2 (`outputs/p/e1m2_a`): 0/20 in all 12 rounds, with 9-16 deaths at about 60 cells. Raising BETA from 0.3 to 0.7 restored collection coverage (about 440 cells) but not the student's score. Next attempt (`e1m2_b`): a training-only auxiliary head on the CNN predicting enemy, waypoint and depth features. These are targets, never inputs.

E1M1 checks on seeds 900-939: quantum 36/40, classical trained from scratch on the same data 33/40, quantum with gain zeroed 34/40. No quantum advantage is shown. PennyLane matches torch at 17/20 on seeds 900-919, with logits within 4e-6. Exported to `results/player/E1M1`.

E1M2 `e1m2_b` with the auxiliary head: 0/20 in rounds 12-15 and after fine-tuning on seeds 900-919, with 9-13 deaths and the remaining episodes timing out at low coverage (about 65 cells). Sampling actions did not help: T=0.5 0/20 (12 deaths), T=1 0/20 (18 deaths).

E1M2 larger model (hidden 512, from scratch on all data, auxiliary head): 0/20 with 8 deaths on seeds 900-919. After a low-rate fine-tune: big2: eval E1M2 completed 0 / 20 | deaths 13 | mean cells 88 | keys 6 06:38.

Where this stands: E1M1 meets the 90% target from the player view. E1M2 does not, at 0/20 across every variant tried, and E1M3 onward were not attempted because they depend on E1M2 working. Held-out accuracy stays near 70% and the student either dies early or wanders at about 65 cells of the roughly 450 the teacher covers. The student has no recurrent memory and only a 64x96 view. Untried next steps: a GRU, higher resolution, and much more data from the early part of the level.

Larger E1M1 check on new seeds 1000-1079:
- model: eval E1M1 completed 74 / 80 | deaths 1 | mean cells 63 | keys 0 06:42
- classical: eval E1M1 completed 64 / 80 | deaths 5 | mean cells 62 | keys 0 06:45
