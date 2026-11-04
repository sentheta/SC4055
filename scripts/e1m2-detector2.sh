#!/bin/bash
# After the streamed enemy detector finishes, swap it into the round-36 student, weight combat frames, and continue DAgger.
P=./.conda/bin/python; V=outputs/p/e1m2_v; DET=outputs/p/vis/enemy_e1m2_big.pt
until grep -q "^epoch 3" outputs/p/vis_stream.log; do
  pgrep -f "[v]ision-pretrain-stream" >/dev/null || break
  $P -c "import time;time.sleep(30)"
done
[ -f $DET ] || DET=outputs/p/vis/enemy_e1m2.pt
echo "detector: $DET $(grep '^epoch' outputs/p/vis_stream.log | tail -1)"
export AUX=1 MEMORY=24 ENEMY=$DET ENEMY_WEIGHT=3
LR=4e-4 THREADS=14 $P scripts/player-train.py $V/m37.pt 5 quantum $V/m36.pt $(ls -v $V/d[0-9]*.npz | tail -n 5) 2>&1 | grep -E "EPOCH|Error" | tail -1 | cut -c1-110
echo "round 37 (new detector, combat weight 3): $($P scripts/player-run.py eval $V/m37.pt E1M2 8000 $(seq 800 819) 2>&1 | grep -E '^eval|Error') $(date +%H:%M)"
unset ENEMY
STOP=1545 START=38 KEEP=5 COLLECT_LIMIT=4000 BETA=0.5 JOBS=12 THREADS=10 LR=4e-4 E1=5 scripts/player-dagger.sh E1M2 8000 e1m2_v 80 8
last=$(ls -v $V/m[0-9]*.pt | tail -1)
echo "final $last on seeds 900-919: $(JOBS=14 $P scripts/player-run.py eval $last E1M2 8000 $(seq 900 919) 2>&1 | grep -E '^eval|Error') $(date +%H:%M)"
echo FINISHED $(date +%H:%M)
