#!/bin/bash
# Pretrain an enemy detector on recent E1M2 frames, add it frozen beside the GRU student's own CNN, then continue DAgger.
P=./.conda/bin/python; D=outputs/p/e1m2_rgb; V=outputs/p/e1m2_v; DET=outputs/p/vis/enemy_e1m2.pt
PYTHONPATH=. $P scripts/vision-pretrain.py $DET 12 $(ls -v $D/d[0-9]*.npz | tail -n 4) 2>&1 | grep -E "^epoch|^frames|Error"
export AUX=1 MEMORY=24 ENEMY=$DET
rm -rf $V; mkdir -p $V; for f in $(ls -v $D/d[0-9]*.npz | tail -n 5); do ln -sf ../e1m2_rgb/$(basename $f) $V/; done
last=$(ls -v $D/d[0-9]*.npz | tail -1 | sed 's/.*d\([0-9]*\).npz/\1/')
LR=5e-4 THREADS=12 $P scripts/player-train.py $V/m$last.pt 6 quantum $D/m$last.pt $(ls -v $V/d[0-9]*.npz) 2>&1 | grep -E "EPOCH|Error" | tail -1 | cut -c1-110
echo "round $last (enemy detector): $(JOBS=14 $P scripts/player-run.py eval $V/m$last.pt E1M2 8000 $(seq 800 819) 2>&1 | grep -E '^eval|Error') $(date +%H:%M)"
STOP=1530 START=$((last+1)) KEEP=5 COLLECT_LIMIT=4000 BETA=0.5 JOBS=12 THREADS=10 LR=4e-4 E1=5 scripts/player-dagger.sh E1M2 8000 e1m2_v 60 8
echo FINISHED $(date +%H:%M)
