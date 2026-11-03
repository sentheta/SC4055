#!/bin/bash
# GRU student on E1M2: DAgger until STOP, then a low-rate fine-tune of the newest round and a fresh-seed evaluation.
P=./.conda/bin/python; D=outputs/p/e1m2_rgb; export AUX=1 MEMORY=24
STOP=1500 START=15 KEEP=5 COLLECT_LIMIT=4000 BETA=0.5 JOBS=12 THREADS=10 LR=5e-4 E1=5 scripts/player-dagger.sh E1M2 8000 e1m2_rgb 60 8
last=$(ls -v $D/m[0-9]*.pt | tail -1)
LR=1.5e-4 THREADS=12 $P scripts/player-train.py $D/gru_polish.pt 4 quantum $last $(ls -v $D/d[0-9]*.npz | tail -n 5) 2>&1 | grep -E "EPOCH|Error" | tail -1 | cut -c1-110
echo "gru polish: $(JOBS=14 $P scripts/player-run.py eval $D/gru_polish.pt E1M2 8000 $(seq 900 919) 2>&1 | grep -E '^eval|Error') $(date +%H:%M)"
echo FINISHED
