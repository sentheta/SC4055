#!/bin/bash
# player-dagger.sh MAP STEPS RUN ROUNDS EPISODES [KIND]: DAgger for the player-view student. Env BETA, START, EVALS.
# Training seeds start at 2000; evaluation seeds 800+ are never trained on.
MAP=$1; STEPS=$2; RUN=$3; ROUNDS=$4; EP=$5; KIND=${6:-quantum}; # env RGB, COLLECT_LIMIT, AUX pass through
 BETA=${BETA:-0.3}; EVALS=${EVALS:-20}
P=./.conda/bin/python; D=outputs/p/$RUN; mkdir -p $D
EVAL="$(seq 800 $((799+EVALS)))"
for r in $(seq ${START:-0} $ROUNDS); do
  S=$((2000+r*100))
  if [ $r = 0 ]; then
    $P scripts/player-run.py collect $D/d0.npz $MAP $STEPS 1 none $(seq $S $((S+EP-1))) 2>&1 | grep -E "^collect|Error"
    INIT=none; E=${E0:-12}
  else
    $P scripts/player-run.py collect $D/d$r.npz $MAP $STEPS $BETA $D/m$((r-1)).pt $(seq $S $((S+EP-1))) 2>&1 | grep -E "^collect|Error"
    INIT=$D/m$((r-1)).pt; E=${E1:-6}
  fi
  LR=${LR:-1e-3} THREADS=${THREADS:-10} $P scripts/player-train.py $D/m$r.pt $E $KIND $INIT $(ls -v $D/d[0-9]*.npz | tail -n ${KEEP:-1000}) 2>&1 | grep EPOCH | tail -1 | cut -c1-110
  echo "round $r: $($P scripts/player-run.py eval $D/m$r.pt $MAP $STEPS $EVAL 2>&1 | grep -E '^eval|Error') $(date +%H:%M)"
done
