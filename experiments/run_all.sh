#!/usr/bin/env bash
# Run every experiment in dependency order (exp3 selects alpha for exp6/exp8;
# exp7 reads exp1 and exp5). Logs go to results/logs/. Takes roughly 1-2 hours.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
for e in exp3_map exp1_length exp5_homophonic exp2_restarts exp4_lm_weight exp6_zodiac exp7_decoding exp8_em_progress; do
  echo "$(date '+%H:%M:%S') start $e"
  "$PY" "$e.py" "$@" > "../results/logs/$e.log" 2>&1
  echo "$(date '+%H:%M:%S') done  $e"
done
