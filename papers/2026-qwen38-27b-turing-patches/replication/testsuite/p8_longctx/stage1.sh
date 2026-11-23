#!/usr/bin/env bash
# P8 long context, stage 1 (research/prereg/2026-11-01-p8-longctx-stage1.md, Amendment 1): one desktop queue step.
#   x-p8lc-stage1|bash /opt/p8/tools/p8_longctx/stage1.sh
# Checks without sudo that the power limit reads 260 W, checks that the w13 overlay is what Python imports
# (with_engine.sh, no GPU), then runs the ladder (stage1_run.py), which resumes from its journal after a guard stop.
# Everything goes to $L/runs/longctx/stage1/. Nothing is deleted.
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
OUT=$L/runs/longctx/stage1
mkdir -p "$OUT"
limit=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
echo "$(date -Is) stage1.sh start; power limit $limit W"
case $limit in 260|260.0|260.00) ;; *) echo "power limit is $limit W, not 260 W; stopping" >&2; exit 4 ;; esac
bash "$L/tools/with_engine.sh" w13 true
nvidia-smi --query-gpu=temperature.gpu,power.limit,clocks.sm,clocks.mem,memory.used --format=csv,noheader
rc=0
"$L/venv/bin/python" "$L/tools/p8_longctx/stage1_run.py" --prompts "$OUT/prompts" --out "$OUT" || rc=$?
echo "$(date -Is) stage1.sh end rc=$rc"
exit $rc
