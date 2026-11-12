#!/usr/bin/env bash
# P8 long context, stages 2-3 (research/prereg/2026-11-01-p8-longctx-stages2-3.md, Amendment 1): one desktop queue
# step per plan.
#   x-p8lc-<plan>|bash /opt/p8/tools/p8_longctx/stage2.sh <plan>
# Checks without sudo that the power limit reads 260 W, checks that the w13 overlay is what Python imports
# (with_engine.sh, no GPU), then runs the plan (stage2_run.py), which resumes item by item after a guard stop.
# Plans: $L/runs/longctx/plans/<plan>.json; items: $L/runs/longctx/items/; results: $L/runs/longctx/<group>/.
# Nothing is deleted.
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
PLAN=$L/runs/longctx/plans/$1.json
GROUP=$("$L/venv/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["group"])' "$PLAN")
OUT=$L/runs/longctx/$GROUP
mkdir -p "$OUT"
limit=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
echo "$(date -Is) stage2.sh $1 start; group $GROUP; power limit $limit W"
case $limit in 260|260.0|260.00) ;; *) echo "power limit is $limit W, not 260 W; stopping" >&2; exit 4 ;; esac
bash "$L/tools/with_engine.sh" w13 true
nvidia-smi --query-gpu=temperature.gpu,power.limit,clocks.sm,clocks.mem,memory.used --format=csv,noheader
rc=0
"$L/venv/bin/python" "$L/tools/p8_longctx/stage2_run.py" "$PLAN" --items "$L/runs/longctx/items" --out "$OUT" || rc=$?
echo "$(date -Is) stage2.sh $1 end rc=$rc"
exit $rc
