#!/usr/bin/env bash
# P8 fd75d logit check (research/prereg/2026-11-03-p8-fd75d-check.md): one desktop queue step.
#   x-p8lc-fd75d|bash /opt/p8/tools/p8_longctx/fd75d_check.sh
# Builds the teacher-forced streams with the model's tokenizer (CPU), then runs tools/turing/numerics.py, unchanged,
# three times in fresh processes: F (EXL3_FD75D=1, fd75d), T1 and T2 (EXL3_FD75D=0, Triton), each one pass of the
# default dispatch with full logits at every position. Then fd75d_analyze.py judges them against the registration.
# A run that would push GPU time past 60 minutes is not started. A finished run is not repeated after a guard stop;
# an interrupted one resumes (numerics --resume). Journal: $L/runs/longctx/fd75d/journal.jsonl, ending with a
# plan_end row. Nothing is deleted.
set -uo pipefail
L=${TURING_LAB:-/opt/p8}
OUT=$L/runs/longctx/fd75d
J=$OUT/journal.jsonl
PY=$L/venv/bin/python
T=$L/tools/p8_longctx
BUDGET_S=3600
RUN_EST_S=1200
mkdir -p "$OUT/numerics"
row() { printf '%s\n' "$1" | tee -a "$J"; }
used() { [ -f "$OUT/used_s" ] && awk '{s += $1} END {print int(s)}' "$OUT/used_s" || echo 0; }

limit=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
row "{\"kind\":\"plan_start\",\"time\":\"$(date -Is)\",\"power_limit_w\":\"$limit\",\"used_before_s\":$(used)}"
case $limit in 260|260.0|260.00) ;; *) row "{\"kind\":\"plan_error\",\"why\":\"power limit $limit W\"}"; exit 4 ;; esac
bash "$L/tools/with_engine.sh" w13 true || { row "{\"kind\":\"plan_error\",\"why\":\"w13 overlay check failed\"}"; exit 3; }
if [ ! -s "$OUT/streams.jsonl" ]; then
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$L/overlay/exl3-w13 "$PY" "$T/fd75d_streams.py" --items "$L/runs/longctx/items" \
    --out "$OUT/streams.jsonl" || { row "{\"kind\":\"plan_error\",\"why\":\"streams build failed\"}"; exit 3; }
fi
for run in F T1 T2; do
  [ -f "$OUT/$run.done" ] && continue
  u=$(used)
  if [ $((u + RUN_EST_S)) -gt $BUDGET_S ]; then
    row "{\"kind\":\"budget_stop\",\"time\":\"$(date -Is)\",\"next_run\":\"$run\",\"used_s\":$u}"
    break
  fi
  fd=0; [ "$run" = F ] && fd=1
  resume=""; [ -d "$OUT/numerics/fd75d-$run" ] && resume="--resume"
  row "{\"kind\":\"run_start\",\"run\":\"$run\",\"EXL3_FD75D\":$fd,\"time\":\"$(date -Is)\",\"used_s\":$u}"
  t0=$(date +%s)
  env PYTHONPATH="$L/overlay/exl3-w13" EXL3_MGEMM_N_THRESHOLD=0 EXL3_FD75D_LIVE=0 EXL3_INT8_SQ_MAX_M=2 EXL3_GEMV_SK=2 \
      EXL3_FD75D=$fd "$PY" "$L/tools/numerics.py" -m "$L/models/qwen38-exl3-2.0" -cq 8,8 -chunk_size 1024 -rcs 2 \
      --streams "$OUT/streams.jsonl" --modes default --max-response-tokens 128 --full-logits-stride 1 \
      --tag "fd75d-$run" --out-dir "$OUT/numerics" $resume
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  echo "$dt" >> "$OUT/used_s"
  row "{\"kind\":\"run_end\",\"run\":\"$run\",\"rc\":$rc,\"seconds\":$dt,\"time\":\"$(date -Is)\"}"
  [ $rc -eq 0 ] || { row "{\"kind\":\"plan_error\",\"why\":\"numerics run $run exited $rc\"}"; exit 3; }
  date -Is > "$OUT/$run.done"
done
if [ -f "$OUT/F.done" ] && [ -f "$OUT/T1.done" ] && [ -f "$OUT/T2.done" ]; then
  "$PY" "$T/fd75d_analyze.py" "$OUT/numerics/fd75d-F.npz" "$OUT/numerics/fd75d-T1.npz" "$OUT/numerics/fd75d-T2.npz" \
    --out "$OUT/results.json" || { row "{\"kind\":\"plan_error\",\"why\":\"analysis failed\"}"; exit 3; }
fi
row "{\"kind\":\"plan_end\",\"time\":\"$(date -Is)\",\"used_s\":$(used)}"
