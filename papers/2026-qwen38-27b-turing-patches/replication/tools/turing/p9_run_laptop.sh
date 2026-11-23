#!/bin/bash
# P9 phase 1 on the RTX 4080 Laptop GPU (research/prereg/2026-11-05-p9-phase1.md): M1 with four arms cycled in each of
# seven rounds (the official wheel, stock89, w13sm89, w13sm89 with W9's host rule forced on), M2 on the stock89 and
# w13sm89 source trees, M3 on the laptop builds and the official wheel (sm_75 and sm_89), and a clock sampler every
# 100 ms for the whole session. Stops a step above 87 C. Writes OUT (default testsuite/results/ada/p9/<stamp>).
#   bash tools/turing/p9_run_laptop.sh [OUT] [--smoke]
# Run it as a capped unit so it cannot crowd the laptop:
#   systemd-run --user --unit=p9-laptop --collect -p MemoryMax=12G -p MemorySwapMax=0 -p Nice=5 \
#     bash tools/turing/p9_run_laptop.sh
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$R/testsuite/results/ada/p9/$(date +%Y%m%dT%H%M%S)}
SMOKE=${2:-}
mkdir -p "$OUT/sass"
PY=$R/engines/exl3-env/bin/python
E=$R/engines
CUDA=/usr/local/cuda-12.8
T=$R/tools/turing
ROUNDS=7
[ "$SMOKE" = "--smoke" ] && ROUNDS=1
say() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }

{
  echo "recorded $(date '+%Y-%m-%d %H:%M:%S %Z') before the run"
  echo "nvidia-powerd: $(systemctl is-active nvidia-powerd)"
  echo "nv_dynamic_boost: $(cat /sys/devices/platform/asus-nb-wmi/nv_dynamic_boost 2>/dev/null) W"
  echo "platform profile: $(powerprofilesctl get 2>/dev/null)"
  nvidia-smi -q -d POWER | grep -E "Current Power Limit|Default Power Limit|Max Power Limit" | head -3
} > "$OUT/power_state.txt"
date +%z > "$OUT/utc_offset.txt"   # nvidia-smi's timestamps are local time

nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms 100 > "$OUT/clocks.csv" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

declare -A SITE=([stock]=$E/exl3-1.6.5-stock/site [stock89]=$E/exl3-1.6.5-stock89/site
                 [w13sm89]=$E/exl3-1.6.5-w13sm89/site [w13sm89fit]=$E/exl3-1.6.5-w13sm89/site)
declare -A XENV=([stock]="" [stock89]="" [w13sm89]="" [w13sm89fit]="EXL3_INT8_SQ_FIT=1")
ARMS=(stock stock89 w13sm89 w13sm89fit)
SM=(); [ "$SMOKE" = "--smoke" ] && SM=(--smoke)

say "M1: ${#ARMS[@]} arms x $ROUNDS rounds"
for r in $(seq 1 $ROUNDS); do
  for i in "${!ARMS[@]}"; do
    arm=${ARMS[$(( (i + r - 1) % ${#ARMS[@]} ))]}   # the arm order rotates with the round
    env PYTHONPATH="${SITE[$arm]}:$T" EXL3_MGEMM_N_THRESHOLD=0 EXL3_INT8_SQ_FIT_LOG=1 ${XENV[$arm]} \
      "$PY" "$T/p9_bench.py" --out "$OUT/m1.jsonl" --arm "$arm" --round "$r" --clocks "$OUT/clocks.csv" --max-temp 87 \
      "${SM[@]}" 2>>"$OUT/m1.$arm.stderr" | tail -1 | sed "s/^/  r$r $arm: /" | tee -a "$OUT/run.log"
  done
done
for arm in "${ARMS[@]}"; do
  env PYTHONPATH="${SITE[$arm]}:$T" EXL3_MGEMM_N_THRESHOLD=0 ${XENV[$arm]} "$PY" "$T/p9_bench.py" \
    --out "$OUT/m1_dispatch.jsonl" --arm "$arm" --part dispatch "${SM[@]}" >/dev/null 2>>"$OUT/m1.$arm.stderr"
done

say "M2: stock89 and w13sm89 sources"
for b in stock89 w13sm89; do
  SRC=$(ls -d "$E/exl3-1.6.5-$b"/src-* | head -1)/exllamav3/exllamav3_ext
  env PYTHONPATH="$E/exl3-1.6.5-$b/site:$T" EXL3_MGEMM_N_THRESHOLD=0 CUDA_HOME=$CUDA \
    PATH="$CUDA/bin:$E/exl3-env/bin:$PATH" CC=gcc-13 CXX=g++-13 MAX_JOBS=4 TORCH_EXTENSIONS_DIR="$E/torch_ext" \
    "$PY" "$T/p9_phases.py" --src "$SRC" --tag "$b" --out "$OUT/m2.jsonl" --rounds $ROUNDS --clocks "$OUT/clocks.csv" \
    --max-temp 87 "${SM[@]}" 2>>"$OUT/m2.$b.stderr" | tail -1 | sed "s/^/  $b: /" | tee -a "$OUT/run.log"
done

say "M3: machine code"
for b in stock89 w13sm89; do
  python3 "$T/p9_sass.py" --so "$E/exl3-1.6.5-$b/site/"exllamav3_ext*.so --arch sm_89 --label "$b" --out-dir "$OUT/sass" \
    --cuobjdump "$CUDA/bin/cuobjdump" | tee -a "$OUT/run.log"
done
for a in sm_75 sm_89; do
  python3 "$T/p9_sass.py" --so "$E/exl3-1.6.5-stock/site/"exllamav3_ext*.so --arch $a --label stock --out-dir "$OUT/sass" \
    --cuobjdump "$CUDA/bin/cuobjdump" | tee -a "$OUT/run.log"
done
say "done: $OUT"
