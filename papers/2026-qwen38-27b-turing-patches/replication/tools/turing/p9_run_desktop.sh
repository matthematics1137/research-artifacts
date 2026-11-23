#!/bin/bash
# P9 phase 1 on the 2080 Ti (research/prereg/2026-11-05-p9-phase1.md): M1 on the w13 overlay for seven rounds, the
# dispatch check, M2 on w13's source tree, M3 on the w13 overlay and the lab venv's official wheel (sm_75), with a
# clock sampler every 100 ms for the whole session. Stops a step above 84 C. Run by the queue as step x-p9-desktop
# (a turing-run unit), so the guard watches it like any GPU step. Expects to sit in $L/tools beside p9_common.py,
# p9_bench.py, p9_phases.py, p9_sass.py and p9/phases.cu.
#   bash $L/tools/p9_run_desktop.sh [OUT] [--smoke]
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
OUT=${1:-$L/runs/p9/$(date +%Y%m%dT%H%M%S)}
SMOKE=${2:-}
mkdir -p "$OUT/sass"
PY=$L/venv/bin/python
TC=$L/toolchains/cuda-12.5-glibc243
T=$L/tools
W13=$L/overlay/exl3-w13
SRC=$(ls -d "$L"/src/exllamav3-1.6.5-w13-2026* | head -1)/exllamav3/exllamav3_ext
ROUNDS=7
[ "$SMOKE" = "--smoke" ] && ROUNDS=1
SM=(); [ "$SMOKE" = "--smoke" ] && SM=(--smoke)
say() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }

sudo -n /usr/bin/nvidia-smi -pl 260 >/dev/null
nvidia-smi -q -d POWER | grep -E "Current Power Limit|Default Power Limit|Max Power Limit" | head -3 > "$OUT/power_state.txt"
date +%z > "$OUT/utc_offset.txt"   # nvidia-smi's timestamps are local time
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms 100 > "$OUT/clocks.csv" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
cd "$T"

say "M1: w13 x $ROUNDS rounds"
for r in $(seq 1 $ROUNDS); do
  env PYTHONPATH="$W13:$T" EXL3_MGEMM_N_THRESHOLD=0 EXL3_INT8_SQ_FIT_LOG=1 "$PY" p9_bench.py --out "$OUT/m1.jsonl" \
    --arm w13 --round "$r" --clocks "$OUT/clocks.csv" --max-temp 84 "${SM[@]}" 2>>"$OUT/m1.w13.stderr" \
    | tail -1 | sed "s/^/  r$r w13: /" | tee -a "$OUT/run.log"
done
env PYTHONPATH="$W13:$T" EXL3_MGEMM_N_THRESHOLD=0 "$PY" p9_bench.py --out "$OUT/m1_dispatch.jsonl" --arm w13 \
  --part dispatch "${SM[@]}" >/dev/null 2>>"$OUT/m1.w13.stderr"

say "M2: w13 source"
env PYTHONPATH="$W13:$T" EXL3_MGEMM_N_THRESHOLD=0 CUDA_HOME="$TC" PATH="$TC/bin:$L/venv/bin:/usr/bin:/bin" CC=gcc-13 \
  CXX=g++-13 MAX_JOBS=2 TORCH_EXTENSIONS_DIR="$L/cache/torch_ext" "$PY" p9_phases.py --src "$SRC" --tag w13 \
  --out "$OUT/m2.jsonl" --rounds $ROUNDS --clocks "$OUT/clocks.csv" --max-temp 84 "${SM[@]}" 2>>"$OUT/m2.w13.stderr" \
  | tail -1 | sed "s/^/  w13: /" | tee -a "$OUT/run.log"

say "M3: machine code"
"$PY" p9_sass.py --so "$W13"/exllamav3_ext*.so --arch sm_75 --label w13 --out-dir "$OUT/sass" \
  --cuobjdump "$TC/bin/cuobjdump" | tee -a "$OUT/run.log"
"$PY" p9_sass.py --so "$L"/overlay/exl3-w11b/exllamav3_ext*.so --arch sm_75 --label w11b --out-dir "$OUT/sass" \
  --cuobjdump "$TC/bin/cuobjdump" | tee -a "$OUT/run.log"   # W12p's sm_75 comparison: same toolchain, W12p absent
"$PY" p9_sass.py --so "$L"/venv/lib/python3.12/site-packages/exllamav3_ext*.so --arch sm_75 --label stock-venv \
  --out-dir "$OUT/sass" --cuobjdump "$TC/bin/cuobjdump" | tee -a "$OUT/run.log"
say "done: $OUT"
