#!/bin/bash
# P8 extension (research/prereg/2026-11-05-p8-call-split-ada.md): census.sh's undrafted config on the laptop's
# RTX 4080 Laptop GPU, with the stock exllamav3 1.6.5 wheel unpacked into engines/exl3-1.6.5-stock/site and put
# first on PYTHONPATH over engines/exl3-env, and EXL3_MGEMM_N_THRESHOLD=0 as in both 2080 Ti censuses. It records the
# GPU's clocks, power and temperature every 500 ms beside the capture, and stops if the GPU passes 87 C.
#   bash tools/turing/census_ada.sh [TAG]        results in testsuite/results/ada/census/TAG
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
NSYS=${NSYS:-/usr/local/bin/nsys}
TAG=${1:-d0-stock-mgemmn0}
D=$R/testsuite/results/ada/census/$TAG
mkdir -p "$D"
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "the GPU has a compute process; not starting (registration: stops)" >&2; exit 3
fi
stamp=$(date +%Y%m%dT%H%M%S)
nvidia-smi --query-gpu=timestamp,clocks.mem,clocks.sm,power.draw,enforced.power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 500 > "$D/telemetry-$stamp.csv" &
LOGGER=$!
watch_heat() {
  while kill -0 "$LOGGER" 2>/dev/null; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    if [ "${t:-0}" -gt 87 ]; then echo "GPU at ${t} C: stopping (registration: stops)" >&2; kill -TERM "$RUN" 2>/dev/null; fi
    sleep 2
  done
}
export EXL3_MGEMM_N_THRESHOLD=0
export PYTHONPATH=$R/engines/exl3-1.6.5-stock/site:$R/tools/turing
"$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat:3 --force-overwrite=false -o "$D/nodraft-$stamp" \
  "$R/engines/exl3-env/bin/python" "$R/tools/turing/census.py" -m "$HOME/models/qwen38-exl3-2.0" -cs 16384 -cq 8,8 \
  -rcs 2 -chunk_size 1024 --tag "ada-$TAG-nodraft-$stamp" --out "$D/decode.jsonl" &
RUN=$!
watch_heat &
HEAT=$!
status=0
wait "$RUN" || status=$?
kill "$LOGGER" 2>/dev/null || true
wait "$HEAT" 2>/dev/null || true
echo "$stamp" > "$D/nodraft.done"
echo "census exit $status; files in $D"
exit "$status"
