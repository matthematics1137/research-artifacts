#!/bin/bash
# Paper 8 W2: decode kernel census on the 2080 Ti, undrafted then MTP, each under nsys, then census_report.py.
# One queue step (full_queue.sh runs it inside a capped turing-run unit). Each config's capture gets a fresh stamp,
# so a try the guard stopped leaves its partial files in place and the retry writes new ones; <cfg>.done names the
# stamp census_report.py reads. Same engine settings as the exl3 lanes: 16K Q8 cache, chunk 1024, MTP 6 dynamic.
#   bash census.sh [TAG] [EXTRA EXLLAMAV3 ARGS...]     results in $L/runs/census/TAG
#   CENSUS_CFGS="nodraft" bash census.sh TAG             only the undrafted capture (dispatch-switch arms)
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
NSYS=${NSYS:-/usr/local/cuda-13.4/bin/nsys}
TAG=${1:-w2}; shift || true
D=$L/runs/census/$TAG
mkdir -p "$D"
M=(-m "$L/models/qwen38-exl3-2.0" -cs 16384 -cq 8,8 -rcs 2 -chunk_size 1024 "$@")
for cfg in ${CENSUS_CFGS:-nodraft mtp}; do
  [ -e "$D/$cfg.done" ] && continue
  extra=(); [ "$cfg" = mtp ] && extra=(-mtp -ndt 6 -dds)
  stamp=$(date +%Y%m%dT%H%M%S)
  "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=repeat:3 --force-overwrite=false -o "$D/$cfg-$stamp" \
    "$L/venv/bin/python" "$L/tools/census.py" "${M[@]}" "${extra[@]}" --tag "$TAG-$cfg-$stamp" --out "$D/decode.jsonl"
  echo "$stamp" > "$D/$cfg.done"
done
"$L/venv/bin/python" "$L/tools/census_report.py" --dir "$D" --nsys "$NSYS"
