#!/bin/bash
# P9 phase 1b (research/prereg/2026-11-05-p9-phase1.md, "Phase 1b registration"): our build's one-row calls with the
# weights held in L2 (one copy) against read from DRAM (rotated copies, as in M1), seven rounds, the order alternating
# by round, with the clock sampler. On the laptop: the seven model shapes at K = 1 and 2. On the desktop (L2 5.5 MB):
# the shapes that fit, attn_kv at K = 1 and 2 and gdn_z and gdn_out at K = 1. Writes OUT (default under
# testsuite/results/ada/p9/ or $L/runs/p9/).
#   bash tools/turing/p9_run_1b.sh laptop [OUT]      (as a capped unit; see p9_run_laptop.sh)
#   bash $L/tools/p9_run_1b.sh desktop [OUT]         (as a queue step)
set -euo pipefail
MACHINE=$1
if [ "$MACHINE" = laptop ]; then
  R=$(cd "$(dirname "$0")/../.." && pwd)
  OUT=${2:-$R/testsuite/results/ada/p9/1b-$(date +%Y%m%dT%H%M%S)}
  PY=$R/engines/exl3-env/bin/python
  SITE=$R/engines/exl3-1.6.5-w13sm89/site
  T=$R/tools/turing
  ARM=w13sm89
  MAXT=87
  ONLY="attn_kv:1,gdn_z:1,gdn_out:1,gdn_qkv:1,attn_q:1,mlp_gate:1,mlp_down:1,attn_kv:2,gdn_z:2,gdn_out:2,gdn_qkv:2,attn_q:2,mlp_gate:2,mlp_down:2"
else
  L=${TURING_LAB:-/opt/p8}
  OUT=${2:-$L/runs/p9/1b-$(date +%Y%m%dT%H%M%S)}
  PY=$L/venv/bin/python
  SITE=$L/overlay/exl3-w13
  T=$L/tools
  ARM=w13
  MAXT=84
  ONLY="attn_kv:1,attn_kv:2,gdn_z:1,gdn_out:1"
  sudo -n /usr/bin/nvidia-smi -pl 260 >/dev/null
fi
mkdir -p "$OUT"
say() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
date +%z > "$OUT/utc_offset.txt"   # nvidia-smi's timestamps are local time
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms 100 > "$OUT/clocks.csv" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

say "M1b: $ARM, L2 (one copy) against DRAM (rotated copies), 7 rounds"
for r in 1 2 3 4 5 6 7; do
  if [ $((r % 2)) = 1 ]; then ORDER="1 0"; else ORDER="0 1"; fi   # copies 1 = L2, 0 = rotated (DRAM)
  for c in $ORDER; do
    tag=$([ "$c" = 1 ] && echo l2 || echo dram)
    env PYTHONPATH="$SITE:$T" EXL3_MGEMM_N_THRESHOLD=0 "$PY" "$T/p9_bench.py" --out "$OUT/m1b.jsonl" \
      --arm "$ARM-$tag" --round "$r" --sets model --only "$ONLY" --copies "$c" --clocks "$OUT/clocks.csv" \
      --max-temp "$MAXT" 2>>"$OUT/m1b.stderr" | tail -1 | sed "s/^/  r$r $tag: /" | tee -a "$OUT/run.log"
  done
done
say "done: $OUT"
