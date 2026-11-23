#!/bin/bash
# Paper 6: start the 24-cell campaign on the laptop GPU once Paper 5's follow-up chain has finished (the goal's
# P5 -> P6 order: never overlap them on the GPU). Same caps as Paper 5's chain (32 GB, no swap, OOM score 500), a load
# guard on Paper 6's own pause marker (testsuite/paper5/load_guard.py with LOAD_GUARD_PAUSE / LOAD_GUARD_LOGDIR), and
# the harness default --idle-seconds 10 (HARNESS_NOTES 2026-09-20). The harness keeps its own hard refusals (pinned
# engine, GGUF hashes, manifest, study ports, P5 phase units).
#   bash testsuite/paper6/launch_campaign.sh
set -euo pipefail
R=/opt/p8/repository
P6=$R/testsuite/paper6
for u in p5-followups p5-phase1 p5-phase2 p5-judges p5-grid; do
  if systemctl --user is-active --quiet "$u"; then echo "refusing: $u is still active (P5 before P6)"; exit 1; fi
done
if systemctl --user is-active --quiet p6-cells; then echo "refusing: p6-cells is already running"; exit 1; fi
mkdir -p "$P6/work/logs"
if ! systemctl --user is-active --quiet p6-load-guard; then
  systemd-run --user --unit=p6-load-guard --collect -p WorkingDirectory=$R \
    -E LOAD_GUARD_PAUSE=testsuite/paper6/PAUSE -E LOAD_GUARD_LOGDIR=testsuite/paper6/work/logs \
    /usr/bin/python3 -B testsuite/paper5/load_guard.py
fi
systemd-run --user --unit=p6-cells --collect -p WorkingDirectory=$P6 \
  -p MemoryMax=32G -p MemorySwapMax=0 -p OOMScoreAdjust=500 \
  -p StandardOutput=append:$P6/work/logs/cells.log -p StandardError=append:$P6/work/logs/cells.log \
  /usr/bin/python3 -B run_cells.py --cells all --idle-seconds 10
echo "p6-cells started; log $P6/work/logs/cells.log, guard log $P6/work/logs/load_guard.log"
