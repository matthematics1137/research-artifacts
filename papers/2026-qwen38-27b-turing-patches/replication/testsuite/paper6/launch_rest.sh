#!/bin/bash
# Paper 6: the rest of the campaign after the 2026-10-31 pause (research/P6_EFFICIENCY_RETHINK_2026-10-31.md).
# Same unit name and caps as launch_campaign.sh; the harness is unchanged (its files are hashed into every journal row).
# What differs is operational only:
#   - run_cells.py invocations in sequence (one spec always runs in canonical order); since Matt chose the primary
#     cells first (see --run below), these are the long-context passes of the six co-primary cells still missing;
#     the original three-step order, kept for the record:
#     27B IQ3_XXS k8v4 (finishing), then the fully resident Qwen3-8B cells, then the partly offloaded 27B Q4_K_XL cells;
#   - --idle-seconds 3 for every primary pass (long-context passes keep the CPU light; the original plan kept the
#     harness default 10 for the partly offloaded 27B cells);
#   - a CPU thermal guard (tools/thermal_guard.py) beside the load guard, on the same PAUSE marker.
#   bash testsuite/paper6/launch_rest.sh            # --run is the unit's own entry point
set -euo pipefail
R=/opt/p8/repository
P6=$R/testsuite/paper6

if [ "${1:-}" = "--run" ]; then
  cd "$P6"
  # Matt, 2026-10-31 ~20:30 EDT: the primary cells first ("we just do the 8hr one with your changes"): the long-context
  # passes of the eight cells the two co-primaries use (design §7), six of them still to run. Everything else waits
  # for a later decision, made on cost and heat.
  for step in "qwen3-8b:Q2_K_XL:f16,qwen3-8b:Q2_K_XL:q4_0,qwen3-8b:Q4_K_XL:f16,qwen3-8b:Q4_K_XL:q4_0 3 longctx" \
              "qwen38-27b:Q4_K_XL:f16,qwen38-27b:Q4_K_XL:q4_0 3 longctx"; do
    read -r cells idle tasks <<< "$step"            # read, not word splitting: no glob, no surprises
    echo "$(date '+%Y-%m-%d %H:%M:%S') launch_rest: --cells $cells --tasks $tasks --idle-seconds $idle"
    rc=0
    /usr/bin/python3 -B run_cells.py --cells "$cells" --tasks "$tasks" --idle-seconds "$idle" || rc=$?
    echo "$(date '+%Y-%m-%d %H:%M:%S') launch_rest: --cells $cells ended rc=$rc"
  done
  exit 0
fi

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
if ! systemctl --user is-active --quiet p6-thermal-guard; then
  systemd-run --user --unit=p6-thermal-guard --collect -p WorkingDirectory=$R -p MemoryMax=256M \
    -E THERMAL_GUARD_PAUSE=testsuite/paper6/PAUSE -E THERMAL_GUARD_LOGDIR=testsuite/paper6/work/logs \
    /usr/bin/python3 -B tools/thermal_guard.py
fi
systemd-run --user --unit=p6-cells --collect -p WorkingDirectory=$P6 \
  -p MemoryMax=32G -p MemorySwapMax=0 -p OOMScoreAdjust=500 \
  -p StandardOutput=append:$P6/work/logs/cells.log -p StandardError=append:$P6/work/logs/cells.log \
  /bin/bash "$P6/launch_rest.sh" --run
echo "p6-cells started; log $P6/work/logs/cells.log, guard logs $P6/work/logs/{load,thermal}_guard.log"
