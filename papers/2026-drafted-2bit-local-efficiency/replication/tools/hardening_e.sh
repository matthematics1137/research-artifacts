#!/usr/bin/env bash
# H38e (pre-registered 2026-09-12 04:10) — GPU-GATED: does exllamav3 `dev` (deterministic GDN reduction) make greedy decoding
# reproducible? Two undrafted greedy runs on the first 100 GSM8K-500 items served by the dev checkout (PYTHONPATH),
# LD_LIBRARY_PATH unset; labels EFF-h38e_nodraft_dev1/2.   usage: hardening_e.sh [all|cells|analyze]   ~40 min of GPU.
set -u
EFF=${EFF:-/path/to/scratch/eff}
source "$EFF/cells.sh"
REPO=/path/to/lab-repo; TOOLS=$REPO/testsuite/results/efficiency/tools; ROWS=$REPO/testsuite/results/efficiency/rows
DEV=${DEV:-/path/to/scratch/upstream/exllamav3}
what=${1:-all}; LOG=$EFF/logs/hardening.log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
reset_suite() { local L=$1 S=$2; rm -f "$RES/$L/$S.jsonl"; python3 - "$L" "$S" <<'PY'
import json, sys
p = '/path/to/lab-repo/testsuite/evals/results/summary.jsonl'; L, S = sys.argv[1], sys.argv[2]
lines = open(p).read().splitlines(True)
open(p, 'w').writelines([l for l in lines if not (l.strip() and json.loads(l).get('label') == L and json.loads(l).get('suite') == S)])
PY
}
cell_dev() { local L=$1 pid; pid=$(cat "$EFF/tabby.pid" 2>/dev/null)
  grep -oE '/[^ ]*(libcublas|libcublasLt|libnvrtc|libcusolver|libcudart)[^ ]*' /proc/$pid/maps 2>/dev/null | sort -u > "$ROWS/${L}_cuda_libs.txt"
  say "$L: engine $(grep -oE 'exllamav3 version: [0-9a-z.+-]+' "$EFF/logs/tabby_${L#EFF-}.log" 2>/dev/null | head -1)  system-12.1 libs: $(grep -c '/usr/local/cuda' "$ROWS/${L}_cuda_libs.txt")  ext: $(ls $DEV/exllamav3/*.so 2>/dev/null | head -1 | xargs -n1 basename)"
  reset_suite "$L" gsm8k_500; say "eval $L gsm8k_500 (limit 100) on :18120"
  ( cd "$KIT" && env -u LD_LIBRARY_PATH nice -n 5 ./run_eval.py --suite gsm8k_500 --label "$L" --base-url http://127.0.0.1:18120 --temperature 0 --limit 100 2>&1 | tail -1 | cut -c1-220 ) | tee -a "$LOG"; }
cells() { for i in 1 2; do
  run_cell "h38e_nodraft_dev$i" "\"$EFF/tabby.sh\" start h38e_nodraft_dev$i \"$TOOLS/tabby_h38_nodraft.yml\" 143 -u LD_LIBRARY_PATH PYTHONPATH=$DEV" "\"$EFF/tabby.sh\" stop" - cell_dev EFF-h38e_nodraft_dev$i
done; }
analyze() {
  say "H38e: dev vs dev"; "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38e_nodraft_dev1 EFF-h38e_nodraft_dev2 --limit 100 --json "$ROWS/h38e_dev_vs_dev.json" | tee -a "$LOG"
  say "H38e: 1.4.7 clean vs dev1"; "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38d_nodraft_clean1 EFF-h38e_nodraft_dev1 --limit 100 --json "$ROWS/h38e_147clean_vs_dev.json" | tee -a "$LOG"
}
if [ "${DRY:-0}" = 1 ]; then run_cell() { echo "DRY cell $1"; echo "   start: $2"; shift 4; echo "   run:   $*"; }; cell_dev() { :; }; LOG=/dev/null; fi
say "=== hardening_e $what start (dev checkout $DEV @ $(git -C $DEV rev-parse --short HEAD))"
case "$what" in all) cells; analyze;; cells) cells;; analyze) analyze;; *) echo "usage"; exit 2;; esac
say "=== hardening_e $what done"
