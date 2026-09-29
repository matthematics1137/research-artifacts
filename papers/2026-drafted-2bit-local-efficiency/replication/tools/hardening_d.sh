#!/usr/bin/env bash
# H38d (pre-registered 2026-09-10 11:45) — GPU-GATED: is the E49 temperature-0 non-determinism an LD_LIBRARY_PATH artifact?
# Two undrafted greedy runs of the daily config on the first 100 GSM8K-500 items with LD_LIBRARY_PATH UNSET (the lab's
# ~/.bashrc exports /usr/local/cuda-12.1/lib64, which let the system libcublasLt.so.12.1 and libnvrtc.so.12.1 preempt
# torch 2.10.0+cu128's bundled libraries in every campaign server). Labels EFF-h38d_nodraft_clean1/2; the server's loaded
# CUDA libraries are recorded per cell.   usage: hardening_d.sh [all|cells|analyze]   ~45 min of GPU.
set -u
EFF=${EFF:-/path/to/scratch/eff}
source "$EFF/cells.sh"
REPO=/path/to/lab-repo; TOOLS=$REPO/testsuite/results/efficiency/tools; ROWS=$REPO/testsuite/results/efficiency/rows
what=${1:-all}; LOG=$EFF/logs/hardening.log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
reset_suite() { local L=$1 S=$2; rm -f "$RES/$L/$S.jsonl"; python3 - "$L" "$S" <<'PY'
import json, sys
p = '/path/to/lab-repo/testsuite/evals/results/summary.jsonl'; L, S = sys.argv[1], sys.argv[2]
lines = open(p).read().splitlines(True)
open(p, 'w').writelines([l for l in lines if not (l.strip() and json.loads(l).get('label') == L and json.loads(l).get('suite') == S)])
PY
}
record_libs() { local L=$1 pid; pid=$(cat "$EFF/tabby.pid" 2>/dev/null); [ -n "$pid" ] || return 0
  grep -oE '/[^ ]*(libcublas|libcublasLt|libnvrtc|libcusolver|libcudart)[^ ]*' /proc/$pid/maps 2>/dev/null | sort -u > "$ROWS/${L}_cuda_libs.txt"
  say "$L: $(grep -c . "$ROWS/${L}_cuda_libs.txt") CUDA libs loaded, system-12.1 among them: $(grep -c '/usr/local/cuda' "$ROWS/${L}_cuda_libs.txt")"; }
cell_clean() { local L=$1; record_libs "$L"; reset_suite "$L" gsm8k_500; say "eval $L gsm8k_500 (limit 100) on :18120, LD_LIBRARY_PATH unset"
  ( cd "$KIT" && env -u LD_LIBRARY_PATH nice -n 5 ./run_eval.py --suite gsm8k_500 --label "$L" --base-url http://127.0.0.1:18120 --temperature 0 --limit 100 2>&1 | tail -1 | cut -c1-220 ) | tee -a "$LOG"; }
cells() { for i in 1 2; do
  run_cell "h38d_clean$i" "\"$EFF/tabby.sh\" start h38d_clean$i \"$TOOLS/tabby_h38_nodraft.yml\" 147 -u LD_LIBRARY_PATH" "\"$EFF/tabby.sh\" stop" - cell_clean EFF-h38d_nodraft_clean$i
done; }
analyze() {
  say "H38d: clean vs clean"; "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38d_nodraft_clean1 EFF-h38d_nodraft_clean2 --limit 100 --json "$ROWS/h38d_clean_vs_clean.json" | tee -a "$LOG"
  say "H38d: clean1 vs the E49 dirty undrafted run"; "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38a_nodraft EFF-h38d_nodraft_clean1 --limit 100 --json "$ROWS/h38d_dirty_vs_clean.json" | tee -a "$LOG"
}
if [ "${DRY:-0}" = 1 ]; then run_cell() { echo "DRY cell $1"; echo "   start: $2"; shift 4; echo "   run:   $*"; }; cell_clean() { :; }; LOG=/dev/null; fi
say "=== hardening_d $what start"
case "$what" in all) cells; analyze;; cells) cells;; analyze) analyze;; *) echo "usage: hardening_d.sh [all|cells|analyze]"; exit 2;; esac
say "=== hardening_d $what done"
