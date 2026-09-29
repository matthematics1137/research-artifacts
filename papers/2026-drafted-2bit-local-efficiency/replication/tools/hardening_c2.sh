#!/usr/bin/env bash
# H38c (iii), pre-registered 14:25 on 2026-09-09 — GPU-GATED: the undrafted daily config re-run at temperature 0 on the
# first 100 GSM8K-500 items (label EFF-h38c_nodraft_rerun), compared with its own H38a run (EFF-h38a_nodraft) to
# measure the base engine's run-to-run determinism. ~20 min of GPU.   usage: hardening_c2.sh [all|cell|analyze]
set -u
EFF=${EFF:-/path/to/scratch/eff}
source "$EFF/cells.sh"
REPO=/path/to/lab-repo; TOOLS=$REPO/testsuite/results/efficiency/tools; ROWS=$REPO/testsuite/results/efficiency/rows
what=${1:-all}; LOG=$EFF/logs/hardening.log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
thermal_watch() { local hot=0
  while :; do
    g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    c=$(sensors 2>/dev/null | awk '/Package id 0|Tctl/ {gsub(/[^0-9.]/,"",$4); print $4; exit}')
    if [ "${g:-0}" -ge 90 ] 2>/dev/null || [ "${c%.*}" -ge 98 ] 2>/dev/null; then hot=$((hot+1)); else hot=0; fi
    if [ $hot -ge 3 ]; then say "THERMAL STOP gpu=${g}C pkg=${c}C"; echo YIELDED-THERMAL > "$EFF/yield.flag"
      for f in "$EFF/tabby.pid" "$EFF/server.pid"; do [ -f "$f" ] && kill -INT "$(cat "$f")" 2>/dev/null; done; hot=0; sleep 120; fi
    sleep 10
  done
}
reset_suite() { local L=$1 S=$2; rm -f "$RES/$L/$S.jsonl"; python3 - "$L" "$S" <<'PY'
import json, sys
p = '/path/to/lab-repo/testsuite/evals/results/summary.jsonl'; L, S = sys.argv[1], sys.argv[2]
lines = open(p).read().splitlines(True)
open(p, 'w').writelines([l for l in lines if not (l.strip() and json.loads(l).get('label') == L and json.loads(l).get('suite') == S)])
PY
}
cell_nodraft_rerun() {
  reset_suite EFF-h38c_nodraft_rerun gsm8k_500; say "eval EFF-h38c_nodraft_rerun gsm8k_500 (limit 100) on :18120"
  ( cd "$KIT" && nice -n 5 ./run_eval.py --suite gsm8k_500 --label EFF-h38c_nodraft_rerun --base-url http://127.0.0.1:18120 --temperature 0 --limit 100 2>&1 | tail -1 | cut -c1-220 ) | tee -a "$LOG"
}
cell() { run_cell h38c_nodraft_rerun "\"$EFF/tabby.sh\" start h38c_nodraft_rerun \"$TOOLS/tabby_h38_nodraft.yml\" 147" "\"$EFF/tabby.sh\" stop" - cell_nodraft_rerun; }
analyze() {
  say "H38c (iii) same-config identity, undrafted vs undrafted rerun, first 100 GSM8K-500 items"
  "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38a_nodraft EFF-h38c_nodraft_rerun --limit 100 --json "$ROWS/h38c_same_config_undrafted.json" | tee -a "$LOG"
}
if [ "${DRY:-0}" = 1 ]; then run_cell() { echo "DRY cell $1"; echo "   start: $2"; shift 4; echo "   run:   $*"; }; cell_nodraft_rerun() { :; }; thermal_watch() { :; }; LOG=/dev/null; fi
say "=== hardening_c2 $what start"
if [ "$what" != analyze ] && [ "${DRY:-0}" != 1 ]; then thermal_watch & TW=$!; echo "$TW" >> "$EFF/our_pids.txt"; trap 'kill $TW 2>/dev/null' EXIT; fi
case "$what" in all) cell; analyze;; cell) cell;; analyze) analyze;; *) echo "usage: hardening_c2.sh [all|cell|analyze]"; exit 2;; esac
say "=== hardening_c2 $what done"
