#!/usr/bin/env bash
# H38c controls (pre-registered §4.8, 2026-09-09 05:30) — GPU-GATED, runs after hardening.sh and the smoke rerun.
#   cell 1  drafted daily config: re-run GSM8K-500 items 1-100 at temperature 0 (label EFF-h38c_mtp_rerun) for the
#           same-config identity control, then 25 greedy traces with top-2 logprobs (rows/traces_h38c_mtp.jsonl)
#   cell 2  undrafted config: the same 25 traces (rows/traces_h38c_nodraft.jsonl)
#   analyze same-config identity (hardening_analyze --limit 100) and the divergence anatomy (divergence_probe compare)
#   usage: hardening_c.sh [all|cells|analyze]      DRY=1 prints the cells.   ~35 min of GPU.
set -u
EFF=${EFF:-/path/to/scratch/eff}
source "$EFF/cells.sh"
REPO=/path/to/lab-repo; TOOLS=$REPO/testsuite/results/efficiency/tools; ROWS=$REPO/testsuite/results/efficiency/rows; SERV=$REPO/serving
what=${1:-all}; LOG=$EFF/logs/hardening.log; mkdir -p "$EFF/logs" "$ROWS"
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
cell_mtp() {
  reset_suite EFF-h38c_mtp_rerun gsm8k_500; say "eval EFF-h38c_mtp_rerun gsm8k_500 (limit 100) on :18120"
  ( cd "$KIT" && nice -n 5 ./run_eval.py --suite gsm8k_500 --label EFF-h38c_mtp_rerun --base-url http://127.0.0.1:18120 --temperature 0 --limit 100 2>&1 | tail -1 | cut -c1-220 ) | tee -a "$LOG"
  rm -f "$ROWS/traces_h38c_mtp.jsonl"; say "traces h38c_mtp"
  nice -n 5 "$TOOLS/divergence_probe.py" collect --label h38c_mtp --port 18120 --limit 25 --top 2 2>&1 | grep -E 'COLLECT_DONE|error' | cut -c1-160 | tee -a "$LOG"
}
cell_nodraft() {
  rm -f "$ROWS/traces_h38c_nodraft.jsonl"; say "traces h38c_nodraft"
  nice -n 5 "$TOOLS/divergence_probe.py" collect --label h38c_nodraft --port 18120 --limit 25 --top 2 2>&1 | grep -E 'COLLECT_DONE|error' | cut -c1-160 | tee -a "$LOG"
}
cells() {
  run_cell h38c_mtp "\"$EFF/tabby.sh\" start h38c_mtp \"$SERV/tabby-daily.yml\" 147 EXL3_CKPT_PP=1024" "\"$EFF/tabby.sh\" stop" - cell_mtp
  run_cell h38c_nodraft "\"$EFF/tabby.sh\" start h38c_nodraft \"$TOOLS/tabby_h38_nodraft.yml\" 147" "\"$EFF/tabby.sh\" stop" - cell_nodraft
}
analyze() {
  say "H38c (i) same-config identity, drafted vs drafted rerun, first 100 GSM8K-500 items"
  "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38a_mtp EFF-h38c_mtp_rerun --limit 100 --json "$ROWS/h38c_same_config.json" | tee -a "$LOG"
  say "H38c (ii) divergence anatomy, reference = undrafted"
  "$TOOLS/divergence_probe.py" compare "$ROWS/traces_h38c_nodraft.jsonl" "$ROWS/traces_h38c_mtp.jsonl" --json "$ROWS/h38c_divergence.json" | tee -a "$LOG"
}
if [ "${DRY:-0}" = 1 ]; then run_cell() { echo "DRY cell $1"; echo "   start: $2"; shift 4; echo "   run:   $*"; }; cell_mtp() { :; }; cell_nodraft() { :; }; thermal_watch() { :; }; LOG=/dev/null; fi
say "=== hardening_c $what start"
if [ "$what" != analyze ] && [ "${DRY:-0}" != 1 ]; then thermal_watch & TW=$!; echo "$TW" >> "$EFF/our_pids.txt"; trap 'kill $TW 2>/dev/null' EXIT; fi
case "$what" in all) cells; analyze;; cells) cells;; analyze) analyze;; *) echo "usage: hardening_c.sh [all|cells|analyze]"; exit 2;; esac
say "=== hardening_c $what done"
