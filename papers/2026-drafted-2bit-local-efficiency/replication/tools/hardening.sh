#!/usr/bin/env bash
# H38 hardening block — GPU-GATED. Run only when Matt has released the GPU (research/GOAL-2026-09-09-nogpu.md, item 3).
# Pre-registration: research/efficiency-hypotheses-2026-09-06.md H38a / H38b. Temperature 0, n >= 500, one cell per suite
# so a yield (foreign GPU process, or a thermal stop) only re-runs the suite it interrupted.
#   H38a  EXL3 SC_2.00 daily config WITH the MTP drafter vs WITHOUT: GSM8K-500 + MMLU-500 (1,000 paired items)
#         -> output identity (sha256) + paired accuracy.                  ~1.3 h + 2.1 h (gsm8k), ~0.9 h + 1.4 h (mmlu)
#   H38b  llama.cpp UD-Q4_K_XL (NGL_B layers offloaded, shared MTP sidecar) vs the H38a drafted run on the first
#         LIMIT_B GSM8K-500 items (dataset order).                         ~2.5-3.5 h at 200 items, ~8 tok/s
#   usage: hardening.sh [all|h38a|h38b|analyze]     env: LIMIT_B=200  NGL_B=30  EFF=<campaign scratchpad eff dir>
# Uses the campaign's scoped launchers (avail gate, yield watcher, our_pids) — never touches foreign GPU processes,
# never edits engines/tabbyAPI/config.yml. Results: testsuite/evals/results/EFF-h38*/ and rows/h38*.json.
set -u
EFF=${EFF:-/path/to/scratch/eff}
source "$EFF/cells.sh"
REPO=/path/to/lab-repo; TOOLS=$REPO/testsuite/results/efficiency/tools; ROWS=$REPO/testsuite/results/efficiency/rows; SERV=$REPO/serving
M=/path/to/models/qwen38; PB=$REPO/engines/llama.cpp-v0.4.0-mtpshared/build/bin/llama-server
MTPS=$M/MTP-shared/mtp-Qwen3.8-27B-Q4_0-shared.gguf
LIMIT_B=${LIMIT_B:-200}; NGL_B=${NGL_B:-30}; what=${1:-all}
LOG=$EFF/logs/hardening.log; mkdir -p "$EFF/logs" "$ROWS"
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

# Thermal guard (standing rule): GPU >= 90 C or CPU package >= 98 C on 3 consecutive 10-s reads -> stop OUR server pid
# and raise the yield flag, so run_cell cleans the partial suite and retries after the cool-down.
thermal_watch() { local hot=0
  while :; do
    g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    c=$(sensors 2>/dev/null | awk '/Package id 0|Tctl/ {gsub(/[^0-9.]/,"",$4); print $4; exit}')
    if [ "${g:-0}" -ge 90 ] 2>/dev/null || [ "${c%.*}" -ge 98 ] 2>/dev/null; then hot=$((hot+1)); else hot=0; fi
    if [ $hot -ge 3 ]; then
      say "THERMAL STOP gpu=${g}C pkg=${c}C"; echo YIELDED-THERMAL > "$EFF/yield.flag"
      for f in "$EFF/tabby.pid" "$EFF/server.pid"; do [ -f "$f" ] && kill -INT "$(cat "$f")" 2>/dev/null; done
      hot=0; sleep 120
    fi
    sleep 10
  done
}

# remove a partial suite from a label (per-item file + its summary rows) so a retried cell starts clean
reset_suite() { local L=$1 S=$2; rm -f "$RES/$L/$S.jsonl"; python3 - "$L" "$S" <<'PY'
import json, sys
p = '/path/to/lab-repo/testsuite/evals/results/summary.jsonl'; L, S = sys.argv[1], sys.argv[2]
lines = open(p).read().splitlines(True)
open(p, 'w').writelines([l for l in lines if not (l.strip() and json.loads(l).get('label') == L and json.loads(l).get('suite') == S)])
PY
}
# one suite at temperature 0 against a running server (idempotent: clears the partial suite first)
evsuite() { local L=$1 S=$2 port=$3 lim=${4:-0}; reset_suite "$L" "$S"; say "eval $L $S (limit $lim) on :$port"
  ( cd "$KIT" && nice -n 5 ./run_eval.py --suite "$S" --label "$L" --base-url "http://127.0.0.1:$port" --temperature 0 --limit "$lim" 2>&1 | tail -1 | cut -c1-220 ) | tee -a "$LOG"; }

h38a() { for s in gsm8k_500 mmlu_500; do
  run_cell "h38a_mtp_$s" "\"$EFF/tabby.sh\" start h38a_mtp_$s \"$SERV/tabby-daily.yml\" 147 EXL3_CKPT_PP=1024" "\"$EFF/tabby.sh\" stop" - evsuite EFF-h38a_mtp "$s" 18120
  run_cell "h38a_nodraft_$s" "\"$EFF/tabby.sh\" start h38a_nodraft_$s \"$TOOLS/tabby_h38_nodraft.yml\" 147" "\"$EFF/tabby.sh\" stop" - evsuite EFF-h38a_nodraft "$s" 18120
done; }

h38b() {
  run_cell h38b_q4kxl "BIN=$PB \"$EFF/srv.sh\" start h38b_q4kxl 9200 -- -m $M/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MTPS --spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 -ctkd q8_0 -ctvd q8_0 -ngl $NGL_B -np 1 -c 8192 -ctk q8_0 -ctv q8_0 -fa on --jinja -ctxcp 4 -cms 1024 -b 512 -ub 256 -t 20" "\"$EFF/srv.sh\" stop" - evsuite EFF-h38b_q4kxl_mtp gsm8k_500 18110 "$LIMIT_B"
}

analyze() {
  for s in gsm8k_500 mmlu_500; do "$TOOLS/hardening_analyze.py" "$s" EFF-h38a_nodraft EFF-h38a_mtp --hyp h38a --json "$ROWS/h38a_$s.json" | tee -a "$LOG"; done
  "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38b_q4kxl_mtp EFF-h38a_mtp --limit "$LIMIT_B" --hyp h38b --json "$ROWS/h38b_gsm8k.json" | tee -a "$LOG"
}

if [ "${DRY:-0}" = 1 ]; then   # DRY=1: print every cell's start command and eval command, start nothing
  run_cell() { echo "DRY cell $1"; echo "   start: $2"; shift 4; echo "   run:   $*"; }
  evsuite() { :; }; thermal_watch() { :; }; LOG=/dev/null
fi
say "=== hardening $what start (LIMIT_B=$LIMIT_B NGL_B=$NGL_B)"
if [ "$what" != analyze ] && [ "${DRY:-0}" != 1 ]; then thermal_watch & TW=$!; echo "$TW" >> "$EFF/our_pids.txt"; trap 'kill $TW 2>/dev/null' EXIT; fi
case "$what" in
  all) h38a; h38b; analyze;;
  h38a) h38a;;  h38b) h38b;;  analyze) analyze;;
  *) echo "usage: hardening.sh [all|h38a|h38b|analyze]"; exit 2;;
esac
say "=== hardening $what done"
