#!/usr/bin/env bash
# H38d speed addendum (E50): the drafted daily config on the first 100 GSM8K-500 items with LD_LIBRARY_PATH unset (label EFF-h38d_mtp_clean1), compared with the mixed-library E47 run. GPU-gated; uses the campaign scratchpad launchers.
set -u; EFF=/path/to/scratch/eff; source "$EFF/cells.sh"
REPO=/path/to/lab-repo; TOOLS=$REPO/testsuite/results/efficiency/tools; ROWS=$REPO/testsuite/results/efficiency/rows; LOG=$EFF/logs/hardening.log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
cell() { pid=$(cat "$EFF/tabby.pid"); grep -oE '/[^ ]*(libcublas|libcublasLt|libnvrtc|libcusolver|libcudart)[^ ]*' /proc/$pid/maps | sort -u > "$ROWS/EFF-h38d_mtp_clean1_cuda_libs.txt"; say "EFF-h38d_mtp_clean1: system-12.1 libs loaded: $(grep -c '/usr/local/cuda' "$ROWS/EFF-h38d_mtp_clean1_cuda_libs.txt")"
  rm -rf "$RES/EFF-h38d_mtp_clean1"; ( cd "$KIT" && env -u LD_LIBRARY_PATH nice -n 5 ./run_eval.py --suite gsm8k_500 --label EFF-h38d_mtp_clean1 --base-url http://127.0.0.1:18120 --temperature 0 --limit 100 2>&1 | tail -1 | cut -c1-220 ) | tee -a "$LOG"; }
say "=== h38d speed addendum start"
run_cell h38d_mtp_clean1 "\"$EFF/tabby.sh\" start h38d_mtp_clean1 \"$REPO/serving/tabby-daily.yml\" 147 -u LD_LIBRARY_PATH EXL3_CKPT_PP=1024" "\"$EFF/tabby.sh\" stop" - cell
say "H38d speed: drafted clean vs drafted mixed (E47 run)"; "$TOOLS/hardening_analyze.py" gsm8k_500 EFF-h38a_mtp EFF-h38d_mtp_clean1 --limit 100 --json "$ROWS/h38d_mtp_clean_vs_mixed.json" | tee -a "$LOG"
say "=== h38d speed addendum done"
