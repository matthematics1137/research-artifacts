#!/bin/bash
# E1 (research/TURING_EXL3_RESEARCH_2026-10-30.md 6.5): the llama.cpp bar on the 2080 Ti, undrafted.
# llama-bench at the plan's settings for the two 2-bit GGUFs; JSON lines to runs/llamabench.jsonl.
# Launch through the capped, guarded launcher:
#   tools/desktop_run.sh cmd e1-llamabench bash /opt/p8/tools/bench_llamacpp.sh [BUILD_DIR]
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
BIN=${1:-$L/src/llama.cpp/build-sm75-cu134/bin}
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64
for model in Qwen3.8-27B-UD-IQ2_XXS.gguf Qwen3.8-27B-UD-IQ2_S.gguf; do
  echo "== $(date -Is) $model with $BIN"
  nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader
  "$BIN/llama-bench" -m "$L/models/gguf/$model" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -p 512 -n 128 -r 3 -d 0,8192,32768 -o jsonl >> "$L/runs/llamabench.jsonl"
done
echo "== $(date -Is) done"
