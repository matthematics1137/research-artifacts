#!/bin/bash
# Launch a capped Turing-lab run on p8-host (run this ON the desktop, from /opt/p8/tools).
#   desktop_run.sh guard                 start the yield guard (idempotent)
#   desktop_run.sh smoke LABEL [ARGS..]  run tools/smoke.py as unit turing-run-LABEL with exllamav3 ARGS
# Caps leave the machine to Matt's agents: 4 CPU cores, 20 GiB RAM (no swap), lowest IO priority, and an
# OOM score that makes this the first process the kernel stops. The guard stops turing-run* units on
# memory pressure, a low disk, a hot GPU or another program on the GPU. Nothing is deleted.
set -euo pipefail
# Everything in one place on the desktop's ssd2 drive; nothing on the 500 GB main drive (Matt, 2026-10-30).
L=${TURING_LAB:-/opt/p8}
# Every unit gets the lab's cache and temp locations, so no library falls back to ~/.triton, ~/.cache, ~/.nv or /tmp
# (2026-10-30 validation: under cmd, Triton compiled its launchers in /tmp; nothing is cleaned by our tools, and a
# library's own temporary directories now land in $L/tmp).
LAB_ENV=(-E XDG_CACHE_HOME="$L/cache" -E TRITON_HOME="$L/cache/triton" -E TRITON_CACHE_DIR="$L/cache/triton/cache"
         -E HF_HOME="$L/cache/huggingface" -E PIP_CACHE_DIR="$L/cache/pip" -E CUDA_CACHE_PATH="$L/cache/nv"
         -E TMPDIR="$L/tmp")
case ${1:-} in
  smoke|cmd) mkdir -p "$L/tmp" "$L/cache/nv" "$L/logs" ;;
esac
case ${1:-} in
  guard)
    systemctl --user is-active --quiet turing-guard && { echo "guard already running"; exit 0; }
    systemd-run --user --unit=turing-guard --collect -p MemoryMax=256M -p Nice=5 \
      /usr/bin/python3 "$L/tools/desktop_guard.py"
    ;;
  smoke)
    label=$2; shift 2
    systemctl --user is-active --quiet turing-guard || { echo "start the guard first"; exit 1; }
    systemd-run --user --unit="turing-run-$label" --collect \
      -p MemoryMax=20G -p MemorySwapMax=0 -p CPUQuota=400% -p Nice=10 -p IOSchedulingClass=idle \
      -p OOMScoreAdjust=800 -p StandardOutput="append:$L/logs/run-$label.log" \
      -p StandardError="append:$L/logs/run-$label.log" \
      -E CUDA_VISIBLE_DEVICES=0 -E PYTHONUNBUFFERED=1 -E HF_HUB_OFFLINE=1 "${LAB_ENV[@]}" \
      "$L/venv/bin/python" "$L/tools/smoke.py" --label "$label" -m "$L/models/qwen38-exl3-2.0" "$@"
    ;;
  cmd)  # any GPU command (llama-bench, llama-server, ...) under the same caps and guard
    label=$2; shift 2
    systemctl --user is-active --quiet turing-guard || { echo "start the guard first"; exit 1; }
    systemd-run --user --unit="turing-run-$label" --collect --working-directory="$L" \
      -p MemoryMax=20G -p MemorySwapMax=0 -p CPUQuota=400% -p Nice=10 -p IOSchedulingClass=idle \
      -p OOMScoreAdjust=800 -p StandardOutput="append:$L/logs/run-$label.log" \
      -p StandardError="append:$L/logs/run-$label.log" \
      -E CUDA_VISIBLE_DEVICES=0 "${LAB_ENV[@]}" "$@"
    ;;
  *) echo "usage: $0 guard | smoke LABEL [exllamav3 args] | cmd LABEL COMMAND..."; exit 64 ;;
esac
