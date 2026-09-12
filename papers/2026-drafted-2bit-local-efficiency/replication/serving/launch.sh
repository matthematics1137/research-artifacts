#!/usr/bin/env bash
# Launch the campaign's recommended TabbyAPI configuration for Qwen3.8-27B on this laptop.
#   serving/launch.sh daily|2agents [--port N] [--ckpt-pp N] [--stock-engine] [--dry-run]
# daily   : 1 slot, 32K, draft 6 (single user)            2agents : 2 slots, 24K (two concurrent agents)
# --ckpt-pp N     recurrent ingestion checkpoint every N tokens (default 1024; E43: edit cost proportional to
#                 distance from the prompt end instead of a full re-prefill; +2-3% cold prefill, +148 MiB host RAM per
#                 checkpoint). Needs the lab-patched engine copy; ignored with --stock-engine.
# --stock-engine  use the pinned exl3-env's own exllamav3 (1.4.3) instead of engines/exl3-1.4.7-lab (loses the
#                 checkpoint knob and the E17-E45 measurements' engine version).
# --dry-run       validate everything and print the command without starting a server.
# Never edits engines/tabbyAPI/config.yml (the saved default); this is a separate process on its own port.
set -u
REPO=/path/to/lab-repo
TABBY=$REPO/engines/tabbyAPI
PY=$REPO/engines/exl3-env/bin/python
LAB=$REPO/engines/exl3-1.4.7-lab
mode=${1:-}; shift || true
port=18120; ckpt=1024; stock=0; dry=0
while [ $# -gt 0 ]; do case "$1" in
  --port) port=$2; shift 2;; --ckpt-pp) ckpt=$2; shift 2;; --stock-engine) stock=1; shift;; --dry-run) dry=1; shift;;
  *) echo "unknown option $1"; exit 2;; esac; done
case "$mode" in daily) cfg=$REPO/serving/tabby-daily.yml;; 2agents) cfg=$REPO/serving/tabby-2agents.yml;; *) sed -n '2,12p' "$0"; exit 2;; esac
fail=0; ok() { echo "  ok   $1"; }; bad() { echo "  FAIL $1"; fail=1; }
echo "== preflight ($mode)"
[ -f "$cfg" ] && ok "config $cfg" || bad "config missing: $cfg"
[ -x "$PY" ] && ok "python $PY" || bad "exl3-env python missing"
[ -f "$TABBY/main.py" ] && ok "TabbyAPI $TABBY" || bad "TabbyAPI missing"
[ -d /path/to/models/qwen38-exl3-2.0 ] && ok "model ~/models/qwen38-exl3-2.0" || bad "model dir missing"
if [ $stock = 0 ]; then
  [ -d "$LAB/exllamav3" ] && ok "lab engine copy $LAB (exllamav3 1.4.7 + lab patches)" || bad "lab engine copy missing: rebuild from the 1.4.7 wheel + testsuite/results/efficiency/tools/*.patch"
  grep -q EXL3_CKPT_PP "$LAB/exllamav3/generator/generator.py" 2>/dev/null && ok "checkpoint-interval patch present" || bad "checkpoint-interval patch not in the lab copy"
  [ $((ckpt % 256)) -eq 0 ] && ok "ckpt-pp $ckpt is a multiple of the 256-token page" || bad "ckpt-pp must be a multiple of 256"
fi
if ss -ltn 2>/dev/null | grep -q ":$port "; then bad "port $port already in use"; else ok "port $port free"; fi
busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | awk -F, '{gsub(/ /,"",$2); if ($2+0 > 300) n++} END {print n+0}')
[ "$busy" = 0 ] && ok "GPU free of other model processes" || bad "GPU has $busy other process(es) using >300 MiB — this launcher never evicts anyone"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${free:-0}" -ge 9500 ] && ok "free VRAM ${free} MiB" || bad "need ~9.5 GiB free VRAM, have ${free:-?}"
sed -i "s/^  port: .*/  port: $port/" "$cfg" 2>/dev/null
[ $fail = 1 ] && { echo "preflight failed"; exit 1; }
env_line=""; [ $stock = 0 ] && env_line="PYTHONPATH=$LAB EXL3_CKPT_PP=$ckpt"
cmd="cd $TABBY && env $env_line nice -n 5 $PY main.py --config $cfg"
echo "== command"; echo "  $cmd"
[ $dry = 1 ] && { echo "(dry run — not started)"; exit 0; }
echo "== starting on port $port (Ctrl-C to stop; logs in engines/tabbyAPI/logs/)"
cd "$TABBY" || exit 1
# LD_LIBRARY_PATH is unset on purpose: ~/.bashrc exports /usr/local/cuda-12.1/lib64, under which torch 2.10+cu128 picked up the
# system libcublasLt/libnvrtc 12.1 instead of its bundled 12.8 libraries (found 2026-09-10; it broke CUDA Cholesky and mixed the
# library set of every server launched from a shell). The pip-bundled CUDA libraries need no library path.
# shellcheck disable=SC2086  # env_line is intentionally word-split (two VAR=value words, or nothing)
exec env -u LD_LIBRARY_PATH $env_line nice -n 5 "$PY" main.py --config "$cfg"
