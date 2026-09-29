#!/usr/bin/env bash
# Scoped TabbyAPI launcher for the efficiency campaign (port 18120, isolated config).
#   tabby.sh start <label> <config.yml> [exl3:147|143] [extra env...]
#   tabby.sh stop
#   tabby.sh vram
set -u
EFF=/path/to/scratch/eff
OURS=$EFF/our_pids.txt
PY=/path/to/lab-repo/engines/exl3-env/bin/python
TABBY=/path/to/lab-repo/engines/tabbyAPI
LOGDIR=$EFF/logs; mkdir -p "$LOGDIR"
cmd=${1:-}; shift || true
case "$cmd" in
  start)
    label=$1; cfg=$2; ver=${3:-143}; shift 3 2>/dev/null || shift $#
    if [ -f "$EFF/tabby.pid" ] && kill -0 "$(cat "$EFF/tabby.pid")" 2>/dev/null; then echo "tabby already running pid $(cat "$EFF/tabby.pid")"; exit 2; fi
    waited=0
    until "$EFF/avail.sh" 9000 2; do
      waited=$((waited+30)); [ $waited -ge 1800 ] && { echo "AVAIL GATE: yielded 30 min, giving up"; exit 3; }
      sleep 30
    done
    log="$LOGDIR/tabby_${label}.log"
    pp=""; [ "$ver" = 147 ] && pp="$EFF/exl3_147"
    ( cd "$TABBY" && PYTHONPATH="$pp" setsid nohup env "$@" nice -n 5 "$PY" main.py --config "$cfg" > "$log" 2>&1 & )
    # setsid may fork, so $! is not reliable: find the real python pid by its config path
    pid=""
    for i in $(seq 1 20); do pid=$(pgrep -f "main.py --config $cfg" | head -1); [ -n "$pid" ] && break; sleep 0.5; done
    [ -z "$pid" ] && { echo "FAILED to find tabby pid"; exit 1; }
    echo "$pid" > "$EFF/tabby.pid"; echo "$pid" >> "$OURS"
    ( setsid nohup "$EFF/yield_watch.sh" "$pid" > /dev/null 2>&1 & )  # GPU-yield watcher: stops us if a foreign GPU process appears
    for i in $(seq 1 300); do
      if curl -sf -m 2 http://127.0.0.1:18120/health >/dev/null 2>&1; then echo "READY pid=$pid label=$label after ${i}s"; grep -E 'exllamav3 version|draft|MTP|Draft|Model successfully|cache|Warning|Error' "$log" | head -12 | cut -c1-160; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid"; exit 0; fi
      kill -0 "$pid" 2>/dev/null || { echo "DIED label=$label"; tail -25 "$log" | cut -c1-200; rm -f "$EFF/tabby.pid"; exit 1; }
      sleep 1
    done
    echo "TIMEOUT"; kill -INT "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; rm -f "$EFF/tabby.pid"; exit 1;;
  stop)
    [ -f "$EFF/tabby.pid" ] || { echo "no tabby"; exit 0; }
    pid=$(cat "$EFF/tabby.pid")
    if kill -0 "$pid" 2>/dev/null; then kill -INT "$pid" 2>/dev/null; for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done; kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null; fi
    for w in $(pgrep -f "yield_watch.sh $pid"); do kill $w 2>/dev/null; done; rm -f "$EFF/tabby.pid"; echo "stopped $pid"; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader;;
  vram)
    pid=$(cat "$EFF/tabby.pid" 2>/dev/null); nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid" || echo "no tabby";;
  *) echo "usage: tabby.sh start|stop|vram"; exit 1;;
esac
