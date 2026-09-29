# cells.sh — sourced by Phase-B drivers. A cell = start a server, run a command, stop. If the yield watcher fired
# during the cell (Matt or another agent took the GPU), the cell's partial eval rows are removed and the cell is
# retried after the GPU is free again (up to 3 attempts, waiting 2 minutes between checks).
EFF=/path/to/scratch/eff
KIT=/path/to/lab-repo/testsuite/evals
RES=/path/to/lab-repo/testsuite/evals/results
yielded() { [ -f "$EFF/yield.flag" ]; }
yield_clear() { rm -f "$EFF/yield.flag"; }
wait_gpu() { until "$EFF/avail.sh" 9000 2 >/dev/null; do sleep 60; done; }
cleanup_label() { local L=$1; [ -d "$RES/$L" ] && rm -rf "$RES/$L"; python3 - "$L" <<'PY'
import json,sys
p='/path/to/lab-repo/testsuite/evals/results/summary.jsonl'; L=sys.argv[1]
lines=open(p).read().splitlines(True)
keep=[l for l in lines if not (l.strip() and json.loads(l).get('label')==L)]
open(p,'w').writelines(keep)
PY
}
# run_cell <name> <start-cmd-string> <stop-cmd-string> <eval-label-or-"-"> <command...>
run_cell() { local name=$1 start=$2 stop=$3 label=$4; shift 4
  for attempt in 1 2 3; do
    yield_clear; wait_gpu
    if ! eval "$start" > "$EFF/logs/cell_${name}_start$attempt.out" 2>&1; then echo "  cell $name: start failed (attempt $attempt)"; eval "$stop" >/dev/null 2>&1; sleep 120; continue; fi
    grep -E 'READY|MiB' "$EFF/logs/cell_${name}_start$attempt.out" | head -2 | cut -c1-120
    "$@"
    if yielded; then
      echo "  cell $name: YIELDED to a foreign GPU process during attempt $attempt — cleaning partial rows, will retry"
      [ "$label" != "-" ] && cleanup_label "$label"
      eval "$stop" >/dev/null 2>&1; sleep 120; continue
    fi
    eval "$stop" >/dev/null 2>&1; return 0
  done
  echo "  cell $name: gave up after 3 attempts"; return 1
}
ev() { ( cd "$KIT" && nice -n 5 ./run_eval.py "$@" 2>&1 | tail -1 | cut -c1-220 ); }
