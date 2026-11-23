#!/usr/bin/env bash
# Paper 8 output quality: one lane x one suite on the 2080 Ti, with Paper 7's E47 flags. Run ON the desktop,
# through the capped, guarded launcher:
#   bash $L/tools/desktop_run.sh cmd q-LANE-SUITE bash $L/tools/quality/run_quality.sh LANE SUITE [LIMIT] [LABEL]
#     LANE   a name from `bash $L/tools/lanes.sh list`
#     SUITE  gsm8k_500 | mmlu_500
#     LIMIT  first N items in dataset order (default 0 = all 500)
#     LABEL  default TURING-<lane>; use TURING-<lane>-r2 for a same-config reproducibility rerun
#   env (pass with `env PORT=8091 bash ...` inside the cmd): PORT (default 8090), PY (default the lab venv python;
#   the eval client is stdlib-only and never touches the GPU); TURING_EXTRA_BODY (a JSON file of extra request-body
#   fields, e.g. a logit_bias; recorded with its sha256; give the run its own LABEL); TURING_LANE_ENV / TURING_LANE_CTX / TURING_LCPP_EXTRA
#   pass through to lanes.sh, with E47's response shape added unless set (e47_shape below)
# E47 flags (testsuite/results/efficiency/tools/hardening.sh, evsuite): --temperature 0, reasoning effort medium
# (run_eval.py's default), --max-tokens 4096 (its default), concurrency 1; E47 passed the first explicitly and
# left the rest at their defaults, so the parsed values and the request bodies are the same.
# Steps: dataset sha256 check -> one-run lock -> lanes.sh start -> run_eval_turing.py -> lanes.sh stop (EXIT trap:
# also on failure, Ctrl-C or a guard stop) -> one JSON run record appended to $L/runs/quality_runs.jsonl.
# Resumable: rerunning the same command skips items that already have an error-free row. Nothing is overwritten
# and no file is discarded; per-run scratch files stay in $L/tmp.
set -uo pipefail
L=${TURING_LAB:-/opt/p8}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
lane=${1:-}; suite=${2:-}; limit=${3:-0}; label=${4:-TURING-${1:-}}
usage() { echo "usage: run_quality.sh LANE SUITE [LIMIT] [LABEL]   (SUITE: gsm8k_500 | mmlu_500)" >&2; exit 64; }
[ -n "$lane" ] && [ -n "$suite" ] || usage
case $suite in   # sha256 of the laptop's testsuite/evals/datasets files (check_fidelity.py re-derives both)
  gsm8k_500) want=7179bc41ad3e06c9f4b62ff1811fd272fa5ae82ddbee91dbf74d8278878ba344 ;;
  mmlu_500)  want=0a7e6b09f8d3ec6a13b00d02d8549a8e48d255a959be6034b932372f3cdea8da ;;
  *) usage ;;
esac
[[ $limit =~ ^[0-9]+$ ]] || usage
[[ $label =~ ^[A-Za-z0-9._-]+$ ]] || usage
PORT=${PORT:-8090}
PY=${PY:-$L/venv/bin/python}; [ -x "$PY" ] || PY=/usr/bin/python3
LANES=$L/tools/lanes.sh
DS=$L/evals/datasets; RES=$L/evals/results; RUNS=$L/runs; TMPD=$L/tmp
mkdir -p "$RES" "$RUNS/quality_gpu" "$TMPD" "$L/logs"
export XDG_CACHE_HOME=$L/cache TRITON_HOME=$L/cache/triton TRITON_CACHE_DIR=$L/cache/triton/cache \
       HF_HOME=$L/cache/huggingface PIP_CACHE_DIR=$L/cache/pip TMPDIR=$TMPD PYTHONDONTWRITEBYTECODE=1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}   # systemctl --user from inside a user unit

run_id=$(date +%Y%m%dT%H%M%S)-$label-$suite
t0=$(date +%s); t_start=$(date -Is)
status=init; eval_rc=null; start_rc=null; stop_rc=null; url=; own=0; sampler=; child=; got=; left=null
snap_before=; snap_loaded=; snap_end=; apps_loaded=; exec_start=; models=; engine=null; tools=null
summary_file=$TMPD/quality-summary-$run_id.json
gpu_csv=$RUNS/quality_gpu/$run_id.csv
unit=$(grep -o 'turing-run-[A-Za-z0-9_.-]*\.service' /proc/self/cgroup 2>/dev/null | head -1)
# This runner's own unit (turing-run-q-...) never touches the GPU. lanes.sh's "one lab unit at a time" check must
# skip its caller's unit, which it reads from its own /proc/self/cgroup (the line above), never from an environment
# variable a caller could set for any unit (README, "Needed in lanes.sh").
lane_record=null; shape_added=; extra_sha=
say() { echo "$(date -Is) [quality $label $suite] $*"; }

# key=value args -> one JSON object (keys ending in :j hold JSON values)
jobj() { "$PY" -c 'import json, sys
o = {}
for a in sys.argv[1:]:
    k, _, v = a.partition("=")
    if k.endswith(":j"):
        k = k[:-2]
        try:
            v = json.loads(v)
        except ValueError:
            pass
    o[k] = v
print(json.dumps(o, ensure_ascii=False))' "$@"; }
gpu_snap() { nvidia-smi --query-gpu=name,driver_version,pstate,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,clocks_event_reasons.active,utilization.gpu --format=csv,noheader 2>&1 | head -1; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | tr '\n' ';'; }
gitid() { local c d; c=$(git -C "$1" rev-parse HEAD 2>/dev/null) || { echo "n/a"; return; }
  d=$(git -C "$1" status --porcelain --untracked-files=no 2>/dev/null | wc -l); echo "$c dirty_files=$d"; }
pyver() { "$PY" -c 'import importlib.metadata as m, sys
try:
    print(m.version(sys.argv[1]))
except Exception:
    print("n/a")' "$1" 2>/dev/null; }
sha() { [ -f "$1" ] && sha256sum "$1" | cut -d" " -f1 || echo "n/a"; }
gpu_summary() { [ -s "$gpu_csv" ] || { echo null; return; }; "$PY" -c 'import json, sys
rows = [[x.strip() for x in l.split(",")] for l in open(sys.argv[1]) if l.count(",") >= 6]
def col(i):
    out = []
    for r in rows:
        try:
            out.append(float(r[i]))
        except (ValueError, IndexError):
            pass
    return out
def f(a, g):
    return round(g(a), 1) if a else None
mean = lambda a: sum(a) / len(a)
t, p, c, mu, u = col(1), col(2), col(3), col(5), col(6)
print(json.dumps({"samples": len(rows), "interval_s": 5, "temp_max_c": f(t, max), "power_mean_w": f(p, mean),
                  "power_max_w": f(p, max), "sm_clock_mean_mhz": f(c, mean), "sm_clock_min_mhz": f(c, min),
                  "mem_used_max_mib": f(mu, max), "util_mean_pct": f(u, mean)}))' "$gpu_csv"; }

# E47's response shape on every lane. E47's server (lab TabbyAPI 4a4f9f4) ran with the reasoning parser and tool
# parsing off (its defaults): content = '<reasoning>\n</think>\n\n<answer>', reasoning_content empty. Upstream TabbyAPI
# (since 2186cdb) parses reasoning and auto-detects a tool format by default, and llama-server splits reasoning
# (--reasoning-format deepseek). A split drops '</think>' from the text, so the same tokens get another content_sha256
# and encode count, and an item truncated inside the reasoning scores on empty content instead of E47's fallbacks.
# So the lane gets E47's settings through the lane overrides, unless the caller already set that key: TabbyAPI reads
# TABBY_MODEL_<KEY> for keys the lane's yml leaves out (the yml wins where it sets one); llama-server takes the last
# --reasoning-format. Sets shape_added (recorded in the run record).
e47_shape() {
  local kv add=
  case $1 in
    exl3-*)
      for kv in TABBY_MODEL_REASONING=false TABBY_MODEL_TOOL_FORMAT=; do
        [[ " ${TURING_LANE_ENV:-} " == *" ${kv%%=*}="* ]] || add="$add $kv"
      done
      add=${add# }
      [ -z "$add" ] || export TURING_LANE_ENV="$add${TURING_LANE_ENV:+ $TURING_LANE_ENV}" ;;
    lcpp-*)
      if [[ " ${TURING_LCPP_EXTRA:-} " != *" --reasoning-format"[\ =]* ]]; then
        add="--reasoning-format none"
        export TURING_LCPP_EXTRA="$add${TURING_LCPP_EXTRA:+ $TURING_LCPP_EXTRA}"
      fi ;;
  esac
  shape_added=$add
}

stop_child() {   # forward a stop to a background child and give it time to write its partial summary
  local pid=$1 i
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
}

cleanup() {
  local rc=$?
  trap - EXIT TERM INT HUP
  stop_child "$child"
  [ -n "$sampler" ] && kill "$sampler" 2>/dev/null
  case $status in eval_running) status=interrupted ;; lane_starting) status=lane_start_interrupted ;; esac
  snap_end=$(gpu_snap)
  if [ "$own" = 1 ]; then
    say "stopping lane $lane"
    bash "$LANES" stop "$lane" 9>&-; stop_rc=$?
  fi
  local esum t1
  esum=$(tail -n 1 "$summary_file" 2>/dev/null); [ -n "$esum" ] || esum=null
  t1=$(date +%s)
  jobj run_id="$run_id" lane="$lane" suite="$suite" label="$label" limit:j="$limit" port:j="$PORT" \
    base_url="$url" status="$status" exit_code:j="$rc" eval_rc:j="$eval_rc" lane_start_rc:j="$start_rc" \
    lane_stop_rc:j="$stop_rc" t_start="$t_start" t_end="$(date -Is)" duration_s:j="$((t1 - t0))" \
    host="$(hostname)" unit="$unit" dataset_sha256="$got" items_left_at_start:j="$left" \
    eval_flags="--temperature 0 --reasoning-effort medium --max-tokens 4096 --concurrency 1 --limit $limit" \
    engine:j="$engine" tools_sha256:j="$tools" lane_exec_start="$exec_start" served_models="$models" \
    lane_start_record:j="$lane_record" lane_env="${TURING_LANE_ENV:-}" lane_ctx_override="${TURING_LANE_CTX:-}" \
    lcpp_extra="${TURING_LCPP_EXTRA:-}" e47_shape_added="$shape_added" \
    extra_body_file="${TURING_EXTRA_BODY:-}" extra_body_sha256="$extra_sha" \
    gpu_before="$snap_before" gpu_loaded="$snap_loaded" gpu_end="$snap_end" gpu_apps_loaded="$apps_loaded" \
    gpu_samples:j="$(gpu_summary)" gpu_samples_csv="$gpu_csv" eval_summary:j="$esum" \
    results="$RES/$label/$suite.jsonl" sidecar="$RES/$label/$suite.full.jsonl" \
    >> "$RUNS/quality_runs.jsonl"
  say "run record appended to $RUNS/quality_runs.jsonl (status $status, exit $rc)"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

# 1. dataset identity (the validator copies the laptop files to $DS; this script only checks them)
got=$(sha256sum "$DS/$suite.jsonl" 2>/dev/null | cut -d" " -f1)
if [ "$got" != "$want" ]; then status=dataset_mismatch; say "REFUSED: $DS/$suite.jsonl sha256 '${got:-missing}' != $want"; exit 65; fi
# optional extra request-body fields (e.g. a logit_bias file); unset = E47's bodies exactly
extra_args=()
if [ -n "${TURING_EXTRA_BODY:-}" ]; then
  [ -f "$TURING_EXTRA_BODY" ] || { status=no_extra_body; say "REFUSED: TURING_EXTRA_BODY file '$TURING_EXTRA_BODY' missing"; exit 66; }
  extra_sha=$(sha "$TURING_EXTRA_BODY"); extra_args=(--extra-body-file "$TURING_EXTRA_BODY")
  say "extra request-body fields from $TURING_EXTRA_BODY (sha256 $extra_sha)"
fi

# 2. one quality run at a time, and never next to another lane (only one GPU job on the desktop)
exec 9>>"$RUNS/quality.lock"   # children get `9>&-` so a lingering child can never hold the lock
if ! flock -n 9; then status=locked; say "another run_quality.sh holds $RUNS/quality.lock"; exit 75; fi
[ -f "$LANES" ] || { status=no_lanes_sh; say "missing $LANES"; exit 69; }
if ! bash "$LANES" list 9>&- | grep -qxF -- "$lane"; then status=unknown_lane; say "unknown lane '$lane' (lanes.sh list)"; exit 64; fi
left=$("$PY" "$HERE/run_eval_turing.py" --suite "$suite" --label "$label" --limit "$limit" \
  --results-dir "$RES" --datasets-dir "$DS" --todo 9>&- 2>&1 | tail -n 1)
if ! [[ $left =~ ^[0-9]+$ ]]; then status=todo_failed; say "could not count the items left: $left"; left=null; exit 70; fi
if [ "$left" = 0 ]; then status=already_complete; say "every item already has an error-free row in $RES/$label/$suite.jsonl; lane not started"; exit 0; fi
active=$(systemctl --user list-units --state=active,activating,deactivating --plain --no-legend 'turing-run-*' 2>/dev/null \
  | awk -v self="$unit" '$1 != self {print $1}' | tr '\n' ' ')
if [ -n "${active// /}" ]; then status=lane_busy; say "another lab unit is active ($active); one GPU job at a time, not touching it"; exit 75; fi

# 3. E47's response shape for this lane, and the identity of what is about to run
e47_shape "$lane"
[ -z "$shape_added" ] || say "E47 response shape: added '$shape_added' to the lane overrides"
engine=$(jobj exllamav3="$(pyver exllamav3)" torch="$(pyver torch)" triton="$(pyver triton)" \
  tabbyAPI="$(gitid "$L/src/tabbyAPI")" llama_cpp="$(gitid "$L/src/llama.cpp")" \
  llama_cpp_v040_mtpshared="$(gitid "$L/src/llama.cpp-v0.4.0-mtpshared")" \
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" \
  client_python="$("$PY" -V 2>&1)")
tools=$(jobj run_eval_turing.py="$(sha "$HERE/run_eval_turing.py")" run_quality.sh="$(sha "$HERE/run_quality.sh")" \
  lanes.sh="$(sha "$LANES")" desktop_run.sh="$(sha "$L/tools/desktop_run.sh")" desktop_guard.py="$(sha "$L/tools/desktop_guard.py")")
snap_before=$(gpu_snap)

# 4. start the lane (blocks until it answers a health check)
say "starting lane $lane on port $PORT"
status=lane_starting; own=1
bash "$LANES" start "$lane" "$PORT" > "$TMPD/quality-lane-start-$run_id.out" 9>&- &
child=$!; wait "$child"; start_rc=$?; child=
url=$(grep -E '^https?://' "$TMPD/quality-lane-start-$run_id.out" | tail -n 1)
if [ "$start_rc" != 0 ]; then status=lane_start_failed; say "lanes.sh start failed (rc $start_rc)"; exit 69; fi
[ -n "$url" ] || url=http://localhost:$PORT
url=${url%/}
snap_loaded=$(gpu_snap); apps_loaded=$(gpu_apps)
lane_record=$(tail -n 1 "$RUNS/lanes.jsonl" 2>/dev/null); [[ $lane_record == *"\"lane\":\"$lane\""* ]] || lane_record=null
exec_start=$(systemctl --user show -p ExecStart --value "turing-run-srv-$lane" 2>/dev/null | head -c 4000)
models=$("$PY" -c 'import sys, urllib.request
print(urllib.request.urlopen(sys.argv[1] + "/v1/models", timeout=20).read().decode("utf-8", "replace")[:2000])' "$url" 2>&1 | head -c 2000)
say "lane up at $url; $snap_loaded"

# 5. the eval (E47 flags), with a 5 s GPU sampler for clocks, power and temperature
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -l 5 >> "$gpu_csv" 2>/dev/null 9>&- &
sampler=$!
status=eval_running
"$PY" "$HERE/run_eval_turing.py" --suite "$suite" --label "$label" --base-url "$url" \
  --temperature 0 --reasoning-effort medium --max-tokens 4096 --concurrency 1 --limit "$limit" \
  --results-dir "$RES" --datasets-dir "$DS" --summary-out "$summary_file" "${extra_args[@]}" 9>&- &
child=$!; wait "$child"; eval_rc=$?; child=
case $eval_rc in
  0) status=done ;; 3) status=server_gone ;; 130) status=interrupted ;; 65) status=dataset_mismatch ;; *) status=eval_failed ;;
esac
say "eval finished: $status (rc $eval_rc)"
exit "$eval_rc"
