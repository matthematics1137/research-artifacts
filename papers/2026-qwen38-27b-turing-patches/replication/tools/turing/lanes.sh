#!/bin/bash
# Turing lab serving lanes on p8-host's RTX 2080 Ti (run this ON the desktop, from $L/tools).
# One OpenAI-compatible server at a time, each a capped, guarded systemd user unit turing-run-srv-LANE started
# through desktop_run.sh cmd (4 cores, 20 GiB RAM, nice, idle IO, OOMScoreAdjust 800; watched by turing-guard).
#   lanes.sh list                 lane names, one per line
#   lanes.sh start LANE [PORT]    start LANE on localhost:PORT (default 8090); blocks until /health answers 200,
#                                 /v1/models answers and a 16-token probe generates (timeout ~600 s); prints the base
#                                 URL on stdout (everything else goes to stderr); non-zero exit on failure, and on
#                                 an interrupt (INT, TERM or HUP: Ctrl-C, a dropped ssh session, a wrapper's
#                                 timeout), in every case after stopping the unit and recording the start
#   lanes.sh stop LANE            systemctl --user stop turing-run-srv-LANE (idempotent)
#   lanes.sh status               one line per active srv lane: LANE <tab> state <tab> base URL <tab> health=CODE
#   lanes.sh show LANE [PORT]     print what a start would run (command, and the TabbyAPI config); starts nothing,
#                                 writes nothing
# Lanes (details and the laptop runs they reproduce: tools/turing/lanes_README.md):
#   exl3-mtp, exl3-nodraft            upstream TabbyAPI + exllamav3 1.6.5, EXL3 2.0 bpw (configs in $L/tools/tabby/);
#                                     GDN recurrent state fp16, the upstream default (plan 6.2 item 6: speed numbers)
#   exl3-mtp-fp32, exl3-nodraft-fp32  the same configs plus recurrent_state_fp32: true, the laptop 1.4.7's fp32 state
#                                     (plan 6.2 item 6: every comparison with the 4080's E47 numbers)
#   lcpp-iq2xxs, lcpp-iq2s            mainline llama-server (d63fbfe7), undrafted
#   lcpp-iq2s-mtp                     v0.4.0-mtpshared llama-server + the stripped shared-embedding MTP sidecar
#   lcpp-iq2xxs-dflash2               v0.4.0-mtpshared llama-server + the DFlash2 drafter (8K context, see README)
#   lcpp-iq2xxs-v040, lcpp-iq2s-v040  v0.4.0-mtpshared llama-server, undrafted: same-build controls for the two lanes
#                                     above, so a drafter speedup does not include the v0.4.0 -> mainline build change
#   lcpp-iq2xxs-dflash2-main          mainline llama-server + the DFlash2 drafter (8K): the drafter on the newer build
# Every llama.cpp lane passes --min-p 0. llama-server defaults to min_p 0.05 and upstream TabbyAPI (no override
# preset) to 0; run_eval.py's body has no min_p, so sampled (temperature > 0) runs would otherwise truncate differently.
# Optional environment, recorded with every start in $L/runs/lanes.jsonl:
#   TURING_LANE_CTX=N          context instead of the lane default (16384; the two dflash2 lanes 8192)
#   TURING_LCPP_EXTRA="..."    extra llama-server flags, appended last so they override (llama.cpp lanes)
#   TURING_LANE_ENV="A=1 B=2"  extra environment for the server process, e.g. GGML_CUDA_DISABLE_GRAPHS=1. TabbyAPI
#                              merges TABBY_<SECTION>_<KEY> variables under the --config file, so they only set keys
#                              that the lane's config leaves out
#   TURING_LANE_TIMEOUT=S      start-up wait in seconds (default 600)
#   TURING_LANE_PROBE=0        skip the post-health generation probe
# Values must not contain '$': systemd-run expands ${VAR} in unit command lines.
# Nothing here deletes files. Logs: $L/logs/run-srv-LANE.log (server console), $L/logs/tabby-LANE/ (TabbyAPI's own
# file log), $L/logs/lanes/ (the exact TabbyAPI config of each start).
set -uo pipefail
L=${TURING_LAB:-/opt/p8}  # the desktop's ssd2 drive, never the main drive
LANES="exl3-mtp exl3-nodraft lcpp-iq2xxs lcpp-iq2s lcpp-iq2s-mtp lcpp-iq2xxs-dflash2
       exl3-mtp-fp32 exl3-nodraft-fp32 lcpp-iq2xxs-v040 lcpp-iq2s-v040 lcpp-iq2xxs-dflash2-main"
LANES=$(echo $LANES)  # one line, single spaces
MAIN=$L/src/llama.cpp/build-sm75-cu134/bin                   # mainline d63fbfe7, built for sm_75 with CUDA 13.4
FORK=$L/src/llama.cpp-v0.4.0-mtpshared/build-sm75-cu134/bin  # v0.4.0 + research/patches/...-mtp-shared-embd.patch
G=$L/models/gguf
CUDA_LIB=/usr/local/cuda-13.4/lib64
DEFAULT_PORT=8090
# The laptop's llama.cpp lane flags (testsuite/results/efficiency/tools/b2.sh COMMON = e8_choice.txt) plus the
# --host/--metrics its srv.sh added. -t 4 instead of srv.sh's -t 8: the unit has 4 cores (CPUQuota=400%).
# --min-p 0 is not from the laptop (its llama-servers ran the 0.05 default); it matches the EXL3 lanes' min_p 0, and
# greedy (temperature 0) requests are unaffected. TURING_LCPP_EXTRA="--min-p 0.05" restores the laptop's default.
LCPP_COMMON="-ngl 99 -np 1 -ctk q8_0 -ctv q8_0 -fa on --jinja -ctxcp 4 -cms 1024 -b 512 -ub 256 --min-p 0 --metrics -t 4 --host localhost"
# TabbyAPI boot: upstream main.py, unchanged, with two loguru adjustments made before it starts: its file log goes to
# $L/logs/tabby-LANE/ instead of src/tabbyAPI/logs/, and without the 1-week retention and zip compression, both of
# which make loguru delete files. Upstream has no other way to configure that sink. One line, no '$' or '%'.
TABBY_BOOT='import os, sys, runpy; from loguru import logger; d = os.environ.get("TURING_TABBY_LOGDIR"); a = logger.add; f = lambda s: (d + s[4:]) if (d and isinstance(s, str) and s.startswith("logs/")) else s; logger.add = lambda *p, **k: a(*((f(p[0]),) + p[1:] if p else ()), **{x: (f(y) if x == "sink" else y) for x, y in k.items() if x not in ("retention", "compression")}); sys.argv = ["main.py"] + sys.argv[1:]; runpy.run_path("main.py", run_name="__main__")'
PROBE_BODY='{"model":"local","messages":[{"role":"user","content":"Reply with one word: ready."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'

die() { echo "lanes.sh: $*" >&2; exit 1; }
note() { echo "lanes.sh: $*" >&2; }
is_lane() { case " $LANES " in *" $1 "*) return 0 ;; esac; return 1; }
lane_kind() { case $1 in exl3-*) echo tabby ;; *) echo lcpp ;; esac; }
tabby_base() { echo "${1%-fp32}"; }  # the yml a TabbyAPI lane is built from: exl3-mtp-fp32 -> exl3-mtp
lane_ctx() {
  local d=16384
  # DFlash2 costs ~2.4 GiB resident; the laptop ran this lane at 8K (9,986 MiB after the first generation on a
  # 12 GB card) and 16K is projected at ~10.9 GiB against the 2080 Ti's ~10.4 GiB free. Not yet measured on the
  # desktop: TURING_LANE_CTX=16384 tries it (README, "Choices made").
  case $1 in lcpp-iq2xxs-dflash2*) d=8192 ;; esac
  echo "${TURING_LANE_CTX:-$d}"
}
lane_rs() {  # LANE -> the GDN recurrent-state precision its start must log; empty when there is nothing to check
  case $1 in
    *-fp32) echo fp32 ;;
    # the base EXL3 lanes are fp16 unless TURING_LANE_ENV sets the TabbyAPI variable for the key they leave out
    exl3-*) case " ${TURING_LANE_ENV:-} " in *TABBY_MODEL_RECURRENT_STATE_FP32=*) ;; *) echo fp16 ;; esac ;;
  esac
}
rs_logged() {  # the precision TabbyAPI logged at load ("Recurrent state storage: fp16|fp32") in this start's part of
  # the console log, else in its file log files written since this start's config; its /v1/model card has no such
  # field. Empty for llama.cpp lanes or when the line is missing.
  [ "$(lane_kind "$lane")" = tabby ] || return 0
  local re='Recurrent state storage: fp(16|32)' got
  got=$(tail -c +"$(( ${log_off:-0} + 1 ))" "$L/logs/run-srv-$lane.log" 2>/dev/null | grep -aoE "$re" | tail -n 1)
  if [ -z "$got" ] && [ -n "${cfg:-}" ] && [ -f "$cfg" ]; then
    got=$(find "$L/logs/tabby-$lane" -maxdepth 1 -name '*.log' -newer "$cfg" -print0 2>/dev/null \
      | xargs -0 -r grep -ahoE "$re" 2>/dev/null | tail -n 1)
  fi
  echo "${got##* }"
}
active_runs() {  # every lab unit that may hold the GPU, srv lanes or not, except the unit this script runs in (a
  # quality runner's turing-run-q-*, which never touches the GPU). The unit comes from this process's own cgroup,
  # never from an environment variable a caller could set to exempt another unit. From an ssh shell it is empty.
  local self
  self=$(grep -o 'turing-run-[A-Za-z0-9_.-]*\.service' /proc/self/cgroup 2>/dev/null | head -1)
  systemctl --user list-units --plain --no-legend --state=active,activating,deactivating 'turing-run-*' 2>/dev/null \
    | awk -v self="$self" '$1 != self {print $1}' | tr '\n' ' '
}
port_busy() { [ -n "$(ss -Hltn "sport = :$1" 2>/dev/null)" ]; }
vram_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

tabby_config() {  # LANE PORT CTX -> the lane's config with this start's port, model dir and context, on stdout
  local base src add=""
  base=$(tabby_base "$1"); src=$L/tools/tabby/$base.yml
  [ -f "$src" ] || die "missing $src (copy tools/turing/tabby/*.yml from the repo to $L/tools/tabby/)"
  # The -fp32 lanes add the key here. The base ymls leave it out, so that the base lanes stay at the upstream default
  # and TABBY_MODEL_RECURRENT_STATE_FP32 in TURING_LANE_ENV still reaches them (the --config file beats the environment).
  if grep -q '^  recurrent_state_fp32:' "$src"; then die "$src sets recurrent_state_fp32; the -fp32 lanes add it"; fi
  [ "$base" != "$1" ] && add="  recurrent_state_fp32: true                 # lane $1 (lanes.sh)"
  sed -e "s|^  port: .*|  port: $2|" -e "s|^  model_dir: .*|  model_dir: $L/models|" \
      -e "s|^  max_seq_len: .*|  max_seq_len: $3|" -e "s|^  cache_size: .*|  cache_size: $3|" "$src" \
    | awk -v add="$add" '{ print } add != "" && $0 == "  backend: exllamav3" { print add }'
}

build_cmd() {  # LANE PORT CTX CONFIG_PATH -> global arrays CMD (unit command line) and NEED (files that must exist)
  local lane=$1 port=$2 ctx=$3 cfg=$4 bin args
  # shellcheck disable=SC2206  # TURING_LANE_ENV / LCPP flags are word lists on purpose
  local extra_env=(${TURING_LANE_ENV:-})
  if [ "$(lane_kind "$lane")" = tabby ]; then
    # env -u LD_LIBRARY_PATH: torch must use its bundled CUDA libraries (laptop E50: a stray system path broke them).
    CMD=(/usr/bin/env -C "$L/src/tabbyAPI" -u LD_LIBRARY_PATH PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
         "XDG_CACHE_HOME=$L/cache" "TRITON_HOME=$L/cache/triton" "TRITON_CACHE_DIR=$L/cache/triton/cache"
         "HF_HOME=$L/cache/huggingface" "PIP_CACHE_DIR=$L/cache/pip" "TURING_TABBY_LOGDIR=$L/logs/tabby-$lane"
         "${extra_env[@]}" "$L/venv/bin/python" -c "$TABBY_BOOT" --config "$cfg")
    NEED=("$L/venv/bin/python" "$L/src/tabbyAPI/main.py" "$L/tools/tabby/$(tabby_base "$lane").yml"
          "$L/models/qwen38-exl3-2.0/config.json")
    return
  fi
  case $lane in
    lcpp-iq2xxs) bin=$MAIN; args="-m $G/Qwen3.8-27B-UD-IQ2_XXS.gguf" ;;
    lcpp-iq2s) bin=$MAIN; args="-m $G/Qwen3.8-27B-UD-IQ2_S.gguf" ;;
    # Undrafted on the drafted lanes' build, as the laptop's e4 undrafted lanes (iq2xxs_none, iq2s_none) ran v0.4.0.
    # The lab patch only touches models without token_embd/output, so these take plain v0.4.0's code path.
    lcpp-iq2xxs-v040) bin=$FORK; args="-m $G/Qwen3.8-27B-UD-IQ2_XXS.gguf" ;;
    lcpp-iq2s-v040) bin=$FORK; args="-m $G/Qwen3.8-27B-UD-IQ2_S.gguf" ;;
    # EFF-iq2s_mtps_36k (b2.sh) / E48: shared sidecar, n-max 3, draft on the GPU with q8_0 draft KV. Fork only:
    # mainline cannot load the stripped sidecar.
    lcpp-iq2s-mtp) bin=$FORK
      args="-m $G/Qwen3.8-27B-UD-IQ2_S.gguf -md $G/mtp-Qwen3.8-27B-Q4_0-shared.gguf --spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 -ctkd q8_0 -ctvd q8_0" ;;
    # EFF-iq2xxs_dflash2_n7 (e8_choice.txt): DFlash2 Q4_K_M, n-max 7, draft on the GPU, default (f16) draft KV.
    # The laptop ran it on plain v0.4.0; the fork's patch only changes models that lack token_embd/output (sidecars).
    # -main: the same flags on mainline d63fbfe7, which has draft-dflash with the same options.
    lcpp-iq2xxs-dflash2|lcpp-iq2xxs-dflash2-main) bin=$FORK
      [ "$lane" = lcpp-iq2xxs-dflash2-main ] && bin=$MAIN
      args="-m $G/Qwen3.8-27B-UD-IQ2_XXS.gguf -md $G/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 -ngld 99" ;;
    *) die "no llama.cpp command for lane $lane" ;;
  esac
  # shellcheck disable=SC2206
  CMD=(/usr/bin/env "LD_LIBRARY_PATH=$CUDA_LIB" "${extra_env[@]}" "$bin/llama-server" $args $LCPP_COMMON
       -c "$ctx" --port "$port" ${TURING_LCPP_EXTRA:-})
  NEED=("$bin/llama-server")
  for a in $args; do case $a in *.gguf) NEED+=("$a") ;; esac; done
}

record() {  # STATUS STARTUP_S PROBE_JSON -> one row in $L/runs/lanes.jsonl (uses lane/port/ctx/cfg/url of start)
  local gpu apps server=null rs
  gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')
  rs=$(rs_logged)
  if [ "$1" = ready ]; then
    if [ "$(lane_kind "$lane")" = tabby ]; then
      server=$(curl -s -m 10 "$url/v1/model" | jq -c '.parameters // .' 2>/dev/null) || server=null
    else
      server=$(curl -s -m 10 "$url/props" | jq -c '{build_info, model_path, n_ctx: .default_generation_settings.n_ctx,
        min_p: .default_generation_settings.params.min_p}' 2>/dev/null) || server=null
    fi
    [ -n "$server" ] || server=null
  fi
  jq -cn --arg time "$(date -Is)" --arg host "$(hostname)" --arg lane "$lane" --argjson port "$port" \
    --argjson ctx "$ctx" --arg status "$1" --argjson startup_s "$2" --argjson probe "${3:-null}" \
    --argjson server "$server" --arg config "$cfg" --arg command "${CMD[*]}" --arg lcpp_extra "${TURING_LCPP_EXTRA:-}" \
    --arg lane_env "${TURING_LANE_ENV:-}" --arg gpu "$gpu" --arg apps "$apps" \
    --arg rs "$rs" --arg rs_want "$(lane_rs "$lane")" \
    '{time: $time, host: $host, lane: $lane, unit: ("turing-run-srv-" + $lane), port: $port, ctx: $ctx,
      status: $status, startup_s: $startup_s, probe: $probe, server: $server, config: $config, command: $command,
      lcpp_extra: $lcpp_extra, lane_env: $lane_env,
      recurrent_state: (if $rs == "" then null else $rs end),
      recurrent_state_expected: (if $rs_want == "" then null else $rs_want end),
      gpu_after: ($gpu | split(", ") | {mem_used_mib: .[0], mem_total_mib: .[1], temp_c: .[2], power_w: .[3], sm_mhz: .[4]}),
      gpu_apps: $apps}' >> "$L/runs/lanes.jsonl"
}

fail_start() {  # MESSAGE STATUS STARTUP_S
  note "$1"
  tail -n 30 "$L/logs/run-srv-$lane.log" 2>/dev/null | sed 's/^/  | /' >&2
  systemctl --user stop "$unit" 2>/dev/null
  record "$2" "$3" null
  exit 1
}

on_interrupt() {  # INT/TERM/HUP/PIPE during start(): never leave a server nobody was told about holding ~10 GB of VRAM.
  # PIPE: an ssh session without a terminal sends no HUP when it drops; the next write to its stdout/stderr does.
  trap '' INT TERM HUP PIPE  # finish the clean-up even if the terminal or the ssh session is already gone
  systemctl --user stop "$unit" 2>/dev/null
  fail_start "interrupted; stopped $unit" interrupted "$(( $(date +%s) - t0 ))"
}

start() {
  lane=$1 port=${2:-$DEFAULT_PORT}
  is_lane "$lane" || die "unknown lane '$lane' (see: lanes.sh list)"
  [[ $port =~ ^[0-9]+$ ]] || die "bad port '$port'"
  ctx=$(lane_ctx "$lane"); [[ $ctx =~ ^[0-9]+$ ]] || die "bad TURING_LANE_CTX '$ctx'"
  unit=turing-run-srv-$lane url=http://localhost:$port cfg="" log_off=0
  systemctl --user is-active --quiet turing-guard || die "the yield guard is not running (bash $L/tools/desktop_run.sh guard)"
  local busy; busy=$(active_runs)
  [ -z "$busy" ] || die "another lab GPU unit is active (${busy% }); one lane or GPU job at a time"
  port_busy "$port" && die "port $port is already in use"
  mkdir -p "$L/logs/lanes" "$L/runs"
  if [ "$(lane_kind "$lane")" = tabby ]; then
    (( ctx % 256 == 0 )) || die "TabbyAPI needs a context that is a multiple of 256 (got $ctx)"
    # upstream merges ./config.yml under the --config file, so a stray one would leak settings into the lane
    [ -e "$L/src/tabbyAPI/config.yml" ] && die "$L/src/tabbyAPI/config.yml exists and would be merged into the lane config"
    cfg=$L/logs/lanes/$lane-$(date +%Y%m%dT%H%M%S)-p$port.yml
    tabby_config "$lane" "$port" "$ctx" > "$cfg" || die "could not write $cfg"
    grep -q "^  port: $port\$" "$cfg" && grep -q "^  cache_size: $ctx\$" "$cfg" || die "port/context not set in $cfg"
    case $lane in
      *-fp32) grep -q '^  recurrent_state_fp32: true ' "$cfg" || die "recurrent_state_fp32 not set in $cfg" ;;
    esac
    mkdir -p "$L/logs/tabby-$lane"
  fi
  build_cmd "$lane" "$port" "$ctx" "$cfg"
  local f; for f in "${NEED[@]}"; do [ -e "$f" ] || die "missing $f"; done
  echo "$port" > "$L/logs/lanes/$lane.port"
  echo "== $(date -Is) lanes.sh start $lane port=$port ctx=$ctx extra='${TURING_LCPP_EXTRA:-}' env='${TURING_LANE_ENV:-}'" \
    >> "$L/logs/run-srv-$lane.log"
  log_off=$(stat -c %s "$L/logs/run-srv-$lane.log" 2>/dev/null || echo 0)  # rs_logged reads only this start's part
  note "starting $lane on port $port (context $ctx) as $unit; console log $L/logs/run-srv-$lane.log"
  local t0 waited=0 code want got
  t0=$(date +%s)
  # From here until the URL has been printed. Signals reach the trap after the running command returns: the health
  # loop's curl and sleep take at most 3 s each, and the probe runs in the background under wait, which returns at once.
  trap on_interrupt INT TERM HUP PIPE
  bash "$L/tools/desktop_run.sh" cmd "srv-$lane" "${CMD[@]}" >&2 || die "desktop_run.sh could not start $unit"
  while :; do
    waited=$(( $(date +%s) - t0 ))
    systemctl --user is-active --quiet "$unit" || fail_start "$unit exited during start-up (guard stop, OOM or bad flags); last log lines:" exited "$waited"
    code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "$url/health")
    [ "$code" = 200 ] && break
    [ "$waited" -ge "${TURING_LANE_TIMEOUT:-600}" ] && fail_start "no 200 from $url/health after ${waited}s; stopping $unit" timeout "$waited"
    sleep 3
  done
  curl -sf -m 10 "$url/v1/models" > /dev/null || fail_start "/health answered but /v1/models did not; stopping $unit" no-models "$waited"
  want=$(lane_rs "$lane")
  if [ -n "$want" ]; then
    got=$(rs_logged)
    if [ -z "$got" ]; then
      note "warning: no 'Recurrent state storage' line in this start's log; precision not confirmed (expected $want)"
    elif [ "$got" != "$want" ]; then
      fail_start "TabbyAPI logged recurrent state $got, but lane $lane expects $want; stopping $unit" precision-mismatch "$waited"
    fi
  fi
  local probe=null raw pf=$L/logs/lanes/$lane.probe.json
  if [ "${TURING_LANE_PROBE:-1}" != 0 ]; then
    curl -sf -m 300 -H 'Content-Type: application/json' -d "$PROBE_BODY" -o "$pf" "$url/v1/chat/completions" &
    wait $! || fail_start "the 16-token generation probe failed; stopping $unit" probe-failed "$waited"
    raw=$(cat "$pf" 2>/dev/null)
    probe=$(printf '%s' "$raw" | jq -c '{completion_tokens: .usage.completion_tokens, text: ((.choices[0].message.content // "")[0:40]),
                                         predicted_per_second: .timings.predicted_per_second}' 2>/dev/null) || probe=null
    [ -n "$probe" ] || probe=null
  fi
  waited=$(( $(date +%s) - t0 ))
  record ready "$waited" "$probe"
  note "$lane ready after ${waited}s; GPU memory used: $(vram_used) MiB; recurrent state: ${got:-n/a}; probe: $probe"
  echo "$url"  # still under the trap: a caller that is gone by now never gets the URL, so the lane is stopped
  trap - INT TERM HUP PIPE
}

stop() {
  is_lane "${1:-}" || die "unknown lane '${1:-}' (see: lanes.sh list)"
  local unit=turing-run-srv-$1
  case $(systemctl --user show -p ActiveState --value "$unit" 2>/dev/null) in
    active|activating|deactivating|reloading) ;;
    *) note "$unit is not running"; return 0 ;;
  esac
  systemctl --user stop "$unit" || die "systemctl --user stop $unit failed"
  echo "== $(date -Is) lanes.sh stop $1" >> "$L/logs/run-srv-$1.log"
  note "stopped $unit; GPU memory used now: $(vram_used) MiB"
}

status() {
  local unit load active sub rest lane port code
  systemctl --user list-units --plain --no-legend --state=active,activating,deactivating 'turing-run-srv-*' 2>/dev/null |
    while read -r unit load active sub rest; do
      lane=${unit#turing-run-srv-}; lane=${lane%.service}
      port=$(cat "$L/logs/lanes/$lane.port" 2>/dev/null || echo "$DEFAULT_PORT")
      code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://localhost:$port/health")
      printf '%s\t%s/%s\thttp://localhost:%s\thealth=%s\n' "$lane" "$active" "$sub" "$port" "$code"
    done
}

show() {
  lane=$1 port=${2:-$DEFAULT_PORT}
  is_lane "$lane" || die "unknown lane '$lane' (see: lanes.sh list)"
  ctx=$(lane_ctx "$lane")
  cfg=""
  if [ "$(lane_kind "$lane")" = tabby ]; then
    cfg="$L/logs/lanes/$lane-<start time>-p$port.yml"
    echo "# config ($cfg):"
    if [ -f "$L/tools/tabby/$(tabby_base "$lane").yml" ]; then
      tabby_config "$lane" "$port" "$ctx" | grep -v '^ *#'
    else
      echo "  MISSING $L/tools/tabby/$(tabby_base "$lane").yml"
    fi
    echo "# expected 'Recurrent state storage' at load: $(lane_rs "$lane" | grep . || echo 'not checked (TURING_LANE_ENV sets it)')"
  fi
  build_cmd "$lane" "$port" "$ctx" "$cfg"
  echo "# unit turing-run-srv-$lane, started by: bash $L/tools/desktop_run.sh cmd srv-$lane \\"
  printf '  %q \\\n' "${CMD[@]}"
  echo "# files:"
  local f; for f in "${NEED[@]}"; do [ -e "$f" ] && echo "  ok      $f" || echo "  MISSING $f"; done
}

case ${1:-} in
  list) printf '%s\n' $LANES ;;
  start) [ $# -ge 2 ] || die "usage: lanes.sh start LANE [PORT]"; start "$2" "${3:-}" ;;
  stop) [ $# -ge 2 ] || die "usage: lanes.sh stop LANE"; stop "$2" ;;
  status) status ;;
  show) [ $# -ge 2 ] || die "usage: lanes.sh show LANE [PORT]"; show "$2" "${3:-}" ;;
  *) sed -n '2,/^set -uo pipefail$/p' "$0" | sed '$d' >&2; exit 64 ;;
esac
