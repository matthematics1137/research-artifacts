#!/bin/bash
# E1 equal-tuning sweep (research/TURING_EXL3_RESEARCH_2026-10-30.md 6.5): before any EXL3-vs-llama.cpp comparison
# counts, give llama.cpp's undrafted bar the tuning effort EXL3 gets, and check offload and clocks. llama-bench on the
# mainline sm_75 build for UD-IQ2_XXS and UD-IQ2_S. Launch through the capped, guarded launcher (refuses otherwise):
#   bash $L/tools/desktop_run.sh cmd sweep bash $L/tools/llamacpp_sweep.sh [--quick]
#   bash $L/tools/llamacpp_sweep.sh --dry-run [--quick]     print the plan only; no GPU, no files written
# Output (nothing is ever deleted):
#   $L/runs/llamabench_sweep.jsonl   llama-bench's own JSON rows, each tagged with sweep_id, sweep_stage, kernel,
#                                    cuda_graphs and bin_dir; plus meta rows (sweep_row = start / offload / end)
#   $L/runs/sweep/<id>/NNN-stage.jsonl   the untagged rows of each llama-bench call
#   $L/logs/sweep-<id>-smi.csv       nvidia-smi at 1 Hz for the whole sweep: clocks, power, temperature,
#                                    utilization, memory, clock-event (throttle) reasons, pstate
#   $L/logs/sweep-<id>-server-*.log  the llama-server start used for the 'offloaded N/N layers' check
#   console (llama-bench progress, CUDA init lines): the unit's log, $L/logs/run-sweep.log
# Design (full mode, about 35-45 min; a 75-min soft budget skips the remaining calls):
#   tg  decode, -p 0 -n 128, depths 0 and 8192: CUDA graphs on/off x {fa 1 + q8_0 KV, fa 1 + f16 KV, fa 0 + f16 KV}.
#       -ub and MMQ/cuBLAS cannot change batch-1 decode: mainline sends ne11 <= 8 to MMVQ before it asks about MMQ
#       (ggml-cuda.cu ggml_cuda_mul_mat), so decode is swept on the axes that can move it.
#   pp  prompt processing, -p 2048 -n 0 -b 2048, -ub 256,512,1024,2048, depths 0 and 8192, same three fa/KV pairs,
#       graphs on (CUDA graphs only capture batch-1 decode).
#   threads  decode at depth 0 with -t 1,4,10 (the unit has 4 cores; E1 ran llama-bench's default, n_threads 10).
#   cublas   MMQ vs cuBLAS. On this build GGML_CUDA_FORCE_MMQ / GGML_CUDA_FORCE_CUBLAS are CMake options, not
#       runtime environment variables (ggml/src/ggml-cuda/CMakeLists.txt:139-144), and on sm_75 the default build
#       already picks MMQ for every IQ2 type (mmq.cu:435 turing_mma_available). So 'mmq' = the default build, and
#       'cublas' runs only if a -DGGML_CUDA_FORCE_CUBLAS=ON build exists at $CUBLAS_BIN (see lanes_README.md);
#       otherwise the sweep records that it was skipped.
#   offload  one llama-server start per model (-ngl 99 -fa on q8_0 KV, 16K) to capture 'offloaded N/N layers',
#       the system_info line and buffer sizes, then a clean stop.
# Every call uses -ngl 99 -r 3 (quick: -r 1) and -t 4 unless it sweeps threads.
# Quick mode (~3 min): IQ2_XXS only, -r 1, decode -n 32 at depth 0 with graphs on/off x {fa1 q8_0, fa0 f16},
# pp -p 512 -ub 256,512 at depth 0, and the offload check.
set -uo pipefail
L=${TURING_LAB:-/opt/p8}  # the desktop's ssd2 drive, never the main drive
BIN=${TURING_SWEEP_BIN:-$L/src/llama.cpp/build-sm75-cu134/bin}                    # mainline d63fbfe7
CUBLAS_BIN=${TURING_SWEEP_CUBLAS_BIN:-$L/src/llama.cpp/build-sm75-cu134-cublas/bin}
G=$L/models/gguf
OUT=$L/runs/llamabench_sweep.jsonl
BUDGET_MIN=${TURING_SWEEP_BUDGET_MIN:-75}
PORT=${TURING_SWEEP_PORT:-18990}
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64
mode=full dry=0
for a in "$@"; do
  case $a in
    --quick) mode=quick ;;
    --dry-run) dry=1 ;;
    *) echo "usage: llamacpp_sweep.sh [--quick] [--dry-run]" >&2; exit 64 ;;
  esac
done
ID=$(date +%Y%m%dT%H%M%S)-$mode
PARTS=$L/runs/sweep/$ID
SMI_LOG=$L/logs/sweep-$ID-smi.csv
T0=$(date +%s) STEP=0 FAILS=0 SKIPS=0
FAKV=("1:q8_0" "1:f16" "0:f16")    # flash attention : KV type (quantized V needs flash attention, so no 0:q8_0)
if [ $mode = quick ]; then
  REPS=1 MODELS=(Qwen3.8-27B-UD-IQ2_XXS.gguf) GRAPHS=(on off) TG_FAKV=("1:q8_0" "0:f16") TG="-p 0 -n 32 -d 0"
  PP_FAKV=("1:q8_0") PP="-p 512 -n 0 -b 2048 -ub 256,512 -d 0" THREADS=""
else
  REPS=3 MODELS=(Qwen3.8-27B-UD-IQ2_XXS.gguf Qwen3.8-27B-UD-IQ2_S.gguf) GRAPHS=(on off) TG_FAKV=("${FAKV[@]}")
  TG="-p 0 -n 128 -d 0,8192" PP_FAKV=("${FAKV[@]}") PP="-p 2048 -n 0 -b 2048 -ub 256,512,1024,2048 -d 0,8192"
  THREADS="-p 0 -n 128 -d 0 -t 1,4,10"
fi

say() { echo "$(date -Is) sweep $ID: $*"; }
elapsed_min() { echo $(( ($(date +%s) - T0) / 60 )); }
over_budget() { [ $(( $(date +%s) - T0 )) -ge $(( BUDGET_MIN * 60 )) ]; }
smi_line() { nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.active,pstate --format=csv,noheader 2>/dev/null | head -1; }

meta() {  # SWEEP_ROW JSON_OBJECT -> one meta row in $OUT
  [ $dry = 1 ] && { say "meta $1: $2"; return 0; }
  jq -cn --arg id "$ID" --arg mode "$mode" --arg row "$1" --arg host "$(hostname)" --arg time "$(date -Is)" \
    --argjson extra "$2" '{sweep_id: $id, sweep_mode: $mode, sweep_row: $row, host: $host, time: $time} + $extra' >> "$OUT"
}

bench() {  # STAGE MODEL KERNEL GRAPHS BIN_DIR FA KV -- extra llama-bench args (word-split string)
  local stage=$1 model=$2 kernel=$3 graphs=$4 bin=$5 fa=$6 kv=$7 extra=$8 part rc envs=()
  STEP=$((STEP + 1))
  part=$PARTS/$(printf '%03d' $STEP)-$stage-${model%.gguf}.jsonl
  [ "$graphs" = off ] && envs=(GGML_CUDA_DISABLE_GRAPHS=1)
  # shellcheck disable=SC2206
  local args=(-m "$G/$model" -ngl 99 -r "$REPS" -fa "$fa" -ctk "$kv" -ctv "$kv" $extra -o jsonl)
  case " $extra " in *" -t "*) ;; *) args+=(-t 4) ;; esac
  if over_budget; then
    SKIPS=$((SKIPS + 1)); say "SKIP $STEP (over the ${BUDGET_MIN}-min budget) $stage $model kernel=$kernel graphs=$graphs fa=$fa kv=$kv"
    return 0
  fi
  say "RUN $STEP $stage $model kernel=$kernel graphs=$graphs: ${envs[*]} llama-bench ${args[*]}"
  [ $dry = 1 ] && return 0
  say "gpu before: $(smi_line)"
  env "${envs[@]}" "$bin/llama-bench" "${args[@]}" > "$part"
  rc=$?
  [ $rc -eq 0 ] || { FAILS=$((FAILS + 1)); say "FAIL $STEP rc=$rc (rows so far kept in $part)"; }
  jq -c --arg id "$ID" --arg mode "$mode" --arg stage "$stage" --arg kernel "$kernel" --arg graphs "$graphs" \
    --arg bin "$bin" --argjson step "$STEP" --arg smi "$SMI_LOG" --arg rc "$rc" \
    '. + {sweep_id: $id, sweep_mode: $mode, sweep_row: "bench", sweep_stage: $stage, sweep_step: $step,
          kernel: $kernel, cuda_graphs: $graphs, bin_dir: $bin, smi_log: $smi, bench_rc: ($rc | tonumber)}' \
    "$part" >> "$OUT" 2>/dev/null || say "WARN $STEP: rows in $part are not all JSON"
  say "done $STEP: $(wc -l < "$part") rows; gpu after: $(smi_line)"
}

offload_check() {  # MODEL -> meta row with the offload line, system_info and buffer sizes
  local model=$1 log=$L/logs/sweep-$ID-server-${1%.gguf}.log pid code t=0 vram offl
  STEP=$((STEP + 1))
  if over_budget; then SKIPS=$((SKIPS + 1)); say "SKIP $STEP offload check $model (budget)"; return 0; fi
  say "RUN $STEP offload check: llama-server -m $model -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1 -t 4 -lv 4 on port $PORT"
  [ $dry = 1 ] && return 0
  if [ -n "$(ss -Hltn "sport = :$PORT" 2>/dev/null)" ]; then
    SKIPS=$((SKIPS + 1)); say "SKIP offload check: port $PORT is in use"; return 0
  fi
  # -lv 4: common/log.cpp maps the library's INFO messages (load_tensors: offloaded N/N layers, buffer sizes) to
  # level 4 (trace), above llama-server's default threshold of 3, so without it the line never reaches the log
  # (2026-10-30 quick sweep: 'NOT FOUND' with a 10-line log)
  "$BIN/llama-server" -m "$G/$model" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 -np 1 -t 4 -lv 4 \
    --host localhost --port "$PORT" > "$log" 2>&1 &
  pid=$!
  while [ $t -lt 300 ]; do
    code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://localhost:$PORT/health")
    [ "$code" = 200 ] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 2; t=$((t + 2))
  done
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  kill -INT "$pid" 2>/dev/null
  for t in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    say "llama-server did not exit on SIGINT; sending SIGTERM"; kill -TERM "$pid"
    for t in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { say "still running; sending SIGKILL"; kill -KILL "$pid"; }
  fi
  wait "$pid" 2>/dev/null
  offl=$(grep -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' "$log" | head -1)
  say "offload $model: '${offl:-NOT FOUND}', health=$code, VRAM used at ready $vram MiB"
  meta offload "$(jq -cn --arg model "$model" --arg offloaded "$offl" --arg health "$code" --arg vram "$vram" \
    --arg log "$log" --arg lines "$(grep -E 'offloaded|system_info|buffer size|CUDA0|ggml_cuda_init|FORCE_|USE_GRAPHS|build' "$log" | head -40)" \
    '{model: $model, offloaded: $offloaded, health_code: $health, vram_used_mib: $vram, server_log: $log,
      log_lines: ($lines | split("\n"))}')"
}

# ---------------------------------------------------------------- preflight
for f in "$BIN/llama-bench" "$BIN/llama-server"; do [ -x "$f" ] || { echo "missing $f" >&2; exit 1; }; done
for m in "${MODELS[@]}"; do [ -f "$G/$m" ] || { echo "missing $G/$m" >&2; exit 1; }; done
have_cublas=0; [ -x "$CUBLAS_BIN/llama-bench" ] && have_cublas=1
if [ $dry = 0 ]; then
  own=$(grep -oE 'turing-run-[A-Za-z0-9_.-]+\.service' /proc/$$/cgroup | head -1)
  [ -n "$own" ] || { echo "run this through: bash $L/tools/desktop_run.sh cmd sweep bash $0 [--quick] (caps + guard)" >&2; exit 1; }
  others=$(systemctl --user list-units --plain --no-legend --state=active,activating,deactivating 'turing-run-*' 2>/dev/null \
    | awk -v own="$own" '$1 != own {print $1}' | tr '\n' ' ')
  [ -z "$others" ] || { echo "another lab GPU unit is active ($others); one GPU job at a time" >&2; exit 1; }
  mkdir -p "$PARTS" "$L/logs"
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active,pstate \
    --format=csv -l 1 > "$SMI_LOG" 2>&1 &
  SMI_PID=$!
  trap 'kill "$SMI_PID" 2>/dev/null' EXIT
fi
say "start: mode=$mode reps=$REPS models=${MODELS[*]} bin=$BIN cublas_build=$have_cublas budget=${BUDGET_MIN}min out=$OUT"
meta start "$(jq -cn --arg bin "$BIN" --arg cublas "$CUBLAS_BIN" --argjson have_cublas $have_cublas \
  --arg commit "$(git -C "$BIN/../.." rev-parse HEAD 2>/dev/null)" --arg smi "$SMI_LOG" --arg parts "$PARTS" \
  --arg gpu "$(nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,clocks.max.mem,memory.total --format=csv,noheader 2>/dev/null | head -1)" \
  --argjson reps "$REPS" --argjson budget "$BUDGET_MIN" \
  '{bin_dir: $bin, build_commit: $commit, cublas_bin: $cublas, cublas_build_present: ($have_cublas == 1),
    reps: $reps, budget_min: $budget, smi_log: $smi, parts_dir: $parts, gpu: $gpu}')"

# ---------------------------------------------------------------- the sweep, most important first
for model in "${MODELS[@]}"; do
  offload_check "$model"
  for graphs in "${GRAPHS[@]}"; do
    for p in "${TG_FAKV[@]}"; do bench tg "$model" mmq "$graphs" "$BIN" "${p%%:*}" "${p#*:}" "$TG"; done
  done
  for p in "${PP_FAKV[@]}"; do bench pp "$model" mmq on "$BIN" "${p%%:*}" "${p#*:}" "$PP"; done
  [ -n "$THREADS" ] && [ "$model" = Qwen3.8-27B-UD-IQ2_XXS.gguf ] && bench threads "$model" mmq on "$BIN" 1 q8_0 "$THREADS"
  if [ $have_cublas = 1 ]; then
    bench cublas-pp "$model" cublas on "$CUBLAS_BIN" 1 q8_0 "$PP"
    bench cublas-tg "$model" cublas on "$CUBLAS_BIN" 1 q8_0 "-p 0 -n 128 -d 0"
  else
    SKIPS=$((SKIPS + 1)); say "SKIP cuBLAS variant for $model: no forced-cuBLAS build at $CUBLAS_BIN"
  fi
done

say "end: $STEP calls, $FAILS failed, $SKIPS skipped, $(elapsed_min) min"
meta end "$(jq -cn --argjson steps $STEP --argjson fails $FAILS --argjson skips $SKIPS --argjson minutes "$(elapsed_min)" \
  '{calls: $steps, failed: $fails, skipped: $skips, minutes: $minutes}')"
if [ $dry = 0 ]; then
  say "summary (avg tok/s; tg = decode, pp = prompt processing):"
  jq -r --arg id "$ID" 'select(.sweep_id == $id and .sweep_row == "bench")
    | [.sweep_stage, (.model_filename | split("/") | last), .kernel, ("graphs=" + .cuda_graphs),
       ("fa=" + (.flash_attn | tostring)), .type_k, ("ub=" + (.n_ubatch | tostring)), ("t=" + (.n_threads | tostring)),
       ("d=" + (.n_depth | tostring)), (if .n_gen > 0 then "tg" + (.n_gen | tostring) else "pp" + (.n_prompt | tostring) end),
       ((.avg_ts * 100 | round) / 100 | tostring), ("+-" + ((.stddev_ts * 100 | round) / 100 | tostring))] | @tsv' \
    "$OUT" 2>/dev/null > "$PARTS/summary.tsv"
  if command -v column > /dev/null; then column -t "$PARTS/summary.tsv"; else cat "$PARTS/summary.tsv"; fi
fi
exit $(( FAILS > 0 ))
