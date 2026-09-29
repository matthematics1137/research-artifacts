#!/usr/bin/env bash
# Fail-closed launcher for the corrective Paper 2 R1 campaign.
# `self-test`, `prepare-*`, `fingerprint-artifacts`, and `preflight` do not run
# model inference.  Only the explicit `run` mode starts servers or sends POSTs.
set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
R1_ROOT="$SCRIPT_DIR/r1"
PRIVATE_INPUTS="$R1_ROOT/private_inputs"
RECEIPT_ROOT="$R1_ROOT/receipts"
ARTIFACT_RECORDS="$RECEIPT_ROOT/artifacts"
RESULTS_ROOT="$REPO_ROOT/testsuite/evals/results"
IDENTITY="$SCRIPT_DIR/identity_guard.py"
CAMPAIGN_GUARD="$SCRIPT_DIR/campaign_guard.py"
RUNTIME_PROVENANCE="$SCRIPT_DIR/runtime_provenance.py"
RUN_GRID="$SCRIPT_DIR/run_grid.py"
GEN_QUESTIONS="$SCRIPT_DIR/gen_questions.py"
WARMUP="$SCRIPT_DIR/warmup_r1.py"
MODEL_SOURCE_MANIFEST="$REPO_ROOT/publication/papers/qwen38_27b_12gb/public/environment/model_artifacts.json"
LINGUA_PYTHON="$REPO_ROOT/engines/lingua-env/bin/python"
PORT=18090
FALLBACK_PORT=18091
BASE_URL="http://127.0.0.1:$PORT"
LLAMA_REVISION=035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
TABBY_REVISION=4a4f9f44820303593844f092d424bb7506008733

usage() {
  echo "usage: $0 {self-test|prepare-inputs|prepare-runtime|fingerprint-artifacts|preflight|run}"
  echo "run: set P2R1_MODEL_ROOT and P2R1_ATTEMPT_ID=p2r1-primary-v5"
}

die() {
  echo "P2R1 ABORTED: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file missing: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory missing: $1"
}

require_model_root() {
  [[ -n "${P2R1_MODEL_ROOT:-}" ]] || die "set P2R1_MODEL_ROOT to the model parent directory"
  MODEL_ROOT=$(cd -- "$P2R1_MODEL_ROOT" && pwd -P)
  Q4_ARTIFACT="$MODEL_ROOT/qwen38/Qwen3.8-27B-UD-Q4_K_XL.gguf"
  IQ2_ARTIFACT="$MODEL_ROOT/qwen38/Qwen3.8-27B-UD-IQ2_S.gguf"
  EXL3_ARTIFACT="$MODEL_ROOT/qwen38-exl3-2.0"
  require_file "$Q4_ARTIFACT"
  require_file "$IQ2_ARTIFACT"
  require_dir "$EXL3_ARTIFACT"
}

require_protocol_frozen() {
  git -C "$REPO_ROOT" ls-files --error-unmatch paper2/R1_PROTOCOL.json >/dev/null 2>&1 \
    || die "paper2/R1_PROTOCOL.json must be committed before the rerun"
  git -C "$REPO_ROOT" diff --quiet -- paper2/R1_PROTOCOL.json \
    || die "paper2/R1_PROTOCOL.json has uncommitted changes"
}

run_self_test() {
  python3 -m py_compile "$SCRIPT_DIR"/*.py
  python3 "$SCRIPT_DIR/selftest_r1.py"
  python3 "$SCRIPT_DIR/warmup_r1.py" --self-test
  python3 "$SCRIPT_DIR/runtime_provenance.py"
  python3 "$REPO_ROOT/paper2/scripts/validate_r1.py" --self-test
  bash -n "$SCRIPT_DIR/p2_grid_r1.sh"
  python3 -m json.tool "$REPO_ROOT/paper2/R1_PROTOCOL.json" >/dev/null
  echo "P2R1 static self-test passed; no inference was run."
}

prepare_pair() {
  local first=$1
  local second=$2
  shift 2
  if [[ -e "$first" || -e "$second" ]]; then
    [[ -e "$first" && -e "$second" ]] \
      || die "partial frozen-input pair exists: $first / $second"
    echo "verified existing frozen pair: $first / $second"
    return
  fi
  "$@"
}

prepare_inputs() {
  require_file "$LINGUA_PYTHON"
  mkdir -p "$PRIVATE_INPUTS/tsi" "$PRIVATE_INPUTS/wildchat"
  chmod 700 "$R1_ROOT" "$PRIVATE_INPUTS" "$PRIVATE_INPUTS/tsi" "$PRIVATE_INPUTS/wildchat"
  prepare_pair \
    "$PRIVATE_INPUTS/tsi/windows_base.jsonl" \
    "$PRIVATE_INPUTS/tsi/source_manifest.json" \
    "$LINGUA_PYTHON" "$SCRIPT_DIR/freeze_tsi_r1.py"
  prepare_pair \
    "$PRIVATE_INPUTS/wildchat/windows_base.jsonl" \
    "$PRIVATE_INPUTS/wildchat/source_manifest.json" \
    "$LINGUA_PYTHON" "$SCRIPT_DIR/build_wildchat_r1.py" \
      --source-dir "$PRIVATE_INPUTS/wildchat/upstream"
  prepare_pair \
    "$PRIVATE_INPUTS/tsi/windows.jsonl" \
    "$PRIVATE_INPUTS/tsi/input_manifest.json" \
    "$LINGUA_PYTHON" "$SCRIPT_DIR/make_conditions_r1.py" \
      --corpus TSI \
      --input "$PRIVATE_INPUTS/tsi/windows_base.jsonl" \
      --source-manifest "$PRIVATE_INPUTS/tsi/source_manifest.json" \
      --out "$PRIVATE_INPUTS/tsi/windows.jsonl" \
      --manifest "$PRIVATE_INPUTS/tsi/input_manifest.json"
  prepare_pair \
    "$PRIVATE_INPUTS/wildchat/windows.jsonl" \
    "$PRIVATE_INPUTS/wildchat/input_manifest.json" \
    "$LINGUA_PYTHON" "$SCRIPT_DIR/make_conditions_r1.py" \
      --corpus WC \
      --input "$PRIVATE_INPUTS/wildchat/windows_base.jsonl" \
      --source-manifest "$PRIVATE_INPUTS/wildchat/source_manifest.json" \
      --out "$PRIVATE_INPUTS/wildchat/windows.jsonl" \
      --manifest "$PRIVATE_INPUTS/wildchat/input_manifest.json"
  "$LINGUA_PYTHON" "$SCRIPT_DIR/validate_inputs_r1.py" --inputs
  echo "P2R1 private inputs are frozen. No benchmark inference was run."
}

prepare_runtime() {
  require_model_root
  mkdir -p "$PRIVATE_INPUTS/runtime"
  chmod 700 "$PRIVATE_INPUTS/runtime"
  prepare_pair \
    "$PRIVATE_INPUTS/runtime/tabby.yml" \
    "$PRIVATE_INPUTS/runtime/tabby_manifest.json" \
    python3 "$SCRIPT_DIR/prepare_tabby_config_r1.py" \
      --model-dir "$MODEL_ROOT" --port "$PORT" \
      --out "$PRIVATE_INPUTS/runtime/tabby.yml" \
      --manifest "$PRIVATE_INPUTS/runtime/tabby_manifest.json"
  python3 "$SCRIPT_DIR/validate_inputs_r1.py" \
    --tabby-config --model-root "$MODEL_ROOT"
}

fingerprint_artifact() {
  local artifact=$1
  local record=$2
  local artifact_id=$3
  if [[ -e "$record" ]]; then
    python3 "$IDENTITY" verify-artifact --record "$record" --full
  else
    python3 "$IDENTITY" fingerprint-artifact \
      --artifact "$artifact" --output "$record" \
      --source-manifest "$MODEL_SOURCE_MANIFEST" --artifact-id "$artifact_id"
  fi
}

fingerprint_artifacts() {
  require_model_root
  mkdir -p "$ARTIFACT_RECORDS"
  chmod 700 "$R1_ROOT" "$RECEIPT_ROOT" "$ARTIFACT_RECORDS"
  require_file "$MODEL_SOURCE_MANIFEST"
  fingerprint_artifact "$Q4_ARTIFACT" "$ARTIFACT_RECORDS/q4kxl.json" UD-Q4_K_XL
  fingerprint_artifact "$EXL3_ARTIFACT" "$ARTIFACT_RECORDS/exl3.json" EXL3-SC_2.00bpw_H3
  fingerprint_artifact "$IQ2_ARTIFACT" "$ARTIFACT_RECORDS/iq2s.json" UD-IQ2_S
  echo "Full artifact hashes verified. Run the campaign only after the cool-state gate; run mode performs fast state checks only."
}

require_inputs() {
  local corpus
  for corpus in tsi wildchat; do
    require_file "$PRIVATE_INPUTS/$corpus/windows.jsonl"
    require_file "$PRIVATE_INPUTS/$corpus/input_manifest.json"
  done
  python3 "$SCRIPT_DIR/validate_inputs_r1.py" --inputs
}

require_artifact_records_fast() {
  python3 "$IDENTITY" verify-artifact --record "$ARTIFACT_RECORDS/q4kxl.json"
  python3 "$IDENTITY" verify-artifact --record "$ARTIFACT_RECORDS/exl3.json"
  python3 "$IDENTITY" verify-artifact --record "$ARTIFACT_RECORDS/iq2s.json"
}

preflight() {
  require_model_root
  run_self_test
  require_protocol_frozen
  require_inputs
  require_file "$PRIVATE_INPUTS/runtime/tabby.yml"
  require_file "$PRIVATE_INPUTS/runtime/tabby_manifest.json"
  python3 "$SCRIPT_DIR/validate_inputs_r1.py" \
    --tabby-config --model-root "$MODEL_ROOT"
  python3 "$IDENTITY" port-free --port "$PORT" --port "$FALLBACK_PORT"
  fingerprint_artifacts
  local check_dir="$RECEIPT_ROOT/preflight-$(tr -d '-' </proc/sys/kernel/random/uuid)"
  mkdir -m 700 "$check_dir"
  python3 "$CAMPAIGN_GUARD" host --attempt-id p2r1-preflight \
    --output "$check_dir/host.json"
  python3 "$CAMPAIGN_GUARD" bind-host --attempt-id p2r1-preflight \
    --host-receipt "$check_dir/host.json" --output "$check_dir/host-binding.json"
  python3 "$CAMPAIGN_GUARD" toolchain --output "$check_dir/toolchain.json" \
    --host-receipt "$check_dir/host.json" \
    --attempt-host-binding "$check_dir/host-binding.json"
  echo "P2R1 preflight passed; no inference was run."
}

ACTIVE_PROCESS_RECORD=""
ACTIVE_RECEIPT=""
ACTIVE_SHUTDOWN=""
ACTIVE_STAGE=""

record_abort_snapshot() {
  if [[ -z "$ACTIVE_STAGE" || -z "${CAMPAIGN_DIR:-}" || -z "$ACTIVE_RECEIPT" ]]; then
    return
  fi
  local normal="$CAMPAIGN_DIR/host/${ACTIVE_STAGE}-post.json"
  local aborted="$CAMPAIGN_DIR/host/${ACTIVE_STAGE}-abort.json"
  if [[ ! -e "$normal" && ! -e "$aborted" ]]; then
    python3 "$CAMPAIGN_GUARD" snapshot \
      --stage "${ACTIVE_STAGE}-abort" --output "$aborted"
  fi
}

cleanup_active() {
  set +e
  if [[ -n "$ACTIVE_RECEIPT" && -f "$ACTIVE_RECEIPT" && ! -e "$ACTIVE_SHUTDOWN" ]]; then
    # A later campaign may safely resume append-only rows only if this
    # interrupted session retains an explicit terminal-state receipt.
    record_abort_snapshot
    python3 "$IDENTITY" terminate \
      --receipt "$ACTIVE_RECEIPT" --shutdown "$ACTIVE_SHUTDOWN" --wait-seconds 30
  elif [[ -n "$ACTIVE_PROCESS_RECORD" && -f "$ACTIVE_PROCESS_RECORD" ]]; then
    python3 "$IDENTITY" terminate-process \
      --process-record "$ACTIVE_PROCESS_RECORD" --wait-seconds 30
  fi
  ACTIVE_PROCESS_RECORD=""
  ACTIVE_RECEIPT=""
  ACTIVE_SHUTDOWN=""
  ACTIVE_STAGE=""
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  cleanup_active
  exit "$rc"
}

on_signal() {
  local rc=$1
  trap - EXIT INT TERM
  cleanup_active
  exit "$rc"
}

wait_for_executable() {
  local pid=$1
  local expected=$2
  local observed=""
  local attempt
  for attempt in $(seq 1 200); do
    [[ -d "/proc/$pid" ]] || die "launched PID $pid exited before executable capture"
    observed=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    [[ "$observed" == "$expected" ]] && return
    sleep 0.01
  done
  die "launched PID $pid did not exec expected binary $expected (observed $observed)"
}

new_log() {
  local path=$1
  ( set -o noclobber; : > "$path" ) || die "refusing to overwrite server log: $path"
  chmod 600 "$path"
}

launch_llama() {
  local tier=$1
  local artifact=$2
  local artifact_record=$3
  local ngl=$4
  local expected_layers=$5
  local stage=$6
  local alias_prefix
  alias_prefix=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact"]["sha256"][:12])' "$artifact_record")
  local alias="P2R1-$tier-$alias_prefix"
  local log="$CAMPAIGN_DIR/logs/$stage.log"
  local process_record="$CAMPAIGN_DIR/process/$stage.json"
  local receipt="$CAMPAIGN_DIR/identity/$stage.json"
  local shutdown="$CAMPAIGN_DIR/shutdown/$stage.json"
  local placement="$CAMPAIGN_DIR/placement/$stage.json"
  python3 "$IDENTITY" port-free --port "$PORT" --port "$FALLBACK_PORT"
  new_log "$log"
  (
    # Clear every inherited dynamic-loader control, including variables not
    # known when this protocol was written.  The exec helper independently
    # derives and rechecks the same CUDA 12.6-only environment.
    local ld_name
    while IFS= read -r ld_name; do
      unset "$ld_name"
    done < <(compgen -A variable LD_)
    export LD_LIBRARY_PATH="$LLAMA_CUDA_LIBRARY_DIR"
    exec setsid python3 "$RUNTIME_PROVENANCE" exec-llama \
      "$LLAMA_SERVER" "$LLAMA_CUDA_LIBRARY_DIR" -- \
      -m "$artifact" -ngl "$ngl" -c 8192 -ctk q8_0 -ctv q8_0 \
      --flash-attn on --jinja --host 127.0.0.1 --port "$PORT" -t 16 \
      --alias "$alias" -lv 4
  ) >>"$log" 2>&1 &
  local pid=$!
  wait_for_executable "$pid" "$(readlink -f "$LLAMA_SERVER")"
  python3 "$IDENTITY" capture-process --pid "$pid" --output "$process_record"
  ACTIVE_PROCESS_RECORD="$process_record"
  ACTIVE_RECEIPT="$receipt"
  ACTIVE_SHUTDOWN="$shutdown"
  ACTIVE_STAGE="$stage"
  python3 "$IDENTITY" create \
    --base-url "$BASE_URL" --expected-pid "$pid" \
    --expected-model-id "$alias" --expected-owned-by llamacpp \
    --engine llama.cpp --engine-revision "$LLAMA_REVISION" \
    --artifact-record "$artifact_record" --process-record "$process_record" \
    --server-log "$log" --receipt "$receipt" \
    --cmd-fragment "$artifact" --cmd-fragment "$alias" --cmd-fragment "$PORT" \
    --forbidden-log-pattern "Address already in use" \
    --forbidden-log-pattern "couldn't bind" \
    --forbidden-log-pattern "failed to bind" \
    --required-log-pattern "offloaded $expected_layers layers to GPU" \
    --wait-seconds 900
  python3 "$CAMPAIGN_GUARD" placement \
    --engine llama.cpp --expected-layers "$expected_layers" \
    --log "$log" --identity-receipt "$receipt" --output "$placement" \
    --toolchain-receipt "$CAMPAIGN_DIR/toolchain.json"
  CURRENT_RECEIPT="$receipt"
}

launch_exl3() {
  local stage=$1
  local config="$PRIVATE_INPUTS/runtime/tabby.yml"
  local log="$CAMPAIGN_DIR/logs/$stage.log"
  local process_record="$CAMPAIGN_DIR/process/$stage.json"
  local receipt="$CAMPAIGN_DIR/identity/$stage.json"
  local shutdown="$CAMPAIGN_DIR/shutdown/$stage.json"
  local placement="$CAMPAIGN_DIR/placement/$stage.json"
  python3 "$IDENTITY" port-free --port "$PORT" --port "$FALLBACK_PORT"
  new_log "$log"
  (
    cd "$TABBY_ROOT"
    # The EXL3 wheel carries its own CUDA user-space stack.  Remove every
    # inherited dynamic-loader control and do not set LD_LIBRARY_PATH; the
    # live placement receipt independently verifies the resulting mappings.
    local ld_name
    while IFS= read -r ld_name; do
      unset "$ld_name"
    done < <(compgen -A variable LD_ || true)
    if compgen -A variable LD_ >/dev/null; then
      die "could not clear all LD_* controls before EXL3 launch"
    fi
    exec setsid "$EXL_PYTHON" start.py --config "$config"
  ) >>"$log" 2>&1 &
  local pid=$!
  wait_for_executable "$pid" "$(readlink -f "$EXL_PYTHON")"
  python3 "$IDENTITY" capture-process --pid "$pid" --output "$process_record"
  ACTIVE_PROCESS_RECORD="$process_record"
  ACTIVE_RECEIPT="$receipt"
  ACTIVE_SHUTDOWN="$shutdown"
  ACTIVE_STAGE="$stage"
  python3 "$IDENTITY" create \
    --base-url "$BASE_URL" --expected-pid "$pid" \
    --expected-model-id qwen38-exl3-2.0 --expected-owned-by tabbyAPI \
    --engine ExLlamaV3/tabbyAPI --engine-revision "$TABBY_REVISION" \
    --artifact-record "$ARTIFACT_RECORDS/exl3.json" \
    --process-record "$process_record" --server-log "$log" --receipt "$receipt" \
    --expected-cwd "$TABBY_ROOT" --config-file "$config" \
    --cmd-fragment start.py --cmd-fragment "$config" \
    --required-config-line "port: $PORT" \
    --required-config-line "model_dir: $MODEL_ROOT" \
    --required-config-line "model_name: qwen38-exl3-2.0" \
    --required-config-line "backend: exllamav3" \
    --required-config-line "max_seq_len: 8192" \
    --required-config-line "cache_size: 8192" \
    --required-config-line "cache_mode: Q8" \
    --required-log-pattern "Loading model: $EXL3_ARTIFACT" \
    --required-log-pattern "Loading with a manual GPU split (or a one GPU setup)" \
    --forbidden-log-pattern "Switching to" \
    --forbidden-log-pattern "Insufficient VRAM" \
    --wait-seconds 900
  python3 "$CAMPAIGN_GUARD" placement \
    --engine ExLlamaV3/tabbyAPI --log "$log" \
    --identity-receipt "$receipt" --output "$placement" \
    --toolchain-receipt "$CAMPAIGN_DIR/toolchain.json"
  CURRENT_RECEIPT="$receipt"
}

stop_current() {
  python3 "$IDENTITY" terminate \
    --receipt "$ACTIVE_RECEIPT" --shutdown "$ACTIVE_SHUTDOWN" --wait-seconds 30
  ACTIVE_PROCESS_RECORD=""
  ACTIVE_RECEIPT=""
  ACTIVE_SHUTDOWN=""
  ACTIVE_STAGE=""
  CURRENT_RECEIPT=""
}

generate_questions() {
  local corpus=$1
  local lc=$2
  local frame=$3
  local target=$4
  local input_dir="$PRIVATE_INPUTS/$lc"
  local question_dir="$ATTEMPT_QUESTION_ROOT/$lc"
  mkdir -p "$question_dir"
  chmod 700 "$question_dir"
  python3 "$GEN_QUESTIONS" \
    --corpus "$corpus" --attempt-id "$ATTEMPT_ID" \
    --windows "$input_dir/windows.jsonl" \
    --out "$question_dir/questions.jsonl" \
    --journal "$question_dir/question_journal.jsonl" \
    --attempt-log "$question_dir/question_attempts.jsonl" \
    --manifest "$question_dir/question_manifest.json" \
    --contract "$question_dir/question_contract.json" \
    --sessions "$question_dir/question_sessions.jsonl" \
    --identity-receipt "$CURRENT_RECEIPT" \
    --max-windows "$frame" --target-items "$target"
}

run_corpus() {
  local tier=$1
  local corpus=$2
  local lc=$3
  local target=$4
  local input_dir="$PRIVATE_INPUTS/$lc"
  local question_dir="$ATTEMPT_QUESTION_ROOT/$lc"
  python3 "$RUN_GRID" \
    --label "P2R1-$tier-$corpus" \
    --attempt-id "$ATTEMPT_ID" \
    --windows "$input_dir/windows.jsonl" \
    --questions "$question_dir/questions.jsonl" \
    --input-manifest "$input_dir/input_manifest.json" \
    --question-manifest "$question_dir/question_manifest.json" \
    --question-contract "$question_dir/question_contract.json" \
    --question-journal "$question_dir/question_journal.jsonl" \
    --question-attempts "$question_dir/question_attempts.jsonl" \
    --question-sessions "$question_dir/question_sessions.jsonl" \
    --identity-receipt "$CURRENT_RECEIPT" --max-items "$target"
}

cool_gate() {
  local stage=$1
  python3 "$CAMPAIGN_GUARD" cool --stage "$stage" \
    --output "$CAMPAIGN_DIR/host/${stage}-pre.json" \
    --max-gpu-temp 50 --max-load 8 --min-available-gib 32 \
    --gpu-utilization-policy record-only-device-total \
    --max-gpu-memory-mib 1536 --timeout 1800
}

warmup_current() {
  local stage=$1
  python3 "$WARMUP" --identity-receipt "$CURRENT_RECEIPT" \
    --output "$CAMPAIGN_DIR/warmup/$stage.json"
}

run_campaign() {
  require_model_root
  require_protocol_frozen
  require_inputs
  require_file "$PRIVATE_INPUTS/runtime/tabby.yml"
  python3 "$SCRIPT_DIR/validate_inputs_r1.py" \
    --tabby-config --model-root "$MODEL_ROOT"
  require_artifact_records_fast
  exec 9>"$R1_ROOT/campaign.lock"
  flock -n 9 || die "another P2R1 campaign holds $R1_ROOT/campaign.lock"
  ATTEMPT_ID=${P2R1_ATTEMPT_ID:-}
  [[ -n "$ATTEMPT_ID" ]] \
    || die "set explicit P2R1_ATTEMPT_ID=p2r1-primary-v5; retries require a committed protocol amendment"
  [[ "$ATTEMPT_ID" == p2r1-primary-v5 ]] \
    || die "official protocol freezes P2R1_ATTEMPT_ID=p2r1-primary-v5; do not resample under a new ID"
  [[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{5,79}$ ]] \
    || die "invalid P2R1_ATTEMPT_ID: $ATTEMPT_ID"
  ATTEMPT_RESULT_ROOT="$RESULTS_ROOT/paper2-r1/$ATTEMPT_ID"
  ATTEMPT_QUESTION_ROOT="$R1_ROOT/private/attempts/$ATTEMPT_ID/questions"
  [[ ! -e "$ATTEMPT_RESULT_ROOT/ATTEMPT_INVALIDATED.json" ]] \
    || die "official attempt is permanently invalidated; a retry requires a committed protocol amendment"
  mkdir -p "$ATTEMPT_RESULT_ROOT" "$ATTEMPT_QUESTION_ROOT"
  chmod 700 "$ATTEMPT_RESULT_ROOT" "$ATTEMPT_QUESTION_ROOT"
  echo "P2R1 attempt namespace: $ATTEMPT_ID"
  echo "Only a provably eligible crash may resume with P2R1_ATTEMPT_ID=$ATTEMPT_ID"
  CAMPAIGN_ID=$(tr -d '-' </proc/sys/kernel/random/uuid)
  CAMPAIGN_DIR="$RECEIPT_ROOT/campaigns/$CAMPAIGN_ID"
  mkdir -m 700 -p \
    "$CAMPAIGN_DIR"/{logs,process,identity,shutdown,placement,host,warmup,artifacts,runtime}
  LLAMA_SERVER="$REPO_ROOT/engines/llama.cpp/build/bin/llama-server"
  TABBY_ROOT="$REPO_ROOT/engines/tabbyAPI"
  EXL_PYTHON="$REPO_ROOT/engines/exl3-env/bin/python"
  require_file "$LLAMA_SERVER"
  require_file "$EXL_PYTHON"
  trap on_exit EXIT
  trap 'on_signal 130' INT
  trap 'on_signal 143' TERM
  python3 "$CAMPAIGN_GUARD" host --attempt-id "$ATTEMPT_ID" \
    --output "$CAMPAIGN_DIR/host/inventory.json"
  python3 "$CAMPAIGN_GUARD" bind-host --attempt-id "$ATTEMPT_ID" \
    --host-receipt "$CAMPAIGN_DIR/host/inventory.json" \
    --output "$ATTEMPT_RESULT_ROOT/HOST_BINDING.json"
  python3 "$CAMPAIGN_GUARD" attempt --attempt-id "$ATTEMPT_ID" \
    --result-root "$ATTEMPT_RESULT_ROOT" \
    --question-root "$ATTEMPT_QUESTION_ROOT" \
    --host-binding "$ATTEMPT_RESULT_ROOT/HOST_BINDING.json" \
    --output "$CAMPAIGN_DIR/attempt.json"
  python3 "$CAMPAIGN_GUARD" toolchain --output "$CAMPAIGN_DIR/toolchain.json" \
    --host-receipt "$CAMPAIGN_DIR/host/inventory.json" \
    --attempt-host-binding "$ATTEMPT_RESULT_ROOT/HOST_BINDING.json"
  LLAMA_CUDA_LIBRARY_DIR=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["llama_cpp"]["cuda_runtime"]["private_library_directory"])' \
    "$CAMPAIGN_DIR/toolchain.json")
  require_dir "$LLAMA_CUDA_LIBRARY_DIR"

  cool_gate q4-question-generation
  launch_llama Q4KXL "$Q4_ARTIFACT" "$ARTIFACT_RECORDS/q4kxl.json" 33 33/66 q4-question-generation
  generate_questions TSI tsi 126 117
  generate_questions WC wildchat 160 120
  python3 "$CAMPAIGN_GUARD" snapshot --stage q4-question-generation-post \
    --output "$CAMPAIGN_DIR/host/q4-question-generation-post.json"
  stop_current

  cool_gate q4-evaluation
  launch_llama Q4KXL "$Q4_ARTIFACT" "$ARTIFACT_RECORDS/q4kxl.json" 33 33/66 q4-evaluation
  warmup_current q4-evaluation
  run_corpus Q4KXL TSI tsi 117
  run_corpus Q4KXL WC wildchat 120
  python3 "$CAMPAIGN_GUARD" snapshot --stage q4-evaluation-post \
    --output "$CAMPAIGN_DIR/host/q4-evaluation-post.json"
  stop_current

  cool_gate exl3-evaluation
  launch_exl3 exl3-evaluation
  warmup_current exl3-evaluation
  run_corpus EXL3 TSI tsi 117
  run_corpus EXL3 WC wildchat 120
  python3 "$CAMPAIGN_GUARD" snapshot --stage exl3-evaluation-post \
    --output "$CAMPAIGN_DIR/host/exl3-evaluation-post.json"
  stop_current

  cool_gate iq2s-evaluation
  launch_llama IQ2S "$IQ2_ARTIFACT" "$ARTIFACT_RECORDS/iq2s.json" 99 65/65 iq2s-evaluation
  warmup_current iq2s-evaluation
  run_corpus IQ2S TSI tsi 117
  run_corpus IQ2S WC wildchat 120
  python3 "$CAMPAIGN_GUARD" snapshot --stage iq2s-evaluation-post \
    --output "$CAMPAIGN_DIR/host/iq2s-evaluation-post.json"
  stop_current

  # Full reads occur only after every timed request and shutdown, so this
  # closes same-size/mtime mutation ambiguity without priming benchmark timing.
  python3 "$CAMPAIGN_GUARD" post-artifacts \
    --q4-record "$ARTIFACT_RECORDS/q4kxl.json" \
    --exl3-record "$ARTIFACT_RECORDS/exl3.json" \
    --iq2-record "$ARTIFACT_RECORDS/iq2s.json" \
    --output "$CAMPAIGN_DIR/artifacts/post-campaign.json"
  python3 "$CAMPAIGN_GUARD" post-runtime \
    --toolchain-receipt "$CAMPAIGN_DIR/toolchain.json" \
    --output "$CAMPAIGN_DIR/runtime/post-campaign.json"
  python3 "$REPO_ROOT/paper2/scripts/validate_r1.py" --campaign "$CAMPAIGN_DIR"
  trap - EXIT INT TERM
  echo "P2R1 campaign complete and validated: $CAMPAIGN_DIR"
}

MODE=${1:-}
case "$MODE" in
  self-test) run_self_test ;;
  prepare-inputs) prepare_inputs ;;
  prepare-runtime) prepare_runtime ;;
  fingerprint-artifacts) fingerprint_artifacts ;;
  preflight) preflight ;;
  run) run_campaign ;;
  *) usage; exit 2 ;;
esac
