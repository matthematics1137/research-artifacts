#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 5 ]]; then
  echo "usage: $0 LABEL MODEL_GGUF NGL_OR_AUTO CONTEXT OUTPUT_JSONL" >&2
  exit 2
fi

label=$1
model=$(realpath -- "$2")
ngl=$3
context=$4
output=$5
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
llama_server=$(realpath -- "${LLAMA_SERVER:-"$PWD/engines/llama.cpp/build/bin/llama-server"}")
model_manifest=$(realpath -- "${MODEL_IDENTITY_MANIFEST:?set MODEL_IDENTITY_MANIFEST to the pre-campaign manifest}")
model_manifest_sha256=${MODEL_IDENTITY_MANIFEST_SHA256:?set MODEL_IDENTITY_MANIFEST_SHA256 to the printed manifest hash}
host_receipt=$(realpath -- "${P4R1_HOST_RECEIPT:?set P4R1_HOST_RECEIPT to the campaign host receipt}")
host_receipt_sha256=${P4R1_HOST_RECEIPT_SHA256:?set P4R1_HOST_RECEIPT_SHA256 to the campaign host-receipt hash}
host=127.0.0.1
port=${PORT:-8090}
base_url="http://${host}:${port}"
dataset=${GSM8K_DATASET:-"$script_dir/work/gsm8k_100.jsonl"}
guard=(python3 -B "$script_dir/server_guard.py")
startup_attempts=${SERVER_START_ATTEMPTS:-600}
shutdown_attempts=${SERVER_STOP_ATTEMPTS:-30}
poll_seconds=${SERVER_POLL_SECONDS:-1}

[[ -x "$llama_server" ]] || { echo "missing executable: $llama_server" >&2; exit 2; }
[[ -f "$model" ]] || { echo "missing model: $model" >&2; exit 2; }
[[ -f "$model_manifest" ]] || { echo "missing model manifest: $model_manifest" >&2; exit 2; }
[[ -f "$dataset" ]] || { echo "missing dataset: $dataset (run prepare_gsm8k.py first)" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite: $output" >&2; exit 2; }
[[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "label must contain only letters, digits, dot, underscore, or hyphen" >&2; exit 2; }
[[ "$model_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "MODEL_IDENTITY_MANIFEST_SHA256 must be 64 lowercase hexadecimal characters" >&2; exit 2; }
[[ "$host_receipt_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "P4R1_HOST_RECEIPT_SHA256 must be 64 lowercase hexadecimal characters" >&2; exit 2; }
[[ -f "$host_receipt" ]] || { echo "missing host receipt: $host_receipt" >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 65535 )) || { echo "PORT must be an integer from 1024 to 65535" >&2; exit 2; }
[[ "$startup_attempts" =~ ^[1-9][0-9]*$ ]] || { echo "SERVER_START_ATTEMPTS must be a positive integer" >&2; exit 2; }
[[ "$shutdown_attempts" =~ ^[1-9][0-9]*$ ]] || { echo "SERVER_STOP_ATTEMPTS must be a positive integer" >&2; exit 2; }
if [[ -n ${BASE_URL+x} ]]; then
  echo "BASE_URL is not accepted; the runner derives its URL from PORT" >&2
  exit 2
fi
"${guard[@]}" port-free --host "$host" --port "$port" || {
  echo "refusing to launch because ${host}:${port} is not free" >&2
  exit 1
}

mkdir -p "$(dirname -- "$output")"
log_dir="$(dirname -- "$output")/${label}-server"
[[ ! -e "$log_dir" ]] || { echo "refusing to overwrite server logs: $log_dir" >&2; exit 2; }
mkdir "$log_dir"
"${guard[@]}" verify-model-entry \
  --manifest "$model_manifest" --manifest-sha256 "$model_manifest_sha256" \
  --model "$model" --alias "$label" \
  --output "$log_dir/model-identity.json"

server=("$llama_server" -m "$model" -c "$context" -t 16 --jinja -lv 4 \
  --host "$host" --port "$port" --alias "$label")
if [[ "$ngl" != auto ]]; then
  server+=(-ngl "$ngl")
fi

server_pid=
server_start_time=
stop_server() {
  local stop_ok=0
  if [[ -n "$server_pid" && -n "$server_start_time" ]] && \
      "${guard[@]}" same-process --pid "$server_pid" --start-time "$server_start_time"; then
    kill -INT "$server_pid" 2>/dev/null || true
    for ((attempt = 0; attempt < shutdown_attempts; attempt++)); do
      "${guard[@]}" same-process --pid "$server_pid" --start-time "$server_start_time" || break
      sleep "$poll_seconds"
    done
    if "${guard[@]}" same-process --pid "$server_pid" --start-time "$server_start_time"; then
      echo "server did not stop after SIGINT; sending SIGTERM to PID $server_pid" >&2
      kill -TERM "$server_pid" 2>/dev/null || true
      for ((attempt = 0; attempt < shutdown_attempts; attempt++)); do
        "${guard[@]}" same-process --pid "$server_pid" --start-time "$server_start_time" || break
        sleep "$poll_seconds"
      done
    fi
    if "${guard[@]}" same-process --pid "$server_pid" --start-time "$server_start_time"; then
      echo "server did not stop after SIGTERM; sending SIGKILL to PID $server_pid" >&2
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  elif [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    echo "refusing to signal PID $server_pid because its recorded process identity changed" >&2
    stop_ok=1
  fi
  for ((attempt = 0; attempt < shutdown_attempts; attempt++)); do
    if "${guard[@]}" port-free --host "$host" --port "$port" >/dev/null 2>&1; then
      server_pid=
      server_start_time=
      return "$stop_ok"
    fi
    sleep "$poll_seconds"
  done
  echo "port ${host}:${port} is still occupied after launched-PID cleanup" >&2
  return 1
}
on_exit() {
  local status=$?
  trap - EXIT
  stop_server || status=1
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${server[@]}" >"$log_dir/stdout.log" 2>"$log_dir/stderr.log" &
server_pid=$!
if ! server_start_time=$("${guard[@]}" fingerprint --pid "$server_pid"); then
  wait "$server_pid" 2>/dev/null || true
  echo "launched server exited before its process identity could be recorded; inspect $log_dir" >&2
  exit 1
fi

verified=0
for ((attempt = 0; attempt < startup_attempts; attempt++)); do
  if ! "${guard[@]}" same-process --pid "$server_pid" --start-time "$server_start_time"; then
    break
  fi
  if "${guard[@]}" ready \
      --pid "$server_pid" --start-time "$server_start_time" \
      --launcher "$llama_server" --model "$model" --alias "$label" \
      --host "$host" --port "$port" --base-url "$base_url" \
      --context "$context" --ngl "$ngl" \
      >/dev/null 2>&1; then
    verified=1
    break
  fi
  sleep "$poll_seconds"
done
if [[ "$verified" -ne 1 ]]; then
  "${guard[@]}" ready \
    --pid "$server_pid" --start-time "$server_start_time" \
    --launcher "$llama_server" --model "$model" --alias "$label" \
    --host "$host" --port "$port" --base-url "$base_url" \
    --context "$context" --ngl "$ngl" || true
  echo "server identity was not verified; inspect $log_dir" >&2
  exit 1
fi

"${guard[@]}" verify-placement-log \
  --alias "$label" --ngl "$ngl" \
  --stdout "$log_dir/stdout.log" --stderr "$log_dir/stderr.log"

"${guard[@]}" record-ready \
  --phase startup --pid "$server_pid" --start-time "$server_start_time" \
  --launcher "$llama_server" --model "$model" \
  --model-identity "$log_dir/model-identity.json" --alias "$label" \
  --host "$host" --port "$port" --base-url "$base_url" \
  --context "$context" --ngl "$ngl" --output "$log_dir/startup-identity.json"

python3 -B "$script_dir/run_gsm8k.py" \
  --dataset "$dataset" --label "$label" --output "$output" \
  --base-url "$base_url" --model-id "$label" \
  --server-guard "$script_dir/server_guard.py" \
  --server-pid "$server_pid" --server-start-time "$server_start_time" \
  --launcher "$llama_server" --model-path "$model" \
  --host "$host" --port "$port" \
  --startup-identity "$log_dir/startup-identity.json" \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --max-tokens 2048 --reasoning-effort medium

"${guard[@]}" record-ready \
  --phase completion --pid "$server_pid" --start-time "$server_start_time" \
  --launcher "$llama_server" --model "$model" \
  --model-identity "$log_dir/model-identity.json" --alias "$label" \
  --host "$host" --port "$port" --base-url "$base_url" \
  --context "$context" --ngl "$ngl" --output "$log_dir/completion-identity.json"

stopped_pid=$server_pid
stopped_start_time=$server_start_time
stop_server
"${guard[@]}" record-shutdown \
  --pid "$stopped_pid" --start-time "$stopped_start_time" --alias "$label" \
  --host "$host" --port "$port" \
  --startup-identity "$log_dir/startup-identity.json" \
  --completion-identity "$log_dir/completion-identity.json" \
  --output "$log_dir/shutdown-identity.json"
server_pid=
server_start_time=
trap - EXIT INT TERM
"${guard[@]}" record-cell \
  --alias "$label" --context "$context" --ngl "$ngl" \
  --result "$output" --stdout "$log_dir/stdout.log" --stderr "$log_dir/stderr.log" \
  --model-identity "$log_dir/model-identity.json" \
  --startup-identity "$log_dir/startup-identity.json" \
  --completion-identity "$log_dir/completion-identity.json" \
  --shutdown-identity "$log_dir/shutdown-identity.json" \
  --dataset "$dataset" --harness "$script_dir/run_gsm8k.py" \
  --guard-source "$script_dir/server_guard.py" \
  --host-receipt "$host_receipt" --host-receipt-sha256 "$host_receipt_sha256" \
  --runner "$script_dir/run_one.sh" --matrix "$script_dir/run_matrix.sh" \
  --output "$log_dir/cell-receipt.json"
chmod 0600 -- "$output" "$log_dir/stdout.log" "$log_dir/stderr.log" \
  "$log_dir/model-identity.json" "$log_dir/startup-identity.json" \
  "$log_dir/completion-identity.json" "$log_dir/shutdown-identity.json" \
  "$log_dir/cell-receipt.json"
chmod 0700 -- "$log_dir"
echo "completed $label; server logs: $log_dir"
