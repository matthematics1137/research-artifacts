#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
llama_server=$(realpath -- "${LLAMA_SERVER:-"$PWD/engines/llama.cpp/build/bin/llama-server"}")
mixtral_dir=$(realpath -- "${MIXTRAL_MODEL_DIR:?set MIXTRAL_MODEL_DIR to the directory containing the three Mixtral GGUFs}")
qwen_dir=$(realpath -- "${QWEN_MODEL_DIR:?set QWEN_MODEL_DIR to the directory containing the three Qwen GGUFs}")
manifest=$(realpath -- "${MODEL_IDENTITY_MANIFEST:?set MODEL_IDENTITY_MANIFEST to the pre-campaign manifest}")
manifest_sha256=${MODEL_IDENTITY_MANIFEST_SHA256:?set MODEL_IDENTITY_MANIFEST_SHA256 to the hash printed by prepare_model_manifest.py}
results=${RESULTS_DIR:?set RESULTS_DIR to a new, nonexistent validation-campaign directory}
prospective_plan=$(realpath -- "${P4R1_PLAN_RECEIPT:?set P4R1_PLAN_RECEIPT to the committed-clean prospective plan}")
prospective_plan_sha256=${P4R1_PLAN_SHA256:?set P4R1_PLAN_SHA256 to the hash printed by prepare_campaign_plan.py}
plan_approval=$(realpath -- "${P4R1_PLAN_APPROVAL:?set P4R1_PLAN_APPROVAL to the interactive approval for this plan}")
guard=(python3 -B "$script_dir/server_guard.py")

[[ "$manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "MODEL_IDENTITY_MANIFEST_SHA256 is malformed" >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "missing model manifest: $manifest" >&2; exit 2; }
[[ -x "$llama_server" ]] || { echo "missing executable: $llama_server" >&2; exit 2; }
[[ "$prospective_plan_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "P4R1_PLAN_SHA256 is malformed" >&2; exit 2; }
[[ -f "$prospective_plan" ]] || { echo "missing prospective plan: $prospective_plan" >&2; exit 2; }
[[ -f "$plan_approval" ]] || { echo "missing prospective-plan approval: $plan_approval" >&2; exit 2; }
[[ ! -e "$results" ]] || { echo "refusing existing validation campaign: $results" >&2; exit 2; }
python3 -B "$script_dir/prepare_campaign_plan.py" \
  --verify "$prospective_plan" --sha256 "$prospective_plan_sha256"
python3 -B "$script_dir/approve_campaign_plan.py" \
  --plan "$prospective_plan" --plan-sha256 "$prospective_plan_sha256" \
  --verify "$plan_approval"
mkdir -p "$(dirname -- "$results")"
mkdir "$results"
mkdir "$results/preflight"
install -m 0600 -- "$prospective_plan" "$results/protocol-plan.json"
install -m 0600 -- "$plan_approval" "$results/protocol-plan-approval.json"
"${guard[@]}" record-host \
  --workspace "$script_dir/../../../../.." \
  --mixtral-dir "$mixtral_dir" --qwen-dir "$qwen_dir" \
  --output "$results/host-receipt.json"
"${guard[@]}" record-toolchain \
  --launcher "$llama_server" --output "$results/toolchain-receipt.json"
host_receipt_sha256=$(sha256sum "$results/host-receipt.json")
host_receipt_sha256=${host_receipt_sha256%% *}
export P4R1_HOST_RECEIPT="$results/host-receipt.json"
export P4R1_HOST_RECEIPT_SHA256="$host_receipt_sha256"
export LLAMA_SERVER="$llama_server"

mixtral_iq1="$mixtral_dir/Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf"
mixtral_iq2="$mixtral_dir/Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf"
mixtral_q4="$mixtral_dir/Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf"
qwen_iq1="$qwen_dir/Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf"
qwen_iq2="$qwen_dir/Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf"
qwen_q4="$qwen_dir/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"

preflight() {
  local alias=$1 model=$2
  [[ -f "$model" ]] || { echo "missing validation artifact: $model" >&2; exit 2; }
  "${guard[@]}" verify-model-entry \
    --manifest "$manifest" --manifest-sha256 "$manifest_sha256" \
    --model "$model" --alias "$alias" \
    --output "$results/preflight/${alias}.json"
}

# Validate every path/stat identity before launching any server. Full content
# hashing is a separate pre-campaign step, so this does not reread GGUF bytes.
preflight P4R1-MIXTRAL-IQ1M "$mixtral_iq1"
preflight P4R1-MIXTRAL-IQ2XXS "$mixtral_iq2"
preflight P4R1-MIXTRAL-Q4KM "$mixtral_q4"
preflight P4R1-QWEN3MOE-IQ1M "$qwen_iq1"
preflight P4R1-QWEN3MOE-IQ2XXS "$qwen_iq2"
preflight P4R1-QWEN3MOE-Q4KM "$qwen_q4"
chmod 0600 -- "$results/preflight/"*.json
chmod 0700 -- "$results/preflight"

run=(bash "$script_dir/run_one.sh")
# P4R1 freezes context at 4096 for every cell. run_gsm8k.py assigns the same
# deterministic per-item sampler seed across all six artifacts.
"${run[@]}" P4R1-MIXTRAL-IQ1M "$mixtral_iq1" 29 4096 "$results/P4R1-MIXTRAL-IQ1M.jsonl"
"${run[@]}" P4R1-MIXTRAL-IQ2XXS "$mixtral_iq2" 26 4096 "$results/P4R1-MIXTRAL-IQ2XXS.jsonl"
"${run[@]}" P4R1-MIXTRAL-Q4KM "$mixtral_q4" 12 4096 "$results/P4R1-MIXTRAL-Q4KM.jsonl"
"${run[@]}" P4R1-QWEN3MOE-IQ1M "$qwen_iq1" auto 4096 "$results/P4R1-QWEN3MOE-IQ1M.jsonl"
"${run[@]}" P4R1-QWEN3MOE-IQ2XXS "$qwen_iq2" auto 4096 "$results/P4R1-QWEN3MOE-IQ2XXS.jsonl"
"${run[@]}" P4R1-QWEN3MOE-Q4KM "$qwen_q4" auto 4096 "$results/P4R1-QWEN3MOE-Q4KM.jsonl"

# Timed cells and their PID-scoped shutdowns are complete. Reread every full
# artifact now so the campaign proves that weight contents did not change
# behind the fast stat identity used during live timing.
"${guard[@]}" verify-manifest-content \
  --manifest "$manifest" --manifest-sha256 "$manifest_sha256" \
  --toolchain-receipt "$results/toolchain-receipt.json" \
  --output "$results/post-model-verification.json"

"${guard[@]}" record-campaign \
  --manifest "$manifest" --manifest-sha256 "$manifest_sha256" \
  --protocol-plan "$results/protocol-plan.json" \
  --protocol-plan-sha256 "$prospective_plan_sha256" \
  --protocol-plan-approval "$results/protocol-plan-approval.json" \
  --host-receipt "$results/host-receipt.json" \
  --post-model-verification "$results/post-model-verification.json" \
  --toolchain-receipt "$results/toolchain-receipt.json" \
  --cell-receipt "$results/P4R1-MIXTRAL-IQ1M-server/cell-receipt.json" \
  --cell-receipt "$results/P4R1-MIXTRAL-IQ2XXS-server/cell-receipt.json" \
  --cell-receipt "$results/P4R1-MIXTRAL-Q4KM-server/cell-receipt.json" \
  --cell-receipt "$results/P4R1-QWEN3MOE-IQ1M-server/cell-receipt.json" \
  --cell-receipt "$results/P4R1-QWEN3MOE-IQ2XXS-server/cell-receipt.json" \
  --cell-receipt "$results/P4R1-QWEN3MOE-Q4KM-server/cell-receipt.json" \
  --output "$results/campaign-receipt.json"
chmod 0600 -- "$results/campaign-receipt.json" "$results/protocol-plan.json" \
  "$results/protocol-plan-approval.json" "$results/toolchain-receipt.json" \
  "$results/post-model-verification.json"
chmod 0600 -- "$results/host-receipt.json"
chmod 0700 -- "$results"
echo "P4R1 six-cell validation complete and hash-bound: $results"
