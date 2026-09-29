# P4R1 corrective validation: complete runbook

Paper 4 is not publishable from its historical six-cell campaign. The files
survive for forensic review, but none of the six primary cells independently
proves that every response came from the intended model endpoint. P4R1 is the
replacement campaign. It is corrective and exploratory, not independent,
because the item IDs, historical outcomes, and hypothesis were known before
this protocol was written.

This runbook does not publish anything and does not overwrite historical data.
Every output path must be new.

## Frozen contract

P4R1 evaluates six GGUF artifact--engine pairs on the same 100 GSM8K items.
Every cell uses context 4,096, one request at a time, temperature 1.0, top-p
0.95, top-k 20, requested reasoning effort `medium`, and at most 2,048 generated
tokens. Item `gsm8k_n` uses inference seed `4,200,000 + n` in all six cells.

Mixtral requests 29, 26, and 12 GPU layers for IQ1_M, IQ2_XXS, and Q4_K_M.
Qwen omits the GPU-layer override and records the actual automatic-fit result.
The complete contract is in `P4R1_PROTOCOL.md`.

The six GGUFs occupy 90,436,239,456 bytes (84.23 GiB). Allow at least 100 GiB
for weights, the engine build, immutable logs, and private evidence.

## Tested host and engine gate

The corrective campaign is bound to this observed environment:

- NVIDIA RTX 4080 Laptop GPU, 12,282 MiB VRAM, 80 W limit
- Intel Core i9-14900HX, 62 GiB system RAM, NVMe storage
- NVIDIA driver 580.167.08; driver-reported CUDA compatibility 13.0
- llama.cpp commit `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`
- CMake 3.28.3, GNU C++ 13.3.0, CUDA toolkit 12.6.85, CUDA architecture 89
- Release build with `GGML_CUDA=ON`, `GGML_NATIVE=ON`, `GGML_OPENMP=ON`,
  `GGML_BLAS=OFF`, and `LLAMA_CURL=OFF`

The runner reads these facts back from the executable, clean Git checkout,
`CMakeCache.txt`, compilers, toolkit, and driver before inference. Driver CUDA
compatibility 13.0 is not the toolkit version used to compile the binary.

```bash
git clone https://github.com/ggml-org/llama.cpp.git engines/llama.cpp
git -C engines/llama.cpp checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
cmake -S engines/llama.cpp -B engines/llama.cpp/build \
  -DCUDAToolkit_ROOT="$HOME/cuda-12.6" \
  -DCMAKE_CUDA_COMPILER="$HOME/cuda-12.6/bin/nvcc" \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_BLAS=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build engines/llama.cpp/build --parallel --target llama-server
```

A different host or toolchain is a portability study, not P4R1. Freeze it under
a new protocol instead of weakening this gate.

## Obtain the six exact artifacts

Download the pinned objects into two explicit directories. The runner does not
guess a model root.

```bash
python3 -m venv hf-env
hf-env/bin/pip install huggingface_hub

hf-env/bin/hf download mradermacher/Mixtral-8x7B-Instruct-v0.1-i1-GGUF \
  Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf \
  Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf \
  Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf \
  --revision 0e8d21e9cc0d28b59173d329348f60f8628d1c05 \
  --local-dir models/mixtral-8x7b

hf-env/bin/hf download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf \
  Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf \
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --revision eea7b2be5805a5f151f8847ede8e5f9a9284bf77 \
  --local-dir models/qwen3-30b-a3b
```

Expected SHA-256 values:

```text
7e8d75c23bd4de55582ac91084a0575f081c1de374b693f5ea7ffa6fa66a2e3c  Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf
db62b07756f50d2f046b0a00dc594f1aa73dba82ebdc182d9f55c96474655f03  Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf
7f2f98f301a74b645d0a4c2bd64c98e9811f2dfa26dbae4787c280eb1183632c  Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf
d527a854db2a1582a3ce746a17b1f42d860334ece18d385ede9e2e395058b39e  Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf
aa0b04ac05f71aa94b542b074500b3de956608365e74f0e1d97ae0d6a6380269  Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf
6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

## Build the private dataset and one-time model manifest

Benchmark questions and answers remain private. The builder downloads GSM8K
test data at revision `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`, checks
source SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`,
and reconstructs the seed-42 subset. Its required output SHA-256 is
`184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37`.

```bash
mkdir -m 700 private-p4r1
python3 replication/prepare_gsm8k.py --output private-p4r1/gsm8k_100.jsonl
python3 replication/run_gsm8k.py --selftest
python3 replication/prepare_model_manifest.py \
  --mixtral-dir models/mixtral-8x7b \
  --qwen-dir models/qwen3-30b-a3b \
  --output private-p4r1/model-manifest.json
```

The manifest fully hashes all 84.23 GiB once. Do this before the campaign, then
allow page-cache and thermal state to normalize. Live cells verify path, size,
device, inode, and mtime without rereading weight contents immediately before
timed requests. After every cell has shut down, `run_matrix.sh` fully hashes all
six artifacts again and binds that post-campaign receipt into the campaign.

## Freeze and approve the prospective plan

Commit the protocol, runner, scoring code, promoter, checker, tests, claims and
figure builders, manuscript gate, and Makefile. The plan command refuses dirty
or untracked listed sources. Inference/scoring and result-interpreting source
inventories are recorded separately. Under the implemented strict gate, any
listed inference, scoring, promotion, or analysis-source change blocks this
campaign's promotion and requires a new protocol, plan, approval, and campaign.
Unpromoted raw files remain forensic material; no amendment path is implemented.

```bash
python3 replication/prepare_campaign_plan.py \
  --output private-p4r1/protocol-plan.json
sha256sum private-p4r1/protocol-plan.json
```

Copy the printed hash into the next command. The author must explicitly
authorize the exact frozen operation; this approval authorizes inference, not
publication. Normally the author types the exact interactive challenge. The
author may instead delegate only the mechanical challenge entry through the
active working conversation, provided the approval receipt's actor field says
that it records explicit chat authorization executed by the named agent. This
delegation does not extend to publication or to a different plan hash.

```bash
python3 replication/approve_campaign_plan.py \
  --plan private-p4r1/protocol-plan.json \
  --plan-sha256 PLAN_SHA256 \
  --output private-p4r1/protocol-plan-approval.json
```

## Run the six cells

Before starting, ensure stable cooling, enough free RAM, and no competing GPU
process. The matrix preflights all six artifacts before the first server. It
uses one port input, refuses an occupied port, proves PID/start-time and
listener ownership, requires the exact alias from `/v1/models` and every
completion, records immutable per-cell logs and placement, and proves the same
process is gone and the port free at shutdown.

Each server is launched with explicit trace verbosity (`-lv 4`). Immediately
after endpoint readiness and before the first benchmark request, the runner's
log probe must find one unambiguous `offloaded X/Y layers to GPU` readback in
the cell's preserved stdout/stderr. For an explicit Mixtral `-ngl`, the
observed `X` must also equal the request. A missing, ambiguous, or mismatched
readback aborts before evaluation; that new campaign directory is preserved as
a failed attempt and is never appended to or promoted.

```bash
export LLAMA_SERVER="$PWD/engines/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="$HOME/cuda-12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MIXTRAL_MODEL_DIR="$PWD/models/mixtral-8x7b"
export QWEN_MODEL_DIR="$PWD/models/qwen3-30b-a3b"
export GSM8K_DATASET="$PWD/private-p4r1/gsm8k_100.jsonl"
export MODEL_IDENTITY_MANIFEST="$PWD/private-p4r1/model-manifest.json"
export MODEL_IDENTITY_MANIFEST_SHA256=MANIFEST_SHA256
export P4R1_PLAN_RECEIPT="$PWD/private-p4r1/protocol-plan.json"
export P4R1_PLAN_SHA256=PLAN_SHA256
export P4R1_PLAN_APPROVAL="$PWD/private-p4r1/protocol-plan-approval.json"
export RESULTS_DIR="$PWD/private-p4r1/campaign-001"
bash replication/run_matrix.sh
```

The matrix command does not emit a campaign receipt until post-campaign
full-content verification of all six model files and a second loader
resolution plus full hash of the CUDA runtime libraries pass.

Historical completion-request time summed to about 79 minutes, but loading,
identity checks, cooling, and host conditions make the wall-clock budget longer.
Reserve roughly two to three hours; this is an operational estimate, not a
performance claim.

An identity or transport failure quarantines the cell. Do not patch or append
to a partial campaign. Correct the cause and use a new campaign directory.

## Promote, check, and build claims without publishing

Promotion preserves private responses and gold answers under owner-only
permissions, independently re-scores all 600 rows, validates exact error
telemetry and lifecycle receipts, and creates a privacy-sanitized public
candidate. It never edits historical evidence.

```bash
python3 replication/promote_validation.py \
  --campaign private-p4r1/campaign-001 \
  --model-manifest private-p4r1/model-manifest.json \
  --dataset private-p4r1/gsm8k_100.jsonl \
  --destination private-p4r1/promoted-001

python3 replication/check_p4r1_evidence.py \
  private-p4r1/promoted-001/public-candidate

python3 ../../../../paper4/scripts/build_p4r1_claims.py \
  --promoted private-p4r1/promoted-001 \
  --promotion-receipt-sha256 PROMOTION_RECEIPT_SHA256 \
  --output-dir ../../../../paper4/derived-p4r1
```

The final manuscript is still blocked after these commands. Its abstract,
methods, limitations, results, explainer, and brief must disclose the historical
provenance failure, describe P4R1 as a post-outcome corrective exploratory run,
and be reviewed against the exact new outcome signature. Only the reviewed,
hash-bound narrative path may produce a canonical release candidate. The
manuscript checker also requires the four generated signed-direction macros,
so a low tier that beats Q4 cannot inherit stale Q4-preservation or low-bit-loss
prose merely because the absolute difference is large.
