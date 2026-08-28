# Full inference replication

The analysis-only route in `README.md` verifies the released numbers without
running a model. This document describes a prospective rerun on comparable
hardware. Exact throughput and the maximum GPU-resident layer count are expected
to change on a different machine or software stack.

## Tested machine

- NVIDIA RTX 4080 Laptop GPU, 12,282 MiB VRAM, 80 W cap
- Intel Core i9-14900HX; 62 GiB dual-channel DDR5; NVMe
- Linux 7.0.0; NVIDIA driver 580.167.08; CUDA toolkit 12.6.3
- Approximately 900 MiB background desktop VRAM during measurements

Exact structured metadata is in `environment/hardware.json` and
`environment/software.json`.

## Download and verify weights

Use the immutable revisions and every filename in
`environment/model_artifacts.json`. For example:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-IQ2_S.gguf \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --local-dir models/qwen38
sha256sum models/qwen38/Qwen3.8-27B-UD-IQ2_S.gguf
```

Do not substitute a newer conversion while calling it a replication.
The ten pinned weight files total 81.83 GiB, including the optional 1.28 GiB
MTP draft; allow at least 90 GiB of free storage for weights and ancillary
tokenizer/configuration files. Download the complete EXL3 repository at its
pinned revision because ExLlamaV3 also needs its configuration and tokenizer
files, then verify the two recorded weight-shard hashes.

## Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git engines/llama.cpp
git -C engines/llama.cpp checkout 1729ed5371cd1ac6f6d6f3226f8803b080042839
cmake -S engines/llama.cpp -B engines/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_COMPILER="$HOME/cuda-12.6/bin/nvcc" \
  -DCUDAToolkit_ROOT="$HOME/cuda-12.6"
cmake --build engines/llama.cpp/build -j 28 \
  --target llama-server llama-bench llama-cli llama-perplexity
```

Set `LLAMA_BIN` to the resulting `build/bin` directory. The guarded GGUF
harness is `testsuite/bench.py`; inspect its command preview and safety constants
before launching it on another machine. If using MTP, set `MTP_DRAFT` to the
downloaded draft checkpoint; the release copy of the harness honors that
environment variable.

## ExLlamaV3 stack

The measured backend reported ExLlamaV3 1.4.3 under TabbyAPI. The local TabbyAPI
checkout used for the campaign is associated with commit
`4a4f9f44820303593844f092d424bb7506008733`, but the server log did not print its
commit, so this association is less strongly preserved than the ExLlamaV3
version. Configure `max_seq_len=16384` for the hard suites and 36,864 for the
long-context validation. The tested EXL3 path used no MTP or vision projector.

The campaign's complete Python lock was not captured before the run. The
strongly preserved runtime facts are Python 3.12.3, Torch 2.10.0+cu128,
ExLlamaV3 1.4.3+cu128.torch2.10.0, and the associated TabbyAPI commit above.
Install the stack following TabbyAPI's instructions at that commit, then check
those versions before treating a run as a close replication. A different
Torch/CUDA wheel is a new software condition, not a byte-for-byte recreation.

Use the supplied local-only configuration and launch from the TabbyAPI checkout:

```bash
git clone https://github.com/theroyallab/tabbyAPI.git engines/tabbyAPI
git -C engines/tabbyAPI checkout 4a4f9f44820303593844f092d424bb7506008733
cp tabby-config.example.yml engines/tabbyAPI/config.yml
cd engines/tabbyAPI
../exl3-env/bin/python main.py
```

The example expects the EXL3 directory at `models/qwen38-exl3-2.0` relative to
the artifact root. It binds only `127.0.0.1:8090`, disables remote fetches, uses
Q8 KV cache, and disables MTP and vision. Change both `max_seq_len` and
`cache_size` to 36,864 for the long-context validation.

## Evaluation settings

- OpenAI-compatible endpoint at `http://127.0.0.1:8090`
- Context 8,192 unless the validation run says otherwise
- KV cache q8_0; flash attention enabled
- Temperature 1.0; top-p 0.95; top-k 20
- `reasoning_effort=medium` except the named xhigh checks
- One stochastic generation per item
- Client timeout 600 seconds and one retry

Run the harness self-test before contacting a server:

```bash
python3 testsuite/evals/run_eval.py --selftest
```

Representative GGUF launch (substitute the matrix's artifact and `-ngl`):

```bash
"$LLAMA_BIN/llama-server" \
  -m models/qwen38/Qwen3.8-27B-UD-IQ2_S.gguf -ngl 99 \
  -c 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
  --port 8090 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 &
SERVER_PID=$!
```

For the MTP row, add `-md "$MTP_DRAFT" --spec-type draft-mtp -ngld 99`.
Stop the exact saved PID with `kill -INT "$SERVER_PID"`; do not use a broad
process-name kill on a shared machine.

Example served-suite invocation:

```bash
EVAL_RESULTS_DIR="$PWD/reproduction-results" \
python3 testsuite/evals/run_eval.py \
  --suite math25 --label replication-UD-IQ2_S-effmed \
  --reasoning-effort medium --max-tokens 8192
```

## Deployment and suite matrix

Serve one configuration at a time, wait for `/health`, run only the suites in
its row, stop that server by its recorded PID, and confirm no inference process
remains before loading the next configuration. All GGUF rows use context 8,192,
Q8/Q8 KV cache, flash attention, temperature 1.0, top-p 0.95, and top-k 20.

| Label | Artifact / engine placement | Suites |
|---|---|---|
| `UD-IQ2_XXS-mtpgpu-effmed` | UD-IQ2_XXS, `-ngl 99`, MTP draft `-ngld 99` | GSM8K, HumanEval, MATH-25, HumanEval+ |
| `UD-IQ2_S-effmed` | UD-IQ2_S, `-ngl 99` | GSM8K, HumanEval, MATH-25, HumanEval+, MMLU, needle |
| `UD-Q2_K_XL-ngl61-effmed` | UD-Q2_K_XL, `-ngl 61` | GSM8K, HumanEval |
| `bart-IQ2_S-ngl59-effmed` | bartowski IQ2_S, `-ngl 59` | GSM8K, HumanEval, MATH-25, HumanEval+ |
| `UD-IQ3_XXS-ngl54-effmed` | UD-IQ3_XXS, `-ngl 54` | GSM8K, HumanEval, MATH-25, HumanEval+ |
| `UD-IQ3_S-ngl49-effmed` | UD-IQ3_S, `-ngl 49` | GSM8K, HumanEval |
| `UD-Q4_K_XL-ngl33-effmed` | UD-Q4_K_XL, `-ngl 33` | GSM8K, HumanEval, MATH-25, HumanEval+ |
| `UD-IQ2_S-effxhigh` | UD-IQ2_S, `-ngl 99`, xhigh | MATH-25, HumanEval+ |
| `UD-IQ2_S-effxhigh-8k` | UD-IQ2_S, `-ngl 99`, xhigh, 8K output budget | HumanEval+ |
| `EXL3-2.0bpw-effmed` | EXL3 under ExLlamaV3/TabbyAPI, fully resident | MATH-25, HumanEval+, MMLU, needle |

For the MMLU/needle validation, use context and cache size 36,864; the GGUF
champion used Q8 keys/Q4 values there. The fixed expected denominator for each
suite is encoded in `check_claims.py`. The original Q4 GSM8K/HumanEval per-item
files were not retained, so a rerun creates new records rather than recovering
the historical ones.

The released scripts intentionally do not offer a one-command unattended GPU
campaign. Fit and thermal limits are machine-specific, and generated-code
execution requires deliberate isolation. HumanEval and HumanEval+ execute
model-generated Python; run them inside a disposable container or VM with no
network and no secrets. Expected throughput values and score denominators are
checked by `check_claims.py`; stochastic reruns need not match individual
answers.
