# Full replication: unmodified llama.cpp mmap near the host-memory boundary

This is the expensive path. The analysis-only check in `README.md` does not
launch a model. A full rerun downloads 126.60 GiB of weights, builds one pinned
engine revision, and runs hours-class inference. Use a machine you are allowed
to load heavily, keep unrelated work closed, and monitor temperature, free
space, memory pressure, and swap.

## Historical system and scope

The measurements came from one Linux laptop:

- Intel Core i9-14900HX;
- 62 GiB dual-channel DDR5 RAM and 8 GiB swap (7.3 GiB used in the retained
  baseline snapshot, not necessarily during each run);
- NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB VRAM, 80 W limit;
- 2 TB WD PC SN740 NVMe;
- Linux `7.0.0-29-generic`, NVIDIA driver `580.167.08`, driver-reported CUDA
  `13.0`.

Run logs identify llama.cpp build `b1-035e227`. The associated clean source
checkout is commit `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, compiled
with CUDA for compute capability 8.9. That revision was the Qwen4Exp pull-
request head when measured; it was not a private patch.

The surviving build record identifies CMake `3.28.3` with the Unix Makefiles
generator, GNU C++ `13.3.0`, and CUDA toolkit compiler `12.6.85` (`nvcc` build
`cuda_12.6.r12.6/compiler.35059454_0`). It records a Release build with
`GGML_CUDA=ON`, `GGML_NATIVE=ON`, `GGML_OPENMP=ON`, and `GGML_BLAS=OFF`.
The `13.0` value above is the NVIDIA driver's reported CUDA compatibility
level, not the toolkit used to compile the historical binary. Machine-local
compiler paths are deliberately excluded; the normalized record is also in
`environment/software.json`.

## Build the associated engine

```bash
mkdir paper3-reproduction
cd paper3-reproduction
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DGGML_BLAS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release --parallel
```

For the closest software reproduction, use the historical compiler and toolkit
versions above as well as the explicit options. Record the compiler, CUDA
toolkit, CMake output, and resulting llama.cpp build line for every rerun. A
newer toolchain or engine can test portability but is not an exact software
rerun.

## Download and verify the four files

Install a current Hugging Face CLI in a disposable environment, then download
from the pinned revisions:

```bash
python3 -m venv hf-env
hf-env/bin/pip install huggingface_hub
hf-env/bin/hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf \
  UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf \
  --revision d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249 \
  --local-dir models/flash
hf-env/bin/hf download ggml-org/gpt-oss-120b-GGUF \
  gpt-oss-120b-MXFP4.gguf \
  --revision 238abdd290bb874b90a5da1b4549881b7d05c091 \
  --local-dir models/gptoss
sha256sum models/flash/UD-IQ1_S/*.gguf models/gptoss/gpt-oss-120b-MXFP4.gguf
```

Expected SHA-256 values, in shard order and then gpt-oss order:

```text
88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6
0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a
582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d
```

The corresponding byte counts are 10,946,624; 49,990,818,368;
22,544,696,352; and 63,387,346,208. Stop if any value differs. Licenses and
repository identities are in `environment/model_artifacts.json`.

## Principal generation protocol

Set explicit paths from the reproduction directory:

```bash
LLAMA_CLI="$PWD/llama.cpp/build/bin/llama-cli"
FLASH_MODEL="$PWD/models/flash/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
GPTOSS_MODEL="$PWD/models/gptoss/gpt-oss-120b-MXFP4.gguf"
PROMPT='Write a detailed technical explanation of how an operating system page cache interacts with memory-mapped files during sequential and random access patterns.'
```

For each artifact, run three consecutive invocations and retain the complete
stdout/stderr, timestamps, `/proc/meminfo`, process exit status, and engine
build line:

```bash
for run_id in 1 2 3; do
  "$LLAMA_CLI" -m "$FLASH_MODEL" -ngl 99 --n-cpu-moe 999 \
    -c 4096 -n 256 -p "$PROMPT" -st --no-display-prompt --simple-io \
    >"flash-generation-${run_id}.log" 2>&1
done

for run_id in 1 2 3; do
  "$LLAMA_CLI" -m "$GPTOSS_MODEL" -ngl 99 --n-cpu-moe 999 \
    -c 4096 -n 256 -p "$PROMPT" -st --no-display-prompt --simple-io \
    >"gptoss-generation-${run_id}.log" 2>&1
done
```

The historical run did not set a sampling seed or force greedy decoding. Keep
that fact if reproducing the historical protocol; add a fixed seed only for a
new controlled study and label it as a protocol change.

The later Flash group used the same invocation after `MemAvailable` was at
least 35 GiB and one-minute load was below 3 for ten consecutive checks spaced
30 seconds apart. That gate describes a host state; it does not create a
randomized cache treatment.

If recording physical-device reads, identify the device explicitly and sample
field 6 of its `/proc/diskstats` row immediately before process launch and
after process exit. Multiply the sector difference by 512. Report it only as a
**whole-invocation physical-device read delta**. It includes unrelated host
reads and must not be divided by generated tokens.

Historical acceptance values are descriptive, not pass/fail tolerances:

| Artifact/group | Decode tokens/s | Whole-invocation device-read GiB |
|---|---|---|
| Flash, initial sequence | 9.4 / 10.4 / 10.0 | 104.2 / 102.1 / 103.6 |
| Flash, later quiet gate | 14.7 / 14.4 / 13.9 | 84.3 / 82.1 / 72.0 |
| gpt-oss, initial sequence | 9.3 / 8.7 / 9.2 | 126.2 / 124.1 / 120.8 |

Substantial differences are expected across kernels, storage, memory state,
power limits, engine revisions, and invocation order. The primary replication
criterion is successful completion with complete provenance; do not tune until
a planned baseline has been retained.

## Task-check protocol

The GSM8K-50 subset is the first 50 rows of the archived seed-42 sample of
100 test items, sorted by source index. This is not a direct random sample of
50; retain those exact IDs when reproducing the historical task check. The
archived client can automatically retry a failed request, and historical
result rows do not include attempt counts. For a new experiment, log every
attempt and its timing explicitly instead of describing retained rows as
single-attempt generations.

Copy the artifact's `evaluation/` directory into the reproduction directory.
The harness uses one request at a time, temperature 1.0, top-p 0.95, top-k 20,
reasoning effort `medium`, and the frozen seed-42 item sets. Start exactly one
server at a time:

```bash
LLAMA_SERVER="$PWD/llama.cpp/build/bin/llama-server"
mkdir -p new-results
EVAL_RESULTS_DIR="$PWD/new-results" \
  "$LLAMA_SERVER" -m "$FLASH_MODEL" -ngl 99 --n-cpu-moe 999 \
  -c 8192 --jinja --port 8090
```

In a second terminal, from the same directory:

```bash
EVAL_RESULTS_DIR="$PWD/new-results" python3 evaluation/run_eval.py \
  --suite math25 --label FLASH-IQ1S-rerun --reasoning-effort medium --max-tokens 8192
EVAL_RESULTS_DIR="$PWD/new-results" python3 evaluation/run_eval.py \
  --suite humaneval_plus --label FLASH-IQ1S-rerun --reasoning-effort medium --max-tokens 4096
EVAL_RESULTS_DIR="$PWD/new-results" python3 evaluation/run_eval.py \
  --suite gsm8k --limit 50 --label FLASH-IQ1S-rerun --reasoning-effort medium --max-tokens 4096
```

Stop that server cleanly, start the gpt-oss model with the same server flags,
and repeat with label `GPTOSS-MXFP4-rerun`. The saved historical Flash
HumanEval+ file is incomplete (17 of 20 rows), so its aggregate is not an
acceptance target. Run generated code only inside a network-disabled,
secret-free container or VM.

Apply the separately reported, post-hoc lenient MATH rule to the two new result
files with the safe release rescorer (it accepts only explicit numeric and
fraction grammar; it does not call `eval`):

```bash
python3 evaluation/rescore_math25.py \
  new-results/FLASH-IQ1S-rerun/math25.jsonl \
  new-results/GPTOSS-MXFP4-rerun/math25.jsonl
```

The historical outputs were 18/25 original and 23/25 lenient for Flash, and
12/25 original and 20/25 lenient for gpt-oss. New stochastic results can
differ. Report the original and lenient scores as separate scoring contracts;
do not merge them.

## Analysis and interpretation

Run `python3 check_claims.py` on the published bundle to verify the historical
evidence. A new stochastic rerun need not reproduce identical item outcomes.
Compare new work with confidence intervals and complete run-level records, not
only aggregate percentages.

This protocol can reproduce the deployment configuration. It cannot turn the
original observational sequence into a causal cache experiment. That would
require randomized repeated host states, consistent before/after residency
metrics, phase- or process-scoped I/O, fixed decoding, and preferably router or
tensor-access instrumentation.
