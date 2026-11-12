# P8 refresh: CPU preparation sources (2026-11-12 UTC / November 11 PST)

This is a delta check for the frozen release experiment, not another unbounded survey. Prior-art record: [synthesis at 349ab98f](https://github.com/matthematics1137/local_llms/blob/349ab98f22939c3815ac83ac4d502dbce63af55b/research/raw/2026-11-12-p8-sota/synthesis.md) and its adjacent searches.md (40 web queries, 89 HTTP receipts, archived FOSCAR PDF). No new paper PDF or model was fetched for this preparation.

## Exact fresh queries and results

- `git ls-remote https://github.com/Brontir/NInferno.git refs/heads/master` -> `65e573fd0c34506478d0f01316da683c47648651`; commit timestamp 2026-11-12 01:06:49 UTC, "Price: the DRAM time from the shared bus and each SM's own warps, no idle-SM charge (#735)".
- `git ls-remote https://github.com/ggml-org/llama.cpp.git refs/heads/master` -> `dd6b945be2188da875ab05a259ce94c6b2736e64`; commit timestamp 2026-11-12 01:01:19 UTC, "server: use the moved test model in the router CORS test (#32117)".
- Shallow, no-checkout public clones of those master branches; verified HEAD equals the resolved pin. Read selected files using `git show HEAD:PATH`; no engine source was executed, compiled or promoted.

Primary paths below were inspected for CLI/source compatibility, with SHA-256 of the fetched file text. Source review validates spellings/options, not binary behavior or old-artifact loading. NInferno parses --version/help before device selection; normal execution selects a device. Passing the existing .ninferno artifact skips the checkpoint conversion path. Explicit --spec none / dflash2 and --draft-tokens are supported. llama.cpp still parses the registered q8, ctxcp/cms and speculation options. No open fit PR is applied.

| Source | Pin / URL | SHA-256 of fetched bytes |
|---|---|---|
| CMakeLists.txt | [pinned source](https://github.com/Brontir/NInferno/blob/65e573fd0c34506478d0f01316da683c47648651/CMakeLists.txt) | `40c99c6968f0a23c9e51a56e3285227c1705ca94cf7827b5b78edb52eb50b6e4` |
| apps/CMakeLists.txt | [pinned source](https://github.com/Brontir/NInferno/blob/65e573fd0c34506478d0f01316da683c47648651/apps/CMakeLists.txt) | `2cd7aa3c5d16b23f266be93232686f3f0cf5eb84759b69db6f0d49bb4c092275` |
| apps/serve/main.cpp | [pinned source](https://github.com/Brontir/NInferno/blob/65e573fd0c34506478d0f01316da683c47648651/apps/serve/main.cpp) | `904d8d5b0a4933be364e481190da8ff656a057287a447a72a2c2f826c5ec1154` |
| src/serve/serve_options.cpp | [pinned source](https://github.com/Brontir/NInferno/blob/65e573fd0c34506478d0f01316da683c47648651/src/serve/serve_options.cpp) | `e09729522021e2d8114f7562f93e597404425a8498ef3c6a4df12ac1853408ef` |
| src/serve/model_source.cpp | [pinned source](https://github.com/Brontir/NInferno/blob/65e573fd0c34506478d0f01316da683c47648651/src/serve/model_source.cpp) | `c84a8b88517b54981200b5a20d3a785d652efab8989319563dae1c2bbc685296` |
| apps/convert/main.cpp | [pinned source](https://github.com/Brontir/NInferno/blob/65e573fd0c34506478d0f01316da683c47648651/apps/convert/main.cpp) | `78c3d9c3c06ec3df1bde816705833da97836b66752f2af6333941bf0540aebc7` |
| LICENSE | [pinned source](https://github.com/Brontir/NInferno/blob/65e573fd0c34506478d0f01316da683c47648651/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| common/arg.cpp | [pinned source](https://github.com/ggml-org/llama.cpp/blob/dd6b945be2188da875ab05a259ce94c6b2736e64/common/arg.cpp) | `51caf9d78f2ddabbc1f0361fcdca105decb19a0daef18272b170debc0015923c` |
| LICENSE | [pinned source](https://github.com/ggml-org/llama.cpp/blob/dd6b945be2188da875ab05a259ce94c6b2736e64/LICENSE) | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` |

## Read-only desktop observations

Through the existing owner-provided SSH route, read tool versions and hashes of code files in the existing w13 overlay only. No model bytes, secrets or credentials were copied, no GPU query was issued, no process/service/queue/file was changed on the desktop.

- CUDA nvcc 13.4.92; gcc/g++-13 13.4.0; lab-venv CMake 4.4.4; Ninja 1.13.2.git.kitware.jobserver-pipe-1; lab Python 3.12.16; system Python 3.14.4. Driver text from /proc/driver/nvidia/version: 580.178.04. No nvidia-smi execution.
- w13: sorted relative .py/.so/.cu/.cuh/.cpp/.h files, excluding __pycache__, SHA-256 of each -> 619 entries in tools/turing/refresh/w13-files.json. This pins the existing overlay as observed now. It does not invent a historical binary hash; the old P8 build/patch records remain historical provenance.
- Local laptop nvcc is 12.8.93, CMake 3.28.3, g++13.3.0. Therefore matching CUDA13.4 engine compilation is a later target-side CPU staging step, not something this worker claims completed.

## Bootstrap and exclusions

Lab confirmation source remains private lab code; reused request/timing/census/port/fit utilities are not republished to engine upstreams. External engine license texts inspected at the pins: NInferno Apache-2.0, llama.cpp MIT. EXL3 and rafatxf's W13 MIT notices remain with the original patches. New tools contain no copied third-party kernel implementation.

Fixed artifact hashes/bytes come from testsuite/results/turing/model_files.json, original plan-df2b.json and the November 4–5 weight audit. No runtime Hub alias, current UD3.1 quant, PR31643 branch, 240 W conditional, new quality lane or kernel experiment is added. CPU-only synthetic checks validate the controls; actual parser/fit/speed behavior remains unmeasured.
