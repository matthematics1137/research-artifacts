# Full rerun: P2R1 (Paper 2)

Analysis reproduction needs only `python3 check_claims.py` (see `README.md`).
This file describes the much larger inference path. A new inference run is a
new experiment: it must regenerate its own questions and cannot resume or
reproduce the frozen private attempt byte for byte.

## Tested system

- **Hardware.** NVIDIA GeForce RTX 4080 Laptop GPU with 12,282 MiB VRAM and
  an 80 W power limit, an Intel Core i9-14900HX with 32 logical CPUs, and
  62.4 GiB RAM. The run was bound to one host inventory; see
  `environment/hardware.json`.
- **Driver and CUDA.** NVIDIA driver 580.167.08. The two llama.cpp tiers use a
  private CUDA 12.6 toolkit whose `libcudart`, `libcublas`, and `libcublasLt`
  are pinned by byte count and SHA-256 (`environment/software.json`). The
  EXL3 tier uses the CUDA 12.8 libraries bundled with torch 2.10.0+cu128, and
  it launches with no inherited `LD_*` loader variables.
- **llama.cpp.** Revision `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, built
  with CUDA for architecture 89. The build settings and tool hashes are in
  `environment/software.json`, and the server binary SHA-256 is recorded
  there.
- **ExLlamaV3 and tabbyAPI.** ExLlamaV3 1.4.3 (cu128, torch 2.10.0) served
  through tabbyAPI revision `4a4f9f44820303593844f092d424bb7506008733`. The
  installed-content fingerprints of every package in the Python runtime are
  in `environment/software.json`.

## Model artifacts

These are linked, not redistributed:

| Tier | Repository | Revision | File | Placement |
|---|---|---|---|---|
| higher footprint | `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | llama.cpp `-ngl 33`, loader readback `33/66` |
| low, EXL3 | `turboderp/Qwen3.8-27B-exl3` | `d7e02d63dddd4e047a8d397acb22aeb2b0bedcec` | the SC_2.00bpw_H3 tree | tabbyAPI, full GPU |
| low, GGUF | `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `Qwen3.8-27B-UD-IQ2_S.gguf` | llama.cpp `-ngl 99`, loader readback `65/65` |

Exact byte counts and SHA-256 values are in `environment/model_artifacts.json`.
All three servers use an 8192-token context and a Q8 KV cache.

## Inputs

- **WildChat.** WildChat-1M (`allenai/WildChat-1M`, ODC-By), rebuilt from the
  pinned dataset revision and shard hashes recorded in
  `replication/testsuite/paper2/build_wildchat_r1.py`. The frame is 160
  windows, 40 in each short/long × code/no-code stratum.
- **Private corpus.** The private corpus cannot be released. A rerun uses
  WildChat alone, or another corpus built to the same window rules.
- **Conditions.** A and E come from the committed deterministic transform
  (`replication/testsuite/paper2/ladder.py`). L2 uses
  `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` at revision
  `ebaba9b0e874dadd3003ffcff828e4397e568089` (MIT), on CPU, with its requested
  rate matched per window to E's character retention
  (`replication/testsuite/paper2/make_conditions_r1.py`).

## Procedure

The launcher `replication/testsuite/paper2/p2_grid_r1.sh` runs every stage in
a fixed order:

1. Cool-state admission gate.
2. Q4_K_XL question generation for both corpora.
3. Scoped shutdown, then the cool gate again.
4. Q4_K_XL evaluation: cold relaunch, one unmeasured warmup, then the TSI and
   WildChat grids.
5. The EXL3 and IQ2_S tiers, each with the same launch, warmup, grid, and
   shutdown sequence.
6. Post-campaign full artifact hashing and runtime verification.
7. Promotion by `replication/paper2/scripts/validate_r1.py`.

The claims build is `replication/paper2/scripts/build_claims.py`. The frozen
parameters are in `replication/paper2/R1_PROTOCOL.json`.

Timings on the tested system, for planning only:

| Stage | Duration |
|---|---|
| Question generation | about 9 h |
| Q4_K_XL evaluation (partly offloaded) | about 31 h |
| EXL3 evaluation | about 2–4 h |
| IQ2_S evaluation | about 3–5 h |
| Post-campaign checks and promotion | under 1 h |

## Expected outcomes and tolerances

A rerun generates new questions, and each cell is one sampled decode
(temperature 0.2). The frozen harness requests seed 42 for evaluation.
llama.cpp accepts that field; the pinned TabbyAPI revision `4a4f9f4` has no
per-request seed parameter and ignores it, so the EXL3 seed was recorded but
not applied. Its sampled responses are not reproducible from that seed.
Q4 question generation through llama.cpp uses its own fixed seed/retry
schedule, and source selection and item ordering are separately seeded.
These controls do not guarantee identical responses across reruns. Exact
analysis reproduction uses the archived outputs. The pre-specified inference
is Holm across the 12 CR2/Satterthwaite tests.

The protocol and replication sources are preserved as the pre-run record;
their seed fields describe the requested settings. This post-run disclosure
does not change those frozen files, responses, scores, or analysis.

The published result is a non-detection bounded by pointwise intervals of
about ±13 points. It is not an equivalence claim, and a rerun should be
judged with its own intervals.

## What cannot be rerun here

- **Private arm.** The private arm's windows, questions, and responses.
- **Frozen evidence.** The exact frozen evidence and promotion record. The
  checker reproduces every statistic from the aggregate histograms, so the
  analysis itself is fully reproducible.
