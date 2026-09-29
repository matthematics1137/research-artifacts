# Third-party notices

No model weights, engine source trees, or benchmark questions and answers are
included in this release.

## Engine code in the patch files

- exllamav3 (turboderp-org/exllamav3): MIT, Copyright (c) 2025 Turboderp. The
  files in `replication/patches/exl3_*.patch` and
  `replication/patches/withdrawn/` modify exllamav3 1.4.7 and include context
  lines of its source. Retained notice: `LICENSES/EXLLAMAV3-MIT.txt`.
- llama.cpp (ggml-org/llama.cpp): MIT, Copyright (c) 2023-2026 The ggml
  authors. `replication/patches/llamacpp-v0.4.0-qwen35-mtp-shared-embd.patch` is
  a diff against v0.4.0 that includes context lines of its source and the
  changes of the ggml-org draft pull request #28243 on which the lab's change
  builds. Retained notice: `LICENSES/LLAMACPP-MIT.txt`.

## Engines and runtimes (linked, not redistributed)

- TabbyAPI (theroyallab/tabbyAPI): AGPL-3.0. The configuration files in
  `replication/serving/` and `replication/tools/` are the lab's own settings
  for it; no TabbyAPI code is included.
- exllamav3: MIT (above). llama.cpp: MIT (above).
- PyTorch 2.10.0+cu128 and the NVIDIA CUDA libraries it bundles: their own
  licenses; not included.
- Tectonic (paper build): MIT.

## Model artifacts (linked, not redistributed)

- Qwen3.8-27B and the tested quantizations (unsloth/Qwen3.8-27B-GGUF,
  turboderp/Qwen3.8-27B-exl3, bartowski/Qwen3.8-27B-GGUF): Apache-2.0 per the
  pinned upstream repositories.
- z-lab/Qwen3.8-27B-DFlash2-GGUF: Apache-2.0 per the file's own metadata.

Exact repositories, revisions or branches, filenames, sizes, SHA-256 values, and
license labels are in `environment/model_artifacts.json`. Recheck the upstream
license before downloading, using, or redistributing a weight. The lab-made
quantizations of E32, E41, and E42 are identified there by shard hashes and are
not included.

## Benchmarks (not redistributed)

The per-item records keep only item identifiers, correctness, token counts,
timings, and SHA-256 digests; they contain no question, answer, or model
response text. The evaluation sets are rebuilt from their upstream sources by
`replication/evals/prepare_datasets.py` and `prepare_hardening.py` from GSM8K
(openai/grade-school-math, MIT), MMLU (the cais/mmlu dataset of
hendrycks/test), MATH (the nlile/hendrycks-MATH-benchmark copy of
hendrycks/math, MIT), HumanEval (openai/human-eval, MIT), and HumanEval+
(evalplus/humanevalplus_release, Apache-2.0). Check each upstream license before
redistributing rebuilt sets.

The bibliography cites third-party scientific work for limited propositions.
Citation does not incorporate those papers or their datasets into this bundle.
