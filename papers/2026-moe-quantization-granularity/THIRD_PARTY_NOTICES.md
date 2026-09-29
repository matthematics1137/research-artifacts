# Third-party notices

No model weights, benchmark questions or answers, or engine source trees are
included in this release.

## Benchmark

- GSM8K, OpenAI, MIT License:
  <https://github.com/openai/grade-school-math>, source revision
  `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`. The artifact redistributes only
  study-authored item identifiers and measurement fields, not GSM8K text; the
  replication builder obtains the benchmark from that upstream revision.

## Model artifacts (linked, not redistributed)

- Mixtral-8x7B-Instruct-v0.1 and the tested mradermacher GGUF conversion:
  Apache-2.0 according to the pinned upstream repositories.
- Qwen3-30B-A3B-Instruct-2507 and the tested Unsloth GGUF conversion:
  Apache-2.0 according to the pinned upstream repositories.

Exact repositories, revisions, filenames, sizes, and LFS SHA-256 identifiers
are in `environment/model_artifacts.json`. Users must recheck the upstream
terms before redistributing any downloaded weight.

## Engine and analysis tools (linked, not redistributed)

- llama.cpp: MIT, <https://github.com/ggml-org/llama.cpp>.
- Tectonic: MIT, <https://github.com/tectonic-typesetting/tectonic>.

The bibliography cites third-party scientific work for limited propositions.
Citation does not incorporate those papers or their datasets into this bundle.
