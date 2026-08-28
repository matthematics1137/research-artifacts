# Third-party notices

No model weights or engine source trees are included in this release.

## Benchmark selections

- GSM8K: OpenAI, MIT License,
  <https://github.com/openai/grade-school-math>.
- MATH: Dan Hendrycks et al., MIT License,
  <https://github.com/hendrycks/math>.
- MMLU: Dan Hendrycks et al., MIT License,
  <https://github.com/hendrycks/test>.
- HumanEval: OpenAI, MIT License,
  <https://github.com/openai/human-eval>.
- HumanEval+: EvalPlus authors, Apache License 2.0,
  <https://github.com/evalplus/evalplus> and
  <https://github.com/evalplus/humanevalplus_release>.
- The needle dataset was generated for this study and contains synthetic filler
  and invented facts.

The release includes the fixed subsets needed to identify and rescore the
reported runs. The source download URLs used by the original selection script
were not all revision-pinned. The released subset files are therefore frozen by
their own SHA-256 values rather than represented as byte-for-byte regeneration
from mutable upstream branches.

## Model artifacts (not redistributed)

- `Qwen/Qwen3.8-27B`, `unsloth/Qwen3.8-27B-GGUF`,
  `bartowski/Qwen3.8-27B-GGUF`, and `turboderp/Qwen3.8-27B-exl3`: Apache-2.0
  according to the tested repositories. Exact revisions and file hashes are in
  `environment/model_artifacts.json`.

## Engines and tools (not redistributed)

- llama.cpp: MIT, <https://github.com/ggml-org/llama.cpp>.
- ExLlamaV3: MIT, <https://github.com/turboderp-org/exllamav3>.
- TabbyAPI: AGPL-3.0, <https://github.com/theroyallab/tabbyAPI>.
- Matplotlib: PSF-based license, <https://matplotlib.org/stable/project/license.html>.
- Tectonic: MIT, <https://github.com/tectonic-typesetting/tectonic>.

The bibliography cites scientific and community sources for the limited
propositions described in the paper; citation does not incorporate those works
into this artifact.
