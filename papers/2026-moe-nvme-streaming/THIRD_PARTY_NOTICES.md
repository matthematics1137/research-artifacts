# Third-party notices

No model weights or engine source trees are included in this release.

## Benchmark inputs and evaluation code

- MATH dataset and reference code: MIT,
  <https://github.com/hendrycks/math>. Retained notice:
  `LICENSES/MATH-MIT.txt`.
- GSM8K: MIT, <https://github.com/openai/grade-school-math>. Retained notice:
  `LICENSES/GSM8K-MIT.txt`.
- HumanEval: MIT, <https://github.com/openai/human-eval>. Retained notice:
  `LICENSES/HUMANEVAL-MIT.txt`.
- HumanEval+: Apache-2.0 and additionally subject to HumanEval's MIT terms,
  <https://github.com/evalplus/humanevalplus_release>. The Apache-2.0 text is
  `LICENSES/Apache-2.0.txt`; the inherited HumanEval notice is retained above.

The `evaluation/datasets/` directory contains only the frozen benchmark subset
needed to rerun this study. The authored harness is MIT-licensed. Sanitized
historical outcome rows omit benchmark prompts and model response prose; they
retain short MATH answer fields needed to reproduce the disclosed rescore.

## Model artifacts (linked, not redistributed)

- Qwen3.8-Flash-Next and the tested Unsloth GGUF conversion: Qwen Community
  License 1.0 at the pinned upstream repository. This is not Apache-2.0.
- gpt-oss-120b and the tested ggml-org GGUF: Apache-2.0 at the pinned upstream
  repositories.

Exact repositories, revisions, filenames, sizes, hashes, and license labels
are in `environment/model_artifacts.json`. Recheck the pinned upstream license
before downloading, using, or redistributing a weight.

## Engine and build tools (linked, not redistributed)

- llama.cpp: MIT, <https://github.com/ggml-org/llama.cpp>.
- CMake: BSD-3-Clause, <https://cmake.org/licensing/>.
- Tectonic: MIT, <https://github.com/tectonic-typesetting/tectonic>.

The bibliography cites third-party scientific work for limited propositions.
Citation does not incorporate those papers or their datasets into this bundle.
