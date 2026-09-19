# Third-party notices

No model weights, dataset text, benchmark questions or answers, or engine
source trees are included in this release.

## Dataset (linked, not redistributed)

- WildChat-1M, Allen Institute for AI, ODC-By 1.0:
  <https://huggingface.co/datasets/allenai/WildChat-1M>. The artifact
  redistributes only study-authored aggregate statistics. It contains no
  conversation text, window, question, answer, response, or identifier. The
  ODC-By database license does not necessarily clear independent rights or
  privacy in individual conversations; see the report's rights section.

## Model artifacts (linked, not redistributed)

- **Qwen3.8-27B.** Apache-2.0 according to the pinned upstream repositories.
  Two conversions were tested:
  - the tested Unsloth GGUF conversions (`unsloth/Qwen3.8-27B-GGUF`);
  - the turboderp EXL3 conversion (`turboderp/Qwen3.8-27B-exl3`).
- **LLMLingua-2** (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`,
  revision `ebaba9b0e874dadd3003ffcff828e4397e568089`): MIT according to its
  model card.

Exact repositories, revisions, filenames, sizes, and SHA-256 identifiers are
in `environment/model_artifacts.json` and `FULL_RERUN.md`. Users must recheck
the upstream terms before redistributing any downloaded weight.

## Engines and tools (linked, not redistributed)

- llama.cpp: MIT, <https://github.com/ggml-org/llama.cpp>.
- ExLlamaV3: MIT, <https://github.com/turboderp-org/exllamav3>.
- tabbyAPI: AGPL-3.0, <https://github.com/theroyallab/tabbyAPI>. Only its
  revision identifier and the study's own configuration template are included.
- LLMLingua (library): MIT, <https://github.com/microsoft/LLMLingua>.
- Tectonic: MIT, <https://github.com/tectonic-typesetting/tectonic>.

The bibliography cites third-party scientific work for limited propositions.
Citation does not incorporate those papers or their datasets into this bundle.
