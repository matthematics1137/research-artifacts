# Third-party notices and redistribution boundary

This is a private candidate; author rights review and top-level license selection remain pending. This inventory does not grant commercial-use clearance or override upstream terms.

- LongMemEval reader-template context: MIT, copyright (c) 2024 Di Wu, pinned commit `d6dc8b50a2d9ac0c99485ea28fa5755c62414c34`. Retained license: `replication/experiment/LICENSES/LongMemEval-MIT.txt`.
- ExLlamaV3 1.4.7 patch context: MIT, copyright (c) 2025 Turboderp. Retained license: `replication/experiment/LICENSES/ExLlamaV3-MIT.txt`. Three patch files preserve historical context and lab modifications; the withdrawn quantizer workaround is historical, not a recommended fix or claimed inference optimization.
- LongMemEval and LoCoMo corpora are obtained separately. LoCoMo at commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` carries CC BY-NC 4.0; this candidate exports no LoCoMo text, item/conversation identifiers or per-question scores. Its result projection is aggregate-only. Derivative-result redistribution still needs explicit rights review.
- Model weights, embeddings, tokenizer files, compression models, native libraries, patched runtime and benchmark corpora are not included. Their exact pins and individually recorded terms are in `replication/experiment/ASSETS.json`; installation and use require separate review.
- Original manuscript, figures, lab code and scored-result projection require Matthew Schwartz's license selection and rights attestation. No new prior-art code is incorporated. Primary-source HTML saved privately for the literature review is not included.

The frozen component's detailed source transformations and retained notices are in `replication/experiment/SOURCE_MANIFEST.json` and `THIRD_PARTY_NOTICES.md`. A source snapshot is not evidence of executed behavior.
