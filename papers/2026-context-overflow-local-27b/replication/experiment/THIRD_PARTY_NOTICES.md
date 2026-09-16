# Third-party source notices — P5 preparation

This is an unapproved source component, not a blanket rights clearance or a
license for model weights, benchmark data, responses or unreviewed lab files.
No model, dataset content, native library or TabbyAPI source is included.

- LongMemEval: the reader template in `replication/testsuite/paper5/p5common.py`
  and associated evaluation approach refer to upstream commit
  `d6dc8b50a2d9ac0c99485ea28fa5755c62414c34`. Copyright (c) 2024 Di Wu,
  MIT license, retained in `LICENSES/LongMemEval-MIT.txt`.
  [Pinned source license](https://github.com/xiaowu0162/LongMemEval/blob/d6dc8b50a2d9ac0c99485ea28fa5755c62414c34/LICENSE).
- ExLlamaV3: the three patch files retain source context from ExLlamaV3 1.4.7
  and the lab's modifications. Copyright (c) 2025 Turboderp, MIT license,
  retained in `LICENSES/ExLlamaV3-MIT.txt`. The combined generator patch and
  the separate architecture hunk are not two sequential whole-file patches;
  follow the runbook's per-file instructions. The third, under `withdrawn/`,
  is an inherited quantizer workaround retained for historical byte fidelity,
  not recommended as a current bug fix or claimed as an inference optimization.
  The optional helper obtains the wheel separately from its publisher; no
  native library or reconstructed overlay is included in this source component.
- Engines, PyTorch/CUDA, tokenizers, the compressor/retrieval encoder and all
  inference-model artifacts are external dependencies, not redistributed here.
  Their individual licenses and exact runtime versions still need the final
  artifact review; the asset catalog is not a commercial-use determination.
- LongMemEval and LoCoMo datasets are fetched separately. LoCoMo's pinned
  repository carries CC BY-NC 4.0; no LoCoMo content, per-item records or
  conversation identifiers are included in this component. The final P5
  results projection remains aggregate-only for that follow-up.
  [Pinned LoCoMo license](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt).

Path-only export changes are listed in `SOURCE_MANIFEST.json`. Modified export
bytes are not represented as the original executed files. The final release
will separately identify the lab's code/data/manuscript licenses and author
approvals; these third-party notices do not satisfy those decisions.
