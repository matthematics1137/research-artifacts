# Third-party notices and rights inventory

This inventory describes the candidate; human clearance is pending.

| Material | Treatment and source | License/notice |
|---|---|---|
| W7, W9, W10, W10b, W11, W11b, W12p lab patches | Code against exllamav3 1.6.5, original hashes in export inventory | MIT; `LICENSES/EXLLAMAV3-MIT.txt`; lab hunks `MIT.txt` |
| W13 split-K kernel | rafatxf/exllamav3 turing commit 15654d3c6fda2714bd91a3a39c0a0903dcf37d26, ported unchanged; lab mode-2 dispatch is identified in the paper and patch | MIT, Copyright (c) 2025 Turboderp; retained patch header and exact `LICENSES/RAFATXF-MIT.txt` |
| Shared-embedding MTP patch | llama.cpp v0.4.0, based on ggml-org PR28243; includes upstream context | MIT; `LLAMACPP-MIT.txt` |
| LongMemEval reader template in p5common.py | xiaowu0162/LongMemEval, d6dc8b50a2d9ac0c99485ea28fa5755c62414c34 | MIT, Copyright (c) 2024 Di Wu; `LongMemEval-MIT.txt` |
| RULER-style needle/question wording | NVIDIA/RULER; borrowed wording is identified in generator source; no RULER harness copied | Apache-2.0; `Apache-2.0.txt` |
| Frozen synthetic gold/questions and outcome projections | Author's generation/scoring, with upstream wording above retained | Original measurements proposed CC BY 4.0; inherited wording terms preserved |
| LongMemEval-S cleaned chat haystacks | Linked at revision 98d7416c24c778c2fee6e6f3006e7a073259d48f; not redistributed | Dataset card records MIT; hash and rebuild inputs in `FULL_RERUN.md` |
| GSM8K and MMLU | Questions/answers and generated response text omitted; item IDs, outcome flags and timing retained | Official repository texts freshly read: MIT, OpenAI 2021 / Dan Hendrycks 2020, retained in `LICENSES/GSM8K-MIT.txt` and `LICENSES/MMLU-MIT.txt`. These repository reads do not clear all incorporated question material; obtain datasets separately. |
| Models/quantizations/drafters | Linked only, pinned in `environment/model_artifacts.json`; no tensor payloads | Apache-2.0 per existing pinned model metadata; local derivatives retain upstream terms |
| Bonsai counterpoint numeric measurements | WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 at revision 841f7812793972b5de6dc48910bdd1e2722ae48b, NInfer f4ab369a; only lab item IDs/outcomes/tokens/times redistributed | Lab measurements; no model weights, benchmark questions/answers, generated response text or NInfer implementation included. Original model/engine terms remain upstream. |
| Bonsai model and MTP adaptation | WaveCut's NInfer conversion of PrismML's ternary target derived from Alibaba Qwen; ProCreations adapted MTP head at efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51 | Apache-2.0 per pinned WaveCut NOTICE and component cards. The NOTICE also lists ProCreations/z-lab DFlash2 and Qwen vision/tokenizer sources; those paths were not measured by the none/MTP-3 comparison. No tensors redistributed. |
| NInfer | iamwavecut/ninfer-all, measured f4ab369a06c728325b706c989d71be7dc0c05a1b; fresh license read at 40d913db2e87e2211fce4089c0c7920480a5e7a0 | Apache-2.0; exact `LICENSES/NINFER-Apache-2.0.txt`. Separate project from Brontir/NInferno. The current license pin is not a new measured engine version. |
| buun-llama-cpp | spiritbuun/buun-llama-cpp at 1e50db7be8e87d80d660f6ea18e01f2a599ffcba; related earlier int8/Turing work, implementation not incorporated | MIT, Copyright (c) 2023-2026 The ggml authors; exact `LICENSES/BUUN-MIT.txt` |
| Fleet numeric projections | Two registered sm_89 deployment slices, shared 2-bit EXL3 weights and w13 chain; no extra engine or queue implementation copied | Author-owned measurements proposed CC BY 4.0; inherited engine/model terms above. Correctness projections cannot independently rescore missing responses. |
| Engines and dependencies | Linked, not redistributed: exllamav3 MIT; llama.cpp/ik_llama.cpp MIT; TabbyAPI AGPL-3.0; NInfer and NInferno Apache-2.0; PyTorch/CUDA, NumPy, Matplotlib and tokenizers under their own terms | Source/build pins in `environment/software.json` and original run records |

Other engines and prior kernels are cited as earlier work; their implementation
code is not incorporated. `CREDITS.md` supplies the fresh component inventory;
`replication/tools/turing/CREDITS.md` retains the historical credits byte content
apart from declared export normalization. Its old unchecked buun/NInfer licence
note is superseded here by the archived primary texts and
`environment/license-reads.json`, without rewriting historical lab evidence.
The full working lab repository and unrelated history
are excluded. Paths and host identifiers are normalized in export copies, with
before/after hashes. The original patches remain private pending publication;
this local candidate does not authorize an upstream push or public disclosure.
