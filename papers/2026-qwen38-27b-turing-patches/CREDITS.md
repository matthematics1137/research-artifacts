# Component credits and the publication hold

W13 remains the unchanged MIT split-K kernel port from
[rafatxf/exllamav3](https://github.com/rafatxf/exllamav3/tree/15654d3c6fda2714bd91a3a39c0a0903dcf37d26).
The lab's mode-2 dispatch change is separate and identified in the patch and paper.
Original patch headers and upstream copyright notices are retained.

The int8/Turing work cites earlier vcruz305, weicj and
[spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp/tree/1e50db7be8e87d80d660f6ea18e01f2a599ffcba).
The fresh buun repository license is MIT, copyright 2023–2026 The ggml authors;
this is related work, with no new borrowed implementation. Historical credit
notes that called its licence unchecked are superseded by the exact fresh text.

The registered Bonsai counterpoint uses WaveCut's NInfer conversion of PrismML's
ternary Qwen-derived target and the ProCreations MTP adaptation. The pinned model
NOTICE credits those components. NInfer is
[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all/tree/40d913db2e87e2211fce4089c0c7920480a5e7a0),
Apache-2.0, separate from Brontir/NInferno. The experiment used NInfer
f4ab369a06c728325b706c989d71be7dc0c05a1b; the newer pin is a license read only.
The container's DFlash2, vision and tokenizer credits are preserved without
claiming to have measured the unused inference paths.

`THIRD_PARTY_NOTICES.md`, `LICENSES/`, `environment/license-reads.json`, and the
pinned model/component receipts distinguish code ports, related ideas, linked
models, generated outcomes and inherited material. Original code/prose licences
remain proposals for Matthew's clearance. AI roles are in `AI_ASSISTANCE.txt`.

All patches stay private until the reporting paper is published. Preparing this
local candidate grants no authority to upload, submit, push a public fork or
contact an upstream project. No third-party engine checkout, runtime or weight
payload is included.
