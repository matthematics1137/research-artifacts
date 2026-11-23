# P8 SOTA refresh: search and source archive

Research date: **2026-11-11 America/Los_Angeles / 2026-11-12 UTC**. Requested interval: changes since October 31 through November 11, with UTC repository queries extending into November 12 while it is still November 11 PST. Source cutoffs are recorded literally below; this is a snapshot, not a claim to have read future commits or every fork. Assignment `assign_c93883177a333673557f2fb3245149ae`; conclusions and per-claim verdicts: [synthesis.md](synthesis.md).

## Method and boundaries

Read the paper before searching. Initial working hypotheses were: upstream could remove the old Turing dispatch disadvantage; a competitor could erase the small undrafted margin; a renamed/replaced quant could make old quality results look current; and a new kernel-cost paper could narrow the novelty statement. These hypotheses guide this release survey; no new experiment line or GPU work was launched. Existing results retain their engine, hash, task, sampler and resource coordinates under `research/LANDSCAPE.md`.

Primary evidence: public GitHub commit/PR/source records, Hugging Face metadata/history/model cards, author benchmark files and arXiv paper text. Search snippets only discover or screen candidates; they do not verify technical claims. Search engines may ignore date restrictions; actual event/commit dates take precedence over “crawled yesterday.” Web calls returned pooled results for each four-query batch, so the inspected-hit list is per batch rather than a fabricated mapping from each result to one query. Repository/API queries are separately archived with exact URLs and response digests below.

No GPU, desktop access, model download, upstream interaction or lab patch publication. Only public source/metadata and one paper PDF were fetched. Model API `blobs=true` returns file metadata/LFS digests, not weights. No tensor payload was inspected; changed file hashes do not by themselves distinguish new weights from metadata changes. Local temporary response files are under `/tmp/p8-sota-c938`; the durable archive is this query/URL/hash record plus the pinned public sources. No credentials were used for these HTTP requests.

The existing related-work note `research/TURING_RELATED_WORK_2026-10-31.md` and P8's November 5 audit provide older baseline context, not proof that current sources are unchanged. Histories were re-read. Source files in the requested paper snapshot `379caf7` match the inspected working source for main, brief and claims (empty diff). Root `AGENTS.md`, `NORTH_STAR.md`, `research/RESEARCH_GATE.md`, `research/LANDSCAPE.md` and the findings mirror/backlog header were read. Worker governance prohibits RIFE writes; proposed entries are in the synthesis for the ambassador.

## Every web search query

All queries below were executed on November 12 UTC / November 11 PST. Unless a row specifies a domain filter, none was used. `after:` and `before:` are literal query text, not separately enforced publication-date guarantees.

| Batch | # | Exact query | Domain filter |
|---|---|---|---|
| A | 1 | `exllamav3 Turing sm_75 November 2026 1.6.7 split-K` | none |
| A | 2 | `rafatxf exllamav3 turing November 2026` | none |
| A | 3 | `"Qwen3.8-27B" "2080" November 2026` | none |
| A | 4 | `"Qwen3.8" "IQ2" November 2026` | none |
| B | 5 | `site:github.com/turboderp-org/exllamav3 "Nov" "2026" "Turing"` | none |
| B | 6 | `site:github.com/ggml-org/llama.cpp "Turing" "November" "2026"` | none |
| B | 7 | `site:github.com/ikawrakow/ik_llama.cpp "Nov" "2026" "IQ2"` | none |
| B | 8 | `site:github.com/Brontir/NInferno "2026-11" "2080"` | none |
| C | 9 | `"Qwen3.8-27B" "2080 Ti" after:2026-10-31` | none |
| C | 10 | `"2-bit" "Turing" kernel after:2026-10-31` | none |
| C | 11 | `"vLLM" "Turing" November 2026 support` | none |
| C | 12 | `"SGLang" "sm75" November 2026` | none |
| D | 13 | `"2 bit" kernel "fixed" "call" November 2026 GPU` | none |
| D | 14 | `"2-bit" "launch" "Turing" after:2026-10-31` | none |
| D | 15 | `"EXL3" "8-bit" "Turing" after:2026-10-31` | none |
| D | 16 | `"NInferno" "2080" November 2026 benchmark` | none |
| E | 17 | `"exllamav3" "fixed" "2-bit" kernel after:2026-10-31` | none |
| E | 18 | `"Turing" "2-bit" GEMV November 2026 paper` | none |
| E | 19 | `"2080 Ti" "27B" "tok/s" after:2026-10-31 before:2026-11-12` | none |
| E | 20 | `"SGLang" "Turing" support sm75` | github.com, docs.sglang.io |
| F | 21 | `"Qwen3.8-27B" "EXL3" "11" after:2026-10-31` | none |
| F | 22 | `"Qwen3.8-27B" "2-bit" "Turing" after:2026-10-31` | none |
| F | 23 | `"Turing" "fixed cost" quantization after:2026-10-31` | none |
| F | 24 | `"2080 Ti" "2.0bpw" after:2026-10-31` | none |
| G | 25 | `"2-bit" "kernel" "Turing" after:2026-10-31 before:2026-11-12` | arxiv.org, github.com |
| G | 26 | `"quantized" "fixed cost per call" GPU` | arxiv.org, github.com |
| G | 27 | `"EXL3" "fd75d"` | github.com, huggingface.co, arxiv.org |
| G | 28 | `"exllamav3" "8-bit cache" "Turing"` | github.com |
| H | 29 | `Qwen3.8 27B 2080Ti 11G 2bit 2026年11月` | none |
| H | 30 | `"Qwen3.8-27B" "2080" after:2026-11-01 before:2026-11-12` | none |
| H | 31 | `"2-bit" "Turing" "kernel" November 2026` | none |
| H | 32 | `"exllamav3" "split-K" after:2026-10-31` | none |
| I | 33 | `"exllamav3" "Turing" after:2026-10-31 before:2026-11-12 "attention"` | none |
| I | 34 | `"exllamav3" "Turing" after:2026-10-31 before:2026-11-12 "int8"` | none |
| I | 35 | `"kyojin" "Turing" "8-bit"` | none |
| I | 36 | `"Qwen3.8-27B" "11GB" after:2026-10-31 before:2026-11-12` | none |
| J | 37 | `site:arxiv.org 2611 2-bit inference kernel overhead GPU Turing` | none |
| J | 38 | `site:arxiv.org "fixed overhead" quantization November 2026` | none |
| J | 39 | `"Turing" "GEMV" "November" "2026"` | none |
| J | 40 | `"EXL3" "2-bit" "fixed"` | none |

### Hits inspected and follow-up sources

The table distinguishes source text read from candidates screened only through metadata/snippets. Repeated hits are consolidated. Unrelated generic results were discarded; no claim derives from them.

| Batches | Hits inspected / source | Read depth, dates and disposition |
|---|---|---|
| A–B | [EXL3 releases](https://github.com/turboderp-org/exllamav3/releases), [rafatxf branch](https://github.com/rafatxf/exllamav3/tree/turing), [NInferno performance](https://github.com/Brontir/NInferno/tree/master/docs/performance) | Followed with primary API histories and pinned source; newest upstream release October 31 1.6.6; unchanged raf branch; new NInferno November 8 rows. |
| A–C | [Unsloth model](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8), [doth4580 2.0 bpw](https://huggingface.co/doth4580/Qwen3.8-27B-exl3-2.0bpw) | Read publisher guide/cards and metadata/history. UD3.1 replacements November 10–11; new EXL3 card November 9 on GB10, not Turing. |
| B–C | [llama.cpp PR 31643](https://github.com/ggml-org/llama.cpp/pull/31643), [PR 31506](https://github.com/ggml-org/llama.cpp/pull/31506), [PR 31927](https://github.com/ggml-org/llama.cpp/pull/31927), [PR 31679](https://github.com/ggml-org/llama.cpp/pull/31679), [PR 31247](https://github.com/ggml-org/llama.cpp/pull/31247) | Read primary bodies/history; 31643 still open; 31506 November 3 already in P8; 31927 November 9 MTP device-buffer prefill change; architecture dispatch fix and Q8_1 numerical fix have narrower scope than an IQ2 performance reversal. |
| B–C | [ik PR 2812](https://github.com/ikawrakow/ik_llama.cpp/pull/2812), [PR 2821](https://github.com/ikawrakow/ik_llama.cpp/pull/2821), [current CUDA source](https://github.com/ikawrakow/ik_llama.cpp/blob/ceee8f888400/ggml/src/ggml-cuda.cu) | Read primary PRs and code. November 2 trellis additions/fix already earlier than November 4 baseline; below-Ampere CUDA graphs still disabled. Bench is MoE on two 3090s. |
| C–E | [weicj vLLM-2080Ti](https://github.com/weicj/vLLM-2080Ti-Definitive), [cookjohn vllm-turing](https://github.com/cookjohn/vllm-turing), [SGLang issue 38515](https://github.com/sgl-project/sglang/issues/38515) | Read available primary descriptions and current metadata. Existing support predates interval; dual modified 22 GB NVFP4 reports do not meet target. SGLang packaging issue is not a blanket engine capability determination. |
| C–F | [Escha W2 GGUF](https://huggingface.co/aj9o9/Qwen3.8-27B-Escha-W2-GGUF), [Sybil registry](https://github.com/sybil-solutions/local-ai-registry) | Read W2 card, registry search results and later metadata. W2 last changed September 11; custom weights/native-op fork, measured 3090, no stock Turing result. Registry was already in P8; no new matching record located. |
| D–G | [FOSCAR](https://arxiv.org/abs/2611.09722), [project-maya issue 23](https://github.com/mw00/project-maya/issues/23), [RK3588 GEMV note](https://github.com/gregordinary/rockchip-npu-notes/blob/main/perf/decode-gemv.md), [Volta bakeoff](https://github.com/christopherrobertbrooks-tech/volta-kernel-bakeoff) | Read FOSCAR abstract/PDF §§2–6; read issue body and note/README. FOSCAR November 6 is new adjacent H100/B200 KV work. Maya opened October 8, Volta archived October 23; RK3588 is another device. No exact new duplicate established. |
| E–G | [NInferno decisions](https://github.com/Brontir/NInferno/blob/72fded0c659ad1f3d9b24b4b56eb9e4c6886092b/docs/maintainer/decisions.md) and kernel roofline/commit records | Read relevant source and decision sections: October 28 fixed speculative pass; November 8 tile law; November 11 Ada launch-node constant and Turing Q4/Q6 MMQ launch law. Partial conceptual overlap, not the same 2-bit EXL3 fit. |
| F–H | [57 tok/s mixed-GPU post](https://www.reddit.com/r/LocalLLM/comments/1xpxm2s/qwen38_27b_131k_context_57_ts_decode_guess_the/), [dual 2080Ti blog](https://ayuan.ink/blog/qwen38-27b-local-test/), [dual-2080Ti runtime](https://github.com/Ray-D-Song/qwen-dual-2080ti/) | Screened hardware descriptions: modified 22 GB or mixed/multiple GPUs. Not a target-coordinate counterexample; no technical conclusion rests on Reddit snippets. |
| H | [RustingStrata author post](https://www.reddit.com/r/Qwen_AI/comments/1xoditt/rust_inference_engine_for_qwen_models_on_a_12_gb/) | Opened/read author post: supports dense 27B but cited speed comparison is Flash-Next or other MoEs on RTX3060 12 GB. Not 27B dense on Turing. Search-engine publication labels were inconsistent with the page; no precise new-event date inferred. |
| H | [AMD Paiton update](https://www.reddit.com/r/Qwen_AI/comments/1xraaey/while_everyone_waits_for_qwen_4_our_final_qwen38/), [Gosset 16 GB post](https://www.reddit.com/r/LocalLLM/comments/1xrug4p/qwen3827b_in_1206_gib_on_a_16_gb_card_benchmarks/) | Screened snippets: different hardware / >12 GB; not target speed evidence. Generic requirement calculators and automated benchmark predictions were not treated as measurements. |
| I | [weicj EXL3 fork](https://github.com/weicj/exllamav3-turing), [vcruz port](https://github.com/vcruz305/exl3-turing), [kyojin](https://github.com/Yamz-Labs/kyojin) | Read current metadata, relevant source/docs and histories. Last pushes September 11/September 2 for first two; kyojin 118 interval commits, pinned qdec describes sm80+. |
| I | [batiai Ollama card](https://ollama.com/batiai/qwen3.8-27b), FieldMouse tags and generic quant guides | Screened model/hardware: M4 Max/Metal or generic size advice, not Turing speed. Could not establish a new matching result. |
| J | [Strata PR mirror 3339](https://www.stratat.com/en/pulls/3339), [T16 CPU format note](https://github.com/lxe/dsx/blob/main/docs/t16.md), older arXiv 2311.16442, 2604.02344, 2606.28565 | Screened dates/abstracts: October 20 Ada-laptop worker comparison, October 24 CPU-format state, and older low-bit/dispatch/simulation papers. No new matching Turing per-call result. No PDF fetched for these. |

Direct GitHub search, after the web batches, found more precise support work: vLLM [65003](https://github.com/vllm-project/vllm/pull/65003) (November 1, open, IQ2 Triton FMA sm75 cross-compile only); [66289](https://github.com/vllm-project/vllm/pull/66289) (merged November 9, Qwen norm startup fix on T4); [55380](https://github.com/vllm-project/vllm/pull/55380) (open, sm75 FlashInfer selector, rebased November 10); [56660](https://github.com/vllm-project/vllm/pull/56660) (open, ModelOpt FP8 gate). SGLang [47690](https://github.com/sgl-project/sglang/pull/47690) is open W4A16 MoE on Tesla T10; [38513](https://github.com/sgl-project/sglang/issues/38513) closed November 9. Primary bodies were read; status/merge fields came from the fresh search API responses, not inferred from titles. No one in these records reports this dense 27B at 2 bits on a stock 11/12 GB Turing card.

## Repository/API search coverage

- EXL3 master: one interval commit; dev: 28; full diff `bd1f51917240...4db6a855c1f1` and current int8 gate inspected. Releases latest 1.6.6. Seven updated issues/PRs matched the Turing/sm75/split-K query; inspected titles/bodies included shared-KV assistant and concurrency lock-buffer work, not an sm75 q8 decode replacement.
- rafatxf: six branch heads, no interval `turing` commits; current turing digest equals P8's. Current Turing document read.
- llama.cpp: all 28 commits under `ggml/src/ggml-cuda` and five under `common/speculative.cpp` within the date filter; selected PR bodies read. Heads separately queried to distinguish path history from repository head.
- ik: all 18 CUDA-path interval commits, selected trellis PRs and graph guard source read. This path-filtered survey does not claim every non-CUDA engine change is performance-neutral.
- buun: default branch is master, corrected after initial main 404; 22 interval commits, read titles and relevant scope. No dense target speed report located.
- NInferno: nine pages, 100+100+100+100+100+100+100+100+29 = 829 commits; no pagination silently dropped. Titles were filtered for Turing/2-bit/attention/speculation; selected patches/notes read rather than claiming line-by-line audit of all 829 diffs. Read complete untruncated tree, pinned performance row/card/methodology/raw bench data, release note, relevant decisions and kernel-cost model. November 8 raw row carries commit `ddb36ce8ad5b5c3774ca9bb970c43e05b080ad42`; November 11 snapshot `72fded0c659ad1f3d9b24b4b56eb9e4c6886092b`.
- vLLM/SGLang: five recent releases each; fresh updated-issue/PR queries returned 37 and six hits, no next page. Relevant bodies and merge fields inspected as above. Existing Turing support is not novel, and no whole-project “unsupported” conclusion was used.
- Kyojin: two pages totaling 118 commits, complete tree, source and README. TurboMind's `src/turbomind` date-filtered commit response was empty; this is a path-specific observation, not proof that all LMDeploy behavior is unchanged.
- Hugging Face: initial 100-model search expanded to 1000 sorted by `lastModified`; oldest returned update October 24 15:56 UTC, spanning the requested interval for the literal `Qwen3.8-27B` name search. Not all naming aliases/private/unlisted models are covered. Reviewed low-bit candidate metadata/cards, old/current Unsloth commits and LFS digests, and the EXL3 revision metadata. Named candidates and hashes are in the synthesis. No benchmark was inferred merely from an uploaded model name.

No absolute absence claim follows from these checks. X, Zhihu and Bilibili were only reachable insofar as indexed web results exposed them; Discord/private communities and video were not surveyed. The search finds a material risk warranting a short matched comparison; it does not establish an external winner.

## Fetched paper PDF

Only one PDF was fetched. Primary source: [FOSCAR v1](https://arxiv.org/abs/2611.09722v1), submitted November 6, 2026 17:00:25 UTC. Title: *FOSCAR: Accelerating Decode Attention over 2- and 4-Bit KV Caches through Affine Folding*; authors Donglin Zhuang, Pragaash Ponnusamy, Xiaoxia Wu, Shuaiwen Leon Song and Zhongzhu Zhou.

- URL: [arXiv PDF v1](https://arxiv.org/pdf/2611.09722v1).
- Retrieved November 12 UTC / November 11 PST; HTTP 200; 1,861,027 bytes.
- Ignored archive path: `engines/archive/papers/2611.09722.pdf`.
- SHA-256: `1f5e99449985e8f403ab03e674700b6b50e596e3e0101ae9e73067ec0cafb735`.
- Verified with `sha256sum`; `git check-ignore` confirms ignored. PDF is not committed. Text extracted to temporary scratch for reading only. No copies placed under paper8 or the research note directory.

Read the abstract and sections on cost, affine folding, operand construction, Hopper/Blackwell pipelines, integration and evaluation. Relevant scope: 2/4-bit KV, H100 80 GB and B200 180 GB; no Turing/EXL3 weight-kernel per-call fit. New adjacent prior art, not the same target or measurement.

## Local evidence read

`paper8/main.tex`, `brief.tex`, `derived/claims.tex`, `derived/model_files.tex`, `references.bib`; `paper8/scripts/build_claims.py` source descriptions; `research/P8_AUDIT_2026-11-05.md`; `research/TURING_RELATED_WORK_2026-10-31.md`; `research/prereg/2026-11-04-p8-confirmation-speed.md` including amendments 1–3; `research/prereg/2026-11-05-p9-phase1.md`; `tools/turing/confirm/{plan.json,plan-fit.json,plan-df2.json,plan-df2b.json,build_confirm.sh,run_confirm.py,run_confirm_fit.py,run_confirm_df2.py,run_confirm_df2b.py}`; quality client/launcher headers; F2/depth rows, quality run summaries/comparisons, confirmation and long-context journals, P9 report and bench records. Read-only; no script that launches GPU work was executed, including a supposed preflight that might touch the GPU.

## Exact public retrieval receipts

The table below preserves every scripted HTTP query/fetch, including failed requests and corrected paths. Times are UTC; response hashes identify fetched bytes. For compactness, HTTP errors preserve the error string instead of an absent digest. GitHub/HF `main` URLs in a receipt describe the actual lookup; the synthesis records the returned immutable revisions for conclusions. The PDF receipt has the date above because its download record did not contain a sub-day timestamp. Public API calls used no token. The anonymous origin lookup was only a private-destination check; a 404 alone is not proof of privacy, so the repository's explicit private-lab statement and existing authorized origin were also checked before the assigned branch push.

| Retrieval | UTC timestamp | Exact URL | HTTP / bytes | SHA-256 or failure |
|---|---|---|---|---|
| exl-master | 2026-11-12T00:51:28.074135+00:00 | [exl-master](https://api.github.com/repos/turboderp-org/exllamav3/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100) | 200 / 3342 | `4bc21e1a02a2a41c5cfd24acd7686fe43e19c8a2716aa776286e248b63cc8d7b` |
| exl-dev | 2026-11-12T00:51:28.074917+00:00 | [exl-dev](https://api.github.com/repos/turboderp-org/exllamav3/commits?sha=dev&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100) | 200 / 100017 | `61ddb16e6fb43c8f2b1ead7e131f4533ef051e0d81c6df8fe760bd7de173be34` |
| raf-turing | 2026-11-12T00:51:28.075355+00:00 | [raf-turing](https://api.github.com/repos/rafatxf/exllamav3/commits?sha=turing&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100) | 200 / 2 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| nif | 2026-11-12T00:51:28.075773+00:00 | [nif](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100) | 200 / 625467 | `b60d3eb2c1faaf22ba9496abeb9a28813702db74e40380c765f43dfd5da68133` |
| llama-cuda | 2026-11-12T00:51:28.076199+00:00 | [llama-cuda](https://api.github.com/repos/ggml-org/llama.cpp/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&path=ggml%2Fsrc%2Fggml-cuda) | 200 / 163769 | `acd022b13b62329eb94fe98e7fdd335e3e446398424e446b7f127a36f2a31244` |
| llama-spec | 2026-11-12T00:51:28.076690+00:00 | [llama-spec](https://api.github.com/repos/ggml-org/llama.cpp/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&path=common%2Fspeculative.cpp) | 200 / 29905 | `d860807f17eaa0876e9eb08e7ecc00d692281ca881481a0e4dd36ddbbf75122e` |
| ik-cuda | 2026-11-12T00:51:28.391952+00:00 | [ik-cuda](https://api.github.com/repos/ikawrakow/ik_llama.cpp/commits?sha=main&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&path=ggml%2Fsrc%2Fggml-cuda) | 200 / 90013 | `6811e2d2b53c04649d47d98c06a82884f9f1a85d8e6bfe9e5065b819671cf290` |
| buun | 2026-11-12T00:51:28.459892+00:00 | [buun](https://api.github.com/repos/spiritbuun/buun-llama-cpp/commits?sha=main&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100) | failed / — | `HTTP Error 404: Not Found` |
| exl-releases | 2026-11-12T00:51:28.508883+00:00 | [exl-releases](https://api.github.com/repos/turboderp-org/exllamav3/releases?per_page=20) | 200 / 2106325 | `19c679bf6a09bb6e60dace959749d4101c612034e5f7966878ed6ee84cb06d34` |
| raf-branches | 2026-11-12T00:51:28.565389+00:00 | [raf-branches](https://api.github.com/repos/rafatxf/exllamav3/branches?per_page=100) | 200 / 1242 | `da731fb578fca2dca94aa48cde5296cfad2da203595172a6d9b33ffa835385d4` |
| nif-releases | 2026-11-12T00:51:28.574827+00:00 | [nif-releases](https://api.github.com/repos/Brontir/NInferno/releases?per_page=20) | 200 / 2 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| origin-public-check | 2026-11-12T00:51:28.672008+00:00 | [origin-public-check](https://api.github.com/repos/matthematics1137/local_llms) | failed / — | `HTTP Error 404: Not Found` |
| nif-p2 | 2026-11-12T00:52:11.080127+00:00 | [nif-p2](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=2) | 200 / 867805 | `9441fa72f18cc43b8049c0f5a37f299958aa82b7c7e5d2cc4a4ad7c5778ce741` |
| nif-p3 | 2026-11-12T00:52:11.080892+00:00 | [nif-p3](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=3) | 200 / 522950 | `b9cf58eead2d5c5cb43723701a05683cfe16820db3fc681647520b57c4c20c0b` |
| nif-p4 | 2026-11-12T00:52:11.081335+00:00 | [nif-p4](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=4) | 200 / 396594 | `5420d5888f5228e16a88dd632c07bf089261494316e40cee57330682a3ab597a` |
| nif-p5 | 2026-11-12T00:52:11.081964+00:00 | [nif-p5](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=5) | 200 / 401544 | `1d142f6f55862af3b93a36bf1d7a45c93022e8b720c6cb5f554c96b8ab0c5932` |
| nif-p6 | 2026-11-12T00:52:11.082432+00:00 | [nif-p6](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=6) | 200 / 400281 | `1da1a134d09a9d3d44d20541c70a00d18a948803761146311f727a71f967c4c2` |
| nif-p7 | 2026-11-12T00:52:11.082875+00:00 | [nif-p7](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=7) | 200 / 404953 | `cb52fb9180830dfa0ecc97e3d7cdc056b33c9eb76e9b384a8cffe720c682f21b` |
| nif-p8 | 2026-11-12T00:52:11.905930+00:00 | [nif-p8](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=8) | 200 / 383157 | `f2e32ed75eba58822372d26b8d75581af717550ce3a92e3e5f2d48fa6f58e3ff` |
| nif-p9 | 2026-11-12T00:52:11.932978+00:00 | [nif-p9](https://api.github.com/repos/Brontir/NInferno/commits?sha=master&since=2026-10-31T00%3A00%3A00Z&until=2026-11-12T00%3A50%3A00Z&per_page=100&page=9) | 200 / 111777 | `8d9aec1d0d80c2831a45c689820d59366f198f004001c9d0cc5521367a9b9cb5` |
| exl-compare | 2026-11-12T00:52:12.000910+00:00 | [exl-compare](https://api.github.com/repos/turboderp-org/exllamav3/compare/bd1f51917240...4db6a855c1f1) | 200 / 411962 | `3cceebbd4706df01b50cdfae11b1d8d621b5273e0feb3e9a685ec0bf5742a977` |
| llama31643 | 2026-11-12T00:52:12.031893+00:00 | [llama31643](https://api.github.com/repos/ggml-org/llama.cpp/pulls/31643) | 200 / 18812 | `eaedff746833b217c25f7366868a5ddf78d7927ea1ab45feada888c90993aaae` |
| ik2812 | 2026-11-12T00:52:12.044046+00:00 | [ik2812](https://api.github.com/repos/ikawrakow/ik_llama.cpp/pulls/2812) | 200 / 20410 | `d7cfc431fb5514619b3b385c37220b30724091725e01c78e1de4e66cd4b580b3` |
| ik2821 | 2026-11-12T00:52:12.260718+00:00 | [ik2821](https://api.github.com/repos/ikawrakow/ik_llama.cpp/pulls/2821) | 200 / 17391 | `6e10e1936faa3e8a98ace435d139aaae5dabafc9dcd6952784a2a78f6c1ea1a2` |
| nif-tree | 2026-11-12T00:52:12.321824+00:00 | [nif-tree](https://api.github.com/repos/Brontir/NInferno/git/trees/master?recursive=1) | 200 / 874719 | `256768299b436ba487fdec136908ba23a205a0eefaaf85d6f5ef0dd048b24c78` |
| buun-meta | 2026-11-12T00:52:12.326487+00:00 | [buun-meta](https://api.github.com/repos/spiritbuun/buun-llama-cpp) | 200 / 5533 | `2114f122c8a3ec5993c9858be4972a74473e56c093336c6b22cd14f0e8cc6c36` |
| vllm-releases | 2026-11-12T00:52:12.376704+00:00 | [vllm-releases](https://api.github.com/repos/vllm-project/vllm/releases?per_page=5) | 200 / 283314 | `c45c1d7ae60c84ae1dc16f322e882203c27e8fadb66918e18432d5aee7b9b557` |
| sglang-releases | 2026-11-12T00:52:12.524480+00:00 | [sglang-releases](https://api.github.com/repos/sgl-project/sglang/releases?per_page=5) | 200 / 450902 | `46d6098a95c9912811e31a205f0454d731adc9a5a5272d9f2c16796a82e28e30` |
| weicj-meta | 2026-11-12T00:52:12.575866+00:00 | [weicj-meta](https://api.github.com/repos/weicj/vLLM-2080Ti-Definitive) | 200 / 5804 | `4e8a3864076edc54198ff3793462652f1ead42429ff1a9f21c9539f267ab9a79` |
| cookjohn-meta | 2026-11-12T00:52:12.586644+00:00 | [cookjohn-meta](https://api.github.com/repos/cookjohn/vllm-turing) | 200 / 5258 | `74530550f2dd4e50c70681adcc57d4485a76cf3b05a80abac3b298f9d8c1efbc` |
| nif-2080 | 2026-11-12T00:53:00.067648+00:00 | [nif-2080](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/docs/performance/runs/rtx2080ti/qwen3.8-27b.md) | 200 / 2345 | `d9c37ed48b9527f08edc4f517911897989e213e12778d2e079e988ef59e1482f` |
| nif-gpu | 2026-11-12T00:53:00.068395+00:00 | [nif-gpu](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/docs/performance/gpus/rtx2080ti.md) | 200 / 1743 | `1929652b15c65940850fb305a106748a1260eb16908353bf78dc34f7466ce0d4` |
| nif-model | 2026-11-12T00:53:00.069002+00:00 | [nif-model](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/docs/performance/models/qwen3.8-27b.md) | 200 / 4128 | `a346636545448e1241156a6a45971a5397ea919416e066c3e65425b661904dbc` |
| nif-notes | 2026-11-12T00:53:00.069378+00:00 | [nif-notes](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/docs/release-notes/v0.1.0.md) | 200 / 28336 | `176ce22178e7a734ed9edbd3e9d775a77554659f67fb3271d3c6d5527f889b52` |
| nif-decisions | 2026-11-12T00:53:00.069831+00:00 | [nif-decisions](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/docs/maintainer/decisions.md) | 200 / 159536 | `42bb032d6f31c1258ad6c1acc5807e2a71fbe20486dd041c08c564fe3ac1a9bb` |
| nif-proposals | 2026-11-12T00:53:00.070141+00:00 | [nif-proposals](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/docs/maintainer/kernel-proposals.md) | 200 / 8272 | `aaeee8fa25578ab6f9a95ac3e16319c770a45522fe2b8c44e94e4fedb856d68e` |
| nif-roofline | 2026-11-12T00:53:00.377360+00:00 | [nif-roofline](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/src/ops/generic/kernel_roofline.h) | 200 / 15579 | `53fe53edc8933f94050ae9b4030fa6b8388bdbda52fb3bc64f625f045cebc57a` |
| exl-int8 | 2026-11-12T00:53:00.389429+00:00 | [exl-int8](https://raw.githubusercontent.com/turboderp-org/exllamav3/4db6a855c1f1/exllamav3/exllamav3_ext/exl3_gemv_int8.cu) | failed / — | `HTTP Error 404: Not Found` |
| raf-doc | 2026-11-12T00:53:00.501709+00:00 | [raf-doc](https://raw.githubusercontent.com/rafatxf/exllamav3/15654d3c6fda2714bd91a3a39c0a0903dcf37d26/doc/turing.md) | 200 / 31879 | `ba6023c5d6ebce9139f3dd30d561b35975b8188e075cb5172a3aa3972da242ff` |
| buun-commits | 2026-11-12T00:53:00.504543+00:00 | [buun-commits](https://api.github.com/repos/spiritbuun/buun-llama-cpp/commits?sha=master&since=2026-10-31T00:00:00Z&until=2026-11-12T00:50:00Z&per_page=100) | 200 / 84622 | `63dfac6318ab378a2643345043f9b187f894374aa8e647ca1bceb03f4788894d` |
| hf-models | 2026-11-12T00:53:00.519337+00:00 | [hf-models](https://huggingface.co/api/models?search=Qwen3.8-27B&sort=lastModified&direction=-1&limit=100&full=true) | 200 / 152555 | `fbec6cfc4e7c1ef422d1cdd65a52ad3b5e8cf046c366fb75f11146014c13d981` |
| hf-exl | 2026-11-12T00:53:00.521812+00:00 | [hf-exl](https://huggingface.co/api/models/turboderp/Qwen3.8-27B-exl3/revision/SC_2.00bpw_H3) | 200 / 20564 | `d85dd5b72af16e5eda4c962c5cca63bc61daccdbec8fb0a643a985b1143d306c` |
| hf-unsloth | 2026-11-12T00:53:00.545741+00:00 | [hf-unsloth](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF?blobs=true) | 200 / 19471 | `e7ba029c3b65a4bcb9627cba3f713afa57203dca06abcf23097556f0db4b94a1` |
| hf-unsloth-history | 2026-11-12T00:53:49.066192+00:00 | [hf-unsloth-history](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/commits/main?limit=100) | 200 / 14943 | `0d40c71c9bd8258717a17edfa7d14c3efe18b08f2cad8d7f92b5f916c1551aa2` |
| hf-unsloth-tree | 2026-11-12T00:53:49.067146+00:00 | [hf-unsloth-tree](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/tree/main?recursive=true&expand=true&limit=100) | 200 / 42879 | `e621a1fcbd3c1c1322b772999a41ca119261054dc435c07d950a34fa937b9a78` |
| hf-wide | 2026-11-12T00:53:49.067569+00:00 | [hf-wide](https://huggingface.co/api/models?search=Qwen3.8-27B&sort=lastModified&direction=-1&limit=1000&full=true) | 200 / 2341467 | `35635a56c319aac7409dc193c863ea861b60d0dbf7dc218f1f01c0af385c2b60` |
| nif-method | 2026-11-12T00:53:49.068024+00:00 | [nif-method](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/docs/performance/methodology.md) | 200 / 6673 | `65231a0cfb76ca58dc0d1a2115231492f61effc6b9d89d8f6c69c0a862412e9e` |
| nif-launch-commit | 2026-11-12T00:53:49.068402+00:00 | [nif-launch-commit](https://api.github.com/repos/Brontir/NInferno/commits/5a949fda2f39) | 200 / 27419 | `2a95b24a263377a04a774fb7362b53e979210a7c4e8d479811d34fcde738b78f` |
| nif-row-commit | 2026-11-12T00:53:49.068753+00:00 | [nif-row-commit](https://api.github.com/repos/Brontir/NInferno/commits/f5e97e0bce) | 200 / 228369 | `be17849f5a780d49e9557304f47dfe5758c5ea13081e996ca33e59df78f4c230` |
| ik-graphs | 2026-11-12T00:53:49.312946+00:00 | [ik-graphs](https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/ceee8f888400/ggml/src/ggml-cuda.cu) | 200 / 257711 | `68ae0038325da4ff45c9df9366fa74bd996459f6e984215f8557135c3bf2c611` |
| llama31506 | 2026-11-12T00:53:49.329640+00:00 | [llama31506](https://api.github.com/repos/ggml-org/llama.cpp/pulls/31506) | 200 / 22572 | `23522188e82e06ea74bc06cf6eff9f9c02be5926aca35daf154b402f5203bdd5` |
| llama-head | 2026-11-12T00:53:49.514418+00:00 | [llama-head](https://api.github.com/repos/ggml-org/llama.cpp/commits?per_page=1) | 200 / 6555 | `2146d7c626f72459c694eb5f214f4ef5ceb3e4eb7eee807c976acf27d2a0dece` |
| ik-head | 2026-11-12T00:53:49.529728+00:00 | [ik-head](https://api.github.com/repos/ikawrakow/ik_llama.cpp/commits?per_page=1) | 200 / 4407 | `50f79d81ab5dc53846f4409992cc0e57d5b62f1576d31bacc5fbb15db0bcd246` |
| exl-int8-correct | 2026-11-12T00:53:49.644845+00:00 | [exl-int8-correct](https://raw.githubusercontent.com/turboderp-org/exllamav3/4db6a855c1f1/exllamav3/exllamav3_ext/quant/exl3_gemv_int8.cu) | 200 / 16306 | `f95ea99d2c0e0767c70873ba4d7c6666282771f94e61b692aee2dc52acebe277` |
| new-exl-meta | 2026-11-12T00:54:27.123617+00:00 | [new-exl-meta](https://huggingface.co/api/models/doth4580/Qwen3.8-27B-exl3-2.0bpw?blobs=true) | 200 / 21769 | `dea9c3ea52596af481b9b9ef19f0684f8efe4f40f4db277e3506bbc79be4224f` |
| new-exl-card | 2026-11-12T00:54:27.124303+00:00 | [new-exl-card](https://huggingface.co/doth4580/Qwen3.8-27B-exl3-2.0bpw/raw/main/README.md) | 200 / 70610 | `f6e0563da1a94599410a5aa0e58741bbe75c32e6aa25f3b99d4183d8eeb9651e` |
| new-trellis-meta | 2026-11-12T00:54:27.125037+00:00 | [new-trellis-meta](https://huggingface.co/api/models/the-sweater-cat/Qwen3.8-27B-trellis-2.5bpw?blobs=true) | 200 / 40846 | `e40fb9fd3c8a1cc1c7c274adf97c763930a2c4de3f5472d151b3ba83ca257ac5` |
| new-trellis-card | 2026-11-12T00:54:27.125597+00:00 | [new-trellis-card](https://huggingface.co/the-sweater-cat/Qwen3.8-27B-trellis-2.5bpw/raw/main/README.md) | 200 / 5466 | `9bc1b6109fd65b05678e92ee3118e61c776d23b324ed853180caa9d57b0e98b2` |
| iq2-refresh-meta | 2026-11-12T00:54:27.126062+00:00 | [iq2-refresh-meta](https://huggingface.co/api/models/daslab-testing/Qwen3.8-27B-IQ2_XXS-refresh?blobs=true) | 200 / 18270 | `4f76d193a05f56b172ecf478ce1c116f86cd65ee335c0864b7dda1d78c8fdf21` |
| iq2-refresh-card | 2026-11-12T00:54:27.126606+00:00 | [iq2-refresh-card](https://huggingface.co/daslab-testing/Qwen3.8-27B-IQ2_XXS-refresh/raw/main/README.md) | failed / — | `HTTP Error 404: Not Found` |
| iq2-alternate-meta | 2026-11-12T00:54:27.362488+00:00 | [iq2-alternate-meta](https://huggingface.co/api/models/daslab-testing/Qwen3.8-27B-IQ2_XXS-alternate?blobs=true) | 200 / 18274 | `331bcedc88a3719d2567eb4844c21e0077c3312f950665a05594c2b685f9b880` |
| iq2-alternate-card | 2026-11-12T00:54:27.380922+00:00 | [iq2-alternate-card](https://huggingface.co/daslab-testing/Qwen3.8-27B-IQ2_XXS-alternate/raw/main/README.md) | failed / — | `HTTP Error 404: Not Found` |
| new-dr-meta | 2026-11-12T00:54:27.381898+00:00 | [new-dr-meta](https://huggingface.co/api/models/Grawnch/Qwen3.8-27B-DR-2.50bpw-GGUF?blobs=true) | 200 / 10528 | `2d52425c94e4127d94e84f3046936c26a98b209f61976b10dd22d43d46424098` |
| new-dr-card | 2026-11-12T00:54:27.385984+00:00 | [new-dr-card](https://huggingface.co/Grawnch/Qwen3.8-27B-DR-2.50bpw-GGUF/raw/main/README.md) | 200 / 1420 | `51a42406e3114a323ea010345197e482b8a18fc5c8a4711f5964cabb1b323523` |
| hf-old-unsloth | 2026-11-12T00:54:27.395924+00:00 | [hf-old-unsloth](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/revision/4ca720788d1e01f1bff70c033e0d0028fd02e502?blobs=true) | 200 / 19224 | `1599e67e96d5368d450fb01041207377fd1621f529c3dc8d4c46f4759dedebe1` |
| unsloth-card | 2026-11-12T00:54:27.404876+00:00 | [unsloth-card](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/raw/ce331efc48d5722b4b8041f72793e530eae18d9b/README.md) | 200 / 6945 | `68884946cd5043f191733b0acf7714dc6a1aeb055ee911ec4c487515178b69a2` |
| exl-file-hashes | 2026-11-12T00:54:27.529530+00:00 | [exl-file-hashes](https://huggingface.co/api/models/turboderp/Qwen3.8-27B-exl3/revision/SC_2.00bpw_H3?blobs=true) | 200 / 22485 | `ad31b51abf21d6ccf9ea91a7b089cd483d2c9529e2622cd1eb34190230114e22` |
| nif-history-2080 | 2026-11-12T00:54:27.532462+00:00 | [nif-history-2080](https://api.github.com/repos/Brontir/NInferno/commits?path=docs/performance/runs/rtx2080ti/qwen3.8-27b.md&per_page=10) | 200 / 18561 | `aa9b23fbd31431121f0590b8145b927e9a925340d64f644fae52b40995caf545` |
| nif-launch-source | 2026-11-12T00:54:27.536321+00:00 | [nif-launch-source](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659a/src/core/launch_node.h) | failed / — | `HTTP Error 404: Not Found` |
| foscar-pdf | 2026-11-12 (see PDF record) | [foscar-pdf](https://arxiv.org/pdf/2611.09722v1) | 200 / 1861027 | `1f5e99449985e8f403ab03e674700b6b50e596e3e0101ae9e73067ec0cafb735` |
| nif-bench | 2026-11-12T01:00:46.023296+00:00 | [nif-bench](https://raw.githubusercontent.com/Brontir/NInferno/72fded0c659ad1f3d9b24b4b56eb9e4c6886092b/docs/performance/data/bench.jsonl) | 200 / 596711 | `4b2591b1c31294de909cec8f12c14b8511bf29443397cdca4f3cc068c8b9dca6` |
| exl-issues | 2026-11-12T01:00:46.024058+00:00 | [exl-issues](https://api.github.com/search/issues?q=repo:turboderp-org/exllamav3+updated:>=2026-10-31+(Turing+OR+sm_75+OR+split-K)&per_page=100) | 200 / 52640 | `4b1f00f8ba1497276fc112346413d88300e1668cfb22dfd7bfaa6a88d9f31cff` |
| escha-meta | 2026-11-12T01:00:46.024703+00:00 | [escha-meta](https://huggingface.co/api/models/aj9o9/Qwen3.8-27B-Escha-W2-GGUF) | 200 / 10803 | `cf8bad7d791a0177b427bd7d33d46fae0ff75aba34b7936fa790c2c01114dc0a` |
| nif-history-decisions | 2026-11-12T01:00:46.025393+00:00 | [nif-history-decisions](https://api.github.com/repos/Brontir/NInferno/commits?path=docs/maintainer/decisions.md&since=2026-11-10T00:00:00Z&until=2026-11-12T01:10:00Z&per_page=100) | 200 / 52633 | `3b35832f7100987ef2d3118b0e12760d2362245016cb04e006fb1acbdbf1955b` |
| vcruz-meta | 2026-11-12T01:01:58.786743+00:00 | [vcruz-meta](https://api.github.com/repos/vcruz305/exl3-turing) | 200 / 5203 | `3959e52a2029ef55eb2f693cdd50db9d64e969c0eacf53b4bb7253ecc77cdb6c` |
| weicj-exl-meta | 2026-11-12T01:01:58.787617+00:00 | [weicj-exl-meta](https://api.github.com/repos/weicj/exllamav3-turing) | 200 / 16353 | `eb43680bea2dc50b5b20d9f1d2e1c54124ac26847e34f9cf2cf254d0879950ca` |
| kyojin-meta | 2026-11-12T01:01:58.788054+00:00 | [kyojin-meta](https://api.github.com/repos/Yamz-Labs/kyojin) | 200 / 6441 | `d4b6b75ab5cf23a4d7e36e9bf3d47e2edc63ee216ca04c3b1505b09a374cc4da` |
| lmdeploy-meta | 2026-11-12T01:01:58.788479+00:00 | [lmdeploy-meta](https://api.github.com/repos/InternLM/lmdeploy) | 200 / 6348 | `de1b71f852b1030ea1b39edd1517dcd6d1d7c1d3bdd782597fee0a1267767208` |
| weicj-sm75 | 2026-11-12T01:01:58.788977+00:00 | [weicj-sm75](https://raw.githubusercontent.com/weicj/exllamav3-turing/master/doc/sm75.md) | failed / — | `HTTP Error 404: Not Found` |
| kyojin-readme | 2026-11-12T01:01:58.789409+00:00 | [kyojin-readme](https://raw.githubusercontent.com/Yamz-Labs/kyojin/main/README.md) | 200 / 12750 | `e1b46e28b2d7fca2fa5f7b0ec583abbb3867fd207f6f471f99c631c0213eb704` |
| vcruz-readme | 2026-11-12T01:01:59.020928+00:00 | [vcruz-readme](https://raw.githubusercontent.com/vcruz305/exl3-turing/main/README.md) | 200 / 9127 | `882ee5d71b9c553ae3711e4c5f95e2e1ef86ba12d2c2d89a69fc8fe8fb47951b` |
| weicj-sm75-correct | 2026-11-12T01:02:22.245602+00:00 | [weicj-sm75-correct](https://raw.githubusercontent.com/weicj/exllamav3-turing/turing-sm75/doc/sm75.md) | 200 / 9480 | `b312c28fd89ce7293923a3c7178e962991f5d9767b22c24dd24bb8e9292ff3ac` |
| kyojin-commits | 2026-11-12T01:02:22.246540+00:00 | [kyojin-commits](https://api.github.com/repos/Yamz-Labs/kyojin/commits?since=2026-10-31T00:00:00Z&until=2026-11-12T01:00:00Z&per_page=100) | 200 / 413874 | `c551d38a18292280229b8154576a34553ad86c53debfe2299124bb4a1b008b9e` |
| lmdeploy-commits | 2026-11-12T01:02:22.246908+00:00 | [lmdeploy-commits](https://api.github.com/repos/InternLM/lmdeploy/commits?path=src/turbomind&since=2026-10-31T00:00:00Z&until=2026-11-12T01:00:00Z&per_page=100) | 200 / 2 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| kyojin-p2 | 2026-11-12T01:03:16.111817+00:00 | [kyojin-p2](https://api.github.com/repos/Yamz-Labs/kyojin/commits?since=2026-10-31T00:00:00Z&until=2026-11-12T01:00:00Z&per_page=100&page=2) | 200 / 64172 | `fa678dad7ae93c86fc2612996e38bd09b153fbca78ef0011fc20f3bbdb2fc1e4` |
| kyojin-tree | 2026-11-12T01:03:16.112368+00:00 | [kyojin-tree](https://api.github.com/repos/Yamz-Labs/kyojin/git/trees/main?recursive=1) | 200 / 480259 | `8879ef1e3503c1a9da6cb54d0ce664ebef265b769ddc60c494c2d55b4e7852c5` |
| kyojin-qdec | 2026-11-12T01:03:34.653892+00:00 | [kyojin-qdec](https://raw.githubusercontent.com/Yamz-Labs/kyojin/1cb0b97b610e5838f78a6c13c0f2d6a2e8e55b8d/exllamav3/modules/attention_fn/qdec_cuda.py) | 200 / 13076 | `3aea86ad0a2b651db7a2319cf61857596bb53af9fd95e9a4e4570cb50c4074a2` |
| vllm-turing-search | 2026-11-12T01:03:55.160571+00:00 | [vllm-turing-search](https://api.github.com/search/issues?q=repo%3Avllm-project/vllm%20updated%3A%3E%3D2026-10-31%20%28Turing%20OR%20sm75%20OR%20sm_75%29&per_page=100) | 200 / 479154 | `39824e2a4a544b5fb8ebe056eb616c73741d9aa08da968198328d7d7205d1df7` |
| sglang-turing-search | 2026-11-12T01:03:55.161332+00:00 | [sglang-turing-search](https://api.github.com/search/issues?q=repo%3Asgl-project/sglang%20updated%3A%3E%3D2026-10-31%20%28Turing%20OR%20sm75%20OR%20sm_75%29&per_page=100) | 200 / 42011 | `d44e331671ce1b1641793a40f0e640887cae69c1514deb6bb0d405e511d86ca0` |
