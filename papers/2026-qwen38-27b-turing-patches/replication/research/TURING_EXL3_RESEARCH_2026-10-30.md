# EXL3 on Turing: running Qwen3.8-27B (2.0 bpw + MTP) on a stock RTX 2080 Ti

Research report, 2026-10-29. Prepared by Claude for Matt.

Scope: this is desk research only. It covers source reading, CPU-only compiles, and GitHub and Hugging Face records. **Nothing has been run on the 2080 Ti or on any other Turing GPU.** Every speed and VRAM number for the 2080 Ti below is a projection unless it is marked as measured.

**Evidence labels used in this report**

- **[Verified]**: checked by an independent verification pass, through a source read, a compile, a `cuobjdump`, or the GitHub API.
- **[Reported]**: a third party's own measurement or statement. It was not reproduced here.
- **[Inference]**: my reasoning or projection from verified facts.
- **[Critic]**: found by the final critic review. The evidence tracks did not re-verify it.

**Citation conventions**

- `v1.6.5:path:line` is upstream exllamav3 at tag v1.6.5 (commit cdc785d).
- `lab:path:line` is `/opt/p8/repository/engines/exl3-1.4.7-lab/` (the lab's exllamav3 1.4.7).
- `labtabby:` is `/opt/p8/repository/engines/tabbyAPI/` (commit 4a4f9f4).
- Paths inside the exllamav3 CUDA extension are under `exllamav3/exllamav3_ext/`, which is abbreviated `ext/`.

---

## Short glossary

These terms are also defined where they first appear.

- **Compute capability / sm_XX.** NVIDIA's version number for a GPU architecture. **sm_75** is Turing: RTX 20-series, Tesla T4, Quadro RTX. sm_80 and sm_86 are Ampere (A100, RTX 30). sm_89 is Ada: RTX 40, which includes the lab's RTX 4080 Laptop.
- **SM.** A streaming multiprocessor, one of the GPU's cores. The 2080 Ti has 68.
- **cubin / SASS / PTX.**
  - A cubin is compiled GPU machine code for one architecture. SASS is the name for those machine instructions.
  - PTX is NVIDIA's portable assembly. The driver can compile PTX at load time, but only for the same architecture or a newer one.
  - **ptxas** is the assembler that turns PTX into SASS. It is the tool that prints "requires .target sm_80" errors.
- **Shared memory.** Fast on-chip scratch memory that a block of threads shares. Turing allows at most 64 KB per block; Ampere GA10x allows 99 KB.
- **cp.async.** An Ampere-only instruction that copies data from global memory (VRAM) into shared memory in the background, so the kernel can compute while the next tile loads.
- **mma / HMMA.** A tensor-core matrix-multiply instruction. **m16n8k16** names the tile shape (16x8 output, 16-deep inner dimension). Turing's fp16 tensor cores only have **m16n8k8**, which is half the depth, so one Ampere instruction becomes two Turing instructions.
- **GEMV / GEMM.**
  - A GEMV is a matrix-vector multiply. It handles one token row and dominates batch-1 decode.
  - A GEMM is a matrix-matrix multiply. It handles many rows and is used in prefill and in draft verification.
- **dp4a (IDP.4A).** A 4-way int8 dot-product instruction, available from sm_61.
- **Triton.** A Python GPU-kernel compiler. Since Triton 3.3 it emits plain scalar fused multiply-adds (**FFMA**) on sm_75 instead of tensor-core instructions.
- **EXL3.** exllamav3's quantization format, built on QTIP trellis coding. Weights are decoded on the fly with integer arithmetic ("trellis decode"), which costs more compute per weight than plain integer quants such as GGUF Q4.
- **bpw.** Bits per weight. **K** is the per-tensor bit width (1-4) inside an EXL3 model.
- **MTP.** Multi-token prediction. The model has one extra layer that drafts the next few tokens; the main model then checks them in one verify pass. This is a form of speculative decoding. **DFlash2** is a separate draft model.
- **GDN.** Gated DeltaNet, the recurrent (linear-attention) layer type used in 48 of Qwen3.8's 64 layers.
- **Q8 cache.** An 8-bit quantized key/value cache for the 16 full-attention layers.
- **Cooperative launch / grid.sync.** A kernel launch in which every thread block can wait for all the others. EXL3's GEMM uses it.
- **Accuracy per minute.** The lab's metric: correct eval answers per minute of wall-clock time. It rewards both speed and spending fewer reasoning tokens.

---

## 1. Bottom line

**Can EXL3 run on a 2080 Ti?** Yes, and the porting work is already done upstream.

- **[Verified]** Turing support was merged into exllamav3 as PR #325 on 2026-09-26 and shipped in v1.5.2 as "Experimental Turing support".
- **[Verified]** The current official wheel, `exllamav3-1.6.5+cu128.torch2.10.0-cp312`, contains 169 sm_75 cubins, including the Turing flash-attention kernel `fa75`. It has the same torch/CUDA combination as the lab's `exl3-env`, so no build is needed to get a working baseline.
- **[Verified]** Upstream TabbyAPI has accepted compute capability 7.5 since PR #463 (2026-09-26).
- **[Reported]** Qwen3.8-27B with MTP or DFlash2 drafting already runs on Turing for at least four independent people. All of them use 16-24 GB cards or two GPUs.

**The lab's current stack cannot run on Turing as it stands.**

- **[Verified]** exllamav3 1.4.7 predates the port. Its compiled `.so` contains only sm_80-sm_120 code. Static analysis plus compile probes show it would abort or fail at three points:
  - the 90 KB shared-memory request of the GEMM (`ext/quant/exl3_gemm_inner.cuh:7`, which exits through `cuda_check` at `ext/quant/coop_autotune.cu:317`);
  - the 98,304-byte Triton prefill-attention tile;
  - the 2-row int8 GEMV.
- **[Verified]** The lab TabbyAPI rejects any GPU below compute capability 8 (`labtabby:common/hardware.py:16`).
- **Choice:** do not port 1.4.7. Run on upstream v1.6.5 instead. The lab's own 1.4.7 patches are three small Python files, about 167 lines, with no CUDA changes, so nothing important is lost.

**How hard is it?**

- **Getting it to run** takes about a day once Matt approves the desktop setup: a venv, the official wheel, a model copy, and an upstream TabbyAPI checkout.
- **Making it competitive** takes about 3-5 weeks. Most of that is measurement. The code changes are small and limited to sm_75: about 60-100 lines that fix how decode work is routed to kernels.
  - **[Inference]** On sm_75, upstream routes roughly 30% of decode weight bytes to a GEMM that has lost its copy/compute overlap.
  - **[Verified code facts]** The causes are three gates:
    - The fast int8 GEMV is switched off on Turing (`v1.6.5:ext/quant/exl3_gemv_int8.cu:279`).
    - The GEMV routing rule has no Turing branch (`v1.6.5:ext/quant/exl3_gemv.cu:46-71`).
    - The CUDA build has no multi-matrix GEMV (`v1.6.5:ext/quant/exl3_gemm.cu:666-703` is ROCm-only).
- Optional custom kernels add 1-2 weeks each and are only worth building if profiling shows a need.

**Expected speed on a stock 2080 Ti compared with llama.cpp on the same card.** All of these are **[Inference]**; nothing has been measured on Turing. Model: 2.0 bpw EXL3, Q8 cache, batch 1, short context.

| Lane | Undrafted decode | Drafted decode, code/math | Prefill | Context with a drafter on 11 GB |
|---|---|---|---|---|
| EXL3 v1.6.5, stock | ~19-32 tok/s | ~25-40 (MTP) | ~600-800 tok/s | ~32K (48K tight) |
| EXL3 + environment-variable dispatch fixes | ~26-38 | ~36-55 | same | same |
| EXL3 + small patches incl. int8 fit-gate (if it proves correct) | ~34-50 | ~47-80 | +10-15% with fp16-compute cuBLAS | same |
| llama.cpp UD-IQ2_XXS | ~37-42 | ~75-85 with DFlash2 | ~900-1,200 (pp512) | ~4-6K with DFlash2 |
| llama.cpp UD-IQ2_S + MTP sidecar | ~35-40 | ~60-72 | same | ~16-24K |

How to read the table:

- **Undrafted decode.** Stock EXL3 is probably about 0.5-0.8x llama.cpp on this card. With the dispatch fixes and a working int8 path it is plausibly 0.85-1.2x. llama.cpp is likely faster at prefill whatever is done to decode.
- **Measured anchors** behind these ranges:
  - **[Verified lab data]** On the 4080 Laptop at 80 W, EXL3 2.0 bpw decoded at 29.2 tok/s undrafted and 48.7-50.4 tok/s with MTP. UD-IQ2_XXS decoded at 25.1 and UD-IQ2_S at 23.6 (`testsuite/results/EFFICIENCY-CAMPAIGN.md`, E1/E7).
  - **[Reported]** Upstream v1.5.3 decoded 4.0 bpw at 17.9 tok/s on an overclocked 22 GB 2080 Ti. rafatxf's unmerged fork got 33.3 tok/s on the same card.
- **Why EXL3 can still win.**
  - In the lab's laptop evals, EXL3 2.0 bpw spent fewer reasoning tokens than UD-IQ2_S: median math25 1,083 vs 1,381-1,447 (`EFFICIENCY-CAMPAIGN.md:347-348`). The mean ratio is about 0.88.
  - It also had higher accuracy per minute: 3.65 vs 2.57 on HumanEval+ and 1.63 vs 1.26 on math25 (`EFFICIENCY-CAMPAIGN.md:317-322`).
  - These come from small samples (n=20-25, single runs, temperature 1.0) and must be re-measured on the 2080 Ti.
  - EXL3 is also probably the only lane that fits 32K context plus a drafter in 11 GB **[Inference]**.

**Recommendation.** Use the "alternative-route" plan with grafts from the other two designs (section 6):

1. Install the official upstream wheel. Do not port.
2. Build a matched llama.cpp baseline on the same card.
3. Apply fixes in layers: environment variables first, then Python, then at most three small CUDA patches. Add each layer only if profiling justifies it.
4. Let a paired accuracy-per-minute shootout decide which engine becomes the 2080 Ti lane.

Each fix that survives goes upstream as an isolated pull request with stock 11 GB 2080 Ti numbers. Nobody has published numbers for that configuration.

---

## 2. Prior art

### 2.1 Plain answer

Running EXL3 on Turing is **not new**: it is upstream and already used. As of 2026-10-29 I found **no** public report of:

- (a) a stock 11 GB 2080 Ti running EXL3; or
- (b) Qwen3.8-27B at about 2.0 bpw on any Turing card.

**[Verified]** One 2.05 bpw EXL3 model has run on Turing: weicj ran Qwen3.8-Flash-Next on dual 22 GB 2080 Tis. So the novelty claim must be specifically "Qwen3.8-27B at 2.0 bpw on a stock 11 GB 2080 Ti", not "2.0 bpw on Turing".

**Decode-side Turing kernels exist but are not upstream.** They are in rafatxf's fork.

### 2.2 Upstream exllamav3 timeline [Verified]

| Date | Event | Source |
|---|---|---|
| 2025-04-13 | Earliest dated attempt: a Colab T4 build fails with "Feature 'cp.async' requires .target sm_80 or higher" | https://github.com/turboderp-org/exllamav3/issues/14 |
| 2026-09-02 | rluisr opens PR #325, "Turing (sm_75) support" | https://github.com/turboderp-org/exllamav3/pull/325 |
| 2026-09-09 | fla (flash-linear-attention) vendored into exllamav3 (92939e5) | git log |
| 2026-09-26 | PR #325 merged into `dev` (8cb1009, 24 commits, +1102/-26); turboderp's rework f4db698 adds a per-device shared-memory budget, config ladders and eager fallback for graph kernels; 12414d0 adds 7.5 to the wheel arch list | PR #325; `v1.5.2:.github/workflows/build.yml:56` |
| 2026-09-26 | v1.5.2 released: "Experimental Turing support" | https://github.com/turboderp-org/exllamav3/releases/tag/v1.5.2 |
| 2026-09-27 | #411 (Pakobbix): sm_75 paged-attention tile choice plus GDN chunk-kernel autotune configs measured on a 2080 Ti; first shipped in v1.5.3 | e86433b |
| 2026-10-08 | #439 (rafatxf, reworked by turboderp): Turing prefill over a dequantized window via SDPA or the new `fa75` kernel. Qwen3.8-27B 4.0 bpw on a 22 GB 2080 Ti: prefill 635/267/93 → 779/739/623 tok/s at 2K/16K/64K | https://github.com/turboderp-org/exllamav3/pull/439, commit 4d8525c |
| 2026-10-10 | v1.6.1: "Turing (sm_75) optimizations", fa75, Triton mixed-architecture autotune workaround (60c0162), two-row GDN state pool (c1136a0) | https://github.com/turboderp-org/exllamav3/releases/tag/v1.6.1 |
| v1.6.2 | GDN recurrent state stored in fp16 by default (0244652) | git tag --contains |
| 2026-10-25 | v1.6.5 (cdc785d); cu128/torch2.10 wheels for cp310-cp314 | https://github.com/turboderp-org/exllamav3/releases/expanded_assets/v1.6.5 |

**[Verified]** The v1.6.5 cp312 wheel (160,944,251 B) contains 169 cubins each for sm_75, 80, 86, 89, 90, 100 and 120. The sm_75 cubin has `fa75` symbols. It has no `fdq4`, `gdnh75` or `gemv_sk` symbols, which are rafatxf's decode kernels.

### 2.3 The maintainer's position over time [Verified quotes]

- **2025-05-28 (#44):** "It's also not the only place where ExLlamaV3 relies on Ampere+ features. So it's not really a priority right now." https://github.com/turboderp-org/exllamav3/issues/44#issuecomment-2916304529
- **2026-03-05 (#162):** "The 2nd gen tensor cores on Turing don't support the m16n8k16 fragment shapes used in the kernel, and I believe the lane mappings for m16n8k8 are different so you can't just run two of those mmas instead. ... If speed isn't a major concern, it could be doable, I guess." https://github.com/turboderp-org/exllamav3/issues/162#issuecomment-4005093819
  - This early view was later overturned. Upstream `v1.6.5:ext/arch.cuh:12-14` now does exactly the two-mma split and calls it "bit-equivalent".
- **2026-03-13 (#14):** "Turing support is not currently planned, but that may change down the line."
- **2026-07-30 (#81, Colab T4):** "Yeah this can't really work on Colab T4."
- **2026-09-26 (TabbyAPI #463):** "Should work now with exllamav3==v1.5.2. Turing support is still largely untested though."
- **2026-10-02 (#417, on rafatxf's ~8k-line fork):** "This is really hard to evaluate as is. ... I can't really tell until I see isolated, minimized pull requests." https://github.com/turboderp-org/exllamav3/issues/417#issuecomment-5960304312
- **2026-10-07 (#162, closing):** "Turing is now supported. Performance is still not amazing, as expected, since the quant format is inherently more compute intensive than integer-quants." https://github.com/turboderp-org/exllamav3/issues/162#issuecomment-6045504989
- **2026-10-08 (#439):** "Don't know what can realistically be done about head_dim=512 models in 64kB of shmem."

### 2.4 Turing runs of Qwen3.8 EXL3 and ports [Reported unless noted]

| Who | Hardware | Model / bpw | Numbers | Source |
|---|---|---|---|---|
| vcruz305 (own port; int8 GEMV patched rather than declined) | Quadro RTX 6000 24 GB | Qwen3.8-27B 4.00 bpw, Q4 cache | 31-33 tok/s decode; MTP draft 1: 42-45 (draft 1 is optimal, deeper drafts are slower); prefill 812/457/296 tok/s at 24K/32K/64K | https://github.com/vcruz305/exl3-turing |
| rafatxf | 2080 Ti **22 GB** mod, 240 W, core capped at 1650 MHz | Qwen3.8-27B 4.0 bpw, Q4 cache | upstream v1.5.3: 17.9 tok/s at 4K, 11.9 at 64K; his fork: 33.3; MTP-3 52.7; DFlash2 81.8 (short prompts) | https://github.com/turboderp-org/exllamav3/issues/417, https://raw.githubusercontent.com/rafatxf/exllamav3/turing/doc/turing.md |
| nalimc | Quadro RTX 5000 mobile 16 GB | Qwen3.8-27B 2.60 bpw | ~500 pp / ~50 tg with MTP (rafatxf branch) | https://huggingface.co/turboderp/Qwen3.8-27B-exl3/discussions/12 |
| Ishannaik | 2x T4 (Kaggle, no P2P) | Qwen3.8-27B 4.0 bpw, tensor-parallel | 18-19 tok/s; DFlash2 55-57 | https://github.com/turboderp-org/exllamav3/issues/568 |
| weicj | dual 2080 Ti 22 GB, layer split | Qwen3.8-Flash-Next **2.05 bpw** | 547 pp / 39.7 tg | PR #325 comments |
| rluisr | T4 | Llama-3.2-1B; Qwen3-30B-A3B | 134 tok/s; 42.5 tok/s | PR #325 body |

Other attempts made before the merge **[Verified]**:

- rluisr branches `sm75`, `sm75-dev`, `sm75-upstream`.
- weicj/exllamav3-turing.
- Pakobbix `sm75-prefill-tiles`.
- Prebuilt T4 wheels: https://huggingface.co/Sigmo23/ext2l-wheels-orca.
- A MiniCPM5 EXL3 model card tested on T4: https://huggingface.co/ewin-reg/MiniCPM5-2B-EXL3-Quantized.
- A marcmy fork that has since been deleted.

Together these show strong community demand and a lot of duplicated effort.

### 2.5 What is not upstream: rafatxf's fork [Verified]

- **What it is.** The `turing` branch of https://github.com/rafatxf/exllamav3 has 46 commits on top of v1.5.3, dated 2026-09-27 to 09-30, tip 15654d3. It contains:
  - `fdq4`, flash-decoding from a Q4 cache;
  - a split-K GEMV and a GEMV rule for older GPUs;
  - unfused multi-projections on sm_75;
  - fp16-accumulate EXL3 GEMM on sm_75;
  - HMMA kernels for GDN;
  - a register-resident GDN recurrent step.
- **What was merged.** Only #421 (sliding-window decode), #424 (a generic generator requeue fix) and #439 (fa75). #423 is still open. #417 is an open issue.
- **What is planned.** rafatxf said on 2026-10-03 that the remaining pieces (decode attention, GEMV, GDN recurrent step) would come "one at a time after #439". None had been opened by 2026-10-29.
- **[Reported]** Measured effects on his 22 GB 2080 Ti at 4.0 bpw:
  - GEMV rule plus unfusing: weight time per token 49.7 → 34.1 ms.
  - fp16-accumulate GEMM on sm_75: +10% decode, KL 8e-5.
  - `CUBLAS_COMPUTE_16F` prefill: +10-15%, perplexity +0.08%.
  - Engine total: 17.9 → 33.3 tok/s at 4K; prefill 588 → 1,249 tok/s at 2K.
- **Caveat [Reported]:** the fork's doc says "nvcc is needed at runtime" for the fdq4 cubins (turing.md:34).

### 2.6 TabbyAPI [Verified]

- **Upstream.** PR #463 (Victor Cruz, merged 2026-09-26, commit e03ee4c) sets `MIN_CUDA_COMPUTE_CAPABILITY = (7, 5)` (upstream `common/hardware.py:3-5`) and still rejects Volta. Upstream TabbyAPI now pins exllamav3 1.6.5. https://github.com/theroyallab/tabbyAPI/pull/463
- **Lab copy.** The lab's copy (4a4f9f4, 2026-08-24) still requires cc ≥ 8 (`labtabby:common/hardware.py:8,16`; `labtabby:backends/exllamav3/model.py:305-316`, "All GPUs must be ampere (30 series) or newer") and pins exllamav3 1.4.3 (`labtabby:pyproject.toml:82-85`).

### 2.7 Other runtimes [Verified unless noted]

- **llama.cpp mainline** has no EXL3 or QTIP support. The only related discussion is #10125, user requests only: https://github.com/ggml-org/llama.cpp/discussions/10125.
- **ik_llama.cpp** has its own integer-only trellis quants (`IQ1_KT`-`IQ4_KT`). The format is not compatible with EXL3, and its CUDA backend officially supports "Turing or newer" (README:13).
  - An EXL3-runtime PR to ik_llama.cpp (#2762) was closed unmerged on 2026-10-28: https://github.com/ikawrakow/ik_llama.cpp/pull/2762. Its quoted performance remarks are about proof-of-concept kernels, so they are not evidence about exllamav3.
  - **[Critic]** Public ik_llama.cpp trellis quants of Qwen3.8-27B already exist: Thireus `IQ1..IQ4_KT`, including MTP variants; IQ2_KT is 7.18 GB, in a per-tensor split format.
- **QTIP reference kernels** (Cornell-RelaxML/qtip) use only `mma m16n8k16` (`qtip-kernels/src/inference.cu:364-365`). The author wrote on 2025-03-12: "Our kernels only run on NVIDIA Ampere GPUs (sm_80) or newer ... While it is possible to rewrite the kernel to work with Volta/Turing GPUs, I do not currently have the bandwidth to do so." https://github.com/Cornell-RelaxML/qtip/issues/25
- **vLLM plugins.** yeasah/vllm-exl3-plugin requires compute capability 8.0+. vcruz305/vllm-exl3 builds against his exllamav3 fork **[Inference: so it inherits that fork's Turing support]**. 1Cat-vLLM #1102 is V100-only. Project Maya v1.1.9 keeps "v1.1.8's decode kernels" on RTX 20/T4, with no Turing EXL3 number published.
- **Non-EXL3 baselines that are not comparable.**
  - weicj's vLLM-2080Ti-Definitive claims 220.84 tok/s. That is on 2x 22 GB cards with NVLink, on "synthetic text inputs with high speculative-hit rates", and the author says "The headline results are not from a stock 11GB card".
  - An X post (TeksEdge) claims ~48 tok/s for Q4_K plus MTP on a 22 GB mod. It is unverified.

### 2.8 Older than Turing [Verified]

- Volta and Pascal EXL3 support is still open territory.
- **Volta (sm_70).** PR #404 (V100, open) reached 14.7 tok/s and reports that "the int8 kernels mis-execute on cc < 8". PR #475 is also open.
- **Pascal (sm_61).** Requests #509 and #512 have no maintainer reply.
- Both TabbyAPI and the merged port deliberately reject Volta, because `m16n8k8` needs sm_75.

### 2.9 Coverage limits [Verified]

- exllamav3 and TabbyAPI have GitHub Discussions disabled (both return HTTP 404), so issues and PRs are the full public record.
- Reddit could not be fetched.
- Not searched: academic microbenchmark papers, the exllama Discord, and Chinese forums, where 22 GB 2080 Ti mods are common. See section 7.

---

## 3. What breaks on Turing and why

### 3.1 The short version

On Turing, three hardware features that EXL3 relies on are missing:

- the cp.async background copy;
- the deep m16n8k16 tensor-core instruction;
- more than 64 KB of shared memory per block.

Separately, current Triton runs every matrix multiply on plain FP32 cores on Turing.

The first three each have a working fallback upstream. The fourth is a speed problem, not a correctness problem. bf16 is not a problem: exllamav3 only converts to and from bf16 and never does bf16 math or bf16 MMA.

### 3.2 Hard blockers and fixes [Verified]

Difficulty scale: 0 means nothing to do, 1 means a few lines, 2 means about a day, 3 means several days, 5 means a new kernel.

| Component | Ampere-only feature (why it breaks) | Where (lab 1.4.7) | Fallback (upstream v1.6.5 fix) | Difficulty | Evidence |
|---|---|---|---|---|---|
| EXL3 GEMM / multi-GEMM / MoE pipeline | `cp.async` background copy: ptxas rejects it on sm_75 | `lab:ext/ptx.cuh:159-168,188-199`; call sites `ext/quant/exl3_gemm_inner.cuh:232,245,257,264,633` | Synchronous 16-byte load+store; commit/wait become no-ops (`v1.6.5:ext/ptx.cuh:246-330`). Correct, because `wait_stage` already does `__syncthreads`. Overlap within a warp is lost | 1 (sync copy); 3 (register-staged prefetch) | Compile fails without it; patched version compiles |
| EXL3 GEMM inner loop and small-m GEMV | `mma.sync m16n8k16` with f16 inputs: sm_80 only | `lab:ext/ptx.cuh:52-99`; `lab:ext/quant/exl3_gemv_kernel.cuh:35-49` | Two `m16n8k8` with the register split (a0,a1,b0) then (a2,a3,b1) (`v1.6.5:ext/ptx.cuh:88-108,140-160`; `v1.6.5:ext/quant/exl3_gemv_kernel.cuh:52-68`). Compiles to native `HMMA.1688` | 1 | Compile fails without it; SASS checked |
| int8 GEMV, staged unit for K=3/5/7 | `cp.async` | `lab:ext/quant/exl3_gemv_int8_kernel.cuh:647-708` | Sync copy, or send K=3 to the existing narrow unit on sm_75 | 1 | Compile fails for sq_k3/k5 and coop_k3/k5/k7 |
| GEMM / MGEMM / MoE launch | Requests a fixed 90 KB of dynamic shared memory; Turing allows 64 KB per block, so `cudaFuncSetAttribute` fails and `cuda_check` calls `exit()` | `lab:ext/quant/exl3_gemm_inner.cuh:7`; `ext/quant/exl3_gemm.cu:243,292,550,615`; `ext/quant/coop_autotune.cu:317`; `ext/util.cuh:92-100` | Per-device request = min(device opt-in limit, 90 KB), and shapes that do not fit are dropped (`v1.6.5:ext/quant/exl3_devctx.cu:56-75`; `ext/quant/exl3_kernel_map.cu:177-210`). Real need is ≤ 53 KB for shapes 1-3 and ≤ 63.5 KB for shape 4 up to K=7; only shape 4 at K=8 (67,584 B) cannot fit | 2 | Code + CUDA docs; 64 KB compile experiment |
| int8 GEMV launch | Shared-memory budget sized for ≥ 80 KB. M=1 fits (50,176 B at K=1/2/4, 56,320 B at K=3, plus 144 B static). M=2 (a one-draft verify) needs 83,968-90,112 B; the coop kernel needs 65,536 + 656 B; residual mode needs 82,944 B or more | `lab:ext/quant/exl3_gemv_int8.cu:146-162,283`; `ext/quant/exl3_gemv_int8_kernel.cuh:951-956,1150` | Upstream declines **all** int8 below 80 KB (`v1.6.5:ext/quant/exl3_gemv_int8.cu:279`; its comment calls this "a performance loss and nothing more"). Better: a per-call fit test that keeps M=1 | 1 (M=1 gate); 2 (M=2/coop, which needs a host+device shared constant) | Arithmetic + `ptxas -v` static sizes; not run |
| Triton prefill attention, head_dim 256 | Tiles sized for ~100 KB: 98,304 B (fp16), 82,176 B (Q8), 98,560 B (Q4) | `lab:exllamav3/modules/attention_fn/triton_paged.py:1779-1800` | Smaller tiles; upstream moved sm_75 prefill to SDPA or `fa75` (`v1.6.5:exllamav3/util/backend.py:74-123`; `v1.6.5:ext/turing/fa75.cu`) | Done upstream | CPU-only Triton compile |
| Build and binaries | Wheels before v1.5.2 carry no sm_75 code; the lab `.so` has sm_80-sm_120 cubins plus sm_120 PTX only | `v1.4.7:.github/workflows/build.yml:51` | Official ≥ v1.5.2 wheel, or build with `TORCH_CUDA_ARCH_LIST=7.5` | 0-1 | `cuobjdump` |
| TabbyAPI | Compute-capability gate at 8.0 | `labtabby:common/hardware.py:16` | Upstream TabbyAPI ≥ e03ee4c | 0-1 | Source read |

### 3.3 Performance-only issues

| Issue | Status | Detail |
|---|---|---|
| Triton has no tensor-core path on sm_75 | **[Verified]** | Since Triton 3.3 (PR #5066, commit 7275ff7291, 2024-11-05), `tl.dot` on compute capability < 8.0 lowers to scalar FFMA. Compiled probe: sm_75 gives 0 mma and 1,024 FFMA; sm_80 gives `mma.sync m16n8k16`. Triton 3.6's README lists "Compute Capability 8.0+". torch 2.10 pins triton 3.6. This affects decode/verify attention and the GDN chunk prefill kernels. https://github.com/triton-lang/triton/pull/5066 |
| Decode and verify attention stay on Triton FFMA | **[Verified]** code; cost **[Inference, low confidence]** | Upstream: "Decode and draft verification stay on the Triton kernels" (`v1.6.5:exllamav3/util/backend.py:81`). Rows are padded to 16 for 6 real GQA heads (`lab:exllamav3/modules/attention_fn/bc_attn.py:268-273`). Estimate at 32K context: ~34 GFLOP per token, roughly 2.6-5 ms. Removing the padding may make it bandwidth-bound (~1.7 ms) |
| Lost copy/compute overlap in the GEMM | **[Inference]** | Upstream: "The cost is the lost global->shared overlap" (`v1.6.5:ext/ptx.cuh:269-273`). Estimated 10-40% kernel slowdown, not measured |
| Dispatch: ~30% of decode bytes on the overlap-less GEMM | **[Inference from code]**, corroborated by rafatxf | K=1 tensors are never GEMV-eligible. Wide K=3 tensors (including the 3-bit lm_head) are rejected because the GEMV rule has no Turing branch. The fused GDN q/k/v+z bundle and k/v pairs go to the cooperative multi-GEMM, because the CUDA multi-GEMV is ROCm-only. Sources: `v1.6.5:ext/quant/exl3_gemv.cu:46-71`, `ext/quant/exl3_gemm.cu:666-703`, `exllamav3/model/config.py:77-93` |
| fp32-accumulate HMMA is half rate on GeForce Turing | **[Reported]** + **[Verified]** code | rafatxf measured 507 vs 987 FLOP/clk/SM on a 2080 Ti (turing.md:7). The fp16-accumulate fast paths are on only for sm_86 (`ext/quant/exl3_gemm_inner.cuh:9-18`) and for ≥ sm_80 (`ext/hgemm_f16acc.cu:403-404`). cuBLAS reconstruct GEMMs use `CUBLAS_COMPUTE_32F` (`ext/hgemm.cu:92-101`). That Quadro/Tesla Turing runs fp32-accumulate at full rate is from memory and unverified |
| GDN prefill (fla chunk kernels) on FFMA | **[Verified]** | All six kernels compile for sm_75. One `chunk_fwd_kernel_o` config (98,304 B) is skipped by the autotuner. Upstream #411 configs: chunk_o 24.6 → 1.4 ms and chunk_h 9.8 → 2.5 ms per GDN layer at T=1792 on a 2080 Ti, measured with Qwen3.5-9B (fewer value heads, so likely an underestimate for Qwen3.8) |
| Lower occupancy | **[Verified]** limits | Turing allows 1,024 resident threads and 64 KB shared memory per SM; Ada allows 1,536 threads and ~100 KB. Patched sm_75 GEMM kernels use up to 203 registers (shape 4 fits one block per SM), and some multi-GEMM instances spill 16-48 B |
| L2 cache size | **[Unverified]** | Believed to be ~5.5 MB on TU102. Read `l2CacheSize` on the desktop |

### 3.4 Unresolved: are two m16n8k8 bit-identical to one m16n8k16?

- **Upstream says yes.** `v1.6.5:ext/arch.cuh:12-14` says "bit-equivalent, not an approximation".
- **rluisr's PR says no.** It says there is "one extra rounding of the partial sum at the k=8 boundary".
- **[Inference]** For the fp16-accumulate GEMV there is certainly an extra fp16 rounding between the two halves. In any case, Turing `HMMA.1688` and Ampere `HMMA.16816` are different hardware operations.
- **Choice:** make no claim that outputs are identical across GPUs. Validate against an fp32 reference using tolerances (section 6, E2).

### 3.5 What does not break [Verified]

These all compile unchanged for sm_75:

- GDN recurrent and conv kernels: `ext/gdn.cu`, 52-78 registers, ≤ 4,160 B shared memory, no spills;
- norms, RoPE, Q8 cache quantize/dequantize (`ext/cache/q_cache.cu`), sampling, reconstruct, Hadamard;
- the attention fallback, graph code, cuBLAS fp16 GEMM;
- the 1-row int8 GEMV at K=1/2/4.

Other points:

- `__reduce_max_sync` in MoE routing is already guarded with a shuffle fallback (`ext/routing.cu:46-74`).
- The sm_80-only L2 cache-hint helper `cp_async_stream` (`ext/ptx.cuh:172-184`) has no callers.
- torch 2.10.0+cu128 ships sm_75 kernels: 456 sm_75 cubins in `libtorch_cuda.so`.
- flash-attn 2 does not support Turing, but exllamav3 does not use it.

---

## 4. The compile-probe results

All probes ran **CPU-only** on the laptop with `CUDA_VISIBLE_DEVICES=''` and `nice`. A verifier then reran them independently.

### 4.1 Single-instruction probe [Verified]

Tool: `/opt/p8-user/cuda-12.6/bin/nvcc -arch=sm_XX -c`.

| PTX feature | sm_75 | sm_80 |
|---|---|---|
| `mma.sync m16n8k16` f16 | **fails**: "Feature '.m16n8k16' requires .target sm_80 or higher" | ok |
| `cp.async`, `cp.async.commit_group`, `cp.async.wait_group` | **fails**: "Feature 'cp.async' requires .target sm_80 or higher" | ok |
| bf16 `mma` | **fails**: "Feature '.bf16' requires .target sm_80 or higher" | ok |
| `createpolicy.fractional`, `.L2::cache_hint` | **fails** | ok |
| `mma.sync m16n8k8` f16 (f32 and f16 accumulate) | ok (`HMMA.1688`) | ok |
| `__dp4a`, `ldmatrix.x4`, `mma m8n8k4` | ok | ok |

### 4.2 Full-extension sweep of lab 1.4.7 [Verified]

Setup:

- All 111 `.cu` translation units: 45 regular plus 66 template-instance units.
- Built with the official wheel's flags plus torch's `cpp_extension` flags, gcc 13.3, nvcc 12.6.
- The lab's extension source is byte-identical to the upstream v1.4.7 tag (`diff -rq` exit 0).

Results:

| Run | Result |
|---|---|
| sm_75, unmodified | **61 pass / 50 fail**; 538 of 1,074 kernel entry points (50.1%) build. Every failure is a ptxas error on exactly four features: cp.async (8,224 lines / 49 files), `.m16n8k16` (8,128 / 45), commit_group (7,408 / 49), wait_group (2,524 / 49). There are no front-end, bf16, static_assert or shared-memory errors. |
| Failing groups | 24 GEMM units (`ext/quant/comp_units/exl3_comp_unit_{1..8}_cb{0,1,2}.cu`, 384 kernels); 20 MoE units (36 kernels); 5 int8 GEMV units for odd K (28 kernels); `ext/quant/exl3_gemv.cu` (88 kernels, 1,408 `.m16n8k16` errors) |
| sm_80 control, same files and flags | **111/111 pass**, which confirms the probe setup |
| CUDA 12.8.93 cross-check (6-file sample) | Same pass/fail pattern and identical error counts |
| Two-header shim (`ptx.cuh` + `exl3_gemv_kernel.cuh`, guarded by `__CUDA_ARCH__ < 800`) | **50/50 previously failing units pass**; all 1,074 kernels build with names identical to sm_80. SASS: `exl3_comp_unit_2_cb2` has 416 `HMMA.1688.F32` and 72 `LDSM`, against 208 `HMMA.16816.F32` and 224 `LDGSTS` on sm_80. `exl3_gemv.cu` has 3,936 `HMMA.1688.F16`. The kernel-audit version of the same fix is a 36-line diff. |
| Shim plus 64 KB `SMEM_MAX` | Only `exl3_comp_unit_8_cb{0,1,2}` fail the kernel's own static_assert (K=8, 512-wide tile, 4 stages: 67,584 B). The 2.0 bpw model uses K=1-4, which all fit. |
| Register and spill cost | GEMV kernels use ≤ 127 registers with no spills (a few instances spill 4-8 B, about the same as sm_86). Patched GEMM kernels use up to 203 registers; some multi-GEMM instances spill 16-48 B. 27 kernels use more stack on sm_75 than on sm_80. |

### 4.3 Upstream v1.6.5 and the official wheel [Verified]

- Upstream HEAD `ext/quant/exl3_gemv.cu` compiles for sm_75 (exit 0, 37.8 s): 3,936 `HMMA.1688.F16`, 3,520 `IDP.4A`, 1,152 evict-first `LDG.E.EF`, and no local-memory use in all 88 kernels. Upstream `gdn.cu` also compiles.
- The official v1.6.5 cp312 cu128/torch2.10 wheel contains 169 sm_75 cubins and the fa75 symbols, so the base port needs no build.

### 4.4 Triton probes, CPU-only, with `GPUTarget('cuda', 75, 32)` [Verified]

- **`tl.dot`:** sm_75 gives 0 mma and FFMA only, for fp16, bf16 and fp32. sm_80 and sm_86 give `mma.sync m16n8k16`.
- **1.4.7 decode split kernel (Qwen3.8 geometry):** fits at 32,832 B on a Q8 cache and 49,152 B on fp16. It uses 255 registers with stack spills, and FFMA only.
- **1.4.7 prefill tiles:** 98,304 B (fp16), 82,176 B (Q8 direct, tile 64x32 with 8 warps), 98,560 B (Q4). All exceed 65,536 B, and on sm_75 `num_stages` does not shrink them. A 32x32 tile with 4 warps is exactly 65,536 B. Upstream's sm_75 tile (16, 32, 8 warps, 2 stages) is 49,152 B with no stack.
- **fla 0.5.2 GDN chunk kernels:** all six compile for sm_75. `chunk_fwd_kernel_o` fits 2 of its 3 configs.

### 4.5 Build constraints found [Verified]

- **CUDA version.** torch cu128 only builds extensions with a CUDA 12.x nvcc, because `torch/utils/cpp_extension.py:523-524` raises on a major-version mismatch. The desktop's nvcc 13.4 cannot be used.
- **Compiler.** The desktop's nvcc 12.5 rejects the default gcc 15.2 and needs `CUDAHOSTCXX=g++-13` (installed).
- **Build tools.** The desktop has no `ninja` and no `cmake`.
- **Choice:** build any patched variant **on the laptop, CPU-only**, as an sm_75-only wheel, and copy only the wheel to the desktop.

### 4.6 What compile probes cannot tell us

- Numerical correctness and speed.
- Runtime shared-memory aborts. Those are inferred from code plus the documented `cudaFuncSetAttribute` behaviour.
- Linking and loading the full `.so`. Only object files were built, though the official wheel sidesteps this.
- The v1.6.5 kernels added after 1.4.7 have not been audited for sm_75 resource use: sparse head, parity-pool scan replay, the 16-row GDN b/a GEMV, the Triton conv1d, and graph kernels falling back to eager mode. This is critic gap #5 and is handled in E0/E2.

### 4.7 Evidence files

These are in the session scratchpad, which is temporary. Copy them into `research/turing/` if they need to persist.

- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/priorart/ptxprobe/probe.cu`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/compile_probe/sm75_summary.txt`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/compile_probe/per_tu_table.tsv`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/compile_probe/sm75_probe_shim.patch`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/compile_probe/sm75_shim64_summary.txt`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/compile_probe/run_tu.sh`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/ext_audit/poc/turing_poc.diff`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/gdn_attn/tprobe/exl3_probe_sm75.txt`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/gdn_attn/tprobe/fla_probe_sm75.txt`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/gdn_attn/tprobe/dot_probe_out.txt`
- `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/priorart/raw/rafatxf_turing.md`
- Upstream clones: `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/upstream/exllamav3` and `/tmp/claude-1000/-home-somebody-local-llms/7a0b57c8-d6dd-4d91-a0c1-0358f48f5f24/scratchpad/turing/upstream/exllamav2`

---

## 5. Model path and VRAM/bandwidth budget

### 5.1 The model [Verified]

`/opt/p8-user/models/qwen38-exl3-2.0`:

- **Layers.** 64 layers: 48 GDN and 16 full-attention (every fourth layer).
- **Full attention.** 24 query heads, 4 KV heads, head_dim 256.
- **GDN layers.** 16 key heads, 48 value heads of dimension 128, conv kernel 4, fp32 state in the config. One MTP layer. Vocabulary 248,320.
- **Quantization.** EXL3 format version 1.4.2, 2.0 bpw, `mul1` codebook, 3-bit lm_head, 2-bit MTP.
- **Bit-width mix.** 401 EXL3 tensors: 334 at K=2, 39 at K=3, 14 at K=1, 14 at K=4.
  - The K=3 tensors include the lm_head and 38 body projections.
  - Trunk k/v projections are 3-bit or 4-bit; every MTP linear is 2-bit.
  - Trellis byte shares: K=2 84.2%, K=3 13.1%, K=1 2.2%, K=4 0.6% of 6.559 GB.
- **Stored size.** 10,195,824,708 B in total:

| Component | Size | Where it lives |
|---|---|---|
| Decoder layers | 6.148 GB | GPU |
| Embedding (bf16) | 2.543 GB | **Host RAM** (`lab:exllamav3/modules/embedding.py:37-39`, `prefer_cpu`; upstream `embedding.py:101`) |
| Vision tower | 0.921 GB | **Not loaded** unless vision is enabled |
| lm_head | 0.477 GB | GPU |
| MTP | 0.106 GB | GPU |

GPU-resident weights are therefore about **6.73 GB (6,420 MiB)**, not 10.2 GB.

### 5.2 Which kernels run on stock v1.6.5 on sm_75

**[Inference from code reading; to be confirmed by the E2 dispatch census.]**

| Phase | Rows per call | What runs | Turing concern |
|---|---|---|---|
| Undrafted decode | 1 | K=2 tensors: fp16 HMMA GEMV. K=4: GEMV, depending on shape. Wide K=3 (incl. lm_head) and all K=1: cooperative GEMM (int8 declined). GDN q/k/v+z bundle and k/v pairs: cooperative multi-GEMM | ~30% of weight bytes on the GEMM with no copy/compute overlap |
| MTP draft step | 1 | MTP layer (2-bit), then either the 3-bit lm_head on the GEMM or v1.6.5's sparse-head on-device draft loop (47d42fe) | Latency-bound; full-head cost is ~5% of a trunk pass per draft token |
| MTP verify | d+1 (2-8) | GEMV path for K=2-4 at m ≤ 8; GEMM for K=1 and the fused bundles; GDN writes one state per token | 2-row int8 would abort on 1.4.7; v1.6.5 declines int8 entirely |
| Prefill tails / mid-size verify | 9-144 | Cooperative GEMM | No overlap |
| Prefill chunks | > 144 | Reconstruct to fp16 + cuBLAS (`COMPUTE_32F`) (`exllamav3/modules/quant/exl3.py:10,140`) | Half-rate accumulate on GeForce |
| Full-attention prefill | — | SDPA or `fa75` over a dequantized fp16 window | Solved upstream |
| Full-attention decode / verify | q_len 1-8 | Triton split/combine reading the packed Q8 cache, FFMA, rows padded to 16 | Grows with context |
| GDN decode | — | exllamav3's own CUDA kernels in a CUDA-graph block; fp16 two-row state pool | Compiles unchanged |
| GDN prefill (≥ 48 tokens) | — | Vendored fla Triton chunk kernels (FFMA) with #411 configs | Slow, but tuned upstream |

### 5.3 VRAM budget on the 2080 Ti

**[Verified]** The desktop reports 11,264 MiB total, 447 MiB driver-reserved, 379 MiB used (Sunshine holds 160 MiB) and **10,440 MiB free**.

**[Inference]** Budget at 32K context, Q8 cache, MTP, batch 1, vision off, v1.6.5:

| Item | MiB | Basis |
|---|---|---|
| Trunk EXL3 weights | 5,863 | Safetensors header sum |
| lm_head (3-bit) | 455 | Header |
| MTP layer | 101.5 | Header |
| GDN b/a transposed copies | ~45 | Code reading |
| Q8 KV cache, 17 attention layers (16 trunk + 1 MTP) | 1,156 | 2,176 B per token per layer (`exllamav3/cache/quant.py:39-42`) |
| GDN recurrent state, fp16 two-row pool, batch 1 | 144 | `v1.6.5:exllamav3/modules/gated_delta_net.py:287-291`. In 1.4.7 this was fp32 with (draft+1) planes: 720 MiB at draft 4, ~1,017 MiB at draft 6 |
| Load-time overhead (laptop-measured on 1.4.7) | ~481 | `testsuite/paper5/tabby-p5.yml` header |
| **At load** | **~8,250** | |
| Growth after a ~30K-deep prompt (laptop-measured) | ~+830 | `EFFICIENCY-CAMPAIGN.md` E28 |
| **In use at 30K depth** | **~9,080** | Leaves ~1.3 GiB |

*Correction (2026-10-31 21:24 EDT): the ~481 MiB load-time overhead row is a residual inferred from the `tabby-p5.yml` header's 9,074 MiB total, not a measurement. Found while drafting Volume I.5, chapter 6; the row is kept as written.*

Items not in that budget, so expect less headroom (realistic range about 0.8-1.4 GiB, unmeasured):

- **[Critic]** The v1.6.5 sparse head mirrors fp16 unembedding columns plus embedding rows at about 20 KiB per token id, for example ~320 MiB at 16K ids.
- The Turing SDPA/fa75 staging window (~128 MiB at 32K in 1.4.7's pool-sized scratch).
- The cuBLAS workspace.
- Differences in CUDA context size on Turing.

Rules that follow from this:

- `max_batch_size` must be 1. TabbyAPI defaults to 4 for recurrent models (`labtabby:backends/exllamav3/model.py:358`).
- 48K is tight at about 9.66 GiB.
- 64K with a Q8 cache does not fit.

### 5.4 Bandwidth and compute budget

- **[Verified spec]** 2080 Ti memory bandwidth: 616 GB/s (352-bit, 14 Gbps).
- **[Inference]** Bytes read per decoded token:

| Component | Bytes per token |
|---|---|
| Weights | ~6.63 GB |
| GDN fp16 state read+write | ~0.15 GB |
| Q8 KV | 34,816 B × context: ~0.14 GB at 4K, ~1.14 GB at 32K |

The hard ceiling is about **88-90 tok/s at short context** and about **76-78 tok/s at 32K**.

Nobody gets near that ceiling:

- **[Verified lab data]** The 4080 Laptop reached 29 tok/s, about 47% of its own 432 GB/s ceiling, while power-capped at 80 W.
- **[Reported]** rafatxf's fork at 4.0 bpw implies about 490 GB/s of effective weight streaming (26.6 ms weight time), and vcruz305 reports a ~38 tok/s roofline at 531 GB/s. At 4.0 bpw, Turing therefore sits near both its compute and bandwidth limits.
- **[Reported, medium confidence]** rafatxf states "fewer bits do not decode faster on Turing (the EXL3 trellis decode is compute-bound there)". That rests on one Gemma 4 31B comparison on his fork: 27.7 vs 28.4 tok/s at 3.5 vs 4.0 bpw (turing.md:76, 386-388).
- **[Inference]** If trellis decode costs the same per weight regardless of bpw, 2.0 bpw buys VRAM fit more than speed, and undrafted decode tops out around 35-45 tok/s without a custom kernel.

That is the single most important unknown. The roofline probe (E3) and the bpw sweep (E7) settle it.

### 5.5 llama.cpp fit on the same card

**[Design parse of GGUF headers, not independently verified]** GPU-resident sizes:

| GGUF | GPU-resident | Fits on 11 GB? |
|---|---|---|
| UD-IQ2_XXS | 6.84 GB | Yes |
| UD-IQ2_S | 7.94 GB | Yes |
| UD-Q2_K_XL | 9.40 GB | Undrafted, short context only |
| UD-IQ3_XXS | 10.51 GB | No; would need partial offload |

**[Verified lab data]** UD-IQ2_S plus MTP needed 10,230 MiB at 32K on the laptop. So the GGUF drafted lanes probably run at 16-24K (MTP) or 4-6K (DFlash2) on the 2080 Ti. A fair comparison therefore matches context at 16K and reports EXL3's longer context separately.

---

## 6. Recommended strategy

### 6.1 Decision

The judge compared three designs and recommended **"alternative-route"**:

- install the official upstream wheel (no port);
- apply fixes in three layers (environment variables, then a Python overlay, then at most three CUDA patches), each added only if profiling justifies it;
- build a matched llama.cpp baseline;
- decide with a paired accuracy-per-minute shootout.

It beat the other two:

- **"Minimal-shim"** was the best-grounded design and was a close second. It never measures llama.cpp or task quality, so it cannot answer the question that matters.
- **"Performance-first"**, a custom Turing GEMV aiming at 60-70 tok/s undrafted, rests on unverified microarchitecture assumptions. It projects nearly twice the best public Turing result and needs about 37 days.

This plan grafts the best parts of both onto the recommended design:

- From minimal-shim: CPU-only laptop builds, isolated per-change A/B tests, the 11 GB budget, and the int8 change done in two steps.
- From performance-first: the roofline probe, the MTP cycle model, per-kernel profiling, and the cubin-diff check before each PR.

The critic's gaps are folded in throughout.

### 6.2 Choices already made (revertible defaults)

1. **Base.** Official exllamav3 **v1.6.5 cp312 cu128/torch2.10 wheel** in a **uv-managed Python 3.12** venv on the desktop. This exact wheel is verified to carry sm_75, and it matches the laptop's `exl3-env`. The desktop only has Python 3.14, and the cp314 torch/exllamav3 combination is unverified, so it is the fallback.
2. **No port of 1.4.7. No lab patches carried over** for the first runs. Measure with upstream `eval/perf.py` and `eval/spec_decode.py`. Port the H32/H33 hooks later only if the 4080 comparison needs the lab's acceptance definitions (critic #15).
3. **Serving.** Upstream TabbyAPI at ≥ b1c2253 in a separate directory on the desktop. The lab TabbyAPI stays untouched.
4. **Builds.** Every patched variant is built **on the laptop, CPU-only**: `TORCH_CUDA_ARCH_LIST=7.5 CUDA_VISIBLE_DEVICES='' nice -n 19 MAX_JOBS=4`, nvcc 12.6, gcc 13. Only the wheel goes to the desktop.
   - Any laptop tool that imports exllamav3 must set `CUDA_VISIBLE_DEVICES=''`, because the vendored fla queries device capability at import.
   - Keep compile jobs at 4 or fewer and watch memory, because Paper 5's load guard pauses grading on CPU pressure or MemAvailable < 6 GiB (critic #14).
5. **llama.cpp baseline.** Build the lab's `v0.4.0-mtpshared` fork (needed for the stripped 0.28 GB MTP sidecar) and current mainline. Cross-build for sm_75 on the laptop CPU (`-DCMAKE_CUDA_ARCHITECTURES=75`, never `native`) and ship the CUDA 12.6 runtime libraries alongside. Add ik_llama.cpp with a public IQ2_KT quant as a third lane (critic #3).
6. **Numerics.** Run at upstream defaults (fp16 GDN state) for performance numbers. Pin `recurrent_dtype` to fp32 for any comparison with the 4080's 1.4.7 numbers.
7. **Operational guards** (free-VRAM floor, GPU-idle check, no foreign compute apps or NVENC sessions) are **adjustable environment defaults**. Runs yield to Matt's processes. They never ask him to stop his own work, and no threshold is hash-bound.
   - The go/no-go numbers below are pre-registered **analysis** criteria for choosing which path to pursue. They do not block launching anything.
8. **Clocks.** Report stock behaviour at 260 W with clocks, temperature, power and throttle reasons logged at 1 Hz. No clock locking.

### 6.3 Approval packet: things only Matt can do

These should be sent as one packet with exact commands, a disk budget and a rollback step (critic #13).

1. Create `~/turing-lab` on the desktop: the uv Python 3.12 download, the venv, torch 2.10 cu128, the exllamav3 1.6.5 wheel, and the upstream TabbyAPI checkout.
2. Copy models: qwen38-exl3-2.0 (10.2 GB) plus the GGUFs (UD-IQ2_XXS 7.27 GB, UD-IQ2_S 8.37 GB, MTP sidecar 0.28 GB, DFlash2 1.14 GB). That is about 27-29 GB, with 65 GB free on a disk that is 86% full.
3. Optional downloads for the bpw sweep and the ik_llama.cpp lane. **[Critic]** turboderp/Qwen3.8-27B-exl3 has branches from 1.40 to 3.50 bpw, including SC 2.00; Thireus IQ2_KT is 7.18 GB.
4. GPU windows on the desktop while his own jobs are idle.
5. Optional `sudo` for `ncu` hardware counters (`RmProfilingAdminOnly: 1`). Without it the plan uses `nsys` only.
6. Any public GitHub comment, issue or PR.

### 6.4 Ordered work items

| # | Work item | What it involves | Effort (days) | Gate |
|---|---|---|---|---|
| W0 | Approval packet + run manifest | Commands, disk budget, rollback. Manifest template: wheel sha256, commits, torch/triton/driver/CUDA versions, model hashes, `nvidia-smi -q`, env vars | 0.5 | — |
| W1 | Desktop environment + static audit of v1.6.5 on sm_75 | Install per 6.2. `cuobjdump --list-elf` and `--dump-resource-usage` on the wheel `.so`: every sm_75 kernel's dynamic + static shared memory ≤ 64 KB, spills listed. CPU-only Triton compiles of the 1.6.5 decode and conv1d kernels at Qwen3.8 geometry. Record `l2CacheSize` | 1.5 | E0 |
| W2 | Baseline engines | llama.cpp (lab fork + mainline) and ik_llama.cpp, sm_75 cross-builds | 1.5 | — |
| W3 | Two-engine harness | One OpenAI-compatible streaming client for both engines; single stream; prefix cache off; TTFT, decode and end-to-end time; same Qwen tokenizer, chat template and reasoning effort; truncation recorded. Interleaved ABBA order. Fresh `XDG_CACHE_HOME` per run, because the sparse head persists token stats and benchmarks itself at init (critic #6) | 1.5 | — |
| W4 | Correctness and numerics tooling | (a) Per-shape kernel tests against an fp32 dequant reference for every model shape, K=1-4, m=1-8. (b) A same-card oracle: force every linear through reconstruct + cuBLAS fp32-accumulate (`AUTO_RECONSTRUCT_THRESHOLD=0`, a one-line change at `exllamav3/modules/quant/exl3.py:10`), then teacher-forced KL (mean, p99, max) and top-1 agreement. (c) Greedy-MTP vs greedy-undrafted divergence rate. (d) inf/NaN and max-activation hooks on every layer output over a 32K prefill. (e) Passkey plus a small RULER subset at 8K/16K/32K | 2 | Used by E2-E9 |
| W5 | Turing roofline probe | Standalone CUDA binaries, cross-built on the laptop. Measures the streaming-read ceiling; L2-resident trellis-decode throughput (weights/clock/SM) for K=1-4 at m=1/4/8 using upstream's int8 and fp16 GEMV cores; HMMA fp16- vs fp32-accumulate rate; HFMA2/HMMA interference | 2 | E3 |
| W6 | L0: environment-variable dispatch | `EXL3_GEMV=2`; unfusing via `EXL3_MGEMM_N_THRESHOLD=0` + `EXL3_MGEMM_K_THRESHOLD=9` (and/or `EXL3_QKV_SLICE=0`); MTP draft length 1/2/3/4/dynamic. **Leave `EXL3_INT8_GEMV` at its default**: setting it to 0 makes the thresholds unused and fuses everything (`exllamav3/model/config.py:28-35`) | 1 | E4 |
| W7 | L2a: int8 fit gate (sm_75 wheel) | Step 1, about 15 host lines in `ext/quant/exl3_gemv_int8.cu`: on devices < 80 KB, allow only the M=1 plain-mode kernel after checking that its shared memory plus static fits; decline cleanly otherwise, with no `exit()`. On sm_75, send K=3 to the narrow unit. Step 2, only if step 1 passes: a host+device shared constant for the M=2 and coop row caps (`ext/quant/exl3_gemv_int8.cu:271` and `ext/quant/exl3_gemv_int8_kernel.cuh:1150`) | 3-4 | E5 |
| W8 | Small levers (Python + one sm_75 wheel) | (a) A Turing branch in `exl3_gemv_cfg`: narrow up to n=12288, wide above. (b) Turing-aware `use_mgemm` so unfusing becomes the default. (c) Decode-attention row floor of 8 on sm_75 (`exllamav3/modules/attention_fn/triton_paged.py:1151-1163`), precompiled CPU-only and required to cut FFMA count by ≥ 35% with no spill growth. (d) `AUTO_RECONSTRUCT_THRESHOLD` sweep. (e) H_ACC at `__CUDA_ARCH__==750`, only if the coop GEMM is still ≥ 10% of token time. (f) `CUBLAS_COMPUTE_16F` for reconstruct prefill, gated on major==7 plus a rate probe rewritten with m16n8k8, with an environment opt-out | 3 | E6 |
| W9 | bpw sweep | turboderp quants at about 1.6/1.8/2.0/2.2/2.5 bpw at short context, plus the lab quant vs SC 2.00 on KL, perplexity and a small eval | 1 (+ downloads) | E7 |
| W10 | Shootout + write-up | Paired accuracy-per-minute comparison; report published for Matt | 3 | E9 |
| W11 | Conditional kernels | (a) Register-staged GEMM prefetch under `EXL3_SM75 && !USE_ROCM`, only if the GEMM is still ≥ 10% of decode after W6-W8. (b) A CUDA flash-decode kernel over the Q8 cache (m16n8k8, 6-way GQA packing, AOT cubins, ≥ 2x68 split blocks), only if attention is ≥ 15% of token time at the target context | 4 + 5-7 | E10 |
| W12 | Upstream PRs | See section 8 | 3 | After the E9 numbers |

**Effort.** The core path (W0-W10, W12) is about 24 working days. Conditional kernels add 9-11. Matt's time is the approvals only.

### 6.5 Desktop experiments and go/no-go criteria

| Exp | What | GO | NO-GO / fallback |
|---|---|---|---|
| **E0** Readiness | Run W1 checks. Guard: no foreign compute apps or NVENC sessions; record free VRAM | sm_75 cubins present; every sm_75 kernel within 64 KB of shared memory or explicitly flagged; llama.cpp binary starts | Missing sm_75 in the chosen wheel: switch Python version. A kernel over 64 KB that is not declined at runtime: file upstream |
| **E1** GGUF bar | `llama-bench -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3 -d 0,8192,32768` for IQ2_XXS and IQ2_S. Servers for IQ2_S + MTP (n-max 2/3, 8K/16K) and IQ2_XXS + DFlash2 (4K/6K). Sweep `-ub/-b`, MMQ vs cuBLAS, CUDA graphs and n-max so llama.cpp gets equal tuning effort | Reference only | If best undrafted < 30 tok/s, explain it (clocks, offload) before going on, because a low bar would flatter EXL3 |
| **E2** Stock EXL3 bring-up + correctness + census | Upstream kernel tests (`tests/kernels/gemm/test_exl3_gemm.py`, `test_exl3_gemv.py`, `test_exl3_gemv_tail.py`, `attention/test_fa75.py`, `recurrent/test_gdn_gemv.py`, `tests/e2e/test_speculative.py`, `tests/modules/linear/test_sparse_head.py`). Load at 32K + Q8 + MTP, batch 1, sparse-head capacity set explicitly. W4 oracle. `nsys` dispatch census: kernel name per linear at m=1, 2, 7; every eager fallback logged | Tests pass; measured headroom ≥ 0.5 GiB at 30K depth; KL against the same-card oracle mean ≤ 2e-4 with top-1 agreement ≥ 99% (larger is treated as a kernel bug); perplexity within 1% of the 4080 value with fp32 state pinned (sanity check only); undrafted ≥ 0.5x the GGUF undrafted bar | OOM at 8K, garbage output, or failing kernel tests: file an upstream issue with logs and recommend the llama.cpp lane |
| **E3** Roofline probe | W5 | A trellis-decode core reaching ≥ 1.25x the DRAM-required rate at the measured sustained clock means a custom GEMV is worth considering later | Below 1.1x: decode is compute-bound on Turing. Do not build a custom GEMV; publish the compute floor |
| **E4** Env-var dispatch | W6 grid at 2K and 32K | Keep a setting if decode is ≥ 5% faster at 2K, with ≤ 2% regression at 32K or in prefill, and KL ≤ 2e-4 | Keep stock dispatch and record the result |
| **E5** int8 fit gate | Per-shape tests at K=1-4, m=1 against fp32; m=2 and coop-shape calls must decline cleanly; `compute-sanitizer --tool racecheck` and `synccheck`; 1,000 repeat launches; then an end-to-end A/B on the E4 winner; 200-request MTP soak | Error within the fp16 GEMV's envelope; no hazards; decode ≥ +10% (projected +15%); no aborts | K=3 fails (the #404 symptom): keep K=3 on the fp16 GEMV and retest K=1/2/4. K=2 fails: drop int8 on Turing and publish the result |
| **E6** Small levers | One A/B per W8 lever | ≥ 3-5% on the lever's target metric; teacher-forced KL ≤ 1e-4 for fp16-accumulate levers; no inf/NaN over a 32K prefill (all layers, including down_proj with k=17408); passkey at 32K unchanged | Drop the lever |
| **E7** bpw sweep | W9. Decode tok/s and kernel time per weight vs bpw | No threshold. Either outcome tests rafatxf's "fewer bits do not decode faster" claim and is publishable. Use whichever quant gives better KL/eval at equal VRAM in E9 | — |
| **E8** Long context + robustness | 2K/8K/16K/24K/30K-deep prompts at 32K Q8 + MTP; fp16 vs fp32 state and fa75 vs SDPA on passkey/RULER; 60-minute mixed soak | No OOM; ≥ 300 MiB spare at peak; passkey/RULER equal to the fp32-state and SDPA references within noise; tok/s drift < 5% from the first window to the last | Drop to 24K and record the limit; if fp16 state fails long-context tests, default to fp32 state and re-budget VRAM |
| **E9** Accuracy-per-minute shootout (the decision) | Best EXL3 configuration vs the best fitting GGUF configurations at matched 16K, on the lab eval kit's 245 paired items (HumanEval+ 20, math25, GSM8K-100, MMLU-100), temperature 1.0, effort medium, 4,096 max tokens, **several samples per item**, truncation reported | EXL3 becomes the 2080 Ti lane if its pooled accuracy-per-minute point estimate is ≥ 1.0x the best GGUF lane and the 95% paired-bootstrap interval rules out a loss larger than 10% | Otherwise llama.cpp is the 2080 Ti lane. EXL3 work continues only as upstream PRs and a paper data point |
| **E10** Conditional kernels | W11 microbenchmarks and end-to-end | Prefetch: ≥ 15% faster kernel for m ≤ 16, bit-identical to the sync-copy path, ≤ 16 B spill growth, ≥ 3% end-to-end. Flash-decode: ≥ 1.5x vs Triton at 32K, q_len 8 cost ≤ 1.2x q_len 1, error no worse than Triton's | Abandon the kernel |

The judge dropped the alternative-route design's fixed "0.8x speed" pivot: it was based on median token ratios, and the mean ratio is about 0.88. E9 decides instead.

**MTP model to fit from E2/E4 data (graft from performance-first).** Cycle time T_cycle(d) = T_verify(d+1) + d × T_draft. Measure T_verify(m)/T1, T_draft and per-position acceptance on code, math and prose. Remove per-step host syncs such as `draft_conf.float().cpu()` where the v1.6.5 loop still has them.

---

## 7. Risks and open questions

### 7.1 Top risks

1. **Trellis decode may be compute-bound on Turing** (F7, medium confidence). If so, 2.0 bpw tops out around 35-45 tok/s undrafted whatever the dispatch fixes. EXL3's case then rests on context length and token efficiency. E3 and E7 test this early.
2. **int8 GEMV correctness on compute capability < 8.** #404 measured int8 kernels giving wrong output on a V100 at K=3/5. vcruz305's sm_75 "no garbage" report has no proof of race freedom. E5 tests this with the sanitizers before any speed claim.
3. **VRAM margin.** Free memory is 10,440 MiB, not the earlier 10,885 figure, which was refuted. Overheads specific to Turing and v1.6.5 (sparse head, staging window, context) are unmeasured. Measured in E2 and E8.
4. **The quality dividend is fragile.** It comes from n=20-25 single laptop runs, and the Turing build changes numerics (fp16 GDN state, k8 split, optional fp16 accumulation). Re-measured in E9 with several samples per item and bootstrap intervals.
5. **Desktop access and contention.** Every step on the desktop is Matt's approval. Sunshine and his projects can perturb clocks and VRAM. The sustained clock at 260 W is unknown. rafatxf's reference numbers come from a tuned 240 W, 1,650 MHz card, and he estimates stock clocks are 10-15% slower.
6. **Upstream churn and overlap.**
   - **[Critic]** exllamav3 `dev` moves daily (#574 and #576 merged on 2026-10-29), and v1.6.x is still labelled experimental on Turing.
   - rafatxf may send the same levers.
   - Triton officially supports only compute capability 8.0+, so torch and triton must stay pinned at 2.10 and 3.6.
7. **Prefill is a structural loss.** About 600-800 tok/s for EXL3 against a projected 900-1,200 for llama.cpp. COMPUTE_16F narrows the gap by 10-15% at most.

### 7.2 The critic's gaps and how this plan handles them

| Critic gap | Handled by |
|---|---|
| P0 #1: no correctness reference that isolates the hardware | W4 same-card oracle (reconstruct + fp32 accumulate), per-shape fp32 tests, greedy-MTP divergence. Cross-GPU logit comparison and the k8-split dispute are deferred until Paper 5 frees the laptop GPU; until then, no cross-GPU equivalence claim |
| P0 #2: long-context fp16 numerics untested | W4(d,e) and E8: all-layer inf/NaN hooks, passkey/RULER, fp16 vs fp32 state, fa75 vs SDPA; every fp16 lever is gated on long-context tests |
| P0 #3: llama.cpp comparison under-specified | W3 shared client; E1 tuning sweep; mainline plus ik_llama.cpp lanes; E9 multi-sample bootstrap; exclusions table below |
| P0 #4: bpw sweep wrongly ruled out; quant choice unexamined | W9/E7 using turboderp's branches; lab quant vs SC 2.00. Optional: HermiHg's optimized MTP heads for acceptance |
| P0 #5: v1.6.5 sm_75 runtime path not audited | W1 resource-usage audit and Triton compiles; E2 upstream tests including speculative and sparse-head tests; eager-fallback logging |
| P0 #6: VRAM budget missing consumers | E2 measured dry run (`torch.cuda.memory_stats` + per-process `nvidia-smi`); explicit sparse-head capacity; fresh cache directory per run |
| P1 #7: stock baseline undetermined (17.9 vs 31-33 tok/s) | E2 measurement plus the `nsys` dispatch census before any patch. **[Verified]** The SDPA-fallback decode bug fixed in 9f811ba is already in v1.5.2, so it does not explain that gap; int8 GEMV and clocks remain candidates |
| P1 #8: sources not searched | One dated search pass each, logging negatives: Jia et al. 2019 (arXiv 1903.07486, Turing T4 microbenchmarks); Yan et al. IPDPS 2020 (tensor-core rates); verify Fasi et al. 2021; Turing whitepaper; CUDA guide throughput table for compute capability 7.5; QTIP, QuIP#, AQLM, FLUTE, LUT-GEMM, Marlin for related work; exllama Discord; Bilibili/Zhihu/V2EX/NGA; Hugging Face discussions; llama.cpp 2080 Ti threads |
| P1 #9: single-source or from-memory claims | Quadro/Tesla full-rate fp32 accumulate: check specs, or measure on a free Kaggle/Colab T4 (also a second Turing data point). L2 size: read in E0. int8 correctness: E5 sanitizers. Laptop-to-Turing scaling: E3 |
| P1 #10: measurement protocol | 1 Hz clock/power/throttle logging, warmups, ABBA, stated n and confidence intervals, NVENC sessions in the idle check. **[Critic]** desktop facts for the methods section: PCIe x8 (gen 3 max), display inactive, i9-7900X with 20 threads, 75 GB RAM; the embedding is on the CPU in both engines, so host-to-device traffic per token is symmetric |
| P1 #11: licensing | Table in section 8.4 |
| P1 #12: upstream mechanics | Section 8.3: PRs target `dev`, rebase and re-validate on the `dev` tip before each PR, no repository-wide formatter run |
| P2 #13: owner access scattered | Section 6.3 consolidated packet |
| P2 #14: Paper 5 interference | 6.2 item 4 |
| P2 #15: scope and reproducibility | Claims limited to a stock GeForce TU102 with 11 GB; manifest (W0); hypotheses and analysis plan stated up front (this section and 6.5); acceptance-definition comparability checked before any MTP comparison with the 4080; watch #417, rafatxf's fork and exllamav3 `dev` |

**Excluded runtimes** (critic #3):

- **vLLM, SGLang, TensorRT-LLM, MLC, Aphrodite.** Their INT4 paths for a 27B model need about 14 GB, which does not fit 11 GB **[Inference]**.
- **The weicj vLLM and TeksEdge figures.** Labelled non-comparable: 22 GB mods or two GPUs, synthetic or unverified inputs.

### 7.3 Open questions

- What does stock v1.6.5 actually decode at on a stock 2080 Ti for this model? No upstream-only baseline exists after v1.5.3.
- Is 2.0 bpw decode compute-bound or bandwidth-bound on TU102 (E3, E7)?
- Does the 1.4.2-format quant load and match quality on v1.6.5? No format-version gate was found in the loader, but this has not been tested at runtime.
- Is the int8 dp4a GEMV correct and faster than the fp16 HMMA GEMV on TU102?
- How much do fp16 GDN state and fa75's fp16 P·V accumulation cost at 32K?
- Are upstream `spec_decode.py` acceptance figures comparable with the lab's H32/H33 definitions?
- Is the DFlash2 lane available for EXL3 on this card? #417 measured DFlash2 at 81.8 tok/s on a 22 GB card. The lab has `/opt/p8-user/models/qwen38-dflash2` as a GGUF; whether an EXL3-format drafter exists or can be converted is unchecked. If not, it is explicitly out of scope.
- Will rafatxf send the decode PRs himself? Ask on #417 first.

---

## 8. Why it matters, and how to contribute upstream

### 8.1 Lifespan of older GPUs

- **[Verified]** Turing cards (RTX 20-series from 2018, Tesla T4, Quadro RTX) are still everywhere. T4s are the free GPUs on Colab and Kaggle.
- **[Verified]** The demand is visible: at least five independent ports or wheel builds appeared before the merge, and people run 27B models on modded 22 GB 2080 Tis.
- **[Inference]** A stock 11 GB 2080 Ti that serves a capable 27B reasoning model at 32K context with drafting, at accuracy-per-minute parity with GGUF, adds years to a large installed base. Below Turing, Volta and Pascal EXL3 are still open (#404, #475, #509, #512) and need a non-tensor-core or dp4a decode path. That is a natural follow-on for a "longer life for older GPUs" paper.

### 8.2 What the lab can add

The basic port is done, so it is not a contribution. What is new:

1. The first measured numbers for Qwen3.8-27B at about 2.0 bpw plus MTP on a **stock 11 GB** 2080 Ti, including VRAM fit at 32K.
2. A bpw sweep on Turing that directly tests "fewer bits do not decode faster".
3. The fit-gated int8 GEMV on Turing, which upstream declines and rafatxf's fork did not attempt.
4. A same-card accuracy-per-minute comparison with llama.cpp and ik_llama.cpp.
5. Small, measured dispatch fixes, credited to rafatxf where they originate.

### 8.3 How to contribute

- **Coordinate first.** Comment on #417 with stock-card numbers and ask rafatxf whether he is opening the GEMV-rule, unfusing, H_ACC or COMPUTE_16F pieces. **[Critic]** #417 and his #423 are still open, and he has opened no decode PRs.
- **One isolated PR per lever, smallest first**, each with stock 11 GB 2080 Ti before/after numbers and KL/perplexity figures:
  1. Docs PR to `doc/env_vars.md` (or an issue) with the Turing environment settings and numbers.
  2. A Turing fusion/dispatch policy in `exllamav3/model/config.py`.
  3. A Turing branch in `exl3_gemv_cfg` (`ext/quant/exl3_gemv.cu`), about 10 lines.
  4. The M=1 int8 fit gate (`ext/quant/exl3_gemv_int8.cu`), about 15 host lines.
  5. The decode row floor on sm_75 (`exllamav3/modules/attention_fn/triton_paged.py`).
  6. H_ACC on sm_75 and COMPUTE_16F with a runtime rate probe. This is the hardest to land, because Quadro/Tesla Turing may not benefit.
- **Hygiene before each PR:**
  - Show with `cuobjdump` diffs that sm_80+ cubins are unchanged.
  - Confirm the ROCm build still compiles, because Turing and ROCm share fallbacks at `v1.6.5:ext/ptx.cuh:248,277,295`.
  - Add tests under `tests/kernels/`, including bit-exact row-invariance tests for any new multi-row GEMV.
- **[Critic] Branch and style:**
  - Target `dev`, not `master`. Measure on v1.6.5, then rebase and re-validate on the `dev` tip.
  - exllamav3 configures no formatter, so match the surrounding style and do not reformat unrelated lines. The lab rule "run the formatter before PRs" applies where a repository configures one.
- **Disclosure and approval.** Disclose AI-assisted authorship and avoid needless @-mentions. Matt approves every public post (lab policy).

### 8.4 Licensing [Critic, via the GitHub API]

| Source | License | Use |
|---|---|---|
| turboderp-org/exllamav3 | MIT | Base |
| rafatxf/exllamav3 (turing) | MIT | Ideas and code may be reused with MIT notice and attribution |
| weicj/exllamav3-turing | MIT | Reference |
| ik_llama.cpp | MIT | Baseline lane |
| llama.cpp (`fattn` design) | MIT | Any derived code keeps the notice |
| vcruz305/exl3-turing | **No license** | Cite ideas only; do not copy patches, including its int8 GEMV patch, unless he adds a license |

---

## Appendix: independent checks by the session after the workflow (2026-10-30)

- GitHub API: turboderp-org/exllamav3 PR #325 "Turing (sm_75) support" (rluisr), merged 2026-09-26T01:14:57Z; theroyallab/tabbyAPI PR #463 (vcruz305), merged 2026-09-26T20:36:27Z.
- Release notes: v1.5.2 (2026-09-26) "Experimental Turing support"; v1.6.1 (2026-10-10) "Turing (sm_75) optimizations"; latest v1.6.5 (2026-10-25).
- `cuobjdump --list-elf` on the downloaded v1.6.5 cp312 wheel (sha256 prefix e184693383192e1b): 169 cubins each for sm_75, 80, 86, 89, 90, 100, 120.
- Workflow: 18 agents (6 survey tracks, 6 adversarial verifiers, 3 designs, judge, critic, report); full structured evidence kept in the session scratchpad (turing_result.json).

## The publication hold (added 2026-10-31 21:50 EDT)

Matt, 2026-10-30: patches to other people's code bases (exllamav3, TabbyAPI, llama.cpp and others) are not pushed to public forks or proposed upstream until the paper that reports them is published. This note's line 109 and section 8.3 plan upstream pull requests. They were written before the rule (this file was committed at 00:01 PDT on 30 October, the rule recorded at 17:20 PDT) and are on hold under it. Credits for the lab's Turing patches are in `tools/turing/CREDITS.md`.

*Correction (2026-10-31 22:14 EDT): section 5.2's "the verify pass writes one state per token" describes exllamav3 1.4.7. Version 1.6.5 keeps two state rows and replays the last pass's inputs on a rejected draft (`scan_replay_kernel_128`). Found while drafting Volume I.5, chapter 14.*

*Correction (2026-11-01 22:45 EST): section 5.3's "64K with a Q8 cache does not fit" is refuted on the 2080 Ti. The pre-registered stage 1
(`research/prereg/2026-11-01-p8-longctx-stage1.md`; results in `testsuite/results/turing/longctx/stage1/report.md`) served a
filled 64K context with the 8-bit cache on the adopted w13 build with MTP at a peak of 10,097 MiB of 11,264, leaving
1,167 MiB free. The longest step with at least 512 MiB headroom was 72K, and the 4-bit cache passed every step to 136K.
The 5.3 budget was derived before any 2080 Ti run and subtracted overheads that the measured peaks already contain. The
original line is kept.*
