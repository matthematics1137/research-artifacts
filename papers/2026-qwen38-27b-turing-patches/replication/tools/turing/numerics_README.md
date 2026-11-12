# Turing numerics check (plan W4(b)): `numerics.py`

This tool answers one question for Paper 8: **does exllamav3 1.6.5's kernel dispatch on the RTX 2080 Ti (sm_75) change
Qwen3.8-27B's next-token distribution, measured independently of task accuracy?**

It loads the model once and runs fixed token streams through it in up to three dispatch modes. It then compares the
next-token distribution at every position. The relevant plan sections are
`research/TURING_EXL3_RESEARCH_2026-10-30.md` §6.4 W4(b) and §6.5 E2. The E2 analysis criterion is KL mean ≤ 2e-4 with
top-1 agreement ≥ 99% against the same-card oracle; a larger value is treated as a kernel bug. The script applies it
to `unfused||ref` (the linear kernels alone, see below), reports it but never uses it to block anything.

## The three modes

| Mode | Decode blocks | Linear kernels |
|---|---|---|
| `default` | Upstream as served: the graph-captured C++ blocks for attention (`BC_Attention`), the gated delta net (`BC_GatedDeltaNetSplit`) and the MLP (`BC_GatedMLP`), plus the fused multi-matrix `exl3_mgemm` bundles | Whatever the wheel dispatches on this GPU (GEMV/GEMM, and the mgemm kernels inside the bundles) |
| `unfused` | Upstream's Python decode paths: the ones `EXL3_BC_ATTN=0` and `EXL3_BC_GDN=0` select (Triton paged attention, the torch GDN conv and scan). Every linear is called on its own. The GDN qkv projection comes out in fp32, as in the C++ block (`--gdn-proj`, below) | The wheel's per-linear GEMV/GEMM kernels |
| `ref` | Same as `unfused` | **Oracle.** The trellis is reconstructed to fp16 weights (exactly the values every kernel sees). The product is a cuBLAS `GemmEx` with `CUBLAS_COMPUTE_32F` (`ext.hgemm`). The output Hadamard and scale run in fp32, then the result is cast to the dtype the module requested |

Every pass prefills its prompt in one fixed mode (`--prefill-mode`, default `default`) and switches to its own mode
for the scored forwards, so all modes start scoring from the same cache and recurrent state (reference-knob item 2).

The KL comparisons, per position and stored in both directions:

- **`unfused||ref`** is the E2 measure: the EXL3 linear kernels against the oracle. Attention, the GDN conv and scan,
  the norms and every intermediate dtype are shared, so only the linear kernels differ.
- **`default||unfused`** is the effect of the fused decode blocks: the C++ attention, GDN and MLP blocks, graph capture,
  and the multi-matrix `exl3_mgemm` bundles. The mgemm kernels run only in `default`, so an error inside them shows up
  here and not in `unfused||ref`.
- **`default||ref`** is end-to-end: the served dispatch against the oracle. It is not a pure kernel test. `ref` runs
  upstream's eager attention and torch GDN paths, not the C++ blocks, so this pair also contains the blocks' own
  arithmetic against the eager paths.
- **`default||default_repeat`**, with `--repeat-default`, measures the run-to-run floor.
- **`default_s1||default`**, when `--step` > 1, is the effect of verify-shaped batching. `--step 5` gives the MTP
  draft-4 verify shape, which addresses E47's open question about which forward causes the divergence (H38c).

## The reference knob: what v1.6.5 actually does

These findings come from reading the source in `engines/src-mirror/turboderp-org__exllamav3.git` at tag `v1.6.5`.

1. **`AUTO_RECONSTRUCT_THRESHOLD`** (`exllamav3/modules/quant/exl3.py:10`, value 144) is read inside
   `LinearEXL3.forward` on every call. That same function also honours a per-call `params["reconstruct"]`. Either
   knob forces reconstruct, but **only for linears that go through the Python `LinearEXL3.forward`**.
2. **At decode shapes most linears never reach that function.** Three paths call the EXL3 GEMV/MGEMM kernels directly:
   - `GatedMLP.bc.run_bszN` for ≤ 8 rows (`modules/mlp.py:868`);
   - the attention `BC_Attention` graph for q_len ≤ 16 (`modules/attn.py:1135`);
   - `BC_GatedDeltaNetSplit` (`modules/gated_delta_net.py:1304`).

   The fused bundles also bypass it for ≤ 32 rows: `multi_gu`, `multi_qkv`, `multi_qg`, `multi_kv` and `multi_qkvz`.
   So **setting the threshold to 0 would leave the decode path almost untouched.**

   **Prefill differs between modes too.** A prefill chunk of up to 144 rows runs the EXL3 GEMM kernel through
   `LinearEXL3.forward` (and ≤ 32 or ≤ 16 rows the fused bundles or C++ blocks). Larger chunks take upstream's
   reconstruct path (`hgemm_recon`, whose output is fp16). In `ref` mode every chunk would go through the oracle.
   Most prompts in the E47 item set are 144 tokens or shorter (median about 80), so if each pass prefilled in its own
   mode, the prompt state would already differ before the first scored token. That prefill-shape error would then
   show up in every position's KL, including the `[0,64)` bucket. The script therefore prefills every pass in one
   fixed mode, `--prefill-mode default` by default (the served path, so the greedy stream still matches serving), and
   switches to the pass's mode for the scored forwards. The comparison then covers exactly the scored decode-shape
   forwards. Two other settings are available:
   - `--prefill-mode ref` prefills through the oracle (the greedy stream then no longer matches serving);
   - `--prefill-mode own` restores per-mode prefill, as an end-to-end variant.
3. **The GDN qkv projection dtype.** Upstream's torch GDN path asks for the qkv projection in fp16 (`_proj_dtype`,
   `modules/gated_delta_net.py:26`, from `EXL3_GDN_PROJ_FP32`). The C++ GDN block (`s_qkv`, `gated_delta_net.py:1186`)
   and the sliced qkvz bundle (`project_qkvz_sliced`) produce it in fp32. Both then round it to bf16 for the conv. In
   `unfused` and `ref` the script sets `_proj_dtype` to fp32. The global is read at call time, and `default` restores
   it. This has two effects: the oracle is never coarser than the served path, and `unfused||ref` differs only in the
   linear kernels. `--gdn-proj upstream` keeps upstream's setting in both modes. The attention projections (fp16) and
   the MLP intermediates (`interm_dtype`) already match on both paths.
4. **`hgemm_recon`, the GEMM inside upstream's reconstruct path, is not always fp32-accumulate.** It first tries
   `hgemm_f16acc` (`exllamav3_ext/hgemm_f16acc.cu`), which accumulates 32-term partial sums in fp16. That kernel is
   disabled for compute capability < 8, so on the 2080 Ti `hgemm_recon` is plain cuBLAS `COMPUTE_32F`. On Ada the
   kernel switches on automatically, and the on/off decision is cached per process. The oracle therefore calls
   `ext.hgemm` directly, which is always cuBLAS `COMPUTE_32F`, so the same reference holds on both GPUs.
5. **What the script does.** Nothing in the installed wheel is edited. Everything happens in-process and is undone
   when the mode switches back:
   - set `exllamav3.modules.attn._bc_attn_enable` to False;
   - set the fused bundles and `GatedMLP.bc` to None;
   - set `GatedDeltaNet.bc_split` to False;
   - set `exllamav3.modules.gated_delta_net._proj_dtype` to fp32 (item 3);
   - replace `LinearEXL3.forward` with the oracle.

   Each item starts from a fresh recurrent state and cache position, so modes can switch between passes. Within a pass
   the mode switches once, after the prefill.
6. **Coverage probe.** Before the timed items start, the script runs one decode forward per mode and counts the
   oracle calls. Every EXL3 linear on the text path must be called exactly once. That is 401 linears for this model:
   48 GDN layers × 3, 16 attention layers × 4, 64 MLPs × 3, plus the head. If any is missing, the run stops; use
   `--allow-incomplete-ref` to override. The same count for `default` mode is the dispatch census: how many linears
   still run through Python. Both are saved in `census.json`.

## Token streams

- **The E47 rows hold no full response text.** Each row stores only `raw_answer_excerpt` (the last 200 characters) and
  a sha256 hash.
- **Prompts.** These are rebuilt from the eval datasets by `id`, with the same strings as `run_eval.build_prompt`.
  They are rendered with the model's `chat_template.jinja` (one user turn, generation prompt, `reasoning_effort`
  medium as in E47). For effort medium that is plain ChatML ending in `<think>\n`, with no system turn.
- **Responses.** By default the stream is the model's own greedy continuation in `default` mode at one token per
  forward. That pass is also the `default` measurement. Rows that carry `response_ids` or `response_text` are
  teacher-forced as given.
- **Reuse.** Every run writes `<tag>.streams.jsonl`, which records who generated each stream (host, GPU, tag, label).
  `--streams` on a later run, for example on the laptop's Ada GPU once Paper 5 frees it, feeds exactly the same token
  ids.
- **Item selection.** `--num-items` (default 50) takes the first N items, interleaved across the `--rows` files: 25
  GSM8K plus 25 MMLU from the two E47 files. Both E47 labels list the same ids in dataset order.
- **Scoring.** `--max-response-tokens` (default 256) positions are scored per item, starting at the last prompt token.
  The E47 medians are 285 tokens (GSM8K) and 558 (MMLU).
- **Target log-probs are selection-biased on a self-generated stream.** A generated stream is the generating mode's
  own argmax. On noise alone, that mode gives the stream tokens a higher log-prob than any other mode does (a
  winner's curse). A review scratch test with equal noise on both sides found a mean difference of +4.8e-4 nats,
  about 1.5 times the KL mean. Modes that share its kernels (`unfused`, `default_repeat`) inherit part of this bias.
  The summary therefore computes NLL and target log-prob differences only on streams that no mode on the compared
  machine(s) generated, for example the desktop's streams replayed on the laptop. On a self-generated run they are
  `null`, and `target_lp_scope` gives the counts. KL, flip rates and top-1 agreement are symmetric and use every
  position. A self-generated run can still give unbiased NLL: replay its streams on another machine.
- **Excerpt check.** With `--gen-max-tokens 4096`, generation continues to EOS, unscored past the scored part. When an
  answer completes, its visible tail is compared with the row's `raw_answer_excerpt`. For a cross-GPU greedy-identity
  check, use the **undrafted** rows (`EFF-h38a_nodraft`), because the drafted rows already differ from undrafted greedy
  output on 5/25 GSM8K and 17/25 MMLU of these items.

## Setup on the desktop

The validating session copies the files in. Everything stays inside `$L`.

```bash
L=/opt/p8
scp tools/turing/numerics.py p8-host:$L/tools/numerics.py
ssh p8-host mkdir -p $L/evals/EFF-h38a_mtp $L/evals/EFF-h38a_nodraft $L/evals/datasets
scp testsuite/evals/results/EFF-h38a_mtp/{gsm8k_500,mmlu_500}.jsonl p8-host:$L/evals/EFF-h38a_mtp/
scp testsuite/evals/results/EFF-h38a_nodraft/{gsm8k_500,mmlu_500}.jsonl p8-host:$L/evals/EFF-h38a_nodraft/
scp testsuite/evals/datasets/{gsm8k_500,mmlu_500}.jsonl p8-host:$L/evals/datasets/
```

Keep each rows file named after its suite (`gsm8k_500.jsonl`): the suite, and with it the prompt format, is taken
from the file name.

## Running (on the desktop, through the capped launcher)

GPU work runs only through `desktop_run.sh`, one job at a time, by an agent cleared for the desktop GPU. The launcher's
`cmd` mode sets only `CUDA_VISIBLE_DEVICES` and `XDG_CACHE_HOME`. The script itself defaults the Triton, HF and pip
caches into `$L/cache`.

```bash
L=/opt/p8
bash $L/tools/desktop_run.sh guard
R="--rows $L/evals/EFF-h38a_mtp/gsm8k_500.jsonl $L/evals/EFF-h38a_mtp/mmlu_500.jsonl"

# 1. smoke: 2 items x 32 positions; checks the probe, the three modes and the outputs (a few minutes incl. load)
bash $L/tools/desktop_run.sh cmd numerics-smoke $L/venv/bin/python $L/tools/numerics.py \
  -m $L/models/qwen38-exl3-2.0 -cq 8,8 $R --limit 2 --max-response-tokens 32 --full-logits-stride 16 --tag w4b-smoke

# 2. the W4(b) run: 50 items x 256 positions, plus the run-to-run floor, plus exact logits on every 16th position
bash $L/tools/desktop_run.sh cmd numerics-w4b $L/venv/bin/python $L/tools/numerics.py \
  -m $L/models/qwen38-exl3-2.0 -cq 8,8 $R --repeat-default --full-logits-stride 16 --tag w4b-e47-n50

# after a guard stop: the same command plus --resume (finished items are kept as shards)
# optional: MTP verify shape   --step 5 --tag w4b-e47-n50-step5
# optional: fp32 GDN state     -rs32   (pins the plan's 6.2 item 6 setting)
# optional: end-to-end prefill --prefill-mode own --tag w4b-e47-n50-ownprefill
```

Logs go to `$L/logs/run-<label>.log`, with one JSON line per item showing per-mode seconds, `kl_unfused_ref`,
`kl_default_ref` and an ETA. The `-cq 8,8` flag (Q8 KV, as in serving) applies to every mode alike. The cache size is
chosen automatically to fit the longest stream; override it with `-cs`.

**Cost.**
- Default mode: about 0.04 s per position (24 tok/s undrafted in the smoke runs).
- Unfused mode: somewhat slower.
- The oracle reconstructs every weight on every forward, about 54 GB of fp16 traffic per position. Expect roughly
  0.3-0.6 s per position; this has **not been measured yet**, so the smoke run's `s_per_position` is the number to plan
  from. With `--prefill-mode default` the prompt prefill never goes through the oracle.
- At 12,800 positions the full run is therefore likely **1.5-2.5 h of GPU**, almost all of it in the oracle.
- VRAM: the model plus a small cache, plus one 320 MiB reconstruct buffer (the largest head slice, 5120 × 32768
  fp16). This should stay well under the smoke runs' 10.0 GB.
- **Full logits (`--full-logits-stride K`).** Each kept row is the whole vocabulary in fp16, about 0.5 MB. Before
  loading the model, the script prints the projected size:

  | Run | Rows per mode | Modes | Size |
  |---|---|---|---|
  | W4(b) at stride 16 | 800 | 4 (or 5 with `--step` > 1) | about 1.5-1.9 GiB of RAM when the run is combined, roughly twice that on disk (shards plus run file; fp16 logits barely compress) |
  | Stride 1 | 12,800 | 5 | about 30 GiB |

  The run refuses a projection above 6 GiB or above the free disk. At stride 1 it would otherwise be OOM-killed under
  the 20 GiB cap after all the GPU work. `finish_run` loads the shards in two passes and keeps one copy of the full
  logits.

## Outputs (`$L/runs/numerics/`)

| File | Content |
|---|---|
| `<tag>/run.json`, `<tag>/census.json` | Configuration (incl. `prefill_mode`, `gdn_proj`, `xref`), versions, GPU state, dispatch facts (int8 GEMV mode and its K cap on this GPU, f16acc status, the GDN projection dtype), coverage probe |
| `<tag>/item_NNNNN.npz` | Per-item shards, written as each item finishes; these are what `--resume` keeps |
| `<tag>.npz` | Everything per position, described below |
| `<tag>.streams.jsonl` | `prompt_ids`, `response_ids` and `stream_generator` per item, for `--streams` |
| `summary.jsonl` | One row per run, described below |

**`<tag>.npz` per-label arrays** (`default`, `unfused`, `ref`, ...):
- `<label>__top_ids` / `__top_lp`: the top-20 token ids and float64 log-probs, sorted with ties broken by id;
- `__tail_lp`: the log of the probability mass outside the top-20, summed directly;
- `__target_lp`: the stream token's log-prob;
- `__entropy`, `__lse`, `__top1`, `__margin` (top-2 gap in nats).

**`<tag>.npz` per-pair arrays:** `kl__<a>__<b>` and the reverse direction.

**`<tag>.npz` cross-run arrays** (in a run with a reference, see below): `xkl__<R>__<L>` and `xklr__<R>__<L>`, the
lower bounds of KL(R||L) and KL(L||R) for reference label R and local label L, plus `xtop1__<R>` and `xmargin__<R>`.
Positions the reference does not cover are NaN (top-1 -1).

**`<tag>.npz` shared arrays:** `targets`, `pos_in_response`, `item_ids`, `item_offsets`, the token streams with their
offsets, and `meta_json`. With `--full-logits-stride K`, the npz also holds full fp16 logits for every K-th position
(`full__<label>`, `full_index`), which gives an exact cross-device KL on that sample.

**`summary.jsonl` row:**
- per pair:
  - KL mean / p50 / p90 / p99 / p99.9 / max, and the reverse-KL mean;
  - top-1 agreement;
  - flip rate where the reference's top-2 margin exceeds 0.1, 0.5 and 1.0 nats (counts included), also measured by
    either side's margin;
  - target log-prob difference, only on streams no compared mode generated (see "Token streams");
  - KL by response position, in buckets `[0,64)`, `[64,256)`, ...;
  - the five worst items;
- per label: NLL (same restriction as the target log-prob difference) and entropy;
- `e2_analysis_criterion`: the E2 verdict on `unfused||ref` (analysis only);
- `end_to_end`: the same thresholds on `default||ref`;
- `fused_blocks`: `default||unfused`;
- `cross_run`: present in a run with a reference;
- `target_lp_scope`: which positions the target log-prob statistics cover;
- timing per mode;
- the excerpt-match count.

**Reading it.**
- If E2 (`unfused||ref`) is above the criterion, the per-linear kernels are at fault. By the plan's rule that is a
  kernel bug.
- If E2 is met but `default||unfused` is large, the fused blocks are at fault: the C++ attention, GDN or MLP block,
  graph capture, or the mgemm bundles. A one-mode run of each block (`EXL3_BC_ATTN=0` or `EXL3_BC_GDN=0` in `default`)
  narrows it down.
- `default||ref` is the end-to-end number for the paper. It is not a kernel-bug test, because `ref`'s eager attention
  and GDN paths are not a more precise copy of the C++ blocks, just a different implementation.
- Compare everything against `default||default_repeat`, the nondeterminism floor.
- The logits are fp16 in every mode, because the head's output dtype is fp16. Rounding the logits to fp16 alone can
  produce KL on the order of 1e-5 between near-identical computations. Treat that as part of the floor.

## Cross-device comparison

The laptop run replays the desktop's streams:

```bash
TURING_LAB=<laptop lab folder> python numerics.py -m <laptop EXL3 2.0 model> -cq 8,8 \
  --streams <desktop runs>/w4b-e47-n50.streams.jsonl --repeat-default --full-logits-stride 16 --tag w4b-e47-n50-ada
```

The comparison then comes from three numbers, tightest first:

1. **Exact KL on a sample.** `--compare` reports `full_kl` when both runs kept full logits at the same positions
   (the stride-16 rows above).
2. **A bound at every position, computed in-process.** A `--streams` run also loads the reference run file beside the
   streams file (`w4b-e47-n50.npz`; `--xref auto`, or name it with `--xref PATH`). It reads only the reference's top-k
   arrays, never its full logits. It then scores every local mode against the reference labels (`--xref-labels`,
   default `default,ref`) at every position:
   - This run's log-probs are exact for every token, and the reference's top-k log-probs and tail mass are stored
     exactly.
   - So the KL of both distributions coarse-grained onto {reference top-k tokens, rest} is a **lower bound** of the
     exact KL in both directions. It is tight when the reference's tail mass is small.
   - It goes in the run's `cross_run` summary and the `xkl__` arrays.
3. **A bound from two stored top-k lists.** `--compare A.npz B.npz --label-a default --label-b default` lines items
   up by id and requires identical prompt and response tokens. It reports `topk_kl_lower_bound`: the KL on {tokens in
   both top-k lists, rest}, where every probability is known on both sides. This is again never above the exact KL,
   but it can be far below it when the lists differ (0 when they are disjoint).
   - Earlier versions reported a coarse top-20 KL that filled B's missing log-probs with B's 20th log-prob instead.
     That estimate is not a bound in either direction. In a review scratch test, fully disjoint top-20 sets gave 0.0
     against an exact 27.0 nats, and half overlap gave 0.023 against 13.4.
   - `--compare` loads both run files whole, including their full logits: at stride 16, about 2 GiB each.

`--compare` also reports top-1 agreement and flip rates. Its NLL and target log-prob differences follow the same
selection-bias rule: positions whose stream was generated on either compared host are left out. Both runs on the
desktop's streams therefore give `null` there, and the laptop run's own summary has unbiased NLL for its modes.

## Other actions (CPU only, numpy only, no model)

```bash
python numerics.py --selftest                                   # metric, bound, flip, combine, compare and prompt checks
python numerics.py --combine TAG --out-dir DIR                  # rebuild <tag>.npz + a summary row from the shards
python numerics.py --compare A.npz B.npz --label-a default --label-b default
```

## Status

- **Checks done (2026-10-30, after the review fixes).**
  - `py_compile` on the laptop with `CUDA_VISIBLE_DEVICES=''`.
  - `--selftest` in the sandbox (analysis-env, numpy): 33/33 passed. The checks cover:
    - KL against an independent logaddexp formula, and exact zero KL for identical logits;
    - top-k order and ties, the tail mass, and flip accounting;
    - E2 on `unfused||ref`;
    - target log-probs left out on self-generated streams, and stream provenance;
    - the cross-run bound: never above the exact KL, equal to it when k = V, and close to it on disjoint top-k sets,
      where the overlap bound is 0;
    - shard combining and the two-pass full-logit fill;
    - compare;
    - the full-logit RAM projection;
    - the prompt strings.
  - A file-level check in the sandbox on a synthetic desktop run plus a laptop replay with `xref`: 9/9 passed. It
    covers `finish_run` (full logits loaded in two passes), the streams-file provenance, `load_xref`, the in-process
    bound (never above the exact KL at any position; 0.8 of it on that synthetic data) and `--compare` with
    `full_kl`.
  - File-destruction word scan (house rule 2) over both files: clean. The script only appends, and never writes over
    an existing result file: a repeated `--combine` writes `<tag>.c2.npz`, and so on.
- **Not run on any GPU yet.** Reaching `--prefill-mode` and the `_proj_dtype` switch needs the model, so the smoke run
  is their first test.
