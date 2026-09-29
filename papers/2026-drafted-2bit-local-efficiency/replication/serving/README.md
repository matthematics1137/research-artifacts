# serving/ — the recommended way to run Qwen3.8-27B on this laptop

Outcome of the efficiency campaign (`testsuite/results/EFFICIENCY-CAMPAIGN.md`, E1-E45, 2026-09-06 → 09-09).
Nothing here touches the saved TabbyAPI default (`engines/tabbyAPI/config.yml`) or the paper-2 pinned engines;
the launcher starts a separate server on its own port with the lab's patched engine copy.

| file | what |
|---|---|
| `launch.sh daily` | single user: EXL3 SC_2.00 + its own MTP head, 1 slot, 32K context, draft length 6, 16 GB recurrent stash. Measured 2026-09-09: up in 10 s, 9,074 MiB resident at load, 9,828 MiB after an ~8K-token prompt; 47-50 tok/s on code/math, ~35 on prose. |
| `launch.sh 2agents` | two concurrent agents: 2 drafted slots, 24K each. Each keeps ~34 tok/s (vs 51 alone); three agents queue. Measured 2026-09-09: 9,202 MiB at load, 9,922 MiB after one ~8K-token prompt, ~10.7 GB with two 10K conversations (E39). |
| `../testsuite/results/efficiency/tools/smoke_serving.sh` | the 15-minute launcher smoke test (both modes: preflight, health, a short and an ~8K-token request, resident VRAM, clean release); last run 2026-09-09 14:07, both modes pass |
| `tabby-daily.yml`, `tabby-2agents.yml` | the two configs, each line annotated with the experiment that set it |
| `logit_bias_marker_penalty_-0.5.json` | optional per-request `logit_bias` (E45): 49 re-verification marker tokens at −0.5 → ~5% fewer reasoning tokens, ~10% less wall, no accuracy change at n=100. Pass as `extra_body` / `--extra-body-file`. |
| `traffic_stats.py` | prompt/cached/generated-token shape from TabbyAPI's per-request Metrics log lines (lengths only, never content); tells you whether the stash and the checkpoint interval are earning their keep |

## Why these settings (one line each, details in the campaign document)
- `max_batch_size: 1` (E28): TabbyAPI pre-allocates (draft length + 1) fp32 recurrent states **per slot** for a drafted
  recurrent model: 728 MiB per slot at draft 4, about 1,020 MiB at this config's draft 6; the default of 4 slots is
  what made 2.20 bpw + MTP OOM at 12K.
- MTP drafting (E17, E35-E37, E47): accuracy-lossless on 1,000 paired items at temperature 0 (GSM8K-500 97.6% vs
  97.2% undrafted, MMLU-500 86.8% vs 86.2%, McNemar p ≥ 0.5) — but not byte-identical: greedy outputs diverge at
  about one argmax near-tie per 400-600 tokens, and the answer almost never changes. ×1.6-1.8 on code/math, ×1.2-1.5
  prose/MMLU; the prose ceiling is the draft head's acceptance, not the verifier.
- draft length 6 (E44): 1-4% better than 4 on prose/math/code; 2 is 5-9% worse everywhere.
- `sysmem_recurrent_cache: 16384` (E30): a repeated 4K system prompt costs 0.9 s instead of 9 s; ~300 MB host RAM per
  live prefix.
- `EXL3_CKPT_PP=1024` via the launcher (E43, lab patch): editing a long prompt re-prefills only from the last 1K-token
  checkpoint instead of from scratch; +2-3% cold prefill, +148 MiB host RAM per checkpoint.
- chunk 1024, Q8 cache: the measured knee for prefill speed vs VRAM at 32K on this card (E28, E40).

## Engine
`engines/exl3-1.4.7-lab/` is a copy of the exllamav3 1.4.7 wheel with the lab patches applied
(`testsuite/results/efficiency/tools/exl3_all_lab_patches.patch`: checkpoint-interval knob, CPU Cholesky fallback for
the converter, and the two measurement-only speculative-decoding hooks that stay off unless their env vars are set).
It is gitignored (1.3 GB); rebuild it from the wheel plus the patch file if it is missing. The launcher's
`--stock-engine` flag runs the pinned exl3-env engine instead (no checkpoint knob).

## Not enabled by default
Enabling this config as the saved TabbyAPI default, the `clock_policy_test.sh` clock experiment (sudo), and any newer
exllamav3 wheel or DFlash drafter download are Matt's calls (see "Items for Matt" in the campaign document).
