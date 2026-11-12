# Pre-registration: P8, the 8-bit cache read by Triton at 64K (accuracy arm)

Registered 2026-11-03 22:11 EST by the stages 2-3 fork, at the ambassador's request. **Not queued.** It runs only if the logit check
(`research/prereg/2026-11-03-p8-fd75d-check.md`) finds fd75d diverging from Triton at 48K or 64K, and only after the
ambassador writes its gate record. The first commit of this file freezes it; later changes are dated amendments
appended at the end.

## Question

At 64K, how much of the split cache's lead over the 8-bit cache (H2s, +11.5 points) comes from the decode kernel
(fd75d against Triton), and how much from the value bits?

## Arm

- **Format.** The 8-bit cache (`8,8`) at 64K, read by **Triton** (`EXL3_FD75D=0`).
- **Items.** The 96 graded 64K items (`graded-64k.jsonl`: multivalue at k = 12, 96 items, SHA-256 589696ca...).
- **Same as stage 2's 64K arms otherwise:**
  - the w13 build and stage 2 environment;
  - TabbyAPI through `stage2_run.py`, one fresh server;
  - `cache_size` = `max_seq_len` = 65,536; chunk 1024; `max_batch_size` 1;
  - MTP with 6 dynamic drafts; `sysmem_recurrent_cache` 2048;
  - thinking off, temperature 0, 768 output tokens.
- **The only difference** from stage 2's `s2-64k-88` arm is `EXL3_FD75D=0`.
- **Runner change.** The runner gains a per-arm environment field to carry it. That is an implementation change
  only, tested before the arm runs.
- **Budget:** about 3 GPU hours (96 items at about 115 s); at most 3.5.

## Comparisons

Paired on the same 96 items. The statistics are those of stages 2-3: a 90% percentile bootstrap over items (10,000
resamples, seed 20261102), usable meaning the lower bound is above -10, equivalent meaning the whole interval lies
within ±5, and exact McNemar.
- **K1 (primary):** `8,8`-Triton minus `8,8`-fd75d (stage 2's `s2-64k-88`). This is the kernel's effect on accuracy
  at 64K.
- **K2:** `8,4`-Triton (stage 2's `s2-64k-84`) minus `8,8`-Triton. This is the value bits' effect with the kernel held
  fixed.
- **Noise.** Same-arm discordance stays stage 2's noise check (0/16).

## Predictions (before any data; meaningful only if the logit check diverged)

- **K1:** the point estimate is above 0, with Triton reading the 8-bit cache more accurately than fd75d at 64K.
- **K2:** equivalent. Four-bit values cost nothing measurable at 64K once the kernel is the same.

## Rules

- **If K1's interval lies above +5,** fd75d loses accuracy at 64K. P8's claims at depth must then carry that, and
  fd75d needs a fix or a length limit; that is a follow-up, not done here. Its length limit would be judged from the
  logit check's lengths.
- **If K1 is equivalent,** the kernel is not what separates the arms, and H2s's lead is the value bits plus noise.
- **K2 is reported either way.**

## Safeguards

As in stage 2:
- one GPU job at a time through the queue, with the guard on, at 260 W, without sudo;
- the 20 GiB cap and the 10 GiB system-RAM floor;
- no retry after an out-of-memory error;
- a 25-minute stop for a hung request, with the item excluded.
