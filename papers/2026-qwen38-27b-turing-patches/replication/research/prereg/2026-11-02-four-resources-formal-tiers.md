# Pre-registration: four resources formalized (C2), part 3: store instead of recompute, composed across storage tiers, and the laptop's NVMe and RAM read rates

Registered 2026-11-02 02:56 EST by the C2 CPU researcher of the ideas program. The first commit of this file freezes it. Later
changes are dated amendments appended at the end.

- Gate: `research/gates/2026-11-02-ideas-program.md`, cluster C2, and its addendum on Matt's checkpointing message
  (2026-11-02 01:18 EST).
- Prior art: `research/prior-art-four-resources-formal-2026-11-02.md`, V9 and O7 as narrowed. Choosing between
  storing, recomputing and offloading is solved for chains (rotor, pofo, checkpoint_schedules). What stays open is
  the measured tier rates on these machines.
- C4 owns persisting and reusing state. This part only composes the exchange rate and measures what the laptop's
  CPU can measure.
- CPU only. No GPU, no model loaded, the desktop untouched.

## Matt's question, in the frame's terms

Matt, 2026-11-02: "storage is very cheap so we can always store lots and lots of information outside of the model
that doesnt need to be recomputed".
- Storing the processed state of a context and reading it back instead of prefilling again is a conversion: bytes at
  a storage tier for operations (prefill), priced in seconds.
- Its exchange rate is
  rho_k(P) = (T_prefill(P) - T_restore,k(P)) / S(P), in seconds of GPU prefill saved per GiB stored at tier k.
- Restoring pays from the break-even length P*_k onward.

## Composition (declared)

- **State size.** S(fmt, P) = s(fmt) x P/1024 + 72 MiB + 2.8 MiB, where:
  - s(fmt) is the cache arithmetic of part 2 (16 attention layers plus MTP);
  - 72 MiB is the Gated DeltaNet state of one sequence in fp16 (48 layers x 48 heads x 128 x 128 x 2 bytes);
  - 2.8 MiB is the convolution state (48 layers x 10,240 channels x 3 taps x 2 bytes).
- **Recompute.** T_prefill(P) = P x (alpha + beta(fmt) x P/2), part 2's PF1 fitted on P8 stage 1.
- **Restore.** T_restore,k(P) = c0 + S(P) / BW_k. c0 is an engine overhead, not measured; predictions are given for
  c0 = 0.05 s and 0.5 s.
- **Tiers:**
  - host RAM to the 2080 Ti over PCIe 3.0 x16, pinned: 12 GB/s (assumed, typical pinned host-to-device rate; not
    measured on this card);
  - the laptop's NVMe (WD PC SN740, 2 TB): **measured below**;
  - a SATA SSD: 0.5 GB/s (assumed);
  - a hard disk, the "16 TB drive" of Matt's 2026-11-02 note: 0.15 GB/s (assumed).
- **A disclosed arithmetic fact, not a prediction.** At 64K, the 8-bit cache holds 36.125 x 64 = 2,312 MiB (2.26
  GiB), plus 75 MiB of recurrent and convolution state. The gate addendum's "about 1.2 GB of 8-bit cache" is the
  4-bit size (19.125 x 64 = 1,224 MiB). Stage 1's measured slopes (37.5-47.5 MiB per 1K for `8,8`) confirm the
  larger figure.

## Measurement (CPU only, laptop)

- **NVMe sequential read.**
  - Python, with `os.O_DIRECT` (bypassing the page cache) and a page-aligned 8 MiB buffer.
  - Three existing GGUF files (Qwen3.8-27B UD-Q2_K_XL, UD-IQ3_XXS, UD-Q4_K_XL), 2 GiB each from offset 1 GiB, then a
    second 2 GiB from offset 4 GiB.
  - Run under `ionice -c3` and `nice -n 19`.
  - Reported: GB/s per read, and the median.
- **RAM copy.** numpy `copyto` of a 1 GiB float64 array into another, 5 repeats, single thread, under nice 19.
  Reported: GB/s copied per read side (1 GiB over the time), median.
- **Nothing is written to disk** except the small result JSON. No file is created or removed in the models folder.

## Predictions

| Id | Claim | Predicted |
|---|---|---|
| T1 | laptop NVMe, O_DIRECT, 8 MiB sequential, median | 2.5-5.2 GB/s (rated 5,150 MB/s) |
| T2 | laptop RAM copy, one thread, median | 8-25 GB/s |
| S1 | break-even P* for every tier | at most 500 tokens at c0 = 0.05 s; at most 1,000 at c0 = 0.5 s |
| S2 | rho at 64K, `8,8`, every tier including the hard disk | at least 40 s of prefill per GiB |
| S3 | rho at 128K, `4,4`, NVMe and host RAM | at least 100 s per GiB |
| S4 | rho as P grows, every format | rises (superlinear prefill against linear storage) |

- T1 and T2 are measured here. A T1 median outside its range is reported as refuted.
- S1-S4 are composed predictions. They are judged only by the GPU design below, or by C4's own measurement.
  Computing them here proves only that the composition says so.

## GPU design (not run; for C4 or a later queue step)

Restore against prefill on the 2080 Ti, adopted w13 build:
- **Arms:** `8,8` and `4,4` at P = 4K, 16K and 64K, plus `4,4` at 128K.
- **Each arm:**
  1. Prefill and time it.
  2. Snapshot the cache pages plus the recurrent and convolution state to pinned host RAM, timing device to host.
  3. Write the snapshot with O_DIRECT and read it back, timing both.
  4. Copy host to device, timing it.
  5. Decode 64 greedy tokens from the restored state, and again from a fresh prefill.
  6. Compare the first-token logits (maximum absolute difference, KL) and the match length of the two continuations.
     The engine is not bit-reproducible (research-ledger-reference), so a same-arm repeat sets the reference.
- **Checks against S1-S4:**
  - the restore time is within 2x of S/BW + c0;
  - P* is below 1,000 tokens;
  - the continuation from the restored state is as close to a fresh prefill as two fresh prefills are to each other.
- **Budget:** about 20 GPU-minutes of compute (two prefills per arm dominate: about 2 x 113 s at 64K and 2 x 273 s at
  128K), at most 45 with loading and margin, at 260 W, through the desktop queue only.

## Software and outputs

- `engines/analysis-env` (numpy 2.3.3), with CUDA_VISIBLE_DEVICES="".
- Code: `tools/ideas/four-resources-formal/tiers.py` (`measure` and `compose`).
- Results: `testsuite/results/ideas/four-resources-formal/tiers.json` and the report.
