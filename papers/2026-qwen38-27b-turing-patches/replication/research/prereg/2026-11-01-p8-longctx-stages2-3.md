# Pre-registration: P8 long context on the 2080 Ti, stages 2-3: accuracy at depth per KV cache format

Registered 2026-11-01 22:49 EST by the ambassador. The first commit of this file freezes it. Later changes are dated amendments
appended at the end. It replaces stages 2-3 of the gate fork's draft
(`research/prereg/2026-11-01-p8-64k-q4-cache-DRAFT.md`) and makes the four changes required at the end of the stage 1
registration.

- Gate: `research/gates/2026-11-01-p8-64k-q4-cache.md`. RIFE: research-ledger-reference; stage 1 fact
  research-ledger-reference.
- Stage 1: `research/prereg/2026-11-01-p8-longctx-stage1.md` and `testsuite/results/turing/longctx/stage1/report.md`.

## Question

How much accuracy does each KV cache format cost the 2-bit 27B at depth on the 2080 Ti? Above all, is the 4-bit cache
usable at 64K, where both it and the deployed 8-bit cache fit? If so, how does its accuracy fall with length beyond
the 8-bit cache's 72K reach?

## Numbers already seen (disclosure)

- **Stage 1** (all of it):
  - reach: the 8-bit cache 72K; the 4-bit cache at least 136K; split at least 96K; fp16 at least 40K;
  - peaks, and prefill times (51 s at 32K, 108-114 s at 64K, 187 s at 96K, 271 s at 128K);
  - one shallow needle per step, answered correctly at every step.
- **P6 at 16K** (llama.cpp, laptop), by family for the 2-bit 27B: single, multikey and multivalue 50/50 in every cell;
  vartrack 4-12/50.
- **Prior art:** rafatxf's needles (25/25 at 32K-250K, Q4 cache, 4.0 bpw), Minima-KV and "Length, Not Bit Width", as
  summarized in `research/prior-art-p8-64k-q4-cache-2026-11-01.md`.
- **Not seen:** any accuracy of exllamav3's quantized caches beyond the stage 1 needles, and any calibration result.

## Common method

- **Machine and engine** as in stage 1: the desktop 2080 Ti through the desktop queue only, at 260 W, with the desktop
  guard on. The adopted w13 build, unchanged. TabbyAPI via the stage 1 launcher (Amendment 1 of stage 1), one fresh
  server per arm:
  - `cache_size` = `max_seq_len` = L;
  - `max_batch_size` 1, chunk 1024;
  - MTP on, with 6 dynamic drafts;
  - sparse draft head off, vision off;
  - `sysmem_recurrent_cache` 2048, and the allocator as in stage 1.
- **Lengths.** L = 32,768 or 65,536 in stage 2; 98,304 and 131,072 in stage 3. Every prompt is at most L - 512 tokens,
  counted with the model's tokenizer, which leaves room for the answer.
- **Requests** as in P6: thinking off, temperature 0, max 128 tokens. Every item is a fresh prompt, with no prefix
  reuse.
- **Scoring:** P6's `score_longctx`, unchanged, giving 0 or 1 per item.
- **Items.** A wrapper imports `testsuite/paper6/make_longctx.py` unchanged and overrides only module constants:
  - `TARGET_TOKENS`, with `TOL_LO` and `TOL_HI` recomputed;
  - `N_NEEDLES` for the graded families;
  - `SEED`.
  - The generator's own self-test runs unmodified at P6's settings; the wrapper adds its own checks for the overridden
    settings. Those checks cover needle count, distinct keys and names, chain order, gold answers, and token count
    within the length limit.
  - Haystacks are LongMemEval-S sessions (MIT, SHA-256 verified).
  - Item manifests are built, hashed and committed before the first GPU step that uses them.

## The graded set (the P6 lesson)

At P6's settings the 27B sat at ceiling on three families and near floor on vartrack. A paired test on such items can
hardly detect a difference, so the primary comparisons use two families whose difficulty is set by calibration:
- **multikey**: one question about one key among k needles with distinct keys. More needles means more distractors.
  Levels: k ∈ {16, 48, 128}.
- **vartrack**: a chain of c assignments; the answer is every variable in the chain. Fewer links is easier. Levels:
  c ∈ {2, 3, 4}.

**Calibration** (before any test item runs):
- It uses the reference arm only (`8,8`), at 64K.
- Separate calibration items, seed SHA-256("p8-longctx-calib-2026-11"), never reused as test items.
- 8 items per level per family: 48 items, about 1.7 GPU hours.
- **Rule:** for each family, pick the level whose `8,8` accuracy is closest to 65%; on a tie, the harder level.
- If the chosen level's accuracy falls outside 30-90%, it is still used, and the report says so.
- The chosen levels are written to the results folder and committed before any test item is built. The same levels
  serve every length, so accuracy by length is comparable.

**Test items:**
- Seed SHA-256("p8-longctx-2026-11").
- 48 items per family at the chosen level, 96 per length, at 32K and 64K.
- Stage 3 uses the same levels at 96K and 128K.

**The cliff set** (descriptive): P6's four families at P6's own settings, 8 items each, 32 per length. It shows whether
easy retrieval holds, and it only runs if the budget allows after the graded arms (see Order).

## Stage 2: paired accuracy (budget at most 18 GPU hours, calibration and rescue included)

**Arms** (paired on the same items):

| Length | Arms |
|---|---|
| 64K | `8,8`, `4,4` |
| 32K | `fp16`, `8,8`, `4,4` |
| 64K, conditional | `8,4`, only if H2 fails its usability bound |
| 64K, noise check | `8,8` again, on the first 16 graded items of each family in manifest order, in a fresh server |

**Hypotheses** on the graded set:
- Each is a paired difference in accuracy points.
- Intervals are 90% percentile bootstrap intervals over items (10,000 resamples, seed 20261102).
- **Usable** means the interval's lower bound lies above -10 points (P6's registered margin). **Equivalent** means the
  whole interval lies within ±5.

| ID | Role | Comparison | Length |
|---|---|---|---|
| H2 | primary | `4,4` minus `8,8` | 64K |
| H1 | secondary | `8,8` minus `fp16` (the deployed default's cost) | 32K |
| H3 | secondary | `4,4` minus `fp16` | 32K |

- Exact McNemar p-values and discordant counts are reported for each, per family and pooled. Holm adjusts the
  McNemar p-values of H1 and H3. The usable and equivalent verdicts come from the intervals.
- **Noise.** Same-arm discordance is reported from the noise check, with a Wilson 95% interval. Greedy decoding on this
  hybrid is not bit-reproducible (research-ledger-reference). If the `4,4` against `8,8` discordance lies inside the
  same-arm interval, the report says the cache's effect cannot be told apart from run-to-run noise.

**Predictions:**
- H2 is usable and equivalent, with a point estimate within ±3 points.
- H1 and H3 are equivalent.
- Any loss concentrates in multikey, where keys must discriminate.

**Rules after stage 2:**
- **R1.** If H2 is usable, the 4-bit cache becomes the lab's mode for contexts beyond the 8-bit cache's 72K reach. The
  deployed default stays `8,8`. If H2 is also equivalent, the report may call the 4-bit cache equivalent at 64K, and
  nothing stronger.
- **R2.** If H2 is not usable, run the `8,4` rescue arm. H2s is `8,4` minus `8,8`, judged by the same bounds. If H2s
  is usable, the split cache takes R1's role, up to its own reach. If neither is usable, no quantized long-context
  mode is adopted, and stage 3 does not run.
- **R3.** If H1 is not equivalent, the deployed 8-bit default gets its own follow-up. It is reported here, not fixed.

## Stage 3: accuracy against length beyond the 8-bit reach (at most 8 GPU hours; only if R1 or R2 adopts a mode)

- **Reach probe first.** The adopted mode at 152K and 168K, under stage 1's rules exactly. Lengths beyond its reach are
  dropped.
- **Accuracy.** The adopted mode on graded items at 96K, 128K, and the longest passing probe length, 48 items each
  (24 per family). The budget governs: an arm that would exceed it stops, and the report gives the N reached.
- **Report (descriptive).**
  - Accuracy by length, from stage 2's 32K and 64K results through stage 3's lengths, with Wilson 95% intervals.
  - **Usable context:** the longest length whose accuracy lies within 10 points of the mode's own 32K accuracy, with
    the gap's 90% interval.
- **Prediction P3b.** At 128K, the 4-bit cache's graded accuracy is within 10 points of its 32K accuracy.
- No equivalence claim is made beyond 64K.

## Order

Calibration; 64K `8,8`; 64K `4,4`; the noise check; 32K `fp16`, `8,8`, `4,4`; the rescue if R2 calls for it; then the
cliff set at 32K and 64K for every stage 2 arm, if budget remains; then stage 3.

Each arm is journaled item by item, so a stopped arm resumes exactly. When an arm stops at its budget, its paired
comparison uses only the items both arms completed.

## Secondary, descriptive only

At each format and length:
- decode and prefill tokens/s (pooled), and peak memory;
- per-family accuracy;
- answer-length statistics.

Decode kernels differ by format (fd75d for `8,8`, Triton otherwise), so speed is not a claim of this registration.

## Safeguards

- Every arm runs at a format and length that passed stage 1, or stage 3's probe, with at least 512 MiB headroom.
- Otherwise as in stage 1:
  - one GPU job at a time, through the queue, with the guard on;
  - the 20 GiB cap, and the 10 GiB system-RAM floor;
  - no retry after an out-of-memory error;
  - a 25-minute stop for a hung request, with the item counted as excluded.
- Exclusions only for a server error unrelated to memory, or a request over 25 minutes. They are counted and listed
  per arm.

## Budget

Stage 2 at most 18 GPU hours; stage 3 at most 8. Together that is at most 26 hours at 260 W, about 6.8 kWh.

## Not covered

- Decode kernels for the 4-bit or split cache (fdq4, a 4-bit fd75d, a split kernel).
- Prefill speed (fa75).
- Prefix reuse across questions.
- Host-RAM offload.
- A short-context T4 for changing the default.

Each needs its own gate or registration.

## Amendment 1 (2026-11-01 23:03 EST): implementation, fixed before calibration ran

Written by the stages 2-3 fork before any GPU step of this registration. The questions, arms, hypotheses, rules,
budgets and predictions do not change.
- Code: `testsuite/p8_longctx/` (`items.py`, `build_stage2_items.py`, `stage2_run.py`, `stage2_stats.py`,
  `stage2.sh`).
- 28 CPU tests pass in zarael-sandbox.
- The generator's own self-test passes, run unmodified at P6's settings in zarael-sandbox. That includes two
  byte-identical builds under different hash seeds and a match against P6's frozen manifest.

1. **Runner.** `stage2_run.py` imports stage 1's launcher, server stop, memory sampler and failure-text checks
   unchanged (`stage1_run.py`).
   - Each arm gets one fresh server.
   - Within an arm, a new server starts only after a hung request, a dead server or a thermal pause (item 8).
   - The runner never scores. P6's scorer runs on the laptop, on the journaled answer text.
2. **Seeds.** Each `SEED` is the SHA-256 hex digest of its seed text.
3. **Cliff seed.** The registration names none. Using the test seed would give the cliff set's multikey and vartrack
   items the same item ids as graded items, hence the same haystack order and first draws. So the cliff set uses
   SHA-256("p8-longctx-cliff-2026-11"), indices 0-7 per family, at P6's needle counts.
4. **Haystack target.** `TARGET_TOKENS` = L - 640 - 2 × the needle count, so that every prompt is at most L - 512
   tokens; `TOL_LO` and `TOL_HI` follow P6's 1%. The calibration prompts measure 64,690-64,973 tokens against the
   65,024 limit.
5. **Calibration items.** Item j (0-7) of level l (0-2) has index 6j + 2l. Items therefore differ across levels and
   spread over the generator's depth grid. Arm order: j, then family, then level, so the levels interleave
   through the arm.
6. **Test items.**
   - Stage 2: indices 0-47 per family, in manifest order index then family (multikey and vartrack interleaved).
     So the noise check's "first 16 of each family" is indices 0-15, and a budget stop leaves the families
     balanced.
   - Stage 3: the even indices 0-46 (24 per family), the same item ids as stage 2's even items. That keeps the depth
     grid and the asked positions spread.
7. **Asked position.** The generator asks multikey's needle number i mod k.
   - With k = 128 and indices up to 47, the asked needle is always among the first 48: an asked depth under about
     0.38.
   - With k = 48 every position is covered.
   - Disclosed, not changed: only module constants may be overridden.
8. **Exclusions and failures.**
   - A request over 25 minutes is excluded, and a fresh server takes the next item.
   - A server error with no out-of-memory cause is excluded, with a fresh server if it died.
   - An out-of-memory error stops that arm with no retry. The plan continues with the next arm, and paired
     comparisons use the items both arms completed.
   - At 86 °C or more on three samples running, our server is stopped. The run waits until the GPU is under 70 °C,
     then a fresh server resumes the arm and asks the paused item again. The guard's own stop is 87 °C.
9. **Budget.**
   - GPU time is the summed wall time of arm server sessions, from launch to settled memory. Stage 3 adds its probe
     steps.
   - Before each item, the arm's mean item time so far must fit; the first item is estimated as stage 1's prefill
     time at that length plus 30 s. If it does not fit, the arm and the plan stop.
   - The cliff set's "if budget remains": cliff arms run in stage 2's arm order. Each starts only if 32 times the
     matching graded arm's mean item time, plus 2 minutes, fits. Otherwise it is skipped and the next one is checked.
10. **Statistics** (`stage2_stats.py`).
    - The bootstrap uses Python's `random.Random(20261102)`, resampling items with replacement 10,000 times, with
      quantiles by linear interpolation.
    - Usable means a lower bound above -10. Equivalent means the interval lies within [-5, +5], inclusive.
    - McNemar is exact and two-sided. Holm adjusts the McNemar p-values of H1 and H3.
    - Noise intervals are Wilson 95%.
11. **Noise comparison.** Same-arm discordance is computed on the 32 noise items: the rerun against the first 64K
    `8,8` arm. The `4,4`-against-`8,8` discordance is computed on those same 32 items and on all 96, and the report
    states the comparison for both.
12. **Safeguard lengths.** An arm counts as passed in stage 1 when its length is within its format's stage 1 reach
    (`8,8` 72K, `4,4` 136K, `8,4` 96K, `fp16` 40K). The runner refuses any other arm unless stage 3's probe passed
    it.
13. **Probe.** Stage 1's step function runs unchanged (`stage1_run.run_step`). Its prompts are built by stage 1's
    builder (seed "p8-longctx-stage1", one single item) at 155,648 and 172,032 tokens. Stage 3 items for the probe
    lengths are built in advance and run only at a length the probe passes.
14. **Sandboxed builds.** zarael-sandbox cannot see `~/models`. The generator's self-test and every item build
    therefore run with HOME pointing at an ignored folder in the repo (`testsuite/p8_longctx/work/opt/p8-user That
    folder holds a copy of the model's tokenizer.json, whose SHA-256 the generator verifies. No code changes.

## Amendment 2 (2026-11-02 00:44 EST, by the ambassador; written after the reference-arm calibration and before any test item ran)

Numbered 2 because the stages 2-3 fork's implementation amendment (2026-11-01 23:03 EST) already holds the number 1. The
text below is the ambassador's, as sent.

The calibration (testsuite/results/turing/longctx/stage2/chosen_levels.json, commit 38c5006) put multikey at 8/8 for k = 16, 48 and 128, and vartrack at 1/8, 0/8 and 1/8 for c = 2, 3 and 4. Nearly every vartrack answer named one variable instead of the chain: a format failure that chain length cannot grade. Run as registered, the graded set would sit at ceiling and floor and could not detect a small loss. Only the reference arm (8,8) has been seen; no comparison arm has run.

Changes:
1. The graded set becomes one family, multivalue: one key with k values, all of which must be listed. Levels: k in {12, 24, 48}. A second calibration runs on 8,8 at 64K, 8 items per level (24 items), from the calibration seed, with items never reused. The registered selection rule applies unchanged: closest to 65%, ties go to the harder level, and a level outside 30-90% is still used and reported.
2. Graded requests get up to 512 output tokens so that the longest answer fits. Every other request keeps 128.
3. Test items: 96 multivalue items per length at the chosen level, at 32K and 64K, and 48 at each stage 3 length.
4. The calibrated multikey (k = 128) and vartrack (c = 4) levels join the descriptive cliff set, 8 items each per length, beside P6's four families, so a retrieval collapse stays visible.
5. Stage 2's budget rises from 18 to 20 GPU hours, to cover the second calibration.

Everything else stands: the hypotheses, margins, order, rules and safeguards.

## Amendment 3 (2026-11-02 00:49 EST, by the ambassador; before the second calibration ran)

Amendment 2's 512 output tokens cannot hold a full 48-value answer in two list styles the 27B uses on about 22% of P6's multivalue answers: 634 and 547 tokens with its preamble (testsuite/results/turing/longctx/stage2/multivalue_check.json, commit 9d58d39). A truncated answer would score as wrong, which is a format failure. Graded requests therefore get up to 768 output tokens, and the levels stay k in {12, 24, 48}. Stage 2's budget rises from 20 to 22 GPU hours to cover the longer answers. Everything else stands.

## Amendment 4 (2026-11-02 00:51 EST, by the stages 2-3 fork): implementing Amendments 2 and 3, before the second calibration ran

Implementation only. Nothing in Amendments 2 and 3 or the rest of the registration changes. 33 CPU tests pass in
zarael-sandbox.

1. **Depth grid for 96 items.** The generator spreads an item's needles by its index over a grid of `N_PER_FAMILY` = 50
   items. At an index of 50 or more, a needle would land past depth 1, and the generator's own validation would
   reject it. So the wrapper also sets `N_PER_FAMILY`:
   - 96 for the graded 32K and 64K sets (indices 0-95);
   - 96 for the stage 3 sets, which use the even indices 0-94, the same item ids as stage 2's even items;
   - unchanged (50) for every other set.

   This adds `N_PER_FAMILY` to the overridden module constants listed under "Items". The item files record it only
   when it is overridden, so the first calibration's set still rebuilds byte for byte (SHA-256 5afe9979..., checked).
2. **Second calibration.** 24 multivalue items at 64K from the calibration seed: item j (0-7) of level l (k = 12, 24,
   48) has index 6j + 2l, in the order j, level. The first calibration used other families, so no item repeats.
3. **Output tokens per arm.** Each plan arm carries its own limit:
   - 768 for the second calibration, because its items are graded items;
   - 768 for every graded arm, including the noise check and the rescue;
   - 128 for the cliff arms.
4. **Cliff set.** 48 items per length, all from the cliff seed:
   - P6's four families at P6's needle counts, indices 0-7;
   - the calibrated multikey k = 128 and vartrack c = 4, indices 8-15, so that they share no item id with P6's
     multikey and vartrack items.
5. **Noise check.** With one graded family, "the first 16 graded items of each family in manifest order" is the
   first 16 multivalue items: indices 0-15, 16 items. Amendment 1's item 11 comparisons apply to those 16.
6. **Item ids.** family-letter-level-index, for example `multivalue-k48-07`. The cliff set's P6-level items use the
   letter n, for example `vartrack-n5-03`.
7. **Budget.** The runner's stage 2 budget is 22 GPU hours. The first calibration's 1.52 hours count toward it.

## Amendment 5 (2026-11-02 02:02 EST, by the stages 2-3 fork): where the test sets were built

Amendment 1, item 14, has every item build run in zarael-sandbox. The eight test sets (graded 32K and 64K, cliff 32K
and 64K, stage 3 at 96K, 128K, 152K and 168K, all at multivalue k = 12) were built there first. The sandbox printed
each set's SHA-256, but other projects' runs pushed the run folder out of the sandbox's newest-20 rotation before its
files could be copied. The sets were therefore rebuilt on the laptop host with the same code and settings. All eight
host builds reproduce the sandboxed SHA-256 exactly; the manifests record them. The builds are deterministic
(byte-identical under different hash seeds, Amendment 1), so the items are the same either way.

## Amendment 6 (2026-11-02 06:28 EST, by the ambassador; written while stage 2 runs, before any stage 3 step)

Two probe steps are added below 152K. A residency model frozen before any stage 2-3 data predicts that both registered
probe lengths fail.

- **Why.** The ideas program's C2 cluster fitted a residency model to stage 1's results and froze its predictions at
  03:03 EST, before stages 2-3 produced any data (`research/prereg/2026-11-02-four-resources-formal-residency.md`,
  `testsuite/results/ideas/four-resources-formal/residency_predictions.json`).
  - It reproduces stage 1's 136K peak within 5 MiB: 10,500 predicted, 10,495 measured.
  - For `4,4` it predicts first-request peaks of 10,806 ± 40 MiB at 152K and 11,112 at 168K, against stage 1's limit
    of 10,752 MiB.
  - Run as registered, the probe would probably pass no length. Stage 3 would then have no longest-length arm, and
    P8's reach would stay at stage 1's lower bound of 136K.
- **Change.**
  1. The probe ladder becomes 147,456 (144K), 151,552 (148K), 155,648 (152K) and 172,032 (168K) tokens, in that order,
     still stopping at the first failure. Each step is stage 1's step function, unchanged, with one fresh server and
     a prompt from stage 1's builder (seed "p8-longctx-stage1", one single item).
  2. Stage 3 item sets at 144K and 148K are built in advance exactly as the 152K and 168K sets were: multivalue at
     k = 12, the even indices 0-94, in zarael-sandbox, with each set's SHA-256 in its manifest.
  3. The longest-length arm runs at the longest length that passes, as registered.
- **Predictions, registered before any probe step runs.** They come from the same frozen model (s = 19.125 MiB per 1K
  tokens for `4,4`) and test that model. They are not P8 hypotheses.
  - 144K: a peak of 10,653 ± 40 MiB, a pass with about 99 MiB to spare; prefill about 318 s.
  - 148K: 10,729 ± 40 MiB, a marginal pass with about 23 MiB to spare; prefill about 330 s.
  - 152K and 168K: as frozen, 10,806 and 11,112 MiB, both failures.
- **A risk stated before data.** C2 also found that a server's peak rises 102-119 MiB after its first request (at 64K,
  after 24-48 requests).
  - An arm at 144K or 148K may therefore run with less than 512 MiB of headroom after its first items, though still
    about 400 MiB under the card's 11,264 MiB.
  - The registered safeguard judges length at the probe, so the arm runs.
  - The report gives the arm's own peak memory. An out-of-memory error ends the arm without retry, as registered.
- **Unchanged.** Everything else:
  - stage 3's 8-hour budget, which now also covers the new steps (about 7 minutes each with loading);
  - the arms and their order, the budget rule, P3b and the safeguards.

  The stages 2-3 fork implements this change and records how in an implementation-only amendment.

## Amendment 7 (2026-11-02 06:41 EST, by the stages 2-3 fork): implementing Amendment 6, before any stage 3 step

Implementation only; nothing in Amendment 6 or the rest of the registration changes. Stage 3 has not started:
stage 2's main plan is still running.

- **Probe prompts.** Stage 1's builder made them unchanged (seed "p8-longctx-stage1", one single item), as at 155,648
  and 172,032 tokens. Manifest: `testsuite/p8_longctx/manifests/probe-prompts-144-148.json`.

  | Length | Prompt tokens | File SHA-256 |
  |---|---|---|
  | 147,456 (144K) | 146,906 | 7e9eaedd9f0aa666a21f40bb3fdd4624cdf0ef41aa52ea94139394e2b6c922f4 |
  | 151,552 (148K) | 151,031 | 65387b188ecca65deb84768684ec5195ae616176eabe6d466e14b106b27ce59b |

- **Stage 3 item sets.** Built as the 152K and 168K sets were: multivalue at k = 12, the even indices 0-94 on the
  96-item depth grid, 48 items each. Manifests `stage3-144k.json` and `stage3-148k.json`.

  | Set | Prompt tokens | Items-file SHA-256 |
  |---|---|---|
  | stage3-144k | 146,814-146,863 | e4da27436191b4541eaf4ce73ed266ce96fd7b3c74fd6d544934810ae279bd7b |
  | stage3-148k | 150,918-150,959 | a24bcfa4739978b9f8d3c7ee1da6885f55da5d4bbad25211a7a9c3aa58e40947 |

- **Reproducibility.** All four were built in zarael-sandbox, with HOME at `testsuite/p8_longctx/work/opt/p8-user and
  rebuilt on the host. The host rebuild reproduced every sandbox SHA-256: the four files above, and both stage 3
  manifests (234700d5... and b5a5ef27...).
- **Plan.** In `testsuite/p8_longctx/plans/s3-44.json`:
  - the probe lengths are 147,456, 151,552, 155,648 and 172,032, in that order, stopping at the first failure;
  - `items_by_length` gains 147,456 → `stage3-144k.jsonl` and 151,552 → `stage3-148k.jsonl`;
  - the arms and their order are unchanged.
- **No runner logic changed.** The probe already ran stage 1's step function over any length list, stopping at the
  first failure. The longest-length arm already took the longest passing length.
- **Deployed and checked.** The new files are in the desktop's items folder (`$L/runs/longctx/items/` and its
  `probe/`), and the plan is in `$L/runs/longctx/plans/`. SHA-256 was checked on the desktop.
- **Tests.** 35 CPU tests pass in zarael-sandbox. That includes two new ones: every plan names committed manifests
  with the right output-token limit, and the stage 3 probe ladder follows Amendment 6.

## Amendment 8 (2026-11-02 08:16 EST, by the ambassador; after H2's result, before the rescue arm and before any stage 3 step)

Stage 3 with the split cache, if R2 adopts it.

- **Context.** H2 is not usable: 4,4 − 8,8 at 64K is −3.1 points, 90% interval −11.5 to +5.2 (the fork's interim
  note). R2's rescue arm (8,4) therefore runs after the 32K arms. If H2s is usable, stage 3 runs with 8,4, whose stage 1
  reach is at least 96K and was tested only up to 96K.
- **Change.** With 8,4, the registered safeguard is applied as follows:
  1. Stage 3 is the 96K arm only. The 128K arm lies outside 8,4's tested reach, so it is not run and is reported as
     outside the tested reach.
  2. The probe steps (144K, 148K, 152K, 168K) are not run. C2's frozen model predicts an 8,4 peak of about 11,870 MiB
     at 144K, above the card's 11,264 MiB, so the first step could only end in an out-of-memory error.
  3. No 8,4 probe steps are added between 96K and 128K. The same model puts 8,4's reach at about 103K, so such steps
     could add only a few thousand tokens of reach, which changes no decision.
- **Unchanged.**
  - With 4,4, Amendments 6 and 7 stand; 4,4 is not adopted now.
  - If neither mode is usable, stage 3 does not run, as registered.
  - Everything else stands.
