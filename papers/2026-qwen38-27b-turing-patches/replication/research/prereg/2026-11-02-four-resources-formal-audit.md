# Pre-registration: four resources formalized (C2), part 1: a cross-session audit of the 2080 Ti speed table, and whether rates compose across operating points

Registered 2026-11-02 02:56 EST by the C2 CPU researcher of the ideas program. The first commit of this file freezes it. Later
changes are dated amendments appended at the end.

- Gate: `research/gates/2026-11-02-ideas-program.md`, cluster C2. Prior art: `research/prior-art-four-resources-formal-2026-11-02.md`
  (with its skeptic pass), open questions O1 and O2 as narrowed there; the base report's test 4
  (`research/FOUR_RESOURCES_PROGRAM_2026-11-01.md`), which designed (b) and (c) below and was never registered.
- RIFE: research-ledger-reference, research-ledger-reference, research-ledger-reference, research-ledger-reference,
  research-ledger-reference, research-ledger-reference.
- CPU only, data already on disk, read only. No GPU on any machine.

## Questions

1. **Q1 (O1, narrowed): do separately measured sessions agree?** Edge semantics, declared: a cell is one configuration
   measured in one session; two cells of one session give a within-session ratio; a cycle in the session-configuration
   graph compares the same ratio measured in two or more sessions. Curl here audits measurement consistency. It does
   not test composition (the skeptic's correction 1).
2. **Q2 (composition across operating points): which conversion rates transfer, and does indexing them by operating
   point make them compose?** Composition along a path in the lattice of operating points (change the fill, then
   drafting, then the cap) is path-independent exactly when the factors do not interact. The test is factorial
   separability plus leave-one-cell-out prediction, judged against Q1's cross-session floor.
3. **Q3 (bottom-up composition against the consensus): do the living map's composed graphs fall inside the
   cross-session band of the configurations they predict?** Tests 2 and 3 judged them against final2 alone.

## Data (read only)

- `testsuite/results/turing/ab/{ab1, ab2, ab3, depth1, depth2, final1.clean, final2, power1, rafatxf1, rafatxf2,
  w12p1, w13ab1, w14ab}.jsonl`: 13 sessions, 1,053 rows, every cell 9 rows (3 tasks x 3 rounds).
  - `final1.jsonl` is left out because the lab's own `final1.clean.jsonl` supersedes it.
  - `w14warm.jsonl` is left out: it is a 3-row warm-up.
- `research/livingmap/graphs/*.json` and `testsuite/results/four_resources/test{2,3}/` (Q3 only).

## Numbers already seen (disclosure)

- `final2.md` in full: every arm's pooled ms/token at both fills.
- Tests 1-3 (reports and RIFE summaries) and the w13 extension's errors.
- The living map's curated rows, including tokens per joule for final1 (`lm-tpj-l1-*`) and power1 (`lm-tpj-l2-*`).
  Tokens per joule mixes time with power, but its ratios carry information about final1's and power1's fill, drafting
  and cap ratios. Predictions about those ratios are informed, not blind.
- The metadata of every session above, read to define configurations: tags, builds, env, fills, cache formats,
  prompt token counts and row counts. No other session's speed value was read.

## Definitions

- **Cell value:** pooled ms/token, sum of `decode_wall_s` over sum of `new_tokens` across the cell's 9 rows. This is the
  lab's pooled convention (medians of per-row ratios overstated MTP gains). **Round value:** the same within one round.
- **Configuration key** (code: `tools/ideas/four-resources-formal/fr_data.py`):
  - **E, engine:** build plus env. `EXL3_INT8_SQ_MAX_M` is dropped when it has the value the lab pairs with the drafting
    state (1 undrafted, 2 drafted). That switch counts as part of the drafting conversion (declared). A non-standard
    value (ab3's j0) stays in E.
  - **D, drafting:** none, or MTP with its draft count, plus the sparse draft head when on.
  - **F, fill:** 0, 8,192 or 28,000.
  - **K, cache:** `8,8` or `4,4`.
  - **P, cap:** 290 or 320 W from power1's tag. Every other cell is 260 W, the desktop default the guard sets
    (declared).
  - **Q, prompt set:** `p1` for the one-token prompts of ab1, ab2 and depth1; `std` otherwise.
- **Computed from metadata before any speed was read:** 117 cells, 13 sessions, 75 configurations, 4 components,
  **cycle rank 33**. Twenty-two configurations appear in two or more sessions. The adopted w11b arms appear in 5-6
  sessions each.

## Q1: the audit

- **Model:** log T(s, c) = a(s) + b(c) + e(s, c), fitted by weighted least squares.
  - Weights: 1 / var(cell), where var(cell) is the pooled within-cell variance of log round values, pooled separately
    for undrafted and drafted cells, divided by the 3 rounds.
  - Session effects sum to zero within each component. They are also reported with final2 as the reference.
- **Outputs:**
  - the session effects, with 95% intervals;
  - the residuals e, the cycle residuals, whose RMS in percent is the **cross-session floor**;
  - chi-squared, the sum of w·e² against 33 degrees of freedom;
  - a between-session variance tau² (method of moments), added to var(cell) for z-scores;
  - outliers at |z| > 3;
  - DiffNet's A-optimal suggestion for the next cell to measure: the cell whose addition most reduces the trace of
    the covariance of b.
- **Predictions:**
  - **A1.** The cycle rank is 33 > 0, so the audit is not vacuous (fixed above; the code must reproduce it).
  - **A2.** The residual RMS is at most 4% for undrafted cells and at most 6% for drafted cells.
  - **A3.** Chi-squared / 33 > 2: sessions disagree beyond their own round-to-round noise. Composition claims must
    then be judged against the cross-session floor, not within-session intervals.
  - **A4.** power1 (24 capped cells in one long session) has the largest |a(s)| in its component. At least half of
    any flagged outliers are power1 cells.
- **Vacuity criterion:** cycle rank 0. It is 33.

## Q2: separability and path composition

**(a) Rates per conversion.**
- For each conversion, take every within-session pair of cells that differ in that factor only. Each pair gives a
  log ratio, with its standard error from the three rounds.
- Pool the pairs per operating point (the levels of the other factors) across sessions, by inverse variance.
- Heterogeneity across operating points: Cochran's Q, I² and the range.

Rates are T(new)/T(base) in ms/token. "Holds" means the pooled 95% interval lies inside the range; "partly" means the
point estimate is inside but the interval crosses an edge; "refuted" means the point estimate is outside.

| Id | Conversion and operating points | Predicted range |
|---|---|---|
| F1 | fill 28K over 0, fd75d engines (w11b, w12p, w13 with FD75D_LIVE=0), undrafted / drafted | 1.06-1.11 / 1.05-1.10 |
| F2 | fill 28K over 0, paged-attention engines: stock undrafted; stock drafted; w9 undrafted | 1.18-1.32; 1.20-1.36; 1.30-1.50 |
| F3 | at the same fd75d engine, \|log r(drafted) - log r(undrafted)\| for fill 28K | at most 0.03 |
| F4 | fill 28K heterogeneity across all engines with both fills, I² | at least 90% |
| D1 | MTP over none, fill 0: stock; w11b; w12p; raf `4,4`; w11b with `4,4` | 0.50-0.60; 0.79-0.87; 0.70-0.86; 0.62-0.80; 0.79-0.90 |
| D2 | the same engine's drafting rate at 28K against fill 0, and across sessions | within 0.04 in log |
| D3 | drafting heterogeneity across engines, I² | at least 95% |
| P1 | cap 320 over 260: w11b arms; stock arms | 0.91-0.99; 0.97-1.02 |
| P1x | cap x engine interaction: log r(320, w11b) - log r(320, stock), mean over arms | at most -0.03 |
| P2 | cap 290 over 260 lies between 1 and that arm's 320 rate | in every arm |
| K1 | `4,4` over `8,8` at fill 0: w11b and raf, undrafted and drafted | 0.97-1.03 |
| K2 | `4,4` over `8,8` at 28K: w11b; raf | 1.05-1.40; 0.92-1.08 |
| L1 | 4 or 8 drafts over 6 (w13 split-K 2, fill 0) | 0.95-1.05 |
| H1 | sparse draft head over none (w13 split-K 2), both fills | 0.93-1.02 |
| W1 | depth2: FD75D off over the w11 engine at 28K | 1.15-1.45 |

**(b) Leave-one-cell-out composition.** All models are weighted least squares on log T with session effects:
- **M0 (replication, no composition):** session + configuration. It predicts a held-out cell from the same
  configuration in other sessions.
- **M1 (flat table):** session + main effects E + D + F + K + P + Q. Each conversion has one rate everywhere.
- **M2 (rates indexed by operating point; interactions declared now):** M1 + E×D + E×F + E×K + D×F + K×F + E×P. The
  physical reasons:
  - verify-row kernels (E×D);
  - the attention kernel and cache format at depth (E×F, E×K, K×F);
  - passes per token under drafting (D×F);
  - the clock sensitivity of each engine's kernels (E×P).

A held-out cell counts only where the model can predict it: every level and interaction it needs must be estimable
without it (M0: its configuration in another session). The score is the RMS of log errors, in percent, on the common
support of all three models, and for M1 against M2 on their own wider support.

- **Predictions:**
  - **B1.** M1's leave-one-out RMS is at least 8%.
  - **B2.** M2's RMS is at most half of M1's (the base report: "the graph halves the flat table's error").
  - **B3.** M2's RMS is at most 1.5 times M0's on the common support.
- **Kill:** M2's RMS exceeds 0.9 times M1's. Indexing rates by operating point would then be no better than a flat
  table.

**(c) Curvature, descriptive.** For every square measured within one session (two factors at two levels each),
report the holonomy h = log T(a', b') - log T(a', b) - log T(a, b') + log T(a, b), with its standard error. This is
the local path-independence test. Report the distribution of |h| / SE by factor pair.

## Q3: bottom-up compositions against the consensus

- **Target:** M0's configuration potentials, converted to ms/token at the weighted mean session effect of the
  configuration's sessions (the consensus). The band is ± max(tau, the spread of that configuration's sessions).
  final2's own value is shown beside it.
- **Compositions** (registered and computed by tests 2-3; this is a re-judgment against a different target, not a new
  prediction):
  - test 2's priced graphs for adopt, w13, stock and w9 at both fills;
  - test 3's drafting cycles for adopted MTP and w13 split-K 2 at both fills: the registered per-cycle host term
    (primary) and the declared per-pass variant;
  - test 3's w13 split-K 1 primary.
- **Nulls:**
  - single-alpha: alpha = consensus / priced graph for adopt at fill 0, applied to every other composition;
  - critical path: the primary per-cycle composition (isolated node costs, no per-pass latency).
- **Predictions:**
  - **C1.** The per-pass variant lies within the consensus band (or within ±4% if the band is narrower) for adopted MTP
    and w13 split-K 2 at both fills. The per-cycle primary lies below the band for at least 3 of the 4.
  - **C2.** The gap between the per-pass and per-cycle errors exceeds twice the cross-session SD of the drafted cells,
    so existing data can resolve it.
  - **C3.** Single-alpha does worse than the per-pass variant on the drafted arms (RMS).

## Not covered

Capability (no capability cells here), energy (power traces are not used), the laptop (no multi-session speed table)
and any new GPU measurement.

## Software and outputs

- `engines/analysis-env`: numpy 2.3.3 and scipy 1.16.2. Scripts in `tools/ideas/four-resources-formal/`.
- Results in `testsuite/results/ideas/four-resources-formal/`. Each process runs with CUDA_VISIBLE_DEVICES="",
  nice 19 and 8 threads.
