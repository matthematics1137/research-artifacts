# Paper 6 amendment, 2026-11-01: primaries-only scope, a stopped cell, and the outcome

Written 2026-11-01 05:15 EST, after the co-primary cells ended and the pre-registered analysis ran. The pre-registration
(`research/design-kv-allocation.md` and its 2026-09-19 amendment) and the hashed harness are unchanged. Every journal
row still carries the same harness hashes. This amendment records scope and operations; it changes no hypothesis,
estimand, test or margin.

## 1. Scope: the co-primary cells only, for now

- **The decision.** It is Matt's, 2026-10-31, after the laptop ran hot: "we just do the 8hr one with your changes and
  then later we amend the paper or it won't even matter". The plan and its reasoning are in
  `research/P6_EFFICIENCY_RETHINK_2026-10-31.md`.
- **Ran on long context.** All eight co-primary cells: both models × {Q2_K_XL, Q4_K_XL} × {f16, q4_0}. Before the
  pause these had also finished: the 27B Q2_K_XL q8_0 and k8v4 cells, and the 27B IQ3_XXS f16, q8_0 and q4_0 cells.
  The 27B IQ3_XXS k8v4 cell has 97 of 200 items.
- **Deferred, to be decided on cost and heat.**
  - Long context: the Qwen3-8B q8_0 and k8v4 cells and its whole IQ3_XXS tier; the 27B Q4_K_XL q8_0 and k8v4 cells;
    the rest of 27B IQ3_XXS k8v4.
  - GSM8K: every cell not listed in `summary.md`.
- **Effect.**
  - The co-primaries need only the eight cells, so both are tested as pre-registered.
  - Secondaries are reported where measured and marked as missing elsewhere.
  - The equal-memory ranking and exchange rates cover only the measured cells.

## 2. Operations (no harness change)

- **The launcher.** `testsuite/paper6/launch_rest.sh` ran the primary cells first, with `--idle-seconds 3`.
- **The thermal guard.** A CPU guard (`tools/thermal_guard.py`) shared the load guard's PAUSE marker. It paused when
  the 60 s mean package temperature reached 95 °C, and resumed after 120 s below 85 °C. It paused the run several
  times. Each pause stopped and restarted the server through the harness's own pause path.

## 3. The stopped cell: qwen38-27b Q4_K_XL q4_0 at 190 of 200

- **The refusal.** At 02:07:03 PDT, after a thermal pause, the server restarted with `--fit` choosing 37/66 layers
  instead of 36/66. Free memory was 10,887 MiB against 10,830 at the previous start, a difference of 57 MiB, and that
  admits one more layer under `--fit`'s 1,024 MiB margin.
- **What the harness did.** It refused as designed (HARNESS_NOTES item 23). The changed start is kept in the journal,
  and no request ran under 37 layers.
- **Why the cell was not resumed.** A relaunch at 02:12 PDT, with free memory back inside the 36-layer window, was
  refused by the journal audit. A journal that holds a changed start cannot be resumed, and the notes forbid deleting
  that record. The cell therefore stops at 190 of 200.
- **Analysis.** The pre-registered rule applies: contrasts use the complete cases across the cells they need, so the
  co-primaries use 190 items. `analyze.py` flags both the missing items and the placement change.

## 4. Sensitivity (labelled, not pre-registered)

`paper6/scripts/sensitivity_missing.py`; the output is
`testsuite/results/paper6/analysis-2026-11-01/sensitivity_missing.md`.

| scenario | I_hybrid | I_GQA | M |
|---|---|---|---|
| complete cases (as pre-registered) | +3.68 (se 2.03, n 190) | +0.00 (se 2.36, n 190) | +3.68 (se 3.28) |
| the 10 imputed correct | +4.00 (se 1.98) | −1.00 (se 2.45) | +5.00 (se 3.23) |
| the 10 imputed wrong | −1.00 (se 2.45) | −1.00 (se 2.45) | +0.00 (se 3.40) |

With the f16 cache, the same model answered 9 of the 10 correctly. Neither conclusion below changes in any scenario.

## 5. Outcome of the pre-registered analysis (2026-11-01 02:13 PDT)

`testsuite/results/paper6/analysis-2026-11-01/analysis.json` (sha256 a626958425f404dc…) and `summary.md`.
Points are accuracy points on the 200 long-context items at 16K.

| co-primary | estimate | p | Holm p | decision |
|---|---|---|---|---|
| M = I_hybrid − I_GQA ≠ 0 (sign-flip permutation) | +3.68 [−2.79, +10.16] | 0.341 | 0.341 | no rejection |
| I_hybrid equivalent to 0 within ±10 (TOST) | +3.68 (90% CI +0.34, +7.03) | 0.00106 | 0.00211 | equivalent within the margin |

- **Prediction "I_hybrid equivalent to 0": supported.** At 2-bit weights, the 4-bit KV cache costs the hybrid no more
  than it does at 4-bit weights, within ±10 points. The point estimate is +3.7 and the 90% interval lies just above
  zero, so a small interaction of a few points is not excluded.
- **Prediction "I_GQA ≥ I_hybrid": not supported in direction, and not significant.** I_GQA = −1.00 and
  I_hybrid = +3.68.
- **D per tier.**
  - Hybrid: Q2 +1.50, IQ3 −1.00, Q4 −2.11.
  - Full attention: Q2 0.00, Q4 +1.00.
  - Every interval includes zero. The q4_0 cache never cost more than 1.5 points on this suite.

## 6. Observations for the paper (descriptive, not tests)

- **Ceiling and floor.** In an interim tally of 12 complete cells:
  - 136 of 200 items were right in every cell, and 10 were wrong in every cell (all variable tracking);
  - single needle scored 50/50 everywhere;
  - variable tracking sat at 2-12/50, except for the 27B at 4-bit weights (29/50 with f16).

  The suite had little room for a small cache effect. This was accepted as a risk in the design and motivates P6b
  (`research/P6B_BREAKING_POINT_DESIGN_2026-11-01.md`).
- **Truncation.** Qwen3-8B truncated 50 (f16) and 53 (q4_0) of 200 long-context answers at Q2_K_XL, against 13 and
  15 at Q4_K_XL. Lower weight precision lengthened the reasoning until it hit the token cap. This is the pattern of
  Book 0's principle that length shows damage before accuracy does.

## 7. Corrections (2026-11-01 05:47 EST, found while reviewing the manuscript)

- **Section 5 compared I values over different item sets.** I_GQA = −1.00 uses all 200 of the full-attention model's
  items, and I_hybrid = +3.68 uses 190. On the 190 items common to all eight cells, I_GQA = 0.00, which is the value M
  uses. The direction is the same either way. The manuscript states both.
- **The hybrid's positive interaction comes mostly from 4-bit weights.** At UD-Q4_K_XL the q4_0 cache scored higher
  than f16 (D = −2.11: 7 items gained, 3 lost). At UD-Q2_K_XL it cost +1.50 (4 lost, 1 gained).
- **Section 6, "Truncation," overstated the pattern.**
  - Long-context answers are capped at 128 tokens (thinking off). The counts are answers that reached that cap, not
    reasoning that ran out of room.
  - Qwen3-8B reached it 50 and 53 times at UD-Q2_K_XL, against 13 and 15 at UD-Q4_K_XL.
  - The 27B reached it most at UD-IQ3_XXS (24-29), at most 2 times at UD-Q2_K_XL, and never at UD-Q4_K_XL.
  - So cap hits do not track weight precision monotonically. The 8B's pattern fits "weight damage lengthens
    answers," but P6 is not clean evidence for Book 0's length principle. The manuscript reports it as an unexplained
    exploratory observation.
- **"136 of 200 right in every cell"** comes from an interim tally of the journals (12 complete cells), not from
  `analysis.json`. The manuscript leaves it out, and its family table shows the same ceiling.
