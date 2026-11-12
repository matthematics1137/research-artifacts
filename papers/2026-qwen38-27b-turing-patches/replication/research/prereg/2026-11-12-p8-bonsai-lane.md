# Registration: P8 counterpoint lane, Bonsai 2 27B on NInfer, same-card accuracy per minute (finalized 2026-11-12T19:41Z)

Drafted on CPU by codex-t (assignment `assign_0123fa399bf1ccd94921128955f92de5`, branch `research/p8-bonsai-lane`, head
8283052b, not merged); finalized by the ambassador, before any Bonsai output. Asked by Matt while reading P8,
2026-11-12: "shouldnt we answer this before publishing?" and "ok lets run it and make sure we dont mix up the names in
the paper". The text below the finalization note is the draft as codex-t wrote it, byte-identical to
`research/prereg/drafts/2026-11-12-p8-bonsai-lane.md` at 8283052b. Gate: research/gates/2026-11-12-p8-bonsai-lane.md.

## Finalization note (ambassador, 2026-11-12T19:41Z, before any Bonsai output)

- **Hold 1, response shape: logical scoring suffices.** NInfer returns reasoning separately; the adapter keeps the
  native response and scores with P8's existing text path and scorer. For a descriptive counterpoint this is enough; the
  paper will not call the outputs byte-identical to the other lanes'.
- **Hold 2, controls: keep the drafted boundary.** P8 has no w13 MMLU or undrafted-w13 quality rows (a limitation P8
  states). The comparison family stays as drafted (Holm m = 5): Bonsai undrafted against stock EXL3 undrafted and
  Bonsai MTP-3 against stock EXL3 MTP on both suites, and Bonsai MTP-3 against w13 on GSM8K. Any w13 sentence names
  GSM8K only. No controls are added.
- **Hold 3, qualification:** the guarded qualification (at most 15 minutes, no benchmark items) runs on the desktop
  after staging and before any data, through the desktop queue; its receipts are recorded before the lane starts.
- **Names:** "NInfer" (iamwavecut/ninfer-all, Apache-2.0, pinned f4ab369a) is a different engine from "NInferno"
  (brontir), P8's main competitor. Every sentence this lane adds to P8 names NInfer and says at its first mention that
  it is a separate engine.
- **Model pin:** HF WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 at 841f7812 (Apache-2.0 card and NOTICE), weight file
  Ternary-Bonsai-2-27B-ninfer-v3.ninfer, 9,520,051,456 bytes, SHA-256 cdc4810b...c14a.
- **Budget:** plan about 5 GPU hours at the 260 W limit; hard cap 6 hours including qualification. A capped run ends
  incomplete; it is never extended.

---

# DRAFT registration: P8 Bonsai/NInfer accuracy per minute

2026-11-12. **CPU preparation; not a registered or launched GPU experiment.** Gate: `research/gates/drafts/2026-11-12-p8-bonsai-lane.md`. RIFE context `research-ledger-reference`; new facts/decision await ambassador filing from `research/raw/2026-11-12-p8-bonsai-lane/rife-candidates.json`. Source synthesis and staging plan in that directory. No P8 paper edit is included.

## Question and estimand

On the lab's 260-W RTX 2080 Ti, compare the complete **Bonsai 2 27B derivative + NInfer** stack with the archived **original Qwen3.8-27B EXL3 2.00-bpw** lanes, at fixed task/request resources. Which produces more correct answers per minute, and what accuracy changes accompany it? These are paired descriptive stack comparisons, not a causal estimate of quantization loss or a claim that the derivative preserves the original weights.

The historical **w13 MTP** quality lane has GSM8K only. There is no w13 MMLU or undrafted w13 quality lane at the frozen base. The draft answers w13 on GSM8K and stock EXL3 on both suites. It does **not** answer a two-suite w13 comparison. Before any Bonsai result, the ambassador must either retain that boundary in P8 or amend the registration/budget to obtain missing controls. Never substitute stock MMLU for w13, infer quality from speed-only rows, or choose another original lane after outcomes.

## Frozen data and request

Dataset identity/order and legacy harness SHA-256 are machine-readable in `research/raw/2026-11-12-p8-bonsai-lane/contract.json`.

| Suite | Items | Dataset SHA-256 |
|---|---:|---|
| `gsm8k_500` | 500 | `7179bc41ad3e06c9f4b62ff1811fd272fa5ae82ddbee91dbf74d8278878ba344` |
| `mmlu_500` | 500 | `0a7e6b09f8d3ec6a13b00d02d8549a8e48d255a959be6034b932372f3cdea8da` |

Use every item once in existing dataset order, GSM8K then MMLU. One request at a time, one sample/item, temperature **0**, top-p **.95**, top-k **20**, reasoning **medium**, **4096 generated tokens total including reasoning**, no extra system prompt, tools, penalties, seed-dependent sampling, early answer extraction or forced conclusion. Reuse legacy `build_prompt`, `SCORERS`, `visible_content` and `run_item_full` unchanged. Their number/letter extraction and truncation fallbacks are deliberately retained. Requests stay single user turns. Translate the legacy ctk effort into NInfer's source-confirmed top-level medium + enable_thinking, never drop it on error.

The original server context/cache is **16384**, Q8 KV, fp16 recurrent state (archived run records and Tabby yml); NInfer uses 16384 max context and KV capacity, int8 KV, `--gdn-state-fp16`, max concurrency 1, prefix reuse enabled. Different cache implementations/prefill kernels and speculative algorithms are part of the stack; Q8/int8 labels do not assert bit-identical state. Preserve actual prompt/cache/timing counts. Do not tune kernels, context, KV, effort, penalties, draft width or output cap on benchmark answers.

## Lanes and controls

NInfer source `f4ab369a06c728325b706c989d71be7dc0c05a1b`; HF `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3` revision `841f7812793972b5de6dc48910bdd1e2722ae48b`, full manifest in raw directory. No new conversion. Text-only, all text weights resident on GPU. Embedded vision/DFlash unused.

1. `TURING-bonsai-ninfer-none`: explicit `--spec none --ngram-draft-tokens 0 --lookup-ngram 0`.
2. `TURING-bonsai-ninfer-mtp3`: explicit `--spec mtp --draft-tokens 3 --draft-min-p 0 --no-adaptive-mtp --full-draft-head --ngram-draft-tokens 0 --lookup-ngram 0`. The embedded MTP is ProCreations' Bonsai-specific Q8 head. No speculative losslessness assumption.

Order: undrafted full 1000, then MTP full 1000. Restart the server between modes, log calibration/load time separately, use read-only health readiness, no benchmark warmup and no benchmark-item probing. A nonbenchmark qualification is described below. Original EXL3 MTP used 6 dynamic drafts; the comparison is between these fixed deployed stacks, not the same drafter. No original rerun is included. Freeze existing rows/hashes from `baseline-inventory.json` at base `ac5d78ed3c7b8b19d2ec96accb7e19943c2c7af8`.

| Original lane | GSM8K correct / seconds / tokens | MMLU correct / seconds / tokens |
|---|---|---|
| stock none | 488 / 7112.03 / 162898 | 433 / 15822.77 / 351734 |
| stock MTP | 488 / 3223.58 / 163618 | 434 / 8318.53 / 354805 |
| w13 MTP | 486 / 2237.44 / 165272 | absent |

These are sums over 500 requests; timings differ from isolated speed-benchmark decode rates. Historical control timing is not interleaved with the new experiment, so report driver/clock/temperature differences and do not infer pure engine speed causality.

## Shape and transport hold

Native NInfer returns split reasoning. `adapter.py` retains native response + exact wire/native hashes, then calls the **existing P8 `laptop_inline_text`** and passes that logical inline representation to the unchanged scorer. For a length finish with reasoning only, it adds no closing tag, preserving E47's reasoning fallback. Usage stays native generated-token counts; no re-encode fallback or fabricated token count. Byte identity is **unverified**: parser trimming/tag boundaries may lose information, especially if the cap lands just after `</think>`. Record native and normalized shapes, normalization case, finish reason, usage and model id per item. No byte- or token-identity accuracy claim.

Before data collection, the integrator must document whether the logical normalization suffices for this descriptive row or obtain an actual raw-output path. If strict parser-disabled response identity is required, this candidate is held: no inspected NInfer flag supplies it. Do not call approximate normalization an exact setting match. A change to the scoring representation requires a registration amendment before benchmark outputs, not post hoc rescoring to improve the result.

No automatic generation retry, including lost replies, HTTP rejection or ctk fallback. The client fsyncs an intent before each request, retains the original outcome/error and stops at the first transport/shape error. Fresh result directories and an atomic campaign claim prevent queue retries from producing additional samples. A crash after intent is an unknown outcome; reconcile evidence without blind resend. Partial data stays saved; no continuation or new cap without a new registered decision.

## Qualification and stopping

Future bounded qualification: <=900 GPU seconds **total**, both modes and termination included, under the existing desktop guard and unit caps. Check source/artifact/binary hashes, source clean state, CUDA/compiler/driver, `sm_75`, GPU name, existing 260-W limit (do not change it), no foreign workload, free port, 16K context/fit, actual KV/state precision and explicit draft mode. Inspect `/apply-template` for the fixed synthetic user prompt `Compute 3 + 4. End with Answer: 7.` with medium versus xhigh; compare the medium rendered template against the original serving template and show thinking is open. No benchmark items. Inspect a native normal response, usage totals and finish/shape; use a separate one-token synthetic cap to exercise reasoning-only length handling. Qualify both modes on the same synthetic material without optimizing settings. Save request/response bytes and logs locally, and a mode-specific manual evidence receipt matching `contract.json`; receipt fields are evidence assertions, not an automatic proof or owner approval.

If medium is ignored, source/artifact incompatible, response fidelity unacceptable, context/OOM, guards stop, transport/usage error, time cap expires or any required item is missing: **incomplete/inconclusive** for that formal comparison. No retuning, replacement artifact, higher cap or cherry-picked completed subset. Stop on first error; preserve all intents, responses, rows, GPU samples, startup logs and exit status. Never stop early for a favorable score. Read-only readiness checks may repeat; generation POSTs may not.

## Analysis fixed before outputs

`tools/turing/bonsai/analyze.py` loads strict unique IDs (no latest-wins selection), checks dataset/baseline hashes, and retains missing/error counts. Primary results require all 500 rows without transport errors per lane/suite. Wrong and truncated answers remain in the denominator. Partial descriptive rates use all observed request time, including failed requests, and are explicitly labelled partial. Report cap-exhausted/unfinished separately from finished-wrong.

For each complete lane/suite: correct/n, Wilson 95% interval, total/mean wall seconds, mean native generated tokens with available-count denominator, truncations, and **correct/min = 60 × sum(correct) / sum(all request wall seconds)**. Include load/calibration/qualification time separately; report whole-campaign correct/min as an additional deployment-cost view, never mix its denominator with historical request-only rates. No speed estimate from vendor throughput substitutes for measured wall time.

Five fixed primary paired comparisons (B minus A): Bonsai none versus stock none on each suite; Bonsai MTP-3 versus stock MTP on each suite; Bonsai MTP-3 versus w13 MTP on GSM8K. Report discordants, paired accuracy difference and exact two-sided McNemar p with **Holm family size 5, alpha .05**. Missing comparisons keep their family slot (p=1 for adjustment, result unavailable). Report deterministic paired-item percentile-bootstrap 95% intervals for accuracy difference and correct/minute difference: 5000 replicates, seed 20261112, same resampled item indices on both sides. These intervals are unadjusted descriptive intervals; only Holm-adjusted McNemar supports the named accuracy test. No equivalence/noninferiority test or margin, pooled suite significance or inference from overlapping Wilson intervals.

Within-Bonsai none/MTP differences are descriptive (no extra significance claim, no assumption of equal outputs). Report task coordinates as GSM8K arithmetic / MMLU knowledge slices, levels/horizon not measured; model state includes derivative/original artifact, engine, KV/state, medium/template and harness hash. Resource coordinates: one sample, 4096 tokens, 16K window, measured generated tokens/seconds/VRAM, 260-W limit. These familiar benchmark slices do not locate a 50% capability frontier. After registered analysis, add four Bonsai points and all named historical controls to `research/self-improvement/scoreboard.jsonl` through the scoreboard source adapter, with registration/artifact identity and finished-wrong/unfinished. Ambassador owns the adapter/source-list change and RIFE entry before P8 integration; no invented generator level or frontier movement. Board power samples give an approximate integration in joules; distinguish it from whole-system energy and the 260-W upper bound.

## Budget and wording rules

From archived generated tokens and NInfer rates: undrafted `(514632/58.5 + 101666/1080)/3600 = 2.470 h`; MTP `(518423/95.9 + 101666/962)/3600 = 1.531 h`; total **4.001 h**. Assume output lengths only for planning. Published rates are a 250-W single CLI sample at another engine/artifact revision, so neither server throughput nor this estimate is promised at 260 W. Plan about **5 GPU h**, allowing <=15 min qualification, startup/calibration and server variance. **Hard 6 h total**: qualification <=900 s and queued data/startup <=20700 s including its termination tail. No automatic extension. At 260 W this is <=1.56 kWh GPU-board-limit energy; log measured power separately. CPU source build/download time does not consume GPU budget and is separately capped by the existing unit/resource policy. Downloads need ~9.521 GB plus cache/partial headroom; build disk/RAM qualification required before scheduling.

Existing stock request wall: 6.371 h undrafted +3.206 h MTP. Existing w13 GSM8K: .622 h. Output-cap decode alone could be 19.449+11.864=31.313 h; therefore a capped run can legitimately end incomplete. Qualification, startup, partial runs and failed attempts count against the cap; a queue retry cannot reset it.

P8 adds **one descriptive counterpoint row**, labelled “Bonsai 2 27B/NInfer (ternary derivative; none / MTP-3)”, with separately labelled mode/suite cells: derivative+engine/pins, n, accuracy/CI, mean tokens, wall/answer, correct/minute, and explicit normalized-shape footnote. It does not enter original-weight kernel ablations or replace their interpretation. Related work gets **one sentence**, selected by the fixed rules (combine suite-specific clauses with a semicolon when both are reported):

- Complete, Bonsai higher observed correct/minute: “On our fixed 500-item [suite] slice, Bonsai 2 27B/NInfer [mode] yielded X correct answers/min versus Y for [named original lane], at A% versus B% accuracy (paired difference D points, Holm-adjusted p=P).” Higher throughput alone permits “higher observed correct/minute on this slice”; it does not permit “same quality” or “lossless.”
- Complete, Bonsai lower/equal observed correct/minute: use the same sentence with “yielded X ... versus Y”; describe lower/equal observed utility. If Holm finds lower accuracy, say a measured accuracy cost on that suite; if not, say the accuracy difference was unresolved, never equivalent. Mixed suites remain separate, with no universal winner.
- Missing/failed/capped/unacceptable match: “Bonsai 2 27B/NInfer is a relevant ternary counterpoint, but our matched [scope] comparison remains incomplete [named reason]; its published decode speed does not establish same-card accuracy per minute.” Use em-dash/NA cells with reasons, no extrapolated quality row.

For w13, the sentence must name **GSM8K only** unless missing controls were registered and measured before Bonsai outcome inspection. No claim about 98.2% retained intelligence follows from this experiment. No headline novelty for NInfer's published speed. RIFE first, then scoreboard and book/P8 updates with the filed item citation.
