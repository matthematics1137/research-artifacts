# Paper 6 harness notes (2026-09-19)

Implements `research/design-kv-allocation.md` and its binding 2026-09-19
narrowed-question amendment. Everything below is a choice the amendment does
**not** fix, or a place where it is ambiguous; nothing here changes a
pre-registered quantity. Files:

| file | what |
|---|---|
| `make_longctx.py` | frozen seeded generator of the 200 long-context items |
| `longctx_manifest.json` | seed, generator hash, per-item hashes, counts (committed) |
| `p6common.py` | paths, the 24-cell grid, model artifacts, scoring, journal reading |
| `run_cells.py` | server + requests + append-only journals per cell |
| `analyze.py` | estimands, co-primaries, secondaries, equal-memory table |
| `work/` | bulk outputs (items, journals, logs, token counts, analysis) — **not for git** |

Nothing in this harness starts a server or touches the GPU except
`run_cells.py` without `--dry-run`/`--selftest`.

September 20 pre-measurement startup correction: the pinned engine's
`common_params_print_info` build line is trace-level (`COM_TRC`), absent at
ordinary verbosity. The evidence checker now requires the running server's
`/props.build_info` to identify the pinned commit; a build line, when present,
must agree. It no longer requires a line the declared launch does not emit.
The model/context/KV/placement log checks remain mandatory. The owned child
also receives `/opt/p8-user/cuda-12.6/lib64` as its CUDA library path without
mutating the parent's environment. The real binary's `--version` readback
passes with this path, but no model-load smoke or P6 measurement is implied.
Nineteen harness tests and five manuscript tests pass, including the full
synthetic 7,200-outcome replay and missing/wrong build-readback rejection.
No P6 measurement journals existed when these corrections were made.

## A. Item generator

1. **Seed 20260919**, a frozen constant in `make_longctx.py`. All randomness is
   SHA-256 counter mode keyed by `(seed, item id, purpose)`; the session order
   is `sort by SHA-256(seed|item|session id)`. Nothing uses Python's `random`,
   so builds are byte-identical across Python versions and `PYTHONHASHSEED`
   (the self-test rebuilds in a subprocess with a different hash seed).
2. **What "16,384 haystack tokens" counts.** The haystack *including* the
   inserted needles, tokenized with the Qwen3.8 tokenizer
   (`~/models/qwen38-exl3-2.0/tokenizer.json`, SHA-256 recorded). Tolerance
   ±1% = 16,221–16,547; the built items land at 16,340–16,384 (mean 16,371).
   The filler is sized to the target minus the needle tokens.
3. **Session pool.** Unique LongMemEval-S sessions by session id (19,195 of
   them, no id ever carries two different bodies), sorted by id. Excluded: 17
   sessions containing `<|` or `|>` (they would tokenize as control tokens);
   the filters for `<think>` tags, "special magic" and `\bVAR\b` matched
   nothing but stay in so a needle phrasing can never occur naturally.
4. **Rendering.** `"{role}: {content}"` per turn, turns joined by `\n\n`,
   sessions joined by `\n\n`. No session headers, dates or separators — the
   amendment says "rendered as plain chat text". Empty turns are dropped.
5. **Filling to length.** Whole sessions in the seeded order while they fit;
   the first session that does not fit is cut at a turn, sentence or line end,
   accepted only if it lands within 48 tokens of the budget, otherwise skipped
   and the next session tried. The joined text is then recounted exactly and
   trimmed back if it is over.
6. **Needle positions.** Sentence ends: a `.`/`!`/`?` followed by whitespace or
   the end of the turn, not inside a ``` fence, where the next word is not
   lower-case and the previous token is not an abbreviation (`Dr.`, `e.g.`), an
   initial (`J.`, `A.J.`) or a numbered-list marker (`3.`). Turn ends that end a
   sentence are included, so "sentence/turn boundaries" is one rule. A needle is
   inserted as `" " + needle sentence`. The abbreviation/initial/list-marker
   exclusions exist only so a needle does not land inside `A.J. Finn` or between
   a list number and its text; they were added before anything was measured and
   the item set was rebuilt.
7. **Depths.** Needle *j* of item *i* targets `(u + j)/k` with `u = (i+0.5)/50`
   and *k* needles: a rotated even grid, so within an item the needles sit
   `1/k` apart and across the 50 items the whole range is swept evenly. Both
   the target depth and the realized token-position fraction are recorded per
   needle.
8. **Which key is asked (multikey).** `i mod 4`, so the asked needle is
   balanced over the four depth quarters.
9. **Needle and question wording** follows RULER: `One of the special magic
   numbers for {key} is: {value}.`, `VAR {name} = {value}.` /
   `VAR {name} = VAR {prev}.`, and RULER's question forms. Values are 7-digit
   numbers (needle families), the variable-tracking value is 5 digits and its
   names are 5 uppercase letters; keys are `adjective-noun` from a 63x63
   embedded word list. Counts are never revealed in the question.
10. **"5-hop assignment chain, all 5 names required" (ambiguous).** Read as
    **5 assignment statements and 5 names**: one value assignment plus four
    variable-to-variable hops. The alternative reading (5 variable-to-variable
    hops) would need 6 names and contradicts "all 5 names required".
11. **Instruction.** One fixed instruction for every family; only the question
    differs, as required. Layout:
    `INSTRUCTION + "\n\n" + haystack + "\n\nQuestion: " + question`.
12. **Collisions.** A drawn value, key or name that occurs in the filler, in the
    instruction or in any question template is redrawn from the same stream.
    Every item is then validated: each needle occurs exactly once, and no
    required answer, distractor value, key or tracked value occurs anywhere in
    the haystack outside its needles.
13. **Item order in the file is the request order**: families interleaved
    (`single-00, multikey-00, multivalue-00, vartrack-00, single-01, ...`).
14. **Scoring is case-insensitive containment** of every required string, like
    RULER's `string_match_all` and the Paper 1 needle scorer. The amendment
    says "deterministic string match"; case-insensitivity is the choice here.
15. **The Qwen3-8B token count** the amendment asks to report cannot be
    produced offline: no Qwen3-8B tokenizer is on disk and the GGUFs were still
    downloading. `run_cells.py` therefore counts every haystack and message with
    the *served* model's own tokenizer through `/tokenize` (CPU, once per
    model) into `work/token_counts/<model>.json`, and `analyze.py` reports it
    beside the Qwen3.8 counts. The frozen items never depend on it.
16. **The manifest carries no timestamp**, so it is a pure function of its
    inputs and can be re-derived.

## B. Runner

17. **argv** is exactly the pre-registered one, wrapped in `nice -n 5` (house
    style, so the lab machine stays usable). The server is started as its own
    process group and stopped with Paper 5's SIGINT/SIGTERM/SIGKILL sequence on
    that pgid only — never `pkill`.
18. **Only these request fields** are sent: `temperature: 0`, `max_tokens`,
    `cache_prompt: false`, `stream: false` and the chat-template kwargs. No
    `top_p`/`top_k`/`seed`: llama.cpp is greedy at temperature 0, so they
    cannot matter, and leaving them out keeps the server defaults visible in
    `/props`.
19. **Identity per server start** (a `server_start` row in the cell journal):
    GGUF path/size/SHA-256 with its receipt, engine commit + CMake flags +
    binary hashes, full argv, `/props` (chat-template hash, build info), the
    parsed load block (fitted layers, model/KV/RS/compute buffers per device,
    the `--fit` lines, raw matching log lines), and nvidia-smi before start and
    once loaded. A further row records nvidia-smi after the first completed
    request, because CUDA allocates a little more on first use.
20. **Hard refusals** (protocol identity, not operational thresholds): engine
    commit not the pinned one or `GGML_CUDA_FA_ALL_QUANTS != ON`; no pinned
    SHA-256 for a GGUF; GGUF hash mismatch; `work/longctx_200.jsonl` not
    matching `longctx_manifest.json`; a study server on 18125/18126/18127/18130
    or an active P5 phase unit. September 20 hardening additionally rejects a
    dirty engine source, wrong receipt repository/revision, or other GPU model.
21. **GGUF hashes are cached** by (path, size, mtime, ctime) in
    `work/gguf_sha_cache.json`, so a 17 GB file is read once, not once per cell.
    `--no-hash` skips hashing and records `verified: false`; it is now refused
    in canonical campaign journals and may only be used for isolated smoke work.
22. **Prompt-token parity is recorded, not gated.** Paper 5 aborts on a
    mismatch; here the harness count (Qwen3.8 tokenizer + the template wrapper)
    and the server's `usage.prompt_tokens` are both stored with their delta.
    A one-token difference between `tokenizers` and llama.cpp must not kill an
    overnight run; `analyze.py` reports the deltas.
23. **`--fit` placement changes** between restarts (Matt using the GPU can
    change free VRAM) are now recorded and stop further requests until inspected.
    This checks model/KV/recurrent buffer placement as well as layer counters,
    including restarts in the first invocation. Do not silently discard the
    recorded changed start or treat it as an approved protocol deviation.
24. **Timeouts** (operational, CLI-adjustable): 1,800 s per long-context
    request, 5,400 s per GSM8K request; 2 retries. A 4xx is terminal without
    retry (a rejected request will be rejected again); connection errors and
    timeouts stop the server, which the gate restarts.
25. **A failed item is missing, never wrong.** Terminal failures are journaled
    and excluded from accuracy; `--retry-failed` re-sends exactly those.
26. **Journals** store the response content, `reasoning_content`, `usage`,
    `timings`, `finish_reason`, wall seconds, the request (long-context prompts
    by SHA-256 + length, since the text is in the frozen item file; GSM8K
    prompts in full) and the score.
27. **Scores are computed at run time and recomputed at analysis** from the
    stored text; disagreements are flagged.
28. **Order**: 200 long-context items then GSM8K-100, in file order, one cell
    per server load (cells are not interleaved — each load is ~10 GB of weights).
29. **Pause/thermal**: Paper 6's own pause file `testsuite/paper6/PAUSE`, and
    Paper 5's thresholds (87 C stop / 75 C resume) as defaults, both adjustable
    on the command line. Pausing stops the server and journals a `pause` event.
    As of September 20, `--idle-seconds` defaults to 10 after each new attempt
    to preserve the owner's gentle-operation preference. Cached outcomes do
    not wait. This changes average utilization/wall time, not peak VRAM use;
    the reported per-request timer does not include the subsequent idle.
30. **`--limit` refuses to write into `work/journals`** so smoke tests cannot
    leave partial cells in the campaign journals.

## C. Analysis

31. **Complete cases.** Every estimand uses the item ids scored in all of its
    cells (4 for I, 8 for M); `n` is reported and any shortfall is flagged.
32. **Outcome per job**: the first terminal ok row wins; a terminal failure with
    no ok row counts as missing.
33. **Primary statistics** (stdlib + numpy): Wilson
    intervals with z from `NormalDist`; exact-binomial two-sided McNemar
    (`min(1, 2 * P(X <= min(b,c)))`, exact integer arithmetic); Student-t from
    the regularized incomplete beta (continued fraction) with quantiles by
    bisection; sign-flip permutation
    `p = (1 + #{|S*| >= |S_obs|}) / (1 + draws)` (Phipson & Smyth, so p is
    never 0), 100,000 draws, seed 20260919, in 5,000-row chunks.
    The separate September 20 verifier uses the now-available SciPy installation
    for an independent numerical check; it does not replace these estimators.
34. **Units**: estimands in accuracy points; 95% t interval for M and for every
    paired difference, 90% for the TOST interval, margin ±10 points.
35. **Holm covers exactly the two co-primaries.** GSM8K D/I/M, its permutation
    p and its TOST are reported as secondary, outside that family.
36. **"Trend over tiers"** is reported as D per tier with intervals plus the two
    adjacent paired contrasts, D(Q2)−D(IQ3) and D(IQ3)−D(Q4). A single linear
    contrast over three equally spaced tiers would just restate I.
37. **Depth bins**: 5 equal-width bins. Item accuracy is binned by the asked
    needle's depth (single, multikey); per-required-answer recall is binned by
    that answer's own needle depth (all families, which is the only sensible
    unit for multivalue and vartrack).
38. **Fully resident** (the equal-memory scope) means: every layer offloaded
    (`offloaded X/Y` with X == Y), no KV or recurrent buffer on the host, and
    no layer split by `--fit` (it can keep `X == Y` while pushing part of a
    layer to the CPU). The first `server_start` row of the cell supplies the
    numbers; restarts and placement changes are counted.
39. **KV at 16,384 and 32,768 tokens** is scaled from the measured buffer and
    the reported cell count (the server runs at `-c 20480`), which matches
    "bytes at 32K are reported as computed". Measured buffers at 20,480 cells
    are reported too, and the exchange-rate line uses resident cells only.
40. **Distractor hits are recorded** (multikey): containment gives credit to an
    answer that lists all four values, so the count of credited items that also
    contain a distractor value is available for audit.

## D. Known risks, recorded not resolved

- **Greedy thinking on GSM8K.** The amendment fixes temperature 0; Qwen3 model
  cards warn that greedy decoding in thinking mode can loop. Per-cell
  truncation counts (`finish_reason == "length"`) and mean completion tokens
  are in the analysis so this is visible.
- **Partly offloaded cells.** The 27B UD-Q4_K_XL (and possibly others at f16
  KV) will not be resident. Their accuracy still enters the interaction
  (amendment §3); they are excluded from the equal-memory table only.
- **Cross-model pairing.** M pairs the same 200 item ids across two models.
  This is what the amendment asks for; the items are identical text, but the
  two models see different tokenizations of it (reported).
- **Ceiling/floor.** Accepted in the amendment: if a cell answers ~100% or ~0%,
  paired flips vanish and power drops. Per-family and per-depth tables make it
  visible early — the first cell run is worth inspecting before committing the
  whole grid.

## E. September 20 prospective evidence hardening — no P6 cells measured

`p6evidence.py` now gates actual-result manuscript and figure commands:

- Full SHA-256s bind the runner, analyzer, verifier, frozen generator, shared
  runtime/scorer sources and design document. The frozen item bytes and
  `p6common.py` were not changed. Keep this measurement harness unchanged
  once the campaign begins; later corrections need a recorded migration.
- Every server start has a unique instance ID, PID, GPU/driver identity,
  the in-tree engine binary/shared-library hashes, and a hash of the exact load
  log prefix. Responses are bound to that active instance. A restart cannot
  silently change the recorded model, request, template or placement.
- Resume audits existing journals before any new request. It reconstructs
  requests, checks typed fields/token counts and re-scores answers. Failed
  attempts may precede one success; duplicate successes, post-success retries,
  foreign IDs, changed requests, missing identities and wrong scores are rejected.
  Missing successful outcomes remain a hard final-draft blocker, never a zero.
- Token-count inventories bind the frozen items, source hashes, model and
  originating server; they are created atomically without overwriting. An
  alternate journal directory uses its own sibling `token_counts` directory.
- Analysis hashes the journals, token inventories and load-log prefixes and
  refuses a changing checkpoint. Final verification replays all tables and
  checks paired t intervals/TOST with SciPy, exact McNemar with its binomial
  test, and the seeded sign-flip counts with separate integer arithmetic.
  Long-context rescoring is independently written; GSM8K deliberately keeps
  the unmodified Paper 1 scoring definition. Raw-log parsing is shared with
  capture and checked against additional protocol constraints, not claimed
  as an independently implemented log parser.

Verified locally with NumPy 2.3.3 and SciPy 1.16.2. Eleven evidence tests
include a complete synthetic 7,200-outcome campaign, adverse journal mutations,
raw-log changes, token-count isolation, pacing and resource-overlap guards.
The seven reporting tests, five manuscript/figure tests and both older
self-tests also pass. Fixtures live only in temporary directories and are
discarded. No real P6 result exists yet and no publication approval is implied.
The six-PDF synthetic layout test also passes: 7/7 main pages, 3/3 explainer
pages and 1/1 brief pages, without TeX/reference/overflow errors. Its figure
fixture calls the pure drawing function inside the disposable labelled test;
there is no synthetic-data bypass in the real figure command.

Run short CPU-only regressions with `make -C paper6 test-evidence`,
`make -C paper6 test-reporting`, and `make -C paper6 test-builders`. After the
real campaign and analysis, `make -C paper6 all` checks the raw evidence before
rendering actual results. The private preparation build remains explicitly
labelled results pending and does not bypass the actual-result gate.

The machine inventory/live-headroom check still belongs before any heavy
launch. Port and identity guards are additional checks, not a scheduler or
proof that a machine currently has enough free resources.

## 2026-10-29 correction before the first cell

No Paper 6 measurement exists yet. Paper 5's gpt-oss judge, served by the same
pinned llama.cpp revision `035e227`, showed that this revision routes the
library's info messages to log level 4 while the server's default is 3. At the
default level the `load_tensors: offloaded X/Y layers to GPU` line and the buffer
sizes, which `parse_server_log` needs for residency, are not printed at all.
`P6Server.cmd_env_cwd` now sets `LLAMA_ARG_LOG_VERBOSITY=4` for the owned
server only. This is the revision's environment form of `-lv 4`. It is logging
only: the server argv, the cells and the requests are unchanged. At this level
the build line (`build N (035e227)`) is also printed, and the existing check
requires it to agree with the pin. The 19 harness tests pass, including a new
assertion for the variable. The environment record (kernel, libraries, Python
analysis packages) is re-captured by the harness at the first real cell.

## 2026-10-30 correction before the first cell: no file deletion

Still no Paper 6 measurement. Since 2026-10-29 a machine guard stops every agent from deleting files.
It also refuses to run any script that contains a delete call. `run_cells.py` had two such calls, and
neither needs to delete anything:

- **Token counts.** `tokenize_pass` wrote each per-model token-count file to a temporary file, linked it
  into place create-only, then removed the temporary. It now opens the final path create-only (`"x"`)
  and writes and fsyncs there. The create-only rule is unchanged, and an existing file is still
  validated, never overwritten. A torn write would fail that validation on the next read, so it stops
  the run instead of being reused.
- **Self-test.** The pause-marker check now lifts the marker by moving it aside (`PAUSE.lifted`)
  instead of removing it, as the load guard does.

Cells, requests, scoring and journals are unchanged. Inside `zarael-sandbox` with the repository's
`engines/exl3-env` Python, `--selftest` passes and 18 of the 19 harness tests pass. The 19th
(`test_reporting`) needs scipy, which that environment lacks; it does not touch the changed code.


## Operational changes after the 2026-10-31 pause (recorded 2026-10-31 20:20 EDT)

Matt paused P6 at about 20:10 EDT to let the laptop cool and asked for the campaign to be made faster before it
continues. The full reasoning, with the measured time split per cell, is in
`research/P6_EFFICIENCY_RETHINK_2026-10-31.md`. Nothing in the hashed harness changes: `p6evidence.py` checks
every journal row against the SHA-256 of the harness files, so an edit there would fail the seven finished cells'
audit. Everything below uses controls the harness already has.

- **Pause.** `PAUSE` was created by hand at 20:12:35 EDT. The runner finished the job in flight and stopped its
  server at 20:13:07 EDT (27B IQ3_XXS k8v4, 97 of 200 long-context items done).
- **CPU thermal guard.** New `tools/thermal_guard.py`, run as unit `p6-thermal-guard` beside the load guard, on
  the same `PAUSE` marker and with the same ownership rule: it lifts only markers whose text starts with
  "thermal-guard", and moves them to `work/logs/pause_history/` instead of deleting them.
  - It pauses when coretemp's "Package id 0" 60 s mean reaches 92 C.
  - It resumes once the mean has stayed below 80 C for 120 s and the pause has lasted 180 s.
  - The runner's own GPU rule (87/75 C) is unchanged.
  - Log: `work/logs/thermal_guard.log`, with one line per minute of CPU mean, CPU maximum and GPU temperature.
- **Idle.** `--idle-seconds 3` for the 12 Qwen3-8B cells, which are fully resident. It stays 10 for the partly
  offloaded 27B cells. Each `server_start` row records the value in `decoding.idle_seconds`, and no check or
  analysis reads it.
- **Order.** `launch_rest.sh` runs `run_cells.py` three times in sequence, with the same unit name and caps as
  `launch_campaign.sh`:
  1. `qwen38-27b:IQ3_XXS:k8v4` (idle 10);
  2. `qwen3-8b:*:*` (idle 3);
  3. `qwen38-27b:Q4_K_XL:*` (idle 10).

  One spec always runs in canonical order, hence three calls. Each cell has its own server and journal.

Cells, requests, server flags, scoring and journals are unchanged. `--dry-run` of the three specs in the sandbox
plans 203, 12 × 300 and 4 × 300 jobs. The guard's `--selftest` passes 16 of 16 checks.
- **Superseded the same evening: primary cells first** (2026-10-31 20:32 EDT). By Matt's decision, `launch_rest.sh` runs only the
  long-context passes of the six co-primary cells still missing, Qwen3-8B before 27B, all at `--idle-seconds 3`. The
  rest of the grid waits for a later decision on cost and heat. Reasons are in
  `research/P6_EFFICIENCY_RETHINK_2026-10-31.md`, "Decision: the primary cells first". The three-step order above is
  kept as written.
- **Thermal guard lines moved to 95/85 C** (2026-10-31 20:38 EDT). At 92/80 the guard paused the first Qwen3-8B cell two minutes in
  (60 s mean 92.4 C at 20:34:50 EDT, with the model fully resident and the GPU at 80 W). It resumed it after the
  minimum 180 s (mean 61.9 C). At that line P6 would run well under half the time. With the GPU busy, about 90 C is
  this laptop's normal CPU level, because the GPU heats the CPU through the shared cooler. The line now marks
  sustained heat beyond that, still 5 C under the 100 C point where the CPU throttles itself. The guard was restarted
  at 20:38:19 EDT, and the self-test passes 16 of 16 checks.
- **Efficiency-core placement tested and reverted** (2026-10-31 20:47 EDT). From 20:40:00 to 20:47:01 EDT, `taskset` held the runner
  and llama-server on CPUs 16-31. The CPU was about 3 C cooler, and host-side time per job rose about 0.6 s, so it
  was reverted by its pre-written rule. Details are in `research/P6_EFFICIENCY_RETHINK_2026-10-31.md`. Rows written in
  that window carry the same harness and server flags as all others.
- **Load guard message** (2026-10-31 21:23 EDT). `testsuite/paper5/load_guard.py` logs "RAM available below 8 GiB" for its two-sample
  pause, but the threshold in its code is 6 GiB (`PAUSE_MEM_AVAILABLE`). The pause uses 6 GiB. The file is
  Paper 5's frozen harness and stays unchanged. Found while drafting Volume I.5, chapter 6.

- **Placement refusal on the last cell, inspected and resumed** (2026-11-01 05:12 EST).
  - **What happened.** At 02:07:03 PDT, after its third thermal-guard pause, qwen38-27b Q4_K_XL q4_0 restarted at
    190/200 with `--fit` choosing 37/66 layers instead of 36/66. The harness refused as designed (item 23).
  - **Cause.** `--fit` saw 10,887 MiB free against 10,830 at the 01:40 start, a difference of 57 MiB. Its 1,024 MiB
    margin moved the target from 9,806 to 9,863 MiB, and that admits layer 37 (9,853 MiB in total, against 9,648 for
    36).
  - **The record.** The changed start is kept in the journal as a server_start with placement_changed=true. No
    request ran under 37 layers.
  - **Window.** `llama-server --list-devices` measures llama's own CUDA context at 195 MiB of device memory. So the
    36-layer window, in nvidia-smi free memory just before launch, is about 11,017-11,222 MiB.
  - **Resume.** Relaunch only this cell, with launch_rest.sh's unit settings, when pre-launch free memory is inside
    that window. The harness's own placement check decides; nothing in it is changed or bypassed.
  - **If a later restart lands outside the window,** the harness refuses again. Free memory is then brought back into
    the window, and the run relaunches.

- **Correction to the entry above** (2026-11-01 05:47 EST). Its title says "resumed", but the 02:12 PDT relaunch was refused by the
  journal audit, because a journal holding a changed start cannot be resumed. The cell stops at 190/200; see
  `research/P6_AMENDMENT_2026-11-01.md` section 3. Nothing ran after the refused start.
