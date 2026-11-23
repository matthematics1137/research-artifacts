# Turing lab: output-quality harness (Paper 8)

This harness asks whether the 2080 Ti's output matches the laptop's in quality, not only in speed. It
replicates Paper 7's E47 quality test on the desktop's RTX 2080 Ti for every serving lane and compares the
result with the laptop's stored rows item by item. E47 is the greedy GSM8K-500 + MMLU-500 run of the daily
EXL3 config, with and without the MTP drafter (`testsuite/results/EFFICIENCY-CAMPAIGN.md`, E47).

## Files

| File | Runs on | What it does |
|---|---|---|
| `run_eval_turing.py` | desktop (lab venv python; stdlib only, never touches the GPU) | A fork of `testsuite/evals/run_eval.py` that never discards files. Prompts, request bodies, scorers and rows are identical; additions are listed below. |
| `run_quality.sh` | desktop, inside a capped `turing-run-q-*` unit | Runs one lane on one suite with E47's flags and E47's response shape. Starts and stops the lane and appends a run record. |
| `compare_quality.py` | laptop or desktop (stdlib only) | Pairs Turing rows with laptop rows by item id and writes markdown and JSON. |
| `check_fidelity.py` | laptop, CPU only | Proves the fork and the comparison are faithful to the E47 tools. |

## What "identical to run_eval.py" means, and the proof

`check_fidelity.py` run on 2026-10-30 (sandbox, CPU only, 3 s) with every check passing:

- **Static.** The 15 functions shared with `run_eval.py` are source-identical: the prompt builder, every scorer
  and math normalizer, `chat_completion`, the thinking estimate and `visible_content`. The constants match. The
  dataset sha256 tables in the fork and in `run_quality.sh` equal the laptop's files, and `run_quality.sh`
  passes E47's flags. Its `e47_shape()` is run by bash on seven override cases and adds E47's response shape
  (see below) unless the caller already set that key.
- **Rescoring.** All 2,000 stored E47 rows (`EFF-h38a_mtp` and `EFF-h38a_nodraft` × `gsm8k_500` and
  `mmlu_500`) were rescored from their stored text.
  - The fork's scorer equals `run_eval.py`'s scorer on 2,000/2,000 rows.
  - It reproduces the stored correctness on 2,000/2,000 rows.
  - 208 of those rows store the whole visible answer, so their check is exact. The other 1,792 store only the
    last 200 characters (`raw_answer_excerpt`) and still agree on every row.
- **Requests and rows.** Both programs parse their own command lines: E47's flags from `hardening.sh` for
  `run_eval.py`, and the command `run_quality.sh` actually builds for the fork. Every one of the 1,000 items
  then gets byte-identical requests (URL, body, headers, method, timeout) and identical rows.
  - Server shapes tested: TabbyAPI with `usage: null` plus `/v1/token/encode`, llama.cpp `usage` plus
    `timings`, inline `<think>`, `chat_template_kwargs` rejected with HTTP 400, one dropped connection, a hard
    failure, and a failing encode route.
  - Four more flag sets were tested too.
  - In total, 15,000 cases agree on requests, rows, `request_body_bytes()` and the sidecar's token source.
- **Statistics.** `compare_quality.py` reproduces all 12 stored `hardening_analyze.py` outputs
  (`testsuite/results/efficiency/rows/h38*.json`, E47-E52) and all 3 stored `identity_anatomy.py` outputs,
  field for field.
- **Resume.** Against a fake server, these behaviours were confirmed:
  - error rows are retried and finished items are skipped;
  - the latest row per id wins, and nothing is truncated;
  - a torn last line is survived;
  - three failed requests in a row stop the suite;
  - `--todo` counts the items left.
- **Response shape.** The same generated texts are served in E47's shape and in upstream TabbyAPI's two shapes,
  and `compare_quality.py` must report the same numbers for both upstream shapes. The texts are the 25 E47
  traces in `rows/traces_h38c_nodraft.jsonl`, plus one item cut inside its reasoning and five items with one
  word more. The shapes:
  - E47's lab TabbyAPI, to `run_eval.py`: inline, `usage: null`;
  - upstream b1c2253 with `reasoning: false`, to the fork: inline, `usage`;
  - upstream's defaults, to the fork: split by upstream's own `TagStreamParser`, which is loaded from the bare
    source mirror and fed in chunks of 1-7 characters.

  Byte identity, matched token identity, mean tokens, the anatomy and the inline-path accuracy agree (0.8, 0.8,
  identical anatomy). The split side's raw identity is 0.04, the defect this guards against. The CHECK line
  fires for the split side only, and the one-word items land in the `1-5` token-delta bucket on the matched basis.
- **Lane table.** A 10-item rerun and a 15-item variant leave the stock lanes' 40 common items alone.
- **Self-test of the checker.** Six deliberate mutations of the fork were each caught: request timeout,
  `max_tokens`, scorer, a row field, prompt text, and the dataset hash. Five in-memory reversions of the fixes
  above were caught too:
  - raw `content_sha256` used across shapes;
  - the split-basis encode count;
  - anatomy deltas on mixed counts;
  - a sidecar that drops the tag;
  - a lane table that intersects reruns.

The fork's additions never change a row:

- **No code-execution suites.** HumanEval and HumanEval+ are left out, because their sandbox cleanup is the
  only code in `run_eval.py` that discards files.
- **Dataset check.** Each dataset's sha256 is checked before a run, and the run refuses on a mismatch.
- **Resumable runs.** Rows are appended. An item whose latest row has no `error` is skipped on a rerun, and
  readers take the latest row per id, as `hardening_analyze.py` does.
- **Full-response sidecar.** `<suite>.full.jsonl` holds the reasoning and answer text, the finish reason,
  `usage`, the server's `timings` (decode tok/s and draft acceptance), the request sha256, and where the
  server supports it, the `/v1/token/encode` count. The laptop's TabbyAPI rows counted tokens that way;
  upstream TabbyAPI reports `usage` instead.
  - The sidecar also records the response shape (`response_shape`: `inline` or `split`), any `tool_calls`,
    and `inline_sha256`, the sha256 of the text in E47's inline shape.
  - For a split response it adds `inline_encode_tokens` and `inline_correct`: the encode count and the score
    of that inline text.
  - `laptop_inline` says how the inline text was rebuilt. It is exact for TabbyAPI. Two edge cases are
    approximate: content of whitespace alone after the tag, and llama-server's trimmed whitespace.
- **Clean stops.** SIGTERM and SIGHUP are handled like Ctrl-C, so the partial summary is still written. A
  suite stops (exit 3) after `--max-consecutive-errors` (default 3) failed requests in a row.
- **Helper flags.** `--summary-out` writes the summary for the run record, and `--todo` prints the number of
  items left.

## On the desktop

Layout inside `$L=/opt/p8`; nothing is written outside it:

| Path | Contents |
|---|---|
| `tools/quality/` | `run_eval_turing.py`, `run_quality.sh` and `compare_quality.py`, copied from this folder |
| `evals/datasets/` | `gsm8k_500.jsonl`, `mmlu_500.jsonl` (copied from the laptop; the runner only checks them) |
| `evals/results/<label>/<suite>.jsonl` | rows in `run_eval.py`'s format, plus `<suite>.full.jsonl` and `summary.jsonl` |
| `runs/quality_runs.jsonl` | one record per run. It holds the times, lane, status and exit codes, plus the engine identity (exllamav3, torch and triton versions; TabbyAPI and both llama.cpp commits with dirty-file counts; driver). It also holds the tool sha256s, the lane's `ExecStart`, `/v1/models`, `nvidia-smi` before, after load and at the end, the GPU sample summary and the eval summary |
| `runs/quality_gpu/<run>.csv` | `nvidia-smi` every 5 s during the eval: temperature, power, SM and memory clocks, VRAM, utilization |
| `tmp/` | lane start output and per-run summaries (left in place) |

Deploy (validator):

```bash
L=/opt/p8
ssh p8-host "mkdir -p $L/tools/quality $L/evals/datasets"
scp tools/turing/quality/{run_eval_turing.py,run_quality.sh,compare_quality.py} p8-host:$L/tools/quality/
scp testsuite/evals/datasets/{gsm8k_500,mmlu_500}.jsonl p8-host:$L/evals/datasets/
```

Run one lane on one suite. This is the only GPU step, and it goes through the capped, guarded launcher, one
job at a time:

```bash
bash $L/tools/desktop_run.sh cmd q-LANE-SUITE bash $L/tools/quality/run_quality.sh LANE SUITE [LIMIT] [LABEL]
#   e.g. a 20-item plumbing pilot, then the full suite (the second command resumes after the first):
bash $L/tools/desktop_run.sh cmd q-exl3-mtp-gsm8k_500 bash $L/tools/quality/run_quality.sh exl3-mtp gsm8k_500 20
bash $L/tools/desktop_run.sh cmd q-exl3-mtp-gsm8k_500 bash $L/tools/quality/run_quality.sh exl3-mtp gsm8k_500
#   same-config reproducibility rerun (fresh server, own label), first 100 items:
bash $L/tools/desktop_run.sh cmd q-exl3-nodraft-r2 bash $L/tools/quality/run_quality.sh exl3-nodraft gsm8k_500 100 TURING-exl3-nodraft-r2
#   another port: ... cmd q-x env PORT=8091 bash $L/tools/quality/run_quality.sh ...
```

The run uses E47's flags: `--temperature 0`, reasoning effort `medium`, `--max-tokens 4096`, concurrency 1. The
default label is `TURING-<lane>`. Progress goes to `$L/logs/run-q-LANE-SUITE.log`.

**Needed in `lanes.sh` before the first run. Applied 2026-10-30 during the desktop validation (exactly the
function below); the 5-item pilots ran lanes.sh from inside their `turing-run-q-*` units.** Before that,
`lanes.sh start` refuses while any `turing-run-*` unit is active (`active_runs()`). That includes the runner's own
`turing-run-q-*` unit, which never touches the GPU. So every quality run would end `lane_start_failed` with
"another lab GPU unit is active". The shared lane interface only refuses for another srv lane.

`active_runs()` should skip exactly the unit that `lanes.sh` itself runs in. It reads that unit from its own
`/proc/self/cgroup`, which is the same grep `run_quality.sh` uses. It must not trust an environment variable,
because any caller could set one to exempt an arbitrary unit, a real GPU job included:

```bash
active_runs() {  # every lab unit that may hold the GPU, except the unit this script runs in (a quality runner)
  local self
  self=$(grep -o 'turing-run-[A-Za-z0-9_.-]*\.service' /proc/self/cgroup 2>/dev/null | head -1)
  systemctl --user list-units --plain --no-legend --state=active,activating,deactivating 'turing-run-*' 2>/dev/null \
    | awk -v self="$self" '$1 != self {print $1}' | tr '\n' ' '
}
```

Run from an ssh shell, `self` is empty and nothing is skipped. Every other `turing-run-*` unit is still refused.
`run_quality.sh` keeps the one-GPU-job rule itself as well: it refuses if any other `turing-run-*` unit is active.
Deploy the changed `lanes.sh` before the pilot.

**E47's response shape (applied by `run_quality.sh` on every lane).** E47's server was the lab TabbyAPI 4a4f9f4.
With its defaults, the reasoning parser and tool parsing were off, and no laptop config turned them on. So it
returned everything inline: `content` = `'<reasoning>\n</think>\n\n<answer>'` with `reasoning_content` empty
(`rows/traces_h38c_nodraft.jsonl`).

The desktop's servers split the reasoning out by default:
- upstream TabbyAPI b1c2253 (commit 2186cdb, 2026-09-22) parses reasoning by default with tag auto-detection, and
  auto-detects a tool format;
- llama-server uses `--reasoning-format deepseek` by default.

A split removes the `'</think>'` string, which causes three problems:
- token-identical output gets a different `content_sha256`, so byte identity against the laptop would read about 0
  whatever the 2080 Ti generates;
- the encode count is one token short;
- an item truncated inside its reasoning scores on empty content, while E47's fallbacks scored its reasoning text.
  For example, E47 undrafted `mmlu_2546` was truncated and scored correct.

So `run_quality.sh` adds E47's settings to the lane overrides, unless the caller already set that key:
- the EXL3 lanes get `TABBY_MODEL_REASONING=false TABBY_MODEL_TOOL_FORMAT=`. TabbyAPI reads `TABBY_MODEL_<KEY>` for
  keys the lane yml leaves out;
- the llama.cpp lanes get `--reasoning-format none`, which puts EXL3 and GGUF on one scoring path.

The run record stores what it added as `e47_shape_added`. If the lanes owner later puts `reasoning: false` and
`tool_format:` in `tabby/exl3-*.yml`, and `--reasoning-format none` in `LCPP_COMMON`, these defaults change nothing.
If a lane splits anyway, `compare_quality.py` falls back to the sidecar's inline fields and prints a CHECK line.

**Lane overrides.** These pass through to `lanes.sh` with `env`, and every run record stores them:

- `TURING_LANE_ENV`, `TURING_LANE_CTX`, `TURING_LCPP_EXTRA`. E47's response shape is added to the first and third
  (see above).
- Plan 6.2 item 6 pins the GDN recurrent state to fp32 for any comparison with the laptop's 1.4.7 numbers. Run the
  VRAM check in step 4b before scheduling it. Use your own label for that run:
  `... cmd q-exl3-mtp-fp32 env TURING_LANE_ENV=TABBY_MODEL_RECURRENT_STATE_FP32=true bash $L/tools/quality/run_quality.sh exl3-mtp gsm8k_500 0 TURING-exl3-mtp-fp32`.
  `compare_quality.py` pairs `TURING-<lane>-<variant>` labels with the stock lane, and the EXL3 ones with E47 too.
- The run record also embeds the lane's own start row from `$L/runs/lanes.jsonl` (command, context, probe, and
  the server's view of its parameters).
- `lcpp-iq2xxs-dflash2` runs at 8K context by the lane's default. That is enough here: the prompt is under 1K
  tokens and the generation cap is 4,096.

The runner refuses to start in these cases, each with its own exit status and a run record:

- the dataset hash differs (65);
- another quality run holds the lock (75);
- any other `turing-run-*` unit is already active (75);
- the lane name is unknown (64).

If every item already has a row, it records `already_complete` without loading the lane.

Whatever happens, an EXIT trap stops the lane and appends the run record: a normal end, a failure, Ctrl-C, or
the guard stopping the unit. To continue after a stop, rerun the same command.

Tested on the laptop against a fake lab (a fake `lanes.sh`, a fake OpenAI server and a fake `nvidia-smi`; no
GPU), these cases all behaved as intended:

- normal run, resume, and nothing-left;
- a SIGTERM to the runner mid-eval: the child stopped cleanly, the partial summary was written, the lane was
  stopped and a record was appended;
- a wrong dataset, a held lock, and an unknown lane;
- after the response-shape change, three more runs against a fake `lanes.sh` that records its overrides and then
  refuses the start. The lanes were `exl3-mtp`, `lcpp-iq2s`, and `exl3-nodraft` with the fp32 variable. `lanes.sh`
  received exactly E47's shape, merged with the caller's variable, and each run ended `lane_start_failed`
  (exit 69) with a run record that carries `lane_env`, `lcpp_extra` and `e47_shape_added`.

The fake server saw only E47's body shape: `temperature 0.0`, `max_tokens 4096`, `top_p 0.95`, `top_k 20`,
`chat_template_kwargs.reasoning_effort = medium`.

### Suggested order and rough durations

These durations are estimates. They scale E47's laptop wall times by the measured 2080 Ti decode rates, and
add about 75 s of model load per run.

| Step | Runs | Answers | Estimate |
|---|---|---|---|
| 1 | `exl3-mtp gsm8k_500 20` | Plumbing: thinking present, no `ctk dropped`, token source, decode tok/s in the sidecar. Shape: the sidecar's `response_shape` is `inline`, `content` holds `'\n</think>\n\n'`, `reasoning_content` and `tool_calls` are null, and `e47_shape_added` is in the run record. Repeat the shape check on the first llama.cpp lane | 5 min |
| 2 | `exl3-mtp` and `exl3-nodraft` on `gsm8k_500` | E47 replication, plus whether the drafter is lossless and sample-identical on 1.6.5 / sm_75 | ~1 h + ~2 h |
| 3 | `exl3-nodraft gsm8k_500 100 TURING-exl3-nodraft-r2` | Is greedy decoding bit-reproducible on Turing? Gives the identity baseline for every other pair | ~25 min |
| 4 | `exl3-mtp` and `exl3-nodraft` on `mmlu_500` | E47 replication on MMLU | ~2.5 h + ~4 h |
| 4b | VRAM check (below), then `exl3-mtp gsm8k_500` with fp32 GDN state, label `TURING-exl3-mtp-fp32`, or `exl3-nodraft` if it does not fit | Does the stock fp16 recurrent state change answers? (plan 6.2 item 6) | 2 min + ~1 h (nodraft: ~2 h) |
| 5 | the four `lcpp-*` lanes on `gsm8k_500`, then on `mmlu_500` | Same-card quality of UD-IQ2_S / UD-IQ2_XXS, with and without drafters, against EXL3 | ~1-2 h + ~2.5-4.5 h each |

**Step 4b's VRAM check.** Upstream documents fp32 GDN state as twice the state and checkpoint memory. On the
laptop, the drafter's per-slot recurrent states took about 1,020 MiB at draft 6 in fp32 (E28). The 2080 Ti
already measured 9.5 GB at draft 4 with 16K context in fp16, against about 10.4 GiB free. Draft 6 with fp32 may
therefore fail to load, or fail mid-run as HTTP 503 error rows and then a `server_gone` stop. The guard makes this
safe, but the run would be wasted. Check before scheduling it:

1. **Load-only check.** From an ssh shell, with no other `turing-run-*` unit active, load the lane with the
   variant's environment and record its VRAM. The start also runs the drafted 16-token probe, and `lanes.jsonl`
   records `gpu_after` too. Then stop it:

   ```bash
   TURING_LANE_ENV="TABBY_MODEL_REASONING=false TABBY_MODEL_TOOL_FORMAT= TABBY_MODEL_RECURRENT_STATE_FP32=true" \
     bash $L/tools/lanes.sh start exl3-mtp
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
   bash $L/tools/lanes.sh stop exl3-mtp
   ```

2. **Decide.** Headroom is `memory.total` minus `memory.used`.
   - If the load fails, or the headroom is under about 300 MiB, run the fp32 variant on `exl3-nodraft` with label
     `TURING-exl3-nodraft-fp32`. That is still a valid laptop comparison: `compare_quality.py` pairs it with
     `EFF-h38a_nodraft`.
   - Otherwise run a 20-item pilot of the drafted variant: `run_quality.sh exl3-mtp gsm8k_500 20
     TURING-exl3-mtp-fp32`. Read `gpu_samples.mem_used_max_mib` and `eval_summary.errors` from its run record.
     Continue only if the headroom during generation stays at or above 300 MiB and there are no error rows.
     Otherwise switch to `exl3-nodraft` as above. The 20 rows stay, and the lane table scores them without
     shrinking the common set.
   - A shorter draft (for example 4) needs a lane config change by the lanes owner. `draft_num_tokens` is set in
     the yml, and `TABBY_*` environment overrides only reach keys the yml leaves out.

   Record the choice and the measured headroom with the run.

## Comparing (laptop)

```bash
rsync -a p8-host:/opt/p8/evals/results/ testsuite/results/turing/evals/
scp p8-host:/opt/p8/runs/quality_runs.jsonl testsuite/results/turing/
python3 tools/turing/quality/compare_quality.py --md testsuite/results/turing/quality_report.md \
    --json testsuite/results/turing/quality_report.json
python3 tools/turing/quality/compare_quality.py --pair gsm8k_500:EFF-h38a_nodraft:TURING-lcpp-iq2s   # extra pairs
```

On the desktop, pass `--turing-dir $L/evals/results --laptop-dir <copy of the laptop rows> --datasets-dir
$L/evals/datasets`.

The report has these sections:

- **Turing lanes table**, one per suite, with these columns:
  - accuracy with its Wilson CI;
  - accuracy on the common items, which every stock lane (`TURING-<lane>`) finished without error. Reruns and
    variants such as `-r2` (100 items) or `-fp32` are scored on their overlap with that set, with n shown when
    smaller, and never shrink it. A note names any stock lane that still limits the set;
  - mean tokens, truncations, wall per item, and accuracy per minute (`run_eval.py`'s definition);
  - median decode tok/s and pooled draft acceptance, from the sidecar timings;
  - thinking share, `ctk dropped`, and the response shape, with any parsed tool calls.
- **Pair tables.** Each row has these fields:
  - n, with errors and missing items;
  - both accuracies with Wilson CIs, delta, the A-only / B-only counts and the exact McNemar p;
  - byte identity with its CI. If one side is inline and the other split, it is computed on E47's inline text,
    from the sidecar's `inline_sha256`, and the raw value is shown next to it;
  - the same final answer;
  - token identity and mean tokens. If the two sides counted tokens differently (laptop encode vs `usage`), they
    are shown on the matched encode basis, and the raw `hardening_analyze` values are marked "mixed basis";
  - wall per item with the A/B ratio, and truncations.

  A `CHECK:` line appears in these cases:
  - the two sides differ in thinking share, dropped template kwargs, or token basis (with no counts to match);
  - the response shapes differ. The line then gives the accuracy and McNemar p on E47's inline scoring path, and
    the ids that score differently on it;
  - a shape is unknown;
  - the server parsed tool calls.

  The JSON adds the discordant item ids and `identity_anatomy.py`'s breakdown for every pair. Its token deltas
  use the matched basis when the two sides count differently (`token_basis` says which), and it uses the inline
  fingerprints when the shapes differ.

### Default pairs (A = reference, B = compared)

| Group | Pairs | Why |
|---|---|---|
| replication | `EFF-h38a_mtp` vs `TURING-exl3-mtp`, `EFF-h38a_nodraft` vs `TURING-exl3-nodraft`, on both suites | E47 on the 2080 Ti: same config, same items |
| engine reference | `EFF-h38f_nodraft_rel1` vs `TURING-exl3-nodraft` and `TURING-exl3-mtp`, first 100 GSM8K-500 items | The laptop on exllamav3 1.5.0, the first release with reproducible greedy decoding (E52). This is the closest laptop engine to 1.6.5 for byte identity |
| on the 2080 Ti | `exl3-nodraft` vs `exl3-mtp`; `lcpp-iq2s` vs `lcpp-iq2s-mtp`; `lcpp-iq2xxs` vs `lcpp-iq2xxs-dflash2`; `exl3-nodraft` vs each undrafted GGUF lane; `exl3-mtp` vs each drafted GGUF lane | Are the drafters lossless on this card, and is 2.0 bpw EXL3 better or worse than the 2-bit GGUFs on the same items? |
| reference | `EFF-h38b_q4kxl_mtp` vs each `lcpp-*` lane, first 200 GSM8K-500 items | A laptop 4-bit GGUF (UD-Q4_K_XL + MTP, E48). This is a reference, not a replication |
| reproducibility | `TURING-X` vs `TURING-X-rN`, whenever a rerun label exists | Greedy run-to-run identity on sm_75 |
| variant | `TURING-<lane>` vs `TURING-<lane>-<variant>` (and E47 vs the EXL3 variants) | The effect of one setting, e.g. fp32 GDN state |

What the laptop has for the llama.cpp lanes (inspected 2026-10-30):

- No `gsm8k_500` or `mmlu_500` rows exist for UD-IQ2_S or UD-IQ2_XXS. The only laptop rows for them are the
  small campaign suites, run at temperature 1.0: `UD-IQ2_S-effmed` (gsm8k 50, mmlu 200),
  `EFF-iq2s_mtp_n3` (gsm8k 100, mmlu 200) and `UD-IQ2_XXS-mtpgpu-effmed` (gsm8k 50).
- Those rows share item ids with the 500-item sets, but they are sampled, not greedy, so the default plan
  does not pair them.
- `EFF-iq2s_mtps_36k` and `EFF-iq2xxs_dflash2_n7` hold only needle, MATH-25 and HumanEval+ rows.
- The llama.cpp lanes are therefore compared on the 2080 Ti itself and against the 4-bit reference.
- E48 (`EFF-h38b_q4kxl_mtp`) ran llama-server v0.4.0 with its default `--reasoning-format deepseek`, so its rows
  are split. The Turing llama.cpp lanes are inline (E47's shape), so these reference pairs always carry a "response
  shapes differ" CHECK, and their byte identity reads "not comparable". This is expected for a 4-bit vs 2-bit
  reference. Accuracy is unaffected here, because E48 had no truncations on its 200 items.

## Reading the identity numbers

- **First, the same shape.** A byte-identity figure only means something when both sides returned the text in
  the same shape. Without E47's shape, the desktop's TabbyAPI splits off the reasoning and drops `'</think>'`.
  Identity against E47 would then read about 0 for token-identical output, and that is an artifact, not
  numerics. Check the lane table's shape column and the CHECK lines before reading anything below.
- **E47 was not reproducible run to run.** It ran exllamav3 1.4.7 with the lab patches and a mixed CUDA
  library set. Its greedy decoding agreed with itself on only 50-64 of 100 GSM8K items (E49/E50).
- **So identity against E47 is not a quality test.** Byte identity between the 2080 Ti and E47 is bounded
  by that run-to-run figure. It measures numerics, not quality. The replication test is the paired accuracy
  (McNemar) together with the same-final-answer rate.
- **Expect some divergence even from a deterministic laptop engine.** From 1.5.0 on, the laptop reproduces
  itself 100/100 (E52). Still, 1.5.0 → 1.6.5 changed reduction orders, and sm_75 runs different kernels:
  m16n8k8 instead of m16n8k16, and fp16 GDN state. Turing therefore shows near-tie divergence even against
  1.5.0.
- **Measure the baseline first.** The reproducibility rerun (step 3) gives the denominator. Drafted against
  undrafted on the same card is the first clean identity test of the drafter since E47.

## Not covered (decided, revertible)

- **HumanEval and HumanEval+.** They are left out. The plan's E9 shootout uses HumanEval+, which would need a
  sandbox that leaves its directories in `$L/tmp`.
- **The laptop 4080 side of the llama.cpp lanes.** It needs laptop GPU time after Paper 5.
- **Context length.** It differs: the laptop used 32K, the lanes use 16K. This does not matter for these
  suites, where prompts are under 1K tokens and the generation cap is 4,096.
