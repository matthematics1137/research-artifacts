# Isolated Paper 5 follow-ups

These tools implement the September 19 protocol's cross-model and LoCoMo
legs without modifying `testsuite/paper5/` or adding files to its prompt
directory. They are experiment tools, **not release approval**. The primary
campaign must complete before follow-up generation or policy selection.

**September 20, 17:37 PDT:** cross-model T/C/M and LoCoMo T prompts are
complete; no follow-up answers exist. A primary-claims JSON-roundtrip loader
bug was corrected after M finished. The original workspaces/source/receipts
were archived and exactly the same prompts adopted into fresh canonical
workspaces. Keep the private archive and do not edit the new bound sources.
See the [migration record](../../research/P5_FOLLOWUP_MIGRATION_2026-09-20.md).
The preparation examples below are a runbook, not requests to redo completed
arms. S and LoCoMo selection still require completed primary results.

## Boundaries

- `cross-model`: the same 500 LongMemEval-S items, T/S/C/M at 4,096 tokens,
  Qwen3-8B UD-Q4_K_XL, its own tokenizer/template, existing 27B session
  summaries, gpt-oss and generator-as-judge scores. Descriptive comparisons,
  outside the primary Holm family.
- `locomo`: the frozen 200 questions, category quotas 70/70/60, T versus the
  primary gpt-oss winner among S/M/C, ties S then M then C. No selection from
  an incomplete primary campaign. Primary judge plus token F1; public output
  remains aggregates only. Item data and raw journals are private.
- The original prompt algorithms and server/pause machinery are imported
  into a separate process with isolated output paths. They are not rewritten
  in place. Job IDs explicitly include the follow-up and generator.
- Every workspace binds its item data, pilot cap, dataset, tokenizers,
  protocol and code by hash. Create-only writes refuse replacement. Prompt
  freezes reject missing/extra files, duplicates, wrong models and over-budget
  records. Symlinks cannot redirect writable paths into the primary campaign.
- Run commands are dry-run by default. `--execute` requires complete primary
  evidence, original-hardware headroom, silent study ports, inactive primary
  units, and exact model/engine bindings. No service starts during import,
  planning, initialization, verification or tests.
- Ten-second request pacing is the follow-up default. It reduces average
  activity, not peak utilization or VRAM. Cached jobs must match the same
  request and execution identity; they are not resent. Only owned server
  process groups are stopped.

## CPU preparation

Commands run from `/path/to/study`. Before a long/model-backed
preparation job, inspect the candidate-machine inventory and live headroom.
Use a suitable machine; do not blindly run it on the laptop. These examples
identify the local pinned environments, not an instruction to bypass that
check. The real GPU measurements stay on the study's original hardware.

```bash
python3 -B testsuite/paper5_followups/followups.py plan cross-model
python3 -B testsuite/paper5_followups/followups.py plan locomo
python3 -B testsuite/paper5_followups/followups.py init cross-model
python3 -B testsuite/paper5_followups/followups.py init locomo
```

Initialization binds the current code. Do not edit it underneath an initialized
workspace and silently continue. On a genuine correction, preserve the old
workspace and document a fresh preparation before any affected measurement.
No existing data are automatically deleted or overwritten to resolve drift.

T/C/M may be prepared before the primary results. Cross-model S requires the
completed primary evidence so the shared summary journal is no longer growing.
Prepare one full arm at a time; a partial file is not accepted as a full arm.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false nice -n 10 \
  engines/exl3-env/bin/python -B testsuite/paper5_followups/prepare.py cross-model --arm T --threads 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false nice -n 10 \
  engines/lingua-env/bin/python -B testsuite/paper5_followups/prepare.py cross-model --arm C --threads 1
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false nice -n 10 \
  engines/lingua-env/bin/python -B testsuite/paper5_followups/prepare.py cross-model --arm M --threads 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false nice -n 10 \
  engines/exl3-env/bin/python -B testsuite/paper5_followups/prepare.py cross-model --arm S --threads 1
```

For LoCoMo, T can be prepared now. After complete primary analysis:

```bash
python3 -B testsuite/paper5_followups/followups.py select locomo
```

Then prepare only the selected alternative. S first needs the isolated
`run.py locomo summaries --execute` stage. C first needs
`engines/lingua-env/bin/python -B testsuite/paper5_followups/preprocess.py --threads 2`
(CPU classifier; repeated conversation histories are deduplicated). M uses
the same prompt command as above with `locomo --arm M`. T/S use `exl3-env`;
C/M use `lingua-env`. All model loads use local files/offline mode.

After all declared arms are present:

```bash
python3 -B testsuite/paper5_followups/followups.py freeze cross-model
python3 -B testsuite/paper5_followups/followups.py verify cross-model
# Corresponding commands for locomo after its selection and two arms exist.
```

## GPU stages — only after live availability checks

These are **dry-run examples**, not launches. Add `--execute` only when the
upstream prerequisites and machine checks pass. `--limit 24` supports an
operator smoke check; it does not create a completed-stage receipt for an
incomplete grid. Inspect that smoke before continuing the full stage.

```bash
LD_LIBRARY_PATH=/path/to/operator/cuda-12.6/lib64 \
  engines/exl3-env/bin/python -B testsuite/paper5_followups/run.py cross-model generate
LD_LIBRARY_PATH=/path/to/operator/cuda-12.6/lib64 \
  engines/exl3-env/bin/python -B testsuite/paper5_followups/run.py cross-model judge-gptoss
LD_LIBRARY_PATH=/path/to/operator/cuda-12.6/lib64 \
  engines/exl3-env/bin/python -B testsuite/paper5_followups/run.py cross-model judge-self
LD_LIBRARY_PATH=/path/to/operator/cuda-12.6/lib64 \
  engines/exl3-env/bin/python -B testsuite/paper5_followups/run.py locomo generate
LD_LIBRARY_PATH=/path/to/operator/cuda-12.6/lib64 \
  engines/exl3-env/bin/python -B testsuite/paper5_followups/run.py locomo judge-gptoss
```

The local pinned llama-server needs that CUDA runtime path: a September 20
read-only `--version` check failed with missing `libcudart.so.12` without it
and passed with it. The prefix applies only to each command and its children;
Tabby's existing launch code removes its child `LD_LIBRARY_PATH`. No bound
follow-up source was edited to make this operating-environment correction.

Qwen3-8B seeds are sent to llama.cpp as prescribed by the original request
seed rule. TabbyAPI's 27B seed remains recorded-but-not-applied. Cross-model
completion counts use its actual tokenizer; gpt-oss's server-reported usage
is retained without inventing a generator-tokenizer completion count.

## Analysis and tests

```bash
python3 -B testsuite/paper5_followups/analyze.py cross-model
python3 -B testsuite/paper5_followups/analyze.py locomo
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 \
  python3 -B -m unittest discover -s testsuite/paper5_followups -p 'test_*.py' -v
```

Analysis requires every declared generation attempt and judge result, exact
execution/freeze bindings and finished-stage journal hashes. Terminal
failures are counted as incorrect; unrun jobs are rejected. Empty successful
gpt-oss replies require the declared retry. A changed generation invalidates
its judge label. No incomplete analysis overwrites a previously good result.

Token F1 uses the primary normalizer. LoCoMo percentile intervals resample
whole conversations with replacement (10,000 draws, seed 20260920); the
item-weighted ratio is recomputed in each draw. These implementation settings
are fixed before LoCoMo measurements. Its item-level Wilson/Newcombe/McNemar
summaries are descriptive and explicitly warn about shared-conversation
dependence. They are not an additional confirmatory family.

As of September 20, 33 tests pass, including a disposable 6,000-response
synthetic generation/two-judge/resume/analysis pipeline. Fake responses never
enter the real workspaces. The actual GPU follow-up stages have **not** yet
been run; a synthetic test is not evidence that a real deployment passed its
token-parity smoke, model startup or scoring validation.

The manuscript's read-only follow-up integration now exists separately in
`paper5/scripts/followup_results.py` and is exercised by synthetic journals.
Still required: remaining real prompt preparation, real smoke/runtime checks,
complete follow-up measurements, independent release verification, final
findings-specific review and the curated artifact. These tools do not make
P5 final-read-ready by themselves.
