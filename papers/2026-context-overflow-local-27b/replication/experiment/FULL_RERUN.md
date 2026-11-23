# Paper 5 inference replication — preparation, not a released package

Component navigation: ASSETS.json and the source manifest are here.
Repository-relative code paths below refer to the replication/ subdirectory,
unless they already start with replication/. Original-workspace audit links
in historical source documentation are context, not bundled private records.

This describes the remaining full-inference route. The separate scored-data
checker and PDF rebuilder reproduce analysis and presentation, not model
responses. No complete real P5 result or author-approved release exists yet.

## Assets and serving conditions

The private preparation inventory is
`ASSETS.json`.
It records 30 assets: the EXL3 shards/configuration/tokenizer, the 8B
follow-up GGUF and tokenizer, the gpt-oss judge, MiniLM retrieval encoder,
LLMLingua compressor, cleaned LongMemEval-S, original oracle evidence and
LoCoMo. Upstream files use immutable revisions; generated outputs are not
mislabelled as upstream downloads. No weights or benchmark content are copied.

The source preparation components also contain `ENVIRONMENT.json` and
`RUNTIME.md`: dated P5 native/metadata observations, current separate
compressor/analysis package inventories, both engine-patch identities and
the distinct llama.cpp build settings. They are not a hash-locked installer
or retrospective proof of every earlier preparation process.

The generator is the lab-patched ExLlamaV3 **1.4.7**, loaded through
`PYTHONPATH`, over the existing environment that also contains stock **1.4.3**.
Those are not interchangeable. TabbyAPI is pinned to
`4a4f9f44820303593844f092d424bb7506008733`. P5's configuration uses one slot,
32,768 context/cache tokens, Q8 KV, 1,024-token chunks, dynamic MTP with draft
length six, and a **4,096 MiB** host recurrent stash. The 16 GiB daily-driver
stash is not this experiment's setting. `EXL3_CKPT_PP=1024` is required;
the experimental measurement/sampled-draft environment switches stay off.

The generator patch has a separate architecture-file hunk. The combined
`exl3_all_lab_patches.patch` alone does not contain both files. A subsequent
full-wheel comparison also identified the inherited, withdrawn quantizer
workaround as a third modified file. The source component's `RUNTIME.md`,
two byte-reconstruction helpers and `runtime-evidence/` records now cover
the exact upstream wheel and all three changes, including the mapped native
extension. All 453 members in the dated snapshot match the reconstructed
overlay; 14 other small files match a separately dated current observation.
This is verified file reconstruction, not dependency installation, native
import, prior model-conversion evidence or a new inference optimization.
The metadata projection alone still covers only its two selected patch entries.
Native dependency setup and actual execution remain separate requirements.
Do not silently substitute a newer engine or the P7 experiment's other builds.

The judge and cross-model llama.cpp use commit
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`. The local binary needs the scoped
CUDA 12.6 runtime-library path. Tabby removes its child's inherited
`LD_LIBRARY_PATH` so torch uses its intended libraries. Retain actual loader
readbacks, not merely requested GPU-layer numbers.

## Continuing the present campaign

Do not repeat completed preparation. Preserve the active summary journal,
pilot/cap, corrected oracle prompts, follow-up migration archive, and the
pre-primary execution snapshot. The pilot selected an 8,192-token answer
cap. The primary has **6,157** requests: twelve 500-item cells and 157
oracle-fit requests. The oracle question/date suffix uses the cleaned primary
item while evidence text still comes from the original oracle dataset.

1. Let the original phase-1 service finish; phase 2 builds S, freezes all
   primary prompts and writes its 24-request smoke into the canonical grid.
2. Inspect and retain that exact smoke prefix. It uses a four-arm order,
   whereas the subsequent complete grid has thirteen cells. It is not
   disposable, and it predates the guarded wrapper's new execution receipts.
3. Check candidate-machine live headroom. The original study's measurements
   stay on its RTX 4080 Laptop GPU; stop for a deliberate protocol decision
   rather than substituting another GPU. Do not overlap GPU campaigns.
4. Preview, then execute generation and both graders with the guarded
   wrapper. The following commands are **dry-run defaults**, not launches:

   ```bash
   engines/exl3-env/bin/python -B paper5/scripts/primary_execution.py generate
   engines/exl3-env/bin/python -B paper5/scripts/primary_execution.py judge-gptoss
   engines/exl3-env/bin/python -B paper5/scripts/primary_execution.py judge-self
   ```

   Add `--execute --idle-seconds 10` only after prerequisites and machine
   checks pass. Ten-second pacing reduces average activity, not peak VRAM.
5. Run the primary-only audit before selecting the LoCoMo alternative.
   Finish the cross-model S preparation and selected LoCoMo alternative,
   then both follow-up measurement/grading/analysis legs. T/C/M cross-model
   and LoCoMo T preparation already exist and must not be overwritten.
6. Build the actual claims, sanitized data and reading set only when the
   complete primary and both follow-up audits pass. Perform findings-specific
   editing, rights/reference checks and final author review afterward.

The follow-up instructions are included in
`replication/testsuite/paper5_followups/README.md`.
The original workspace's dated execution audit is historical context,
not a redistributed private baseline or an authorization for a new run.

## A new experiment elsewhere is a different path

The original wrapper deliberately binds the original snapshot, hardware,
paths and pilot. It is **not a portable installer**. Do not copy the old
execution receipts into a fresh campaign or weaken the original guards to
make a foreign run pass. The source component now includes
`prepare_p5_rerun.py`, a create-only adapter for explicit model, input-data,
HF-cache, engine and CUDA-library paths. From the component directory:

```bash
python3 -B prepare_p5_rerun.py --output /absolute/new-p5-rerun --models-root /absolute/model-library --repo-assets-root /absolute/asset-repository --hf-cache-root /absolute/hf-cache --engines-root /absolute/engines --llama-library-dir /absolute/cuda-library-directory
```

All supplied directories must use physical, non-symlink paths; inspect and
resolve aliases such as CUDA's `lib64` before configuration. No asset, model,
native runtime, old cache or execution receipt is copied. Only new source/config
files are written, with declared expression-level changes and input hashes in
`RERUN_SETUP.json`. Its verifier checks setup inputs, not measurement outputs.
Nine copied Python files receive path/loader substitutions, and the copied YAML
model path plus two stale comments change. All scientific/serving settings and
original study files remain intact. The complete real oracle prompt rebuild
and date-only correction reproduce the original 157-fit-item files exactly,
with a new source-bound correction record.

The new workspace also contains `freeze_inputs.py`. Once its own complete
pilot/cap, all thirteen prompt cells and oracle correction exist, the default
command checks those inputs without writing anything. After checking live
CPU/RAM headroom, run from the new workspace:

```bash
python3 -B freeze_inputs.py
python3 -B freeze_inputs.py --freeze
python3 -B freeze_inputs.py --verify
```

The explicit freeze creates `testsuite/paper5/PROMPTS_MANIFEST.json` and then
the private `work/RERUN_INPUT_FREEZE.json`. It rechecks dataset/tokenizer pins,
all twenty excluded pilot requests and their cap rule, every prompt's source
question/date/hash/token count, the preserved oracle correction, and adapted
source hashes. No file is overwritten. An identical manifest left by an
interrupted freeze can be completed; changed manifests and a freeze attempted
after primary measurement journals exist are refused. The original 24-request
smoke exception is not inherited. Later `--verify` checks inputs only, not any
new generation results.

This is input-consistency evidence, **not** evidence that the pilot actually
used the claimed runtime, model, GPU or owned endpoint. The original-study
execution wrappers remain snapshot-bound and must refuse the new workspace.
A separately guarded prospective pilot/baseline/execution path remains to be
completed; do not bypass those guards or treat the legacy server-starting
scripts as preflight commands. The complete new freeze path is tested with
synthetic inputs, not claimed as a newly executed empirical campaign.

A fresh experiment must recreate preparation, run its own pilot/cap decision,
record its own model/runtime identity and responses, and preserve its own
follow-up selection evidence. In particular, preserve the oracle date
alignment semantics; do not regenerate the historical mismatch uncorrected.
The existing Tabby seed behavior and stochastic generation mean new answers
need not be byte-identical. No dataset, model or patched runtime should be
deleted on the assumption that this inventory is an offline backup.

Asset commands, licensing boundaries and precisely verified source facts:
[source-component guide](README.md).
