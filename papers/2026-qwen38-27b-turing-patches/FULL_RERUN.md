# Full experimental replication

These commands are documentation, not permission to start a GPU job. This
candidate has run CPU replay only. Preserve the frozen measurements; reruns
must write new output directories and retain failures and deviations.

## Resources and system

The primary system is one unmodified RTX 2080 Ti with 11,264 MiB, sm_75,
260 W and driver 580.178.04. Other programs shared CPU/RAM. Use at least
32 GiB host RAM and about 100 GiB free storage for pinned models, engines,
cache and new results; compilation and model acquisition are additional.
The original long-context stages plus kernel checks/replication used about
40 GPU-hours; the November 12 refresh has one combined 9,000-second ceiling.
Plan costs from the individual registrations rather than treating this as a
single short benchmark. No paid model service is needed.

`environment/software.json` and the exported run records pin the measured
runtime. exllamav3 1.6.5 used torch 2.10.0+cu128; the comparison servers used
the recorded CUDA 13.4 sm75 builds. The eight patches apply in order W7,
W9, W10, W10b, W11, W11b, W12p, W13. W12c was not adopted. W13's split-K
kernel is rafatxf's; mode 2 changes dispatch. Replaying the experiments on a
different GPU, driver, kernel, model file or prompt is a new landscape point.

## Recover the exact weights and prompt inputs

`environment/model_artifacts.json` is the download specification. Use its
full revision and SHA-256, not a moving branch or a filename alone. For
example, after installing a separately chosen Hugging Face CLI:

```bash
hf download turboderp/Qwen3.8-27B-exl3 \
  --revision d7e02d63dddd4e047a8d397acb22aeb2b0bedcec \
  --local-dir /opt/p8/models/qwen38-exl3-2.0
hf download unsloth/Qwen3.8-27B-GGUF \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  Qwen3.8-27B-UD-IQ2_XXS.gguf Qwen3.8-27B-UD-IQ2_S.gguf \
  MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir /opt/p8/models/gguf
```

Before loading, check each file's size and hash against the JSON inventory.
The old EXL3 shards are 8,573,967,630 and 1,622,158,601 bytes. The old
GGUF files are 7,266,070,528 and 8,371,970,048 bytes. Replacements uploaded
November 10–11 are unmeasured in this paper. The stripped shared-embedding
sidecar has a local derivative hash, not a Hub digest. Its reconstruction
and llama.cpp patch are exported under `replication/research/patches/`.
The NInferno container must be reconstructed with the documented conversion
and pinned drafter, then hash-checked; filenames alone cannot verify it.

To regenerate complete long-context prompts, obtain:

- EXL3 `tokenizer.json`, SHA-256
  `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`;
- `xiaowu0162/longmemeval-cleaned` at
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`, file
  `longmemeval_s_cleaned.json` (277,383,467 bytes), SHA-256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`;
- Python with tokenizers 0.23.1.

Place the tokenizer at `~/models/qwen38-exl3-2.0/tokenizer.json` and the
data at `replication/testsuite/paper5_data/longmemeval/longmemeval_s_cleaned.json`.
Then, from the artifact root, for example:

```bash
python3 -B replication/testsuite/p8_longctx/build_stage2_items.py graded \
  --length 65536 --levels multivalue=12 --out new-prompts
python3 -B replication/testsuite/p8_longctx/build_replication_items.py --out new-replication-prompts
```

Repeat the generator modes/lengths/seeds in the committed manifests. Compare
complete generated item bytes with the original file SHA-256 in each
manifest. The `replication/.../work/items/` files are compact scoring
projections and are deliberately not complete prompts. Do not compare their
file hashes to the complete-item hash or send them to a server.

## Install the measured engine and patches

Acquire exllamav3 at `cdc785db89e263e65e5849cb53e1ada156411414`
(v1.6.5) and TabbyAPI at `b1c225394d0fb45e8d26bddec344318dfd85096e`.
On the compatible CUDA/torch toolchain, apply the exported patches before
building an isolated engine tree. Preserve the stock wheel separately.
An illustrative application sequence from an exllamav3 checkout is:

```bash
for step in w7-int8-gemv-fit w9-int8-sq-fit w10-int8-sq-m4 \
  w10b-int8-sq-rowcap w11-fd75d-decode w11b-fd75d-live-splits \
  w12p-int8-stepped-pointer w13-gemv-sk; do
  git apply --check "$P8_ARTIFACT/replication/tools/turing/patches/exllamav3-1.6.5-$step.patch"
  git apply "$P8_ARTIFACT/replication/tools/turing/patches/exllamav3-1.6.5-$step.patch"
done
```

Use the upstream build procedure for that exact revision and toolchain;
record the source, binary, wheel and patch digests before running. The
historical wheel and resulting extension are linked/hashed in original
records, not redistributed. The refreshed NInferno/llama.cpp build recipes
are in `replication/tools/turing/refresh/build_cpu.py` and the registration;
read them before installing. No build/download is part of the quick check.

## Layout, serving and measurements

The exported paths are normalized to `/opt/p8`; choose a disposable layout
with the same structure. Set `TURING_LAB` explicitly. Place `replication/tools/turing/`
contents at `$TURING_LAB/tools/`, and the plan/config/patch dependencies at
their declared relative paths. `venv/`, `models/`, `src/`, `overlay/`,
`logs/` and `runs/` are separate. Review guards for the replication machine;
do not install the original machine's services or copy its account settings.

The exact F2 arm list and flags are in the W13/F2 registration section of
`replication/research/TURING_W7_INT8_GEMV_2026-10-30.md`. It uses
`tools/ab_speed.sh TAG 3 'NAME|ENV|ARGS' ...`, fresh processes, rotated
rounds, three fixed prompts, at most 256 tokens, temperature 0.8 and min-p
0.08, thinking off. The explicit `-cs 32768` in the registered extra
arguments overrides the shell wrapper's older 16384 default. Do not omit it.
Keep the stock arm, both contexts and patch modes exactly as registered;
compare within the new session rather than treating historical speeds as
pass/fail tolerances.

For the confirmation protocol, from the prepared lab directory:

```bash
python3 tools/confirm/run_confirm.py --plan tools/confirm/plan.json --out runs/new-confirmation
python3 tools/confirm/analyze_confirm.py --help
```

Read the dated amendments and their separate fit/DFlash2 plans first. The
November 12 refresh requires its exact new engine build receipt, artifact
inventory and a newly reviewed seal. Historical seals record the original
code bytes and will reject normalized export sources. Do not edit those
seals to pretend a rerun is the original campaign.

Long-context plans and launchers are in `replication/testsuite/p8_longctx/`.
Use a new workspace and the regenerated full prompts, then run the
registered stage 1 ladder, calibrations, paired stages 2–3, logit check,
accuracy arm and value-bits replication in their prescribed order. The
launchers' `--help`/source and dated registrations give the exact modes,
stop rules and GPU budgets. One TabbyAPI process per arm owns its caches;
retain every journal, server config, server log and telemetry file. Pooled
requests, greedy retrieval and sampler settings must match the registration.

The quality lane commands/configs are in `replication/tools/turing/quality/`
and `lanes.sh`; obtain GSM8K/MMLU separately from the original dataset pins.
Use the original seeded 500 IDs, thinking on, greedy, 4096 output-token cap,
one request at a time and the shared scorer. Exported projections support
aggregate/pair audits, not re-scoring absent response text. Full inference
re-scoring needs newly retained responses and the independently obtained
benchmark gold. Generated code must never execute outside a sandbox.

## Outputs, tolerances and known boundaries

The CPU checker expects exact exported bytes, exact regenerated macros and
tables, and—under the pinned Matplotlib version—exact figure PDFs. GPU
speeds and greedy outputs need not be bit-reproducible: compare new arms
within a session and apply the original registered bounds, not a new
post-hoc tolerance. Missing/cut/failed cells remain missing; partial offload
must not become full-offload evidence. A 240 W repeat in the old campaign
actually ran at 260 W and is identified as an unplanned repeat.

The export includes normalized source, not native binaries or a tested
installer. A full hardware replication has not been performed from this
candidate. Human review of these instructions remains pending. In
particular, archived refreshed-engine startup logs/build binaries are not
bundled; their recorded hashes and summarized outcomes are retained. Any
unavailable upstream file or incompatible historical binary is a replication
blocker, not authority to substitute current weights or silently widen a claim.

## Bonsai derivative counterpoint

The frozen registration is `replication/research/prereg/2026-11-12-p8-bonsai-lane.md`.
The original adapter, qualification, lane and analysis sources are included under
`replication/tools/turing/bonsai/`, with the original contract, budget, baseline
inventory and model/rights receipts under `replication/research/raw/2026-11-12-p8-bonsai-lane/`.
The public-source normalization inventory records any path/host changes; historical
hash seals describe the original experiment, not authorization to launch a normalized copy.

Use NInfer `f4ab369a06c728325b706c989d71be7dc0c05a1b` and
`WaveCut/Ternary-Bonsai-2-27B-NInfer-v3` revision
`841f7812793972b5de6dc48910bdd1e2722ae48b`, artifact
`Ternary-Bonsai-2-27B-ninfer-v3.ninfer`, SHA-256
`cdc4810b0ff17c40d0f62cf214b6e0bcd08346e9eb05ca53371507037793c14a`.
The model is linked only; its pinned metadata/NOTICE records Apache-2.0.
NInfer is separate from NInferno. The registered card is a stock RTX 2080 Ti
at an existing 260-W limit, one request at a time, 16,384 context capacity,
int8 KV and fp16 recurrent state. The registration fixes undrafted and MTP-3
modes, the 500-item GSM8K and MMLU slices, a 4,096 generated-token cap and
medium reasoning effort. Read its full sampling/normalization contract and
qualification/stop conditions before a prospective hardware replication.
The queue snippets describe the original owner's machine and must not be
executed as generic install scripts. New runs require their own explicit
machine placement, qualification, budget and authorization.

For the model-free check, the default root `check_claims.py` already replays
all nine numeric lanes and all five paired comparisons, with the registered
Wilson convention, exact McNemar tests, family-five Holm and fixed-seed
5,000-replicate bootstrap intervals. Compact records omit benchmark text and
responses. Native split reasoning was logically normalized for the unchanged
scorer; neither that representation nor the numeric replay proves byte/token
identity or equivalent quality. Request-only correct/minute excludes startup
and qualification and must not be mixed with whole-campaign denominators.

## Two fleet deployment slices

The fleet extension retains only the paper's numeric P8 slice: two rotated
36-request speed fixtures (four conditions, three tasks, three rounds/card),
eight 500-item quality verdict projections, the P8 summaries and assessed
prediction verdicts. Logical card labels appear under `replication/fixtures/fleet/`;
the P8-only analysis metadata is under `replication/research/raw/2026-11-12-fleet-lanes/`.
Original/export hashes and projection field sets are in the inventory.
No machine queue wrappers or the rest of the fleet workspace are shipped.

The targets are RTX 4070 and RTX 4080 SUPER (sm_89), EXL3 1.6.5 plus the w13
chain, and the same pinned 2-bit weights. Registered configured limits are
200 W and 320 W; pulled evidence lacks per-run actual power-limit/device/driver
snapshots. TabbyAPI is b1c225394d0fb45e8d26bddec344318dfd85096e. Its request path
does not forward the nominal seed labels. The HTTP speed timer uses server
`decode_wall_s`, not process lifetime or whole-campaign time. These slices cannot
be pooled with in-process F2 as controlled card scaling or repeated RNG states.

A prospective hardware replication must reconstruct the declared conditions,
rotate the same four condition cells and three speed tasks, serialize requests,
use the archived paired quality IDs/scorer and obtain its own device/runtime/power
snapshots, qualification, thermal/stop guards, placement and run authorization.
These portable numeric fixtures are not a tested fleet launcher. A new model
run must retain fresh raw responses for independent correctness re-scoring;
saved flags alone only reproduce aggregate and paired arithmetic.

## November 14 replacement evidence

The new numeric dependency is
`replication/research/raw/2026-11-14-p8-rc-refresh-a72c/refresh14-numeric.json`.
It projects the independently reviewed pooled, round and task tables, preserving
their original SHA-256 bindings. The portable claim builder replays 69 added
macros and the replacement table from this reviewed projection. It does not
reopen the 201-row private journal, recheck the 49 raw request receipts, or
repeat the target/drafter placement audit. Those checks belong to the reviewed
scientific analysis; passing this artifact check cannot replace them.

The registered source package seal is
`6b2853cb07b030dc5229d1b1a219db2086d2f30fac774d15778ee147a0b9fe4c`,
scientific registration SHA-256
`e6092cfdd3562dd98c914de6ccde63c479af0000b679b7082b1cb9811009ef20`.
The replacement uses NInferno f8143b9e, llama.cpp 61dbf492, stock EXL3 1.6.7
and unchanged patched EXL3 anchors, with the retained model files. Presence 0,
frequency 0 and repetition 1 were explicit on every NInferno request; top-k 0
resolved to 64 and min-p ordering differed from EXL3. The earlier November 12
NInferno comparison is replaced, not pooled. Its default presence 1.5 was
reconstructed from startup logs and pinned source rather than an old
per-request receipt. The concurrent version change prevents causal attribution.

A new hardware replication needs the exact registered source package and
artifacts, fresh registration/authorization, runtime qualification, raw request
receipts, placement checks, phase budgets, fit ladder and both required warmups.
No new GPU run or runtime launcher is executed or certified by this candidate.
The compact artifact intentionally establishes numeric replay, not a tested
end-to-end launcher for the new run. Coverage is 9/14 complete and four eligible
NInferno leads; missing, incomplete and placement-unknown cells are not losses.

## Remaining candidate gates

The old per-prompt attribution defect is resolved in the selected scientific
source; the portable check independently selects its named arm. This package
also exports the fleet table and current factual AI roles. It does not grant
new scientific, editorial, rights, disclosure or full hardware-replication
approval. The accepted scientific refresh must be frozen and this exporter
rerun; independent review must bind that final export rather than the older RC.

A successful exact-stage controller `verify-analysis` receipt is still required
in an approved compatible namespace context. The previous candidate failed
before Python at nested bubblewrap namespace creation. Direct portable CPU replay
is separate evidence, not a substitute or policy bypass. No controller retry,
root helper modification or successful controller receipt is claimed here.
All manifest gates remain pending; author approval is false, approvals empty,
artifact commit/tag and all public identifiers blank. Current route/platform
rules, human declarations/licences, Matthew's final read, exact-hash approval,
and separately authorized provider/submission operations remain outstanding.
