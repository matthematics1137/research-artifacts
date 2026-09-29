# Paper 2 corrective rerun plan (P2R1)

Status: **prepared; inference not started**. The original P2G grid is
quarantined. It must never be used to rebuild claims or to promote a
manuscript.

P2R1 is a corrective exploratory rerun designed after the hypotheses, corpus
frames, and contaminated P2G outputs were observed. Fresh Q4-generated
questions repair the measurement chain; they do not make this an independent
confirmation. Any promoted paper must say that plainly.

The prospective machine-readable contract is `R1_PROTOCOL.json`. Commit that
contract, this plan, and the recovery harness before `run`. The repository's
Git DAG records chronology; it is not an independently registered
preregistration.

## Pre-request operational amendment on 2026-08-31

Campaign `725e2db91e9840f6a766a5b616decc78` loaded the intended Q4_K_XL
artifact and reached a healthy loopback server, but its identity gate could
not find the required placement line. The gate remained before question
generation: there was no generator or evaluation POST, no durable request
intent, no generated question, no response, no score, and no timed result.
The private campaign directory and its verbosity-3 diagnostic log are retained
as failed pre-request provenance.

The cause was logging configuration, not a model-load failure. In the pinned
llama.cpp revision, the library-level INFO line that reports layer placement is
exposed at server verbosity 4, while the default is 3. Before any empirical
request, the deployed command for both llama.cpp tiers was therefore amended
to include exact `-lv 4`. The identity guard requires that exact flag and value
in its exact-position argv contract. The placement gate still requires exactly
one authoritative loader line; it has not been weakened to infer placement
from requested arguments.

The observed artifact-specific expectations are `33/66` for Q4_K_XL and
`65/65` for IQ2_S. Q4_K_XL has 65 model blocks, including its MTP-related
block, plus one separately counted output/offload unit. IQ2_S has 64 blocks
plus its output unit. Missing, duplicate, or different readbacks stop the stage
before any request.

Verbosity 4 is now part of the frozen deployed llama.cpp configuration. Its
trace output stays in owner-only campaign logs. The trace sites are bounded,
but their runtime overhead is included in reported llama.cpp HTTP wall times
and is not assumed to be zero. The official attempt ID remains
`p2r1-primary-v1`: because the failed campaign stopped before the first request
boundary, this is an eligible pre-request restart rather than outcome-based
resampling. Frozen inputs, artifacts, quotas, prompts, decode/scoring rules,
analysis, and stage order are unchanged.

## Pre-request runtime-map amendment on 2026-08-31

The next launch, private campaign `4c01d78c03714b9e9f932846b738e76c`,
passed the intended Q4_K_XL identity, and its private log contained the exact
`33/66` placement readback. Its live CUDA mapping gate then stopped before it
could write a placement receipt, before every durable request intent, and
before every question-generation or evaluation POST. No question, response,
score, or timed result exists, so the same frozen `p2r1-primary-v1` attempt
remains eligible. The campaign and its scoped shutdown evidence are retained.

The guard failed because it rejected any `/proc/<pid>/maps` pathname ending in
` (deleted)` before determining whether that pathname belonged to the three
CUDA-family libraries in scope. A separate five-minute, resource-capped,
GET-only diagnostic reproduced the error from seven normal
`/dev/zero (deleted)` mappings among 327 map lines. All 12 mapped segments for
the pinned libcudart, libcublas, and libcublasLt files were canonical and
non-deleted; zero classified CUDA-family segments were deleted. No benchmark
POST was sent. The private diagnostic script, log, process receipt, and map
receipt have SHA-256 hashes
`e57ad5636150ce3e40eeda116fc0f25cbe35b670dd5e2d03c51e4fac158dc1a8`,
`27eb9bf94b6969062df4866c89d5ac67eebc686058a249319316c3b395456221`,
`56cea97ea1eefde5e3871a5e78300969e325590255220764f7b22101378f40cc`,
and `44b952c69f65098c5fd5689a2824fb5800f1c35f9e577a0fbc26163400211370`.
The private incident receipt is
`.publication/private/p2r1/pre-request-runtime-map-failure-002.json`, SHA-256
`2fb6b9f59e9027b19912d92d906d22a5b7825bee8c70d1247a9aff8d4def232a`.

Prospectively, the parser now removes only the terminal deletion marker for
classification, then ignores deleted mappings outside the declared
libcudart/libcublas/libcublasLt family. A deleted absolute or nonabsolute
CUDA-family mapping still fails closed. Exact canonical mapping-set equality,
private-directory containment, byte counts, and SHA-256 checks are unchanged.
Tests cover benign deleted `/dev/zero` and NVIDIA memfd mappings, deleted
absolute and nonabsolute CUDA-family mappings, and missing or extra family
files. This guard-only correction changes no deployed artifact, input,
placement expectation, quota, prompt, decode/scoring rule, timed HTTP request,
analysis, or stage order.

## Pre-model graphical-desktop admission amendment on 2026-08-31

Campaign `09ce310e6db745e98206790f096c02e0` stopped after spending 1,800
seconds in the first Q4 question-generation cool gate. The ordered launcher had
not started a model. Eight model-stage receipt directories are empty; `host/`
contains only `inventory.json`, and the campaign root contains only
`attempt.json`, `toolchain.json`, and that host inventory. The official
question tree is empty, and the result root still contains only the
pre-existing host binding. There was no endpoint session, durable request
intent, benchmark GET or POST, question, response, result row, score,
promotion, or invalidation. The same `p2r1-primary-v1` attempt therefore
remains eligible; the stopped campaign is retained and will not be reused.

The terminal sample met every other retained admission limit: 43 C GPU,
993 MiB allocated VRAM, 1.83 one-minute load, 40.7 GiB `MemAvailable`, and no
NVIDIA compute process. It failed only because device-total GPU utilization was
24%, above the original 5% ceiling. That aggregate counter can include ordinary
Xorg/GNOME/Chrome graphics on this graphical host and is not a reliable proxy
for a foreign CUDA workload.

Prospectively, device-total GPU utilization is still recorded, required to be
finite, and range-checked to 0--100%, but it is not an admission threshold.
The gate still requires three consecutive ten-second samples with no NVIDIA
compute process, GPU temperature at most 50 C, baseline VRAM at most 1,024 MiB,
one-minute load at most 2, and at least 32 GiB `MemAvailable`. The campaign
cgroup, swap cap, thermal watchdog, model identity, and placement gates remain
unchanged. The electronic gate cannot distinguish ordinary composition from an
intentionally heavy game, video, or WebGL workload that uses only graphics
contexts. Avoiding those workloads is therefore an explicit operator conduct
control in the private launch authorization, not a machine-enforced claim.
Request timing remains a descriptive measurement of the deployed laptop state,
not isolated-GPU throughput and not part of the primary accuracy-interaction
branch.

The owner-only incident receipt is
`.publication/private/p2r1/pre-request-graphics-gate-failure-003.json`
(SHA-256 `5a51009cab487ce76d4eb9c54d859dd6f49c5bcf2b21e3ce4895e92e6a34a493`).
Its archived raw systemd journal exports have SHA-256
`06f64638926fa5c124c6d59572f9fec964402db64ed558e62f4d74ce2048e3a8`
and `90df20d5fa2d0001706c8b2be6d3b302f3960c9884f9aaa3d2fbbc8f344fd600`.

## Pre-session generator-defect amendment on 2026-09-01

Campaign `e2a83a22bd024cca8776cc9d1e4b18a5` passed the cool gate, launched the
intended Q4_K_XL server, passed the full endpoint-identity gate (exact
`33/66` placement readback), and then crashed on the first generator
invocation with `NameError: name 'sha256_file' is not defined` at
`testsuite/paper2/gen_questions.py:882` — the module used the helper without
importing it, a defect `py_compile` cannot detect. The crash occurred
immediately after the TSI generation contract was written and before any
generation session, journal row, attempt-ledger row, question, durable request
intent, benchmark POST, response, score, or timed result existed. The EXIT trap
terminated the launched server through its scoped process receipt
(`shutdown/q4-question-generation.json`, status `closed`), wrote the
`q4-question-generation-abort.json` host snapshot, and both ports were
verified free afterwards with zero NVIDIA compute processes.

The campaign's single durable pre-session file, the TSI question contract
(SHA-256
`8f29c68b8135d53cd1088eee41f0595fb9e1a4219d04195373c87691af72a3bd`), embeds
the defective harness's `recovery_harness_sha256` and would fail the immutable
contract check against any corrected harness. It was moved byte-identically
(hash-verified before and after) into the failed campaign's private evidence
tree at `quarantine/pre-session-question-contract-tsi.json`, beside the launch
console log (SHA-256
`ec541289c93d043b475d75f645d38bb235b3bd2d7b36112bf66045572942a1d3`) and the
quarantine receipt (SHA-256
`1c15b77ddc10b87c7736ff54071960ef9912ba9919d77af843083dfb4880b8ae`). The
eligible attempt's question namespace is empty again; no other attempt state
existed.

The prospective fix adds `sha256_file` to the module's existing
`identity_guard` import — one line, changing no request, input, quota, prompt,
decode/scoring rule, gate threshold, or stage order. A static undefined-name
audit (pyflakes 3.4.0 under Python 3.12.3) over every
`testsuite/paper2/*.py` and `paper2/scripts/*.py` module found exactly this
one undefined name; the only other findings were unused imports and unused
local variables, which cannot raise at runtime.

Because the campaign stopped before the first generation session was fsynced
and before every durable request intent, the official attempt
`p2r1-primary-v1` remains eligible for a pre-request restart under the same
frozen inputs, artifacts, and protocol.

## Completion-cap amendment and official attempt p2r1-primary-v2 on 2026-09-01

Campaign `90007baf553f4eea83a684138699943c` ran the corrected generator
through all 126 TSI frame windows (three predeclared seeded attempts each,
378 generator requests, every identity gate passing). Deterministic selection
then stopped:

    TSI/L-nocode has 5 eligible windows; frozen quota is 6 after 3 attempts

and the harness permanently invalidated attempt `p2r1-primary-v1`
(`ATTEMPT_INVALIDATED.json`, reason label
`question-generation-final-evidence-ambiguity`, with the exact error quoted
in its details). No WildChat generation request, evaluation request, warmup,
or timed result was ever created. The launched Q4 server was terminated
through its scoped shutdown receipt and both ports were verified free.

The private journals show the mechanism precisely. Seven windows accepted
zero questions; in six of them every attempt returned HTTP&nbsp;200 with
exactly 512 completion tokens --- the frozen cap --- and zero parseable
question/answer pairs, and in the seventh the only parsed pairs failed the
verbatim verifier before two further capped attempts. Accepted windows ran at
a 445-token median with 32 percent of attempts at the cap. The deployed
model's reasoning simply exceeds 512 tokens before it emits the required
format on verbose windows, and under fixed seeds and temperature 0.2 that
truncation is effectively deterministic. The L-nocode stratum has exactly six
frame windows for a quota of six, so one such window makes the quota
infeasible and a retry under the same configuration must fail the same way.

The prospective amendment raises the completion cap to 2048 tokens for both
question generation and evaluation, freezes it in every generation and run
contract, and registers the new official attempt `p2r1-primary-v2` in
`R1_PROTOCOL.json`. The evaluation cap is amended in the same step because no
evaluation request has ever been sent and the identical truncation mechanism
would otherwise convert verbose-reasoning items into parse failures scored
incorrect, contaminating the primary accuracy grid. Request timing remains a
descriptive end-to-end measurement; the longer permitted completions are
disclosed. Frozen inputs, artifacts, quotas, prompts, verbatim and scoring
rules, retry seed offsets, condition order, temperature, top-p, seed, the
600-second timeout, placement expectations, stage order, and the statistical
analysis are unchanged. The `p2r1-primary-v1` namespace is preserved
forensic-only under its invalidation marker.

## Warmup-cap amendment and official attempt p2r1-primary-v3 on 2026-09-01

Under attempt `p2r1-primary-v2`, campaign
`409fc7a5727d45cb8c853fabe15f2395` completed question generation exactly on
quota --- 117 TSI and 120 WildChat accepted items, zero zero-accept windows,
normal generation post-state, scoped shutdown --- then passed the
q4-evaluation cool, identity, and placement gates and stopped inside the
deterministic unmeasured warmup:

    P2R1 WARMUP ABORTED: deterministic warmup response omitted its extractive token

No timed evaluation request, request intent, response row, or score was ever
created. The abort snapshot and scoped shutdown receipts were written and the
ports verified free.

The cause is the same completion-cap mechanism as the previous amendment, in
the one request path that had not yet executed: `warmup_r1.py` froze
`max_tokens=64`, and the deployed model's reasoning consumes that budget
before the extractive token can appear. The amendment raises the warmup cap
to 2048 (harness and validator expectation in lockstep). Because
`warmup_r1.py` is inside the frozen measurement bundle embedded by every
question contract, the amended harness cannot resume `p2r1-primary-v2`: its
contracts fail the immutable-contract equality check by design, so the v2
namespace --- including its complete on-quota question set --- is preserved
forensic-only under an explicit operator-written invalidation marker, and the
new official attempt `p2r1-primary-v3` regenerates questions under identical
frozen inputs, prompts, seeds, and rules. The warmup remains a single
recorded, unmeasured request excluded from analysis. Everything else is
unchanged.

## Operator-pause amendment and official attempt p2r1-primary-v4 on 2026-09-11

Campaign `e62d9382f15948a4862ae9d4ea8fa815` under `p2r1-primary-v3` had completed
Q4KXL question generation for both corpora, the whole `P2R1-Q4KXL-TSI` grid
(468 of 468 rows) and 404 of 480 `P2R1-Q4KXL-WC` requests when, at
2026-09-11 07:40:01, the operator's pause procedure stopped the systemd user
unit that had launched the campaign. That unit's cgroup contained the launcher,
`run_grid.py` and the llama.cpp server, so all three received SIGTERM at once
while evaluation request `034399bb03ef4baf881ae8285ef41c08` (`wc0134q0`,
condition A) was in flight. The launcher's trap recorded
`host/q4-evaluation-abort.json` and the scoped server shutdown; `run_grid.py`
died before writing that request's terminal. The request ledger therefore
holds 405 durable intents and 404 terminals. By the retained rules ("exactly
one terminal must follow" every intent; an intent without a terminal is
ambiguous response completion), `run_grid.py` refuses to resume that grid and
the attempt is forensic-only. Every retained row, question and receipt is kept;
nothing from v3 is resumed or re-labelled.

The cause was operator tooling, not the harness or the model. The pause script
is corrected (`testsuite/paper2/p2r1_pause.sh`): it freezes only the recorded
evaluator PID at an even ledger count with consistent heads, lets the launcher
take its own abort path, and never stops a unit; the auto-resume
(`testsuite/paper2/p2r1_autoresume.sh`) now launches the campaign as its own
transient unit so stopping the auto-resume cannot touch it.

Prospectively, the official attempt is `p2r1-primary-v4`. Questions are
regenerated under identical seeds and all six grids are redone. Because a new
attempt is the only point at which the immutable harness may change, the same
amendment relaxes the cool gate's operational thresholds for this working
desktop: admission still requires three consecutive ten-second samples with no
NVIDIA compute process, GPU temperature at most 50 C and at least 32 GiB
`MemAvailable`, and now admits at desktop VRAM at most 1,536 MiB (was 1,024)
and one-minute load at most 8 (was 2); device-total utilization, load and
desktop VRAM stay recorded in every gate receipt. The headroom analysis of the
withdrawn 2026-09-10 note holds: the completed Q4KXL evaluation peaked at
10,312 MiB device-total on a 713 MiB desktop, so the worst admitted desktop
leaves about 1.1 GiB of the 12,282 MiB device free on the tightest tier, and the
exact placement readbacks (33/66, 65/65) still fail closed. Request timing
remains a descriptive measurement of the deployed laptop, now under a broader
admitted host state, and is not part of the primary accuracy-interaction branch.

The owner-only incident receipt is
`.publication/private/p2r1/pre-request-operator-pause-failure-005.json`
(SHA-256 `07b4b1529c6087abf72106d29b7cae3c6b68f344e8347d3a540c434f4ecc0629`).

## EXL3 provenance-gate amendment and official attempt p2r1-primary-v5 on 2026-09-13

Attempt `p2r1-primary-v4` completed Q4KXL question generation for both corpora
(117 TSI + 120 WildChat items), the whole `P2R1-Q4KXL-TSI` grid (468 of 468
rows) and the whole `P2R1-Q4KXL-WC` grid (480 of 480 rows plus one retained
failure row) under campaign `a3c471b006994fe3a5c4c5beb2b04e4d`. Its EXL3 stage
first stopped inside tabbyAPI `start.py`'s first-run dependency installer
(operator runtime state, repaired outside the harness with the ignored file
`engines/tabbyAPI/start_options.json`). On relaunch (campaign
`f0682da7bb5f44a793bef23f0a937dd1`, 2026-09-13 21:15:43 local) the EXL3
identity gate passed and the placement gate aborted with "live EXL3 mapping is
unreadable: /dev/zero (deleted)". The launcher's trap recorded
`host/exl3-evaluation-abort.json` and the scoped shutdown; no EXL3 warmup,
request intent, response row, or score was created.

The cause was three defects in the EXL3 branch of the placement gate, a code
path no earlier attempt had reached (v1 stopped in question generation, v2 in
the Q4 warmup, v3 in the operator pause, v4's first launch in the installer):

1. The EXL3 `/proc/<pid>/maps` parser resolved every absolute path as a file.
   The NVIDIA driver maps anonymous shared memory that the kernel reports as
   `/dev/zero (deleted)` in every CUDA process (five segments after
   `torch.cuda.init` in the pinned environment, six in the loaded server). The
   llama.cpp parser strips that suffix; the EXL3 parser did not.
2. With (1) repaired, the classifier rejected Triton's runtime-compiled CUDA
   driver-API helper (`cuda_utils.cpython-312-x86_64-linux-gnu.so` beneath the
   user's Triton cache, 31,944 bytes) as a CUDA user-space library outside the
   pinned `site-packages` tree. Triton 3.6.0, a pinned torch dependency whose
   version is bound by the distribution inventory, compiles it on first use;
   the model's Triton kernels load through it.
3. With (1) and (2) repaired, the gate required a per-process GPU allocation
   of at least 8,000 MiB, an unmeasured estimate. The frozen 8192-token
   Q8-cache deployment allocates 7,666 MiB at load and 7,706 MiB after a
   2,048-token completion (measured 2026-09-13 on driver 580.167.08).

Prospectively, the official attempt is `p2r1-primary-v5`; `p2r1-primary-v4` is
forensic-only, retained and never resumed or relabelled. Questions are
regenerated under identical seeds and all six grids are redone. Because a new
attempt is the only point at which the immutable harness may change, the same
amendment repairs the gate: anonymous shared memory reported as a deleted
pseudo-file (`/dev/zero`, `memfd`, SysV segments) carries no file identity and
is skipped, while any other deleted mapping, library or not, remains a fault;
Triton's runtime helper is classified separately (`triton_jit_cache`), admitted
only beneath the launching user's Triton cache root, and recorded with path,
byte count and SHA-256, so it is the only runtime-compiled shared object
allowed outside the pinned tree and the precompiled `exllamav3_ext` gate is
unchanged; and the resident-allocation bound is 7,000 MiB
(`campaign_guard.EXL3_MIN_RESIDENT_ALLOCATION_MIB`, mirrored by the post-run
validator), with the exact allocation recorded in every placement receipt as
before. Inputs, artifacts, quotas, prompts, decode/scoring rules, statistical
analysis, llama.cpp placement checks, campaign order, cool-gate thresholds and
the completion/warmup caps are unchanged.

Before this amendment the repaired stage functions were rehearsed with the
launcher's own functions against a scratch campaign directory (no attempt
namespace, question set, ledger or result row was touched): EXL3 identity,
placement (20 classified CUDA user-space, 2 system-driver and 1 Triton-helper
mappings; 7,666 MiB), warmup, six real requests through `run_grid.py`'s request
path (HTTP 200, answer line present, 4.4-9.2 s), snapshot and scoped shutdown;
IQ2S identity (exact `65/65` readback), placement, warmup and shutdown; and the
post-campaign artifact and runtime verifications. tabbyAPI returns no
token-usage object for chat completions, so EXL3 rows carry `usage: {}` while
llama.cpp rows carry usage; no analysis depends on that field.

The owner-only incident receipt is
`.publication/private/p2r1/pre-request-exl3-provenance-gate-failure-006.json`
(SHA-256 recorded in `R1_PROTOCOL.json` under
`prospective_exl3_provenance_gate_amendment`).

## What the recovery changes

- Port `18090` and fallback port `18091` must both be free before every load.
- Every server is bound to the PID/start time/executable captured immediately
  after launch, the listener-owning process tree, exact API model ID and owner,
  engine revision, artifact identity, command, config, working directory, and
  load-log prefix.
- Q4 and IQ2 use unique llama.cpp aliases. EXL3 uses a dedicated immutable
  private tabbyAPI config whose bytes and artifact path are checked live.
- Q4 and IQ2 freeze exact llama.cpp verbosity 4. Their private loader logs must
  contain exactly one placement readback (`33/66` and `65/65`, respectively)
  before any request is permitted.
- Full multi-GiB artifact hashing happens before the campaign and again only
  after every timed stage and verified shutdown. Live identity checks use the
  frozen hash record plus file count, size, and mtime, so they do not prime the
  page cache between load and timed requests. Promotion binds the exclusive
  post-campaign full-hash receipt to the original three artifact records.
- The llama.cpp launch environment resolves `libcudart.so.12`,
  `libcublas.so.12`, and `libcublasLt.so.12` only from the private CUDA 12.6
  toolkit frozen by the build. Their versioned filenames, byte counts, and
  SHA-256 hashes are checked before timing; each live llama.cpp process must
  map those exact files; and the same resolution and hashes are checked after
  all timing and shutdowns. Private absolute paths remain in campaign evidence;
  public claims receive only a path-free projection.
- EXL3 package probes, installed-content fingerprinting, and the tabbyAPI launch
  remove every inherited `LD_*` variable and leave `LD_LIBRARY_PATH` unset. The
  toolchain receipt privately pins the virtual environment's `site-packages`
  tree and the precompiled `exllamav3_ext` file. After model load, the placement
  gate records paths, byte counts, and SHA-256 hashes for every mapping matched
  by the frozen CUDA/PyTorch-CUDA/ExLlama shared-library classifier across the
  live launched process tree. Classified CUDA user-space files must resolve
  inside that tree; only system-root `libcuda.so` and `libnvidia-*.so` driver
  files and Triton's runtime-compiled `cuda_utils` helper (beneath the user's
  Triton cache root, recorded with its hash; 2026-09-13 amendment) may sit
  outside it. Anonymous shared memory that the kernel reports as a deleted
  pseudo-file is skipped; any other deleted mapping is a fault. The launched
  tree's per-process GPU allocation must be at least 7,000 MiB (measured
  7,666-7,706 MiB). This is a containment check for the declared classifier
  and one live snapshot, not a claim to enumerate every possible GPU-runtime
  mechanism.
- The campaign is bound to one exclusive host inventory: i9-14900HX, 32 logical
  CPUs, 62.4 GiB system RAM, RTX 4080 Laptop GPU with 12,282 MiB reported VRAM,
  current/requested/default 80 W limits, and NVIDIA driver 580.167.08. Every
  session campaign, promotion, and claims rebuild must agree with that receipt.
- Q4 generates candidates across all 126 TSI and 160 WildChat frame windows.
  Selection is one QA per window, exact predeclared stratum quotas, deterministic
  ID-only ranking, and preference for distinct source conversations.
- The private question journal retains the exact HTTP response body bytes,
  exact message content, and the separately derived parser-visible text, each
  with a hash. This permits replay of parsing and rejection decisions without
  publishing raw conversation or response content.
- Q4 is shut down after question generation. The host must pass the same cool
  gate before a clean Q4 evaluation reload. Each evaluation deployment receives
  one deterministic, recorded, unmeasured warmup before TSI and WildChat.
- Before every model stage, three consecutive ten-second cool-gate samples must
  show at least 32 GiB `MemAvailable`, no NVIDIA compute process, at most
  1,024 MiB baseline VRAM, at most 50 C GPU temperature, and one-minute load at
  most 2. Device-total utilization is recorded and range-checked but does not
  gate ordinary graphical-desktop composition.
- The fixed evaluation order is Q4, EXL3, IQ2; within a deployment it is TSI,
  then WildChat. Conditions are deterministically interleaved within item.
- Request `wall_s` times only the chat-completion HTTP call. Endpoint guards,
  compressor preprocessing, model load, cool waits, and warmup are excluded.
- Every evaluation terminal privately retains bounded exact HTTP body bytes,
  body hash, status, reason, headers, exact raw message, and the separately
  hashed visible/scored projection. Identity-preserved HTTP or parsing failures
  are retained once as incorrect outcomes with their transport evidence.
  Response-model mismatch or loss/overflow while retaining evidence invalidates
  the named attempt; neither is converted into an ordinary incorrect answer.
- Shutdown is scoped to the exact recorded process group. There is no `pkill`
  or pattern-based termination.
- An interrupted stage records an exclusive `*-abort.json` host-state receipt
  before scoped shutdown. Append-only rows may be resumed in a later verified
  session; promotion requires exactly one normal post-state or explicit abort
  state for every session it uses.
- Every question, session, response, and failure event carries the previous
  event's SHA-256 plus its own checksum and is fsynced. Each append is then
  anchored by an atomically replaced, owner-only `.head.json` recording the
  exact path, event count, byte count, and head digest. A missing/stale head,
  chain mismatch, selective row/stream deletion, crash after event fsync but
  before head replacement, or unterminated tail permanently writes
  `ATTEMPT_INVALIDATED.json`. The original tail is retained in private forensic
  quarantine and any verified prefix is inspection-only. Preserve the whole
  namespace. A retry is forbidden unless a prospective, committed protocol
  amendment records the failure, new official attempt ID, and reason before
  any new inference; never resample merely until a usable result appears.
- Generated questions and all six grids live beneath one explicit attempt ID.
  A crash after a generation/evaluation session is fsynced but before its first
  request intent permanently marks that attempt namespace forensic-only. No
  replacement session may hide the gap. The frozen attempt becomes
  forensic-only; a complete retry requires the explicit committed amendment
  described above. Eligible later-session resumes must reuse the same ID.
- Immediately before every generator or evaluation POST, the harness fsyncs a
  unique intent containing the item/condition or window/attempt key, seed,
  prompt hash, session, endpoint receipt, and sampling contract. Exactly one
  terminal must follow. A kill after the response but before evidence is
  durable therefore leaves an unmatched intent and invalidates that named
  attempt instead of causing a silent reissue. Evaluation request errors with
  endpoint identity still intact are retained once as incorrect outcomes;
  generator errors advance only through the three predeclared retry seeds.

## Inputs and rights boundary

WildChat is rebuilt from revision
`7d6490e462285cf85d91eabea0f9a954fbddcd1f`, with the Parquet shard size and
SHA-256 checked. ODC-By covers database rights; it does not establish all
independent content or privacy rights. Raw WildChat text, excerpts, questions,
answers, upstream IDs/hashes, responses, and the source map remain private.
The deterministic local IDs are reproducible pseudonyms and therefore stay
private too. Public derivatives are aggregate-only: counts, rates, timing
summaries, test statistics, and WildChat attribution. No per-item rows or IDs
are released unless a later rights/privacy review approves a secret-random,
nonlinkable remap under a separately documented policy.

TSI text and per-item content also remain private. Its old A/L2/E derived
columns are not trusted. The R1 freeze keeps original text and opaque
source-conversation grouping, recomputes E from the original with the same
committed deterministic ladder used for WildChat, and then recomputes A and L2
under the pinned R1 toolchain. The frozen output hashes are authoritative.

## Exact commands

Run the commands below from the repository root. They assume the existing
model layout beneath `$HOME/models` and do not publish anything.

Static self-test (no inference, no network):

```bash
bash testsuite/paper2/p2_grid_r1.sh self-test
```

Freeze corpus inputs and recompute E/A/L2. This may download the pinned
WildChat shard and runs the local LLMLingua compressor on CPU, but starts no
benchmark server and sends no benchmark request:

```bash
bash testsuite/paper2/p2_grid_r1.sh prepare-inputs
```

Create the private EXL3 runtime config:

```bash
P2R1_MODEL_ROOT="$HOME/models" bash testsuite/paper2/p2_grid_r1.sh prepare-runtime
```

After committing the prospective protocol/harness, run the complete
analysis-only preflight. It performs full artifact checksums and static/live
host checks but starts no server and makes zero POST requests:

```bash
P2R1_MODEL_ROOT="$HOME/models" bash testsuite/paper2/p2_grid_r1.sh preflight
```

The explicit inference command is separate. Do **not** run it while another
GPU workload is active:

```bash
P2R1_ATTEMPT_ID=p2r1-primary-v3 \
P2R1_MODEL_ROOT="$HOME/models" \
bash testsuite/paper2/p2_grid_r1.sh run
```

`run` holds a campaign-wide file lock, refuses dirty/uncommitted protocol state,
uses non-overwriting private paths, and ends by invoking the post-run validator.
It never publishes or updates a paper automatically. Reuse
`P2R1_ATTEMPT_ID=p2r1-primary-v3` only to resume a still-eligible interrupted
campaign or restart after the documented zero-request campaign above. If the
harness creates `ATTEMPT_INVALIDATED.json`, preserve that namespace and stop.
A retry is forbidden until a prospective committed
protocol amendment freezes a replacement official attempt ID and the reason;
do not move or rewrite the old evidence or invent an ID at runtime.

## Runtime and human prerequisites

The six grids contain 2,844 timed requests. Question generation adds 286 to
858 unscored requests depending on retries, plus three warmups. Based on the
invalid campaign's observed request times—but not its scientific labels—the
likely wall time is about 11--16 hours. Cooling, loads, or retries can extend it;
reserve 20 hours. Input preparation and full hashing are additional.

Before `run`, a human must ensure:

1. the recovery protocol and harness are committed;
2. no unrelated GPU/CPU benchmark is running;
3. the machine can remain powered and thermally stable for the full window;
4. the exact private inputs, runtime config, and artifact records passed
   preflight; and
5. at least 32 GiB of system memory is available at every stage gate; and
6. enough disk space remains for full private responses and receipts.

No manual choice is permitted after results are seen. A failed or incomplete
stage is resumed only through its immutable contract and a newly registered
identity session; historical rows are never overwritten.

The exception is an append that was interrupted inside an event. Because it is
impossible to know whether the model response completed before the durable row,
the tail-recovery receipt makes that result series non-resumable. Recovery is
forensic, not scientific evidence.

The prospective measurement bundle is hash-bound to every run. Post-run
validator and statistics implementations are a separate, committed analysis
bundle captured in the promotion record. A reviewed analysis-code correction
therefore requires a fresh promotion audit, but does not relabel or invalidate
otherwise valid inference requests.

## Promotion gate after the run

The manuscript and explainer stay invalidated until all of the following pass:

- six exact `P2R1-{Q4KXL,EXL3,IQ2S}-{TSI,WC}` runs, 24 complete cells, 117 TSI
  and 120 WildChat items per condition;
- valid contracts, registered sessions, identity/placement/warmup/shutdown
  receipts, and recomputed scores for every retained response;
- exactly one Q4_K_XL `33/66` and one IQ2_S `65/65` loader placement readback
  in each applicable private stage log, with exact deployed `-lv 4` argv;
- an LD-free EXL3 launch receipt whose live classified CUDA user-space mappings
  are contained by the frozen private `site-packages` tree and whose mapped
  precompiled extension matches the installed-content hash;
- the post-run promotion record produced by `paper2/scripts/validate_r1.py`;
- unchanged pre/live/post CUDA 12.6 dynamic-runtime evidence for both
  llama.cpp tiers, with only its path-free projection entering public claims;
- claims rebuilt only from P2R1 evidence, with interaction inference clustered
  by the private `source_cluster_id` grouping and the actual cluster count;
- the study-level interaction branch set only by Holm step-down across all 12
  two-sided model-assisted CR2/Satterthwaite source-conversation-cluster-robust
  tests at nominal familywise alpha 0.05; all 12 Satterthwaite reference degrees
  of freedom must be at least 5 within numerical tolerance, and a nonzero
  contrast with zero CR2 standard error stops promotion for statistical review;
  clustered 95% intervals remain pointwise
  descriptive. Exhaustive sign
  flipping is secondary and exact only conditional on cluster-sign symmetry;
- those model-assisted intervals and tests condition on the one frozen
  Q4-generated, eligibility-selected QA set and one seeded decode per cell.
  They use an independent source-conversation-cluster superpopulation
  approximation; they do not integrate alternate question-generation seeds,
  decoder/run-to-run variation, or provide design-based inference for the
  deterministic stratified source frame;
- exact aggregate disclosure of evaluation request outcomes and generator
  transport errors, separated from parser/verifier rejection counts;
- no equivalence or exact-independence claim from a nonsignificant test;
- deployed artifact--engine--placement language rather than a causal claim
  about weight precision alone;
- character-retention language rather than unmeasured KV-cache/VRAM savings;
- end-to-end request-time language without an unsupported output-length,
  prefill, or decode mechanism;
- a fresh bibliography/prior-art check, privacy/export scan, light/dark render
  review, and numerical assertion audit; and
- a SHA-256-bound public allowlist covering every canonical source/figure plus
  an exact inventory of each dedicated `publication/papers/.../public` staging
  root; unlisted extra files and per-item/linkable IDs are release blockers;
- regenerated light/dark quantitative figures whose eight exact hashes are
  bound by the post-result narrative receipt; figure code and this Makefile are
  part of the promotion-time analysis source inventory;
- an explicit post-result narrative receipt binding the observed qualitative
  result branch to the title, abstract, conclusion, explainer, and their exact
  source hashes (synthetic or contrary results cannot open canonical builds);
- a truthful AI-assistance disclosure describing drafting/coding assistance,
  Matthew Schwartz's verification of claims and citations, and his full
  responsibility for the work.

Only after those gates should `main.tex`, `explainer.tex`, figures, derived
claims, and a public release package be promoted. Publication remains a
separate explicit approval.

After rewriting the result-specific sources, render the four unmistakably
noncanonical review PDFs with `make -C paper2 review-candidates`. The candidate
source gate rejects the invalid P2G body and any result/protocol mismatch. The
author visually reviews those exact light/dark paper and explainer PDFs.

After the aggregate claims, eight figures, four reviewed PDFs, and a dedicated
sanitized public tree exist, build (but do not publish) the exact privacy-scan
inventory with:

```bash
P2_PUBLIC_RELEASE_ROOT=publication/papers/<paper-2-slug>/public \
make -C paper2 public-allowlist
```

The builder runs the pre-figure promotion/claims gate, hashes every canonical
source, figure, and reviewed PDF, inventories every file beneath that exact
public root, and writes `derived/public_release_allowlist.json`. It copies and
publishes nothing. The later narrative receipt binds the allowlist hash and all
four reviewed-PDF hashes. Canonical Make targets run the full privacy/editorial
gate and then copy those exact reviewed bytes; they never rerender a different
PDF after review.

The canonical PDF gate expects
`paper2/derived/postrun_narrative_audit.json` with schema
`paper2-r1-postrun-narrative-audit-v1`. It must bind the exact promotion,
claims JSON/TeX, manuscript, explainer, and bibliography hashes; copy the
claims file's complete qualitative-assertion object and interaction result
branch; record affirmative title/abstract/conclusion, explainer, and
AI-disclosure review; affirm explicit review of P2R1 protocol facts,
rights/privacy, the timing boundary, multiplicity, the hash/receipt
reproducibility chain, host provenance, and all quantitative figures; bind the
exact hashes of the public allowlist, four reviewed light/dark paper/explainer
PDFs, and the four light and four dark figure PDFs; and quote one exact result-specific
source assertion
from each of the title, abstract, conclusion, and explainer. Both TeX sources
must carry the machine-readable comment
`P2R1_RESULT_BRANCH: <observed-branch>`. This receipt is intentionally a human
editorial gate, not something the inference launcher creates. The canonical
gate also rejects P2G-only statements such as unseeded generation, the old
105/15/0 and 103/14 frame summaries, or claims about an already prepared public
artifact. Both sources must instead state the 117/120 totals, exact stratified
quotas and predeclared retry-seed schedule, corrective/non-independent status,
character rather than KV-cache/VRAM estimand, request-only timing boundary,
WildChat rights/privacy boundary, deployed-tier interpretation, Holm familywise
rule, and receipt/hash reproducibility chain.
Both sources must also state the CR2/Satterthwaite primary rule, the secondary
conditional role of sign flipping, the single-question-set/single-decode
inference scope, and the exact observed evaluation and question-generation
request-error counts.
