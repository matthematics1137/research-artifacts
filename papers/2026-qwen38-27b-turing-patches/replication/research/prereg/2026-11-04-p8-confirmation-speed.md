# Pre-registration: P8's confirmation speed run on the 2080 Ti

Registered 2026-11-04 14:12 EST by the P8 fork, before any arm of this run has executed, and committed before the queue
step exists. Gate record: `research/gates/2026-11-04-p8-confirmation-speed.md`. The arms follow the competition
refresh (`research/TURING_COMPETITION_REFRESH_2026-11-04.md`, commit 02d3eb3) and the ambassador's arm list. It runs
after `x-p8lc-replication` ends. No new optimization is part of it (Matt on research-ledger-reference: "only the story
so far").

## Question

In one session on the stock 11 GiB RTX 2080 Ti at 260 W, how fast does each engine decode the unmodified
Qwen3.8-27B, undrafted and drafted, at an empty context and at 28K? Under test:
- F2's speed of record for w13 and its ratios over the stock wheel;
- w13 against exllamav3 1.6.6, NInferno, llama.cpp v0.9.0, rafatxf's branch, ik_llama.cpp and buun-llama-cpp;
- the wording "the fastest original-weights decode we found on a Turing card of 12 GB or less".

## Files (frozen by this commit)

| File | sha256 (first 16) | Role |
|---|---|---|
| `tools/turing/confirm/plan.json` | f670ed92af01d5e9 | every engine, model path, flag, condition and priority |
| `tools/turing/confirm/run_confirm.py` | 9088eadffce1830a | the session driver and the server client |
| `tools/turing/confirm/analyze_confirm.py` | d713742661b4b82e | the analysis below, written before any row exists |
| `tools/turing/confirm/build_confirm.sh` | fabb2c1676fc8503 | the builds (started 2026-11-04 14:02 EST, CPU only) |
| `tools/turing/confirm/test_confirm.py` | 23f37023b5daf31d | CPU tests: plan, commands, both stream formats, budget rule, analysis (6 pass) |

## Arms

Every condition is a fresh process. Three rounds, rotated as `ab_speed.sh` rotates. Fills: 0, or 28,000 tokens of
`census.filler` text ahead of the prompt (28,050-token prompts).

| Priority | Engine (build) | Weights | Conditions |
|---|---|---|---|
| 1 (A5 and F2's anchor) | stock exllamav3 1.6.5 wheel; w13 on 1.6.5 | EXL3 2.0 bpw, cache 8,8 | undrafted and MTP, at 0 and 28K (8) |
| 1 (A5) | stock exllamav3 1.6.6 wheel (cu128, torch 2.10.0, `--no-deps` overlay); w13 rebuilt on 1.6.6 (`exl3-w13on166`) | same | undrafted and MTP, at 0 (4) |
| 1 (A1) | NInferno 1d3b43e3 (Apache-2.0), sm_75, CUDA 13.4, `--spec none`, its automatic KV and activation choices | UD-IQ2_XXS and UD-IQ2_S | undrafted, at 0 and 28K (4) |
| 1 (A2) | llama.cpp v0.9.0 cd959bae9, sm_75 | UD-IQ2_XXS + DFlash2 Q4_K_M (`draft-dflash`, n-max 7); UD-IQ2_S + the full 1.37 GB MTP sidecar (`draft-mtp`, n-max 3) | drafted, at 0 and 28K (4) |
| 2 | llama.cpp v0.9.0 | UD-IQ2_XXS | undrafted, at 0 and 28K (2) |
| 2 | rafatxf turing 15654d3 (1.5.3), cache 4,4 with fdq4, gcc 13 | EXL3 2.0 bpw | MTP, at 0 and 28K (2) |
| 3 (A3) | ik_llama.cpp 009dce19, sm_75 | UD-IQ2_XXS, UD-IQ2_S; UD-IQ2_S + full sidecar (`mtp:n_max=3`) | undrafted (2) and MTP, at 0 (3) |
| 4 (A4) | buun-llama-cpp a8022d24, sm_75 | the EXL3 directory itself, its MTP layer | undrafted and `draft-mtp`, at 0 (2), plus 100 GSM8K items with MTP |

Weights: the EXL3 conversion and the two GGUFs were hash-audited against the Hub on 2026-09-06. The full MTP sidecar
was copied from the laptop on 2026-11-04 and matches its audited SHA-256 (50d9ce5a…). No model is downloaded.

**The exllamav3 family** runs `tools/census.py` unchanged, as F2 did: `-cs 32768 -rcs 2 -chunk_size 1024`, MTP as
`-mtp -ndt 6 -dds`, the overlay on `PYTHONPATH` and the switches of F2 (w13 undrafted `EXL3_GEMV_SK=1
EXL3_INT8_SQ_MAX_M=1`, MTP `2` and `2`, both `EXL3_MGEMM_N_THRESHOLD=0 EXL3_FD75D_LIVE=0`). Its sampler is exllamav3's
default (temperature 0.8, min-p 0.08, unseeded), thinking off, three warm-up decodes, then 256 tokens per prompt.

**Server engines** get the same three prompts, Qwen's template with thinking off, the same filler text, the same
sampler (temperature 0.8, min-p 0.08, top-k off, top-p 1) and 256 tokens with the stop tokens ignored. llama.cpp-family
servers take the raw templated prompt on `/completion`; NInferno takes chat messages with `enable_thinking: false`
and its `ignore_eos`, `top_k` and `min_p` extensions. Server flags follow the lab's lanes: `-ngl 99 -np 1 -ctk q8_0
-ctv q8_0 -fa on --jinja -b 512 -ub 256 -c 32768` (with `-ctxcp 4 -cms 1024` where the engine has them); NInferno
`--max-context 32768 --max-concurrency 1`. One 32-token warm-up request precedes the measured ones.

**Timing.** As `census.py` times it: from the first streamed token to the last, over the tokens generated. The
server's own `timings` are recorded beside it. Speed is pooled: total tokens over total decode time.

**Before round 1:** each overlay is checked by importing it without the GPU; each server binary must exist; NInferno
converts each GGUF once (into the GGUF's folder) in an untimed start.

## Budget and cut rule

- **Expected:** about 2.3 GPU-hours. A round is about 38 minutes (F2's measured process times for the exllamav3 arms,
  about 1 minute per server condition at an empty context and 2.5 at 28K), plus about 10 minutes of preparation and
  12 for the GSM8K check.
- **Cap: 2.5 hours.** Before rounds 2 and 3 the driver projects the session's end from the measured condition times.
  Over the cap, it cuts the lowest priority first (4, then 3, then 2), the 28K conditions of a priority before its
  empty-context ones. Priority 1 (A1, A2, A5 and the 1.6.5 anchor) is never cut.
- **Failures.** A condition that fails to start, runs out of memory, errors or times out (15 minutes) is dropped for
  the rest of the session and reported. Nothing falls back silently: each overlay's version is checked, and each
  server's command line and startup log are kept.
- **Limits:** one job in the desktop queue, 260 W, the guard on, no sudo beyond the queue's power rule, a pause
  above 84 °C until 78 °C, and memory back at the session's baseline before the next condition starts.

## Conditional arms (each decided at run time, never by downloading)

- **NInferno with DFlash2.** It needs `z-lab/Qwen3.8-27B-DFlash2` (rev 50307d4c, safetensors, 3.85 GB), a download
  only Matt can approve. It runs, one round at 0 and 28K, only if those files are in the lab's Hugging Face cache before
  the step starts. Otherwise it is reported as not run.
- **240 W replicate** (NInferno UD-IQ2_XXS and w13 undrafted, at 0 and 28K, one round), to match NInferno's published
  rows. It runs only if `sudo -n -l /usr/bin/nvidia-smi -pl 240` succeeds. On 2026-11-04 the desktop's rule allows 260,
  290 and 320 only, so it is expected not to run.

## Predictions (written before any arm runs)

1. **w13 on 1.6.5, F2 discounted:** undrafted at 0, 43-48 tokens/s (F2 47.53, a fast session); MTP at 0, 64-72 (F2
   71.61).
2. **Over the stock wheel:** 1.75-1.95x undrafted and 1.45-1.65x with MTP at 0; at least 1.9x undrafted at 28K.
3. **1.6.6:** each 1.6.6 arm within ±5% of its 1.6.5 counterpart (the decode path is byte-identical).
4. **rafatxf:** w13 with MTP ahead by 3-12% at 0 (F2 +8.2%); at 28K between −3% and +10% (F2 +5.5%).
5. **NInferno, undrafted:** UD-IQ2_XXS at 42-48 tokens/s at 0 (published 43.7 at 240 W), and UD-IQ2_S 8-18% slower
   than UD-IQ2_XXS (its per-token bytes are 16% more). w13 undrafted ahead of NInferno UD-IQ2_XXS by 0-12% at 0.
6. **llama.cpp v0.9.0:** IQ2_XXS + DFlash2 at 50-65 and IQ2_S + MTP at 40-55 at 0; undrafted IQ2_XXS at 23-28. w13 with
   MTP at least 1.2x every llama.cpp drafted arm at 0.
7. **ik_llama.cpp:** undrafted UD-IQ2_XXS 0-25% faster than llama.cpp v0.9.0's.
8. **buun-llama-cpp:** undrafted 20-35 tokens/s on the EXL3 directory (its dp4a paths at 2.0 bpw); its GSM8K accuracy
   within 3 points of the stock EXL3 lane on the same 100 items.
9. **The fastest wording** survives undrafted and drafted at 0.

## Analysis (`analyze_confirm.py`, fixed now)

- Pooled tokens/s per condition, and per round.
- **F2:** each w13-on-1.6.5 condition against F2's pooled value. Inside the cross-session floor (3.6% undrafted,
  5.5% drafted), F2's numbers stand. Outside it, this run becomes the speed of record and the paper calls F2 a fast
  session.
- **Ratios and leads:** w13 against every other arm of the same drafting and fill (the pairs in the script), each with
  its per-round record.
- **Wording rules:**
  - "faster than X" only if w13 wins every round at that fill and drafting, otherwise the per-round numbers as they
    are;
  - "the fastest original-weights decode we found on a Turing card of 12 GB or less" only if w13 beats every other
    arm of the same drafting and fill in every round, undrafted and drafted at 0; otherwise the sentence goes, and the
    paper gives the ranking;
  - 1.6.6 replaces 1.6.5 in the paper only if a 1.6.6 arm beats its 1.6.5 counterpart by more than 5% in every round.
- **buun's GSM8K check:** accuracy on the first 100 GSM8K-500 items against the stock EXL3 MTP lane's same items,
  paired, exact McNemar, descriptive.

## What it does not change

Accuracy claims rest on the quality lanes and the long-context registrations. Only buun's 100-item check measures
accuracy here. Any deviation is written as a dated amendment before the step it affects, or reported as found.

## Deviations recorded before the step ran

- **The first confirm build failed FFmpeg's configure.** The desktop has neither nasm nor yasm, so NInferno, which
  links FFmpeg's libraries, failed after it. The ambassador rebuilt both with `--disable-x86asm` (commit 8c84aea, unit
  `turing-build-confirm2`, `logs/build-confirm-2.log`). That flag slows only FFmpeg's own media code, which no
  text-decode path uses. The queued step now waits for that log's "confirm builds done" line, or for the unit to stop,
  before its preflight. No other build changed, and no condition, item or analysis rule changed. Recorded 2026-11-04 18:30 EST
  from the ambassador's note; the confirmation analysis lists it as a deviation.

## Amendment 1 (2026-11-04 21:04 EST): follow-up step `x-p8-confirm-fit`

Written before any row of either session was read. What prompted it is all from the ambassador's note of the desktop
logs:
- **Session 1** (`runs/confirm/20261104T194157`, 19:41-20:52 EST) passed every check, NInferno included, completed
  round 1 and reached `raf44-mtp-0` in round 2.
- **The guard stopped it** at 20:52:12 for a memory stall (PSI full avg10 ≥ 15). Matt's own processes hold the
  memory; the swap is full.
- **A guard stop uses no try.** The queue restarted the registered step at 20:54:40 as a fresh session
  (`runs/confirm/20261104T205440`), which follows this registration untouched. Session 1 is reported as a stopped
  session and never pooled with it.
- **In round 1, session 1 dropped 7 competitor lanes,** none of them during decoding:
  - `lcpp9-xxs-df-0` and `-28k`: out of memory allocating the DFlash2 context's compute buffer at load, with "failed
    to fit params ... n_gpu_layers already set by user to 99";
  - `lcpp9-s-mtp-0` and `-28k`: out of memory allocating the MTP context's KV buffer at load;
  - `ik-s-mtp-0`: out of CUDA memory on the first request;
  - `buun-nd-0` and `buun-mtp-0`: "couldn't bind HTTP server socket". This was the harness: its two alternating
    ports were still held by servers that had just failed.
- Session 2 is expected to drop the same lanes, which is reported as registered.

**(a) What the step runs.** One session of three rotated rounds, after `x-p8-confirm`:
- the 7 dropped lanes;
- same-session anchors: w13 on 1.6.5, undrafted and MTP, at 0 and 28K, so every comparison stays inside one session;
- buun's 100-item GSM8K check, if the registered session did not complete it;
- the conditional arms, under the registered rules:
  - NInferno + DFlash2 only if its drafter is in the lab's Hugging Face cache before this step starts (Matt has not
    approved the download);
  - the 240 W replicate only if `sudo -n -l` allows it.

**(b) The harness fix.** Before each server start, the wrapper takes a port from 18321-18340 that can be bound now (no
`SO_REUSEADDR`), waiting up to 120 s if none can. The fix and the ladder live in a wrapper,
`tools/turing/confirm/run_confirm_fit.py` with `plan-fit.json`, which imports `run_confirm.py`. `run_confirm.py`,
`plan.json` and `analyze_confirm.py` stay byte-identical: a further guard retry of the registered session loads them
from disk. The wrapper refuses to run if their SHA-256 differs from the registration.

**(c) The fit ladder, fixed before any fitted result.** Each server configuration is started at the rungs below, in
order and cumulatively. The first rung that loads and completes the warm-up is used in every round, and each lane's
rung is reported (`rungs.json`, with the GPU layers the engine logged):
- **R0:** as registered.
- **R1:** `-ub 128`, the prefill micro-batch only; decode is untouched.
- **R2:** the smallest `-c`, a multiple of 1024, that holds the longest filled 28K prompt plus 256 new tokens plus a
  1K margin. The prompt is counted by the GGUF tokenizer: llama.cpp v0.9.0's `llama-tokenize`, vocabulary only, on
  UD-IQ2_XXS (the same vocabulary as UD-IQ2_S and the EXL3 folder), with special tokens parsed. It runs on the CPU and
  is built in the step if missing.
- **R3:** no `-ngl 99`: the engine's own fit to free memory. It applies to llama.cpp v0.9.0 and buun-llama-cpp.
  ik_llama.cpp has no automatic fit, so its ladder ends at R2. An R3 result is labelled fitted with partial offload,
  with the layers left on the CPU, and is never reported as the full-offload speed.
- **The warm-up for a rung** is the registered 32-token request. A configuration that has a 28K condition must also
  complete the same request after the 28K filler, so the rung is known to hold that context. Neither request is
  timed.
- A configuration that passes no rung is dropped as "did not fit on 11 GB at full offload".

**(d) Analysis.**
- `analyze_confirm.py`, unchanged, runs on the fit session as its own session. A dropped arm is never counted as
  beaten.
- The "fastest original-weights decode we found" sentence needs two things: the registered session's rule, and w13
  beating every fitted competitor of the same drafting and fill in every round of the fit session.
- If a lane fails every rung, the paper says it did not fit on 11 GB at full offload and claims no speed lead over it.

**(e) Budget.** About 55-65 minutes, estimated from F2's measured process times and the registered per-condition
estimates:
- the ladder, about 5-10 minutes;
- 11 conditions, about 14 minutes a round;
- the GSM8K check, about 12 minutes, if buun fits.

The cap is 1 hour. Before rounds 2 and 3, the registered rule cuts priority 4 (buun), then 3 (ik_llama.cpp); the
anchors and the llama.cpp lanes are priority 1. A buun whose conditions are cut skips the GSM8K check. Predictions 6-8
apply as registered.

**Files (frozen by this amendment's commit):**

| File | sha256 (first 16) |
|---|---|
| `tools/turing/confirm/run_confirm_fit.py` | a4ab18a638b16382 |
| `tools/turing/confirm/plan-fit.json` | bab4f803f1ada7c0 |
| `tools/turing/confirm/test_confirm_fit.py` | e508f9dcaf7ffdbe |

The CPU tests (`test_confirm_fit.py`: the frozen hashes, the port wait, the ladder's order, arguments and context,
the plan, and a dry run) pass in the sandbox, with the registered suite (13 in all).


## Amendment 2 (2026-11-05 05:45 EST): follow-up step `x-p8-confirm-df2`

Written before the step runs and before any of its rows exist. The registered and fit sessions are over and judged
(commit 7a24d11).
- **What prompted it.** The registered conditional arm NInferno + DFlash2 needs z-lab/Qwen3.8-27B-DFlash2, a 3.85 GB
  download that only Matt could approve. Neither session ran the arm, because the drafter was not in the lab cache.
  Matt approved the download on 2026-11-05: "yes download it and run the session".
- **The download.**
  - The drafter is z-lab/Qwen3.8-27B-DFlash2 at 50307d4c4cde6860d4eee73e2547cd786fe8e8a4, NInferno's pinned revision
    and the repository's main.
  - It was downloaded on 2026-11-05 from 05:37:48 to 05:38:31 EST into the lab cache (`$L/cache/huggingface/hub`).
  - `model.safetensors` is 3,848,817,896 bytes, with SHA-256 `67fc76d68dc5a941…4bab65c`, equal to the Hub's LFS
    digest. Its figure also matches. config.json, README.md and .gitattributes are small git files.
- **NInferno's own plan for the registered arguments.** A dry plan on the CPU (`--plan --no-load --plan-device
  'RTX 2080 Ti,headless,linux'`, 2026-11-05) resolves `--spec dflash2 --max-context 32768 --max-concurrency 1` to:
  - DFlash2 with 7 draft tokens;
  - int8 KV and fast activations;
  - rung 1 of 6, which fits.

  That is the configuration of NInferno's published 2080 Ti row: "dflash2 ×7", KV auto, activations fast. That row
  gives 52.1 tokens/s at C=1 and 56.5 at 24K, greedy, at 240 W, measured 2026-10-19.

**(a) What the step runs.** One session of three rounds, rotated as registered. The six conditions, in base order:

| Condition | Priority | Why |
|---|---|---|
| `w13-165-mtp-0` | 1 | anchor, drafted |
| `nif-xxs-df2-0` | 1 | the registered conditional arm |
| `w13-165-nd-0` | 2 | anchor, undrafted: w13's drafting gain |
| `nif-xxs-0` | 2 | NInferno undrafted: its DFlash2 gain |
| `w13-165-mtp-28k` | 1 | anchor, drafted |
| `nif-xxs-df2-28k` | 1 | the registered conditional arm |

- **The DFlash2 arm keeps what `run_confirm.py`'s `conditional()` gives it:** its id, its engine, the config
  `nif-xxs-df2` (drafted), and the arguments `["{iq2xxs}", "--spec", "dflash2"]` plus NInferno's registered common
  arguments.
- **Everything else is as registered:**
  - the prompts and the sampler (temperature 0.8, min-p 0.08);
  - 256 new tokens and the 28K filler;
  - the timing and the rows.
- **NInferno's prepare start** runs once for each configuration before round 1, as registered. The DFlash2
  configuration's first start attaches the drafter, converting it if needed.
- **No fit ladder.** If the DFlash2 start fails to load or warm up, both its conditions are dropped and reported, as
  registered.

**(b) What it does not run.**
- **The 240 W conditional.** The desktop has no sudoers rule for it. Its registered test, `sudo -n -l`, passed in both
  sessions through the password rule, and the arms ran at 260 W (deviation 5).
- **The buun GSM8K check,** which the fit session completed.
- **Every other competitor,** measured in the registered and fit sessions.

**(c) Departures from the registered conditional, stated now.**
1. The registered conditional runs each DFlash2 fill once, at the end of a session (priority 5, round 1). Here it runs
   in all three rounds, interleaved with the anchors. That is the standard every other competitor met, so the change
   is stricter, not looser.
2. Before anything runs, the drafter is checked by size and SHA-256 against the download above. The registered rule
   only asked that its folder exist.
3. Ports come from 18341-18360 through `run_confirm_fit.py`'s chooser (Amendment 1's fix), not from the registered
   alternating pair.

**(d) Predictions (written before any row).**
- **P10.** w13 with MTP is faster than NInferno with DFlash2 in every round, at both fills.
  - The pooled lead is +20% to +50% at 0 and +5% to +35% at 28K.
  - **Basis:**
    - NInferno's published greedy rates are 52.1 and 56.5 tokens/s.
    - This session samples at temperature 0.8, which should lower DFlash2's acceptance.
    - In the registered session, NInferno's undrafted rate read 5% above its published one (45.91 against 43.7).
- **P11.** NInferno's DFlash2 gain at 0 (`nif-xxs-df2-0` over `nif-xxs-0`) lies between 1.00 and 1.30. Its
  published greedy gain is 52.1 / 43.7 = 1.19.
- **P12.** w13's MTP gain at 0 lies between 1.45 and 1.56. It was 1.50 in the registered session and 1.507 in F2.
- **P13.** The w13 anchors agree with the registered session within the cross-session floor (3.6% undrafted, 5.5%
  drafted), and `analyze_confirm.py`'s F2 check holds for each.
- **P14.** The DFlash2 start fits at 32K. NInferno's plan keeps rung 1.

**(e) Analysis, fixed now.**
- **`analyze_confirm.py`** runs unchanged on this session's rows, as its own session.
  - It writes `results.json`: the F2 check for the three w13 anchors, `over_nif_xxs_0`, and the fastest wording.
  - The fastest rule's drafted rivals at 0 and 28K are `nif-xxs-df2-0` and `-28k`, whose names hold "df".
- **`analyze_confirm_df2.py`** imports `analyze_confirm.py`'s `load()` and `versus()` unchanged. It refuses to run
  if `analyze_confirm.py` differs from its registered SHA-256. It adds four pairs and writes them to
  `results-df2.json`:
  - `over_nif_df2_0` and `over_nif_df2_28k`;
  - `nif_df2_gain_0`;
  - `w13_mtp_gain_0`.
- **Wording.**
  - "w13 is faster than NInferno with DFlash2" at a fill only if w13 wins every round at that fill in this session.
  - The "fastest original-weights decode we found" sentence keeps its registered conditions. For drafting, it now
    also needs this session's fastest verdict at both fills.
  - If the DFlash2 arm drops, the paper says so and keeps citing NInferno's published rates.

**(f) Budget.** About 45-55 minutes:
- the preflight and filler, about 3 minutes;
- NInferno's two prepare starts, about 2 minutes, plus 3-10 for the drafter's first start;
- six conditions, about 12 minutes a round, from the registered session's durations.

The cap is 1 hour. Before rounds 2 and 3, the registered rule cuts priority 2 first. Priority 1 is never cut.

**(g) Step.** It is appended to the desktop queue after this commit:

```text
x-p8-confirm-df2|sudo -n /usr/bin/nvidia-smi -pl 260 >/dev/null; export PATH=$L/venv/bin:$PATH; cd $L/tools/confirm && $L/venv/bin/python run_confirm_df2.py --plan plan.json --df2 plan-df2.json --out $L/runs/confirm-df2/$(date +%Y%m%dT%H%M%S)
```

**Files (frozen by this amendment's commit):**

| File | sha256 (first 16) |
|---|---|
| `tools/turing/confirm/run_confirm_df2.py` | 730d92cd09c504a0 |
| `tools/turing/confirm/plan-df2.json` | 2acc1eeeadeba42c |
| `tools/turing/confirm/analyze_confirm_df2.py` | caad682127dfc1fe |
| `tools/turing/confirm/test_confirm_df2.py` | 559eb085baa75c98 |

The CPU tests pass in the sandbox: 24 in all, `test_confirm_df2.py`'s 10 with the registered and Amendment 1 suites.
They cover the frozen hashes, the conditions and the DFlash2 arm's registered arguments, the drafter check, the
ports, a dry run and the addendum pairs.

## Amendment 3 (2026-11-05 06:12 EST): follow-up step `x-p8-confirm-df2b`

Written after Amendment 2's session ran and before this step runs. Amendment 2's anchors-only session has been
analysed, as registered, with `analyze_confirm.py` and `analyze_confirm_df2.py`. Its results do not bear on this step's
DFlash2 comparison, which has no rows yet.

- **What happened in Amendment 2's step.**
  - It started at 05:46:00 EST. The drafter check passed, on size and SHA-256.
  - The DFlash2 arm failed at its first start, within 4 seconds: "FATAL server failed during startup | Qwen3.5
    config: missing component dflash2".
  - Both DFlash2 conditions were dropped, as registered. The session ran its other four conditions in three rounds
    and ended at 06:03:25.
- **Why the arm failed.**
  - NInferno serves a local GGUF from its own text-only conversion, in `<dir>/.ninferno-cache/`. A start does not
    attach a local GGUF's separate drafters: `src/product/plan/plan_speculation.cpp:252` says "a hub model's start
    attaches them, a local one's not".
  - NInferno's published 2080 Ti command serves the hub model, `unsloth/Qwen3.8-27B-GGUF --quant UD-IQ2_XXS
    --with-drafters --spec dflash2 --draft-tokens 7`, with no `--max-context`. Its hub flow converts the target
    together with the registry's drafter.
  - The registration pointed the arm at the local GGUF. NInferno's CPU dry plan, which Amendment 2 cites, does not
    check an artifact's components.
- **CPU checks after that session, on the desktop, with no GPU visible:**
  1. `ninferno-serve <GGUF> --with-drafters --spec dflash2 --prepare-only` reused the text-only conversion and
     attached nothing.
  2. NInferno's converter, built from the same commit (1d3b43e3) as `ninferno-convert`, has to be given the
     component explicitly.
     - Without `--components` it writes a text-only artifact and reports the DFlash2 drafter as "not in the
       artifact". That artifact is in `models/gguf/ninferno-df2/` and is not used.
     - The `mtp` component cannot be built: the GGUF has no MTP layers.
  3. With the component, the registry's drafter source and its quantization, the converter wrote the artifact this
     step serves, in 95 s:
     - the artifact: `models/gguf/ninferno-df2b/Qwen3.8-27B-UD-IQ2_XXS.qwen3_5_bf16.ninferno`, 9,486,035,968
       bytes, SHA-256 `98fafe128d9dccef…df9f5896`;
     - the command: `ninferno-convert --model <GGUF> --out <artifact> --components text,dflash2 --source
       dflash2=<the drafter's snapshot> --drafter-quant q8 --with-drafters --model-id unsloth/Qwen3.8-27B-GGUF`,
       offline;
     - its log reports `drafter dflash2 … z-lab/Qwen3.8-27B-DFlash2 embedded`, and its speculation manifest names
       revision 50307d4c and quantization q8, as NInferno's registry does.
  4. NInferno's dry plan for that artifact, with its drafter's weights counted, on a named 11 GB 2080 Ti:
     - at the registered `--max-context 32768`: "no candidate holds 32768 tokens";
     - at 29,696, the smallest context that holds the 28K-filled prompt (Amendment 1's R2 rule, as computed in the
       fit session): "no candidate holds 29696 tokens";
     - with NInferno's automatic context: 24,192 tokens, its rung 58 of 132. That fits NInferno's own published
       2080 Ti DFlash2 rows, measured after a 22,224-token prompt.

**(a) The changes, all for the DFlash2 arm.**
1. **The model.** It serves the artifact above in place of the GGUF, with the registered `--spec dflash2`. The
   artifact's text component comes from the same registered GGUF, by the converter that NInferno's own start uses.
   Its DFlash2 component is the downloaded drafter at NInferno's pinned revision, quantized to q8 as the registry
   names. Before anything runs, `run_confirm_df2b.py` checks the artifact by size and SHA-256 beside the drafter
   check; a mismatch drops the arm.
2. **The context.**
   - At an empty context, NInferno's automatic choice, as its published command: no `--max-context`.
   - At 28K, `--max-context 29696`, Amendment 1's R2 rung, since 32768 cannot fit. If that start fails, the
     condition is dropped as "did not fit on 11 GB at full offload with its drafter", as Amendment 1 reports such
     lanes. No further rung is tried: NInferno's automatic context cannot hold the 28K prompt.
3. **The stop rule.** If the DFlash2 arm's first start fails at NInferno's prepare step, the session stops before
   round 1. It would otherwise only repeat Amendment 2's anchors.

**Unchanged, as in Amendment 2:**
- the conditions, priorities, rounds, rotation and cap;
- the undrafted NInferno arm on the GGUF at 32768;
- the analysis (`analyze_confirm.py` and `analyze_confirm_df2.py`) and the wording rules.

**Implementation.**
- The step uses ports 18361-18380.
- `run_confirm_df2.py` and `plan-df2.json` are checked against Amendment 2's SHA-256 and imported, never edited.
- The journal's session event keeps Amendment 2's label. An `amendment` event before it names this step and its
  DFlash2 arguments.

**(b) Predictions.** P10-P13 stand as registered. P14 ("the DFlash2 start fits at 32K") is false by NInferno's own
plan, before any start, and is reported as failed.

New, written before any row:
- **P15.** At an empty context, NInferno's DFlash2 arm starts and decodes at 40-60 tokens/s. P10's lead at 0
  (+20% to +50% for w13) stands.
- **P16.** At 28K, the start at 29,696 fails, by NInferno's plan. So there is no drafted comparison at depth, and the
  paper says NInferno with DFlash2 holds at most about 24K tokens of context on this card.

**(c) Amendment 2's session, reported as run.** It ran anchors and NInferno undrafted, three rounds each, and
`analyze_confirm.py` analyses it as its own session:
- w13 with MTP at 0 and 28K: 73.00 and 66.48 tokens/s;
- w13 undrafted at 0: 47.96;
- NInferno undrafted at 0: 46.01.

Against those results:
- P12 holds: w13's MTP gain at 0 is 1.522.
- P13 holds: the anchors are within 0.2-1.7% of the registered session and within F2's floor.
- w13 led NInferno undrafted by 4.2%, in every round.
- The DFlash2 arm is reported as failed at its first start, for the reason above. Its rows are not pooled with this
  step's.

**(d) Budget.**
- About 35-45 minutes: the DFlash2 arm at 28K is expected to fail within seconds.
- The cap is 1 hour.

**Files (frozen by this amendment's commit):**

| File | sha256 (first 16) |
|---|---|
| `tools/turing/confirm/run_confirm_df2b.py` | 2841d4df659c9c69 |
| `tools/turing/confirm/plan-df2b.json` | 52b40daccc1156b0 |
| `tools/turing/confirm/test_confirm_df2b.py` | 8145ec474f2b366f |

The CPU tests pass in the sandbox: 32 in all, `test_confirm_df2b.py`'s 8 with the earlier suites. They cover:
- the frozen files, including Amendment 2's;
- that only the artifact, contexts, ports and labels differ from Amendment 2;
- the DFlash2 arguments and contexts;
- the artifact check;
- the stop rule;
- a dry run.
