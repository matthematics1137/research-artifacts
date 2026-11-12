# Turing lab: serving lanes, speed client and the llama.cpp tuning sweep

Paper 8 tooling for Qwen3.8-27B on p8-host's stock 11 GB RTX 2080 Ti (plan:
`research/TURING_EXL3_RESEARCH_2026-10-30.md` sections 5-6; results so far: `research/TURING_SMOKE_2026-10-30.md`).
Everything runs on the desktop inside `/opt/p8` (`$L`), through `desktop_run.sh` (caps)
under the `turing-guard` unit (yield guard). Nothing here deletes files.

| File (repo `tools/turing/`) | Deployed to | What |
|---|---|---|
| `lanes.sh` | `$L/tools/lanes.sh` | start, stop and list the eleven OpenAI-compatible serving lanes, one at a time |
| `tabby/exl3-mtp.yml`, `tabby/exl3-nodraft.yml` | `$L/tools/tabby/` | upstream TabbyAPI configs for the EXL3 lanes (`lanes.sh` derives the `-fp32` lanes from them) |
| `speed_bench.py` | `$L/tools/speed_bench.py` | stdlib streaming speed client; rows to `$L/runs/speed.jsonl` |
| `llamacpp_sweep.sh` | `$L/tools/llamacpp_sweep.sh` | E1 equal-tuning llama-bench sweep; rows to `$L/runs/llamabench_sweep.jsonl` |

Deploy from the laptop (always edit the repo copy):

```bash
L=/opt/p8
ssh p8-host "mkdir -p $L/tools/tabby"
scp tools/turing/lanes.sh tools/turing/speed_bench.py tools/turing/llamacpp_sweep.sh p8-host:$L/tools/
scp tools/turing/tabby/exl3-mtp.yml tools/turing/tabby/exl3-nodraft.yml p8-host:$L/tools/tabby/
```

## Serving lanes (`lanes.sh`)

```bash
bash $L/tools/desktop_run.sh guard              # once; every start refuses without it
bash $L/tools/lanes.sh list                     # one lane name per line
url=$(bash $L/tools/lanes.sh start exl3-mtp)    # [PORT], default 8090; prints http://localhost:8090
bash $L/tools/lanes.sh status                   # LANE <tab> state <tab> URL <tab> health=CODE, per active lane
bash $L/tools/lanes.sh stop exl3-mtp            # systemctl --user stop turing-run-srv-exl3-mtp
bash $L/tools/lanes.sh show lcpp-iq2s-mtp       # the exact command (and Tabby config) a start would use
```

`start` checks that the guard is running, that no other `turing-run-*` unit is active (no other lane, sweep or
smoke run, because only one GPU job runs at a time) and that the port is free. It then launches the server as unit
`turing-run-srv-LANE` through `desktop_run.sh cmd`: 4 cores, 20 GiB RAM, nice, idle IO, OOMScoreAdjust 800.
Before printing the base URL it waits for three things:
1. `/health` answers 200 (up to 600 s);
2. `/v1/models` answers;
3. a 16-token greedy probe generates.

If any of these fails, `start` stops the unit, prints the last log lines and exits non-zero. It does the same if it is
interrupted before it has printed the URL. That covers Ctrl-C, a wrapper's timeout (TERM), a dropped ssh session with
a terminal (HUP), and one without a terminal, which shows up as a broken pipe on the next write (PIPE). The unit is
stopped first, and the row's status is `interrupted`. The health loop's commands return within 3 s and the probe runs
under `wait`, so the clean-up starts within seconds. Without this, an interrupted start left the server running with
about 10 GB of VRAM and no row. For TabbyAPI lanes, `start` also refuses a server whose logged GDN state precision
differs from the lane's (status `precision-mismatch`; see below).

Every start, failed or not, appends one row to `$L/runs/lanes.jsonl`. The row records:
- the full command, context and overrides;
- the startup time and the probe result;
- the server's own view (TabbyAPI `/v1/model` parameters, or llama-server `/props` build info, `n_ctx` and default
  `min_p`);
- `recurrent_state` (EXL3 lanes): the `Recurrent state storage: fp16|fp32` line that TabbyAPI logs at load, read
  from this start's part of the console log (or else its file log), next to `recurrent_state_expected`.
  `/v1/model` does not report this setting;
- the GPU state after load.

Port 8090 is also `run_eval.py`'s default `--base-url`. Servers bind localhost with auth off, so clients run on the
desktop. From the laptop, use `ssh -L 8090:localhost:8090 p8-host`.

### The lanes

All lanes run at 16,384 context unless noted, with a single slot. The first six are the shared lane interface; the
other five are controls added on review (2026-10-30).

| Lane | Engine and build | Model and drafter | Reproduces (laptop) |
|---|---|---|---|
| `exl3-mtp` | upstream TabbyAPI b1c2253 + exllamav3 1.6.5 wheel | EXL3 SC 2.00 bpw + its 2-bit MTP layer: draft 6, dynamic, Q8 cache and draft cache, chunk 1024, `max_batch_size` 1, prompt checkpoint every 1024; GDN state fp16 (upstream default) | `serving/tabby-daily.yml` = E47 `EFF-h38a_mtp`, except the state precision |
| `exl3-nodraft` | same | same, `draft_mode: disabled` | `tabby_h38_nodraft.yml` = E47 `EFF-h38a_nodraft`, except the state precision |
| `lcpp-iq2xxs` | mainline llama-server d63fbfe7 | UD-IQ2_XXS, undrafted | `iq2xxs_none` (e4_configs_xxs.txt; laptop build v0.4.0) |
| `lcpp-iq2s` | mainline | UD-IQ2_S, undrafted | `iq2s_none` (e4_configs_s.txt) |
| `lcpp-iq2s-mtp` | v0.4.0-mtpshared | UD-IQ2_S + stripped shared-embedding MTP sidecar: `--spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 -ctkd q8_0 -ctvd q8_0` | `EFF-iq2s_mtps_36k` (b2.sh; same sidecar and n-max as E48) |
| `lcpp-iq2xxs-dflash2` | v0.4.0-mtpshared | UD-IQ2_XXS + DFlash2 Q4_K_M: `--spec-type draft-dflash --spec-draft-n-max 7 -ngld 99`, **8,192 context** | `EFF-iq2xxs_dflash2_n7` (e8_choice.txt, 8K) |
| `exl3-mtp-fp32` | as `exl3-mtp` | `exl3-mtp` + `recurrent_state_fp32: true` | E47 `EFF-h38a_mtp` including its fp32 state |
| `exl3-nodraft-fp32` | as `exl3-nodraft` | `exl3-nodraft` + `recurrent_state_fp32: true` | E47 `EFF-h38a_nodraft` including its fp32 state |
| `lcpp-iq2xxs-v040` | v0.4.0-mtpshared | UD-IQ2_XXS, undrafted | `iq2xxs_none` on its own build; the control for `lcpp-iq2xxs-dflash2` |
| `lcpp-iq2s-v040` | v0.4.0-mtpshared | UD-IQ2_S, undrafted | `iq2s_none` on its own build; the control for `lcpp-iq2s-mtp` |
| `lcpp-iq2xxs-dflash2-main` | mainline d63fbfe7 | as `lcpp-iq2xxs-dflash2`, **8,192 context** | none; DFlash2 on the newer build, paired with `lcpp-iq2xxs` |

Every llama.cpp lane adds the laptop's common flags, `-ngl 99 -np 1 -ctk q8_0 -ctv q8_0 -fa on --jinja -ctxcp 4
-cms 1024 -b 512 -ub 256`, srv.sh's `--host localhost --metrics`, and `--min-p 0` (see below). They also run with
`LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64`.

**Choices made, and why** (revertible with the overrides below):
- **Context 16K everywhere except the two DFlash2 lanes at 8K.** On the laptop, DFlash2 cost 2.4 GiB resident:
  9,986 MiB after the first generation at 8K on a 12 GB card. At 16K the projection is about 10.9 GiB, against about
  10.4 GiB free on the 2080 Ti. **This has not been measured on the desktop yet.** Next GPU session, run
  `TURING_LANE_CTX=16384 lanes.sh start lcpp-iq2xxs-dflash2`, then
  `speed_bench.py --lane lcpp-iq2xxs-dflash2 --tag ctx16k --prompts long8k`. A failed load stops the unit and is
  recorded. If the lane loads and long8k finishes with `stop_kind` `max_tokens` or `stop`, change `lane_ctx` to 16K
  for both DFlash2 lanes, as the lane spec asks. Until then, the 8K lanes run every speed prompt: long8k now fits 8K
  with 512 generated tokens (see below). For eval prompts at 8K, see `quality/README.md`.
- **Same-build controls for the drafted llama.cpp lanes.** The undrafted lanes run mainline d63fbfe7 (0.8.0-dev).
  The drafted lanes run the v0.4.0 fork, two months older, with different MMVQ, FA and CUDA-graph code. A
  drafted-vs-undrafted ratio across the two builds would fold the build difference into the drafter's speedup.
  `lcpp-iq2xxs-v040` and `lcpp-iq2s-v040` are the undrafted lanes on the drafted lanes' build. This is how the
  laptop's e4 undrafted lanes and its drafted lanes all ran v0.4.0. `lcpp-iq2xxs-dflash2-main` is the drafter on
  mainline, which has `draft-dflash` with the same options. The MTP sidecar lane has no mainline twin, because
  mainline cannot load the stripped sidecar. That gives two decompositions:
  - drafter effect = `lcpp-iq2xxs-dflash2` / `lcpp-iq2xxs-v040`, or `lcpp-iq2xxs-dflash2-main` / `lcpp-iq2xxs`;
  - build effect = `lcpp-iq2xxs` / `lcpp-iq2xxs-v040`.
- **DFlash2 on the v0.4.0-mtpshared build.** The laptop ran it on plain v0.4.0. The lab patch only changes models
  that lack `token_embd`/`output` (the MTP sidecar), so a full target model plus DFlash takes the plain v0.4.0 code
  path. The same holds for the two `-v040` lanes.
- **`--min-p 0` on every llama.cpp lane.** llama-server fills any sampler a request omits from its command line, and
  defaults to `min_p` 0.05 (`common/common.h`, both builds). Upstream TabbyAPI without an override preset uses
  `min_p` 0 (`common/sampling.py:154`). `run_eval.py`'s body has no `min_p`. At temperature 0 this makes no difference.
  At temperature above 0 (plan E9: temperature 1.0, several samples) the GGUF lanes would sample from a truncated
  distribution and the EXL3 lanes would not. The laptop's lanes had the same asymmetry. 0 is the one value both
  engines then share, and it is also Qwen's recommended `min_p`. lanes.jsonl's `server.min_p` and speed rows'
  `server.min_p_default` read it back from `/props`. To restore the laptop's GGUF behaviour, use
  `TURING_LCPP_EXTRA="--min-p 0.05"`; the last flag wins.
- **`-t 4`, not srv.sh's `-t 8`.** The unit is capped at 4 cores.
- **The EXL3 lanes follow upstream wherever the laptop's lab build has no upstream equivalent.** The yml headers
  list these. In short:
  - GDN recurrent state: `exl3-mtp` and `exl3-nodraft` keep upstream's fp16 default, for speed numbers (plan 6.2
    item 6). The laptop's 1.4.7 kept fp32. `exl3-mtp-fp32` and `exl3-nodraft-fp32` pin fp32 for every comparison
    with the 4080's E47 numbers (also plan 6.2 item 6), so that a desktop-vs-E47 pair does not also change numerics.
    `lanes.sh` makes those two lanes by adding `recurrent_state_fp32: true` to the base yml's per-start copy.
    Upstream merges `TABBY_*` environment variables under the `--config` file, so `TURING_LANE_ENV=
    "TABBY_MODEL_RECURRENT_STATE_FP32=true"` still works on the base lanes (the key stays out of their ymls). The
    `-fp32` lane names are clearer in eval labels. Every start records the precision TabbyAPI logged
    (`recurrent_state`), and fails if it differs from the lane's.
  - Dynamic drafting uses upstream's confidence rule (`draft_confidence` 0.4).
  - `sysmem_recurrent_cache` is 4,096 MB, not 16,384, because of the 20 GiB unit cap. It affects prefix-reuse
    cost only.
  - The laptop's `EXL3_CKPT_PP=1024` lab patch is upstream's `recurrent_checkpoint_interval_pp: 1024`.
  - Web UI and CORS are off.
  - Both ymls validate against upstream's own `TabbyConfigModel` with no unknown keys.

**Request bodies.** The lanes accept `run_eval.py`'s body unchanged:
- `model`, `messages`, `temperature`, `top_p` 0.95, `top_k` 20, `max_tokens`;
- `chat_template_kwargs: {"reasoning_effort": "medium"}`. Upstream TabbyAPI takes this as an alias of
  `template_vars`; llama-server with `--jinja` passes it to the template.

None of the 1,000 E47 rows, nor the EFF-iq2xxs_dflash2_n7 and EFF-iq2s_mtps_36k rows, had
`chat_template_kwargs_dropped`, so the laptop servers honoured it too. No lane sets server-side template defaults,
as on the laptop. Upstream's sampler `override_preset` defaults to none, also as on the laptop. The body has no
`min_p`, so each engine falls back to its own default; `--min-p 0` (above) makes that default the same for both.

**Chat templates differ between the engines, as they did on the laptop.** TabbyAPI uses the model folder's
`chat_template.jinja` (checked: the model folder wins over `templates/`). llama-server uses the template embedded
in the GGUF. The two templates differ only in:
- system/developer-message merging;
- a `high`→`xhigh` effort alias;
- tool-call validation.

A single user message at effort `medium`, or with thinking off, takes none of those branches. This was not rendered
token for token. `speed_bench.py` records each lane's `prompt_tokens`, so any mismatch shows up there.

**Overrides** (recorded in `lanes.jsonl`):
- `TURING_LANE_CTX=N`.
- `TURING_LCPP_EXTRA="-ub 512 ..."`: appended last, so these flags win.
- `TURING_LANE_ENV="GGML_CUDA_DISABLE_GRAPHS=1"`: environment for the server process. For TabbyAPI,
  `TABBY_<SECTION>_<KEY>` variables only set keys that the lane's config leaves out, because the `--config` file
  wins.
- `TURING_LANE_TIMEOUT=S`.
- `TURING_LANE_PROBE=0`.

Values must not contain `$`, because systemd-run expands `${VAR}` in unit command lines.

**Logs:**
- `$L/logs/run-srv-LANE.log`: the server console, appended with a start/stop marker.
- `$L/logs/tabby-LANE/`: TabbyAPI's own file log.
- `$L/logs/lanes/LANE-<time>-p<port>.yml`: the exact config of each TabbyAPI start.

**TabbyAPI boot shim.** Upstream `main.py` runs unchanged from `$L/src/tabbyAPI` (`env -C`, with `LD_LIBRARY_PATH`
unset as in laptop E50, and every cache inside `$L`). Before it runs, a one-line wrapper (`TABBY_BOOT` in
`lanes.sh`) changes loguru's file sink in two ways:
- it writes to `$L/logs/tabby-LANE/` instead of `src/tabbyAPI/logs/`;
- it drops the sink's `retention="1 week"` and `compression="zip"`, both of which make loguru delete files.

This was tested on a stand-in `main.py` with the same `logger.add` call: the sink moved and both settings were gone.
Upstream has no config key for this sink.

## Speed client (`speed_bench.py`)

```bash
python3 $L/tools/speed_bench.py --lane exl3-mtp --base-url "$url" --repeats 3          # desktop system python3
python3 $L/tools/speed_bench.py --lane lcpp-iq2s-mtp --base-url "$url" --limit 3      # first 3 prompts
python3 $L/tools/speed_bench.py --list                                                 # the prompt set
```

**Prompt set,** in this order. The token counts are from the Qwen3.8 tokenizer.
- `code`, `math`, `prose`: verbatim from `smoke.py`.
- `code2`: a token-bucket class.
- `math2`: a proof, by induction and geometrically.
- `prose2`: an explanation of hash-map collisions.
- `summarize`: an original 525-token incident report.
- `long8k`: a synthetic shipment ledger with a question that needs all of it. A fixed LCG generates the ledger,
  so it is identical on every machine. With 175 rows it is 7,400 tokens of user text, and 7,417 with the template
  and the `Run 1.` tag (thinking off). Prompt plus 512 generated tokens therefore fits the 8K DFlash2 lanes with
  263 tokens to spare. The first version used 188 rows (7,957 tokens). That left those lanes 235 tokens before
  llama-server's context stop, which ends with finish_reason `length`, the same as max_tokens.

`PROMPT_TOKENS` in the script holds every prompt's measured count, with `--thinking` offsets (low +28, medium -2,
xhigh +40). Before sending, the client checks each prompt against the server's context (llama-server `/props` `n_ctx`,
TabbyAPI `max_seq_len`). It skips any prompt whose count plus `max_tokens` plus a 16-token margin does not fit. For
example, `--thinking medium --max-tokens 4096` skips long8k on an 8K lane. A skipped prompt still writes a row, with
`skipped` set to the reason and no measurements, so filter on `skipped` when aggregating. `--no-ctx-check` sends the
prompt anyway, which is useful for testing the context stop itself.

**Every lane gets byte-identical requests:**
- greedy decoding (`temperature` 0, with the eval kit's `top_p`/`top_k`);
- `max_tokens` 512;
- streaming with `stream_options.include_usage`;
- the same `chat_template_kwargs`. The default is `--thinking off` (`enable_thinking: false`, as in `smoke.py`);
  `--thinking medium` sends the eval kit's `reasoning_effort`.

**Prefix caching.** Each request starts with `Run <repeat>.`, so a repeat cannot reuse the previous repeat's prompt
cache on either engine. All lanes still see identical prompts. Use `--no-cache-bust --repeats 1` for a like-for-like
comparison with `runs/smoke.jsonl`. One untimed warm-up request runs first.

**Row fields** (one row per request, appended to `$L/runs/speed.jsonl`):

| Field | Meaning |
|---|---|
| `ttft_s` | time to the first streamed text (content or reasoning) |
| `stream_decode_tps` | client-side decode rate: tokens after the first text chunk over the streaming window. TabbyAPI sends several tokens per chunk when drafts are accepted, so the first chunk's share is estimated from its characters. |
| `e2e_s`, `e2e_tps` | whole-request time and rate |
| `server_decode_tps`, `server_prefill_tps`, `server_timings` | the server's own timings. Both engines emit llama-server-style `timings`. |
| `draft_n`, `draft_n_accepted`, `draft_accept` | drafting statistics, when the server reports them |
| `completion_tokens`, `prompt_tokens`, `cached_tokens`, `tokens_source` | token counts (`usage`, then `timings`, then the chunk count) |
| `stop_kind`, `hit_max_tokens`, `hit_ctx_limit` | why generation ended. `max_tokens`: completion_tokens reached `max_tokens`. `context`: finish_reason `length` with fewer tokens, so the server's context ran out; the row's rates, energy and J/token are not comparable, and the run exits non-zero. Otherwise `stop`, `error`, or the server's finish_reason. When the token count comes only from counting chunks, a short `length` stays `length`. |
| `skipped` | set (with the reason) on rows the context check did not send; such rows carry no measurements |
| `server.lane_start` | the newest `$L/runs/lanes.jsonl` row for the server's port: lane, context, `recurrent_state`, overrides, config. Read only when the base URL is loopback on the desktop. A lane or status that does not match `--lane` prints a warning. |
| `server.n_ctx`, `server.min_p_default` | the server's context (llama-server `n_ctx`, TabbyAPI `max_seq_len`) and llama-server's default `min_p` |
| `max_gap_s`, `tokens_per_chunk` | run-shape checks |
| `output_sha256` | the eval kit's `content_sha256` definition, for comparing greedy outputs across lanes |
| `gpu_during`, `gpu_after` | 1 Hz nvidia-smi samples while the request streams (power, SM/memory clocks, temperature, utilization, memory, OR of clock-event reasons), and a snapshot after it. Collected only when the client runs on the desktop; `--smi auto`. |
| `energy_j_est`, `j_per_token_est` | mean sampled power × `e2e_s`, total and per token |

For paired lane comparisons, run lanes in ABBA order: start A, bench, stop; then B, B, A.

## llama.cpp equal-tuning sweep (`llamacpp_sweep.sh`)

```bash
bash $L/tools/desktop_run.sh cmd sweep bash $L/tools/llamacpp_sweep.sh --quick   # smoke, ~3 min
bash $L/tools/desktop_run.sh cmd sweep bash $L/tools/llamacpp_sweep.sh           # full, ~35-45 min
bash $L/tools/llamacpp_sweep.sh --dry-run                                         # plan only, no GPU, no writes
tail -f $L/logs/run-sweep.log                                                     # progress; summary table at the end
```

The sweep refuses to run outside a `turing-run-*` unit and while another lab unit is active. All calls use
mainline d63fbfe7 with `-ngl 99 -r 3 -t 4`. In order of importance, per model (IQ2_XXS first, then IQ2_S):

1. **offload**: one llama-server start (`-ngl 99 -fa on`, q8_0 KV, 16K) records:
   - `offloaded N/N layers to GPU`;
   - the `system_info` line, buffer sizes and VRAM at ready.

   The server is then stopped cleanly. This checks the plan's "verify full offload" item.
2. **tg** (decode, `-p 0 -n 128 -d 0,8192`): CUDA graphs on/off × {fa 1 + q8_0 KV, fa 1 + f16 KV, fa 0 + f16 KV}.
   Quantized V needs flash attention, so there is no fa 0 + q8_0 pair.
3. **pp** (prompt processing, `-p 2048 -n 0 -b 2048 -ub 256,512,1024,2048 -d 0,8192`): the same three fa/KV pairs,
   graphs on.
4. **threads** (IQ2_XXS only): decode at depth 0 with `-t 1,4,10`. E1 used llama-bench's default thread count:
   its rows record `n_threads` 10, inside the unit's 4-core quota. This stage shows whether that cost decode speed.
5. **cublas**: pp and tg, only if a forced-cuBLAS build exists (see below).

A 75-minute soft budget (`TURING_SWEEP_BUDGET_MIN`) skips whatever is left. An nvidia-smi log samples clocks,
power, temperature, utilization, memory, clock-event (throttle) reasons and pstate at 1 Hz for the whole sweep:
`$L/logs/sweep-<id>-smi.csv`. That covers the plan's "explain a low bar by clocks" check.

**Output.** llama-bench's own rows go to `$L/runs/llamabench_sweep.jsonl`, tagged with `sweep_id`, `sweep_stage`,
`kernel`, `cuda_graphs` and `bin_dir`. Meta rows (`sweep_row` = start / offload / end) go to the same file. The
untagged per-call files are in `$L/runs/sweep/<id>/`.

**Axes that were narrowed or replaced, from reading the mainline source:**
- **MMQ vs cuBLAS is a build choice, not an environment variable.** `GGML_CUDA_FORCE_MMQ` and
  `GGML_CUDA_FORCE_CUBLAS` are CMake options (`ggml/src/ggml-cuda/CMakeLists.txt:139-144`), so exporting them would
  measure nothing. On sm_75 the default build already uses MMQ for every IQ2 type (`mmq.cu:435`,
  `turing_mma_available`), so `kernel=mmq` is the default build. The cuBLAS side needs its own build. To include it
  in a sweep, build `llama-bench` once, in a capped unit like `build_llamacpp.sh`:

  ```bash
  cmake -S $L/src/llama.cpp -B $L/src/llama.cpp/build-sm75-cu134-cublas -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON -DGGML_CUDA_FORCE_CUBLAS=ON
  cmake --build $L/src/llama.cpp/build-sm75-cu134-cublas -j 4 --target llama-bench
  ```

  Run it with `PATH=$L/venv/bin:$PATH` for cmake and ninja. The sweep picks it up automatically (or set
  `TURING_SWEEP_CUBLAS_BIN`). Without it, the sweep logs the skip and records `cublas_build_present: false`.
- **Batch-1 decode cannot see `-ub` or MMQ/cuBLAS.** Mainline routes `ne11 <= 8` to MMVQ before it consults
  `should_use_mmq` (`ggml_cuda_mul_mat`), so those axes are swept on prompt processing only. CUDA graphs are swept
  on decode, the only path they capture.
- **Other runtime knobs exist but are not swept:** `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_GRAPH_OPT`,
  `GGML_CUDA_CUBLAS_COMPUTE_TYPE` (relevant only to the cuBLAS build) and `GGML_CUDA_MMQ_PREC`.
- **A sweep winner can be carried into a lane**, e.g. `TURING_LCPP_EXTRA="-ub 512"` or
  `TURING_LANE_ENV=GGML_CUDA_DISABLE_GRAPHS=1`.

## Checks done (2026-10-30, no GPU used)

- `bash -n` on both scripts. Python compile check of `speed_bench.py`, without writing bytecode. The delete-word
  grep (`grep -F -f` with a pattern file) found nothing. shellcheck is not installed on either machine.
- Both ymls, and a per-start rewrite with a different port and context, validate against upstream TabbyAPI's
  `TabbyConfigModel` on the desktop (CPU only, config piped over stdin): no unknown keys, values as intended.
- `lanes.sh list/status/show` and `llamacpp_sweep.sh --dry-run [--quick]` ran on the desktop with the scripts piped
  over ssh, writing nothing. Every binary, model and sidecar path resolves. The only missing files are the two
  ymls, which are not deployed yet.
- uutils `env -C … -u …` on the desktop passes `python -c CODE --config X` arguments through intact.
- The TabbyAPI boot shim was checked against a stand-in `main.py` with upstream's exact `logger.add` call.
- `speed_bench.py` ran against a stand-in streaming server in llama-server style and TabbyAPI style, and against a
  closed port. Usage, timings, draft fields, reasoning/content split and multi-token chunks were all parsed; errors
  exit non-zero. On the desktop it ran under the system Python 3.14 in `--list` mode, and its nvidia-smi telemetry
  parsed.
- Every jq filter in `lanes.sh` and the sweep was run against sample rows shaped like llama-bench output.

Review fixes (2026-10-30, second round, no GPU used):
- **long8k.** Recalibrated on the laptop CPU with the Qwen3.8 `tokenizer.json` and `chat_template.jinja`. At 188 rows
  the count was 7,957, matching the review. At 175 rows it is 7,417 (thinking off), 7,415 (medium), 7,445 (low) and
  7,457 (xhigh). `PROMPT_TOKENS` holds these counts.
- **`speed_bench.py`** ran against a stand-in server in both styles:
  - 8K llama-server style: long8k sent at 512 tokens; skipped at 1,024 tokens, with a skip row.
  - `--no-ctx-check` against a server that stops at 235 tokens: `stop_kind` `context`, exit 1.
  - TabbyAPI style with multi-token chunks: `n_ctx` comes from `max_seq_len`.
  - `lane_start` is linked from a lanes.jsonl row, and a mismatched `--lane` prints a warning.
  - A closed port exits 1.
  - Unit cases: a `length` counted from chunks stays `length`; an error gives `error`.
- **`lanes.sh`** ran end to end on the laptop CPU against a fake lab in the scratchpad, with stubbed `systemctl` and
  `nvidia-smi` and a stub `desktop_run.sh` that starts a stdlib stand-in server. No systemd unit was started:
  - `-fp32` lanes get `recurrent_state_fp32: true` and start when the log says fp32.
  - An fp16 log on an `-fp32` lane gives `precision-mismatch`.
  - The `TURING_LANE_ENV` override records no expectation.
  - TERM during the health wait exits in 1 s with status `interrupted` and the unit stopped.
  - TERM during a probe that hangs for 120 s exits in under 1 s.
  - HUP, and a closed stdout/stderr pipe (PIPE), also give `interrupted` and a stopped unit.
  - Refusals still work: another unit active, or an unknown lane.
- **On the desktop**, read-only with the script piped over ssh:
  - `show` resolves every binary and GGUF of the new llama.cpp lanes.
  - The generated `exl3-mtp-fp32`, `exl3-nodraft-fp32` and `exl3-mtp` configs validate against upstream TabbyAPI's
    `TabbyConfigModel`: no unknown keys, and `recurrent_state_fp32` is True only on the `-fp32` lanes.
  - Both llama-server builds contain the `--min-p` option.
- **Upstream source** (laptop mirrors): llama-server seeds every request's sampling from the command line
  (`server-schema.cpp`, `params.sampling = params_base.sampling`) with `min_p` 0.05 in both builds. TabbyAPI's
  config load merges `config.yml`, then the environment, then the `--config` file, last wins. TabbyAPI logs
  `Recurrent state storage: fp16|fp32` at load (`backends/exllamav3/model.py:650`).
