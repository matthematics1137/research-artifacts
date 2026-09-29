# Full replication: self-drafted 2-bit Qwen3.8-27B on an 80 W, 12 GB laptop GPU

This is the expensive path. The analysis-only check in `README.md` runs no
model. A full rerun downloads about 56 GB of weights (about 29 GB for the daily
driver and the hardening block alone), builds or stages two engines with the lab
patches, and runs GPU-hours of inference: the hardening block alone took 8.0 h
(H38a), 1.7 h (H38b), 52 min (H38c), and about 45 min each for H38d and H38e.
Use a machine you may load heavily and keep unrelated GPU work closed.

Exact historical numbers are not expected. The original laptop was a loaded
machine (foreign agents; one foreign Python job held about four CPU cores),
temperature-1 runs are stochastic, and temperature-0 decoding on exllamav3 1.4.7
is itself not bit-reproducible from run to run (E49, E50). Treat the historical
values below as descriptive reference points. Observed rerun differences
included three items on GSM8K-100 at temperature 1 (E26) and about 8% in wall
time per item between identically configured 100-item runs (E50). These few
repeats do not establish general noise floors or calibrated acceptance thresholds.

## Historical system

- NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB, power limit 80 W, compute
  capability 8.9; the desktop session held about 485 MiB (E23).
- Intel Core i9-14900HX, approximately 62 GiB of reported system RAM.
- Ubuntu 24.04.3 LTS, Linux 7.0.0-29-generic; NVIDIA driver 580.167.08 on this
  laptop's 2026-08-31 host readback (the campaign itself did not record it).
- GPU telemetry by `nvidia-smi` every 1–2 s; energy is GPU-board energy only.

`environment/hardware.json` and `environment/software.json` hold the historical
machine-readable metadata. The former retains the campaign's `ram_gb: 62`
field; this is a legacy unit label for the approximately 62 GiB reported here.

## Model files (linked, not redistributed)

`environment/model_artifacts.json` lists every file with repository, branch or
revision, filename, byte count, SHA-256, and license. The files used by the
reported experiments:

| Role | Repository | Branch / revision | File | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| EXL3 2.00 bpw (daily driver) | `turboderp/Qwen3.8-27B-exl3` | `SC_2.00bpw_H3` / `d7e02d63dddd4e047a8d397acb22aeb2b0bedcec` | `model-00001-of-00002.safetensors` | 8,573,967,630 | `cf8966ebc88599d904d4b1ac65d567a15b00b7f9c97ef54c8d77eb534ff8eb93` |
| | | | `model-00002-of-00002.safetensors` | 1,622,158,601 | `2a72afda6bf041732018ee34aca772a4a268ef90905da0c710397e68a9e1a746` |
| EXL3 2.20 bpw (E23) | `turboderp/Qwen3.8-27B-exl3` | `SC_2.20bpw_H3_V3` at `25019f1663e0e0bcfc363609f1dbabd5604b04e4` | `model-00001-of-00002.safetensors` | 8,500,866,052 | `522f94e79dfc119eeb02cd96978513eb24f86682183eb07084c956a4299e28c4` |
| | | | `model-00002-of-00002.safetensors` | 1,780,094,352 | `b89cafe9c614119b2faedbfd5e635a75af09928b058f3734ed14d6d0044b5f33` |
| GGUF `UD-IQ2_S` | `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `Qwen3.8-27B-UD-IQ2_S.gguf` | 8,371,970,048 | `7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe` |
| GGUF `UD-IQ2_XXS` | same | same | `Qwen3.8-27B-UD-IQ2_XXS.gguf` | 7,266,070,528 | `e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d` |
| GGUF `UD-Q4_K_XL` (4-bit reference) | same | same | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,559,178,144 | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| MTP sidecar | same | same | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1,369,590,656 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
| DFlash2 drafter | `z-lab/Qwen3.8-27B-DFlash2-GGUF` | `2d9571f8ce46e151f61c6499c99dee6079e1d610` | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1,143,006,816 | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |

The Unsloth repository was re-cut on 2026-08-20, so the SHA-256 values, not the
names, identify the evidence; the listed Unsloth and EXL3 revisions are those
the lab's Paper 1 artifact recorded for the same SHA-256. Where no revision was
recorded, download the named file and stop if its SHA-256 differs.

```bash
python3 -m venv hf-env
hf-env/bin/pip install huggingface_hub
hf-env/bin/hf download turboderp/Qwen3.8-27B-exl3 --revision d7e02d63dddd4e047a8d397acb22aeb2b0bedcec \
  --local-dir models/qwen38-exl3-2.0
hf-env/bin/hf download unsloth/Qwen3.8-27B-GGUF --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  Qwen3.8-27B-UD-IQ2_S.gguf Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  --local-dir models/qwen38
sha256sum models/qwen38-exl3-2.0/*.safetensors models/qwen38/*.gguf models/qwen38/MTP/*.gguf
```

The **lab-made quantizations** of E32 (own-trace recalibration), E41 (variant A
bit recipe), and E42 (off-policy calibration) are not included and are not on
the Hub. They were converted on this laptop with exllamav3 1.4.7 from the
Qwen3.8-27B weights (Apache-2.0 per the upstream records), deleted from the lab
disk on 2026-09-08, and archived off-machine; their shard SHA-256 values are in
`environment/model_artifacts.json` and `data/lab_quant_hashes.json`. The license
would permit redistribution, but this release does not carry them. Recreating
them means re-running the conversions (about 3 h each on this card) with the
bit recipes in `replication/tools/` (`recipe_stock20.yaml`, the stock
per-tensor widths; `recipe_variantA.yaml`, E41) and calibration sets built as E32
and E42 describe (`gen_traces.py`, `pack_cal.py`, `pack_offpolicy.py`,
`make_recipe.py`); the calibration text itself (the model's own reasoning traces
for E32) is not redistributed. `error_anatomy.py` reproduces E33 from the
converter logs, which are not included either.

## Engines and patches

**EXL3 lane.** TabbyAPI at checkout `4a4f9f44820303593844f092d424bb7506008733`
(the lab's pinned checkout) with exllamav3 1.4.7
(`exllamav3-1.4.7+cu128.torch2.10.0`, CPython 3.12) staged outside the pinned
environment and loaded through `PYTHONPATH`, over torch 2.10.0+cu128. Apply the
lab patches to an unpacked copy of that exllamav3 package:

- `replication/patches/exl3_all_lab_patches.patch` — the three patches' hunks
  in `exllamav3/generator/generator.py` (measurement, sampled-draft verifier,
  checkpoint intervals); off unless `EXL3_SPEC_MEASURE`, `EXL3_SPEC_SAMPLED`,
  `EXL3_CKPT_PP`, or `EXL3_CKPT_GEN` is set.
- The `exllamav3/architecture/qwen3_5_mtp.py` hunk of
  `replication/patches/exl3_spec_sampled.patch` (a superset of the
  measurement patch's hunk for that file); the combined file above does not
  contain it, so E35/E36 also need this hunk.
- `replication/patches/exl3_ckpt_interval.patch` is a historical, incomplete
  excerpt, **not an applicable patch**. Its checkpoint changes (E43 and the
  daily driver's `EXL3_CKPT_PP=1024`) are present in the combined generator
  patch above. Do not apply the excerpt or stack the overlapping generator
  sections of the measurement/sampled patches on the combined patch.
- `replication/patches/withdrawn/exl3_cholesky_cpu_fallback.patch` is history,
  not a fix: the failure it worked around was the library-path fault below.

The patch headers name the lab's scratch copy (`/path/to/scratch/eff/exl3_147/…`),
not a destination on your machine. Run the following from an **unmodified
unpacked exllamav3 1.4.7 package**, with `P7_PATCHES` set to the absolute path
of this artifact's `replication/patches` directory. Both files are required
for E35/E36; the generator patch also contains the checkpoint overrides.
The checks fail if the input is already patched or belongs to another version.

```bash
P7_PATCHES=/absolute/path/to/artifact/replication/patches
printf '%s\n' \
  'd9679f40791c091382e00e5278d815a19fa7856b9788c376b2ae7093d1cf8231  exllamav3/generator/generator.py' \
  '91a71f29dd7c80336527a7cac6aa566e7b7b97ce9e38f0afe3469eeeff51d829  exllamav3/architecture/qwen3_5_mtp.py' \
  | sha256sum --check --strict &&
patch --batch --fuzz=0 exllamav3/generator/generator.py \
  < "$P7_PATCHES/exl3_all_lab_patches.patch" &&
sed -n '\|^--- .*exllamav3/architecture/qwen3_5_mtp.py|,$p' \
  "$P7_PATCHES/exl3_spec_sampled.patch" \
  | patch --batch --fuzz=0 exllamav3/architecture/qwen3_5_mtp.py &&
printf '%s\n' \
  'f0d7bee9e713b5b755e199babdc8fe8fa4e6c0ad116d072cef679ef8c353148e  exllamav3/generator/generator.py' \
  '8f619e54c0423c9174ed5c75869db49d2334c181e2e0362a789f64b4fa6000a2  exllamav3/architecture/qwen3_5_mtp.py' \
  | sha256sum --check --strict
```

On 2026-09-28 these two patch sections were checked against the upstream
v1.4.7 Python files: exact input hashes, zero-fuzz dry runs, exact hunk/context
application, Python syntax, and the output hashes above passed. This verifies
source reconstruction only, not installation, native execution or a new model
run. Experimental measurement and sampled-draft switches remain off unless
explicitly enabled by the corresponding experiment.

**Library environment.** Launch every torch/exllamav3/TabbyAPI process with
`LD_LIBRARY_PATH` unset (`replication/serving/launch.sh` does `env -u
LD_LIBRARY_PATH`). The historical E7–E49 servers inherited a system CUDA 12.1
path and loaded its cuBLASLt and NVRTC beside torch's bundled 12.8 libraries
(E50); record `/proc/<pid>/maps` of the server, as
`data/rows/EFF-h38*_cuda_libs.txt` does, to show which set you ran.

**GGUF lane.** llama.cpp tag `v0.4.0` (commit
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`), CUDA 12.6, `sm_89`, with the
2026-08-26 build `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3` as a control. The
stripped-sidecar runs (E12, E18, E20, E22, E38, E48) use v0.4.0 with
`replication/patches/llamacpp-v0.4.0-qwen35-mtp-shared-embd.patch`
(`git apply` from the checkout root; in this copy one hunk starts two context
lines later and one comment line is reworded, which changes no code and still
applies to v0.4.0); it builds on the ggml-org draft pull
request #28243 (`--mtp-shared-embd`) and adds the Qwen3.5-family MTP rules
(campaign infrastructure notes). The stripped sidecar (15 tensors, 0.25 GiB) is
made by `replication/tools/strip_mtp.py`, which drops the `token_embd`,
`output`, and `output_norm` tensors that duplicate the target's (set its
`gguf-py` path placeholder first). Build as usual:

```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git checkout 5266f24da75dc449bd56cbed7addb9c8e4a6a73e
git apply ../replication/patches/llamacpp-v0.4.0-qwen35-mtp-shared-embd.patch   # patched lane only
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Server flags of the GGUF timing runs (E4a): `-ngl 99 -np 1 -fa on -ctk q8_0
-ctv q8_0 -ctxcp 4 -cms 1024 -b 512 -ub 256`. Always set `-np 1` (auto slots
try to allocate a multi-GiB recurrent cache) and never mix K/V cache types
(`-ctk q8_0 -ctv q4_0` runs attention off the fused kernels, E5).

## The daily driver and the shared server

`replication/serving/tabby-daily.yml`: EXL3 SC_2.00bpw_H3, `draft_mode: mtp`,
dynamic drafting, `draft_num_tokens: 6`, `max_batch_size: 1`, 32K context, Q8
cache, `chunk_size: 1024`, 16 GB recurrent stash; launched by
`replication/serving/launch.sh daily`, which sets `PYTHONPATH` to the patched
exllamav3 copy and `EXL3_CKPT_PP=1024` and unsets `LD_LIBRARY_PATH`. Replace the
`/path/to/...` placeholders first. `replication/serving/tabby-2agents.yml` is the
two-slot 24K shared configuration (E39); `logit_bias_marker_penalty_-0.5.json`
is the optional per-request marker penalty (E45).

## Evaluation sets and harness

`replication/evals/run_eval.py` is the lab's harness at the evidence freeze
(standard library only; `--selftest` runs offline). The small suites come from
`replication/evals/prepare_datasets.py`; the hardening sets GSM8K-500 and
MMLU-500 are built by `replication/evals/prepare_hardening.py` as supersets of
GSM8K-100 and MMLU-200 from the public upstream sets (GSM8K test set, MIT;
`cais/mmlu`, MIT). Both scripts download from the upstream sources; set their
cache directory placeholder (`/path/to/scratch/prep_cache`) to a writable path.
Compare the resulting item IDs and their order with
`data/dataset_item_order.json` before running: the first-100 and first-200
comparisons use dataset order.

Harness settings for the campaign suites: temperature 1.0, top-p 0.95, top-k 20,
`--reasoning-effort medium`, `--max-tokens 4096`, one run per item. HumanEval+
executes model-generated code; run it only in a network-disabled, secret-free
container or VM.

## Hardening block (H38, E47–E52)

Serve the daily driver (drafted) or `replication/tools/tabby_h38_nodraft.yml`
(the same configuration without its `draft_model` block) on port 18120, one
process at a time, then:

```bash
EVAL_RESULTS_DIR=$PWD/new-results python3 replication/evals/run_eval.py \
  --suite gsm8k_500 --label EFF-h38a_mtp --base-url http://127.0.0.1:18120 --temperature 0
EVAL_RESULTS_DIR=$PWD/new-results python3 replication/evals/run_eval.py \
  --suite mmlu_500 --label EFF-h38a_mtp --base-url http://127.0.0.1:18120 --temperature 0
# repeat both with --label EFF-h38a_nodraft against tabby_h38_nodraft.yml
```

H38b serves `UD-Q4_K_XL` on the patched llama.cpp with 30 of 64 layers on the
GPU and the stripped MTP sidecar (`-m Qwen3.8-27B-UD-Q4_K_XL.gguf -md
<stripped sidecar> --spec-type draft-mtp --spec-draft-n-max 3 -ngld 99 -ctkd q8_0
-ctvd q8_0 -ngl 30 -np 1 -c 8192 -ctk q8_0 -ctv q8_0 -fa on --jinja -ctxcp 4 -cms
1024 -b 512 -ub 256 -t 20`) and runs `--suite gsm8k_500 --limit 200 --temperature
0`. H38c, H38d, and H38e are 100-item re-runs (`--limit 100`) of the same
configurations: a drafted and an undrafted re-run (H38c), two undrafted runs with
`LD_LIBRARY_PATH` unset plus one drafted (H38d), and two undrafted runs on the
exllamav3 development branch at `08849e3` (H38e), and two undrafted runs on the
unmodified exllamav3 1.5.0 release wheel served through `PYTHONPATH` (H38f). The lab's drivers
(`replication/tools/hardening*.sh`) show the exact cells; they depend on the lab's
scratch orchestration (`cells.sh`, `tabby.sh`, an availability gate and a yield
watcher) and need their placeholders replaced before use. Compare two labels
with `replication/tools/hardening_analyze.py <suite> <A> <B> [--limit N]` and the
divergence anatomy with `replication/tools/identity_anatomy.py` and
`replication/tools/divergence_probe.py` (after pointing their `RES`/`KIT`
constants at your results).

Historical reference values (all temperature 0; see Tables A6–A9):

| Comparison | n | Accuracy | Byte-identical |
|---|---:|---|---|
| E47 GSM8K-500, undrafted vs drafted | 500 | 97.2 / 97.6% (p = 0.50) | 318 |
| E47 MMLU-500, undrafted vs drafted | 500 | 86.2 / 86.8% (p = 0.61) | 105 |
| E48 GSM8K, Q4_K_XL vs 2.00 bpw drafted | 200 | 98.0 / 99.0% (p = 0.63) | — |
| E49 drafted vs drafted re-run (mixed libraries) | 100 | 99 / 99 | 91 |
| E49 undrafted vs undrafted re-run (mixed) | 100 | 98 / 98 | 50 |
| E50 undrafted clean vs clean | 100 | 98 / 99 | 64 |
| E51 exllamav3 dev vs dev | 100 | 97 / 97 | 100 |
| E52 exllamav3 1.5.0 vs 1.5.0 | 100 | 98 / 98 | 100 |

Accuracy should fall inside the Wilson intervals of the paper; byte identity on
exllamav3 1.4.7 will vary from run to run (from 1.5.0 it is reproducible, E52) and should not be used as a pass/fail
criterion without a same-configuration control.

## Throughput, energy, memory, prefill, and levers

The measurement tools are in `replication/tools/`: `sustained_tpj.py` (E25
sustained tokens per joule, E29 concurrency), `memsnap.py` (E28 instrumented
loads), `prefill_profile.py` (E37), `edit_probe.py` (E38, E43),
`agents_sim.py` (E39), `traces_compare.py` (E40), `spec_meas_analyze.py`
(E35/E36 with the measurement patch), `deer_proxy.py` (E24, E26, E31), and
`hash_audit.py`. Each file's header documents its arguments. Historical
reference values include: 29 tok/s undrafted and 48.7 / 50.4 / 35.7 tok/s
drafted on code / math / prose (E7); 0.559 and 0.577 tokens per joule gross
over 30.4 and 15.1 sustained minutes (E25); 10,226 MiB resident at load with
TabbyAPI's default four slots and 8,050 MiB with `max_batch_size: 1` (E27,
E28); 583 tok/s bare-API prefill at chunk 1,024 with 73.8% of GPU time in fp16
GEMM (E37). Throughput depends on host load, clocks under the power cap, and
the CUDA library set, so treat differences within about 10% as unresolved.

## Analysis and interpretation

Run `python3 check_claims.py` on this bundle to verify the historical evidence.
New runs can be compared with the same estimands by pointing
`hardening_analyze.py` and the tools above at new result directories. The paper's
claims are bounded to this model, this card, the 80 W cap, and the two engines;
a rerun on other hardware tests portability, not the historical claims.
