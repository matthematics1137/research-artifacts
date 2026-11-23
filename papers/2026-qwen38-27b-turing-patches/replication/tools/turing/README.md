# Turing lab: Qwen3.8-27B on the desktop's RTX 2080 Ti

Matt approved this on 2026-10-30. His other agents run on p8-host and use its CPU, RAM
and disk, but not its GPU. This lab uses the GPU only while yielding everything else to them. The
plan and its go/no-go criteria are in `research/TURING_EXL3_RESEARCH_2026-10-30.md` (sections 6.2-6.5).

## Layout on the desktop (`/opt/p8`, nothing outside it)

Everything lives in this one folder on the desktop's 2 TB `ssd2` SSD (Matt, 2026-10-30): nothing on the
500 GB main drive and nothing split across drives. Every cache (Triton, exllamav3, torch, pip) is pointed
inside the lab, and the environment holds no reference to any path outside it.

| Path | What |
|---|---|
| `bootstrap/` | venv holding only `uv` |
| `python/` | uv-managed CPython 3.12 (the system Python is 3.14) |
| `venv/` | torch 2.10.0+cu128 (its pinned triton) and exllamav3 1.6.5 (sha256 `e1846933…`, 169 sm_75 cubins); upstream TabbyAPI, editable |
| `src/tabbyAPI` | upstream TabbyAPI checkout, separate from the lab's copy |
| `models/qwen38-exl3-2.0` | the laptop's 2.0 bpw EXL3 model, hash-checked file by file after the copy |
| `models/qwen38-bf16` | original Qwen/Qwen3.8-27B BF16 checkpoint @ `1d4bf0f2` (55.6 GB, all 32 files SHA-256-checked against the Hub, 2026-10-30; same revision as the 2026-09-12 Drive backup), for our own quantization and fine-tuning |
| `wheelhouse/` | the verified exllamav3 wheel; the same file is archived on the laptop in `engines/wheelhouse/turing-cp312/` |
| `tools/` | `desktop_guard.py`, `desktop_run.sh` and `smoke.py` from this folder |
| `logs/`, `runs/` | setup, guard and run logs; JSON result rows |

## Not disturbing the machine

- Every run is a systemd user unit, `turing-run-<label>`, with these limits:
  - 4 CPU cores, 20 GiB RAM and no swap;
  - Nice 10 and idle IO priority;
  - `OOMScoreAdjust=800`, so under memory pressure the kernel stops our run first.
- The guard (`turing-guard` unit, `desktop_guard.py`) stops every `turing-run*` unit when any of
  these happens:
  - available RAM stays under 10 GiB for two samples;
  - the memory-stall pressure reading reaches 15;
  - free space on the lab's drive (ssd2) falls under 50 GB;
  - the GPU reaches 87 °C (the card's own target is 84 °C, max operating 89 °C, slowdown 91 °C);
  - another program starts using the GPU.
- These thresholds are environment defaults, adjustable at any time. They are not study parameters.
- The guard never restarts anything. Nothing here deletes files.

## Commands (on the desktop)

```bash
/opt/p8/tools/desktop_run.sh guard
/opt/p8/tools/desktop_run.sh smoke base-32k -cs 32768 -cq 8,8 -rcs 2 --long 30000
/opt/p8/tools/desktop_run.sh smoke mtp-16k  -cs 16384 -cq 8,8 -rcs 2 -mtp -ndt 4
```

Results land in `runs/smoke.jsonl` inside the lab, one JSON row per prompt.

## Self-reliance

`tools/mirror_sources.sh` keeps full-history mirrors of the code this lab depends on in
`engines/src-mirror/` on the laptop, with a `MANIFEST.tsv` listing each mirror's head commit. That
code is exllamav3 plus the community Turing forks, TabbyAPI, flash-linear-attention, Triton,
tokenizers, safetensors, QTIP, exllamav2, llama.cpp, ik_llama.cpp and PyTorch (main plus a full v2.10.0
checkout with submodules). Rerunning the script refreshes the mirrors and never deletes anything.
