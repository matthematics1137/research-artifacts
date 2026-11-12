#!/usr/bin/env python3
"""Paper 8 W12b limiter study: time the int8 sq GEMV's per-weight loop alone on the 2080 Ti (tools/turing/w12/sqmb.cu).

Builds the microbenchmark against a patched exllamav3 source tree (its exllamav3_ext directory on the include path),
checks that variants 0-3 and 8-15 give bit-identical per-slice partials, then times every variant on the model's large K = 2
shapes. Each timing replays a CUDA graph of 20 launches. Grid and slice height follow the sq kernel's own rule:
resident blocks per SM x 68, and the smallest slice with units <= grid (at least 16 rows). A slice-height sweep on
the MLP gate shape follows. Then the M-row part (MTP verify): upstream's dp4a narrow unit against an integer
tensor-core (IMMA m8n8k16) form at M = 1-8 rows, bit-identical check and timing. Rows go to --out (JSON lines).
  python3 w12_sqmb.py --src SRC/exllamav3/exllamav3_ext --out ROWS.jsonl
"""
import argparse
import json
import os
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {0: "upstream narrow", 1: "upstream staged", 2: "narrow + PRMT", 3: "staged + PRMT", 4: "no loads",
         5: "no dp4a", 6: "no hash multiply", 7: "loads only", 8: "running pointers", 9: "pointers + PRMT",
         10: "prefetch 1 row", 11: "prefetch 2 rows", 12: "prefetch 2 + PRMT", 13: "1 word/lane + shfl",
         14: "shfl + prefetch 1", 15: "shfl + prefetch 2", 16: "stepped row ptr"}
EXACT = {1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16}   # variants that must reproduce V0's partials bit for bit
REPEAT = [0, 4, 7, 8, 9, 10, 11, 12, 2]     # sqmb2: rotated, repeated timings (--variants overrides; W12e: 0,7,8,13,14,15)
SHAPES = [("mlp_gate", 5120, 17408), ("mlp_down", 17408, 5120), ("gdn_qkv", 5120, 10240), ("attn_q", 5120, 12288)]
PER_GRAPH = 20
SMS = 68


def build(src):
    return load(name="w12_sqmb", sources=[os.path.join(HERE, "w12", "sqmb.cu")], extra_include_paths=[src],
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_75,code=sm_75", "-lineinfo"],
                verbose=False)


def rows_rule(k, n, grid):
    """The sq kernel's slice height for a grid (exl3_gemv_int8_kernel.cuh, rows_per_host == 0)."""
    rows_total, nb = k // 16, n // 256
    r = -(-rows_total * nb // grid)
    rows = (max(r, min(2 * r, 32)) + 7) & ~7
    return min(max(rows, 16), (rows_total + 7) & ~7)


def timed_rows(ext, v, args, iters):
    return timed(ext, v, args, iters, launch=ext.launch_rows)


def timed(ext, v, args, iters, launch=None):
    launch = launch or ext.launch
    call = lambda: launch(v, *args)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(PER_GRAPH):
            call()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (iters * PER_GRAPH)


def repeat(ext, a, dev, gen, emit):
    """sqmb2/W12e: per shape, every variant in the list timed once per round, in an order rotated each round; medians."""
    import statistics as st
    REPEAT = [int(x) for x in a.variants.split(",")] if a.variants else globals()["REPEAT"]
    assert REPEAT[0] == 0, "V0 must come first (it is the reference)"
    for name, k, n in SHAPES:
        B = torch.randint(-32768, 32767, (k // 16, n // 16, 32), dtype=torch.int16, device=dev, generator=gen)
        splats = torch.randint(-2**31, 2**31 - 1, (k,), dtype=torch.int32, device=dev, generator=gen)
        setup, ref = {}, None
        for v in REPEAT:
            grid = ext.info(v, 160)[1] * SMS
            rows = rows_rule(k, n, grid)
            partials = torch.zeros((-(-(k // 16) // rows), n), dtype=torch.int32, device=dev)
            args = (B, splats, partials, k, n, rows, grid)
            ext.launch(v, *args)
            torch.cuda.synchronize()
            same = None
            if v == 0:
                ref = partials.clone()
            elif v in EXACT:
                same = bool(torch.equal(partials, ref))
            setup[v] = (args, grid, rows, same)
        times = {v: [] for v in REPEAT}
        for r in range(a.rounds):
            order = REPEAT[r % len(REPEAT):] + REPEAT[:r % len(REPEAT)]
            for v in order:
                us = timed(ext, v, setup[v][0], a.iters)
                times[v].append(us)
                emit({"kind": "rep", "shape": name, "variant": v, "name": NAMES[v], "round": r + 1, "us": round(us, 2)})
        base = st.median(times[0])
        for v in REPEAT:
            med = st.median(times[v])
            emit({"kind": "rep_summary", "shape": name, "k": k, "n": n, "variant": v, "name": NAMES[v],
                  "grid": setup[v][1], "rows_per": setup[v][2], "rounds": a.rounds, "median_us": round(med, 2),
                  "min_us": round(min(times[v]), 2), "max_us": round(max(times[v]), 2),
                  "vs_v0": round(med / base, 4), "bit_identical_to_v0": setup[v][3]})
            print(f"{name:9s} V{v:<2d} {NAMES[v]:18s} median {med:7.2f} us [{min(times[v]):7.2f}, {max(times[v]):7.2f}] "
                  f"{med / base:5.3f}x V0  identical={setup[v][3]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--part", choices=("all", "repeat"), default="all",
                    help="all: the sqmb1 study; repeat: sqmb2, rotated repeated timings with medians")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--variants", default=None, help="repeat part: comma-separated variants, V0 first (default: sqmb2's list)")
    a = ap.parse_args()
    ext = build(a.src)
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1031)
    stamp = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": torch.cuda.get_device_name(dev), "src": a.src}
    out = open(a.out, "a")

    def emit(row):
        out.write(json.dumps({**row, **stamp}) + "\n")
        out.flush()

    if a.part == "repeat":
        repeat(ext, a, dev, gen, emit)
        out.close()
        return

    for v in NAMES:
        regs, blocks = ext.info(v, 160)
        emit({"kind": "info", "variant": v, "name": NAMES[v], "registers": regs, "blocks_per_sm_at_160_rows": blocks})
        print(f"V{v} {NAMES[v]:18s} {regs} registers, {blocks} blocks per SM", flush=True)

    for name, k, n in SHAPES:
        B = torch.randint(-32768, 32767, (k // 16, n // 16, 32), dtype=torch.int16, device=dev, generator=gen)
        splats = torch.randint(-2**31, 2**31 - 1, (k,), dtype=torch.int32, device=dev, generator=gen)
        mweights = k * n / 1e6
        floor_us = k * n * 2 / 8 / 616e3          # bytes at K = 2 over 616 GB/s, in us
        ref = None
        for v in NAMES:
            blocks = ext.info(v, 160)[1]
            grid = blocks * SMS
            rows = rows_rule(k, n, grid)
            ksplit = -(-(k // 16) // rows)
            partials = torch.zeros((ksplit, n), dtype=torch.int32, device=dev)
            args = (B, splats, partials, k, n, rows, grid)
            ext.launch(v, *args)
            torch.cuda.synchronize()
            same = None
            if v == 0 or v in EXACT:
                if ref is None:
                    ref = partials.clone()
                same = bool(torch.equal(partials, ref))
            us = timed(ext, v, args, a.iters)
            row = {"kind": "time", "shape": name, "k": k, "n": n, "variant": v, "name": NAMES[v], "grid": grid,
                   "rows_per": rows, "us": round(us, 2), "us_per_mweight": round(us / mweights, 4),
                   "gbs_at_k2": round(k * n * 2 / 8 / us / 1e3, 1), "floor_us": round(floor_us, 2),
                   "bit_identical_to_v0": same}
            emit(row)
            print(f"{name:9s} V{v} {NAMES[v]:18s} grid {grid:4d} rows {rows:4d} {us:8.2f} us "
                  f"{row['us_per_mweight']:.3f} us/M {row['gbs_at_k2']:6.1f} GB/s  identical={same}", flush=True)

    # Several activation rows (MTP verify): upstream's dp4a narrow unit (v 0) against the IMMA form (v 1)
    for name, k, n in SHAPES[:3]:
        B = torch.randint(-32768, 32767, (k // 16, n // 16, 32), dtype=torch.int16, device=dev, generator=gen)
        for m in (1, 2, 3, 4, 6, 8):
            splats = torch.randint(-2**31, 2**31 - 1, (m, k), dtype=torch.int32, device=dev, generator=gen)
            ref = None
            for v in (0, 1):
                regs, blocks = ext.rows_info(v, m, 32)
                grid = max(blocks, 1) * SMS
                rows = rows_rule(k, n, grid)
                partials = torch.zeros((-(-(k // 16) // rows), m, n), dtype=torch.int32, device=dev)
                args = (m, B, splats, partials, k, n, rows, grid)
                ext.launch_rows(v, *args)
                torch.cuda.synchronize()
                if ref is None:
                    ref = partials.clone()
                same = bool(torch.equal(partials, ref))
                call_us = timed_rows(ext, v, args, a.iters)
                emit({"kind": "rows", "shape": name, "k": k, "n": n, "m": m, "variant": v,
                      "name": ("upstream dp4a", "IMMA m8n8k16")[v], "registers": regs, "blocks_per_sm": blocks,
                      "grid": grid, "rows_per": rows, "us": round(call_us, 2), "bit_identical_to_dp4a": same})
                print(f"rows {name:9s} M={m} {('dp4a', 'IMMA')[v]:5s} regs {regs:3d} blocks {blocks} rows {rows:4d} "
                      f"{call_us:8.2f} us identical={same}", flush=True)

    name, k, n = SHAPES[0]
    B = torch.randint(-32768, 32767, (k // 16, n // 16, 32), dtype=torch.int16, device=dev, generator=gen)
    splats = torch.randint(-2**31, 2**31 - 1, (k,), dtype=torch.int32, device=dev, generator=gen)
    for v in (0, 1, 2, 3, 4, 7, 8, 9):
        grid = ext.info(v, 160)[1] * SMS
        for rows in (16, 32, 64, 96, 160, 320):
            partials = torch.zeros((-(-(k // 16) // rows), n), dtype=torch.int32, device=dev)
            us = timed(ext, v, (B, splats, partials, k, n, rows, grid), a.iters)
            emit({"kind": "sweep", "shape": name, "variant": v, "name": NAMES[v], "grid": grid, "rows_per": rows,
                  "us": round(us, 2), "us_per_mweight": round(us / (k * n / 1e6), 4)})
            print(f"sweep {name} V{v} rows {rows:4d} {us:8.2f} us", flush=True)
    out.close()


if __name__ == "__main__":
    main()
