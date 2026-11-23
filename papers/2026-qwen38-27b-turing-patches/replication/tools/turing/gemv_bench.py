#!/usr/bin/env python3
"""Paper 8 W12 scouting: time the EXL3 int8 GEMV alone on the 2080 Ti, over shapes and bit widths.

Synthetic mul1 tensors (random trellis words, random +-1 input and output flips) of the model's shapes, plus tiny
and fused ones, one activation row, through ext.exl3_gemm (the dispatch the model uses for mul1 tensors: the int8
sq kernel where it applies). Each timing replays a CUDA graph of 20 back-to-back calls, so launch cost from
Python is excluded and the per-call figure includes the in-graph gap a decode step also pays. For each K it fits
t = a + b * weights (least squares over the shapes) and prints the fixed part a and the per-million-weight part b.
The fused shapes (gate+up as one 5120 x 34816, qkv+z as 5120 x 16384, q+k+v as 5120 x 14336) show what one launch
per bundle would cost against the separate launches (W12a).
  python3 gemv_bench.py --out ROWS.jsonl [--iters 30] [--bits 1,2,3,4] [--m 1,3,5,7] [--shapes mlp_gate,gdn_qkv]
"""
import argparse
import json
import time

import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.version import __version__

SHAPES = [  # (name, k, n)
    ("tiny", 128, 256), ("small", 1024, 1024), ("attn_kv", 5120, 1024), ("gdn_z", 5120, 6144),
    ("gdn_out", 6144, 5120), ("gdn_qkv", 5120, 10240), ("attn_q", 5120, 12288), ("mlp_gate", 5120, 17408),
    ("mlp_down", 17408, 5120),
    ("fused_qkv_z", 5120, 16384), ("fused_q_k_v", 5120, 14336), ("fused_gate_up", 5120, 34816),
]
PER_GRAPH = 20


def tensors(k, n, K, dev, gen, m=1):
    B = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev, generator=gen)
    suh = (torch.randint(0, 2, (k,), device=dev, generator=gen) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), device=dev, generator=gen) * 2 - 1).half()
    A = (torch.randn((m, k), device=dev, generator=gen) * 0.1).half()
    return A, B, suh, svh, torch.empty_like(A), torch.empty((m, n), dtype=torch.half, device=dev)


def time_shape(k, n, K, dev, gen, iters, m=1):
    A, B, suh, svh, A_had, C = tensors(k, n, K, dev, gen, m)
    call = lambda: ext.exl3_gemm(A, B, C, suh, A_had, svh, -1, False, True, 0)
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


def fit(points):
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    b = sum((p[0] - mx) * (p[1] - my) for p in points) / sum((p[0] - mx) ** 2 for p in points)
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--bits", default="1,2,3,4")
    ap.add_argument("--m", default="1", help="activation rows per call, e.g. 1,3,5,7 (MTP verify); the engine's own "
                    "dispatch picks the kernel (int8 up to EXL3_INT8_SQ_MAX_M rows, else fp16)")
    ap.add_argument("--shapes", default="", help="comma list of shape names (default all)")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1030)
    env = {k: v for k, v in __import__("os").environ.items() if k.startswith("EXL3_")}
    shapes = [s for s in SHAPES if not a.shapes or s[0] in a.shapes.split(",")]
    with open(a.out, "a") as out:
        for m in [int(x) for x in a.m.split(",")]:
            for K in [int(x) for x in a.bits.split(",")]:
                pts = []
                for name, k, n in shapes:
                    us = time_shape(k, n, K, dev, gen, a.iters, m)
                    row = {"K": K, "m": m, "shape": name, "k": k, "n": n, "us": round(us, 2), "mweights": k * n / 1e6,
                           "gbs": round(k * n * K / 8 / us / 1e3, 1), "env": env, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                           "device": torch.cuda.get_device_name(dev), "exllamav3": __version__}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    if not name.startswith("fused"):
                        pts.append((k * n / 1e6, us))
                    print(f"m={m} K={K} {name:14s} {k:>6} x {n:<6} {us:8.2f} us  {row['gbs']:6.1f} GB/s", flush=True)
                if len(pts) >= 2:
                    a0, b0 = fit(pts)
                    print(f"m={m} K={K} fit (unfused shapes): {a0:.2f} us fixed + {b0:.3f} us per million weights", flush=True)
                    out.write(json.dumps({"K": K, "m": m, "fit_fixed_us": round(a0, 2), "fit_us_per_mweight": round(b0, 4)}) + "\n")


if __name__ == "__main__":
    main()
