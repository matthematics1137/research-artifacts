#!/usr/bin/env python3
"""P9 phase 1, M1 (research/prereg/2026-11-05-p9-phase1.md): one-row EXL3 matrix calls through ext.exl3_gemm, the
dispatch the model uses for mul1 tensors, unprofiled, with each weight tensor rotated through copies past three L2
sizes (p9_common.py). One process runs one round of one arm (a build on PYTHONPATH plus its EXL3_* environment); the
driver (p9_run.sh) loops rounds and arms, samples clocks, and keeps stderr, where EXL3_INT8_SQ_FIT_LOG=1 prints the
host's slice height, ksplit and grid for every shape where W9's host rule is on.
  python3 p9_bench.py --out ROWS.jsonl --arm NAME --round R [--sets model,tiny,grid,fused] [--smoke]
  python3 p9_bench.py --out ROWS.jsonl --arm NAME --part dispatch     the kernels one call launches, per case
Synthetic tensors as in gemv_bench.py: random trellis words, random +-1 input and output flips, one activation row.
"""
import argparse
import json
import os
import time

import torch

import p9_common as pc
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.version import __version__
import exllamav3


def setup(k, n, K, dev, gen, ncopies):
    A = (torch.randn((1, k), device=dev, generator=gen) * 0.1).half()
    Bs = [pc.trellis(k, n, K, dev, gen) for _ in range(ncopies)]
    suh, svh = pc.flips(k, dev, gen), pc.flips(n, dev, gen)
    return A, Bs, suh, svh, torch.empty_like(A), torch.empty((1, n), dtype=torch.half, device=dev)


def calls_for(A, Bs, suh, svh, A_had, C):
    return [lambda B=B: ext.exl3_gemm(A, B, C, suh, A_had, svh, -1, False, True, 0) for B in Bs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--sets", default="model,tiny,grid,fused")
    ap.add_argument("--part", choices=("time", "dispatch"), default="time")
    ap.add_argument("--smoke", action="store_true", help="two model shapes at K = 1 and 2; rows tagged smoke")
    ap.add_argument("--clocks", default="", help="the clock sampler's CSV, for the temperature stop")
    ap.add_argument("--max-temp", type=float, default=0.0)
    ap.add_argument("--copies", type=int, default=0, help="phase 1b: weight copies per case (1: replays read L2); "
                    "0 rotates past three L2 sizes as in M1")
    ap.add_argument("--only", default="", help="phase 1b: comma list of shape:K cases to run (default all)")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1105 + a.round)
    stamp = {"arm": a.arm, "round": a.round, "exllamav3": __version__, "overlay": os.path.dirname(exllamav3.__file__),
             "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, "smoke": a.smoke,
             **pc.device_stamp(dev)}
    todo = pc.cases(a.sets.split(","))
    if a.only:
        want = {tuple(x.split(":")) for x in a.only.split(",")}
        todo = [c for c in todo if (c[1], str(c[4])) in want]
    if a.smoke:
        todo = [c for c in todo if c[1] in ("attn_kv", "mlp_gate") and c[4] in (1, 2)]
    with open(a.out, "a") as out:
        def emit(row):
            out.write(json.dumps({**row, **stamp}) + "\n")
            out.flush()

        if a.part == "dispatch":
            from torch.profiler import profile, ProfilerActivity
            for s, name, k, n, K in todo:
                A, Bs, suh, svh, A_had, C = setup(k, n, K, dev, gen, 1)
                f = calls_for(A, Bs, suh, svh, A_had, C)[0]
                f()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    f()
                    torch.cuda.synchronize()
                names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                emit({"kind": "dispatch", "set": s, "shape": name, "k": k, "n": n, "K": K, "kernels": names})
                print(f"{name:14s} K={K}: {len(names)} kernel(s): {'; '.join(x[:70] for x in names)}", flush=True)
            return

        for i, (s, name, k, n, K) in enumerate(pc.rotated(todo, a.round - 1)):
            waited = pc.cool(a.clocks)
            if waited:
                emit({"kind": "cooled", "before": name, "K": K, "seconds": round(waited, 1)})
            if pc.too_hot(a.clocks, a.max_temp):
                emit({"kind": "stopped", "reason": f"temperature above {a.max_temp} C", "at": name, "K": K})
                print(f"stopped: temperature above {a.max_temp} C", flush=True)
                return
            nc = a.copies or pc.copies(k, n, K, stamp["l2_bytes"])
            A, Bs, suh, svh, A_had, C = setup(k, n, K, dev, gen, nc)
            us, ncalls, t0, t1 = pc.time_calls(calls_for(A, Bs, suh, svh, A_had, C))
            row = {"kind": "time", "set": s, "shape": name, "k": k, "n": n, "K": K, "order": i, "copies": nc,
                   "calls": ncalls, "us": round(us, 3), "mweights": k * n / 1e6, "bytes": k * n * K // 8,
                   "t0": round(t0, 3), "t1": round(t1, 3)}
            emit(row)
            print(f"r{a.round} {a.arm:12s} {name:14s} K={K} x{nc:<3d} {us:8.2f} us "
                  f"{k * n * K / 8 / us / 1e3:6.1f} GB/s", flush=True)
            del A, Bs, suh, svh, A_had, C
            torch.cuda.empty_cache()
    print(f"done {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)


if __name__ == "__main__":
    main()
