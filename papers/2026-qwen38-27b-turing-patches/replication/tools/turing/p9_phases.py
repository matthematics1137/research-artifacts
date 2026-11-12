#!/usr/bin/env python3
"""P9 phase 1, M2 (research/prereg/2026-11-05-p9-phase1.md): the phase split of the one-row int8 sq GEMV.

Builds tools/turing/p9/phases.cu against a patched exllamav3 source tree (its exllamav3_ext directory on the include
path, the engine's own nvcc flags) and times its variants (P flags: 1 stage, 2 loop, 4 epilogue; 16 the null kernel)
on the model shapes, the near-empty shape and the grid at K = 1 and 2, with the weight tensor rotated through copies
past three L2 sizes (p9_common.py). The build on PYTHONPATH must be the same source tree's, because each case first
asks the engine for its decomposition and output:
- where W9's host rule is on (EXL3_INT8_SQ_FIT_LOG=1 prints it), its slice height and grid; otherwise the kernel's own
  rule, computed here from the full variant's occupancy as exl3_gemv_int8_sq does;
- the full variant (P = 7) must reproduce ext.exl3_gemm's output bit for bit at that decomposition.
Then every variant is timed, in an order rotated per round, and the loop alone runs at the other bit width's
decomposition (model shapes). One process runs all rounds for one build.
  python3 p9_phases.py --src SRC/exllamav3/exllamav3_ext --tag NAME --out ROWS.jsonl [--rounds 7] [--smoke]
"""
import argparse
import json
import os
import re
import tempfile
import time

import torch
from torch.utils.cpp_extension import load

import p9_common as pc

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {7: "full", 6: "no stage", 5: "no loop", 3: "no epilogue", 2: "loop only", 1: "stage only",
            4: "epilogue only", 0: "counter only", 16: "null"}
FIT_RE = re.compile(r"EXL3_INT8_SQ_FIT k=(\d+) n=(\d+) K=(\d+) M=1 residual=0: kernel rule rows_per=(\d+) ksplit=(\d+) "
                    r"grid=(\d+) -> host rows_per=(\d+) ksplit=(\d+) grid=(\d+)")


def build(src, tag, minb=0):
    """minb 0: a probe build without the occupancy bound; otherwise every variant held to minb blocks per SM."""
    p = torch.cuda.get_device_properties(0)
    cc = f"{p.major}{p.minor}"
    flags = ["-O3", "--use_fast_math", "-std=c++17", f"-gencode=arch=compute_{cc},code=sm_{cc}",
             "-Xcudafe", "--diag_suppress=177", "-Xcudafe", "--diag_suppress=20012", "-lineinfo"]
    if minb:
        flags.append(f"-DPH_MINB={minb}")
    return load(name=f"p9_phases_{tag}_b{minb}", sources=[os.path.join(HERE, "p9", "phases.cu")],
                extra_include_paths=[src], extra_cuda_cflags=flags, verbose=False)


def engine_call(eng, A, B, suh, svh, n):
    C = torch.empty((1, n), dtype=torch.half, device=A.device)
    eng.exl3_gemm(A, B, C, suh, torch.empty_like(A), svh, -1, False, True, 0)
    return C


def with_stderr(fn):
    """Run fn with file descriptor 2 sent to a temporary file; return fn's result and what was written."""
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tf:
        os.dup2(tf.fileno(), 2)
        try:
            res = fn()
            torch.cuda.synchronize()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tf.seek(0)
        return res, tf.read().decode(errors="replace")


def kernel_rule(ph, K, k, n, sms, rows_cap):
    """exl3_gemv_int8_sq's decomposition without the host's choice: seed at 6 x SMs, grid from occupancy, then again."""
    rows_total, nb = k // 16, n // 256

    def decomp(grid):
        r = -(-rows_total * nb // grid)
        rp = (max(r, min(2 * r, 32)) + 7) & ~7
        return min(max(rp, 16), rows_cap, (rows_total + 7) & ~7)

    rp = decomp(6 * sms)
    blocks = ph.info(K, 7, rp)[2]
    grid = min(max(blocks, 1) * sms, 1024)
    return decomp(grid), grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="two model shapes, one round; rows tagged smoke")
    ap.add_argument("--clocks", default="")
    ap.add_argument("--max-temp", type=float, default=0.0)
    a = ap.parse_args()
    os.environ["EXL3_INT8_SQ_FIT_LOG"] = "1"     # read on the engine's first one-row call
    from exllamav3.ext import exllamav3_ext as eng
    from exllamav3.version import __version__
    # The full variant's residency, unbounded, at a small slice (registers or threads limit it, not shared memory),
    # then the real build with every variant held to it (Amendment 1)
    minb = build(a.src, a.tag).info(1, 7, 16)[2]
    ph = build(a.src, a.tag, minb)
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1105)
    stamp = {"tag": a.tag, "src": a.src, "exllamav3": __version__, "smoke": a.smoke, "ph_minb": minb,
             **pc.device_stamp(dev)}
    sms, l2 = stamp["sms"], stamp["l2_bytes"]
    rows_cap = ph.rows_max()
    shapes = pc.MODEL + pc.TINY + pc.GRID
    if a.smoke:
        shapes = [s for s in pc.MODEL if s[0] in ("attn_kv", "mlp_gate")]
        a.rounds = 1
    out = open(a.out, "a")

    def emit(row):
        out.write(json.dumps({**row, **stamp}) + "\n")
        out.flush()

    for K in (1, 2):
        for v, name in VARIANTS.items():
            regs, static, blocks, dyn, optin, local = ph.info(K, v, 64)
            emit({"kind": "info", "K": K, "variant": v, "name": name, "registers": regs, "static_smem": static,
                  "blocks_per_sm_at_64_rows": blocks, "optin_smem": optin, "local_bytes": local})

    # Decomposition and the bit-identical check, once per (shape, K)
    decomp, host_seen = {}, {}
    for sname, k, n in shapes:
        for K in (1, 2):
            A = (torch.randn((1, k), device=dev, generator=gen) * 0.1).half()
            B = pc.trellis(k, n, K, dev, gen)
            suh, svh = pc.flips(k, dev, gen), pc.flips(n, dev, gen)
            C_eng, log = with_stderr(lambda: engine_call(eng, A, B, suh, svh, n))
            host = [m for m in FIT_RE.finditer(log) if (int(m[1]), int(m[2]), int(m[3])) == (k, n, K)]
            if host and int(host[-1][7]) > 0:
                rp, grid, rule = int(host[-1][7]), int(host[-1][9]), "host (W9)"
                host_seen[(k, n, K)] = (rp, grid)
            elif (k, n, K) in host_seen:   # the engine logs its choice once per shape: a grid shape may repeat one
                (rp, grid), rule = host_seen[(k, n, K)], "host (W9)"
            else:
                (rp, grid), rule = kernel_rule(ph, K, k, n, sms, rows_cap), "kernel"
            ksplit = -(-(k // 16) // rp)
            locks = torch.zeros(ph.ws_reserved() + ksplit * n, dtype=torch.int32, device=dev)
            C_ph = torch.empty((1, n), dtype=torch.half, device=dev)
            ph.launch(K, 7, A, B, C_ph, locks, suh, svh, k, n, rp, grid)
            torch.cuda.synchronize()
            same = bool(torch.equal(C_ph, C_eng))
            decomp[(sname, K)] = (rp, grid, ksplit)
            # Waves each variant needs at this grid; more than one is excluded from the analysis (Amendment 1)
            waves = {v: -(-grid // (max(ph.info(K, v, rp)[2], 1) * sms)) for v in VARIANTS}
            emit({"kind": "decomp", "shape": sname, "k": k, "n": n, "K": K, "rule": rule, "rows_per": rp, "grid": grid,
                  "ksplit": ksplit, "bit_identical_to_engine": same, "waves": waves})
            multi = [VARIANTS[v] for v, w in waves.items() if w > 1]
            print(f"{sname:12s} K={K} {rule:9s} rows_per {rp:4d} grid {grid:4d} ksplit {ksplit:3d} identical={same}"
                  f"{'  multi-wave: ' + ', '.join(multi) if multi else ''}", flush=True)

    # Timed items: (shape, K, decomposition source K) groups; within a group every variant, rotated per round
    groups = [(s, k, n, K, K) for s, k, n in shapes for K in (1, 2)]
    groups += [(s, k, n, K, 3 - K) for s, k, n in pc.MODEL if not a.smoke or s in ("attn_kv", "mlp_gate") for K in (1, 2)]
    vlist = list(VARIANTS)
    for r in range(a.rounds):
        for gi, (sname, k, n, K, dK) in enumerate(pc.rotated(groups, r)):
            waited = pc.cool(a.clocks)
            if waited:
                emit({"kind": "cooled", "before": sname, "K": K, "round": r + 1, "seconds": round(waited, 1)})
            if pc.too_hot(a.clocks, a.max_temp):
                emit({"kind": "stopped", "reason": f"temperature above {a.max_temp} C", "round": r + 1})
                print("stopped: temperature", flush=True)
                return
            rp, grid, ksplit = decomp[(sname, dK)]
            nc = pc.copies(k, n, K, l2)
            A = (torch.randn((1, k), device=dev, generator=gen) * 0.1).half()
            Bs = [pc.trellis(k, n, K, dev, gen) for _ in range(nc)]
            suh, svh = pc.flips(k, dev, gen), pc.flips(n, dev, gen)
            C = torch.empty((1, n), dtype=torch.half, device=dev)
            locks = torch.zeros(ph.ws_reserved() + ksplit * n, dtype=torch.int32, device=dev)
            vs = vlist if dK == K else [2]
            for v in pc.rotated(vs, r + gi, step=1):
                calls = [lambda B=B, v=v: ph.launch(K, v, A, B, C, locks, suh, svh, k, n, rp, grid) for B in Bs]
                us, ncalls, t0, t1 = pc.time_calls(calls)
                emit({"kind": "time", "shape": sname, "k": k, "n": n, "K": K, "decomp_K": dK, "variant": v,
                      "name": VARIANTS[v], "round": r + 1, "rows_per": rp, "grid": grid, "ksplit": ksplit, "copies": nc,
                      "calls": ncalls, "us": round(us, 3), "mweights": k * n / 1e6, "t0": round(t0, 3), "t1": round(t1, 3)})
            print(f"r{r + 1} {sname:12s} K={K} (decomp K={dK}) done", flush=True)
            del A, Bs, suh, svh, C, locks
            torch.cuda.empty_cache()
    out.close()
    print(f"done {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)


if __name__ == "__main__":
    main()
