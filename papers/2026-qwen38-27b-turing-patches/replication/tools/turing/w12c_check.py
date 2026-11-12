#!/usr/bin/env python3
"""Paper 8 W12c: end-to-end exactness check of the IMMA verify-row path against W10's dp4a path.

Dump mode runs ext.exl3_gemm on fixed synthetic mul1 tensors (seeded, the same in every process) for each shape, bit
width and row count, under whatever overlay and EXL3_* switches the process has, and saves the outputs:
  PYTHONPATH=$L/overlay/exl3-w11b EXL3_INT8_SQ_MAX_M=4 python3 w12c_check.py dump --out dp4a.pt
  PYTHONPATH=$L/overlay/exl3-w12c EXL3_INT8_SQ_MAX_M=4 python3 w12c_check.py dump --out imma.pt
Compare mode reports, per case, whether two dumps are bit-identical and the largest relative difference:
  python3 w12c_check.py compare dp4a.pt imma.pt --json ROWS.jsonl
At M = 3-4 both builds run int8 (dp4a against IMMA) with the same partials and epilogue, so they must match bit for bit.
At M = 5-7 the reference is the fp16 kernel (w11b, MAX_M=2), and only the size of the difference is reported.
"""
import argparse
import json
import os
import zlib

import torch

SHAPES = [("attn_kv", 5120, 1024), ("gdn_z", 5120, 6144), ("gdn_qkv", 5120, 10240), ("mlp_gate", 5120, 17408),
          ("mlp_down", 17408, 5120)]
BITS = (1, 2, 3, 4)
ROWS = (1, 2, 3, 4, 5, 6, 7)


def dump(out):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.version import __version__
    dev = torch.device("cuda:0")
    res = {"exllamav3": __version__, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, "cases": {}}
    for name, k, n in SHAPES:
        for K in BITS:
            gen = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{name}|{K}".encode()))   # same in every process
            B = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev, generator=gen)
            suh = (torch.randint(0, 2, (k,), device=dev, generator=gen) * 2 - 1).half()
            svh = (torch.randint(0, 2, (n,), device=dev, generator=gen) * 2 - 1).half()
            A_all = (torch.randn((max(ROWS), k), device=dev, generator=gen) * 0.1).half()
            for m in ROWS:
                A = A_all[:m].contiguous()
                C = torch.empty((m, n), dtype=torch.half, device=dev)
                ext.exl3_gemm(A, B, C, suh, torch.empty_like(A), svh, -1, False, True, 0)
                torch.cuda.synchronize()
                res["cases"][f"{name}|K{K}|m{m}"] = C.cpu()
    torch.save(res, out)
    print(f"dumped {len(res['cases'])} cases from exllamav3 {__version__} {res['env']} -> {out}")


def compare(a_path, b_path, json_out):
    a, b = torch.load(a_path), torch.load(b_path)
    rows = []
    for key in a["cases"]:
        x, y = a["cases"][key].float(), b["cases"][key].float()
        same = bool(torch.equal(a["cases"][key], b["cases"][key]))
        rel = ((x - y).abs().max() / x.abs().max().clamp_min(1e-6)).item()
        finite = bool(torch.isfinite(y).all())
        rows.append({"case": key, "bit_identical": same, "max_rel_diff": rel, "finite": finite})
        print(f"{key:22s} identical={same!s:5s} max_rel_diff={rel:.3e} finite={finite}")
    if json_out:
        with open(json_out, "a") as f:
            for r in rows:
                f.write(json.dumps({**r, "a": a_path, "b": b_path, "a_env": a["env"], "b_env": b["env"],
                                    "a_version": a["exllamav3"], "b_version": b["exllamav3"]}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json", default="")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.out)
    else:
        compare(args.a, args.b, args.json)


if __name__ == "__main__":
    main()
