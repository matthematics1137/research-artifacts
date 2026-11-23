#!/usr/bin/env python3
"""Bits per weight of every EXL3 trellis tensor in a model directory, from safetensors headers only (stdlib).

An EXL3 trellis is stored as uint16 [in/16, out/16, 16*K] (16*K + 8 at the half-integer rates), so K = last dim / 16.
Prints one line per module type with the K of each layer (compact), and a table of K by module type.
  python3 layer_bits.py MODEL_DIR [--json OUT]
"""
import collections
import glob
import json
import os
import re
import struct
import sys


def trellis(model_dir):
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        if os.path.basename(path).startswith("cal_"):
            continue
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for name, meta in h.items():
            if name.endswith(".trellis"):
                yield name[: -len(".trellis")], meta["shape"]


def main():
    per = collections.defaultdict(dict)
    for name, shape in trellis(sys.argv[1]):
        k = shape[2] / 16
        m = re.search(r"layers\.(\d+)\.(.+)$", name)
        if m:
            # the drafting layer is mtp.layers.0: keep it apart, or it overwrites text layer 0 (fixed 2026-10-31)
            mod = ("mtp." if name.startswith("mtp.") else "") + m.group(2)
            per[mod][int(m.group(1))] = (k, shape[0] * 16, shape[1] * 16)
        else:
            per[name][-1] = (k, shape[0] * 16, shape[1] * 16)
    for mod, layers in sorted(per.items()):
        ks = collections.Counter(v[0] for v in layers.values())
        dims = {(v[1], v[2]) for v in layers.values()}
        print(f"{mod:<34} {dict(sorted(ks.items()))}  in x out {sorted(dims)}")
        if len(ks) > 1:
            print("    by layer: " + " ".join(f"{i}:{v[0]:g}" for i, v in sorted(layers.items())))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump({m: {str(i): v for i, v in l.items()} for m, l in per.items()}, f, indent=1)


if __name__ == "__main__":
    main()
