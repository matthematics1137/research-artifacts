#!/usr/bin/env python3
"""Bytes of weights by group in a safetensors model directory, from the headers only (stdlib, reads no tensor data).

Paper 8 uses it to turn kernel time into achieved bandwidth: decode streams every weight the text model uses once per
token (plus the MTP layer and the head once per drafted token), so bytes per group / time per group = GB/s.
Groups are keyed on the tensor name: embed, lm_head, mtp, vision, and for each text layer its module (the trellis
and its scales grouped together) with the tensor suffix kept, e.g. 'layer.mlp.gate_proj'.
  python3 weight_bytes.py MODEL_DIR [--json OUT]
"""
import collections
import glob
import json
import os
import re
import struct
import sys


def headers(model_dir):
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        if os.path.basename(path).startswith("cal_"):
            continue
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for name, meta in h.items():
            if name != "__metadata__":
                a, b = meta["data_offsets"]
                yield name, b - a, meta["dtype"], meta["shape"]


def group(name):
    if "visual" in name or "vision" in name:
        return "vision"
    if name.startswith("mtp") or ".mtp." in name:
        return "mtp"
    if "embed_tokens" in name:
        return "embed"
    if name.startswith("lm_head"):
        return "lm_head"
    m = re.search(r"layers\.\d+\.(.+?)\.(trellis|suh|svh|su|sv|weight|bias|mcg|mul1|scale|A_log|dt_bias|conv1d.*)$", name)
    if m:
        return "layer." + m.group(1)
    m = re.search(r"layers\.\d+\.(.+)$", name)
    return "layer." + m.group(1) if m else "other:" + name


def main():
    model_dir = sys.argv[1]
    by = collections.Counter()
    for name, size, dtype, shape in headers(model_dir):
        by[group(name)] += size
    total = sum(by.values())
    rows = sorted(by.items(), key=lambda kv: -kv[1])
    for g, b in rows:
        print(f"{b / 2**20:10.1f} MiB  {g}")
    text = sum(b for g, b in by.items() if g.startswith("layer."))
    print(f"total {total / 2**30:.3f} GiB; text layers {text / 2**30:.3f} GiB")
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump({"model_dir": model_dir, "bytes": dict(rows), "text_layer_bytes": text}, f, indent=1)


if __name__ == "__main__":
    main()
