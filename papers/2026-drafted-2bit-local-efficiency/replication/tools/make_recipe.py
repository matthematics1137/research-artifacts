#!/usr/bin/env python3
"""Recipe YAML reproducing the stock SC_2.00bpw_H3 per-tensor allocation (from stock_bits_per_tensor.json)."""
import json, sys
per = json.load(open(sys.argv[1])); out = sys.argv[2]
body = {k: v for k, v in per.items() if not k.startswith(("lm_head", "mtp")) and "visual" not in k}
with open(out, "w") as f:
    f.write("# per-tensor bitrates copied from turboderp/Qwen3.8-27B-exl3@SC_2.00bpw_H3 (safetensors trellis widths)\n")
    f.write("achieved_bpw: 2.0\nhead_bits: 3\ntensors:\n")
    for k in sorted(body): f.write(f"  {k}: {body[k]}\n")
print(len(body), "body tensors ->", out)
