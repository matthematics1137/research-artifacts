#!/usr/bin/env python3
"""A1: quantization-error anatomy from the E32 conversion logs (per-tensor proxy_err at the assigned bits, per-layer
rfn/cos/sqnr against the fp16 reference trajectory). Prints tables and writes error_anatomy.json.
usage: error_anatomy.py <dry.out> <convert.out> [head.out]"""
import collections, json, re, statistics, sys

TENSOR = re.compile(r"-- Quantized: (model\.language_model\.layers\.(\d+)\.(\S+))\s+bpw:\s+([\d.]+)\s+proxy_err:\s+([\d.]+)")
LAYER = re.compile(r"-- Quantized: model\.language_model\.layers\.(\d+)\s+bpw:\s+([\d.]+)\s+rfn:\s+([\d.]+)\s+cos:\s+([\d.]+)\s+sqnr:\s+([\d.]+)\s+\[([\d.]+) s\]")

tensors, layers = {}, {}
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        m = TENSOR.search(line)
        if m:
            tensors[m.group(1)] = {"layer": int(m.group(2)), "comp": m.group(3), "bpw": float(m.group(4)), "err": float(m.group(5))}
        m = LAYER.search(line)
        if m:
            layers[int(m.group(1))] = {"bpw": float(m.group(2)), "rfn": float(m.group(3)), "cos": float(m.group(4)), "sqnr": float(m.group(5)), "s": float(m.group(6))}
print(f"tensors {len(tensors)}  layers {len(layers)}")

# --- per-layer: rfn vs depth, attention layers flagged
attn_layers = sorted({v["layer"] for v in tensors.values() if v["comp"].startswith("self_attn")})
print("\nper-layer rfn (cumulative divergence from the fp16 trajectory), attention layers marked *:")
row = []
for i in sorted(layers):
    d = layers[i]; row.append(f"{i}{'*' if i in attn_layers else ''}:{d['rfn']:.4f}")
for k in range(0, len(row), 8): print("  " + "  ".join(row[k:k + 8]))
inc = [(i, layers[i]["rfn"] - layers[i - 1]["rfn"]) for i in sorted(layers) if i - 1 in layers]
inc.sort(key=lambda x: -x[1])
print("largest per-layer rfn increments:", [(i, round(v, 4)) for i, v in inc[:8]])
print("attention-layer mean increment", round(statistics.mean(v for i, v in inc if i in attn_layers), 5), "| GDN-layer mean increment", round(statistics.mean(v for i, v in inc if i not in attn_layers), 5))
print("layer time s: mean", round(statistics.mean(l["s"] for l in layers.values()), 1), "max", max(l["s"] for l in layers.values()))

# --- per-tensor: error by component type and by bits
by_comp = collections.defaultdict(list)
for k, v in tensors.items(): by_comp[v["comp"]].append(v)
print("\nper-tensor proxy_err by component (n, bits histogram, median err at 2 bpw, max err):")
for comp, vs in sorted(by_comp.items(), key=lambda kv: -statistics.median(v["err"] for v in kv[1])):
    bits = collections.Counter(int(v["bpw"]) for v in vs)
    e2 = [v["err"] for v in vs if int(v["bpw"]) == 2]
    print(f"  {comp:28s} n={len(vs):3d} bits={dict(sorted(bits.items()))} med2={statistics.median(e2) if e2 else float('nan'):.4f} max={max(v['err'] for v in vs):.4f}")
print("\nby bits (median proxy_err):", {b: round(statistics.median(v["err"] for v in tensors.values() if int(v["bpw"]) == b), 4) for b in (1, 2, 3, 4)})
print("\ntop-12 error tensors overall:")
for k, v in sorted(tensors.items(), key=lambda kv: -kv[1]["err"])[:12]: print(f"  {k:60s} bpw {v['bpw']:.0f} err {v['err']:.4f}")
print("\nerror by depth band (2-bpw tensors only, median):")
for lo, hi in ((0, 8), (8, 24), (24, 40), (40, 56), (56, 64)):
    e = [v["err"] for v in tensors.values() if int(v["bpw"]) == 2 and lo <= v["layer"] < hi]
    print(f"  layers {lo:2d}-{hi - 1:2d}: {statistics.median(e):.4f} (n={len(e)})")
json.dump({"tensors": tensors, "layers": layers}, open(sys.argv[2].rsplit("/", 1)[0] + "/../error_anatomy.json", "w"), indent=0)
