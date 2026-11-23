#!/usr/bin/env python3
"""P8 extension (research/prereg/2026-11-05-p8-call-split-ada.md): the per-call split of the laptop census, judged
against the registered predictions P1-P3. Reads the census's int8_by_bits.md (tools/turing/census_bits.py) and its
GPU telemetry; the read rate is the RTX 4080 Laptop GPU's rated 432 GB/s scaled by the memory clock recorded during
decode (rated at 9001 MHz), times 550/616 for the primary analysis (the 2080 Ti's assumed fraction) and 1.0 for the
sensitivity. The split is computed as call_split() in paper8/scripts/build_claims.py. Writes split.json and split.md.
  python3 tools/turing/call_split_ada.py testsuite/results/ada/census/d0-stock-mgemmn0
"""
import csv
import glob
import json
import os
import statistics as st
import sys

RATED, RATED_MHZ, FRACTION = 432e9, 9001.0, 550 / 616


def md_rows(path):
    out = []
    for line in open(path):
        if line.startswith("|") and not line.startswith("|---"):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def split(rows, rate):
    calls = [r for r in rows if len(r) == 7 and r[0] != "role"]
    fits = {int(r[0]): (float(r[2]), float(r[3])) for r in rows if len(r) == 5 and r[0] != "K"}
    tokens = next(int(r[2]) for r in calls if r[0] == "head")
    read = lambda k: k / 8 * 1e6 / rate * 1e6          # µs per million K-bit weights
    n = t_all = floor_all = t_fit = floor_fit = fixed = extra = 0.0
    for role, k, launches, med, mb, _, _ in calls:
        k, per_tok, med, mb = int(k), int(launches) / tokens, float(med), float(mb)
        floor = mb * 1e6 / rate * 1e6
        n, t_all, floor_all = n + per_tok, t_all + per_tok * med, floor_all + per_tok * floor
        if role != "head" and k in fits:
            a, b = fits[k]
            t_fit, floor_fit = t_fit + per_tok * med, floor_fit + per_tok * floor
            fixed += per_tok * a
            extra += per_tok * (mb * 8 / k) * (b - read(k))
    gap = t_fit - floor_fit
    return {"rate_gbs": rate / 1e9, "calls": n, "matrix_ms": t_all / 1e3, "floor_ms": floor_all / 1e3,
            "gap_ms": gap / 1e3, "fixed_ms": fixed / 1e3, "per_weight_ms": extra / 1e3,
            "residual_ms": (gap - fixed - extra) / 1e3, "fixed_share": fixed / gap if gap else None,
            "fits": {str(k): {"fixed_us": a, "us_per_mweights": b, "per_weight_over_read": b / read(k)}
                     for k, (a, b) in fits.items()}}


def main():
    d = sys.argv[1]
    rows = md_rows(os.path.join(d, "int8_by_bits.md"))
    calls = [r for r in rows if len(r) == 7 and r[0] != "role"]
    tel = []
    for f in sorted(glob.glob(os.path.join(d, "telemetry-*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh, skipinitialspace=True):
                try:
                    util = float(r["utilization.gpu [%]"].split()[0])
                    mhz = float(r["clocks.current.memory [MHz]"].split()[0])
                except (KeyError, ValueError):
                    continue
                tel.append((util, mhz, r))
    busy = [m for u, m, _ in tel if u >= 50]
    mem_mhz = st.median(busy) if busy else None
    temps = [float(r["temperature.gpu"]) for _, _, r in tel if r.get("temperature.gpu", "").strip().isdigit()]
    rated_at_clock = RATED * (mem_mhz / RATED_MHZ) if mem_mhz else RATED
    primary, rated = split(rows, rated_at_clock * FRACTION), split(rows, rated_at_clock)
    by = {}
    for role, k, *_rest in calls:
        by.setdefault(role, {})[int(k)] = float(_rest[1])
    p1 = {role: v[1] / v[2] for role, v in by.items() if 1 in v and 2 in v}
    verdict = {
        "P1": all(x >= 0.90 for x in p1.values()) and bool(p1),
        "P2": primary["fits"].get("2", {}).get("fixed_us", 0) > 0 and (primary["fixed_share"] or 0) > 0.5,
        "P3": (rated["fixed_share"] or 0) > 0.5,
    }
    out = {"memory_clock_mhz_during_decode": mem_mhz, "rated_gbs_at_clock": rated_at_clock / 1e9,
           "max_temperature_c": max(temps) if temps else None, "one_bit_over_two_bit": p1,
           "primary": primary, "rated": rated, "verdict": verdict}
    json.dump(out, open(os.path.join(d, "split.json"), "w"), indent=1)
    lines = [f"# Per-call split on the RTX 4080 Laptop GPU ({os.path.basename(d)})", "",
             f"Memory clock during decode (median, utilization >= 50%): {mem_mhz} MHz; rated bandwidth at that clock "
             f"{rated_at_clock / 1e9:.0f} GB/s. Highest GPU temperature: {out['max_temperature_c']} C.", "",
             "| read rate | calls/token | matrix ms | read ms | fitted excess ms | fixed ms | per-weight ms | residual ms "
             "| fixed share |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("primary (89.3% of rated)", primary), ("rated (100%)", rated)):
        lines.append(f"| {name}: {s['rate_gbs']:.0f} GB/s | {s['calls']:.0f} | {s['matrix_ms']:.2f} | {s['floor_ms']:.2f} "
                     f"| {s['gap_ms']:.2f} | {s['fixed_ms']:.2f} | {s['per_weight_ms']:.2f} | {s['residual_ms']:.2f} "
                     f"| {100 * s['fixed_share']:.0f}% |")
    lines += ["", "1-bit call time over 2-bit call time, same role: " +
              ", ".join(f"{r} {v:.3f}" for r, v in sorted(p1.items())), "",
              "Fits (primary rate): " + "; ".join(f"K={k}: {v['fixed_us']:.1f} µs fixed, {v['us_per_mweights']:.3f} µs "
                                                    f"per M weights = {v['per_weight_over_read']:.2f}x the read"
                                                    for k, v in sorted(primary["fits"].items())), "",
              f"Registered predictions: P1 {'holds' if verdict['P1'] else 'fails'}; "
              f"P2 {'holds' if verdict['P2'] else 'fails'}; P3 {'holds' if verdict['P3'] else 'fails'}."]
    open(os.path.join(d, "split.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
