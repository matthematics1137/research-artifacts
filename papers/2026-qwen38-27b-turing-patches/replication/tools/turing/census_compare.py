#!/usr/bin/env python3
"""Paper 8: compare census arms against a baseline (stdlib only), from each arm's report.json.

Per task: ms per decoded token (capture span) and its change against the baseline; per launch shape: us per token
in each arm, so a dispatch switch shows which kernels it moved. Undrafted captures by default.
  python3 census_compare.py --base DIR [--cfg nodraft] ARM_DIR... [--md OUT]
"""
import argparse
import json
import os


def load(d, cfg):
    r = json.load(open(os.path.join(d, "report.json")))
    return r.get(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cfg", default="nodraft")
    ap.add_argument("--md")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    dirs = [a.base] + a.arms
    reps = {os.path.basename(d.rstrip("/")): load(d, a.cfg) for d in dirs}
    names = [n for n in reps if reps[n]]
    base = names[0]
    out = [f"# Census arms vs {base} ({a.cfg})", "", "| arm | " + " | ".join(t["task"] for t in reps[base]["tasks"]) +
           " | mean ms/token | change |", "|---" * (len(reps[base]["tasks"]) + 3) + "|"]
    bmean = sum(t["ms_per_token"] for t in reps[base]["tasks"]) / len(reps[base]["tasks"])
    for n in names:
        ts = reps[n]["tasks"]
        mean = sum(t["ms_per_token"] for t in ts) / len(ts)
        out.append(f"| {n} | " + " | ".join(f"{t['ms_per_token']:.2f}" for t in ts) +
                   f" | {mean:.2f} | {100 * (mean - bmean) / bmean:+.1f}% |")
    keys = []
    for n in names:
        for s in reps[n]["shapes"][:10]:
            k = (s["family"], s["grid"], s["block"])
            if k not in keys:
                keys.append(k)
    out += ["", "Launch shapes, us per decoded token (launches per token):", "",
            "| family | grid | block | " + " | ".join(names) + " |", "|---" * (3 + len(names)) + "|"]
    for k in keys:
        cells = []
        for n in names:
            m = next((s for s in reps[n]["shapes"] if (s["family"], s["grid"], s["block"]) == k), None)
            cells.append(f"{m['us_per_token']:.0f} ({m['launches_per_token']:g})" if m else "-")
        out.append(f"| `{k[0]}` | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    main()
