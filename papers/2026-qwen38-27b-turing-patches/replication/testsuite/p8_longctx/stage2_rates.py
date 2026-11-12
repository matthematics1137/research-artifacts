#!/usr/bin/env python3
"""P8 long context, stages 2-3: living-map rows (research/livingmap/SCHEMA.md) from stage2_analyze.py's results.

  stage2: each graded arm's accuracy (measured, Wilson 95%), and each registered comparison as a paired difference in
          accuracy points (measured; the registered 90% bootstrap interval is in the note, ci95 is the 95% one from the
          same seed) with its usable and equivalent verdicts
  stage3: accuracy against context length for the adopted mode (measured, Wilson 95%) and the usable context (derived)

  python stage2_rates.py stage2 RESULTS_JSON DATE --out rates.jsonl
  python stage2_rates.py stage3 RESULTS_JSON DATE --out rates.jsonl
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PREREG = "research/prereg/2026-11-01-p8-longctx-stages2-3.md"
RIFE = "research-ledger-reference"
FORMAT_OF = {"88": "8,8", "44": "4,4", "84": "8,4", "fp16": "fp16"}
NAMES = {"8,8": "8-bit", "4,4": "4-bit", "8,4": "split (8-bit keys, 4-bit values)", "fp16": "fp16"}
LABEL = {"h2": ("4,4", "8,8", 64), "h1": ("8,8", "fp16", 32), "h3": ("4,4", "fp16", 32), "h2s": ("8,4", "8,8", 64)}


def op(fmt, context, task):
    return {"card": "RTX 2080 Ti 11 GB", "engine": "exllamav3 1.6.5+w13 (mode 2) under TabbyAPI b1c2253",
            "model": "Qwen3.8-27B", "weights": "EXL3 2.0 bpw", "kv": fmt, "context": context,
            "drafting": "MTP, 6 tokens, dynamic", "power_cap_w": 260, "sampler": "greedy (temperature 0)", "task": task}


def row(rid, quantity, value, unit, ci, n, fmt, context, status, note, date, evidence, task, gives="information",
        gets="capability"):
    return {"id": rid, "conversion": "kv-quant", "gives": gives, "gets": gets, "quantity": quantity, "value": value,
            "unit": unit, "ci95": ci, "n": n, "operating_point": op(fmt, context, task), "status": status,
            "evidence": evidence, "prereg": PREREG, "rife": RIFE, "date": date, "supersedes": None, "note": note}


TASK = "multivalue retrieval, k = 12 values per item (P6 generator, calibrated level), all values required"


def stage2_rows(res, date, evidence):
    out = []
    for arm, s in res["arms"].items():
        if not s or arm.startswith(("calib", "cliff")) or arm.endswith("noise"):
            continue
        _, length, code = arm.split("-")
        fmt, a = FORMAT_OF[code], s["accuracy"]
        lo, hi = a["wilson95"]
        out.append(row(f"p8lc-s2-acc-{length}-{code}", f"graded accuracy at {length.upper()} with the {NAMES[fmt]} cache",
                       round(100 * a["acc"], 1), "percent", [round(100 * lo, 1), round(100 * hi, 1)], a["n"], fmt,
                       int(length[:-1]) * 1024, "measured",
                       f"{a['k']} of {a['n']} items fully right; excluded {len(s['excluded'])}.", date, evidence, TASK))
    for key, c in res["comparisons"].items():
        if not c:
            continue
        fa, fb, k = LABEL[key]
        verdict = ("usable and equivalent" if c["equivalent"] else "usable, not shown equivalent" if c["usable"]
                   else "not usable")
        out.append(row(f"p8lc-s2-{key}", f"accuracy cost of the {NAMES[fa]} cache against {NAMES[fb]} at {k}K "
                       f"(paired difference, {key.upper()})", round(c["diff_points"], 1), "accuracy points",
                       c.get("ci95_points"), c["n"], f"{fa} vs {fb}", k * 1024, "measured",
                       f"{verdict}: registered 90% bootstrap interval [{c['ci90'][0]:+.1f}, {c['ci90'][1]:+.1f}] "
                       f"(usable: lower bound above -10; equivalent: within +-5); discordant {c['n10']} / {c['n01']}, "
                       f"exact McNemar p {c['mcnemar_p']:.3g}.", date, evidence, TASK))
    return out


def stage3_rows(res, date, evidence):
    out, fmt = [], res["mode"]
    for p in res["curve"]:
        lo, hi = p["wilson95"]
        out.append(row(f"p8lc-s3-acc-{p['length_k']}k", f"graded accuracy at {p['length_k']}K with the {NAMES[fmt]} "
                       "cache", round(100 * p["acc"], 1), "percent", [round(100 * lo, 1), round(100 * hi, 1)], p["n"],
                       fmt, p["length_k"] * 1024, "measured",
                       f"{p['k']} of {p['n']} items fully right"
                       + (f"; {p['gap_vs_32k']['points']:+.1f} points against 32K on {p['gap_vs_32k']['n']} shared items"
                          if p.get("gap_vs_32k") else "") + ".", date, evidence, TASK))
    if res.get("usable_context_k"):
        out.append(row(f"p8lc-s3-usable-{fmt.replace(',', '')}", f"usable context with the {NAMES[fmt]} cache "
                       "(accuracy within 10 points of its own 32K accuracy)", res["usable_context_k"] * 1024, "tokens",
                       None, None, fmt, res["usable_context_k"] * 1024, "derived",
                       "Longest measured length whose paired accuracy gap to 32K is above -10 points (descriptive; "
                       "no equivalence claim beyond 64K).", date, evidence, TASK))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("stage2", "stage3"))
    ap.add_argument("results")
    ap.add_argument("date")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.results) as f:
        res = json.load(f)
    folder = f"testsuite/results/turing/longctx/{args.what}"
    evidence = [f"{folder}/report.md", f"{folder}/results.json"]
    rows = (stage2_rows if args.what == "stage2" else stage3_rows)(res, args.date, evidence)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
