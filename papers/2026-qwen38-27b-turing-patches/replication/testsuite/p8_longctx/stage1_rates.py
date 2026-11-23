#!/usr/bin/env python3
"""P8 long context, stage 1: living-map rows (research/livingmap/SCHEMA.md) from stage1_summarize.py's results.json.

Rows, all for the adopted w13 build on the 2080 Ti with MTP on:
  - each completed ladder step's peak GPU memory at its filled context (measured);
  - each format's reach, the longest filled context with at least 512 MiB headroom (measured; bracketed by the first
    failing step when the ladder found one);
  - each format's memory cost per 1K tokens of context between its two longest completed steps (derived);
  - each registered prediction the data refuted (refuted, no value).

  python stage1_rates.py RESULTS_JSON DATE --out rates.jsonl
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_spec as S  # noqa: E402

PREREG = "research/prereg/2026-11-01-p8-longctx-stage1.md"
RIFE = "research-ledger-reference"
EVIDENCE = ["testsuite/results/turing/longctx/stage1/report.md", "testsuite/results/turing/longctx/stage1/results.json"]
NAMES = {"8,8": "8-bit", "4,4": "4-bit", "8,4": "split (8-bit keys, 4-bit values)", "fp16": "fp16"}


def op(fmt, context):
    return {"card": "RTX 2080 Ti 11 GB", "engine": "exllamav3 1.6.5+w13 (mode 2) under TabbyAPI b1c2253",
            "model": "Qwen3.8-27B", "weights": "EXL3 2.0 bpw", "kv": fmt, "context": context,
            "drafting": "MTP, 6 tokens, dynamic", "power_cap_w": 260, "sampler": "greedy (temperature 0)",
            "task": "one filled single-needle prompt (P6 generator), 64 decoded tokens"}


def row(rid, quantity, value, unit, fmt, context, status, note, date, n=1, conversion="kv-quant",
        gives="information", gets="MiB"):
    return {"id": rid, "conversion": conversion, "gives": gives, "gets": gets, "quantity": quantity,
            "value": value, "unit": unit, "ci95": None, "n": n, "operating_point": op(fmt, context),
            "status": status, "evidence": EVIDENCE, "prereg": PREREG, "rife": RIFE, "date": date,
            "supersedes": None, "note": note}


def rows_from(res, date):
    out = []
    by_fmt = {}
    for s in res["steps"]:
        by_fmt.setdefault(s["format"], []).append(s)
        if s["outcome"] == "completed" and s["peak_mib"] is not None:
            k = s["length"] // S.K
            out.append(row(f"p8lc-s1-peak-{s['step']}",
                           f"peak GPU memory serving a filled {k}K-token context with the {NAMES[s['format']]} cache",
                           s["peak_mib"], "MiB", s["format"], s["length"], "measured",
                           f"nvidia-smi memory.used maximum at 1 Hz from server start to exit, baseline "
                           f"{s['baseline_mib']} MiB included (display, sunshine). Projected "
                           f"{s['projected'][0]}-{s['projected'][1]} MiB from the census anchors; step {s['verdict']}.",
                           date))
    for fmt, f in res["formats"].items():
        if f["reach"] is not None:
            bracket = (f"; the next step ({f['first_failure'] // S.K}K) failed, so the true reach lies below it"
                       if f["first_failure"] else "; the ladder ended without a failure, so the reach may be longer")
            out.append(row(f"p8lc-s1-reach-{fmt.replace(',', '')}",
                           f"longest filled context with at least 512 MiB headroom, {NAMES[fmt]} cache",
                           f["reach"], "tokens", fmt, f["reach"], "measured",
                           "Longest passing ladder step: prefill and 64-token decode completed with the peak at or "
                           f"below 10,752 MiB{bracket}.", date, gets="capability"))
        done = [s for s in by_fmt.get(fmt, []) if s["outcome"] == "completed" and s["peak_mib"] is not None]
        if len(done) >= 2:
            a, b = done[-2], done[-1]
            slope = round((b["peak_mib"] - a["peak_mib"]) / ((b["length"] - a["length"]) / S.K), 2)
            out.append(row(f"p8lc-s1-slope-{fmt.replace(',', '')}",
                           f"GPU memory per 1K tokens of context, {NAMES[fmt]} cache (main and MTP layers)",
                           slope, "MiB per 1K tokens", fmt, [a["length"], b["length"]], "derived",
                           f"Difference of two measured peaks ({a['step']}, {b['step']}) over their context "
                           "difference; includes the cache and any prefill buffers that grow with context.",
                           date, n=2, gives="MiB", gets="capability"))
    v = res["verdicts"]
    claims = {
        "P1": ("measured peaks within -150 to +250 MiB of the census-based projection", None),
        "P2": ("the 8-bit cache passes at 64K", "8,8"),
        "P3": ("the 4-bit cache reaches at least 128K", "4,4"),
    }
    for key, (what, fmt) in claims.items():
        if v.get(key) and str(v[key]).startswith("refuted"):
            out.append({"id": f"p8lc-s1-{key.lower()}-refuted", "conversion": "kv-quant", "gives": "information",
                        "gets": "capability", "quantity": f"registered prediction {key}: {what}", "value": None,
                        "unit": "claim", "ci95": None, "n": None, "operating_point": op(fmt or "all formats", None),
                        "status": "refuted", "evidence": EVIDENCE, "prereg": PREREG, "rife": RIFE, "date": date,
                        "supersedes": None, "note": f"{key} {v[key]} (stage 1 report)."})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("date")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.results) as f:
        res = json.load(f)
    rows = rows_from(res, args.date)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
