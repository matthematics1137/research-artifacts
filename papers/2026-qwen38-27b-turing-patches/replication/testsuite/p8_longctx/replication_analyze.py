#!/usr/bin/env python3
"""P8 value-bits replication: judge R1, N, R2 and R3 as research/prereg/2026-11-04-p8-value-bits-replication.md
registers them. Written and committed before any arm's answers were read.

  R1 (primary): D1 = rep-64k-84 - rep-64k-88-triton, on the 96 fresh items
       replicated if the 90% lower bound > 0 (and noted if > +5); equivalent if the interval lies within +-5 with the
       lower bound <= 0; otherwise not replicated (reversed if the upper bound < 0)
  N:   D_N = noise-64k-88-triton-qcs128 - s2-64k-88-triton, on the 96 original 64K items; K2's 17 discordant items
       are beyond the harmless spread if the Wilson 95% upper bound of N's discordance (of 96) is below 17/96. The
       first-16 basis: N's discordance on the first 16 items against 8,4's 6 of 16 there (8,4 - 8,8-fd75d, stage 2)
  R2:  D2 = rep-32k-84 - fp16 at 32K; the reference is stage 2's s2-32k-fp16 when the replication journal shows the
       16-item check reproduced it (rep-32k-fp16 skipped as "reproduced"), else rep-32k-fp16. Outcomes as R1.
       Descriptive: rep-32k-84 - s2-32k-88 (fd75d-read 8-bit at 32K)
  R3:  descriptive: D3a = rep-64k-86 - s2-64k-88-triton, D3b = s2-64k-84 - rep-64k-86; monotone if both >= 0

Statistics and scorer: stage2_analyze (stage 2's Arms and comparison: P6's scorer, 90% percentile bootstrap with
10,000 resamples and seed 20261102, exact McNemar, Wilson). Arms the budget skipped, or that did not finish, are
reported as such; paired comparisons use the items both arms completed. Laptop only; the journals are small JSON.

  python replication_analyze.py --replication J --stage2 J2 --arm JA --items DIR [--out results.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_analyze as A  # noqa: E402
import stage2_stats as T  # noqa: E402

K2_DISCORDANT, K2_N = 17, 96          # K2: 14 + 3 discordant of 96 (the accuracy arm)
SPLIT_FIRST16 = 6                     # 8,4 against 8,8-fd75d on the first 16 graded items (number check, F14)


def outcome(c):
    lo, hi = c["ci90"]
    if lo > 0:
        return "replicated, interval above +5" if lo > 5 else "replicated"
    if lo >= -5 and hi <= 5:
        return "equivalent"
    return "not replicated, reversed" if hi < 0 else "not replicated"


def texts(rows, arm):
    return {r["uid"]: r.get("text") or "" for r in rows
            if r.get("kind") == "item" and r["arm"] == arm and r.get("status") == "ok"}


def identity(rows, a, b):
    ta, tb = texts(rows, a), texts(rows, b)
    common = [u for u in ta if u in tb]
    return {"n": len(common), "identical": sum(1 for u in common if ta[u] == tb[u])}


def status(rows, arm):
    closed = next((r for r in rows if r.get("kind") in ("arm_end", "arm_skipped") and r["arm"] == arm), None)
    if closed is None:
        return "not closed" if any(r.get("kind") == "item" and r["arm"] == arm for r in rows) else "not run"
    return closed.get("status") or f"skipped ({closed.get('why')})"


def first16_discordance(arms, a, b):
    """Discordant items between a and b on the first 16 items of a's item file, in file order."""
    first = list(arms.items[a])[:16]
    sa, sb = arms.scores[a], arms.scores[b]
    common = [u for u in first if u in sa and u in sb]
    return {"n": len(common), "discordant": sum(1 for u in common if sa[u] != sb[u])}


def judge(rep_rows, stage2_rows, arm_rows, items_dir):
    rows = stage2_rows + arm_rows + rep_rows
    arms = A.Arms(rows, items_dir)
    has = lambda a: a in arms.scores  # noqa: E731
    res = {"arms": {}, "status": {}}
    for a in ("rep-64k-88-triton", "rep-64k-84", "noise-64k-88-triton-qcs128", "rep-32k-84", "rep-32k-fp16-check",
              "rep-32k-fp16", "rep-64k-86"):
        res["status"][a] = status(rep_rows, a)
        if has(a):
            res["arms"][a] = A.arm_secondary(arms, a)
    if has("rep-64k-84") and has("rep-64k-88-triton"):
        c = A.comparison(arms, "rep-64k-84", "rep-64k-88-triton")
        res["R1"] = {**c, "outcome": outcome(c), "text_identity": identity(rows, "rep-64k-84", "rep-64k-88-triton")}
    if has("noise-64k-88-triton-qcs128"):
        c = A.comparison(arms, "noise-64k-88-triton-qcs128", "s2-64k-88-triton")
        disc = c["n10"] + c["n01"]
        lo, hi = T.wilson(disc, c["n"])
        res["N"] = {**c, "discordant": disc, "wilson95": [lo, hi],
                    "K2_discordance": K2_DISCORDANT / K2_N,
                    "K2_beyond_harmless_spread": hi < K2_DISCORDANT / K2_N,
                    "text_identity": identity(rows, "noise-64k-88-triton-qcs128", "s2-64k-88-triton"),
                    "first16": {"N": first16_discordance(arms, "noise-64k-88-triton-qcs128", "s2-64k-88-triton"),
                                "split_against_8bit": SPLIT_FIRST16}}
    if has("rep-32k-84"):
        reproduced = any(r.get("kind") == "arm_skipped" and r["arm"] == "rep-32k-fp16" and r.get("why") == "reproduced"
                         for r in rep_rows)
        ref = "s2-32k-fp16" if reproduced else ("rep-32k-fp16" if has("rep-32k-fp16") else None)
        if ref is not None:
            c = A.comparison(arms, "rep-32k-84", ref)
            res["R2"] = {**c, "outcome": outcome(c), "fp16_reference": ref, "reproduced": reproduced,
                         "check": next((r for r in rep_rows if r.get("kind") == "reproduction_check"), None)}
        if has("s2-32k-88"):
            res["R2_against_8bit_fd75d"] = A.comparison(arms, "rep-32k-84", "s2-32k-88")
    if has("rep-64k-86"):
        d3a = A.comparison(arms, "rep-64k-86", "s2-64k-88-triton")
        d3b = A.comparison(arms, "s2-64k-84", "rep-64k-86")
        res["R3"] = {"D3a_6_minus_8": d3a, "D3b_4_minus_6": d3b,
                     "monotone": d3a["diff_points"] >= 0 and d3b["diff_points"] >= 0}
    res["excluded"] = {a: arms.excluded[a] for a in arms.excluded if a.startswith(("rep-", "noise-"))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replication", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = judge(A.load_rows(args.replication), A.load_rows(args.stage2), A.load_rows(args.arm), args.items)
    text = json.dumps(res, indent=1, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    keys = ("n", "acc_a", "acc_b", "diff_points", "ci90", "n10", "n01", "mcnemar_p", "outcome")
    print(json.dumps({"status": res["status"], **{k: {x: res[k].get(x) for x in keys} for k in ("R1", "N", "R2")
                                                  if k in res}}, indent=1, default=str))


if __name__ == "__main__":
    main()
