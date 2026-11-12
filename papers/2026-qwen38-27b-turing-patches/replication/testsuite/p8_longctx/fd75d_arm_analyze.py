#!/usr/bin/env python3
"""P8 fd75d accuracy arm: judge K1 and K2 as research/prereg/2026-11-03-p8-fd75d-accuracy-arm.md registers them.

  K1 (primary): 8,8 read by Triton (arm s2-64k-88-triton) minus 8,8 read by fd75d (stage 2's s2-64k-88)
  K2:           8,4 read by Triton (stage 2's s2-64k-84) minus 8,8 read by Triton
Both paired on the 96 graded 64K items, with stage 2's statistics (stage2_stats: P6's scorer, 90% percentile bootstrap,
10,000 resamples, seed 20261102, usable > -10, equivalent within +-5, exact McNemar) and the registered rules.

Descriptive, not registered: text identity between the arms, the discordant split among differing texts, and the
bound |K1| <= share of items whose 8,8-Triton and 8,8-fd75d texts differ (an item's score can change only if its text
does). Also H2s = K1 + K2 exactly on these items. Laptop only; the journals are small JSON.

  python fd75d_arm_analyze.py --stage2 J2 --arm JA --items DIR [--out results.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_analyze as A  # noqa: E402

TRITON, FD75D, SPLIT, NOISE = "s2-64k-88-triton", "s2-64k-88", "s2-64k-84", "s2-64k-88-noise"


def texts(rows, arm):
    return {r["uid"]: r.get("text") or "" for r in rows if r.get("kind") == "item" and r["arm"] == arm
            and r["status"] == "ok"}


def identity(rows, arms, a, b):
    ta, tb = texts(rows, a), texts(rows, b)
    common = sorted(u for u in ta if u in tb)
    differ = [u for u in common if ta[u] != tb[u]]
    sa, sb = arms.scores[a], arms.scores[b]
    changed = [u for u in differ if sa[u] != sb[u]]
    return {"a": a, "b": b, "n": len(common), "identical": len(common) - len(differ), "differ": len(differ),
            "correctness_changed": len(changed),
            "a_only_right": sum(1 for u in changed if sa[u] and not sb[u]),
            "b_only_right": sum(1 for u in changed if sb[u] and not sa[u]),
            "score_change_without_text_change": sum(1 for u in common if ta[u] == tb[u] and sa[u] != sb[u])}


def judge(rows, items_dir):
    arms = A.Arms(rows, items_dir)
    k1, k2, h2s = (A.comparison(arms, TRITON, FD75D), A.comparison(arms, SPLIT, TRITON),
                   A.comparison(arms, SPLIT, FD75D))
    rule = ("K1's interval lies above +5: fd75d loses accuracy at 64K" if k1["ci90"][0] > 5 else
            "K1 is equivalent: the kernel is not what separates the arms; H2s's lead is the value bits plus noise"
            if k1["equivalent"] else
            "neither registered case: K1's interval is not above +5 and not within +-5")
    pred = {"K1_point_above_0": "holds" if k1["diff_points"] > 0 else "refuted",
            "K2_equivalent": "holds" if k2["equivalent"] else "refuted"}
    ids = {"triton_vs_fd75d": identity(rows, arms, TRITON, FD75D), "split_vs_triton": identity(rows, arms, SPLIT, TRITON),
           "split_vs_fd75d": identity(rows, arms, SPLIT, FD75D)}
    if NOISE in arms.scores:
        ids["triton_vs_fd75d_rerun_16"] = identity(rows, arms, TRITON, NOISE)
    bound = 100.0 * ids["triton_vs_fd75d"]["differ"] / ids["triton_vs_fd75d"]["n"]
    return {"K1": k1, "K2": k2, "H2s_check": h2s,
            "decomposition": {"K1_plus_K2": k1["diff_points"] + k2["diff_points"], "H2s": h2s["diff_points"]},
            "rule": rule, "predictions": pred,
            "descriptive": {"text_identity": ids, "K1_bound_points": bound,
                            "K1_within_bound": abs(k1["diff_points"]) <= bound + 1e-9},
            "arms": {a: A.arm_secondary(arms, a) for a in (TRITON, FD75D, SPLIT)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = judge(A.load_rows(args.stage2) + A.load_rows(args.arm), args.items)
    text = json.dumps(res, indent=1, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    short = {k: {x: res[k][x] for x in ("n", "acc_a", "acc_b", "diff_points", "ci90", "usable", "equivalent",
                                        "n10", "n01", "mcnemar_p")} for k in ("K1", "K2", "H2s_check")}
    print(json.dumps({**short, "rule": res["rule"], "predictions": res["predictions"],
                      "decomposition": res["decomposition"], "descriptive": res["descriptive"]}, indent=1))


if __name__ == "__main__":
    main()
