#!/usr/bin/env python3
"""Analysis of the P8 confirmation speed run, written before any of its rows exist
(research/prereg/2026-11-04-p8-confirmation-speed.md, section "Analysis").

Pooled speed per condition: total decoded tokens over total decode wall time, over its prompts and rounds; per-round
pooled speeds the same way within each round. Every comparison below pairs two conditions of this one session.

  python3 analyze_confirm.py ROWS.jsonl [--plan plan.json] [--quality-dir DIR --reference REF.jsonl] > report.md
Writes results.json beside ROWS.jsonl. Stdlib only.
"""
import argparse
import collections
import json
import math
import os
import re

F2 = {"w13-165-nd-0": 47.53, "w13-165-mtp-0": 71.61, "w13-165-nd-28k": 43.88, "w13-165-mtp-28k": 66.57}  # final2.md
FLOOR = {"nd": 3.6, "mtp": 5.5}   # the cross-session floor (testsuite/results/ideas/four-resources-formal/audit.json)


def load(rows_path):
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for line in open(rows_path):
        if not line.strip():
            continue
        r = json.loads(line)
        m = re.fullmatch(r"confirm-(.+)-r(\d)", r.get("tag", ""))
        if not m:
            continue
        cond, rnd = m.group(1), int(m.group(2))
        acc[cond][rnd][0] += r["new_tokens"]
        acc[cond][rnd][1] += r["decode_wall_s"]
    out = {}
    for cond, rounds in acc.items():
        tok = sum(v[0] for v in rounds.values())
        sec = sum(v[1] for v in rounds.values())
        out[cond] = {"pooled": tok / sec, "rounds": {k: v[0] / v[1] for k, v in sorted(rounds.items())}}
    return out


def versus(s, a, b):
    """a against b: pooled change and the per-round record, on the rounds both ran."""
    if a not in s or b not in s:
        return None
    common = sorted(set(s[a]["rounds"]) & set(s[b]["rounds"]))
    per = [100 * (s[a]["rounds"][k] / s[b]["rounds"][k] - 1) for k in common]
    return {"a": a, "b": b, "pooled_pct": 100 * (s[a]["pooled"] / s[b]["pooled"] - 1),
            "ratio": s[a]["pooled"] / s[b]["pooled"], "per_round_pct": per, "rounds": len(common),
            "wins": sum(1 for x in per if x > 0), "faster_every_round": bool(per) and all(x > 0 for x in per),
            "slower_every_round": bool(per) and all(x < 0 for x in per)}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def quality(qdir, reference, label):
    rows = {}
    path = os.path.join(qdir, label, "gsm8k_500.jsonl")
    if not os.path.exists(path):
        return None
    for line in open(path):
        r = json.loads(line)
        rows[r["id"]] = r
    ref = {}
    for line in open(reference):
        r = json.loads(line)
        ref[r["id"]] = r
    ids = [i for i in rows if i in ref and "error" not in rows[i] and "error" not in ref[i]]
    a = sum(1 for i in ids if rows[i].get("correct"))
    r_ok = sum(1 for i in ids if ref[i].get("correct"))
    b_only = sum(1 for i in ids if ref[i].get("correct") and not rows[i].get("correct"))
    c_only = sum(1 for i in ids if rows[i].get("correct") and not ref[i].get("correct"))
    return {"n": len(ids), "acc": 100 * a / len(ids), "ref_acc": 100 * r_ok / len(ids),
            "diff_pts": 100 * (a - r_ok) / len(ids), "ref_only": b_only, "buun_only": c_only,
            "mcnemar_p": mcnemar(b_only, c_only)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows")
    ap.add_argument("--quality-dir")
    ap.add_argument("--reference", help="the stock EXL3 MTP lane's gsm8k_500.jsonl (TURING-exl3-mtp)")
    args = ap.parse_args()
    s = load(args.rows)
    res = {"pooled": s, "f2": {}, "comparisons": {}}
    for cond, f2 in F2.items():
        if cond in s:
            d = 100 * (s[cond]["pooled"] / f2 - 1)
            mode = "mtp" if "-mtp-" in cond else "nd"
            res["f2"][cond] = {"now": s[cond]["pooled"], "f2": f2, "change_pct": d, "within_floor": abs(d) <= FLOOR[mode]}
    pairs = {  # name: (a, b); a is the w13 arm unless stated
        "over_stock_nd_0": ("w13-165-nd-0", "stock165-nd-0"), "over_stock_mtp_0": ("w13-165-mtp-0", "stock165-mtp-0"),
        "over_stock_nd_28k": ("w13-165-nd-28k", "stock165-nd-28k"),
        "over_stock_mtp_28k": ("w13-165-mtp-28k", "stock165-mtp-28k"),
        "stock166_vs_165_nd": ("stock166-nd-0", "stock165-nd-0"), "stock166_vs_165_mtp": ("stock166-mtp-0", "stock165-mtp-0"),
        "w13_166_vs_165_nd": ("w13-166-nd-0", "w13-165-nd-0"), "w13_166_vs_165_mtp": ("w13-166-mtp-0", "w13-165-mtp-0"),
        "over_raf_0": ("w13-165-mtp-0", "raf44-mtp-0"), "over_raf_28k": ("w13-165-mtp-28k", "raf44-mtp-28k"),
        "over_nif_xxs_0": ("w13-165-nd-0", "nif-xxs-0"), "over_nif_xxs_28k": ("w13-165-nd-28k", "nif-xxs-28k"),
        "over_nif_s_0": ("w13-165-nd-0", "nif-s-0"), "over_nif_s_28k": ("w13-165-nd-28k", "nif-s-28k"),
        "over_lcpp_df_0": ("w13-165-mtp-0", "lcpp9-xxs-df-0"), "over_lcpp_df_28k": ("w13-165-mtp-28k", "lcpp9-xxs-df-28k"),
        "over_lcpp_mtp_0": ("w13-165-mtp-0", "lcpp9-s-mtp-0"), "over_lcpp_mtp_28k": ("w13-165-mtp-28k", "lcpp9-s-mtp-28k"),
        "over_lcpp_nd_0": ("w13-165-nd-0", "lcpp9-xxs-0"), "over_lcpp_nd_28k": ("w13-165-nd-28k", "lcpp9-xxs-28k"),
        "over_ik_xxs_0": ("w13-165-nd-0", "ik-xxs-0"), "over_ik_s_0": ("w13-165-nd-0", "ik-s-0"),
        "over_ik_mtp_0": ("w13-165-mtp-0", "ik-s-mtp-0"), "ik_vs_lcpp_nd_0": ("ik-xxs-0", "lcpp9-xxs-0"),
        "over_buun_nd_0": ("w13-165-nd-0", "buun-nd-0"), "over_buun_mtp_0": ("w13-165-mtp-0", "buun-mtp-0"),
    }
    for name, (a, b) in pairs.items():
        v = versus(s, a, b)
        if v:
            res["comparisons"][name] = v
    # the "fastest" wording: w13's arm is faster than every other engine's arm of the same drafting and fill, every round
    fastest = {}
    for mode, fill in (("nd", "0"), ("mtp", "0"), ("nd", "28k"), ("mtp", "28k")):
        ours = f"w13-165-{mode}-{fill}"
        rivals = [k for k in s if not k.startswith(("w13-", "stock")) and k.endswith("-" + fill)
                  and (("mtp" in k or "df" in k) == (mode == "mtp"))]
        rec = {r: versus(s, ours, r) for r in rivals}
        fastest[f"{mode}-{fill}"] = {"rivals": sorted(rivals),
                                     "holds": bool(rivals) and all(v and v["faster_every_round"] for v in rec.values())}
    res["fastest"] = fastest
    if args.quality_dir and args.reference:
        res["quality_buun"] = quality(args.quality_dir, args.reference, "TURING-buun-exl3-mtp")
    with open(os.path.join(os.path.dirname(os.path.abspath(args.rows)), "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("# P8 confirmation speed run: pooled decode tokens/s\n")
    print("| condition | pooled | per round |\n|---|---|---|")
    for cond in sorted(s):
        print(f"| {cond} | {s[cond]['pooled']:.2f} | {', '.join(f'{v:.2f}' for v in s[cond]['rounds'].values())} |")
    print("\n## Against F2\n")
    for cond, v in res["f2"].items():
        print(f"- {cond}: {v['now']:.2f} against {v['f2']} ({v['change_pct']:+.1f}%), within the floor: {v['within_floor']}")
    print("\n## Comparisons (a against b; per-round wins)\n")
    for name, v in res["comparisons"].items():
        print(f"- {name}: {v['a']} / {v['b']} = {v['ratio']:.3f} ({v['pooled_pct']:+.1f}%), "
              f"{v['wins']} of {v['rounds']} rounds faster; per round {', '.join(f'{x:+.1f}%' for x in v['per_round_pct'])}")
    print("\n## The fastest wording\n")
    for k, v in fastest.items():
        print(f"- {k}: holds {v['holds']} against {', '.join(v['rivals']) or 'no rival arm'}")
    if res.get("quality_buun"):
        q = res["quality_buun"]
        print(f"\n## buun GSM8K check\n\n- {q['n']} items: buun {q['acc']:.1f}% against {q['ref_acc']:.1f}% "
              f"({q['diff_pts']:+.1f} points; {q['buun_only']} right only with buun, {q['ref_only']} only with the "
              f"stock EXL3 lane; exact McNemar p {q['mcnemar_p']:.3g})")


if __name__ == "__main__":
    main()
