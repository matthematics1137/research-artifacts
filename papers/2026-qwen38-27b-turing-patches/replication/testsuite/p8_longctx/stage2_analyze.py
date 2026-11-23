#!/usr/bin/env python3
"""P8 long context, stages 2-3: score the journals and judge them against research/prereg/2026-11-01-p8-longctx-stages2-3.md.

  calib   the calibration rule: per family, the level whose 8,8 accuracy at 64K is closest to 65% (harder on a tie);
          --arm calib-64k-88 (the first calibration) or calib2-64k-88 (the second, multivalue: Amendment 2)
  stage2  H2 (4,4 - 8,8 at 64K, primary), H1 (8,8 - fp16 at 32K), H3 (4,4 - fp16 at 32K), H2s if the rescue ran:
          paired 90% bootstrap intervals, usable (lower bound > -10) and equivalent (within +-5), exact McNemar per
          family and pooled, Holm on H1 and H3; the noise check; per-family accuracy; the cliff set; R1-R3
  stage3  the probe, accuracy by length with Wilson intervals, usable context and P3b, on the items shared with 32K

Scores come from P6's scorer on the journaled answers (stage2_stats.score); only items with status ok are scored, and
excluded items are listed. Paired comparisons use the items both arms completed.

  python stage2_analyze.py calib  --journal J --items DIR [--arm calib2-64k-88] [--out results.json]
  python stage2_analyze.py stage2 --journal J --items DIR [--out results.json] [--markdown tables.md]
  python stage2_analyze.py stage3 --journal J3 --stage2 J2 --items DIR [--out results.json] [--markdown tables.md]
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_stats as T  # noqa: E402

ARMS2 = {"h2": ("s2-64k-44", "s2-64k-88"), "h1": ("s2-32k-88", "s2-32k-fp16"), "h3": ("s2-32k-44", "s2-32k-fp16"),
         "h2s": ("s2-64k-84", "s2-64k-88")}
NOISE = ("s2-64k-88-noise", "s2-64k-88")


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(items_dir, name):
    with open(os.path.join(items_dir, name), encoding="utf-8") as f:
        return {it["uid"]: it for it in (json.loads(line) for line in f if line.strip())}


class Arms:
    """Per arm: scored items (uid -> 0/1), excluded items, and the item rows."""

    def __init__(self, rows, items_dir):
        self.rows = rows
        self.files = {r["arm"]: r["items_file"] for r in rows if r.get("kind") == "arm_plan"}
        self.items = {}
        self.scores, self.excluded, self.item_rows = {}, {}, {}
        for r in rows:
            if r.get("kind") != "item":
                continue
            arm = r["arm"]
            if arm not in self.items:
                self.items[arm] = load_items(items_dir, self.files[arm])
            self.item_rows.setdefault(arm, []).append(r)
            if r["status"] == "ok":
                self.scores.setdefault(arm, {})[r["uid"]] = T.score(r.get("text") or "", self.items[arm][r["uid"]])
            else:
                self.excluded.setdefault(arm, []).append({"uid": r["uid"], "status": r["status"],
                                                          "error": (r.get("error") or "")[:200]})

    def has(self, arm):
        return arm in self.scores

    def family_of(self, arm, uid):
        return self.items[arm][uid]["family"]

    def paired(self, a, b, family=None):
        common = [u for u in self.scores.get(a, {}) if u in self.scores.get(b, {})
                  and (family is None or self.family_of(a, u) == family)]
        common.sort()
        return common, [self.scores[a][u] for u in common], [self.scores[b][u] for u in common]


def acc(scores):
    k, n = sum(scores), len(scores)
    lo, hi = T.wilson(k, n)
    return {"k": k, "n": n, "acc": None if n == 0 else k / n, "wilson95": [lo, hi]}


def comparison(arms, a, b):
    common, sa, sb = arms.paired(a, b)
    if not common:
        return None
    mean, lo, hi = T.paired_interval(sa, sb)
    _, lo95, hi95 = T.paired_interval(sa, sb, level=0.95)   # descriptive, for the living map; verdicts use the 90%
    n10, n01, p = T.mcnemar(sa, sb)
    fam = {}
    for f in sorted({arms.family_of(a, u) for u in common}):
        c, fa, fb = arms.paired(a, b, f)
        if c:
            m10, m01, pf = T.mcnemar(fa, fb)
            fam[f] = {"n": len(c), "acc_a": sum(fa) / len(c), "acc_b": sum(fb) / len(c),
                      "diff_points": 100.0 * (sum(fa) - sum(fb)) / len(c), "n10": m10, "n01": m01, "mcnemar_p": pf}
    return {"a": a, "b": b, "n": len(common), "acc_a": sum(sa) / len(common), "acc_b": sum(sb) / len(common),
            "diff_points": mean, "ci90": [lo, hi], "ci95_points": [lo95, hi95], **T.verdict(lo, hi),
            "n10": n10, "n01": n01, "mcnemar_p": p,
            "by_family": fam}


def arm_secondary(arms, arm):
    rows = [r for r in arms.item_rows.get(arm, []) if r["status"] == "ok"]
    sess = [r for r in arms.rows if r.get("kind") == "session_end" and r["arm"] == arm]
    if not rows:
        return None
    pt = sum(r.get("prompt_tokens") or 0 for r in rows)
    ptime = sum(r.get("prompt_time_s") or 0 for r in rows)
    ct = [r.get("completion_tokens") or 0 for r in rows]
    ctime = sum(r.get("completion_time_s") or 0 for r in rows)
    mism = [r["uid"] for r in rows if r.get("prompt_tokens") != r.get("prompt_tokens_built")]
    by_fam = {}
    for f in sorted({arms.family_of(arm, r["uid"]) for r in rows}):
        by_fam[f] = acc([arms.scores[arm][r["uid"]] for r in rows if arms.family_of(arm, r["uid"]) == f])
    return {"items_ok": len(rows), "excluded": arms.excluded.get(arm, []), "accuracy": acc(list(arms.scores[arm].values())),
            "by_family": by_fam, "prefill_tok_s_pooled": pt / ptime if ptime else None,
            "decode_tok_s_pooled": sum(ct) / ctime if ctime else None,
            "answer_tokens": {"mean": statistics.mean(ct), "median": statistics.median(ct), "max": max(ct)},
            "finish_length": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "prompt_token_mismatches": mism, "peak_mib": max([s.get("peak_mib") or 0 for s in sess] or [None]),
            "gpu_s": round(sum(s.get("wall_s") or 0 for s in sess), 1), "sessions": len(sess),
            "mean_item_s": statistics.mean(r["wall_s"] for r in rows)}


def calib(rows, items_dir, arm="calib-64k-88"):
    arms = Arms(rows, items_dir)
    out = {"arm": arm, "families": {}}
    for f in sorted({it["family"] for uid, it in arms.items.get(arm, {}).items() if uid in arms.scores.get(arm, {})}):
        by_level = {}
        for uid, s in arms.scores.get(arm, {}).items():
            it = arms.items[arm][uid]
            if it["family"] == f:
                by_level.setdefault(it["level"], []).append(s)
        accs = {lv: sum(v) / len(v) for lv, v in by_level.items()}
        chosen = T.select_level(accs) if accs else None
        out["families"][f] = {"by_level": {str(lv): acc(v) for lv, v in sorted(by_level.items())},
                              "chosen": chosen, "chosen_acc": accs.get(chosen),
                              "outside_30_90": None if chosen is None else not (0.30 <= accs[chosen] <= 0.90)}
    out["excluded"] = arms.excluded.get(arm, [])
    out["secondary"] = arm_secondary(arms, arm)
    return out


def stage2(rows, items_dir):
    arms = Arms(rows, items_dir)
    res = {"comparisons": {}, "arms": {}, "noise": None, "rules": {}, "cliff": {}}
    for key, (a, b) in ARMS2.items():
        if arms.has(a) and arms.has(b):
            res["comparisons"][key] = comparison(arms, a, b)
    h1, h3 = res["comparisons"].get("h1"), res["comparisons"].get("h3")
    if h1 and h3:
        adj = T.holm({"h1": h1["mcnemar_p"], "h3": h3["mcnemar_p"]})
        h1["mcnemar_p_holm"], h3["mcnemar_p_holm"] = adj["h1"], adj["h3"]
    for arm in sorted(arms.scores):
        res["arms"][arm] = arm_secondary(arms, arm)
    if arms.has(NOISE[0]) and arms.has(NOISE[1]):
        common, sn, sr = arms.paired(*NOISE)
        k = sum(1 for x, y in zip(sn, sr) if x != y)
        lo, hi = T.wilson(k, len(common))
        noise = {"n": len(common), "discordant": k, "rate": k / len(common), "wilson95": [lo, hi]}
        if arms.has("s2-64k-44"):
            c32, s44, s88 = [], [], []
            for u in common:
                if u in arms.scores["s2-64k-44"]:
                    c32.append(u)
                    s44.append(arms.scores["s2-64k-44"][u])
                    s88.append(arms.scores["s2-64k-88"][u])
            d32 = sum(1 for x, y in zip(s44, s88) if x != y)
            call, a44, a88 = arms.paired("s2-64k-44", "s2-64k-88")
            dall = sum(1 for x, y in zip(a44, a88) if x != y)
            noise["cache_discordance_same_items"] = {"n": len(c32), "discordant": d32,
                                                     "rate": d32 / len(c32) if c32 else None}
            noise["cache_discordance_all"] = {"n": len(call), "discordant": dall, "rate": dall / len(call) if call else None}
            noise["cache_inside_noise_interval"] = {
                "same_items": None if not c32 else lo <= d32 / len(c32) <= hi,
                "all": None if not call else lo <= dall / len(call) <= hi}
        res["noise"] = noise
    for arm in sorted(arms.scores):
        if arm.startswith("cliff-"):
            res["cliff"][arm] = res["arms"][arm]
    res["skipped"] = [{k: r.get(k) for k in ("arm", "why", "estimate_s", "used_s", "budget_s")}
                      for r in rows if r.get("kind") == "arm_skipped"]
    h2, h2s = res["comparisons"].get("h2"), res["comparisons"].get("h2s")
    r = res["rules"]
    if h2:
        r["R1"] = {"applies": h2["usable"], "mode": "4,4" if h2["usable"] else None,
                   "equivalent_wording_allowed": h2["usable"] and h2["equivalent"]}
        r["R2"] = {"rescue_needed": not h2["usable"]}
        if not h2["usable"]:
            r["R2"].update({"h2s_usable": None if not h2s else h2s["usable"],
                            "mode": None if not h2s else ("8,4" if h2s["usable"] else None)})
    if h1:
        r["R3"] = {"follow_up_for_8bit_default": not h1["equivalent"]}
    adopted = (r.get("R1") or {}).get("mode") or (r.get("R2") or {}).get("mode")
    r["adopted_mode"] = adopted
    r["stage3_runs"] = adopted is not None
    return res


def stage3(rows3, rows2, items_dir, mode):
    """Stage 3 arms are named s3-<L/1024>k (the adopted mode); stage 2's base arms s2-32k-<fmt> and s2-64k-<fmt>."""
    a3, a2 = Arms(rows3, items_dir), Arms(rows2, items_dir)
    probe = [{"step": r["step"], "verdict": r["verdict"], "peak_mib": r.get("peak_mib"), "prefill_s": r.get("prompt_time_s"),
              "reason": r.get("reason")} for r in rows3 if r.get("kind") == "result"]
    fmt = mode.replace(",", "")
    base32, base64 = f"s2-32k-{fmt}", f"s2-64k-{fmt}"
    curve = []
    for arm, scores in [(base32, a2.scores.get(base32, {})), (base64, a2.scores.get(base64, {}))] + \
            sorted(a3.scores.items(), key=lambda kv: int(kv[0].split("-")[1].rstrip("k"))):
        if not scores:
            continue
        length = int(arm.split("-")[1].rstrip("k"))
        row = {"arm": arm, "length_k": length, **acc(list(scores.values()))}
        if arm != base32 and a2.scores.get(base32):
            common = sorted(u for u in scores if u in a2.scores[base32])
            if common:
                sl = [scores[u] for u in common]
                s32 = [a2.scores[base32][u] for u in common]
                mean, lo, hi = T.paired_interval(sl, s32)
                row["gap_vs_32k"] = {"n": len(common), "points": mean, "ci90": [lo, hi], "within_10": mean >= -10}
        curve.append(row)
    usable = None
    for row in curve:
        if row["arm"] == base32 or row.get("gap_vs_32k", {}).get("within_10"):
            usable = row["length_k"]
        else:
            break
    p3b = next((row["gap_vs_32k"] for row in curve if row["length_k"] == 128 and "gap_vs_32k" in row), None)
    return {"mode": mode, "probe": probe, "curve": curve, "usable_context_k": usable,
            "P3b": None if p3b is None else ("holds" if abs(p3b["points"]) <= 10 else "refuted"),
            "P3b_gap": p3b, "arms": {arm: arm_secondary(a3, arm) for arm in a3.scores},
            "excluded": a3.excluded}


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def markdown2(res):
    out = ["| Comparison | n | Accuracy (a / b) | Difference, points | 90% interval | Usable | Equivalent | "
           "Discordant (a only / b only) | McNemar p | Holm p |", "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"h2": "H2: `4,4` - `8,8`, 64K", "h1": "H1: `8,8` - `fp16`, 32K", "h3": "H3: `4,4` - `fp16`, 32K",
             "h2s": "H2s: `8,4` - `8,8`, 64K"}
    for key in ("h2", "h1", "h3", "h2s"):
        c = res["comparisons"].get(key)
        if not c:
            continue
        holm = f"{c['mcnemar_p_holm']:.3g}" if "mcnemar_p_holm" in c else "-"
        out.append(f"| {names[key]} | {c['n']} | {pct(c['acc_a'])} / {pct(c['acc_b'])} | {c['diff_points']:+.1f} | "
                   f"[{c['ci90'][0]:+.1f}, {c['ci90'][1]:+.1f}] | {'yes' if c['usable'] else 'no'} | "
                   f"{'yes' if c['equivalent'] else 'no'} | {c['n10']} / {c['n01']} | {c['mcnemar_p']:.3g} | {holm} |")
    out += ["", "| Comparison | Family | n | Accuracy (a / b) | Difference, points | Discordant | McNemar p |",
            "|---|---|---|---|---|---|---|"]
    for key in ("h2", "h1", "h3", "h2s"):
        c = res["comparisons"].get(key)
        if not c:
            continue
        for f, v in c["by_family"].items():
            out.append(f"| {key.upper()} | {f} | {v['n']} | {pct(v['acc_a'])} / {pct(v['acc_b'])} | "
                       f"{v['diff_points']:+.1f} | {v['n10']} / {v['n01']} | {v['mcnemar_p']:.3g} |")
    out += ["", "| Arm | Items ok | Excluded | Accuracy (Wilson 95%) | By family | Prefill tok/s | "
            "Decode tok/s | Answer tokens (mean / max) | Peak MiB | GPU time |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, s in res["arms"].items():
        if not s:
            continue
        a = s["accuracy"]
        fams = ", ".join(f"{f} {v['k']}/{v['n']}" for f, v in s["by_family"].items())
        out.append(f"| `{arm}` | {s['items_ok']} | {len(s['excluded'])} | {pct(a['acc'])} "
                   f"[{pct(a['wilson95'][0])}, {pct(a['wilson95'][1])}] | {fams} | "
                   f"{s['prefill_tok_s_pooled'] or 0:.0f} | {s['decode_tok_s_pooled'] or 0:.1f} | "
                   f"{s['answer_tokens']['mean']:.1f} / {s['answer_tokens']['max']} | {s['peak_mib']} | "
                   f"{s['gpu_s'] / 3600:.2f} h |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("calib", "stage2", "stage3"))
    ap.add_argument("--journal", required=True)
    ap.add_argument("--stage2")
    ap.add_argument("--mode")
    ap.add_argument("--arm", default="calib-64k-88")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out")
    ap.add_argument("--markdown")
    args = ap.parse_args()
    rows = load_rows(args.journal)
    if args.what == "calib":
        res = calib(rows, args.items, args.arm)
    elif args.what == "stage2":
        res = stage2(rows, args.items)
        if args.markdown:
            with open(args.markdown, "w") as f:
                f.write(markdown2(res))
    else:
        res = stage3(rows, load_rows(args.stage2), args.items, args.mode)
    text = json.dumps(res, indent=1, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
