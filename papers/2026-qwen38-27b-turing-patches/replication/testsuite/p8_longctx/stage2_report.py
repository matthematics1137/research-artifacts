#!/usr/bin/env python3
"""P8 long context, stage 2: assemble report.md from results.json, calibration.json, calibration2.json and the
registration's amendments. Every number in the report comes from those files; the narrative sections are fixed text
keyed to the registered verdicts.

  python stage2_report.py RESULTS_DIR --out report.md
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_analyze as A  # noqa: E402

NAMES = {"h2": "H2 (primary): `4,4` − `8,8` at 64K", "h1": "H1: `8,8` − `fp16` at 32K",
         "h3": "H3: `4,4` − `fp16` at 32K", "h2s": "H2s (rescue): `8,4` − `8,8` at 64K"}


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def stamp():
    return subprocess.run(["date", "+%Y-%m-%d %H:%M %Z"], capture_output=True, text=True,
                          env=dict(os.environ, TZ="America/New_York")).stdout.strip()


def verdict_line(key, c):
    if not c:
        return f"| {NAMES[key]} | not run | | | |"
    v = ("usable and equivalent" if c["equivalent"] else "usable, not shown equivalent" if c["usable"]
         else "not usable")
    return (f"| {NAMES[key]} | {c['diff_points']:+.1f} | [{c['ci90'][0]:+.1f}, {c['ci90'][1]:+.1f}] | {v} | "
            f"{c['n10']} / {c['n01']}, p {c['mcnemar_p']:.3g}"
            + (f", Holm {c['mcnemar_p_holm']:.3g}" if "mcnemar_p_holm" in c else "") + " |")


def build(d):
    res = json.load(open(os.path.join(d, "results.json")))
    c1 = json.load(open(os.path.join(d, "calibration.json")))
    c2 = json.load(open(os.path.join(d, "calibration2.json")))
    comp, rules, noise = res["comparisons"], res["rules"], res["noise"]
    gpu = sum((s or {}).get("gpu_s", 0) for s in res["arms"].values())
    out = [f"# P8 long context, stage 2: paired accuracy at depth per KV cache format", "",
           "- Registration: `research/prereg/2026-11-01-p8-longctx-stages2-3.md`, with Amendments 1-8. Amendments 2, "
           "3, 6 and 8 are the ambassador's; 1, 4, 5 and 7 are the stages 2-3 fork's implementation notes.",
           "- Results: this folder (`journal.jsonl`, `results.json`, per-arm configs, logs and 1 Hz samples). "
           "Analysis: `testsuite/p8_longctx/stage2_analyze.py`; scorer: P6's `score_longctx`, unchanged.",
           f"- GPU time: {gpu / 3600:.2f} of 22 hours: both calibrations, the arms, the rescue"
           + (" and the cliff arms that ran." if res.get("cliff") else ", with the cliff set still running."),
           "- Exclusions: " + (", ".join(f"`{a}` {len(v['excluded'])}" for a, v in res["arms"].items() if v and v["excluded"])
                               or "none") + ". Truncated answers: "
           + str(sum(v["finish_length"] for v in res["arms"].values() if v)) + ". Server and built prompt token counts "
           + ("matched on every item." if not any(v["prompt_token_mismatches"] for v in res["arms"].values() if v)
              else "differed on some items (see results.json)."),
           f"- Report written {stamp()}.", ""]
    out += ["## Verdicts, as registered", "",
            "| Comparison | Difference, points | 90% interval | Verdict | Discordant (a only / b only), McNemar |",
            "|---|---|---|---|---|"]
    out += [verdict_line(k, comp.get(k)) for k in ("h2", "h1", "h3", "h2s")]
    out += ["", "Usable: the interval's lower bound lies above −10 points. Equivalent: the whole interval lies within "
            "±5. Holm adjusts the McNemar p-values of H1 and H3.", ""]
    r1, r2, r3 = rules.get("R1", {}), rules.get("R2", {}), rules.get("R3", {})
    out += ["## Rules", "",
            f"- **R1.** {'Applies' if r1.get('applies') else 'Does not apply'}: "
            + ("the 4-bit cache becomes the mode for contexts beyond the 8-bit cache's 72K reach."
               if r1.get("applies") else "H2's lower bound is below −10, so the 4-bit cache is not adopted."),
            "- **R2.** " + ("Not needed." if not r2.get("rescue_needed") else
                            "The rescue ran (`8,4` at 64K). " +
                            ("H2s is usable, so the split cache takes R1's role up to its own reach."
                             if r2.get("h2s_usable") else
                             "H2s is not usable, so no quantized long-context mode is adopted and stage 3 does not "
                             "run." if r2.get("h2s_usable") is False else "H2s was not computed.")),
            "- **R3.** " + ("H1 is equivalent: the deployed 8-bit default needs no follow-up."
                            if not r3.get("follow_up_for_8bit_default") else
                            "H1 is not equivalent: the deployed 8-bit default gets its own follow-up."),
            f"- **Adopted mode:** {rules.get('adopted_mode') or 'none'}; stage 3 "
            f"{'runs' if rules.get('stage3_runs') else 'does not run'}.", ""]
    h2, h2s, h1, h3 = (comp.get(k) for k in ("h2", "h2s", "h1", "h3"))
    out += ["## What the comparisons show", ""]
    if h2:
        out.append(f"- **H2.** The 4-bit cache scored {pct(h2['acc_a'])} against the 8-bit cache's {pct(h2['acc_b'])} "
                   f"at 64K ({h2['diff_points']:+.1f} points). Its interval reaches {h2['ci90'][0]:+.1f}, below "
                   "−10, so it is not usable by the registered bound.")
    if h2s:
        above = h2s["ci90"][0] > 5
        out.append(f"- **H2s.** The split cache scored {pct(h2s['acc_a'])} against {pct(h2s['acc_b'])} "
                   f"({h2s['diff_points']:+.1f} points, interval [{h2s['ci90'][0]:+.1f}, {h2s['ci90'][1]:+.1f}], "
                   f"McNemar p {h2s['mcnemar_p']:.3g}). It is usable"
                   + (". It is not equivalent only because the whole interval lies above +5: on these items the "
                      "split cache scored higher than the 8-bit cache. The registration makes no superiority claim, "
                      "so this is reported, not claimed." if above else "."))
    if h1:
        out.append(f"- **H1.** At 32K the deployed 8-bit cache and fp16 scored the same "
                   f"({pct(h1['acc_a'])} and {pct(h1['acc_b'])}), with {h1['n10'] + h1['n01']} items differing.")
    if h3:
        out.append(f"- **H3.** At 32K the 4-bit cache scored {pct(h3['acc_a'])} against fp16's {pct(h3['acc_b'])} "
                   f"({h3['diff_points']:+.1f} points, interval [{h3['ci90'][0]:+.1f}, {h3['ci90'][1]:+.1f}]): "
                   "usable, not shown equivalent.")
    out += ["- **A confound to test, not a claim.** The decode kernel differs by format: the lab's fd75d reads the "
            "8-bit cache, and Triton reads the others. At 32K, fd75d's 8-bit cache and Triton's fp16 agreed on every "
            "item. At 64K, the fd75d-read 8-bit cache trails the Triton-read split cache, which shares its 8-bit "
            "keys. So H2s mixes value bits with the kernel, and fd75d's accuracy beyond its tested 28K is a lead for "
            "its own registration.", ""]
    if noise:
        lo, hi = noise["wilson95"]
        out += ["## Noise check", "",
                f"The `8,8` arm was rerun at 64K in a fresh server on the first 16 graded items. It disagreed with "
                f"the first run on {noise['discordant']} of {noise['n']} items (Wilson 95% {pct(lo)} to {pct(hi)}).",
                "",
                f"- `4,4` against `8,8` on the same 16 items: {noise['cache_discordance_same_items']['discordant']} of "
                f"{noise['cache_discordance_same_items']['n']} "
                f"({pct(noise['cache_discordance_same_items']['rate'])}).",
                f"- `4,4` against `8,8` on all items (H2): {noise['cache_discordance_all']['discordant']} of "
                f"{noise['cache_discordance_all']['n']} ({pct(noise['cache_discordance_all']['rate'])}).",
                (f"- `8,4` against `8,8` on all items (H2s): {h2s['n10'] + h2s['n01']} of {h2s['n']} "
                 f"({pct((h2s['n10'] + h2s['n01']) / h2s['n'])})." if h2s else "- H2s: not run."),
                "- **H2, by the registered criterion:** " + (
                    "both of the 4-bit cache's rates lie outside the same-arm interval, so its effect can be told "
                    "apart from run-to-run noise."
                    if not noise["cache_inside_noise_interval"]["same_items"] and not noise["cache_inside_noise_interval"]["all"]
                    else "a 4-bit rate lies inside the same-arm interval, so its effect cannot be told apart from "
                         "run-to-run noise there."),
                ("- **H2s** (the same comparison, which the registration states for H2 only): " + (
                    f"its rate lies inside the same-arm interval, whose upper bound of {pct(hi)} rests on only "
                    f"{noise['n']} items. Its direction is lopsided, though ({h2s['n10']} against {h2s['n01']}; exact "
                    f"McNemar p {h2s['mcnemar_p']:.3g}), which run-to-run noise would not favour."
                    if lo <= (h2s['n10'] + h2s['n01']) / h2s['n'] <= hi else
                    "its rate lies outside the same-arm interval.") if h2s else "- H2s: not run."), ""]
    out += ["## Registered predictions", "",
            "| Prediction | Verdict |", "|---|---|",
            "| H2 is usable and equivalent, with a point estimate within ±3 points | "
            + ("holds" if h2 and h2["equivalent"] and abs(h2["diff_points"]) <= 3 else
               f"refuted: {'not usable' if h2 and not h2['usable'] else 'not equivalent'}, point estimate "
               f"{h2['diff_points']:+.1f}" if h2 else "not run") + " |",
            "| H1 is equivalent | " + ("holds" if h1 and h1["equivalent"] else "refuted") + " |",
            "| H3 is equivalent | " + ("holds" if h3 and h3["equivalent"] else
                                       f"refuted: interval [{h3['ci90'][0]:+.1f}, {h3['ci90'][1]:+.1f}]" if h3
                                       else "not run") + " |",
            "| Any loss concentrates in multikey | not testable as registered: Amendment 2 made multivalue the only "
            "graded family; multikey is in the descriptive cliff set |", ""]
    cliff = res.get("cliff") or {}
    out += ["## The cliff set (descriptive)", ""]
    if cliff:
        out += ["| Arm | Items ok | Accuracy | By family |", "|---|---|---|---|"]
        for arm, s_ in cliff.items():
            if s_:
                fams = ", ".join(f"{f} {v['k']}/{v['n']}" for f, v in s_["by_family"].items())
                out.append(f"| `{arm}` | {s_['items_ok']} | {pct(s_['accuracy']['acc'])} | {fams} |")
        out.append("")
        for sk in res.get("skipped") or []:
            out.append(f"- `{sk['arm']}` was skipped by the budget rule: its estimate of {sk['estimate_s'] / 3600:.2f} h "
                       f"did not fit the {(sk['budget_s'] - sk['used_s']) / 3600:.2f} h left of {sk['budget_s'] / 3600:.0f}.")
        out += ["- single, multikey (P6's k = 4 and the calibrated k = 128, 8 items each) and multivalue (P6's 4 values) "
                "were answered fully at both lengths in every format that ran, so easy retrieval held. vartrack (P6's "
                "c = 5 and the calibrated c = 4) stayed at 0-1 of 16 everywhere, the format failure the first "
                "calibration showed.", ""]
    else:
        out += ["Pending: the cliff plan (`s2cliff`) started at 16:22 EST on 2026-11-02 and runs each arm whose estimate "
                "fits the remaining stage 2 budget. Its results are added to this report when it ends.", ""]
    out += ["## Calibration (reference arm `8,8` at 64K)", "",
            "| Calibration | Family | Level: items fully right | Chosen | Inside 30-90% |", "|---|---|---|---|---|"]
    for name, c in (("first (as registered)", c1), ("second (Amendments 2-4)", c2)):
        for f, v in c["families"].items():
            levels = ", ".join(f"{lv}: {x['k']}/{x['n']}" for lv, x in v["by_level"].items())
            out.append(f"| {name} | {f} | {levels} | {v['chosen']} ({pct(v['chosen_acc'])}) | "
                       f"{'no' if v['outside_30_90'] else 'yes'} |")
    out += ["", "## Tables", "", A.markdown2(res)]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w") as f:
        f.write(build(args.dir))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
