#!/usr/bin/env python3
"""Paper 8 claims: every result number in the manuscript, brief and explainer, as a LaTeX macro (derived/claims.tex).

Sources, all under testsuite/results/turing/longctx/ (each stage's report, results files, journals and server logs):
  stage1/      the reach ladder (registered 2026-11-01, c75ee2e, with Amendment 1)
  stage2/      calibrations, H1-H3, H2s, the noise check, the cliff set, per-arm measures (Amendments 1-8)
  stage3/      the split cache at 96K; the paired 96K-64K gap is recomputed from the journals, as its report did
  fd75d/       the logit check (registered 2026-11-03, 075283a)
  fd75d-arm/   K1 and K2, text identity, decode speeds (registered 2026-11-03)
P6's numbers, for the section on what Paper 6 measured, come from paper6/derived/claims.tex, which P6's own builder
generates from testsuite/results/paper6/analysis-2026-11-01/.

The speed, mechanism and same-card quality numbers (the 2080 Ti story so far, Matt 2026-11-04) come from
testsuite/results/:
  turing/ab/final2.jsonl, final2.md   the speed of record F2 (registered 2026-10-31 13:42 EDT, x-final2), recomputed
                                      from the raw rows and checked against the report; final1.md and depth2.md
  ideas/four-resources-formal/        the cross-session audit (registered 2026-11-02)
  turing/census/                      the stock wheel's census (w2), the int8 census (compare_w7.md) and the 28K
                                      census (compare_w11_d28.md)
  turing/attn/                        the decode-attention bench (bench1.jsonl) and fd75d's (fd75d_test2.jsonl)
  four_resources/test2, test3/        the priced graph, and the verify-pass matrix ratios with the w13 composition
  turing/quality/compare_lcpp_fp32.json   GSM8K-500 and MMLU-500 lanes on the card, with Holm across the 14
                                      same-card pairs
The dated fleet replication uses research/raw/2026-11-12-fleet-lanes/results-2026-11-13-7c2e and
results-2026-11-14-c4d8 (summaries and text-excluded verdict exports), plus the committed P8 speed journals.

The builder recomputes what it can from the journals with the registered statistics (testsuite/p8_longctx/
stage2_analyze.py and stage2_stats.py) and refuses to write if a recomputed value disagrees with a results file, if a
registered verdict differs from the one the manuscript states, or if a server log disagrees with its journal. Scoring
needs the item files, which testsuite/p8_longctx/build_stage2_items.py rebuilds from the committed manifests into
testsuite/p8_longctx/work/items/.

  python3 scripts/build_claims.py
"""
import functools
import glob
import json
import statistics
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
TABLES = {}   # generated LaTeX tables, written beside claims.tex at the end
REPO = os.path.dirname(PAPER)
R = os.path.join(REPO, "testsuite", "results", "turing", "longctx")
ITEMS = os.path.join(REPO, "testsuite", "p8_longctx", "work", "items")
sys.path.insert(0, os.path.join(REPO, "testsuite", "p8_longctx"))
import stage2_analyze as A  # noqa: E402
import stage2_stats as T  # noqa: E402

A.load_items = functools.lru_cache(maxsize=None)(A.load_items)   # one copy of each item file, shared by its arms

K = 1024
CARD_MIB, LIMIT_MIB = 11264, 10752          # the card, and the registered pass limit (512 MiB headroom)
# KV bytes per token in one attention layer: 4 KV heads x 256 dims = 1,024 values each for keys and values; a quantized
# cache stores b bits per value plus one fp16 scale per 32 values (exllamav3's cache layout). 16 attention layers and
# the MTP layer hold a cache. The stage 1 report's "cache arithmetic" column.
FORMATS = ("8,8", "8,4", "4,4", "fp16")
BITS = {"8,8": (8, 8), "8,4": (8, 4), "4,4": (4, 4), "fp16": (16, 16)}
CACHE_LAYERS = 16 + 1
NAME = {"8,8": "EE", "8,4": "EF", "4,4": "FF", "fp16": "Fp"}
WORD = {32: "ThirtyTwo", 36: "ThirtySix", 40: "Forty", 48: "FortyEight", 64: "SixtyFour", 72: "SeventyTwo",
        80: "Eighty", 88: "EightyEight", 96: "NinetySix", 128: "OneTwentyEight", 136: "OneThirtySix"}
COUNT = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}   # counts in prose

M = {}


def put(name, value):
    if not re.fullmatch(r"Pe[A-Za-z]+", name):
        raise SystemExit(f"bad macro name {name}")
    if name in M:
        raise SystemExit(f"duplicate macro {name}")
    M[name] = str(value)


def need(cond, what):
    if not cond:
        raise SystemExit(f"claim check failed: {what}")


def load(*parts):
    with open(os.path.join(R, *parts)) as f:
        return json.load(f)


def signed(x, nd=1):
    s = f"{x:+.{nd}f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def pts(x, nd=1):
    return "\\ensuremath{" + signed(x, nd) + "}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def num(x):
    return f"{x:,}"


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"


def kv_bytes(fmt):
    kb, vb = BITS[fmt]
    side = lambda b: 2048 if b == 16 else 1024 * b // 8 + 2 * (1024 // 32)  # noqa: E731
    return side(kb) + side(vb)


def tango(n10, n01, n, z=1.6448536269514722):
    """Tango's score interval for a paired difference of proportions, in points (90% by default): a check that the
    percentile bootstrap's verdicts do not depend on the interval method when few items are discordant."""
    d, m = (n10 - n01) / n, n - n10 - n01

    def zstat(delta):
        b = n10 * (1 - delta) + n01 * (1 - 3 * delta) - 2 * m * delta
        c = n01 * delta * (1 - delta)
        x = (b + math.sqrt(max(b * b + 8 * n * c, 0.0))) / (4 * n)
        var = (2 * x + delta - delta * delta) / n
        return (d - delta) / math.sqrt(var) if var > 0 else (math.inf if d > delta else -math.inf)

    def solve(lo, hi, target):     # zstat decreases in delta
        for _ in range(200):
            mid = (lo + hi) / 2
            if zstat(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    eps = 1e-12
    return 100 * solve(-1 + eps, d, z), 100 * solve(d, 1 - eps, -z)


def stage1():
    s = load("stage1", "results.json")
    f = s["formats"]
    need(s["verdicts"]["P2"] == "holds" and s["verdicts"]["P3"] == "holds", "stage 1: P2 and P3 hold")
    need(s["verdicts"]["P1"] == "refuted for 8,8, 4,4, 8,4", "stage 1: P1 refuted for the quantized formats")
    need(f["8,8"]["reach"] == 72 * K and f["8,8"]["first_failure"] == 80 * K, "8,8 reaches 72K and fails at 80K")
    for fmt, top in (("4,4", 136), ("8,4", 96), ("fp16", 40)):
        need(f[fmt]["reach"] == top * K and f[fmt]["first_failure"] is None and f[fmt]["ladder_finished"],
             f"{fmt} passes its whole ladder to {top}K")
    step = {(x["format"], x["length"] // K): x for x in s["steps"]}
    need(all(x["outcome"] == "completed" and x["prompt_tokens"] == x["prompt_tokens_expected"] for x in s["steps"]),
         "every ladder step completed with the built prompt length")
    for fmt in FORMATS:
        put(f"PeReach{NAME[fmt]}", f"{f[fmt]['reach'] // K}K")
        reach = step[(fmt, f[fmt]["reach"] // K)]
        put(f"PeReachPeak{NAME[fmt]}", num(reach["peak_mib"]))
        put(f"PeReachFree{NAME[fmt]}", num(CARD_MIB - reach["peak_mib"]))
        b = kv_bytes(fmt)
        put(f"PeBytes{NAME[fmt]}", num(b))
        put(f"PeCacheMiB{NAME[fmt]}", f"{b * CACHE_LAYERS * K / 2 ** 20:.1f}")
    for (fmt, length), x in step.items():
        put(f"PePeak{NAME[fmt]}{WORD[length]}", num(x["peak_mib"]))
        put(f"PePrefill{NAME[fmt]}{WORD[length]}", f"{x['prefill_s']:.0f}")
        put(f"PePrompt{NAME[fmt]}{WORD[length]}", num(x["prompt_tokens"]))
    need(step[("8,8", 80)]["verdict"] == "fail" and step[("8,8", 80)]["peak_mib"] > LIMIT_MIB, "80K over the limit")
    put("PeEEEightyOver", step[("8,8", 80)]["peak_mib"] - LIMIT_MIB)
    put("PeLimitMiB", num(LIMIT_MIB))
    put("PeCardMiB", num(CARD_MIB))
    put("PeHeadroomMiB", CARD_MIB - LIMIT_MIB)
    put("PeEESixtyFourFree", num(CARD_MIB - step[("8,8", 64)]["peak_mib"]))
    put("PeStageOneMin", f"{s['gpu_s'] / 60:.1f}")
    # beyond each format's first interval, the measured slopes agree with the cache arithmetic within 2 MiB per 1K
    # tokens; the quantized formats' first interval above 64K carries an extra 90-120 MiB (stage 1 report)
    for fmt, pairs in (("8,8", [(72, 80)]), ("4,4", [(96, 128), (128, 136)]), ("8,4", [(88, 96)]),
                       ("fp16", [(36, 40)])):
        per_k = kv_bytes(fmt) * CACHE_LAYERS * K / 2 ** 20
        for a, b in pairs:
            slope = (step[(fmt, b)]["peak_mib"] - step[(fmt, a)]["peak_mib"]) / (b - a)
            need(abs(slope - per_k) <= 2.0 + 1e-9, f"{fmt} {a}-{b}K slope {slope:.1f} near {per_k:.1f} MiB per 1K")
    extra = {}
    for fmt, b in (("8,8", 72), ("4,4", 96), ("8,4", 88)):
        per_k = kv_bytes(fmt) * CACHE_LAYERS * K / 2 ** 20
        extra[fmt] = round(step[(fmt, b)]["peak_mib"] - step[(fmt, 64)]["peak_mib"] - (b - 64) * per_k)
        put(f"PeExtra{NAME[fmt]}", extra[fmt])
    need(min(extra.values()) >= 90 and max(extra.values()) <= 120, "the 64K step's extra 90-120 MiB")
    under = [x["p1_offset_mib"] for x in s["steps"]]
    put("PeProjUnderMin", -max(under))
    put("PeProjUnderMax", -min(under))
    put("PeBaselineMiB", step[("8,8", 64)]["baseline_mib"])
    put("PeStageOneRss", f"{max(x['peak_rss_mib'] for x in s['steps']) / 1024:.1f}")
    put("PeStageOneSteps", len(s["steps"]))
    # P1 (every peak within -150..+250 MiB of its projected range): refuted by the quantized formats' 64K steps
    off = [-step[(f, 64)]["p1_offset_mib"] for f in ("8,8", "4,4", "8,4")]
    need(all(not step[(f, 64)]["p1_in_band"] for f in ("8,8", "4,4", "8,4")) and min(off) > 150, "P1's three misses")
    put("PePOneMin", min(off))
    put("PePOneMax", max(off))
    # the three ladder tops predicted to fail that passed
    tops = [(f, l) for (f, l), x in step.items() if x["predicted"] == "fail" and x["verdict"] == "pass"]
    need(sorted(tops) == [("4,4", 136), ("8,4", 96), ("fp16", 40)], "three predicted failures passed")
    # everything but the cache at 64K (derived): peak - cache arithmetic - baseline, for the quantized formats
    rest = [step[(f, 64)]["peak_mib"] - 64 * kv_bytes(f) * CACHE_LAYERS * K / 2 ** 20 - step[(f, 64)]["baseline_mib"]
            for f in ("8,8", "4,4", "8,4")]
    need(max(rest) - min(rest) < 10, "the non-cache footprint agrees across formats")
    put("PeRestMin", num(round(min(rest))))
    put("PeRestMax", num(round(max(rest))))
    put("PeRestGiB", f"{sum(rest) / len(rest) / 1024:.1f}")
    dev = max(abs((step[(f, b)]["peak_mib"] - step[(f, a)]["peak_mib"]) / (b - a)
                  - kv_bytes(f) * CACHE_LAYERS * K / 2 ** 20)
              for f, a, b in (("8,8", 72, 80), ("4,4", 96, 128), ("4,4", 128, 136), ("8,4", 88, 96), ("fp16", 36, 40)))
    put("PeSlopeDev", f"{dev:.1f}")
    gaps = [x["length"] - x["prompt_tokens"] for x in s["steps"]]
    put("PeFillGapMin", min(gaps))
    put("PeFillGapMax", max(gaps))


def server_logs(paths):
    """Per log: the requests the server logged (prompt tokens, temperature, sampler, max_tokens) and their results
    (prompt tokens cached). Wrapped lines are joined first."""
    req = re.compile(r"#(\d+) chat/completions: ([\d,]+) prompt tokens \S temperature: ([\d.]+), (\w+) \(req\), "
                     r"max_tokens: (\d+) \(req\)")
    done = re.compile(r"#(\d+) chat/completions: (\d+) tokens generated at [\d.]+ T/s \S prompt ([\d,]+) tokens, "
                      r"(none cached|[\d,]+ cached)")
    out = {}
    for path in paths:
        lines = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("    ") and lines:
                    lines[-1] += " " + line.strip()
                else:
                    lines.append(line)
        text = "\n".join(lines)
        out[os.path.basename(path)[:-4]] = {"req": req.findall(text), "done": done.findall(text)}
    return out


def logs_agree(rows, logs, plans):
    """Every journaled item has one logged request: temperature 0, greedy, the arm's output limit, no cached tokens."""
    items = {}
    for r in rows:
        if r.get("kind") == "item":
            items.setdefault(r["arm"], []).append(r)
    total = 0
    for arm, its in items.items():
        lg = logs[arm]
        need(len(lg["req"]) == len(its) == len(lg["done"]), f"{arm}: one logged request per journaled item")
        need(all(t == "0" and how == "greedy" for _, _, t, how, _ in lg["req"]), f"{arm}: temperature 0, greedy")
        need({int(m) for *_, m in lg["req"]} == {plans.get(arm, 128)}, f"{arm}: the registered output limit")
        need(all(c == "none cached" for *_, c in lg["done"]), f"{arm}: no prompt tokens reused")
        logged = sorted(int(p.replace(",", "")) for _, p, *_ in lg["req"])
        need(logged == sorted(r["prompt_tokens"] for r in its), f"{arm}: logged prompt lengths match the journal")
        total += len(its)
    return total


def plan_limits():
    out = {}
    for p in glob.glob(os.path.join(REPO, "testsuite", "p8_longctx", "plans", "*.json")):
        with open(p) as f:
            for a in json.load(f)["arms"]:
                if isinstance(a.get("length"), int):
                    out.setdefault(a["arm"], a.get("max_tokens", 128))
    return out


def acc_macros(prefix, a):
    lo, hi = a["wilson95"]
    put(f"{prefix}K", a["k"])
    put(f"{prefix}N", a["n"])
    put(f"{prefix}Acc", pct(a["k"] / a["n"]))
    put(f"{prefix}Lo", pct(lo))
    put(f"{prefix}Hi", pct(hi))


def comp_macros(prefix, c):
    put(f"{prefix}Diff", pts(c["diff_points"]))
    put(f"{prefix}Lo", pts(c["ci90"][0]))
    put(f"{prefix}Hi", pts(c["ci90"][1]))
    put(f"{prefix}Ten", c["n10"])
    put(f"{prefix}Ott", c["n01"])
    put(f"{prefix}P", f"{c['mcnemar_p']:.3g}")
    put(f"{prefix}N", c["n"])
    put(f"{prefix}Abs", f"{abs(c['diff_points']):.1f}")


def arm_macros(prefix, arm, acc=True):
    if acc:
        acc_macros(prefix, arm["accuracy"])
    put(f"{prefix}Decode", f"{arm['decode_tok_s_pooled']:.1f}")
    put(f"{prefix}Prefill", f"{arm['prefill_tok_s_pooled']:.0f}")
    put(f"{prefix}Peak", num(arm["peak_mib"]))
    put(f"{prefix}ItemS", f"{arm['mean_item_s']:.1f}")
    need(not arm["excluded"] and arm["finish_length"] == 0 and not arm["prompt_token_mismatches"],
         f"{prefix}: no exclusion, no truncation, server and built prompt lengths agree")


STAGE2_ARMS = (("s2-64k-88", "PeArmEESixtyFour"), ("s2-64k-44", "PeArmFFSixtyFour"), ("s2-64k-84", "PeArmEFSixtyFour"),
               ("s2-32k-fp16", "PeArmFpThirtyTwo"), ("s2-32k-88", "PeArmEEThirtyTwo"),
               ("s2-32k-44", "PeArmFFThirtyTwo"))


def cycles(rows):
    """Decode time per draft-and-verify cycle in the 32K graded arms (Table 4's fp16 note): tokens generated, draft
    tokens accepted (the server log's "draft A/B accepted"), cycles = tokens - accepted, and the server's own decode
    time (completion_time_s) per cycle."""
    out = {}
    for arm, tag in (("s2-32k-fp16", "Fp"), ("s2-32k-88", "EE"), ("s2-32k-44", "FF")):
        its = [r for r in rows if r.get("kind") == "item" and r["arm"] == arm]
        lines = []
        with open(os.path.join(R, "stage2", "logs", f"{arm}.log"), encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("    ") and lines:
                    lines[-1] += " " + line.strip()
                else:
                    lines.append(line)
        acc = re.findall(r"draft (\d+)/(\d+) accepted", "\n".join(lines))
        need(len(acc) == len(its), f"{arm}: one draft line per request")
        tok = sum(r["completion_tokens"] for r in its)
        accepted, drafted = sum(int(a) for a, _ in acc), sum(int(d) for _, d in acc)
        n_cycles = tok - accepted
        ms = 1000 * sum(r["completion_time_s"] for r in its) / n_cycles
        out[tag] = ms
        put(f"PeCycle{tag}Ms", f"{ms:.0f}" if ms >= 100 else f"{ms:.1f}")
        put(f"PeAccept{tag}", f"{100 * accepted / drafted:.1f}")
        put(f"PeTokPerCycle{tag}", f"{tok / n_cycles:.2f}")
    need(out["Fp"] > 250 and out["EE"] < 45, "fp16's slow cycles against fd75d's")
    return out


def stage2(limits):
    s = load("stage2", "results.json")
    c = s["comparisons"]
    rows = A.load_rows(os.path.join(R, "stage2", "journal.jsonl"))
    again = A.stage2(rows, ITEMS)
    again_arms = A.Arms(rows, ITEMS)
    for key in ("h1", "h2", "h3", "h2s"):
        need(abs(again["comparisons"][key]["diff_points"] - c[key]["diff_points"]) < 1e-9
             and again["comparisons"][key]["ci90"] == c[key]["ci90"], f"{key} recomputes from the journal")
    need(not c["h2"]["usable"] and c["h2"]["ci90"][0] < -10, "H2 is not usable")
    need(c["h1"]["usable"] and c["h1"]["equivalent"] and c["h1"]["n10"] + c["h1"]["n01"] == 0, "H1 is equivalent")
    need(c["h3"]["usable"] and not c["h3"]["equivalent"], "H3 is usable, not equivalent")
    need(c["h2s"]["usable"] and not c["h2s"]["equivalent"] and c["h2s"]["ci90"][0] > 5, "H2s usable, above +5")
    need(s["rules"]["adopted_mode"] == "8,4" and not s["rules"]["R1"]["applies"] and s["rules"]["R2"]["h2s_usable"]
         and not s["rules"]["R3"]["follow_up_for_8bit_default"], "R1 does not apply, R2 adopts 8,4, R3 none")
    comp_macros("PeHTwo", c["h2"])
    comp_macros("PeHOne", c["h1"])
    comp_macros("PeHThree", c["h3"])
    comp_macros("PeHTwoS", c["h2s"])
    put("PeHThreeHolm", f"{c['h3']['mcnemar_p_holm']:.2g}")
    for key, tag in (("h1", "HOne"), ("h2", "HTwo"), ("h3", "HThree"), ("h2s", "HTwoS")):
        lo, hi = tango(c[key]["n10"], c[key]["n01"], c[key]["n"])
        need(T.verdict(lo, hi) == {"usable": c[key]["usable"], "equivalent": c[key]["equivalent"]},
             f"{key}: the score interval gives the same verdict")
        put(f"PeTango{tag}Lo", pts(lo))
        put(f"PeTango{tag}Hi", pts(hi))
    put("PeTangoHOneHalf", f"{tango(0, 0, c['h1']['n'])[1]:.1f}")
    # pairing by item across lengths (number check F15): the split cache at 64K against fp16 at 32K, descriptive
    cross = A.comparison(again_arms, "s2-64k-84", "s2-32k-fp16")
    need(cross["n"] == 96, "the cross-length pairing covers the 96 items")
    for k_, v_ in (("Diff", pts(cross["diff_points"])), ("Lo", pts(cross["ci90"][0])), ("Hi", pts(cross["ci90"][1])),
                   ("Ten", cross["n10"]), ("Ott", cross["n01"]), ("P", f"{cross['mcnemar_p']:.2g}")):
        put(f"PeCross{k_}", v_)
    arms = s["arms"]
    for arm, prefix in STAGE2_ARMS:
        arm_macros(prefix, arms[arm])
    graded = [a for a, _ in STAGE2_ARMS] + ["s2-64k-88-noise"]
    put("PeAnswerMax", max(arms[a]["answer_tokens"]["max"] for a in graded))
    n = s["noise"]
    need(n["discordant"] == 0 and n["n"] == 16, "the noise check: 0 of 16")
    put("PeNoiseK", n["discordant"])
    put("PeNoiseN", n["n"])
    put("PeNoiseHi", pct(n["wilson95"][1]))
    put("PeNoiseSameK", n["cache_discordance_same_items"]["discordant"])
    # the split cache against the 8-bit cache on the noise check's 16 items (number check F14)
    first16 = list(again_arms.items["s2-64k-88-noise"])[:16]
    s84, s88 = again_arms.scores["s2-64k-84"], again_arms.scores["s2-64k-88"]
    same16 = sum(1 for u in first16 if s84[u] != s88[u])
    need(len(first16) == 16 and same16 == 6, "the split cache's 6 of 16 on the noise check's items")
    put("PeHTwoSSameK", same16)
    put("PeHTwoSSamePct", pct(same16 / 16))
    put("PeHTwoDisc", n["cache_discordance_all"]["discordant"])
    put("PeHTwoDiscPct", pct(n["cache_discordance_all"]["rate"]))
    put("PeHTwoSDisc", c["h2s"]["n10"] + c["h2s"]["n01"])
    put("PeHTwoSDiscPct", pct((c["h2s"]["n10"] + c["h2s"]["n01"]) / c["h2s"]["n"]))
    used = sum(a["gpu_s"] for a in arms.values())
    need(len(s["skipped"]) == 1 and s["skipped"][0]["arm"] == "cliff-64k-84" and s["skipped"][0]["why"] == "budget"
         and abs(s["skipped"][0]["used_s"] - used) < 1, "only the split cache's cliff arm was skipped, by budget")
    put("PeStageTwoHours", f"{used / 3600:.2f}")
    put("PeSkipEstimate", f"{s['skipped'][0]['estimate_s'] / 3600:.2f}")
    put("PeSkipLeft", f"{(s['skipped'][0]['budget_s'] - used) / 3600:.2f}")
    for arm, v in s["cliff"].items():
        fam = v["by_family"]
        need(all(fam[x]["k"] == fam[x]["n"] for x in ("single", "multikey", "multivalue")),
             f"{arm}: easy retrieval fully answered")
        need(fam["vartrack"]["k"] <= 1 and fam["vartrack"]["n"] == 16, f"{arm}: vartrack at 0-1 of 16")
    put("PeCliffArms", len(s["cliff"]))
    put("PeCliffArmsWord", COUNT[len(s["cliff"])])
    put("PeCliffVtMax", max(v["by_family"]["vartrack"]["k"] for v in s["cliff"].values()))
    put("PeStageTwoJobs", sum(1 for r in rows if r.get("kind") == "plan_start"))
    lengths = {r["arm"]: r["items_file"] for r in rows if r.get("kind") == "arm_plan"}
    room_all = min(int(lengths[r["arm"]].split("-")[1][:-len("k.jsonl")]) * K - r["prompt_tokens"]
                   for r in rows if r.get("kind") == "item")
    put("PeRoomAll", num(room_all))
    put("PeAnswerMaxCalibTwo", arms["calib2-64k-88"]["answer_tokens"]["max"])
    same, n_common = text_identity(rows, "s2-32k-88", "s2-32k-fp16")
    need(n_common == 96, "H1's 96 items")
    put("PeHOneSameTexts", same)
    # calibrations
    c1 = load("stage2", "calibration.json")["families"]
    c2 = load("stage2", "calibration2.json")["families"]
    need(c1["multikey"]["chosen"] == 128 and c1["vartrack"]["chosen"] == 4
         and c1["multikey"]["outside_30_90"] and c1["vartrack"]["outside_30_90"], "first calibration: both outside")
    need(c2["multivalue"]["chosen"] == 12 and not c2["multivalue"]["outside_30_90"], "second calibration: k 12")
    lv_word = {"2": "Two", "3": "Three", "4": "Four", "12": "Twelve", "16": "Sixteen", "24": "TwentyFour",
               "48": "FortyEight", "128": "OneTwentyEight"}
    for fam, src, tag in (("multikey", c1, "Mk"), ("vartrack", c1, "Vt"), ("multivalue", c2, "Mv")):
        for lv, x in src[fam]["by_level"].items():
            put(f"PeCal{tag}{lv_word[lv]}", f"{x['k']}/{x['n']}")
    put("PeCalMvAcc", pct(c2["multivalue"]["chosen_acc"]))
    # the server's own records: one logged request per item, temperature 0, greedy, no cached prompt tokens
    logs = server_logs(sorted(glob.glob(os.path.join(R, "stage2", "logs", "*.log"))))
    n_req = logs_agree(rows, logs, limits)
    cycles(rows)
    prompts = {}
    for r in rows:
        if r.get("kind") == "item" and r["arm"] in graded:
            prompts.setdefault(r["arm"].split("-")[1], []).append(r["prompt_tokens"])
    for length, word in (("32k", "ThirtyTwo"), ("64k", "SixtyFour")):
        put(f"PePromptMin{word}", num(min(prompts[length])))
        put(f"PePromptMax{word}", num(max(prompts[length])))
        put(f"PeRoom{word}", num(int(length[:-1]) * K - max(prompts[length])))
    sessions = [r for r in rows if r.get("kind") == "session_end"]
    starts = [r for r in rows if r.get("kind") == "session_start"]
    plans = [r for r in rows if r.get("kind") == "plan_start"]
    return {"n_req": n_req, "temps": [r["max_temp_c"] for r in sessions], "rss": [r["peak_rss_mib"] for r in sessions],
            "baselines": {r["baseline_mib"] for r in starts}, "power": {r["power_limit_w"] for r in plans},
            "sessions": len(sessions)}


def stage3(limits):
    s = load("stage3", "results.json")
    p96 = next(p for p in s["curve"] if p["arm"] == "s3-96k")
    need(s["mode"] == "8,4" and s["usable_context_k"] is None and s["P3b"] is None and not s["probe"],
         "stage 3: the split cache, no usable-context value, no probe")
    arm = s["arms"]["s3-96k"]
    need(arm["accuracy"]["k"] == p96["k"] and arm["accuracy"]["n"] == p96["n"], "stage 3's curve and arm agree")
    arm_macros("PeArmEFNinetySix", arm)
    put("PeStageThreeHours", f"{arm['gpu_s'] / 3600:.2f}")
    put("PeArmEFNinetySixFree", num(CARD_MIB - arm["peak_mib"]))
    rows3 = A.load_rows(os.path.join(R, "stage3", "journal.jsonl"))
    rows2 = A.load_rows(os.path.join(R, "stage2", "journal.jsonl"))
    a2, a3 = A.Arms(rows2, ITEMS), A.Arms(rows3, ITEMS)
    s64, s96 = a2.scores["s2-64k-84"], a3.scores["s3-96k"]
    common = sorted(u for u in s96 if u in s64)
    x, y = [s96[u] for u in common], [s64[u] for u in common]
    mean, lo, hi = T.paired_interval(x, y)
    n10, n01, p = T.mcnemar(x, y)
    need(len(common) == 48 and signed(mean) == "-27.1" and signed(lo) == "-41.7" and signed(hi) == "-14.6"
         and (n10, n01) == (3, 16), "the stage 3 report's paired 96K-64K gap")
    put("PeGapN", len(common))
    put("PeGapSixtyFourK", sum(y))
    put("PeGapNinetySixK", sum(x))
    put("PeGapDiff", pts(mean))
    put("PeGapLo", pts(lo))
    put("PeGapHi", pts(hi))
    put("PeGapTen", n10)
    put("PeGapOtt", n01)
    put("PeGapP", f"{p:.2g}")
    put("PeGapSixtyFourAcc", pct(sum(y) / len(y)))
    put("PeGapAbs", f"{abs(mean):.1f}")
    put("PeGapAbsRound", f"{abs(mean):.0f}")
    # descriptive (number check F3): the same 48 items against the 8-bit cache at 64K and fp16 at 32K
    for ref, tag in (("s2-64k-88", "EE"), ("s2-32k-fp16", "Fp")):
        sr = a2.scores[ref]
        c2 = [u for u in common if u in sr]
        xr, yr = [s96[u] for u in c2], [sr[u] for u in c2]
        m_, lo_, hi_ = T.paired_interval(xr, yr)
        a10, a01, pp = T.mcnemar(xr, yr)
        need(len(c2) == 48, f"96K against {ref}: 48 items")
        put(f"PeGap{tag}Diff", pts(m_))
        put(f"PeGap{tag}Lo", pts(lo_))
        put(f"PeGap{tag}Hi", pts(hi_))
        put(f"PeGap{tag}Ten", a10)
        put(f"PeGap{tag}Ott", a01)
        put(f"PeGap{tag}P", f"{pp:.2g}")
        put(f"PeGap{tag}K", sum(yr))
        put(f"PeGap{tag}Abs", f"{abs(m_):.1f}")
        put(f"PeGap{tag}AbsRound", f"{abs(m_):.0f}")
    put("PeNinetySixUnderLimit", LIMIT_MIB - arm["peak_mib"])
    logs = server_logs(sorted(glob.glob(os.path.join(R, "stage3", "logs", "*.log"))))
    n_req = logs_agree(rows3, logs, limits)
    prompts = [r["prompt_tokens"] for r in rows3 if r.get("kind") == "item"]
    put("PePromptMinNinetySix", num(min(prompts)))
    put("PePromptMaxNinetySix", num(max(prompts)))
    put("PeRoomNinetySix", num(96 * K - max(prompts)))
    put("PeAnswerMaxNinetySix", arm["answer_tokens"]["max"])
    sessions = [r for r in rows3 if r.get("kind") == "session_end"]
    need({r["power_limit_w"] for r in rows3 if r.get("kind") == "plan_start"} == {"260.00"}, "stage 3 at 260 W")
    return {"n_req": n_req, "temps": [r["max_temp_c"] for r in sessions], "rss": [r["peak_rss_mib"] for r in sessions],
            "sessions": len(sessions)}


def fd75d():
    s = load("fd75d", "results.json")
    L = s["lengths"]
    need(s["decision"]["diverges_48k_or_64k"] and not s["decision"]["diverges_28k"], "logit check: diverges at 48K")
    need(L["48k"]["diverges"] and not L["64k"]["diverges"] and not L["28k"]["diverges"], "only 48K diverges")
    need(all(L[x]["kl_floor"]["mean"] == 0 and L[x]["kl_floor"]["max"] == 0 for x in ("28k", "48k", "64k")),
         "the Triton-Triton floor is exactly zero")
    need(all(L[x]["top1_f"] == 1.0 and L[x]["top1_floor"] == 1.0 for x in L), "top-1 agreement 100% everywhere")
    need(s["rule"]["kl_abs"] == 1e-4 and s["rule"]["kl_factor"] == 2.0, "the registered threshold")
    for x, w in (("28k", "TwentyEight"), ("48k", "FortyEight"), ("64k", "SixtyFour")):
        put(f"PeKl{w}", sci(L[x]["kl_f"]["mean"]))
    put("PeKlMax", f"{L['pooled']['kl_f']['max']:.3f}")
    put("PeKlPositions", L["pooled"]["positions"])
    put("PeKlPerLength", L["48k"]["positions"])
    with open(os.path.join(R, "fd75d", "used_s")) as f:
        used = [float(x) for x in f.read().split()]
    need(len(used) == 3, "three logit-check runs")
    put("PeKlMinutes", f"{sum(used) / 60:.1f}")
    with open(os.path.join(R, "fd75d", "run-F", "run.json")) as f:
        return json.load(f)


def arm(limits):
    s = load("fd75d-arm", "results.json")
    need(s["K1"]["usable"] and s["K1"]["equivalent"] and s["K2"]["usable"] and not s["K2"]["equivalent"]
         and s["K2"]["ci90"][0] > 5, "K1 equivalent; K2 usable with its interval above +5")
    need(s["rule"].startswith("K1 is equivalent"), "the registered rule: the kernel is not what separates the arms")
    need(s["predictions"] == {"K1_point_above_0": "refuted", "K2_equivalent": "refuted"}, "both predictions refuted")
    need(abs(s["decomposition"]["K1_plus_K2"] - s["decomposition"]["H2s"]) < 1e-9, "H2s = K1 + K2")
    arm_rows = A.load_rows(os.path.join(R, "fd75d-arm", "journal.jsonl"))
    rows = A.load_rows(os.path.join(R, "stage2", "journal.jsonl")) + arm_rows
    again = A.comparison(A.Arms(rows, ITEMS), "s2-64k-88-triton", "s2-64k-88")
    need(again["ci90"] == s["K1"]["ci90"] and again["diff_points"] == s["K1"]["diff_points"], "K1 recomputes")
    comp_macros("PeKOne", s["K1"])
    comp_macros("PeKTwo", s["K2"])
    for key, tag in (("K1", "KOne"), ("K2", "KTwo")):
        lo, hi = tango(s[key]["n10"], s[key]["n01"], s[key]["n"])
        need(T.verdict(lo, hi) == {"usable": s[key]["usable"], "equivalent": s[key]["equivalent"]},
             f"{key}: the score interval gives the same verdict")
        put(f"PeTango{tag}Lo", pts(lo))
        put(f"PeTango{tag}Hi", pts(hi))
    t = s["descriptive"]["text_identity"]
    need(all(v["score_change_without_text_change"] == 0 for v in t.values()), "no score change without a text change")
    put("PeKOneSame", t["triton_vs_fd75d"]["identical"])
    put("PeKOneDiffer", t["triton_vs_fd75d"]["differ"])
    put("PeKOneChanged", t["triton_vs_fd75d"]["correctness_changed"])
    put("PeKOneBound", f"{s['descriptive']['K1_bound_points']:.1f}")
    put("PeKTwoSame", t["split_vs_triton"]["identical"])
    put("PeKTwoDiffer", t["split_vs_triton"]["differ"])
    put("PeKTwoChanged", t["split_vs_triton"]["correctness_changed"])
    put("PeRerunSame", t["triton_vs_fd75d_rerun_16"]["identical"])
    a = s["arms"]
    tri, fd, split = a["s2-64k-88-triton"], a["s2-64k-88"], a["s2-64k-84"]
    need(tri["accuracy"]["k"] == 68 and tri["sessions"] == 1, "the Triton arm: 68/96 in one session")
    arm_macros("PeArmTriton", tri)
    put("PeDecodeRatio", f"{fd['decode_tok_s_pooled'] / tri['decode_tok_s_pooled']:.1f}")
    put("PeArmHours", f"{tri['gpu_s'] / 3600:.2f}")
    put("PeItemSDiffPct", f"{100 * (split['mean_item_s'] - fd['mean_item_s']) / fd['mean_item_s']:.1f}")
    h = load("stage2", "results.json")["comparisons"]
    widths = [c["ci90"][1] - c["ci90"][0] for c in (h["h2"], h["h2s"], s["K2"])]
    put("PeWidthMin", f"{min(widths):.1f}")
    put("PeWidthMax", f"{max(widths):.1f}")
    prefill = [x["prefill_tok_s_pooled"] for x in (tri, fd, split)]
    put("PePrefillArmMin", f"{min(prefill):.0f}")
    put("PePrefillArmMax", f"{max(prefill):.0f}")
    env_rows = [r for r in arm_rows if r.get("kind") in ("arm_plan", "server_ready")]
    need(len(env_rows) == 2 and all(r.get("arm_env") == {"EXL3_FD75D": "0"} for r in env_rows),
         "the arm's plan and server journaled EXL3_FD75D=0")
    need(all("arm_env" not in r for r in rows if r.get("kind") in ("arm_plan", "server_ready")
             and r["arm"] == "s2-64k-88"), "stage 2's 8-bit arm ran the build's default kernel, fd75d")
    logs = server_logs([os.path.join(R, "fd75d-arm", "logs", "s2-64k-88-triton.log")])
    n_req = logs_agree(arm_rows, logs, limits)
    sessions = [r for r in arm_rows if r.get("kind") == "session_end"]
    need({r["power_limit_w"] for r in arm_rows if r.get("kind") == "plan_start"} == {"260.00"}, "the arm at 260 W")
    return {"n_req": n_req, "temps": [r["max_temp_c"] for r in sessions], "rss": [r["peak_rss_mib"] for r in sessions],
            "sessions": len(sessions)}


def envelope(e2, e3, ea, run):
    need(e2["power"] == {"260.00"}, "every stage 2 plan journaled the 260 W limit")
    need(e2["baselines"] == {379}, "the 379 MiB baseline before every stage 2 session")
    put("PeNRequests", num(e2["n_req"] + e3["n_req"] + ea["n_req"]))
    put("PeNSessions", e2["sessions"] + e3["sessions"] + ea["sessions"])
    put("PeMaxTemp", f"{max(e2['temps'] + e3['temps'] + ea['temps']):.0f}")
    put("PePeakRss", f"{max(e2['rss'] + e3['rss'] + ea['rss']) / 1024:.1f}")
    d, v = run["device"], run["versions"]
    need(d["name"] == "NVIDIA GeForce RTX 2080 Ti" and d["cc"] == [7, 5] and d["start"]["mem_total_mib"] == "11264",
         "the logit check's device record")
    put("PeDriver", d["start"]["driver"])
    put("PeTorch", v["torch"].replace("+", "{+}"))
    put("PeTriton", v["triton"])
    put("PeCuda", v["cuda"])
    put("PePython", v["python"])
    with open(os.path.join(R, "stage1", "journal.jsonl")) as f:
        first = json.loads(f.readline())
    need(first["env"] == {"PYTHONPATH": "/opt/p8/overlay/exl3-w13", "EXL3_MGEMM_N_THRESHOLD": "0",
                          "EXL3_FD75D_LIVE": "0", "EXL3_INT8_SQ_MAX_M": "2", "EXL3_GEMV_SK": "2"},
         "the w13 build's environment, as stage 1 journaled it")
    versions = set()
    for p in glob.glob(os.path.join(R, "stage*", "logs", "*.log")) + glob.glob(os.path.join(R, "fd75d-arm", "logs",
                                                                                           "*.log")):
        with open(p, encoding="utf-8", errors="replace") as f:
            versions |= set(re.findall(r"exllamav3 version: (\S+)", f.read()))
    need(versions == {"1.6.5"}, "every server logged exllamav3 1.6.5")
    put("PeExlVersionShort", "1.6.5")


def matched_items():
    """The graded items at 32K, 64K and 96K share identifiers, key, values and question; the haystack's length differs
    (P6's generator keys its randomness by the seed and the item id)."""
    def items(name):
        return {u: it for u, it in A.load_items(ITEMS, name).items()}
    i32, i64, i96 = items("graded-32k.jsonl"), items("graded-64k.jsonl"), items("stage3-96k.jsonl")
    need(set(i32) == set(i64) and set(i96) <= set(i64) and len(i64) == 96 and len(i96) == 48, "item identifiers")
    for other in (i32, i96):
        for u, it in other.items():
            need(it["answers"] == i64[u]["answers"] and it["question"] == i64[u]["question"],
                 f"{u}: the same key, values and question at every length")
    put("PeNeedles", len(i64["multivalue-k12-00"]["answers"]))


def text_identity(rows, a, b):
    ta = {r["uid"]: r.get("text") or "" for r in rows if r.get("kind") == "item" and r["arm"] == a}
    tb = {r["uid"]: r.get("text") or "" for r in rows if r.get("kind") == "item" and r["arm"] == b}
    return sum(1 for u in ta if u in tb and ta[u] == tb[u]), len(set(ta) & set(tb))


def peak_rise():
    """Peak memory of each graded arm over its requests, against stage 1's single filled request at that length."""
    s1 = {(x["format"], x["length"] // K): x["peak_mib"] for x in load("stage1", "results.json")["steps"]}
    s2, s3 = load("stage2", "results.json")["arms"], load("stage3", "results.json")["arms"]
    tri = load("fd75d-arm", "results.json")["arms"]["s2-64k-88-triton"]
    arms = [("8,8", 64, s2["s2-64k-88"]), ("4,4", 64, s2["s2-64k-44"]), ("8,4", 64, s2["s2-64k-84"]),
            ("fp16", 32, s2["s2-32k-fp16"]), ("8,4", 96, s3["s3-96k"]), ("8,8", 64, tri)]
    rise = [a["peak_mib"] - s1[(fmt, length)] for fmt, length, a in arms]
    need(all(a["peak_mib"] <= LIMIT_MIB for *_, a in arms), "every graded arm peaked under the pass limit")
    put("PeRiseMin", min(rise))
    put("PeRiseMax", max(rise))
    put("PeArmPeakMax", num(max(a["peak_mib"] for *_, a in arms)))


def replication_r1():
    """The value-bits replication's R1 (research/prereg/2026-11-04-p8-value-bits-replication.md), from the journal copied
    after both R1 arms closed: recomputed with the registered statistics and checked against r1.json."""
    d = os.path.join(R, "replication")
    with open(os.path.join(d, "r1.json")) as f:
        r1 = json.load(f)
    rows = A.load_rows(os.path.join(d, "journal_at_r1.jsonl"))
    arms = A.Arms(rows, ITEMS)
    c = A.comparison(arms, "rep-64k-84", "rep-64k-88-triton")
    need(c["ci90"] == r1["R1"]["ci90"] and c["diff_points"] == r1["R1"]["diff_points"] and c["n"] == 96,
         "R1 recomputes from the journal copy")
    lo, hi = c["ci90"]
    need(not lo > 0 and not (lo >= -5 and hi <= 5) and hi >= 0 and r1["verdict"] == "not replicated",
         "R1: not replicated by the registered rule")
    comp_macros("PeROne", c)
    for arm, tag in (("rep-64k-84", "Split"), ("rep-64k-88-triton", "Eight")):
        a = r1["arms"][arm]
        need(not a["excluded"] and a["finish_length"] == 0 and not a["prompt_token_mismatches"], f"{arm}: clean")
        acc_macros(f"PeROne{tag}", a["accuracy"])
    put("PeROneSame", r1["R1"]["text_identity"]["identical"])
    put("PeROneDiffer", r1["R1"]["text_identity"]["n"] - r1["R1"]["text_identity"]["identical"])
    logs = server_logs([os.path.join(d, "logs", f"{a}.log") for a in ("rep-64k-88-triton", "rep-64k-84")])
    n_req = logs_agree([r for r in rows if r.get("arm") in ("rep-64k-88-triton", "rep-64k-84")], logs, plan_limits())
    put("PeROneRequests", n_req)


def replication_final():
    """The replication's N, R2 and R3 from the final journal (copied after plan_end, 2026-11-04 18:24 EST), judged by
    the registered analysis (testsuite/p8_longctx/replication_analyze.py, c0260db) and checked against results.json;
    R1 in the final journal must equal R1 at r1.json."""
    import replication_analyze as RA
    d = os.path.join(R, "replication")
    res = json.load(open(os.path.join(d, "results.json")))
    rep = A.load_rows(os.path.join(d, "journal.jsonl"))
    got = RA.judge(rep, A.load_rows(os.path.join(R, "stage2", "journal.jsonl")),
                   A.load_rows(os.path.join(R, "fd75d-arm", "journal.jsonl")), ITEMS)
    for k in ("R1", "N", "R2"):
        need(got[k]["ci90"] == res[k]["ci90"] and got[k]["diff_points"] == res[k]["diff_points"], f"{k} recomputes")
    for k in ("D3a_6_minus_8", "D3b_4_minus_6"):
        need(got["R3"][k]["ci90"] == res["R3"][k]["ci90"], f"R3 {k} recomputes")
    r1 = json.load(open(os.path.join(d, "r1.json")))["R1"]
    need(res["R1"]["ci90"] == r1["ci90"] and res["R1"]["diff_points"] == r1["diff_points"], "R1 unchanged at the end")
    need(all(v == "done" for k, v in res["status"].items() if k != "rep-32k-fp16")
         and res["status"]["rep-32k-fp16"] == "skipped (reproduced)" and not res["excluded"], "every arm closed clean")
    plan_end = next(r for r in rep if r.get("kind") == "plan_end")
    put("PeRepHours", f"{plan_end['used_s'] / 3600:.1f}")
    chk = res["R2"]["check"]
    need(chk["reproduced"] and chk["identical"] == chk["n_check"] == 16, "R2's fp16 check reproduced stage 2")
    put("PeRTwoCheck", chk["identical"])
    n = res["N"]
    comp_macros("PeNz", n)
    put("PeNzDisc", n["discordant"])
    put("PeNzWilsonHi", f"{100 * n['wilson95'][1]:.1f}")
    put("PeNzKTwoDisc", pct(n["K2_discordance"]))
    need(n["K2_beyond_harmless_spread"] and n["wilson95"][1] < 17 / 96, "N: K2's discordance beyond the harmless spread")
    need(-5 <= n["ci90"][0] and n["ci90"][1] <= 5, "N: equivalent, as predicted")
    put("PeNzSame", n["text_identity"]["identical"])
    put("PeNzFirstSixteen", n["first16"]["N"]["discordant"])
    put("PeNzAccA", pct(n["acc_a"]))
    put("PeNzAccB", pct(n["acc_b"]))
    r2 = res["R2"]
    comp_macros("PeRTwo", r2)
    need(r2["outcome"] == "not replicated" and r2["fp16_reference"] == "s2-32k-fp16", "R2's registered outcome")
    put("PeRTwoAccA", pct(r2["acc_a"]))
    put("PeRTwoAccB", pct(r2["acc_b"]))
    a, b = res["R3"]["D3a_6_minus_8"], res["R3"]["D3b_4_minus_6"]
    comp_macros("PeRThreeA", a)
    comp_macros("PeRThreeB", b)
    need(res["R3"]["monotone"], "R3: monotone as described")
    put("PeRThreeSixAcc", pct(a["acc_a"]))
    put("PeRThreeEightAcc", pct(a["acc_b"]))
    put("PeRThreeFourAcc", pct(b["acc_a"]))


def weight_hashes():
    """The EXL3 weight files' SHA-256, from the 2026-09-06 hash audit against the Hub's LFS digests."""
    with open(os.path.join(REPO, "testsuite", "results", "efficiency", "hash_audit_2026-09-06.json")) as f:
        audit = json.load(f)
    rows = [x for x in audit if "qwen38-exl3-2.0/" in x["local"]]
    need(len(rows) == 2 and all(x["status"] == "MATCH" and "SC_2.00bpw_H3" in str(x["hub"]) for x in rows),
         "both EXL3 2.0 bpw files matched the Hub at revision SC_2.00bpw_H3")
    rows.sort(key=lambda x: x["local"])
    put("PeShaOne", rows[0]["sha256"][:12])
    put("PeShaTwo", rows[1]["sha256"][:12])
    put("PeAuditN", len(audit))


RT = os.path.join(REPO, "testsuite", "results")


def md_rows(path):
    """Rows of every markdown table in a report, as lists of stripped cells."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("|") and not set(line.strip()) <= set("|-: "):
                out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def p9_bench():
    """P9's registered phase-1 measurements on the 2080 Ti (research/prereg/2026-11-05-p9-phase1.md), which P8's
    section 4.3 cites under that registration's section 5: testsuite/results/p9/analysis/report.json, written by
    tools/turing/p9_analyze.py from testsuite/results/turing/p9/20261105T142814/."""
    r = json.load(open(os.path.join(RT, "p9", "analysis", "report.json")))
    f = r["fits"]["desktop/w13/K2"]
    need(f["rounds"] == 7 and f["a_half"] < 1.0, "P9: seven rounds, the 2-bit fixed part within 1 us")
    put("PeNineFixedTwo", f"{f['a']:.1f}")
    put("PeNineFixedTwoHalf", f"{f['a_half']:.1f}")
    put("PeNinePerWeightTwoPct", f"{100 * (f['b'] / 0.4545 - 1):.0f}")   # above the 2-bit read at 550 GB/s
    f1, f4 = r["fits"]["desktop/w13/K1"], r["fits"]["desktop/w13/K4"]
    need(f1["rounds"] == f4["rounds"] == 7, "P9: seven rounds at 1 and 4 bits")
    put("PeNinePerWeightOneX", f"{f1['b'] / (0.4545 / 2):.1f}")          # a 1-bit weight against its read at 550 GB/s
    put("PeNineOneTwoPerWeight", f"{100 * f1['b'] / f['b']:.0f}")        # the 1-bit time per weight, % of the 2-bit
    put("PeNineFourBitGbs", f"{0.5e3 / f4['b']:.0f}")                     # 4-bit calls' read rate: 0.5 MB per M weights
    put("PeNineShare", f"{100 * r['h2']['share']:.0f}")
    put("PeNineShareRated", f"{100 * r['h2']['share_rated_616']:.0f}")   # the same at the rated 616 GB/s
    ot = r["one_over_two"]
    six = [v for s, v in ot.items() if s != "attn_kv"]
    need(len(six) == 6 and max(six) < 1.0 and ot["attn_kv"] > 1.0, "P9: 1-bit faster on six shapes, slower on attn_kv")
    put("PeNineOneTwoLo", f"{100 * min(six):.0f}")
    put("PeNineOneTwoHi", f"{100 * max(six):.0f}")
    put("PeNineOneTwoKv", f"{100 * ot['attn_kv']:.0f}")
    fu = r["fusion_saving_in_a"]
    put("PeNineFuseQkv", f"{fu['fused_q_k_v']:.2f}")
    put("PeNineFuseQkvZ", f"{fu['fused_qkv_z']:.2f}")
    put("PeNineFuseGateUp", f"{fu['fused_gate_up']:.2f}")
    pc = r["phase_contrib_desktop_K2"]
    need(pc["epilogue"] > pc["loop"] > pc["stage"], "P9: epilogue, then loop, then stage")
    put("PeNinePhFull", f"{pc['full']:.1f}")
    put("PeNinePhEpi", f"{pc['epilogue']:.1f}")
    put("PeNinePhLoop", f"{pc['loop']:.1f}")
    put("PeNinePhStage", f"{pc['stage']:.1f}")
    put("PeNinePhNull", f"{pc['null']:.1f}")
    s = r["sass"]["w13/sm_75"]
    put("PeNineIpwOne", f"{s['K1']:.2f}")
    put("PeNineIpwTwo", f"{s['K2']:.2f}")
    lim = r["limiter"]["desktop/w13"]
    put("PeNineIssuePct", f"{100 * lim['issue_floor'] / lim['b']:.0f}")


def model_files():
    """The model files' table (derived/model_files.tex) from testsuite/results/turing/model_files.json, which
    tools/turing/model_files.py measured from the files' own headers; sizes and digests checked against the hash audit."""
    rows = json.load(open(os.path.join(RT, "turing", "model_files.json")))
    audit = json.load(open(os.path.join(REPO, "testsuite", "results", "efficiency", "hash_audit_2026-09-06.json")))
    by_name = {"/".join(x["local"].split("/")[-2:]): x for x in audit}   # folder and name: shard names repeat
    need([r["key"] for r in rows] == ["exl3", "iq2xxs", "iq2s", "mtp_full", "mtp_shared", "dflash2"], "model files listed")
    for r in rows:
        need(abs(r["tensor_bytes"] * 8 / r["parameters"] - r["bits_per_weight"]) < 1e-9, f"{r['key']}: bits per weight")
        for f, sha in zip(r["files"], r["sha256"]):
            a = by_name.get("/".join(f.split("/")[-2:]))
            need(a is None or a["sha256"] == sha, f"{f}: digest matches the hash audit")
        need(r["hub_digest_checked"] == (r["key"] != "mtp_shared"), f"{r['key']}: Hub digest checked except our sidecar")
    ex = rows[0]
    put("PeFilesExlParams", f"{ex['parameters'] / 1e9:.2f}")
    put("PeFilesExlTrellisParams", f"{ex['trellis_parameters'] / 1e9:.2f}")
    put("PeFilesExlTrellisBpw", f"{ex['trellis_bits_per_weight']:.2f}")
    put("PeFilesExlBpw", f"{ex['bits_per_weight']:.2f}")
    put("PeFilesGgufParams", f"{rows[1]['parameters'] / 1e9:.2f}")
    need(rows[1]["parameters"] == rows[2]["parameters"], "both 2-bit GGUFs hold the same parameters")
    # where the EXL3 file's parts live in a text-only run: exllamav3 1.6.5 loads an unquantized embedding table to host
    # memory (modules/embedding.py, caps "prefer_cpu"), the servers ran with vision off, and the rest sits on the card
    wb = open(os.path.join(RT, "turing", "census", "weight_bytes_qwen38-exl3-2.0.txt")).read()
    mib = lambda part: float(re.search(rf"([\d.]+) MiB\s+{part}\n", wb).group(1))   # noqa: E731
    text, total = (float(x) for x in re.search(r"total ([\d.]+) GiB; text layers ([\d.]+) GiB", wb).groups()[::-1])
    on_card = text + (mib("lm_head") + mib("mtp")) / 1024
    need(abs(on_card + (mib("embed") + mib("vision")) / 1024 - total) < 0.005
         and abs(total - ex["file_bytes"] / 2**30) < 0.005, "the EXL3 file: embedding, vision and on-card parts add up")
    put("PeEmbedGiB", f"{mib('embed') / 1024:.2f}")
    put("PeVisionGiB", f"{mib('vision') / 1024:.2f}")
    put("PeOnCardGiB", f"{on_card:.2f}")
    names = {"exl3": r"EXL3 2.0\,bpw (two shards)", "iq2xxs": r"UD-IQ2\_XXS", "iq2s": r"UD-IQ2\_S",
             "mtp_full": "MTP sidecar, full", "mtp_shared": "MTP sidecar, shared", "dflash2": "DFlash2 drafter"}
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lL{3.4cm}rrrl}", r"\toprule",
             r"File & Source; used by & Parameters & GiB & Bits/weight & SHA-256\\", r"\midrule"]
    for r in rows:
        lines.append(f"{names[r['key']]} & {r['source'].replace('_', chr(92) + '_')}; {r['used_by']} & "
                     f"{r['parameters'] / 1e9:.2f}\\,B & {r['file_bytes'] / 2**30:.2f} & {r['bits_per_weight']:.2f} & "
                     f"\\q{{{r['sha256'][0][:12]}}}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Model files. Parameters and average bits per weight are counted from each file's own headers "
              r"(an EXL3 trellis tensor counts its $k\times n$ weights); sizes are on disk. The EXL3 file holds the MTP "
              r"layer, which the GGUFs do not, and a 16-bit vision encoder that a text-only run does not load. The full "
              r"MTP sidecar carries its own embedding "
              r"and output tables; ours, the shared sidecar, strips them and uses the base model's. SHA-256 prefixes; "
              r"every file but ours matched the Hub's LFS digest.}",
              r"\label{tab:files}", r"\end{table}"]
    TABLES["model_files.tex"] = ("% Generated by paper8/scripts/build_claims.py from testsuite/results/turing/"
                                 "model_files.json. Do not edit by hand.\n" + "\n".join(lines) + "\n")


def f2_per_prompt(arms, pooled):
    """F2 by prompt (exploratory, not registered): each arm's tokens over decode time per prompt, pooled over the three
    rounds, written as the table derived/f2_per_prompt.tex; the prompts pooled again must give the registered rate."""
    order = [("stock-0", "stock", "0", "none"), ("adopt-0", "w11b", "0", "none"), ("w13-0", "w13", "0", "none"),
             ("adopt-28k", "w11b", "28K", "none"), ("w13-28k", "w13", "28K", "none"),
             ("stock-mtp-0", "stock", "0", "MTP"), ("adopt-mtp-0", "w11b", "0", "MTP"),
             ("w13s1-mtp-0", "w13, mode 1", "0", "MTP"), ("w13s2-mtp-0", "w13, mode 2", "0", "MTP"),
             ("raf-mtp-0", "rafatxf", "0", "MTP"), ("adopt-mtp-28k", "w11b", "28K", "MTP"),
             ("w13s1-mtp-28k", "w13, mode 1", "28K", "MTP"), ("w13s2-mtp-28k", "w13, mode 2", "28K", "MTP"),
             ("raf-mtp-28k", "rafatxf", "28K", "MTP")]
    need(sorted(a for a, *_ in order) == sorted(arms), "the per-prompt table covers every F2 arm")
    lines = [r"\begin{table}[!htbp]", r"\centering\small", r"\begin{tabular}{lllrrrrc}", r"\toprule",
             r"Build & Context & Drafting & Code & Math & Prose & Pooled & Tokens decoded\\", r"\midrule"]
    eq = {}
    for a, build, ctx, draft in order:
        per, toks = {}, {}
        for task in ("code", "math", "prose"):
            rr = [x for v in arms[a].values() for x in v if x["task"] == task]
            need(len(rr) == 3, f"F2 {a} {task}: three rounds")
            per[task] = sum(x["new_tokens"] for x in rr) / sum(x["decode_wall_s"] for x in rr)
            toks[task] = sum(x["new_tokens"] for x in rr)
            need(all(x["new_tokens"] <= 256 for x in rr), f"F2 {a} {task}: at most 256 tokens a run")
        if a == "w13s2-mtp-0":
            for task in ("code", "math", "prose"):
                put(f"PeFtPpSTwo{task.capitalize()}", f"{per[task]:.1f}")
        if not draft == "MTP":
            need(max(per.values()) / min(per.values()) < 1.005, f"F2 {a}: undrafted rates within 0.5% across prompts")
        eq[a] = 3 / sum(1 / v for v in per.values())    # each prompt weighted equally (exploratory)
        allr = [x for v in arms[a].values() for x in v]
        need(abs(sum(x["new_tokens"] for x in allr) / sum(x["decode_wall_s"] for x in allr) - pooled[a]) < 1e-9,
             f"F2 {a}: the prompts pool back to the registered rate")
        lines.append(f"{build} & {ctx} & {draft} & {per['code']:.1f} & {per['math']:.1f} & {per['prose']:.1f} & "
                     f"{pooled[a]:.1f} & {toks['code']}/{toks['math']}/{toks['prose']}\\\\")
    for a, b, tag in (("w13s2-mtp-0", "raf-mtp-0", "OverRafZero"), ("w13s2-mtp-28k", "raf-mtp-28k", "OverRafTwentyEight")):
        put(f"PeFt{tag}EqPct", f"{100 * (eq[a] / eq[b] - 1):.1f}")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{F2's decode rates by prompt: tokens per second, each prompt pooled over its three rounds, "
              r"with the registered rate pooled over all three prompts, and the tokens each prompt decoded over its "
              r"three runs (code/math/prose, of 768 each; a run ends early at the end-of-sequence token). The rafatxf "
              r"arms ran on that fork's 4-bit cache. Exploratory: the registered comparisons pool the prompts.}",
              r"\label{tab:f2prompts}", r"\end{table}"]
    TABLES["f2_per_prompt.tex"] = ("% Generated by paper8/scripts/build_claims.py from testsuite/results/turing/ab/"
                                   "final2.jsonl. Do not edit by hand.\n" + "\n".join(lines) + "\n")


def speed_final2():
    """The speed of record F2 (research/TURING_W7_INT8_GEMV_2026-10-30.md, x-final2, registered 2026-10-31 13:42 EDT):
    pooled tok/s per arm from the raw rows, checked against final2.md's pooled column, and the per-round wins."""
    rows = [json.loads(x) for x in open(os.path.join(RT, "turing", "ab", "final2.jsonl")) if x.strip()]
    arms = {}
    for r in rows:
        arm, rnd = re.fullmatch(r"final2-(.+)-r(\d)", r["tag"]).groups()
        arms.setdefault(arm, {}).setdefault(int(rnd), []).append(r)
    pooled = {a: sum(x["new_tokens"] for rr in v.values() for x in rr) / sum(x["decode_wall_s"] for rr in v.values()
                                                                             for x in rr) for a, v in arms.items()}
    md = {r[0]: r for r in md_rows(os.path.join(RT, "turing", "ab", "final2.md")) if r[0] in pooled}
    need(len(md) == 14 and all(abs(float(md[a][7]) - pooled[a]) < 0.006 for a in pooled), "final2 pooled tok/s")
    need(all(x["cache_quant"] == ("4,4" if a.startswith("raf") else "8,8") for a, v in arms.items() for rr in v.values()
             for x in rr), "final2 caches: ours 8,8, rafatxf 4,4 (its best configuration, as registered)")
    need(all(x["exllamav3"].startswith("1.5.3") for a, v in arms.items() if a.startswith("raf") for rr in v.values()
             for x in rr), "rafatxf's arms report exllamav3 1.5.3")
    names = {"stock-0": "StockZero", "adopt-0": "AdoptZero", "w13-0": "WZero", "adopt-28k": "AdoptTwentyEight",
             "w13-28k": "WTwentyEight", "stock-mtp-0": "StockMtpZero", "adopt-mtp-0": "AdoptMtpZero",
             "w13s1-mtp-0": "SOneMtpZero", "w13s2-mtp-0": "STwoMtpZero", "raf-mtp-0": "RafMtpZero",
             "adopt-mtp-28k": "AdoptMtpTwentyEight", "w13s1-mtp-28k": "SOneMtpTwentyEight",
             "w13s2-mtp-28k": "STwoMtpTwentyEight", "raf-mtp-28k": "RafMtpTwentyEight"}
    for a, tag in names.items():
        put(f"PeFt{tag}", f"{pooled[a]:.2f}")
    f2_per_prompt(arms, pooled)
    per_round = {a: {k: sum(x["new_tokens"] for x in v) / sum(x["decode_wall_s"] for x in v) for k, v in rr.items()}
                 for a, rr in arms.items()}

    def wins(a, b):
        return sum(1 for k in per_round[a] if per_round[a][k] > per_round[b][k])

    for a, b, tag in (("w13-0", "stock-0", "UndraftedOverStock"), ("w13s2-mtp-0", "stock-mtp-0", "MtpOverStock"),
                      ("w13s2-mtp-0", "raf-mtp-0", "OverRafZero"), ("w13s2-mtp-28k", "raf-mtp-28k", "OverRafTwentyEight"),
                      ("w13s1-mtp-0", "raf-mtp-0", "SOneOverRafZero"), ("w13s1-mtp-0", "adopt-mtp-0", "SplitKMtp"),
                      ("w13-0", "adopt-0", "SplitKUndrafted")):
        change = 100 * (pooled[a] / pooled[b] - 1)
        need((change < 0) == (tag == "SOneOverRafZero"), f"{tag}: the direction the text states")
        put(f"PeFt{tag}Pct", f"{abs(change):.1f}")     # unsigned: the text says faster or slower
        put(f"PeFt{tag}Wins", wins(a, b))
    def rounds(a, b):
        return [100 * (per_round[a][k] / per_round[b][k] - 1) for k in sorted(per_round[a])]

    for a, b, tag in (("w13-0", "stock-0", "UndraftedOverStock"), ("w13s2-mtp-0", "stock-mtp-0", "MtpOverStock"),
                      ("w13s2-mtp-0", "raf-mtp-0", "OverRafZero"), ("w13s2-mtp-28k", "raf-mtp-28k", "OverRafTwentyEight")):
        r = rounds(a, b)
        put(f"PeFt{tag}Lo", f"{min(r):+.1f}")
        put(f"PeFt{tag}Hi", f"{max(r):+.1f}")
    for a, b, tag in (("w13s1-mtp-28k", "adopt-mtp-28k", "SplitKMtpTwentyEight"),
                      ("w13-28k", "adopt-28k", "SplitKUndraftedTwentyEight")):
        change = 100 * (pooled[a] / pooled[b] - 1)
        need(change > 0, f"{tag}: w13 faster than w11b")
        put(f"PeFt{tag}Pct", f"{change:.1f}")
        put(f"PeFt{tag}Wins", wins(a, b))
    acc = {}
    for r in rows:
        if r["mtp"]:
            a = acc.setdefault(r["tag"].rsplit("-r", 1)[0], [0, 0])
            a[0] += r["accepted"]
            a[1] += r["accepted"] + r["rejected"]
    pooled_acc = [a / n for a, n in acc.values()]
    need(len(acc) == 9, "nine drafted arms")
    put("PeFtAcceptLo", f"{min(pooled_acc):.2f}")
    put("PeFtAcceptHi", f"{max(pooled_acc):.2f}")
    temps = [int(x["gpu"]["temp_c"]) for x in rows]
    put("PeFtTempLo", min(temps))
    put("PeFtTempHi", max(temps))
    load_avg = [float(m) for m in re.findall(r"load average: ([\d.]+),", open(os.path.join(RT, "turing", "ab",
                                                                                      "final2-load.log")).read())]
    need(len(load_avg) == 2, "final2's load log: start and end")
    put("PeFtLoadStart", f"{load_avg[0]:.1f}")
    put("PeFtLoadEnd", f"{load_avg[1]:.1f}")
    times = sorted(x["time"] for x in rows)
    put("PeFtFirst", times[0][11:16])
    put("PeFtLast", times[-1][11:16])
    need(times[0][:10] == times[-1][:10] == "2026-10-31", "final2 ran on one day")
    put("PeFtRows", len(rows))
    put("PeFtUndraftedRatio", f"{pooled['w13-0'] / pooled['stock-0']:.2f}")
    put("PeFtMtpRatio", f"{pooled['w13s2-mtp-0'] / pooled['stock-mtp-0']:.2f}")
    put("PeTitleMtpRate", f"{pooled['w13s2-mtp-0']:.0f}")    # the abstract's rate: drafted, empty context
    put("PeFtWMs", f"{1000 / pooled['w13-0']:.1f}")              # w13 undrafted, ms per token, unprofiled
    put("PeTitleRatioLo", f"{pooled['w13s2-mtp-0'] / pooled['stock-mtp-0']:.1f}")   # the title: drafted ratio
    put("PeTitleRatioHi", f"{pooled['w13-0'] / pooled['stock-0']:.1f}")             # the title: undrafted ratio
    need(wins("w13s2-mtp-0", "raf-mtp-0") == 3 and wins("w13s2-mtp-28k", "raf-mtp-28k") == 3
         and wins("w13s1-mtp-0", "raf-mtp-0") == 0, "R3: mode 2 wins every round, mode 1 none at fill 0")
    # final1 (registered S1-S4): w11b over stock at 28K, means of per-prompt tok/s
    f1 = {r[0]: r for r in md_rows(os.path.join(RT, "turing", "ab", "final1.md"))}
    put("PeFoneStockTwentyEight", f1["stock-28k"][5])
    put("PeFoneAdoptTwentyEight", f1["adopt-28k"][5])
    put("PeFoneRatioTwentyEight", f"{float(f1['adopt-28k'][5]) / float(f1['stock-28k'][5]):.2f}")
    put("PeFoneRatioMtpTwentyEight", f"{float(f1['adopt-mtp-28k'][5]) / float(f1['stock-mtp-28k'][5]):.2f}")


def audit():
    """The cross-session audit of the 2080 Ti speed table (research/prereg/2026-11-02-four-resources-formal-audit.md,
    testsuite/results/ideas/four-resources-formal/audit.json): session effects against final2, in time per token."""
    a = json.load(open(os.path.join(RT, "ideas", "four-resources-formal", "audit.json")))["audit"]
    put("PeAuditSessions", a["graph"]["sessions"])
    put("PeAuditChi", f"{a['chi2_over_df']:.1f}")
    put("PeAuditFloorUndrafted", f"{a['resid_rms_pct']['undrafted']:.1f}")
    put("PeAuditFloorDrafted", f"{a['resid_rms_pct']['drafted']:.1f}")
    eff = {k: v for k, v in a["session_effects"].items() if v["reference"] == "final2" and k != "final2"}
    slow = {k: v["pct"] for k, v in eff.items() if v["ci95_pct"][0] > 0}
    close = [k for k, v in eff.items() if abs(v["pct"]) <= 2]
    need(sorted(slow) == ["final1.clean", "power1", "rafatxf1"], "three sessions slower than final2")
    need(all(v["pct"] > -2 for v in eff.values()), "no session faster than final2 by more than 2%")
    put("PeAuditSlowLo", f"{min(slow.values()):.1f}")
    put("PeAuditSlowHi", f"{max(slow.values()):.1f}")
    put("PeAuditCloseN", len(close))
    put("PeAuditCloseWord", COUNT[len(close)])


def bytes_per_token():
    """Weight bytes read per decoded token: EXL3 from the census's byte accounting (text layers plus the head; the
    embedding, vision and MTP tensors are not read undrafted), the GGUFs from their tensor tables
    (tools/turing/gguf_weight_bytes.py: every tensor but token_embd). NInferno's published undrafted rows on the same
    card (research/raw/turing-competition-refresh-2026-11-04/forks/ninferno/bench_rows_rtx2080ti.jsonl, 240 W, its
    own prompts) are quoted beside them, not measured here."""
    txt = open(os.path.join(RT, "turing", "census", "weight_bytes_qwen38-exl3-2.0.txt")).read()
    text_gib = float(re.search(r"text layers ([\d.]+) GiB", txt).group(1))
    head_mib = float(re.search(r"([\d.]+) MiB\s+lm_head", txt).group(1))
    exl3 = text_gib * 2**30 + head_mib * 2**20
    g = json.load(open(os.path.join(RT, "turing", "census", "weight_bytes_gguf.json")))
    xxs, s2 = g["Qwen3.8-27B-UD-IQ2_XXS.gguf"], g["Qwen3.8-27B-UD-IQ2_S.gguf"]
    need(xxs["file_bytes"] == 7266070528 and s2["file_bytes"] == 8371970048, "the audited GGUF files")
    put("PeTokGbExl", f"{exl3 / 1e9:.2f}")
    put("PeTokGbIxxs", f"{xxs['per_token_bytes'] / 1e9:.2f}")
    put("PeTokGbIs", f"{s2['per_token_bytes'] / 1e9:.2f}")
    put("PeFileGbIxxs", f"{xxs['file_bytes'] / 1e9:.2f}")
    w13 = float(M["PeFtWZero"])
    put("PeGbsW", f"{exl3 * w13 / 1e9:.0f}")
    rows = [json.loads(x) for x in open(os.path.join(REPO, "research", "raw", "turing-competition-refresh-2026-11-04",
                                                       "forks", "ninferno", "bench_rows_rtx2080ti.jsonl")) if x.strip()]
    nif = [r for r in rows if r["model"] == "Qwen/Qwen3.8-27B" and r["metric"] == "decode_aggregate"
           and r["concurrency"] == 1 and r["prompt_set"] == "short_chat_20req"]
    und = {r["spec"]: r["value"] for r in nif}
    deep = {r["spec"]: (r["value"], r["prompt_len"]) for r in rows if r["model"] == "Qwen/Qwen3.8-27B"
            and r["metric"] == "decode" and r["concurrency"] == 1}
    put("PeNifDeepUndrafted", f"{deep['none'][0]:.1f}")
    put("PeNifDeepDflash", f"{deep['dflash2'][0]:.1f}")
    put("PeNifDeepTokens", num(deep["none"][1]))
    need(abs(und["none"] - 43.74) < 1e-9 and all(r["machine"]["power_limit_w"] == 240.0 for r in nif),
         "NInferno's published short-chat rows, 240 W")
    put("PeNifUndrafted", f"{und['none']:.1f}")
    put("PeNifDflash", f"{und['dflash2']:.1f}")
    put("PeGbsNif", f"{xxs['per_token_bytes'] * und['none'] / 1e9:.0f}")
    put("PeNifLeadPct", f"{100 * (w13 / und['none'] - 1):.0f}")
    put("PeTokBytesPct", f"{100 * (xxs['per_token_bytes'] / exl3 - 1):.0f}")


def confirmation_timing(base):
    """Compare client first-to-last-token seconds with reported server decode milliseconds.

    Only the completed registered and fit sessions support this sentence; in-process rows have no server timing.
    The published upper bound is rounded UP to whole milliseconds, never to a value below an observed difference.
    """
    close_ms = 0.1
    differences = []
    for directory, expected_n, expected_close in (("registered-20261104T205440", 78, 41),
                                                  ("fit-20261104T225354", 36, 16)):
        rows = [json.loads(x) for x in open(os.path.join(base, directory, "rows.jsonl")) if x.strip()]
        paired = [r for r in rows if (r.get("server_timings") or {}).get("predicted_ms") is not None]
        need(len(paired) == expected_n, f"{directory}: every audited client/server timing row is present")
        need(len({(r["tag"], r["task"]) for r in paired}) == len(paired),
             f"{directory}: unique client/server timing rows")
        delta = []
        for row in paired:
            client_s, server_ms = row["decode_wall_s"], row["server_timings"]["predicted_ms"]
            need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (client_s, server_ms)),
                 f"{directory}: finite positive client seconds and server milliseconds")
            delta.append(abs(1000 * client_s - server_ms))
        need(sum(d <= close_ms for d in delta) == expected_close,
             f"{directory}: audited count within 0.1 ms")
        differences.extend(delta)
    upper_ms = math.ceil(max(differences))
    close_n = sum(d <= close_ms for d in differences)
    need(len(differences) == 114 and close_n == 57 and upper_ms == 47,
         "confirmation timing: 114 paired rows, 57 within 0.1 ms, maximum rounds up to 47 ms")
    need(all(d <= upper_ms for d in differences), "confirmation timing: whole-ms bound covers every row")
    put("PeCfTimingMaxMs", upper_ms)
    put("PeCfTimingN", len(differences))
    put("PeCfTimingCloseN", close_n)
    put("PeCfTimingCloseMs", f"{close_ms:.1f}")


def confirmation():
    """The registered confirmation session, its follow-up fit session (Amendment 1) and the stopped session 1
    (research/prereg/2026-11-04-p8-confirmation-speed.md), from the rows copied after both sessions ended: pooled
    speeds recomputed with analyze_confirm.py's own functions and checked against its results.json in each session."""
    sys.path.insert(0, os.path.join(REPO, "tools", "turing", "confirm"))
    import analyze_confirm as AC
    base = os.path.join(RT, "turing", "confirm")
    confirmation_timing(base)
    sess = {}
    for tag, d in (("reg", "registered-20261104T205440"), ("fit", "fit-20261104T225354"),
                   ("stop", "stopped-20261104T194157")):
        s_ = AC.load(os.path.join(base, d, "rows.jsonl"))
        res = json.load(open(os.path.join(base, d, "results.json")))
        need(all(abs(s_[k]["pooled"] - res["pooled"][k]["pooled"]) < 1e-9 for k in s_), f"{d}: pooled speeds recompute")
        sess[tag] = (s_, res)
    reg, rres = sess["reg"]
    fit, fres = sess["fit"]
    names = {"w13-165-nd-0": "WNdZero", "w13-165-mtp-0": "WMtpZero", "w13-165-nd-28k": "WNdDeep",
             "w13-165-mtp-28k": "WMtpDeep", "stock165-nd-0": "StockNdZero", "stock165-mtp-0": "StockMtpZero",
             "stock165-nd-28k": "StockNdDeep", "stock165-mtp-28k": "StockMtpDeep", "stock166-nd-0": "SixNdZero",
             "stock166-mtp-0": "SixMtpZero", "w13-166-nd-0": "WSixNdZero", "w13-166-mtp-0": "WSixMtpZero",
             "raf44-mtp-0": "RafZero", "raf44-mtp-28k": "RafDeep", "nif-xxs-0": "NifXxsZero",
             "nif-xxs-28k": "NifXxsDeep", "nif-s-0": "NifSZero", "nif-s-28k": "NifSDeep",
             "lcpp9-xxs-0": "LcppZero", "lcpp9-xxs-28k": "LcppDeep", "ik-xxs-0": "IkXxsZero", "ik-s-0": "IkSZero"}
    for k, tag in names.items():
        put(f"PeCf{tag}", f"{reg[k]['pooled']:.2f}")
    for k, tag in (("buun-nd-0", "BuunNdZero"), ("buun-mtp-0", "BuunMtpZero"), ("lcpp9-s-mtp-0", "LcppMtpZero"),
                   ("lcpp9-s-mtp-28k", "LcppMtpDeep")):
        put(f"PeCf{tag}", f"{fit[k]['pooled']:.2f}")
    need(len(fit["buun-nd-0"]["rounds"]) == 1, "buun undrafted ran one round (cut by the cap)")
    f2 = rres["f2"]
    need(all(v["within_floor"] for v in f2.values()), "F2 stands: every w13 condition within the cross-session floor")
    put("PeCfFtwoMin", f"{min(v['change_pct'] for v in f2.values()):+.1f}")
    put("PeCfFtwoMax", f"{max(v['change_pct'] for v in f2.values()):+.1f}")
    stop = sess["stop"][1]["f2"]
    put("PeCfStopMax", f"{max(abs(v['change_pct']) for v in stop.values()):.1f}")
    comp = rres["comparisons"]
    for key, tag in (("over_stock_nd_0", "StockNdZero"), ("over_stock_mtp_0", "StockMtpZero"),
                     ("over_stock_nd_28k", "StockNdDeep"), ("over_stock_mtp_28k", "StockMtpDeep"),
                     ("over_lcpp_nd_0", "LcppZero"), ("over_lcpp_nd_28k", "LcppDeep"),
                     ("over_ik_xxs_0", "IkXxsZero"), ("over_ik_s_0", "IkSZero")):
        need(comp[key]["faster_every_round"], f"{key}: every round")
        put(f"PeCfX{tag}", f"{comp[key]['ratio']:.2f}")
    for key, tag in (("over_raf_0", "RafZero"), ("over_raf_28k", "RafDeep"), ("over_nif_xxs_0", "NifXxsZero"),
                     ("over_nif_xxs_28k", "NifXxsDeep"), ("over_nif_s_0", "NifSZero"), ("over_nif_s_28k", "NifSDeep")):
        c = comp[key]
        need(c["faster_every_round"], f"{key}: every round")
        put(f"PeCfP{tag}", f"{c['pooled_pct']:.1f}")
        put(f"PeCfP{tag}Lo", f"{min(c['per_round_pct']):.1f}")
        put(f"PeCfP{tag}Hi", f"{max(c['per_round_pct']):.1f}")
    # the lead over rafatxf with each prompt weighted equally (exploratory, beside the registered pooling, which weighs
    # a prompt by the tokens it decoded: the exllamav3 runs can end early at the end-of-sequence token)
    rows_reg = [json.loads(x) for x in open(os.path.join(base, "registered-20261104T205440", "rows.jsonl")) if x.strip()]

    def eqw(cond):
        rr = [r for r in rows_reg if r["tag"].rsplit("-r", 1)[0] == "confirm-" + cond]
        need(abs(sum(x["new_tokens"] for x in rr) / sum(x["decode_wall_s"] for x in rr) - reg[cond]["pooled"]) < 1e-9,
             f"{cond}: the rows pool to the session's rate")
        per = [sum(x["new_tokens"] for x in rr if x["task"] == t) / sum(x["decode_wall_s"] for x in rr if x["task"] == t)
               for t in ("code", "math", "prose")]
        return 3 / sum(1 / v for v in per)
    put("PeCfPRafZeroEq", f"{100 * (eqw('w13-165-mtp-0') / eqw('raf44-mtp-0') - 1):.1f}")
    put("PeCfPRafDeepEq", f"{100 * (eqw('w13-165-mtp-28k') / eqw('raf44-mtp-28k') - 1):.1f}")
    six = [comp[k]["pooled_pct"] for k in ("stock166_vs_165_nd", "stock166_vs_165_mtp", "w13_166_vs_165_nd",
                                            "w13_166_vs_165_mtp")]
    need(all(abs(x) <= 5 for x in six) and not any(comp[k]["faster_every_round"] and comp[k]["pooled_pct"] > 5
                                                      for k in comp if "166" in k), "1.6.6: no change by the rule")
    put("PeCfSixMax", f"{max(abs(x) for x in six):.1f}")
    put("PeCfIkVsLcpp", f"{-comp['ik_vs_lcpp_nd_0']['pooled_pct']:.1f}")
    need(comp["ik_vs_lcpp_nd_0"]["slower_every_round"], "ik slower than llama.cpp undrafted, every round")
    put("PeCfNifSBelowXxs", f"{100 * (1 - reg['nif-s-0']['pooled'] / reg['nif-xxs-0']['pooled']):.1f}")
    need(all(rres["fastest"][k]["holds"] for k in ("nd-0", "mtp-0", "nd-28k", "mtp-28k")), "fastest: registered rule")
    fc = fres["comparisons"]
    need(fc["over_buun_mtp_0"]["faster_every_round"] and fc["over_lcpp_mtp_0"]["faster_every_round"]
         and fc["over_lcpp_mtp_28k"]["faster_every_round"], "fit session: w13 ahead in every round")
    for k, tag in (("w13-165-nd-0", "NdZero"), ("w13-165-mtp-0", "MtpZero"), ("w13-165-nd-28k", "NdDeep"),
                   ("w13-165-mtp-28k", "MtpDeep")):
        put(f"PeCfFitW{tag}", f"{fit[k]['pooled']:.2f}")   # the follow-up session's own w13 anchors (Table 2)
    # Amendment 2's session (5 Nov): its DFlash2 arm failed at its first start, so it ran anchors and NInferno undrafted
    d = os.path.join(base, "df2-20261105T054600")
    dfa = AC.load(os.path.join(d, "rows.jsonl"))
    ares, aadd = json.load(open(os.path.join(d, "results.json"))), json.load(open(os.path.join(d, "results-df2.json")))
    need(all(abs(dfa[k]["pooled"] - ares["pooled"][k]["pooled"]) < 1e-9 for k in dfa), "df2 session: pooled recompute")
    need(sorted(dfa) == ["nif-xxs-0", "w13-165-mtp-0", "w13-165-mtp-28k", "w13-165-nd-0"]
         and all(len(v["rounds"]) == 3 for v in dfa.values()), "df2 session: four conditions, three rounds each")
    journal = [json.loads(x) for x in open(os.path.join(d, "journal.jsonl"))]
    need(any(e["kind"] == "prepare" and e.get("config") == "nif-xxs-df2" and not e["ok"] for e in journal),
         "df2 session: the DFlash2 arm failed at its first start")
    need(all(v["within_floor"] for v in ares["f2"].values()), "df2 session: anchors within F2's floor")
    for k, tag in (("w13-165-nd-0", "WNdZero"), ("w13-165-mtp-0", "WMtpZero"), ("w13-165-mtp-28k", "WMtpDeep"),
                   ("nif-xxs-0", "NifXxsZero")):
        put(f"PeDfa{tag}", f"{dfa[k]['pooled']:.2f}")
    c = ares["comparisons"]["over_nif_xxs_0"]
    need(c["faster_every_round"], "df2 session: w13 ahead of NInferno undrafted in every round")
    put("PeDfaPNifXxsZero", f"{c['pooled_pct']:.1f}")
    put("PeDfaMtpGain", f"{aadd['comparisons']['w13_mtp_gain_0']['ratio']:.2f}")
    put("PeCfXBuunNdZero", f"{fc['over_buun_nd_0']['ratio']:.2f}")
    put("PeCfXBuunMtpZero", f"{fc['over_buun_mtp_0']['ratio']:.2f}")
    put("PeCfXLcppMtpZero", f"{fc['over_lcpp_mtp_0']['ratio']:.1f}")
    put("PeCfXLcppMtpDeep", f"{fc['over_lcpp_mtp_28k']['ratio']:.1f}")
    q = fres["quality_buun"]
    need(q["n"] == 100, "buun's GSM8K check: 100 items")
    put("PeCfBuunAcc", f"{q['acc']:.1f}")
    put("PeCfBuunRefAcc", f"{q['ref_acc']:.1f}")
    put("PeCfBuunDiff", pts(q["diff_pts"]))
    put("PeCfBuunTen", q["buun_only"])
    put("PeCfBuunOtt", q["ref_only"])
    put("PeCfBuunP", f"{q['mcnemar_p']:.2g}")
    rungs = json.load(open(os.path.join(base, "fit-20261104T225354", "rungs.json")))
    need(rungs["lcpp-xxs-df"]["rung"] is None and rungs["ik-s-mtp"]["rung"] is None
         and rungs["lcpp-s-mtp"]["rung"] == "R3" and rungs["lcpp-s-mtp"]["gpu_layers"] is None
         and rungs["buun-nd"]["rung"] == rungs["buun-mtp"]["rung"] == "R0", "the fit ladder's outcomes")
    tok = json.load(open(os.path.join(RT, "turing", "census", "weight_bytes_gguf.json")))
    exl3_txt = open(os.path.join(RT, "turing", "census", "weight_bytes_qwen38-exl3-2.0.txt")).read()
    exl3 = (float(re.search(r"text layers ([\d.]+) GiB", exl3_txt).group(1)) * 2**30
            + float(re.search(r"([\d.]+) MiB\s+lm_head", exl3_txt).group(1)) * 2**20)
    put("PeCfGbsW", f"{exl3 * reg['w13-165-nd-0']['pooled'] / 1e9:.0f}")
    put("PeCfGbsNif", f"{tok['Qwen3.8-27B-UD-IQ2_XXS.gguf']['per_token_bytes'] * reg['nif-xxs-0']['pooled'] / 1e9:.0f}")
    dflash2_session(AC, base)


def dflash2_session(AC, base):
    """Amendment 3's session (5 Nov): NInferno's DFlash2 arm on its own drafter-attached conversion, beside w13 anchors
    and NInferno undrafted. Pooled rates and the registered comparisons from its results files; the per-prompt split
    (exploratory) from per_prompt.py's per-prompt.json; the DFlash2 arm's contexts from its start logs."""
    d = os.path.join(base, "df2b-20261105T061246")
    s = AC.load(os.path.join(d, "rows.jsonl"))
    res, add = json.load(open(os.path.join(d, "results.json"))), json.load(open(os.path.join(d, "results-df2.json")))
    need(all(abs(s[k]["pooled"] - res["pooled"][k]["pooled"]) < 1e-9 for k in s), "df2b session: pooled recompute")
    need(all(v["within_floor"] for v in res["f2"].values()), "df2b session: anchors within F2's floor")
    for k, tag in (("w13-165-nd-0", "WNdZero"), ("w13-165-mtp-0", "WMtpZero"), ("w13-165-mtp-28k", "WMtpDeep"),
                   ("nif-xxs-0", "NifXxsZero"), ("nif-xxs-df2-0", "NifDfZero"), ("nif-xxs-df2-28k", "NifDfDeep")):
        need(len(s[k]["rounds"]) == 3, f"df2b session: {k} ran three rounds")
        put(f"PeDfb{tag}", f"{s[k]['pooled']:.2f}")
    comp = add["comparisons"]
    for key, tag in (("over_nif_df2_0", "Zero"), ("over_nif_df2_28k", "Deep")):
        c = comp[key]
        put(f"PeDfbPNifDf{tag}", f"{c['pooled_pct']:.1f}")
        put(f"PeDfbPNifDf{tag}Lo", f"{min(c['per_round_pct']):.1f}")
        put(f"PeDfbPNifDf{tag}Hi", f"{max(c['per_round_pct']):.1f}")
        put(f"PeDfbPNifDf{tag}Wins", str(c["wins"]))
    put("PeDfbNifDfGain", f"{comp['nif_df2_gain_0']['ratio']:.2f}")
    put("PeDfbMtpGain", f"{comp['w13_mtp_gain_0']['ratio']:.2f}")
    put("PeDfbFastestZero", "holds" if res["fastest"]["mtp-0"]["holds"] else "fails")
    put("PeDfbFastestDeep", "holds" if res["fastest"]["mtp-28k"]["holds"] else "fails")
    ctx = {}
    for cond in ("prepare-nif-xxs-df2", "nif-xxs-df2-28k"):
        m = re.findall(r"plan \| rung (\d+) of (\d+): ([\d,]+) tokens", open(os.path.join(d, "logs", f"{cond}.log")).read())
        need(len(set(m)) == 1, f"df2b session: {cond} started at one rung every time")
        ctx[cond] = m[0]
    put("PeDfbNifCtx", ctx["prepare-nif-xxs-df2"][2])
    put("PeDfbNifDeepCtx", ctx["nif-xxs-df2-28k"][2])
    put("PeDfbNifDeepRung", f"{ctx['nif-xxs-df2-28k'][0]} of {ctx['nif-xxs-df2-28k'][1]}")
    pp = json.load(open(os.path.join(d, "per-prompt.json")))
    for cond, tag in (("w13-165-mtp-0", "WMtpZero"), ("nif-xxs-df2-0", "NifDfZero"), ("w13-165-mtp-28k", "WMtpDeep"),
                      ("nif-xxs-df2-28k", "NifDfDeep"), ("nif-xxs-0", "NifXxsZero")):
        for task in ("code", "math", "prose"):
            put(f"PeDfbPp{tag}{task.capitalize()}", f"{pp['rates'][f'{cond}|{task}']['all']:.1f}")
    for cond, tag in (("nif-xxs-df2-0", "Zero"), ("nif-xxs-df2-28k", "Deep")):
        for task in ("code", "math", "prose"):
            put(f"PeDfbAcc{tag}{task.capitalize()}", f"{100 * pp['acceptance'][cond][task]['rate']:.0f}")
    put("PeDfbFtwoMax", f"{max(abs(v['change_pct']) for v in res['f2'].values()):.1f}")
    # the prose the paper writes about the split by prompt (exploratory)
    pr = pp["rates"]
    need(all(0.95 < pr[f"nif-xxs-df2-0|{t}"]["all"] / pr[f"w13-165-mtp-0|{t}"]["all"] < 1.05 for t in ("code", "math")),
         "empty context: DFlash2 level with w13 on code and math (within 5%)")
    need(pr["nif-xxs-df2-0|prose"]["all"] < pr["nif-xxs-0|prose"]["all"] < pr["w13-165-mtp-0|prose"]["all"],
         "empty context: DFlash2 on prose below its own undrafted rate and w13's")
    need(all(pr[f"w13-165-mtp-28k|{t}"]["all"] > pr[f"nif-xxs-df2-28k|{t}"]["all"] for t in ("code", "math", "prose")),
         "28K: w13 ahead on every prompt pooled")
    need(sum(pr["nif-xxs-df2-28k|math"][r] > pr["w13-165-mtp-28k|math"][r] for r in "123") == 2,
         "28K math: DFlash2 faster in two of the three rounds")


def census_groups():
    """GAP 10: kernel time by group, the stock wheel's censuses against w13's (tools/turing/census_groups.py, recomputed
    here from each census report and checked against the committed table)."""
    sys.path.insert(0, os.path.join(REPO, "tools", "turing"))
    import census_groups as CG
    table = json.load(open(os.path.join(RT, "turing", "census", "groups_stock_vs_w13.json")))["captures"]
    caps = {label: CG.capture(d, sec) for label, d, sec in CG.CAPTURES}
    need(all(abs(caps[k]["kernel_us_per_token"] - table[k]["kernel_us_per_token"]) < 1e-6 for k in caps),
         "the committed group table matches the census reports")
    tags = {"stock 1.6.5, undrafted, fill 0": "StockNd", "stock 1.6.5, MTP, fill 0": "StockMtp",
            "w13 mode 2, MTP, fill 0": "W", "w13 mode 2, MTP, 28K": "WDeep"}
    short = {"matrix products": "Matrix", "attention": "Attn", "GDN recurrent state": "Gdn", "small kernels": "Small"}
    for label, tag in tags.items():
        c = caps[label]
        put(f"PeCg{tag}Ms", f"{c['ms_per_token_mean']:.2f}")
        for g, sg in short.items():
            put(f"PeCg{tag}{sg}Us", num(round(c["group_us_per_token"][g])))
            put(f"PeCg{tag}{sg}Share", f"{100 * c['group_share'][g]:.1f}")
    st, w = caps["stock 1.6.5, MTP, fill 0"], caps["w13 mode 2, MTP, fill 0"]
    put("PeCgMsCut", f"{100 * (1 - w['ms_per_token_mean'] / st['ms_per_token_mean']):.0f}")
    for g, sg in short.items():
        put(f"PeCg{sg}Cut", f"{100 * (1 - w['group_us_per_token'][g] / st['group_us_per_token'][g]):.0f}")
    need(abs(100 * caps["stock 1.6.5, undrafted, fill 0"]["group_share"]["matrix products"] - 91.9) < 0.05,
         "the undrafted stock grouping reproduces D1's matrix share")


CENSUS_RATE = 550e9  # bytes/s: the card's streaming rate used for every floor in section 4
RATED_RATE = 616e9   # bytes/s: the card's rated memory bandwidth, for the split's sensitivity


def byte_us_per_mweights(k, rate=CENSUS_RATE):
    """Microseconds to read a million K-bit weights at a streaming rate (CENSUS_RATE unless given)."""
    return k / 8 * 1e6 / rate * 1e6


def call_split(census, rate=CENSUS_RATE):
    """One census's fit of each matrix call as a fixed cost plus a cost per weight, by bits per weight
    (<census>/int8_by_bits.md, from tools/turing/census_bits.py), summed over the calls of one undrafted token. The output
    head, which the fit leaves out, is in the token's matrix time and floor but not in the split."""
    rows = md_rows(os.path.join(RT, "turing", "census", census, "int8_by_bits.md"))
    calls = [r for r in rows if len(r) == 7 and r[0] != "role"]
    fits = {int(r[0]): (float(r[2]), float(r[3])) for r in rows if len(r) == 5 and r[0] != "K"}
    tokens = next(int(r[2]) for r in calls if r[0] == "head")   # the head runs once per token
    n = t_all = floor_all = t_fit = floor_fit = fixed = extra = 0.0
    for role, k, launches, med, mb, _, _ in calls:
        k, per_tok, med, mb = int(k), int(launches) / tokens, float(med), float(mb)
        floor = mb * 1e6 / rate * 1e6
        n, t_all, floor_all = n + per_tok, t_all + per_tok * med, floor_all + per_tok * floor
        if role != "head" and k in fits:
            a, b = fits[k]
            t_fit, floor_fit = t_fit + per_tok * med, floor_fit + per_tok * floor
            fixed += per_tok * a
            extra += per_tok * (mb * 8 / k) * (b - byte_us_per_mweights(k, rate))
    gate_up = {int(r[1]): float(r[3]) for r in calls if r[0] == "mlp_gate_up"}
    return {"n": n, "matrix": t_all / 1e3, "floor": floor_all / 1e3, "gap": (t_fit - floor_fit) / 1e3,
            "fixed": fixed / 1e3, "extra": extra / 1e3, "fits": fits, "gate_up": gate_up}


def call_costs():
    """Where the matrix kernels' time above their byte floor goes (research-ledger-reference). An exploratory split of two
    registered censuses: the w9 build's (d0-f1-mgemmn0) and the final build's (d0-w12p, whose one-row kernels w13 runs
    unchanged when undrafted; its captures' SHA-256 match the desktop originals)."""
    for census, tag, want in (("d0-f1-mgemmn0", "", (18.17, 4.69, 1.56)), ("d0-w12p", "Fin", (17.47, 4.57, 0.93))):
        s = call_split(census)
        need(round(s["n"]) == 401 and abs(s["matrix"] - want[0]) < 0.01 and abs(s["floor"] - 11.92) < 0.01,
             f"the call split ({census}): 401 matrix calls per token, {want[0]} ms, a floor of 11.92 ms")
        need(abs(s["fixed"] - want[1]) < 0.01 and abs(s["extra"] - want[2]) < 0.01,
             f"the call split ({census}): {want[1]} ms fixed per call, {want[2]} ms per weight above the byte cost")
        need(s["gate_up"][1] < s["gate_up"][2], f"the call split ({census}): a 1-bit gate-up call is no slower")
        rated = call_split(census, RATED_RATE)
        need(rated["fixed"] / rated["gap"] > 0.5, f"the call split ({census}): fixed time is most of the excess "
             "even at the rated 616 GB/s")
    # the registered extension on Ada (research/prereg/2026-11-05-p8-call-split-ada.md): the laptop's RTX 4080
    # Laptop GPU, stock exllamav3 1.6.5; its read rate is the rated 432 GB/s scaled by the recorded memory clock
    ada_dir = os.path.join(RT, "ada", "census", "d0-stock-mgemmn0")
    ada = json.load(open(os.path.join(ada_dir, "split.json")))
    rated = 432e9 * ada["memory_clock_mhz_during_decode"] / 9001
    for frac, key in ((550 / 616, "primary"), (1.0, "rated")):
        again = call_split("../../ada/census/d0-stock-mgemmn0", rated * frac)
        need(abs(again["fixed"] - ada[key]["fixed_ms"]) < 1e-6 and abs(again["gap"] - ada[key]["gap_ms"]) < 1e-6,
             f"the Ada split ({key}) recomputes")
    v = ada["verdict"]
    need(not v["P1"] and v["P2"] and v["P3"], "the Ada extension: P1 fails, P2 and P3 hold, as reported")
    ratios = sorted(ada["one_bit_over_two_bit"].values())
    put("PeAdaOneFasterLo", f"{100 * (1 - ratios[-1]):.0f}")
    put("PeAdaOneFasterHi", f"{100 * (1 - ratios[0]):.0f}")
    put("PeAdaMemMhz", f"{ada['memory_clock_mhz_during_decode']:,.0f}")
    rep2 = json.load(open(os.path.join(RT, "ada", "census", "d0-stock-mgemmn0-perf", "split.json")))
    need(rep2["memory_clock_mhz_during_decode"] == ada["memory_clock_mhz_during_decode"] and not rep2["verdict"]["P1"]
         and rep2["verdict"]["P2"] and rep2["verdict"]["P3"], "the Ada replicate: same memory clock, same verdicts")
    r2 = sorted(rep2["one_bit_over_two_bit"].values())
    # Amendment 2: the GPU limit raised to 175 W (nvidia-powerd, ASUS dynamic boost 25 W), memory at 9,001 MHz
    bdir = os.path.join(RT, "ada", "census", "d0-stock-mgemmn0-boost")
    boost = json.load(open(os.path.join(bdir, "split.json")))
    need(boost["memory_clock_mhz_during_decode"] >= 8001 and not boost["verdict"]["P1"] and boost["verdict"]["P2"]
         and boost["verdict"]["P3"], "the raised-limit run: memory at full clock, P1 fails, P2 and P3 hold")
    rb = sorted(boost["one_bit_over_two_bit"].values())
    need(all(boost["one_bit_over_two_bit"][k] < ada["one_bit_over_two_bit"][k] for k in boost["one_bit_over_two_bit"]),
         "B2 fails: every role's 1-bit advantage grew at the raised limit")
    put("PeAdaBoostOneFasterLo", f"{100 * (1 - rb[-1]):.0f}")
    put("PeAdaBoostOneFasterHi", f"{100 * (1 - rb[0]):.0f}")
    put("PeAdaBoostMemMhz", f"{boost['memory_clock_mhz_during_decode']:,.0f}")
    brep = json.load(open(os.path.join(bdir, "report.json")))["nodraft"]
    b_ms = sum(t["ms_per_token"] for t in brep["tasks"]) / len(brep["tasks"])
    a_ms = sum(t["ms_per_token"] for t in json.load(open(os.path.join(ada_dir, "report.json")))["nodraft"]["tasks"]) / 3
    def sm_ghz(folder):
        rows = list(csv.DictReader(open(glob.glob(os.path.join(folder, "telemetry-*.csv"))[0]), skipinitialspace=True))
        busy = [float(r["clocks.current.sm [MHz]"].split()[0]) for r in rows
                if float(r["utilization.gpu [%]"].split()[0]) >= 50]
        return statistics.median(busy) / 1000
    ada_rep = json.load(open(os.path.join(ada_dir, "report.json")))["nodraft"]
    fin = json.load(open(os.path.join(RT, "turing", "census", "d0-w12p", "report.json")))["nodraft"]
    put("PeCensusFinMs", f"{sum(t['ms_per_token'] for t in fin['tasks']) / len(fin['tasks']):.1f}")


def mechanism():
    """The why: the stock census (E1-E2), the int8 fit (W7), fd75d at depth (W11 census and x-depth2), split-K verify
    rows (test 3, w13 extension), the priced graph (test 2) and the attention bench."""
    w2 = json.load(open(os.path.join(RT, "turing", "census", "w2", "report.json")))["nodraft"]
    fam = {f["family"]: f["share"] for f in w2["families"]}
    share = fam["exl3_gemv_kernel"] + fam["exl3_mgemm_kernel"] + fam["exl3_gemm_kernel"]
    need(abs(share - 0.92) < 0.005, "the census: matrix kernels 92% of undrafted kernel time")
    put("PeCensusMatrix", f"{100 * share:.0f}")
    put("PeCensusGemv", f"{100 * fam['exl3_gemv_kernel']:.1f}")
    put("PeCensusMgemm", f"{100 * fam['exl3_mgemm_kernel']:.1f}")
    put("PeCensusGemm", f"{100 * fam['exl3_gemm_kernel']:.1f}")
    busy = [t["busy_frac"] for t in w2["tasks"]]
    put("PeCensusBusy", f"{100 * min(busy):.0f}")
    put("PeCensusMs", f"{sum(t['ms_per_token'] for t in w2['tasks']) / len(w2['tasks']):.1f}")
    idle = [t["idle_ms_per_token"] for t in w2["tasks"]]
    put("PeCensusIdleLo", f"{min(idle):.1f}")
    put("PeCensusIdleHi", f"{max(idle):.1f}")
    put("PeCensusKernels", f"{sum(t['kernels_per_token'] for t in w2['tasks']) / len(w2['tasks']):.0f}")
    w7 = {r[0]: r for r in md_rows(os.path.join(RT, "turing", "census", "compare_w7.md"))}
    before, after = float(w7["w2"][4]), float(w7["c2g1-w7-mgemmn0"][4])
    need(w7["c2g1-w7-mgemmn0"][5] == "-39.1%", "the int8 fit: -39.1% per token")
    put("PeIntEightBefore", f"{before:.2f}")
    put("PeIntEightAfter", f"{after:.2f}")
    put("PeIntEightChange", f"{100 * (1 - after / before):.0f}")
    put("PeIntEightUnfused", w7["g1-mgemmn0"][5].lstrip("-").rstrip("%"))
    put("PeIntEightAlone", w7["c2-w7"][5].lstrip("-").rstrip("%"))
    need(w7["g1-mgemmn0"][5] == "-6.8%" and w7["c2-w7"][5] == "-22.9%", "unfusing alone and int8 alone")
    # W9 in x-ab2 (interleaved, no profiler; S3 written before any W9 run): the W7 build (unfused), w9 with the host's
    # slice choice off (f0) and on (f1), undrafted at an empty context, three rotated rounds
    by = {}
    for r in (json.loads(x) for x in open(os.path.join(RT, "turing", "ab", "ab2.jsonl")) if x.strip()):
        arm, rnd = re.fullmatch(r"ab2-(.+)-r(\d)", r["tag"]).groups()
        need(not r["mtp"] and r["fill"] == 0, "ab2: undrafted at an empty context")
        by.setdefault(arm, {}).setdefault(int(rnd), []).append(r)
    rate = lambda rr: sum(x["new_tokens"] for x in rr) / sum(x["decode_wall_s"] for x in rr)   # noqa: E731
    need(sorted(by) == ["f0", "f1", "w7g1"] and all(sorted(v) == [1, 2, 3] for v in by.values()), "ab2: 3 arms x 3")
    need(all(rate(by["f1"][k]) > rate(by[b][k]) for b in ("w7g1", "f0") for k in (1, 2, 3)), "ab2: W9 every round")
    nine = {a: rate([x for rr in v.values() for x in rr]) for a, v in by.items()}
    put("PeWNineGain", f"{100 * (nine['f1'] / nine['w7g1'] - 1):.1f}")
    put("PeWNineFitGain", f"{100 * (nine['f1'] / nine['f0'] - 1):.1f}")
    d28 = {r[0]: r for r in md_rows(os.path.join(RT, "turing", "census", "compare_w11_d28.md"))}
    put("PeDepthBefore", d28["d28-f1-mgemmn0"][4])
    put("PeDepthAfter", d28["d28-w11-mgemmn0"][4])
    put("PeDepthChange", d28["d28-w11-mgemmn0"][5].lstrip("-").rstrip("%"))
    split = [r for r in md_rows(os.path.join(RT, "turing", "census", "compare_w11_d28.md"))
             if r[0] in ("`_paged_attn_decode_split_kernel`", "`fd75d_split_kernel`")]
    us = {r[0].strip("`"): r for r in split}
    put("PeDepthTritonUs", num(int(us["_paged_attn_decode_split_kernel"][3].split()[0])))
    put("PeDepthFdUs", num(int(us["fd75d_split_kernel"][4].split()[0])))
    calls = []
    for k, col in (("_paged_attn_decode_split_kernel", 3), ("fd75d_split_kernel", 4)):
        t, n = re.fullmatch(r"(\d+) \(([\d.]+)\)", us[k][col]).groups()
        calls.append(int(t) / float(n))
    put("PeDepthTritonCall", f"{calls[0]:.0f}")
    put("PeDepthFdCall", f"{calls[1]:.0f}")
    put("PeDepthCallRatio", f"{calls[0] / calls[1]:.1f}")
    dep = {r[0]: r for r in md_rows(os.path.join(RT, "turing", "ab", "depth2.md"))}
    pooled = {r[0]: r for r in md_rows(os.path.join(RT, "turing", "ab", "depth2.pooled.md"))}
    # pooled tokens over pooled decode time, as section 2.3 defines speed (the audit of 2026-11-05)
    f1, w11, off = (float(pooled[k][7]) for k in ("f1-28k", "w11-28k", "w11off-28k"))
    put("PeDepthTwoFOne", f"{f1:.2f}")
    put("PeDepthTwoW", f"{w11:.2f}")
    put("PeDepthTwoGain", f"{100 * (w11 / f1 - 1):.1f}")
    put("PeDepthTwoOff", f"{100 * (off / f1 - 1):.1f}")
    need(all(float(dep["w11-28k"][i]) > float(dep["f1-28k"][i]) for i in (2, 3, 4)), "fd75d at 28K: every round")
    # the registered empty-context check (w11b within +-3% of w9), which failed
    z9, z11 = float(pooled["f1-0"][7]), float(pooled["w11-0"][7])
    need(sum(float(dep["w11-0"][i]) > float(dep["f1-0"][i]) for i in (2, 3, 4)) == 2 and z11 / z9 - 1 > 0.03,
         "w11b over w9 at an empty context: two of three rounds, outside +-3%")
    put("PeDepthTwoZeroGain", f"{100 * (z11 / z9 - 1):.1f}")
    t3 = {(r[0], r[1]): r for r in md_rows(os.path.join(RT, "four_resources", "test3", "w13_report.md"))
          if len(r) == 5 and r[1] == "pooled"}
    put("PeTThreeErrZero", t3[("0", "pooled")][4].lstrip("-").rstrip("%"))
    put("PeTThreeErrTwentyEight", t3[("28000", "pooled")][4].lstrip("-").rstrip("%"))
    need(t3[("0", "pooled")][4] == "-13.2%", "test 3: w13 mode 2 underpredicted by 13.2% at fill 0")
    bench = [json.loads(x) for x in open(os.path.join(RT, "turing", "attn", "bench1.jsonl")) if x.strip()]
    b28 = [r for r in bench if r["ctx"] == 28050 and r["qlen"] == 1]
    put("PeBenchDefault", f"{next(r['us'] for r in b28 if r['config'] == 'default'):.1f}")
    put("PeBenchBest", f"{min(r['us'] for r in b28 if isinstance(r['config'], dict)):.1f}")
    put("PeBenchConfigs", sum(1 for r in b28 if isinstance(r["config"], dict)))
    put("PeBenchFloor", f"{b28[0]['floor_us_at_550gbs']:.0f}")
    t2 = [json.loads(x) for x in open(os.path.join(RT, "turing", "attn", "fd75d_test2.jsonl")) if x.strip()]
    fd = next(r for r in t2 if r["ctx"] == 28050 and r["q_len"] == 1)
    put("PeBenchFd", f"{fd['fdlive_us']:.1f}")
    put("PeBenchFdOverFloor", f"{fd['fdlive_us'] / b28[0]['floor_us_at_550gbs']:.1f}")
    pred = json.load(open(os.path.join(RT, "four_resources", "test3", "w13_predictions.json")))
    ratio = None
    stack = [pred]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if "verify_matrix_ratio" in x:
                ratio = x["verify_matrix_ratio"]
                break
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    need(ratio is not None and {"w11b", "sk1", "sk2"} <= set(ratio), "test 3's verify-pass matrix ratios")
    for disp, tag in (("w11b", "Adopt"), ("sk1", "SOne"), ("sk2", "STwo")):
        v = [ratio[disp][m] for m in ("3", "4", "5")]
        put(f"PeVerify{tag}Lo", f"{min(v):.2f}")
        put(f"PeVerify{tag}Hi", f"{max(v):.2f}")
    t2r = json.load(open(os.path.join(RT, "four_resources", "test2", "results.json")))["two_ti"]
    errs = [t2r[k]["priced_err"] for k in ("final2-adopt-0", "final2-adopt-28k", "final2-w13-0", "final2-w13-28k")]
    berr = [t2r[k]["bytes_err"] for k in ("final2-adopt-0", "final2-adopt-28k", "final2-w13-0", "final2-w13-28k")]
    put("PePricedMin", f"{100 * min(errs):.1f}")
    put("PePricedMax", f"{100 * max(errs):.1f}")
    put("PeBytesErrMin", f"{-100 * max(berr):.0f}")
    put("PeBytesErrMax", f"{-100 * min(berr):.0f}")


def quality_lanes():
    """Same-card GSM8K-500 and MMLU-500 against llama.cpp's 2-bit GGUFs (testsuite/results/turing/quality/
    compare_lcpp_fp32.json; descriptive, Holm across the 14 same-card pairs)."""
    q = json.load(open(os.path.join(RT, "turing", "quality", "compare_lcpp_fp32.json")))
    lanes = {(s, x["label"]): x for s, xs in q["lanes"].items() for x in xs}
    tags = {"TURING-exl3-nodraft": "ExlNo", "TURING-exl3-mtp": "ExlMtp", "TURING-exl3-mtp-w13s2": "ExlW",
            "TURING-lcpp-iq2s": "Is", "TURING-lcpp-iq2xxs": "Ixxs", "TURING-lcpp-iq2s-mtp": "IsMtp",
            "TURING-lcpp-iq2xxs-dflash2": "IxxsDf"}
    for (suite, label), x in lanes.items():
        if label in tags:
            st = {"gsm8k_500": "Gsm", "mmlu_500": "Mmlu"}[suite]
            need(x["errors"] == 0 and x["n"] == 500, f"{label} {suite}: 500 rows, no errors")
            put(f"PeQ{st}{tags[label]}Acc", f"{x['acc']:.1f}")
            if suite == "gsm8k_500":
                put(f"PeQ{st}{tags[label]}Tok", f"{x['mean_tok']:.0f}")
                put(f"PeQ{st}{tags[label]}Wall", f"{x['mean_wall']:.2f}")
                put(f"PeQ{st}{tags[label]}Tps", f"{x['decode_tps_median']:.1f}")
    gsm = [x["acc"] for (s, l), x in lanes.items() if s == "gsm8k_500" and l in tags]
    put("PeQGsmMin", f"{min(gsm):.1f}")
    put("PeQGsmMax", f"{max(gsm):.1f}")
    pairs = [p for p in q["pairs"] if p["group"] == "turing-internal"]
    need(len(pairs) == 14, "14 same-card pairs")
    ps = sorted((p["stats"]["mcnemar_p"], i) for i, p in enumerate(pairs))
    holm, run = {}, 0.0
    for rank, (pv, i) in enumerate(ps):
        run = max(run, min(1.0, (len(ps) - rank) * pv))
        holm[i] = run
    for i, p in enumerate(pairs):
        a, b, suite = p["A"], p["B"], p["suite"]
        key = {("TURING-exl3-nodraft", "TURING-lcpp-iq2s", "mmlu_500"): "MmluIs",
               ("TURING-exl3-nodraft", "TURING-lcpp-iq2xxs", "mmlu_500"): "MmluIxxs",
               ("TURING-exl3-nodraft", "TURING-lcpp-iq2s", "gsm8k_500"): "GsmIs",
               ("TURING-exl3-nodraft", "TURING-lcpp-iq2xxs", "gsm8k_500"): "GsmIxxs",
               ("TURING-exl3-mtp", "TURING-lcpp-iq2xxs-dflash2", "mmlu_500"): "MmluDf"}.get((a, b, suite))
        if key:
            st = p["stats"]
            put(f"PeQP{key}Diff", pts(st["delta_acc_pts"]))
            put(f"PeQP{key}Abs", f"{abs(st['delta_acc_pts']):.1f}")
            put(f"PeQP{key}Ten", st["A_only_correct"])
            put(f"PeQP{key}Ott", st["B_only_correct"])
            put(f"PeQP{key}P", f"{st['mcnemar_p']:.2g}")
            put(f"PeQP{key}Holm", f"{holm[i]:.2f}")
    w, s_mtp, x_df = (lanes[("gsm8k_500", "TURING-exl3-mtp-w13s2")], lanes[("gsm8k_500", "TURING-lcpp-iq2s-mtp")],
                      lanes[("gsm8k_500", "TURING-lcpp-iq2xxs-dflash2")])
    put("PeQRatioIsMtp", f"{s_mtp['mean_wall'] / w['mean_wall']:.2f}")
    put("PeQRatioIxxsDf", f"{x_df['mean_wall'] / w['mean_wall']:.2f}")
    put("PeQTokRatioIsMtp", f"{s_mtp['mean_tok'] / w['mean_tok']:.2f}")
    put("PeQTpsRatioIsMtp", f"{w['decode_tps_median'] / s_mtp['decode_tps_median']:.2f}")
    put("PeQTokRatioIxxsDf", f"{x_df['mean_tok'] / w['mean_tok']:.2f}")
    put("PeQTpsRatioIxxsDf", f"{w['decode_tps_median'] / x_df['decode_tps_median']:.2f}")
    var = {(p["A"], p["B"], p["suite"]): p["stats"] for p in q["pairs"] if "stats" in p}
    st = var[("TURING-exl3-mtp", "TURING-exl3-mtp-w13s2", "gsm8k_500")]
    need(st["A_only_correct"] == st["B_only_correct"] == 0, "w13 mode 2 with MTP: no discordant GSM8K item")
    put("PeQWSame", st["extra"]["same_final_answer"])
    t4 = json.load(open(os.path.join(RT, "turing", "quality", "compare_t4_w13s2.json")))
    ref = next(p["stats"] for p in t4["pairs"] if p["A"] == "TURING-exl3-mtp-w11j2" and p["B"] == "TURING-exl3-mtp-w13s2")
    need(ref["delta_acc_pts"] >= -1.0 and ref["A_only_correct"] == ref["B_only_correct"] == 0,
         "mode 2's registered check against w11b with MTP: inside -1.0 point, no discordant item")
    put("PeQWModeTwoRefAcc", f"{ref['accB']:.1f}")
    put("PeQWModeTwoRefSame", ref["extra"]["same_final_answer"])
    put("PeQWIdent", st["extra"]["anatomy"]["identical"])
    st = var[("TURING-exl3-nodraft", "TURING-exl3-nodraft-w11", "gsm8k_500")]
    put("PeQWElevenAcc", f"{st['accB']:.1f}")
    put("PeQWElevenDiff", pts(st["delta_acc_pts"]))
    put("PeQWElevenTen", st["A_only_correct"])
    put("PeQWElevenOtt", st["B_only_correct"])
    put("PeQWElevenP", f"{st['mcnemar_p']:.3g}")
    w11 = lanes[("gsm8k_500", "TURING-exl3-nodraft-w11")]
    put("PeQWElevenTok", f"{w11['mean_tok']:.0f}")
    put("PeQWElevenTps", f"{w11['decode_tps_median']:.1f}")
    put("PeQWElevenWall", f"{w11['mean_wall']:.2f}")
    w11m = lanes[("gsm8k_500", "TURING-exl3-mtp-w11j2")]    # w11b with MTP, mode 2's registered reference
    need(abs(w11m["acc"] - ref["accA"]) < 1e-9 and abs(w11m["mean_tok"] - ref["mean_tok_A"]) < 1e-9,
         "the w11b MTP lane matches the reference pair")
    put("PeQWElevenMtpAcc", f"{w11m['acc']:.1f}")
    put("PeQWElevenMtpTok", f"{w11m['mean_tok']:.0f}")
    put("PeQWElevenMtpTps", f"{w11m['decode_tps_median']:.1f}")
    put("PeQWElevenMtpWall", f"{w11m['mean_wall']:.2f}")
    for suite, tag in (("gsm8k_500", "Gsm"), ("mmlu_500", "Mmlu")):
        st = var[("TURING-exl3-nodraft", "TURING-exl3-mtp", suite)]
        put(f"PeQMtp{tag}Ten", st["A_only_correct"])
        put(f"PeQMtp{tag}Ott", st["B_only_correct"])
    no = lanes[("gsm8k_500", "TURING-exl3-nodraft")]
    longer = [lanes[("gsm8k_500", l)]["mean_tok"] / no["mean_tok"] - 1 for l in ("TURING-lcpp-iq2s", "TURING-lcpp-iq2xxs")]
    put("PeQLongerMin", f"{100 * min(longer):.0f}")
    put("PeQLongerMax", f"{100 * max(longer):.0f}")


def configs_agree():
    """Every arm's server config and start-up log: one request at a time, chunk 1024, MTP with 6 dynamic drafts in the
    cache's own format, context = cache size = the arm's length, a 2,048 MiB host-RAM recurrent cache, fp16 state."""
    want = {"max_batch_size": "1", "chunk_size": "1024", "draft_mode": "mtp", "dynamic_draft": "true",
            "draft_num_tokens": "6", "sysmem_recurrent_cache": "2048", "log_generation_params": "false"}
    paths = sorted(glob.glob(os.path.join(R, "stage[23]", "configs", "*.yml")) +
                   glob.glob(os.path.join(R, "fd75d-arm", "configs", "*.yml")))
    for p in paths:
        with open(p) as f:
            cfg = dict(re.findall(r"^\s+(\w+): \"?([^\"\n]*)\"?$", f.read(), re.M))
        need(all(cfg.get(k) == v for k, v in want.items()), f"{os.path.basename(p)}: the lane's server settings")
        need(cfg["cache_mode"] == cfg["draft_cache_mode"] and cfg["max_seq_len"] == cfg["cache_size"],
             f"{os.path.basename(p)}: draft cache in the cache's format, context = cache size")
    logs = glob.glob(os.path.join(R, "stage[23]", "logs", "*.log")) + glob.glob(os.path.join(R, "fd75d-arm", "logs",
                                                                                            "*.log"))
    starts = 0
    for p in logs:
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        n = text.count("Recurrent state storage: fp16")
        need(n >= 1 and n == text.count("Using main model MTP component for drafting"), f"{p}: fp16 state, MTP")
        starts += n
    put("PeNConfigs", len(paths))
    put("PeDraftTokens", want["draft_num_tokens"])
    put("PeChunk", num(int(want["chunk_size"])))
    put("PeRecurrentMiB", num(int(want["sysmem_recurrent_cache"])))
    return starts


def p6():
    """P6's numbers, from P6's own generated macros."""
    with open(os.path.join(REPO, "paper6", "derived", "claims.tex")) as f:
        six = dict(re.findall(r"\\newcommand\{\\(Psix[A-Za-z]+)\}\{(.*)\}", f.read()))
    for src, dst in (("PsixIHyb", "PeSixI"), ("PsixICIloHyb", "PeSixILo"), ("PsixICIhiHyb", "PeSixIHi"),
                     ("PsixEqHolm", "PeSixHolm"), ("PsixMargin", "PeSixMargin"), ("PsixKvFreedHybGiB", "PeSixFreedHyb"),
                     ("PsixKvFreedGqaGiB", "PeSixFreedGqa"), ("PsixStepHybTwoThreeGiB", "PeSixStepHyb"),
                     ("PsixStepGqaTwoFourGiB", "PeSixStepGqa"), ("PsixNItems", "PeSixItems"),
                     ("PsixStopAt", "PeSixStopAt"), ("PsixFamMinVartrackHyb", "PeSixVtMin"),
                     ("PsixFamMaxVartrackHyb", "PeSixVtMax"), ("PsixDMax", "PeSixDMax"), ("PsixM", "PeSixM"),
                     ("PsixMLo", "PeSixMLo"), ("PsixMHi", "PeSixMHi"), ("PsixMHolm", "PeSixMHolm"),
                     ("PsixEngine", "PeSixEngine")):
        need(src in six, f"P6 macro {src}")
        put(dst, six[src])
    easy = [v for k, v in six.items() if re.fullmatch(r"PsixFam(Single|Multikey|Multivalue)Hyb[A-Za-z]+", k)]
    need(easy and all(v == "100.0" for v in easy), "P6: the hybrid's three easy families at 100% in every cell")


def audit_numbers():
    """Numbers the full audit of 2026-11-05 added (research/P8_AUDIT_2026-11-05.md): the census against F2 under the
    profiler, the unprofiled bench of the final kernels against the census fit, 1-bit against 2-bit calls on both cards,
    the usable context against the split cache's own 32K accuracy, and rafatxf's fork across the two sessions."""
    ft = lambda k: float(M[k])
    # the stock census's ms per token at two decimals (PeIntEightBefore, the same w2 census), so no rounding is chained
    need(abs(ft("PeIntEightBefore") - ft("PeCensusMs")) < 0.05, "the stock census: one value at two precisions")
    put("PeProfStockPct", f"{100 * (ft('PeIntEightBefore') / (1000 / ft('PeFtStockZero')) - 1):.0f}")
    put("PeProfStockMtpPct", f"{100 * (ft('PeCgStockMtpMs') / (1000 / ft('PeFtStockMtpZero')) - 1):.0f}")
    put("PeProfWMtpPct", f"{100 * (1 - ft('PeCgWMs') / (1000 / ft('PeFtSTwoMtpZero'))):.0f}")
    put("PeProfFinPct", f"{100 * (ft('PeCensusFinMs') / ft('PeFtWMs') - 1):.0f}")
    put("PeFtMtpCutPct", f"{100 * (1 - ft('PeFtStockMtpZero') / ft('PeFtSTwoMtpZero')):.0f}")
    put("PeCfRafVsFtPct", f"{100 * (ft('PeCfRafZero') / ft('PeFtRafMtpZero') - 1):.1f}")
    # 1-bit against 2-bit calls of the same role, in the 2080 Ti censuses and in the unprofiled benches
    ratios = []
    for census in ("d0-f1-mgemmn0", "d0-w12p"):
        rows = md_rows(os.path.join(RT, "turing", "census", census, "int8_by_bits.md"))
        med = {(r[0], int(r[1])): float(r[3]) for r in rows if len(r) == 7 and r[0] != "role"}
        ratios += [med[(role, 1)] / med[(role, 2)] for role, k in med if k == 1 and (role, 2) in med]
    need(all(0.95 <= x <= 1.0 for x in ratios), "1-bit calls take 95-100% of the 2-bit time in the censuses")
    bench_ratios, fin_fixed, fin_share, fin_rated, nine_fixed, nine_share = [], [], [], [], [], []
    w12p = md_rows(os.path.join(RT, "turing", "census", "d0-w12p", "int8_by_bits.md"))
    calls = [r for r in w12p if len(r) == 7 and r[0] not in ("role", "head")]
    tokens = next(int(r[2]) for r in w12p if len(r) == 7 and r[0] == "head")

    def share(fits, rate):
        fixed = extra = 0.0
        for role, k, launches, _, mb, _, _ in calls:
            k, per_tok, mb = int(k), int(launches) / tokens, float(mb)
            if k in fits:
                a, b = fits[k]
                fixed += per_tok * a
                extra += per_tok * (mb * 8 / k) * (b - byte_us_per_mweights(k, rate))
        return fixed / (fixed + extra)

    def fit(rows):
        out = {}
        for k in sorted({r["K"] for r in rows}):
            pts = [(r["mweights"], r["us"]) for r in rows if r["K"] == k]
            if len(pts) < 3:
                continue
            mx = sum(x for x, _ in pts) / len(pts)
            my = sum(y for _, y in pts) / len(pts)
            b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
            out[k] = (my - b * mx, b)
        return out

    load = lambda pat: [r for f in sorted(glob.glob(os.path.join(RT, "turing", "w12", pat)))
                        for r in map(json.loads, open(f)) if r.get("m") == 1 and "mweights" in r and "us" in r]
    for mode in ("sk1", "sk2"):                    # each mode's two rounds pooled, as the audit fitted them
        rows = load(f"t3w13_{mode}_r[12]_20261101T191124.jsonl")
        fits = fit(rows)
        fin_fixed.append(fits[2][0])
        fin_share.append(share(fits, CENSUS_RATE))
        fin_rated.append(share(fits, RATED_RATE))
        for role in ("attn_q", "gdn_qkv", "mlp_down", "mlp_gate_up"):   # the roles the censuses hold at 1 bit
            one = [r["us"] for r in rows if r["shape"] == role and r["K"] == 1]
            two = [r["us"] for r in rows if r["shape"] == role and r["K"] == 2]
            if one and two:
                bench_ratios.append(statistics.mean(one) / statistics.mean(two))
    rows = load("t3w13_w11b_r[12]_20261101T191124.jsonl")
    fits = fit(rows)
    nine_fixed.append(fits[2][0])
    nine_share.append(share(fits, CENSUS_RATE))
    need(len(fin_fixed) == 2 and len(nine_fixed) == 1 and len(bench_ratios) >= 4,
         "the unprofiled benches of 1 November: two final modes, the w11b kernel, the 1-bit roles")
    need(min(fin_share) > 0.5 and min(fin_rated) > 0.5, "unprofiled: the fixed part is most of the final kernels' excess")
    # Ada: time per weight at 1 bit against the read (primary rate)
    for tag, d in (("", "d0-stock-mgemmn0"), ("Boost", "d0-stock-mgemmn0-boost")):
        sp = json.load(open(os.path.join(RT, "ada", "census", d, "split.json")))
    # the registered priced-graph test's own verdicts (four_resources test 2), beside its good fits
    rep = open(os.path.join(RT, "four_resources", "test2", "report.md")).read()
    km = re.search(r"Kill 2.*?TRIGGERED\. (\d+) of (\d+) miss by more than 20%", rep)
    mm = re.search(r"refuted\. max \|error\| ([\d.]+)%, median ([\d.]+)%", rep)
    need(km is not None and mm is not None, "test 2's registered verdicts: kill 2 triggered, the +-12% claim refuted")
    put("PeTTwoKillN", km.group(1))
    put("PeTTwoKillOf", km.group(2))
    put("PeTTwoMaxErr", mm.group(1))
    put("PeTTwoMedErr", mm.group(2))
    put("PeFtModeTwoOverAdoptPct", f"{100 * (ft('PeFtSTwoMtpZero') / ft('PeFtAdoptMtpZero') - 1):.1f}")
    # the registered usable context, now computable against the split cache's own 32K arm (replication R2's arm)
    s96 = A.Arms(A.load_rows(os.path.join(R, "stage3", "journal.jsonl")), ITEMS).scores["s3-96k"]
    s32 = A.Arms(A.load_rows(os.path.join(R, "replication", "journal.jsonl")), ITEMS).scores["rep-32k-84"]
    common = sorted(u for u in s96 if u in s32)
    x, y = [s96[u] for u in common], [s32[u] for u in common]
    m, lo, hi = T.paired_interval(x, y)
    n10, n01, _ = T.mcnemar(x, y)
    need(len(common) == 48, "the usable-context gap: 48 shared items")
    put("PeUseGapN", len(common))
    put("PeUseGapDiff", pts(m))
    put("PeUseGapAbs", f"{abs(m):.1f}")
    put("PeUseGapLo", pts(lo))
    put("PeUseGapHi", pts(hi))
    put("PeUseGapTen", n10)
    put("PeUseGapOtt", n01)


def refresh():
    """The release speed refresh of 12 November (research/prereg/2026-11-12-p8-refresh.md): current NInferno
    (65e573fd) and llama.cpp (dd6b945be) against unchanged w13 on the old pinned artifacts, three rotated rounds.
    Pooled rates recomputed from rows.jsonl with analyze_refresh.py's own functions and checked against its output."""
    sys.path.insert(0, os.path.join(REPO, "tools", "turing", "refresh"))
    import analyze_refresh as AR
    d = os.path.join(RT, "turing", "p8-refresh-76b5")
    res = json.load(open(os.path.join(d, "analysis.json")))
    rows = [json.loads(x) for x in open(os.path.join(d, "rows.jsonl")) if x.strip()]
    rungs = json.load(open(os.path.join(d, "rungs.json")))
    by = {c["id"]: c for c in res["comparisons"]}
    for cid, c in by.items():
        if not c["complete"]:
            continue
        a, ma = AR.cells(rows, c["anchor"], ["code", "math", "prose"])
        b, mb = AR.cells(rows, cid, ["code", "math", "prose"])
        need(not ma and not mb, f"refresh {cid}: complete cells recompute")
        need(abs(AR.pooled(list(a.values())) - c["w13"]) < 1e-9 and abs(AR.pooled(list(b.values())) - c["competitor"]) < 1e-9,
             f"refresh {cid}: pooled rates recompute")
    names = {"r1-ninferno-nif-nd-0": "NifNdZero", "r1-ninferno-nif-nd-28k": "NifNdDeep",
             "r1-ninferno-nif-df2-0": "NifDfZero", "r1-ninferno-nif-df2-28k": "NifDfDeep",
             "r2nd-llama-lcpp-nd-0": "LcppNdZero", "r2nd-llama-lcpp-nd-28k": "LcppNdDeep",
             "r2draft-llama-lcpp-mtp-0": "LcppMtpZero", "r2draft-llama-lcpp-mtp-28k": "LcppMtpDeep"}
    for cid, tag in names.items():
        c = by[cid]
        need(c["complete"] and c["w13_wins_every_round"], f"refresh {cid}: complete, w13 ahead in every round")
        put(f"PeRf{tag}", f"{c['competitor']:.2f}")
        put(f"PeRfW{tag}", f"{c['w13']:.2f}")
        put(f"PeRfP{tag}", pct(c["w13"] / c["competitor"] - 1))
        put(f"PeRfX{tag}", f"{c['w13'] / c['competitor']:.2f}")
        rr = [r["ratio"] for r in c["rounds"]]
        put(f"PeRfP{tag}Lo", pct(min(rr) - 1))
        put(f"PeRfP{tag}Hi", pct(max(rr) - 1))
    for cid in ("r1-ninferno-nif-nd-0", "r1-ninferno-nif-nd-28k", "r1-ninferno-nif-df2-0", "r1-ninferno-nif-df2-28k"):
        need(by[cid]["can_keep_current_pooled_lead"], f"refresh {cid}: the registered rule keeps the current lead")
    for cid in ("r2nd-llama-lcpp-nd-0", "r2nd-llama-lcpp-nd-28k", "r2draft-llama-lcpp-mtp-0", "r2draft-llama-lcpp-mtp-28k"):
        need(not by[cid]["eligible_full_residency_comparison"], f"refresh {cid}: residency unverified, no current lead")
    for cid in ("r2draft-llama-lcpp-df2-0", "r2draft-llama-lcpp-df2-28k"):
        need(not by[cid]["complete"], f"refresh {cid}: no speed cell")
    need(rungs["lcpp-df2"]["status"] == "failed-all-attempted-rungs", "refresh: llama.cpp DFlash2 failed every rung")
    need(rungs["lcpp-mtp"]["rung"] == "R3" and rungs["lcpp-mtp"]["partial_or_unknown"], "refresh: llama.cpp MTP fit only at R3")
    math0 = by["r1-ninferno-nif-df2-0"]["per_prompt"]["math"]
    math28 = by["r1-ninferno-nif-df2-28k"]["per_prompt"]["math"]
    need(math0["w13"] < math0["competitor"] and math28["w13"] < math28["competitor"], "refresh: DFlash2 ahead on math")
    put("PeRfNifDfZeroMath", f"{math0['competitor']:.2f}")
    put("PeRfWNifDfZeroMath", f"{math0['w13']:.2f}")
    put("PeRfNifDfDeepMath", f"{math28['competitor']:.2f}")
    put("PeRfWNifDfDeepMath", f"{math28['w13']:.2f}")
    for cid, tag in (("r1-ninferno-nif-df2-0", "Zero"), ("r1-ninferno-nif-df2-28k", "Deep")):
        pp = by[cid]["per_prompt"]["prose"]
        put(f"PeRfXNifDf{tag}Prose", f"{pp['w13'] / pp['competitor']:.2f}")


def refresh_fourteen():
    """Generate the replacement comparison from independently reviewed aggregate outputs.

    The numeric projection binds the original reviewed tables. It does not
    re-score outputs or infer identical sampling from neutral penalties.
    """
    path = os.path.join(REPO, "research", "raw", "2026-11-14-p8-rc-refresh-a72c", "refresh14-numeric.json")
    data = json.load(open(path))
    need(data["schema"] == "p8-refresh14-reviewed-numeric-v1" and data["session_valid"], "14 November evidence identity")
    by = {c["id"]: c for c in data["comparisons"]}
    need(len(by) == 14 and sum(c["complete"] for c in by.values()) == 9, "14 November coverage 9/14")
    need(sum(c["keep"] for c in by.values()) == sum(c["eligible"] for c in by.values()) == 4,
         "14 November only four eligible leads")
    need(not data["complete"] and not data["can_keep_all_tested_leads"] and not data["sampling_equivalence"],
         "14 November no comprehensive ranking or identical sampler")
    need(data["resolved_penalties"] == dict(requests=49, presence=0.0, frequency=0.0, repetition=1.0),
         "49 neutral-penalty requests")
    names = {"r1-ninferno-nif-nd-0": "NifNdZero", "r1-ninferno-nif-nd-28k": "NifNdDeep",
             "r1-ninferno-nif-df2-0": "NifDfZero", "r1-ninferno-nif-df2-28k": "NifDfDeep",
             "r2nd-llama-lcpp-nd-0": "LcppNdZero", "r2nd-llama-lcpp-nd-28k": "LcppNdDeep",
             "r2draft-llama-lcpp-mtp-0": "LcppMtpZero", "r2draft-llama-lcpp-mtp-28k": "LcppMtpDeep",
             "r0-stock-mtp-0": "StockMtpZero"}
    for cid, tag in names.items():
        c = by[cid]
        need(c["complete"] and len(c["rounds"]) == 3 and len(c["tasks"]) == 3, "complete reviewed cell " + cid)
        need(abs(c["ratio"] - c["w13"] / c["competitor"]) < 1e-12, "reviewed ratio " + cid)
        for r in c["rounds"] + c["tasks"]:
            need(all(math.isfinite(r[k]) and r[k] > 0 for k in ("w13", "competitor", "ratio")) and
                 abs(r["ratio"] - r["w13"] / r["competitor"]) < 1e-12, "finite reviewed subcell " + cid)
        need({r["round"] for r in c["rounds"]} == {1, 2, 3} and
             {r["task"] for r in c["tasks"]} == {"code", "math", "prose"}, "reviewed cell identities " + cid)
        if tag.startswith("Nif"):
            need(c["eligible"] and c["keep"] and all(r["ratio"] > 1 for r in c["rounds"]), "NInferno pooled wins every round")
        else:
            need(not c["eligible"] and not c["keep"], "descriptive comparison only " + cid)
        put("PeFresh" + tag, f"{c['competitor']:.2f}")
        put("PeFreshW" + tag, f"{c['w13']:.2f}")
        put("PeFreshX" + tag, f"{c['ratio']:.2f}")
        precision = 2 if "Nd" in tag else 1
        put("PeFreshP" + tag, pct(c["ratio"] - 1, precision))
        put("PeFreshP" + tag + "Lo", pct(min(r["ratio"] for r in c["rounds"]) - 1, precision))
        put("PeFreshP" + tag + "Hi", pct(max(r["ratio"] for r in c["rounds"]) - 1, precision))
        if tag.startswith("NifDf"):
            for r in c["tasks"]:
                need((r["ratio"] < 1) == (r["task"] == "math"), "drafted code/prose win and math loss")
                put("PeFresh" + tag + r["task"].capitalize(), f"{r['competitor']:.2f}")
                put("PeFreshW" + tag + r["task"].capitalize(), f"{r['w13']:.2f}")
    put("PeFreshComplete", 9)
    put("PeFreshPlanned", 14)
    put("PeFreshEligible", 4)
    TABLES["refresh14.tex"] = r"""% Generated from reviewed 14 November aggregate tables; descriptive ranges.
\begin{table}[!tbp]\centering\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llrrll@{}}\toprule
Drafting & Fill & Patched EXL3 & NInferno & Excess (\%) & Round range (\%)\\\midrule
none / none & empty & \PeFreshWNifNdZero & \PeFreshNifNdZero & \PeFreshPNifNdZero & \PeFreshPNifNdZeroLo--\PeFreshPNifNdZeroHi\\
none / none & 28K & \PeFreshWNifNdDeep & \PeFreshNifNdDeep & \PeFreshPNifNdDeep & \PeFreshPNifNdDeepLo--\PeFreshPNifNdDeepHi\\
MTP / DFlash2 & empty & \PeFreshWNifDfZero & \PeFreshNifDfZero & \PeFreshPNifDfZero & \PeFreshPNifDfZeroLo--\PeFreshPNifDfZeroHi\\
MTP / DFlash2 & 28K & \PeFreshWNifDfDeep & \PeFreshNifDfDeep & \PeFreshPNifDfDeep & \PeFreshPNifDfDeepLo--\PeFreshPNifDfDeepHi\\
\bottomrule\end{tabular}
\caption{Replacement comparison, 14 November 2026: NInferno f8143b9e
against unchanged patched exllamav3, tokens per second pooled over three
prompts and three rotated rounds. Both use neutral penalties; top-k and
sampling order still differ. All four pooled leads hold in every round.
Ranges describe those rounds, not confidence intervals.}
\label{tab:refresh14}
\end{table}
"""


def fleet_replication():
    """Dated HTTP deployment replication, not an update to the 2080 Ti measurements.

    Committed 7c2e/c4d8 results (13--14 November); fleet program parent research-ledger-reference.
    Recompute speed from measured journal rows and quality counts from text-excluded verdict exports.
    Those exports permit count checks, not independent answer rescoring or controlled-seed claims.
    """
    base = os.path.join(REPO, "research", "raw", "2026-11-12-fleet-lanes")

    def read(root, name):
        with open(os.path.join(root, name)) as f:
            return json.load(f)

    current = os.path.join(base, "results-2026-11-14-c4d8")
    exported = os.path.join(current, "available")
    summaries = read(exported, "registered-summaries.json")
    previous = read(os.path.join(base, "results-2026-11-13-7c2e", "analysis"), "registered-summaries.json")
    verdicts = read(current, "prediction-verdicts.json")
    need(verdicts["p8_slice_complete"], "fleet: P8 slice complete")
    need(len({p["id"] for p in verdicts["predictions"]}) == len(verdicts["predictions"]), "fleet: unique predictions")
    assessed = {p["id"]: p for p in verdicts["predictions"] if p["id"].startswith(("A1-", "A2-"))}
    expected, speed, quality = {}, {}, {}

    def quality_rows(arm, suite, source=os.path.join(exported, "quality-rows")):
        with open(os.path.join(source, arm, suite + ".jsonl")) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        need(len(rows) == len({r["id"] for r in rows}) == 500, f"fleet {arm}/{suite}: complete unique IDs")
        need(all(type(r["correct"]) is bool and not r.get("error") for r in rows), f"fleet {arm}/{suite}: verdicts")
        return rows

    anchor = quality_rows("TURING-exl3-mtp-w13", "gsm8k_500", os.path.join(RT, "turing", "quality"))
    anchor_correct = sum(r["correct"] for r in anchor)
    # Private source labels live outside the paper; tags name the two card fixtures.
    cards = read(base, "p8-card-labels.json")
    need(set(cards) == {"CardA", "CardB"} and len(set(cards.values())) == 2, "fleet: two distinct card fixtures")
    for tag, machine in sorted(cards.items()):
        journal = os.path.join(REPO, "fixtures", "fleet", machine,
                               "runs", "ns-20261112-p8-speed", "journal.jsonl")
        with open(journal) as f:
            rows = [r for line in f if line.strip() and (r := json.loads(line)).get("kind") == "result"]
        need(len(rows) == len({r["id"] for r in rows}) == 36, f"fleet {machine}: complete measured speed rows")
        summary = summaries[machine + "/p8-speed"]
        need(summary["complete"] and summary["n"] == summary["planned"] == len(rows), f"fleet {machine}: speed summary")
        cells = {tuple(c["coordinate"]): c for c in summary["cells"]}
        need(set(cells) == {(d, fill) for d in (False, True) for fill in (0, 28000)}, f"fleet {machine}: speed cells")
        for fill, label in ((0, "Zero"), (28000, "Deep")):
            for drafted, mode in ((False, "Nd"), (True, "Mtp")):
                rr = [r for r in rows if (r["drafted"], r["fill"]) == (drafted, fill)]
                need({(r["round"], r["task"]) for r in rr} ==
                     {(rnd, task) for rnd in range(3) for task in ("code", "math", "prose")} and len(rr) == 9,
                     f"fleet {machine}/{mode}/{fill}: three rotated rounds")
                need(all(r["new_tokens"] > 0 and math.isfinite(r["decode_wall_s"]) and r["decode_wall_s"] > 0
                         for r in rr), f"fleet {machine}: finite positive speed rows")
                rate = sum(r["new_tokens"] for r in rr) / sum(r["decode_wall_s"] for r in rr)
                need(cells[drafted, fill]["complete_run"] and cells[drafted, fill]["n"] == len(rr) and
                     abs(rate - cells[drafted, fill]["pooled_tokens_per_s"]) < 1e-9, f"fleet {machine}: pooled rate recomputes")
                speed[machine, drafted, fill] = rate
                put(f"PeFleet{tag}{mode}{label}", f"{rate:.3f}")
            expected[f"A1-{machine}-{fill}"] = speed[machine, True, fill] > speed[machine, False, fill]
        for suite, short, label in (("gsm8k_500", "gsm", "Gsm"), ("mmlu_500", "mmlu", "Mmlu")):
            arms = []
            for mode, mode_label in (("nd", "Nd"), ("mtp", "Mtp")):
                lane = f"{machine}/p8-{short}-{mode}"
                summary = summaries[lane]
                rr = quality_rows(f"FLEET-{machine}-w13-{mode}", suite)
                arms.append(rr)
                correct = sum(r["correct"] for r in rr)
                need(summary["complete"] and summary["n"] == summary["planned"] == len(rr), f"fleet {lane}: complete")
                need(len(summary["cells"]) == 1 and summary["cells"][0]["complete_run"] and
                     summary["cells"][0]["n"] == len(rr) and
                     abs(summary["cells"][0]["pass1"] - correct / len(rr)) < 1e-12, f"fleet {lane}: correct count recomputes")
                quality[machine, suite, mode] = correct
                put(f"PeFleet{tag}{label}{mode_label}", correct)
            need([r["id"] for r in arms[0]] == [r["id"] for r in arms[1]], f"fleet {machine}/{suite}: paired IDs/order")
            expected[f"A2-{machine}-{suite}-nd-mtp"] = abs(quality[machine, suite, "mtp"] - quality[machine, suite, "nd"]) / len(arms[0]) <= .03
            if short == "gsm":
                need([r["id"] for r in anchor] == [r["id"] for r in arms[1]], f"fleet {machine}: archived anchor IDs/order")
                expected[f"A2-{machine}-gsm-replication"] = abs(quality[machine, suite, "mtp"] - anchor_correct) / len(anchor) <= .05
        if tag == "CardA":
            for lane in ("p8-speed", "p8-gsm-nd", "p8-gsm-mtp", "p8-mmlu-nd", "p8-mmlu-mtp"):
                need(previous[machine + "/" + lane] == summaries[machine + "/" + lane], "fleet: first-card archived results unchanged")
    expected["A1-cross-card"] = all(speed[cards["CardA"], d, fill] > speed[cards["CardB"], d, fill]
                                    for d in (False, True) for fill in (0, 28000))
    need(set(expected) == set(assessed) and all(expected.values()) and
         all(p["status"] == "held" for p in assessed.values()), "fleet: all and only assessed P8 predictions hold")
    put("PeFleetAssessedPredictions", len(expected))
    put("PeFleetQualityN", len(anchor))
    put("PeFleetSpeedRequestsPerCell", cells[False, 0]["n"])
    TABLES["fleet_replication.tex"] = r"""% Generated by build_claims.py; committed fleet journals/verdict exports.
\begin{table}[!tbp]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrr@{}}\toprule
 & \multicolumn{4}{c}{HTTP decode tokens/s} & \multicolumn{2}{c}{Correct (none / MTP)}\\
Card & 0, none & 0, MTP & 28K, none & 28K, MTP & GSM8K & MMLU\\\midrule
RTX 4080 SUPER & \PeFleetCardANdZero & \PeFleetCardAMtpZero & \PeFleetCardANdDeep & \PeFleetCardAMtpDeep & \PeFleetCardAGsmNd/\PeFleetCardAGsmMtp & \PeFleetCardAMmluNd/\PeFleetCardAMmluMtp\\
RTX 4070 & \PeFleetCardBNdZero & \PeFleetCardBMtpZero & \PeFleetCardBNdDeep & \PeFleetCardBMtpDeep & \PeFleetCardBGsmNd/\PeFleetCardBGsmMtp & \PeFleetCardBMmluNd/\PeFleetCardBMmluMtp\\
\bottomrule\end{tabular}
\caption{Dated fleet deployment replication, reports of 13--14 November 2026.
Speed pools actual generated tokens over decode seconds for
\PeFleetSpeedRequestsPerCell{} requests per card/condition: three prompts in
three rotated rounds, warm-ups excluded. Quality is correct out of
\PeFleetQualityN{} items per suite/arm. Both cards serve the same EXL3 weights
with w13 over TabbyAPI HTTP; these are descriptive deployment results, not
speedup ratios against the 2080 Ti.}
\label{tab:fleet-replication}
\end{table}
"""


def main():
    if not os.path.isdir(ITEMS):
        raise SystemExit(f"{ITEMS} is missing: rebuild the items from the committed manifests first "
                         "(testsuite/p8_longctx/build_stage2_items.py)")
    limits = plan_limits()
    stage1()
    e2 = stage2(limits)
    e3 = stage3(limits)
    run = fd75d()
    ea = arm(limits)
    envelope(e2, e3, ea, run)
    put("PeNStarts", configs_agree())
    matched_items()
    peak_rise()
    replication_r1()
    replication_final()
    weight_hashes()
    speed_final2()
    model_files()
    p9_bench()
    bytes_per_token()
    census_groups()
    confirmation()
    refresh()
    refresh_fourteen()
    fleet_replication()
    audit()
    mechanism()
    call_costs()
    audit_numbers()
    quality_lanes()
    p6()
    os.makedirs(os.path.join(PAPER, "derived"), exist_ok=True)
    lines = ["% Generated by paper8/scripts/build_claims.py from testsuite/results/turing/longctx/ (and P6's generated",
             "% claims for the section on Paper 6). Do not edit by hand."]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v in sorted(M.items())]
    with open(os.path.join(PAPER, "derived", "claims.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name, text in TABLES.items():   # generated tables, written only once every check has passed
        with open(os.path.join(PAPER, "derived", name), "w") as f:
            f.write(text)
    print(f"{len(M)} macros -> paper8/derived/claims.tex; {len(TABLES)} table(s); every claim check passed")


if __name__ == "__main__":
    main()
