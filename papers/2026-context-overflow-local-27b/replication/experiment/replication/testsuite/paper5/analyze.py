#!/usr/bin/env python3
"""Paper 5 analysis (amendment item 5). Reads the frozen prompts, the generation
journals, and the judge journals; writes paper5/derived/claims.json and a Markdown
summary. Statistics are the audited Paper 4 functions (Wilson, exact McNemar,
Newcombe method 10, Holm), imported unmodified.

Primary family (confirmatory): exact two-sided McNemar of S, C, M against T at
B = 4,096 on the primary-judge labels, all 500 items, Holm across the three.
A missing or failed generation scores incorrect (counted and reported).
Everything else is descriptive with pointwise intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import string
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5common as C  # noqa: E402
from p5run import split_thinking  # noqa: E402

sys.path.insert(0, os.path.join(C.REPO, "paper4", "scripts"))
from verify_claims import (holm_adjust, mcnemar_exact_two_sided,  # noqa: E402
                           newcombe_paired_risk_difference_interval, wilson_interval)

J = os.path.join(C.WORK, "journals")
P = os.path.join(C.WORK, "prompts")
OUT = os.path.join(C.REPO, "paper5", "derived")

PRIMARY_ARMS = ("S", "C", "M")
ALL_ARMS = ("T", "S", "C", "M", "Cw", "Mbm25")


# ------------------------------------------------------------------ loading
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def terminal(rows):
    out = {}
    for r in rows:
        if r.get("terminal"):
            out[r["job_id"]] = r
    return out


def load_prompts():
    out = {}
    for name in sorted(os.listdir(P)):
        if name.endswith(".jsonl"):
            for r in read_jsonl(os.path.join(P, name)):
                out[r["job_id"]] = r
    return out


def judge_labels(journal_name):
    """gen job id -> (label, failed). The retry row, when present, replaces the base row."""
    rows = terminal(read_jsonl(os.path.join(J, journal_name)))
    out = {}
    for jid, r in rows.items():
        base = jid[:-len("|retry")] if jid.endswith("|retry") else jid
        gen = r["meta"]["gen_job_id"]
        content = r.get("text") if r.get("ok") else None
        failed = not (content or "").strip()
        if jid.endswith("|retry") or gen not in out:
            out[gen] = ("yes" in (content or "").lower(), failed)
    return out


# ------------------------------------------------------------------ scorers
_PUNCT = set(string.punctuation)


def normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def containment(answer, gold):
    g = normalize(gold)
    return bool(g) and g in normalize(answer)


def token_f1(answer, gold):
    a, g = normalize(answer).split(), normalize(gold).split()
    if not a or not g:
        return float(a == g)
    common = Counter(a) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(a), same / len(g)
    return 2 * p * r / (p + r)


def kappa(x, y):
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return None if pe == 1 else (po - pe) / (1 - pe)


# ------------------------------------------------------------------ statistics
def acc_summary(labels):
    n, k = len(labels), sum(labels)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "correct": k, "accuracy": k / n, "wilson95": [lo, hi]}


def paired(a, b):
    """a, b: dicts qid->bool on the same items. Returns the a-minus-b paired analysis."""
    ids = sorted(set(a) & set(b))
    both = sum(a[i] and b[i] for i in ids)
    a_only = sum(a[i] and not b[i] for i in ids)
    b_only = sum(b[i] and not a[i] for i in ids)
    neither = len(ids) - both - a_only - b_only
    diff, lo, hi, phi = newcombe_paired_risk_difference_interval(both, b_only, a_only, neither)
    return {"n": len(ids), "both": both, "arm_only": a_only, "reference_only": b_only, "neither": neither,
            "difference": diff, "newcombe95": [lo, hi], "phi": phi,
            "mcnemar_p": mcnemar_exact_two_sided(a_only, b_only)}


# ------------------------------------------------------------------ main analysis
def analyze(gen_journal="grid.jsonl", primary="gptoss", secondary="self"):
    items = {it["question_id"]: it for it in C.load_json_verified(C.LME_S)}
    prompts = load_prompts()
    gens = terminal(read_jsonl(os.path.join(J, gen_journal)))
    lab1 = judge_labels(f"judge_{primary}__{gen_journal}")
    lab2 = judge_labels(f"judge_{secondary}__{gen_journal}")

    cells = defaultdict(dict)        # (arm, budget) -> qid -> record
    for jid, p in prompts.items():
        if p["arm"] not in ALL_ARMS + ("O", "E"):
            continue
        g = gens.get(jid)
        ok = bool(g and g.get("ok"))
        th, ans, closed = split_thinking(g.get("text")) if ok else ("", "", False)
        usage = (g or {}).get("usage") or {}
        l1, f1 = lab1.get(jid, (False, True))
        l2, _ = lab2.get(jid, (False, True))
        it = items[p["question_id"]]
        short = not ("_abs" in it["question_id"] or it["question_type"] == "single-session-preference")
        cells[(p["arm"], p["budget"])][p["question_id"]] = {
            "generated": ok, "closed_thinking": closed, "judge": l1 and ok, "judge_failed": f1,
            "self_judge": l2 and ok, "containment": containment(ans, it["answer"]) if (ok and short) else None,
            "n_prompt": p["n_prompt"], "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": (g or {}).get("finish_reason"), "seconds": (g or {}).get("seconds"),
            "type": it["question_type"], "abstention": "_abs" in it["question_id"], "diag": p.get("diag", {})}

    def labels(arm, budget, key="judge"):
        return {q: bool(r[key]) for q, r in cells.get((arm, budget), {}).items()}

    out = {"schema": "paper5-claims-v1", "gen_journal": gen_journal, "primary_judge": primary,
           "secondary_judge": secondary, "cells": {}, "primary": {}, "secondary": {}}

    for (arm, budget), recs in sorted(cells.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
        L = [r["judge"] for r in recs.values()]
        comp = [r["completion_tokens"] for r in recs.values() if r["completion_tokens"] is not None]
        secs = [r["seconds"] for r in recs.values() if r["seconds"] is not None]
        npr = [r["n_prompt"] for r in recs.values()]
        agree = [(r["judge"], r["self_judge"]) for r in recs.values() if r["generated"]]
        cont = [(r["judge"], r["containment"]) for r in recs.values() if r["containment"] is not None]
        out["cells"][f"{arm}@{budget}"] = {
            **acc_summary(L),
            "self_judge": acc_summary([r["self_judge"] for r in recs.values()]),
            "missing_or_failed_generations": sum(not r["generated"] for r in recs.values()),
            "unterminated_thinking": sum(r["generated"] and not r["closed_thinking"] for r in recs.values()),
            "cap_hits": sum(r["finish_reason"] == "length" for r in recs.values()),
            "judge_failures": sum(r["judge_failed"] for r in recs.values() if r["generated"]),
            "delivered_prompt_tokens_mean": statistics.fmean(npr) if npr else None,
            "delivered_prompt_tokens_median": statistics.median(npr) if npr else None,
            "completion_tokens_mean": statistics.fmean(comp) if comp else None,
            "generation_seconds_mean": statistics.fmean(secs) if secs else None,
            "judge_agreement": {"raw": (sum(a == b for a, b in agree) / len(agree)) if agree else None,
                                "kappa": kappa([a for a, _ in agree], [b for _, b in agree])},
            "containment_vs_judge": {"n": len(cont),
                                     "containment_accuracy": (sum(b for _, b in cont) / len(cont)) if cont else None,
                                     "agreement": (sum(a == b for a, b in cont) / len(cont)) if cont else None,
                                     "kappa": kappa([a for a, _ in cont], [b for _, b in cont])},
            "by_type": {t: acc_summary([r["judge"] for r in recs.values() if r["type"] == t])
                        for t in sorted({r["type"] for r in recs.values()})},
            "abstention": acc_summary([r["judge"] for r in recs.values() if r["abstention"]])
            if any(r["abstention"] for r in recs.values()) else None,
            "diag_means": {k: statistics.fmean([r["diag"][k] for r in recs.values()
                                                if isinstance(r["diag"].get(k), (int, float))
                                                and not isinstance(r["diag"].get(k), bool)])
                           for k in sorted({k for r in recs.values() for k in r["diag"]})
                           if any(isinstance(r["diag"].get(k), (int, float)) and not isinstance(r["diag"].get(k), bool)
                                  for r in recs.values())},
            "diag_counts": {k: sum(bool(r["diag"].get(k)) for r in recs.values())
                            for k in sorted({k for r in recs.values() for k in r["diag"]})
                            if any(isinstance(r["diag"].get(k), bool) for r in recs.values())},
        }

    # primary family
    T4 = labels("T", 4096)
    fam = {a: paired(labels(a, 4096), T4) for a in PRIMARY_ARMS if (a, 4096) in cells}
    holm = holm_adjust({a: v["mcnemar_p"] for a, v in fam.items()}) if fam else {}
    for a in fam:
        fam[a]["holm_p"] = holm[a]
        fam[a]["rejected_at_0.05"] = holm[a] <= 0.05
    out["primary"] = {"contrasts_vs_T_at_4096": fam, "family": list(fam), "alpha": 0.05}

    # secondary
    sec = {}
    T1 = labels("T", 1024)
    for a in ("S", "C", "M", "Cw"):
        if (a, 1024) in cells and T1:
            sec[f"{a}_vs_T@1024"] = paired(labels(a, 1024), T1)
    for a in ("Cw", "Mbm25"):
        if (a, 4096) in cells:
            sec[f"{a}_vs_T@4096"] = paired(labels(a, 4096), T4)
    if ("Cw", 4096) in cells and ("C", 4096) in cells:
        sec["Cw_vs_C@4096"] = paired(labels("Cw", 4096), labels("C", 4096))
    if ("Mbm25", 4096) in cells and ("M", 4096) in cells:
        sec["Mbm25_vs_M@4096"] = paired(labels("Mbm25", 4096), labels("M", 4096))
    # ranking by budget
    rank = {}
    for B in (1024, 4096):
        accs = {a: out["cells"][f"{a}@{B}"]["accuracy"] for a in ("T", "S", "C", "M") if f"{a}@{B}" in out["cells"]}
        rank[B] = sorted(accs, key=lambda a: -accs[a])
    sec["ranking"] = rank
    # oracle gap on the fitting subset
    if ("O", 4096) in cells:
        O = labels("O", 4096)
        gap = {}
        for a in ("T", "S", "C", "M"):
            if (a, 4096) in cells:
                A = {q: v for q, v in labels(a, 4096).items() if q in O}
                gap[a] = paired(O, A)
        sec["oracle_gap_on_fit_subset"] = {"n_fit": len(O), "O_minus_arm": gap}
    out["secondary"] = sec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="grid.jsonl")
    ap.add_argument("--out", default="claims.json")
    a = ap.parse_args()
    res = analyze(a.gen)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, a.out), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    for k, v in res["primary"].get("contrasts_vs_T_at_4096", {}).items():
        print(f"{k} vs T @4096: diff {v['difference']:+.3f} [{v['newcombe95'][0]:+.3f}, {v['newcombe95'][1]:+.3f}] "
              f"McNemar p {v['mcnemar_p']:.3g} Holm {v['holm_p']:.3g}")
    for k, v in sorted(res["cells"].items()):
        print(f"{k:12s} acc {v['accuracy']:.3f} n {v['n']} tokens {v['delivered_prompt_tokens_mean']}")


if __name__ == "__main__":
    main()
