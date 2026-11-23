"""Independent reconstruction of every P5 primary analysis aggregate.

Uses audited raw requests plus source items, not reported cell percentages.
Wilson intervals and exact binomial p-values come from SciPy. The paired
interval uses the contingency-count phi formula, separately from the frozen
analyzer's probability-based implementation. No inference or output writes.
"""
from collections import defaultdict
import json
import math

import diagnostics as D


def mean(values):
    return math.fsum(values) / len(values) if values else None


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n//2] if n % 2 else (ordered[n//2-1] + ordered[n//2]) / 2


def summary(values):
    from scipy.stats import binomtest
    n, k = len(values), sum(values)
    D.require(n > 0, "empty binomial cohort")
    ci = binomtest(k, n).proportion_ci(confidence_level=.95, method="wilson")
    return {"n": n, "correct": k, "accuracy": k/n, "wilson95": [float(ci.low), float(ci.high)]}


def agreement(pairs):
    if not pairs:
        return None, None
    n = len(pairs)
    agree = sum(x == y for x, y in pairs) / n
    # Expected disagreement under independent marginals. This is algebraically
    # equivalent to Cohen's kappa but avoids reusing the analyzer's function.
    nx, ny = sum(x for x, _ in pairs), sum(y for _, y in pairs)
    chance_disagreements = nx*(n-ny) + ny*(n-nx)
    kappa = 1 - (sum(x != y for x, y in pairs) * n / chance_disagreements) if chance_disagreements else None
    return agree, kappa


def paired(left, right):
    from scipy.stats import binomtest
    D.require(left.keys() == right.keys() and bool(left), "paired cohorts are not identical")
    a = sum(left[k] and right[k] for k in left)
    b = sum(left[k] and not right[k] for k in left)
    c = sum(not left[k] and right[k] for k in left)
    d = sum(not left[k] and not right[k] for k in left)
    n = a+b+c+d
    x, y = (a+b)/n, (a+c)/n
    sx, sy = summary(list(left.values())), summary(list(right.values()))
    denominator = math.sqrt((a+b)*(a+c)*(b+d)*(c+d))
    phi = (a*d-b*c)/denominator if denominator else 0.
    diff = x-y
    x_low, x_high = sx["wilson95"]
    y_low, y_high = sy["wilson95"]
    # Squared distance combining the two one-sided marginal errors with phi.
    def radius(u, v):
        return math.sqrt(max(0., (u-v)**2 + 2*(1-phi)*u*v))
    limits = [max(-1., diff-radius(x-x_low, y_high-y)), min(1., diff+radius(x_high-x, y-y_low))]
    p = float(binomtest(b, b+c, .5, alternative="two-sided").pvalue) if b+c else 1.
    return {"n": n, "both": a, "arm_only": b, "reference_only": c, "neither": d,
            "difference": diff, "newcombe95": limits, "phi": phi, "mcnemar_p": p}


def reconstruct(items, prompts, generations, judges):
    by_id = {it["question_id"]: it for it in items}
    scores = {name: D.verdicts(generations, records, name) for name, records in judges.items()}
    cells = defaultdict(dict)
    for p in prompts:
        jid, qid = p["job_id"], p["question_id"]
        g, it = generations[jid], by_id[qid]
        answer, closed = D.visible(g.get("text") if g["ok"] else None)
        lexical = None
        if g["ok"] and "_abs" not in qid and it["question_type"] != "single-session-preference":
            gold = D.normalize(it["answer"])
            lexical = bool(gold) and gold in D.normalize(answer)
        cells[f"{p['arm']}@{p['budget']}"][qid] = {
            "correct": scores["gptoss"][jid]["label"], "self_correct": scores["self"][jid]["label"],
            "judge_failed": scores["gptoss"][jid]["failed"], "lexical": lexical,
            "generated": g["ok"], "unclosed": g["ok"] and not closed, "cap_hit": g.get("finish_reason") == "length",
            "prompt_tokens": p["n_prompt"], "completion_tokens": (g.get("usage") or {}).get("completion_tokens"),
            "seconds": g.get("seconds"), "type": it["question_type"], "abstention": "_abs" in qid, "diag": p.get("diag", {})}
    out = {"schema": "paper5-claims-v1", "gen_journal": "grid.jsonl", "primary_judge": "gptoss", "secondary_judge": "self",
           "cells": {}, "primary": {}, "secondary": {}}
    for key, observations in cells.items():
        records = list(observations.values())
        local_self = [(r["correct"], r["self_correct"]) for r in records if r["generated"]]
        local_lexical = [(r["correct"], r["lexical"]) for r in records if r["lexical"] is not None]
        raw, kappa = agreement(local_self)
        lex_raw, lex_kappa = agreement(local_lexical)
        types = sorted({r["type"] for r in records})
        abstentions = [r["correct"] for r in records if r["abstention"]]
        diagnostic_values = defaultdict(list)
        for r in records:
            for name, value in r["diag"].items():
                diagnostic_values[name].append(value)
        out["cells"][key] = {
            **summary([r["correct"] for r in records]), "self_judge": summary([r["self_correct"] for r in records]),
            "missing_or_failed_generations": sum(not r["generated"] for r in records),
            "unterminated_thinking": sum(r["unclosed"] for r in records), "cap_hits": sum(r["cap_hit"] for r in records),
            "judge_failures": sum(r["judge_failed"] for r in records if r["generated"]),
            "delivered_prompt_tokens_mean": mean([r["prompt_tokens"] for r in records]),
            "delivered_prompt_tokens_median": median([r["prompt_tokens"] for r in records]),
            "completion_tokens_mean": mean([r["completion_tokens"] for r in records if r["completion_tokens"] is not None]),
            "generation_seconds_mean": mean([r["seconds"] for r in records if r["seconds"] is not None]),
            "judge_agreement": {"raw": raw, "kappa": kappa},
            "containment_vs_judge": {"n": len(local_lexical), "containment_accuracy": mean([y for _, y in local_lexical]),
                                     "agreement": lex_raw, "kappa": lex_kappa},
            "by_type": {t: summary([r["correct"] for r in records if r["type"] == t]) for t in types},
            "abstention": summary(abstentions) if abstentions else None,
            "diag_means": {name: mean([v for v in values if type(v) in (int, float)])
                           for name, values in diagnostic_values.items() if any(type(v) in (int, float) for v in values)},
            "diag_counts": {name: sum(bool(r["diag"].get(name)) for r in records)
                            for name, values in diagnostic_values.items() if any(type(v) is bool for v in values)},
        }
    def labels(arm, budget):
        return {qid: row["correct"] for qid, row in cells[f"{arm}@{budget}"].items()}
    family = {arm: paired(labels(arm, 4096), labels("T", 4096)) for arm in ("S", "C", "M")}
    ordered = sorted(family, key=lambda arm: (family[arm]["mcnemar_p"], arm))
    adjusted = []
    for i, arm in enumerate(ordered):
        adjusted.append((3-i)*family[arm]["mcnemar_p"])
        p = min(1., max(adjusted))
        family[arm].update(holm_p=p, rejected_at_0_05=p <= .05)
        family[arm]["rejected_at_0.05"] = family[arm].pop("rejected_at_0_05")
    out["primary"] = {"contrasts_vs_T_at_4096": family, "family": ["S", "C", "M"], "alpha": .05}
    secondary = {}
    contrasts = [(arm, "T", 1024) for arm in ("S", "C", "M", "Cw")]
    contrasts += [("Cw", "T", 4096), ("Mbm25", "T", 4096), ("Cw", "C", 4096), ("Mbm25", "M", 4096)]
    for arm, ref, budget in contrasts:
        secondary[f"{arm}_vs_{ref}@{budget}"] = paired(labels(arm, budget), labels(ref, budget))
    secondary["ranking"] = {str(budget): sorted(("T", "S", "C", "M"),
                                               key=lambda arm: -out["cells"][f"{arm}@{budget}"]["accuracy"])
                            for budget in (1024, 4096)}
    oracle = labels("O", 4096)
    secondary["oracle_gap_on_fit_subset"] = {"n_fit": len(oracle), "O_minus_arm": {
        arm: paired(oracle, {qid: value for qid, value in labels(arm, 4096).items() if qid in oracle})
        for arm in ("T", "S", "C", "M")}}
    out["secondary"] = secondary
    return out


def compare(actual, expected, path="claims"):
    if isinstance(expected, dict):
        D.require(isinstance(actual, dict) and actual.keys() == expected.keys(), "independent keys differ: " + path)
        for key, value in expected.items():
            compare(actual[key], value, path + "." + key)
    elif isinstance(expected, list):
        D.require(isinstance(actual, list) and len(actual) == len(expected), "independent list differs: " + path)
        for i, value in enumerate(expected):
            compare(actual[i], value, f"{path}[{i}]")
    elif type(expected) is float:
        D.require(type(actual) in (float, int) and math.isfinite(actual), "invalid reported number: " + path)
        # A tiny nonzero p-value must not be accepted as zero merely because
        # an absolute tolerance would hide the difference.
        abs_tol = 0. if path.endswith((".mcnemar_p", ".holm_p")) else 2e-12
        D.require(math.isclose(actual, expected, rel_tol=5e-10, abs_tol=abs_tol), "independent number differs: " + path)
    else:
        D.require(type(actual) is type(expected) and actual == expected, "independent value differs: " + path)


def verify(claims, items, prompts, generations, judges):
    import scipy
    expected = reconstruct(items, prompts, generations, judges)
    # Treat on-disk JSON consistently with the analyzer's numeric ranking keys.
    saved = json.loads(json.dumps(claims, allow_nan=False))
    compare(saved, expected)
    return {"schema": "p5-independent-primary-statistics-v1", "scipy": scipy.__version__,
            "scope": "all primary analysis fields, paired tables, Wilson/Newcombe intervals, exact McNemar p-values, Holm, subgroups and descriptive summaries"}
