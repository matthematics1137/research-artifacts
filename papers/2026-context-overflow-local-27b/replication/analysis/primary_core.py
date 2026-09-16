"""Verbatim primary aggregation after text-to-score projection."""
import statistics
PRIMARY_ARMS = ("S", "C", "M")
ALL_ARMS = ("T", "S", "C", "M", "Cw", "Mbm25")

# Full source SHA256: d12981918b8c0b65715b525b607ec3ad6e81f55c26c0d775d8efa2cd05027cb3
def kappa(x, y):
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return None if pe == 1 else (po - pe) / (1 - pe)


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


def aggregate(cells, gen_journal="grid.jsonl", primary="gptoss", secondary="self"):
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

