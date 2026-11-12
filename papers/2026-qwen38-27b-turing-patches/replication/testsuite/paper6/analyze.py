#!/usr/bin/env python3
"""Paper 6 analysis (2026-09-19 amendment §7-8): paired item-level contrasts over the same
200 long-context items, the two co-primaries with Holm, the pre-registered secondaries, and
the measured equal-memory ranking.

  analyze.py                      # work/journals -> work/analysis/{analysis.json,summary.md}
  analyze.py --selftest           # synthetic journals with hand-computed answers

Definitions (accuracy points; every contrast is paired on the item id):
  D_m(w) = acc(w, f16) - acc(w, q4_0)        per item: y(w,f16) - y(w,q4_0)
  I_m    = D_m(Q2_K_XL) - D_m(Q4_K_XL)       per item: a sum of +-1 indicators over 4 cells
  M      = I_hybrid - I_GQA                  per item: over 8 cells
Co-primary 1: M != 0, two-sided sign-flip permutation (100,000 draws, fixed seed) with a
t 95% interval. Co-primary 2: I_hybrid equivalent to 0 within +-10 points (TOST at 0.05:
the 90% t interval and the two one-sided p-values, p = max). Holm across the two.

Statistics use the standard library and numpy only: Wilson
intervals, an exact-binomial McNemar, Student-t quantiles from the regularized incomplete
beta, and the permutation test on integer per-item contrasts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from fractions import Fraction
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6common as P  # noqa: E402

Z975 = NormalDist().inv_cdf(0.975)
PERM_DRAWS = 100_000
PERM_SEED = 20260919
DELTA_POINTS = 10.0                 # TOST margin (accuracy points)
DEPTH_BINS = 5


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = Z975) -> dict:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 400, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d, c = (FPMIN if abs(d) < FPMIN else d), (FPMIN if abs(c) < FPMIN else c)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d, c = (FPMIN if abs(d) < FPMIN else d), (FPMIN if abs(c) < FPMIN else c)
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) (continued fraction, Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    if df <= 0:
        return float("nan")
    p = 0.5 * betainc_reg(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t > 0 else p


def t_sf(t: float, df: float) -> float:
    return t_cdf(-t, df)


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile by bisection on the CDF (exact to ~1e-12 for the df used here)."""
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_ci(x, level: float = 0.95) -> dict:
    """Mean of paired per-item contrasts with a t interval, in accuracy points."""
    n = len(x)
    if n == 0:
        return {"n": 0, "mean_pts": None, "lo_pts": None, "hi_pts": None, "sd_pts": None, "se_pts": None}
    m = statistics.fmean(x)
    sd = statistics.stdev(x) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    q = t_ppf(1 - (1 - level) / 2, n - 1) if n > 1 else 0.0
    return {"n": n, "level": level, "mean_pts": 100 * m, "sd_pts": 100 * sd, "se_pts": 100 * se,
            "t_quantile": q, "lo_pts": 100 * (m - q * se), "hi_pts": 100 * (m + q * se)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial sign test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)
    return min(1.0, float(2 * tail))


def signflip_p(x, draws: int = PERM_DRAWS, seed: int = PERM_SEED, chunk: int = 5000) -> dict:
    """Two-sided sign-flip permutation p for the mean of paired contrasts.
    p = (1 + #{|sum*| >= |sum_obs|}) / (1 + draws)  (Phipson & Smyth: never reports 0)."""
    v = np.asarray(x, dtype=np.float64)
    if v.size == 0:
        return {"p": None, "draws": draws, "seed": seed, "n": 0}
    obs = abs(float(v.sum()))
    rng = np.random.default_rng(seed)
    ge, left = 0, draws
    while left > 0:
        b = min(chunk, left)
        signs = rng.integers(0, 2, size=(b, v.size), dtype=np.int8).astype(np.float64) * 2.0 - 1.0
        ge += int((np.abs(signs @ v) >= obs - 1e-9).sum())
        left -= b
    return {"p": (1 + ge) / (1 + draws), "draws": draws, "seed": seed, "n": int(v.size),
            "observed_sum": float(v.sum()), "n_ge": ge}


def tost(x, delta_pts: float = DELTA_POINTS) -> dict:
    """Two one-sided tests for equivalence to 0 within +-delta (accuracy points)."""
    d = delta_pts / 100.0
    n = len(x)
    if n < 2:
        return {"n": n, "equivalent": None}
    m = statistics.fmean(x)
    se = statistics.stdev(x) / math.sqrt(n)
    df = n - 1
    if se == 0:
        p_lo = 0.0 if m > -d else 1.0
        p_hi = 0.0 if m < d else 1.0
        lo = hi = m
        # No finite t statistic exists at zero standard error. Keep the
        # limiting p-values above, but serialize the diagnostic as null,
        # not non-standard JSON Infinity (nor a misleading +inf for both tails).
        t_lo = t_hi = None
    else:
        t_lo, t_hi = (m + d) / se, (m - d) / se
        p_lo, p_hi = t_sf(t_lo, df), t_cdf(t_hi, df)
        q = t_ppf(0.95, df)
        lo, hi = m - q * se, m + q * se
    return {"n": n, "delta_pts": delta_pts, "mean_pts": 100 * m, "se_pts": 100 * se, "df": df,
            "ci90_lo_pts": 100 * lo, "ci90_hi_pts": 100 * hi,
            "t_lower": t_lo, "t_upper": t_hi, "zero_standard_error": se == 0,
            "p_lower": p_lo, "p_upper": p_hi,
            "p_tost": max(p_lo, p_hi), "equivalent": bool(-delta_pts < 100 * lo and 100 * hi < delta_pts)}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values."""
    order = sorted(pvals.items(), key=lambda kv: (kv[1] is None, kv[1]))
    k = len(order)
    adj, prev = {}, 0.0
    for i, (name, p) in enumerate(order):
        if p is None:
            adj[name] = None
            continue
        a = max(prev, min(1.0, (k - i) * p))
        adj[name] = a
        prev = a
    return adj


# ---------------------------------------------------------------- data
def collect(journal_dir: str, items_lc: list[dict], items_gsm: list[dict]) -> dict:
    """Re-score every journal from the stored response text (the journal's own score is only
    checked against it) and index the outcomes by cell, task and item id."""
    by_id = {"longctx": {it["id"]: it for it in items_lc}, "gsm8k": {it["id"]: it for it in items_gsm}}
    cells = {}
    for cell in P.all_cells():
        name = P.cell_name(*cell)
        J = P.read_journal(P.journal_path(*cell, journals=journal_dir))
        res = {"cell": cell, "name": name, "longctx": {}, "gsm8k": {}, "hits": {}, "identity": J["identity"],
               "failed": [], "unknown_items": [], "rescore_mismatch": [], "truncated": {"longctx": 0, "gsm8k": 0},
               "seconds": {"longctx": 0.0, "gsm8k": 0.0}, "prompt_tokens": [], "prompt_deltas": [],
               "completion_tokens": {"longctx": [], "gsm8k": []}, "n_rows": len(J["outcomes"])}
        for jid, row in J["outcomes"].items():
            task, item_id = jid.split("|", 1)
            item = by_id.get(task, {}).get(item_id)
            if item is None:
                res["unknown_items"].append(jid)
                continue
            if not row.get("ok"):
                res["failed"].append(jid)
                continue
            sc = P.score_row(task, row, item)
            stored = (row.get("score") or {}).get("correct")
            if stored is not None and bool(stored) != sc["correct"]:
                res["rescore_mismatch"].append(jid)
            res[task][item_id] = 1 if sc["correct"] else 0
            if task == "longctx":
                res["hits"][item_id] = sc["answers_found"]
            if row.get("finish_reason") == "length":
                res["truncated"][task] += 1
            res["seconds"][task] += row.get("seconds") or 0.0
            u = row.get("usage") or {}
            if u.get("prompt_tokens") and task == "longctx":
                res["prompt_tokens"].append(u["prompt_tokens"])
            if u.get("completion_tokens"):
                res["completion_tokens"][task].append(u["completion_tokens"])
            if row.get("meta", {}).get("prompt_tokens_delta") is not None:
                res["prompt_deltas"].append(row["meta"]["prompt_tokens_delta"])
        cells[name] = res
    return cells


def y(cells: dict, model: str, tier: str, kv: str, task: str) -> dict:
    return cells[P.cell_name(model, tier, kv)][task]


def common_ids(cells: dict, need: list[tuple], task: str, ids: list[str]) -> list[str]:
    """Item ids scored in every cell of `need` (complete cases; the paired unit of analysis)."""
    maps = [y(cells, *c, task) for c in need]
    return [i for i in ids if all(i in m for m in maps)]


def d_vector(cells, model, tier, task, ids):
    a, b = y(cells, model, tier, "f16", task), y(cells, model, tier, "q4_0", task)
    return [a[i] - b[i] for i in ids]


def i_vector(cells, model, task, ids):
    return [d2 - d4 for d2, d4 in zip(d_vector(cells, model, "Q2_K_XL", task, ids),
                                      d_vector(cells, model, "Q4_K_XL", task, ids))]


def estimands(cells: dict, task: str, ids: list[str], draws: int, seed: int, delta: float) -> dict:
    """D per (model, tier), I per model, and M, all as paired item-level contrasts."""
    out = {"task": task, "D": {}, "I": {}, "M": None, "n_items_total": len(ids)}
    for model in P.MODELS:
        out["D"][model] = {}
        for tier in P.TIERS:
            need = [(model, tier, "f16"), (model, tier, "q4_0")]
            cids = common_ids(cells, need, task, ids)
            out["D"][model][tier] = {**mean_ci(d_vector(cells, model, tier, task, cids)),
                                     "cells": [P.cell_name(*c) for c in need]}
        need4 = [(model, t, k) for t in ("Q2_K_XL", "Q4_K_XL") for k in ("f16", "q4_0")]
        cids = common_ids(cells, need4, task, ids)
        v = i_vector(cells, model, task, cids)
        out["I"][model] = {**mean_ci(v), "cells": [P.cell_name(*c) for c in need4],
                           "tost": tost(v, delta), "permutation": signflip_p(v, draws, seed) if v else None}
    need8 = [(m, t, k) for m in (P.HYBRID, P.GQA) for t in ("Q2_K_XL", "Q4_K_XL") for k in ("f16", "q4_0")]
    cids = common_ids(cells, need8, task, ids)
    mv = [a - b for a, b in zip(i_vector(cells, P.HYBRID, task, cids), i_vector(cells, P.GQA, task, cids))]
    out["M"] = {**mean_ci(mv), "definition": "I_hybrid - I_GQA", "cells": [P.cell_name(*c) for c in need8],
                "permutation": signflip_p(mv, draws, seed) if mv else None,
                "missing_items": [i for i in ids if i not in cids][:20]}
    return out


def secondary(cells: dict, task: str, ids: list[str]) -> dict:
    """Exact McNemar for each KV type vs f16, the K8/V4 comparisons, and paired flip counts."""
    rows, k8 = [], []
    for model in P.MODELS:
        for tier in P.TIERS:
            f16 = y(cells, model, tier, "f16", task)
            for kv in P.KV_ORDER:
                if kv == "f16":
                    continue
                other = y(cells, model, tier, kv, task)
                cids = [i for i in ids if i in f16 and i in other]
                b = sum(1 for i in cids if f16[i] == 1 and other[i] == 0)     # lost vs f16
                c = sum(1 for i in cids if f16[i] == 0 and other[i] == 1)     # gained vs f16
                rows.append({"model": model, "tier": tier, "kv": kv, "n": len(cids),
                             "acc_f16": (sum(f16[i] for i in cids) / len(cids)) if cids else None,
                             "acc_kv": (sum(other[i] for i in cids) / len(cids)) if cids else None,
                             "lost": b, "gained": c, "flips": b + c,
                             "diff_pts": 100 * (sum(other[i] - f16[i] for i in cids) / len(cids)) if cids else None,
                             "mcnemar_p": mcnemar_exact(b, c),
                             "paired_diff": mean_ci([other[i] - f16[i] for i in cids])})
            for ref in ("q8_0", "q4_0"):
                a, o = y(cells, model, tier, "k8v4", task), y(cells, model, tier, ref, task)
                cids = [i for i in ids if i in a and i in o]
                b = sum(1 for i in cids if o[i] == 1 and a[i] == 0)
                c = sum(1 for i in cids if o[i] == 0 and a[i] == 1)
                k8.append({"model": model, "tier": tier, "vs": ref, "n": len(cids),
                           "k8v4_better": c, "k8v4_worse": b, "mcnemar_p": mcnemar_exact(b, c),
                           "paired_diff": mean_ci([a[i] - o[i] for i in cids])})
    return {"kv_vs_f16": rows, "k8v4": k8}


def trend(cells: dict, task: str, ids: list[str]) -> list[dict]:
    """The IQ3 tier's D and the adjacent-tier differences (the trend over tiers)."""
    out = []
    for model in P.MODELS:
        for lo, hi in (("Q2_K_XL", "IQ3_XXS"), ("IQ3_XXS", "Q4_K_XL")):
            need = [(model, t, k) for t in (lo, hi) for k in ("f16", "q4_0")]
            cids = common_ids(cells, need, task, ids)
            v = [a - b for a, b in zip(d_vector(cells, model, lo, task, cids),
                                       d_vector(cells, model, hi, task, cids))]
            out.append({"model": model, "contrast": f"D({lo}) - D({hi})", **mean_ci(v)})
    return out


def depth_bins(cells: dict, items_lc: list[dict]) -> dict:
    """Long-context accuracy by needle depth: item accuracy by the asked needle's depth
    (single, multikey) and per-required-answer recall by that answer's needle depth (all)."""
    by_id = {it["id"]: it for it in items_lc}
    out = {}
    for name, res in cells.items():
        item_bins = [[0, 0] for _ in range(DEPTH_BINS)]
        ans_bins = [[0, 0] for _ in range(DEPTH_BINS)]
        for item_id, ok in res["longctx"].items():
            it = by_id.get(item_id)
            if it is None:
                continue
            if it["asked_depth"] is not None:
                b = min(DEPTH_BINS - 1, int(it["asked_depth"] * DEPTH_BINS))
                item_bins[b][0] += ok
                item_bins[b][1] += 1
            hits = set(res["hits"].get(item_id, []))
            for ans, depth in it["answer_depths"].items():
                b = min(DEPTH_BINS - 1, int(depth * DEPTH_BINS))
                ans_bins[b][0] += 1 if ans in hits else 0
                ans_bins[b][1] += 1
        out[name] = {"bin_edges": [round(i / DEPTH_BINS, 2) for i in range(DEPTH_BINS + 1)],
                     "item_accuracy": [{"k": k, "n": n, "acc": (k / n) if n else None} for k, n in item_bins],
                     "answer_recall": [{"k": k, "n": n, "recall": (k / n) if n else None} for k, n in ans_bins]}
    return out


def by_family(cells: dict, items_lc: list[dict]) -> dict:
    fam = {it["id"]: it["family"] for it in items_lc}
    out = {}
    for name, res in cells.items():
        d = {}
        for item_id, ok in res["longctx"].items():
            f = fam.get(item_id)
            if f:
                d.setdefault(f, [0, 0])
                d[f][0] += ok
                d[f][1] += 1
        out[name] = {f: wilson(k, n) for f, (k, n) in sorted(d.items())}
    return out


def equal_memory(cells: dict, ids: list[str]) -> dict:
    """Measured device memory and long-context accuracy of the fully resident cells
    (amendment §8). Partly offloaded cells are listed but excluded."""
    out = {}
    for model in P.MODELS:
        resident, excluded = [], []
        for tier in P.TIERS:
            for kv in P.KV_ORDER:
                name = P.cell_name(model, tier, kv)
                res = cells[name]
                if not res["identity"]:
                    continue
                ident = res["identity"][0]
                load = ident.get("load") or {}
                kvc = load.get("kv_cache") or {}
                cellsn = kvc.get("cells")
                scored = [i for i in ids if i in res["longctx"]]
                row = {"cell": name, "tier": tier, "kv": kv,
                       "layers": f"{load.get('offloaded_layers')}/{load.get('total_layers')}",
                       "fully_resident": load.get("fully_resident"),
                       "device_mib": load.get("device_mib"), "host_mib": load.get("host_mib"),
                       "kv_total_mib": kvc.get("total_mib"), "kv_cells": cellsn,
                       "kv_mib_at_16k": (kvc["total_mib"] * 16384 / cellsn) if kvc.get("total_mib") and cellsn else None,
                       "kv_mib_at_32k_computed": (kvc["total_mib"] * 32768 / cellsn) if kvc.get("total_mib") and cellsn else None,
                       "recurrent_mib": (load.get("recurrent") or {}).get("total_mib"),
                       "nvidia_process_mib": (ident.get("gpu_loaded") or {}).get("process_mib"),
                       "starts": len(res["identity"]),
                       "placement_changed": any(r.get("placement_changed") for r in res["identity"]),
                       "accuracy": wilson(sum(res["longctx"][i] for i in scored), len(scored))}
                (resident if load.get("fully_resident") else excluded).append(row)
        resident.sort(key=lambda r: (-(r["accuracy"]["p"] if r["accuracy"]["p"] is not None else -1),
                                     r["device_mib"]["total_model_kv_rs_compute"] if r["device_mib"] else 0))
        # the amendment's exchange-rate prediction, checked against the measured buffers
        pred = []
        kv16 = {(r["tier"], r["kv"]): r["kv_mib_at_16k"] for r in resident if r["kv_mib_at_16k"]}
        model_mib = {}
        for r in resident + excluded:
            if r["device_mib"] and r["kv"] == "f16":
                model_mib[r["tier"]] = (r["device_mib"]["model"] or 0) + ((r["host_mib"] or {}).get("model") or 0)
        for tier in P.TIERS:
            if (tier, "f16") in kv16 and (tier, "q4_0") in kv16:
                pred.append({"tier": tier, "kv_freed_f16_to_q4_at16k_mib": kv16[(tier, "f16")] - kv16[(tier, "q4_0")]})
        for lo, hi in (("Q2_K_XL", "IQ3_XXS"), ("IQ3_XXS", "Q4_K_XL")):
            if lo in model_mib and hi in model_mib:
                pred.append({"weight_step": f"{lo} -> {hi}", "model_bytes_step_mib": model_mib[hi] - model_mib[lo]})
        out[model] = {"resident": resident, "excluded_partly_offloaded": excluded, "exchange_rate": pred}
    return out


def token_report(cells: dict, items_lc: list[dict], journal_dir: str) -> dict:
    hay = [it["haystack_tokens"] for it in items_lc]
    out = {"qwen38_tokenizer": {"haystack_min": min(hay), "haystack_max": max(hay),
                                "haystack_mean": round(sum(hay) / len(hay), 1),
                                "content_mean": round(sum(it["content_tokens"] for it in items_lc) / len(items_lc), 1)},
           "served_tokenizer": {}, "prompt_tokens_observed": {}, "prompt_token_deltas_hybrid": {}}
    counts_dir = os.path.join(os.path.dirname(journal_dir), "token_counts")
    for model in P.MODELS:
        p = os.path.join(counts_dir, f"{model}.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            hs = [v["haystack"] for v in d["items"].values()]
            out["served_tokenizer"][model] = {"n": len(hs), "haystack_min": min(hs), "haystack_max": max(hs),
                                              "haystack_mean": round(sum(hs) / len(hs), 1)}
        obs = [t for name, res in cells.items() if name.startswith(model + "__") for t in res["prompt_tokens"]]
        if obs:
            out["prompt_tokens_observed"][model] = {"n": len(obs), "min": min(obs), "max": max(obs),
                                                    "median": round(statistics.median(obs), 1)}
        dl = [t for name, res in cells.items() if name.startswith(model + "__") for t in res["prompt_deltas"]]
        if dl:
            out["prompt_token_deltas_hybrid"][model] = {"n": len(dl), "min": min(dl), "max": max(dl),
                                                        "nonzero": sum(1 for t in dl if t)}
    return out


# ---------------------------------------------------------------- report
def analyse(journal_dir: str, items_path: str, draws: int, seed: int, delta: float) -> dict:
    from p6evidence import evidence_snapshot
    evidence_before = evidence_snapshot(journal_dir)
    items_lc = P.load_longctx(items_path)
    items_gsm = P.load_gsm8k(verify=False)
    ids_lc = [it["id"] for it in items_lc]
    ids_gsm = [it["id"] for it in items_gsm]
    cells = collect(journal_dir, items_lc, items_gsm)

    per_cell, flags = {}, []
    for name, res in cells.items():
        entry = {"identity_starts": len(res["identity"]), "failed_jobs": len(res["failed"]),
                 "rescore_mismatches": len(res["rescore_mismatch"]),
                 "seconds": {k: round(v, 1) for k, v in res["seconds"].items()}}
        for task, ids in (("longctx", ids_lc), ("gsm8k", ids_gsm)):
            got = res[task]
            entry[task] = {**wilson(sum(got.values()), len(got)), "expected": len(ids),
                           "missing": [i for i in ids if i not in got][:10],
                           "n_missing": sum(1 for i in ids if i not in got),
                           "truncated": res["truncated"][task],
                           "mean_completion_tokens": round(statistics.fmean(res["completion_tokens"][task]), 1)
                           if res["completion_tokens"][task] else None}
            if got and entry[task]["n_missing"]:
                flags.append(f"{name}/{task}: {entry[task]['n_missing']} of {len(ids)} items missing")
        if res["rescore_mismatch"]:
            flags.append(f"{name}: {len(res['rescore_mismatch'])} rows re-score differently from the stored score")
        if res["failed"]:
            flags.append(f"{name}: {len(res['failed'])} failed requests")
        if res["identity"]:
            ident = res["identity"][0]
            g = ident.get("gguf") or {}
            entry["gguf_sha256"] = g.get("sha256")
            entry["gguf_verified"] = g.get("verified")
            entry["engine_commit"] = (ident.get("engine") or {}).get("commit")
            entry["fit_layers"] = (ident.get("load") or {}).get("offloaded_layers")
            entry["fully_resident"] = (ident.get("load") or {}).get("fully_resident")
            if not g.get("verified"):
                flags.append(f"{name}: GGUF SHA-256 not verified against a receipt")
            if any(r.get("placement_changed") for r in res["identity"]):
                flags.append(f"{name}: --fit chose a different layer count on a restart")
        else:
            entry["gguf_sha256"] = None
        per_cell[name] = entry

    est = {t: estimands(cells, t, ids, draws, seed, delta)
           for t, ids in (("longctx", ids_lc), ("gsm8k", ids_gsm))}
    co1 = est["longctx"]["M"]
    co2 = est["longctx"]["I"][P.HYBRID]["tost"]
    p_holm = holm({"co_primary_1_M_not_zero": (co1.get("permutation") or {}).get("p"),
                   "co_primary_2_I_hybrid_equivalence": co2.get("p_tost")})
    co_primary = {
        "1_moderation": {"estimand": "M = I_hybrid - I_GQA (long context)", "estimate": co1,
                         "p_permutation": (co1.get("permutation") or {}).get("p"),
                         "p_holm": p_holm["co_primary_1_M_not_zero"],
                         "reject_at_0.05": (p_holm["co_primary_1_M_not_zero"] is not None
                                            and p_holm["co_primary_1_M_not_zero"] < 0.05)},
        "2_equivalence": {"estimand": f"I_hybrid within +-{delta:g} points (long context)", "tost": co2,
                          "p_holm": p_holm["co_primary_2_I_hybrid_equivalence"],
                          "reject_at_0.05": (p_holm["co_primary_2_I_hybrid_equivalence"] is not None
                                             and p_holm["co_primary_2_I_hybrid_equivalence"] < 0.05)},
        "holm": p_holm, "predictions": "I_hybrid equivalent to 0, and I_GQA >= I_hybrid",
    }
    if co1["n"] < len(ids_lc):
        flags.append(f"co-primaries use {co1['n']}/{len(ids_lc)} items (complete cases across the 8 cells)")

    out = {
        "what": "Paper 6 analysis (2026-09-19 amendment §7-8)",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {"journal_dir": journal_dir, "items": items_path, "items_sha256": P.sha256_file(items_path),
                   "gsm8k_sha256": P.sha256_file(P.GSM8K_ITEMS), "n_longctx": len(ids_lc), "n_gsm8k": len(ids_gsm),
                   "permutation_draws": draws, "permutation_seed": seed, "tost_delta_points": delta,
                   "evidence_sha256": evidence_before,
                   "harness_sha256": {os.path.basename(f): P.sha256_file(os.path.join(HERE, f))[:16]
                                      for f in ("p6common.py", "analyze.py")}},
        "cells": per_cell,
        "co_primary": co_primary,
        "estimands": est,
        "secondary": {"longctx": secondary(cells, "longctx", ids_lc), "gsm8k": secondary(cells, "gsm8k", ids_gsm),
                      "trend_longctx": trend(cells, "longctx", ids_lc),
                      "trend_gsm8k": trend(cells, "gsm8k", ids_gsm),
                      "by_family_longctx": by_family(cells, items_lc),
                      "depth_bins_longctx": depth_bins(cells, items_lc)},
        "equal_memory": equal_memory(cells, ids_lc),
        "tokens": token_report(cells, items_lc, journal_dir),
        "flags": flags,
    }
    if evidence_snapshot(journal_dir) != evidence_before:
        raise ValueError("P6 journals/token counts changed during analysis; wait for a stable checkpoint")
    return out


def _pct(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def _pts(d, key="mean_pts"):
    if not d or d.get(key) is None:
        return "-"
    return f"{d[key]:+.2f} [{d['lo_pts']:+.2f}, {d['hi_pts']:+.2f}]"


def _p(x):
    if x is None:
        return "-"
    return f"{x:.3g}"


def equivalence_conclusion(co_primary: dict) -> str:
    """The primary claim uses the adjusted family decision, not raw TOST/CI90."""
    p = co_primary.get("p_holm")
    if p is None:
        return "not evaluable"
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("equivalence Holm p must be a finite probability")
    rejected = p < 0.05  # match the pre-existing analysis decision convention
    if co_primary.get("reject_at_0.05") is not rejected:
        raise ValueError("equivalence decision disagrees with Holm-adjusted p")
    return "equivalent within the specified margin (Holm)" if rejected else "equivalence not established (Holm)"


def markdown(a: dict) -> str:
    L = [f"# Paper 6 analysis ({a['ts']})", "",
         f"Journals: `{a['inputs']['journal_dir']}`; items `{a['inputs']['items']}` "
         f"(sha256 {a['inputs']['items_sha256'][:16]}); permutation {a['inputs']['permutation_draws']} draws, "
         f"seed {a['inputs']['permutation_seed']}; TOST margin +-{a['inputs']['tost_delta_points']} points.", ""]
    if a["flags"]:
        L += ["## Flags", ""] + [f"- {f}" for f in a["flags"]] + [""]
    else:
        L += ["## Flags", "", "- none", ""]

    L += ["## Co-primary outcomes (long context, Holm across the two)", "",
          "| outcome | estimate (points) | p | Holm p | decision |", "|---|---|---|---|---|"]
    c1 = a["co_primary"]["1_moderation"]
    L.append(f"| M = I_hybrid - I_GQA != 0 (sign-flip permutation) | {_pts(c1['estimate'])} | "
             f"{_p(c1['p_permutation'])} | {_p(c1['p_holm'])} | "
             f"{'reject H0' if c1['reject_at_0.05'] else 'no rejection'} |")
    t = a["co_primary"]["2_equivalence"]["tost"]
    if t.get("n"):
        L.append(f"| I_hybrid equivalent to 0 within +-{t['delta_pts']} (TOST) | {t['mean_pts']:+.2f} "
                 f"(90% CI {t['ci90_lo_pts']:+.2f}, {t['ci90_hi_pts']:+.2f}) | {_p(t['p_tost'])} | "
                 f"{_p(a['co_primary']['2_equivalence']['p_holm'])} | "
                 f"{equivalence_conclusion(a['co_primary']['2_equivalence'])} |")
    L += ["", f"Pre-registered predictions: {a['co_primary']['predictions']}.", ""]

    for task in ("longctx", "gsm8k"):
        e = a["estimands"][task]
        L += [f"## Estimands — {task} ({'primary' if task == 'longctx' else 'secondary'})", "",
              "| model | D(Q2) | D(IQ3) | D(Q4) | I = D(Q2)-D(Q4) | TOST p | perm p (I) |", "|---|---|---|---|---|---|---|"]
        for m in P.MODELS:
            i = e["I"][m]
            L.append(f"| {m} ({P.MODELS[m]['arch']}) | " +
                     " | ".join(_pts(e["D"][m][t]) for t in P.TIERS) +
                     f" | {_pts(i)} | {_p((i.get('tost') or {}).get('p_tost'))} | "
                     f"{_p((i.get('permutation') or {}).get('p'))} |")
        M = e["M"]
        L += ["", f"M = I_hybrid - I_GQA: {_pts(M)} points, n = {M['n']}, "
              f"sign-flip p = {_p((M.get('permutation') or {}).get('p'))}", ""]

    L += ["## Per-cell accuracy", "", "| cell | long-context | 95% Wilson | GSM8K | 95% Wilson | layers | resident | "
          "missing (lc/gsm) | trunc (lc/gsm) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, c in a["cells"].items():
        lc, gs = c["longctx"], c["gsm8k"]
        L.append(f"| {name} | {_pct(lc['p'])} ({lc['k']}/{lc['n']}) | "
                 f"{_pct(lc['lo'])}-{_pct(lc['hi'])} | {_pct(gs['p'])} ({gs['k']}/{gs['n']}) | "
                 f"{_pct(gs['lo'])}-{_pct(gs['hi'])} | {c.get('fit_layers')} | {c.get('fully_resident')} | "
                 f"{lc['n_missing']}/{gs['n_missing']} | {lc['truncated']}/{gs['truncated']} |")
    L.append("")

    for task in ("longctx", "gsm8k"):
        L += [f"## Secondary — each KV type vs f16 ({task})", "",
              "| model | tier | kv | n | acc f16 | acc kv | diff (points) | lost | gained | exact McNemar p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in a["secondary"][task]["kv_vs_f16"]:
            L.append(f"| {r['model']} | {r['tier']} | {r['kv']} | {r['n']} | {_pct(r['acc_f16'])} | "
                     f"{_pct(r['acc_kv'])} | {_pts(r['paired_diff'])} | {r['lost']} | {r['gained']} | "
                     f"{_p(r['mcnemar_p'])} |")
        L += ["", f"### K8/V4 vs q8_0 and q4_0 ({task})", "",
              "| model | tier | vs | n | diff (points) | k8v4 better | k8v4 worse | exact McNemar p |",
              "|---|---|---|---|---|---|---|---|"]
        for r in a["secondary"][task]["k8v4"]:
            L.append(f"| {r['model']} | {r['tier']} | {r['vs']} | {r['n']} | {_pts(r['paired_diff'])} | "
                     f"{r['k8v4_better']} | {r['k8v4_worse']} | {_p(r['mcnemar_p'])} |")
        L.append("")

    L += ["## Secondary — trend of D over weight tiers", "", "| model | contrast | points |", "|---|---|---|"]
    for r in a["secondary"]["trend_longctx"]:
        L.append(f"| {r['model']} | {r['contrast']} (long context) | {_pts(r)} |")
    L.append("")

    L += ["## Secondary — long-context accuracy by family", "",
          "| cell | " + " | ".join(("single", "multikey", "multivalue", "vartrack")) + " |",
          "|---|---|---|---|---|"]
    for name, d in a["secondary"]["by_family_longctx"].items():
        L.append(f"| {name} | " + " | ".join(_pct(d.get(f, {}).get("p")) for f in
                                             ("single", "multikey", "multivalue", "vartrack")) + " |")
    L.append("")

    L += ["## Secondary — long-context accuracy by needle depth (answer recall per bin)", "",
          "| cell | " + " | ".join(f"{i / DEPTH_BINS:.1f}-{(i + 1) / DEPTH_BINS:.1f}" for i in range(DEPTH_BINS)) + " |",
          "|---|" + "---|" * DEPTH_BINS]
    for name, d in a["secondary"]["depth_bins_longctx"].items():
        L.append(f"| {name} | " + " | ".join(_pct(b["recall"]) + f" ({b['n']})" for b in d["answer_recall"]) + " |")
    L.append("")

    L += ["## Equal-memory ranking (fully resident cells only; measured buffers)", ""]
    for model, d in a["equal_memory"].items():
        L += [f"### {model}", "",
              "| cell | layers | model MiB | KV MiB | KV@16k MiB | KV@32k MiB (computed) | recurrent MiB | "
              "compute MiB | device total MiB | nvidia-smi MiB | accuracy | 95% Wilson |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in d["resident"]:
            dm = r["device_mib"] or {}
            L.append(f"| {r['cell']} | {r['layers']} | {dm.get('model')} | {dm.get('kv')} | "
                     f"{r['kv_mib_at_16k'] and round(r['kv_mib_at_16k'], 1)} | "
                     f"{r['kv_mib_at_32k_computed'] and round(r['kv_mib_at_32k_computed'], 1)} | "
                     f"{r['recurrent_mib']} | {dm.get('compute')} | {dm.get('total_model_kv_rs_compute')} | "
                     f"{r['nvidia_process_mib']} | {_pct(r['accuracy']['p'])} | "
                     f"{_pct(r['accuracy']['lo'])}-{_pct(r['accuracy']['hi'])} |")
        if d["excluded_partly_offloaded"]:
            L += ["", "Excluded (partly offloaded): " +
                  ", ".join(f"{r['cell']} ({r['layers']})" for r in d["excluded_partly_offloaded"]), ""]
        if d["exchange_rate"]:
            L += ["", "Exchange rate (KV from the resident cells' measured buffers; weight steps from "
                  "device+host model buffers, which do not depend on placement): " + "; ".join(
                f"{r.get('tier') or r.get('weight_step')}: " +
                (f"KV f16->q4_0 at 16k frees {r['kv_freed_f16_to_q4_at16k_mib']:.0f} MiB"
                 if "kv_freed_f16_to_q4_at16k_mib" in r else f"weights +{r['model_bytes_step_mib']:.0f} MiB")
                for r in d["exchange_rate"]), ""]
        L.append("")

    tk = a["tokens"]
    L += ["## Token counts", "",
          f"- Qwen3.8 tokenizer (frozen items): haystack {tk['qwen38_tokenizer']['haystack_min']}-"
          f"{tk['qwen38_tokenizer']['haystack_max']} (mean {tk['qwen38_tokenizer']['haystack_mean']}), "
          f"full message mean {tk['qwen38_tokenizer']['content_mean']}"]
    for m, d in tk["served_tokenizer"].items():
        L.append(f"- served tokenizer ({m}): haystack {d['haystack_min']}-{d['haystack_max']} "
                 f"(mean {d['haystack_mean']}, n={d['n']})")
    for m, d in tk["prompt_tokens_observed"].items():
        L.append(f"- server prompt_tokens ({m}, long context): median {d['median']}, range {d['min']}-{d['max']}")
    for m, d in tk["prompt_token_deltas_hybrid"].items():
        L.append(f"- harness-vs-server prompt token delta ({m}): {d['nonzero']}/{d['n']} nonzero, "
                 f"range {d['min']}..{d['max']}")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- selftest
def _synthetic(tmp: str, items_lc, items_gsm) -> str:
    """Journals with a known answer. Correct cells answer with every required string.

    long context (n=200):  hybrid Q2/f16 wrong on items 0-9, Q2/q4_0 wrong on 10-19
                           GQA   Q2/q4_0 wrong on 0-39, Q4/q4_0 wrong on 0-9
      -> D_h(Q2) = 0, D_h(Q4) = 0, I_hybrid = 0 (per item -1 x10, +1 x10)
      -> D_g(Q2) = 20 pts, D_g(Q4) = 5 pts, I_GQA = 15 pts
      -> M = -15 points exactly; per item -1 on 30 items, 0 elsewhere
    GSM8K (n=100): the same shape halved -> D_g(Q2) = 20, D_g(Q4) = 5, M = -15 points.
    Hybrid Q4 cells are recorded as partly offloaded (33/66) so they leave the
    equal-memory table; one IQ3/k8v4 item is dropped to exercise complete-case handling."""
    jd = os.path.join(tmp, "journals")
    os.makedirs(jd, exist_ok=True)
    lc_wrong = {("qwen38-27b", "Q2_K_XL", "f16"): set(range(0, 10)),
                ("qwen38-27b", "Q2_K_XL", "q4_0"): set(range(10, 20)),
                ("qwen3-8b", "Q2_K_XL", "q4_0"): set(range(0, 40)),
                ("qwen3-8b", "Q4_K_XL", "q4_0"): set(range(0, 10))}
    gs_wrong = {("qwen38-27b", "Q2_K_XL", "f16"): set(range(0, 5)),
                ("qwen38-27b", "Q2_K_XL", "q4_0"): set(range(5, 10)),
                ("qwen3-8b", "Q2_K_XL", "q4_0"): set(range(0, 20)),
                ("qwen3-8b", "Q4_K_XL", "q4_0"): set(range(0, 5))}
    kv_bytes = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625, "k8v4": 0.8125}   # per f16 byte
    for cell in P.all_cells():
        model, tier, kv = cell
        hybrid = model == P.HYBRID
        base_kv = 1280.0 if hybrid else 2880.0          # f16 MiB at 20,480 cells
        kvm = round(base_kv * kv_bytes[kv] / 2.0, 2)
        partial = hybrid and tier == "Q4_K_XL"
        layers = ("33/66" if partial else ("65/65" if hybrid else "37/37"))
        log_text = "\n".join([
            "I cmn common_param: common_params_print_info: build 1 (035e227) with GNU 13.3.0 for Linux x86_64",
            "I common_fit_params: successfully fit params to free device memory",
            f"I load_tensors: offloaded {layers} layers to GPU",
            f"I load_tensors:   CPU_Mapped model buffer size =  {'7898.27' if partial else '397.85'} MiB",
            f"I load_tensors:        CUDA0 model buffer size =  {8500.0 if hybrid else 4000.0} MiB",
            "I llama_context: n_ctx                 = 20480",
            "I llama_context: flash_attn            = enabled",
            f"I llama_kv_cache:      CUDA0 KV buffer size =  {kvm} MiB",
            f"I llama_kv_cache: size = {kvm} MiB ( 20480 cells,  {16 if hybrid else 36} layers,  1/1 seqs), "
            f"K ({P.KV_TYPES[kv][0]}):  {kvm / 2} MiB, V ({P.KV_TYPES[kv][1]}):  {kvm / 2} MiB",
        ] + ([f"I llama_kv_cache:        CPU KV buffer size =  {kvm} MiB"] if partial else []) +
            ([f"I llama_memory_recurrent:      CUDA0 RS buffer size =   149.62 MiB",
              "I llama_memory_recurrent: size =  149.62 MiB (     1 cells,  64 layers,  1 seqs  0 rs_seq), "
              "R (f32):    5.62 MiB, S (f32):  144.00 MiB"] if hybrid else []) +
            ["I sched_reserve:      CUDA0 compute buffer size =   200.00 MiB"])
        import run_cells as R
        load = R.parse_server_log(log_text)
        rows = [{"kind": "server_start", "cell": {"model": model, "tier": tier, "kv": kv},
                 "gguf": {"path": P.gguf_path(model, tier), "sha256": "ab" * 32, "verified": True},
                 "engine": {"commit": P.ENGINE_COMMIT}, "load": load, "start_seq": 1,
                 "gpu_loaded": {"process_mib": int((load["device_mib"]["total_model_kv_rs_compute"]) + 400)},
                 "props": {"build_info": "b1-035e227"}}]
        for n, it in enumerate(items_lc):
            if cell == ("qwen38-27b", "IQ3_XXS", "k8v4") and n == len(items_lc) - 1:
                continue                                   # one missing item on a secondary cell
            ok = n not in lc_wrong.get(cell, set())
            content = (" and ".join(it["answers"]) if ok else "I could not find it in the text.")
            rows.append({"job_id": P.job_id("longctx", it["id"]), "task": "longctx", "item_id": it["id"],
                         "terminal": True, "ok": True, "content": content, "reasoning_content": None,
                         "finish_reason": "stop", "usage": {"prompt_tokens": 16400, "completion_tokens": 12},
                         "seconds": 1.0, "score": {"correct": ok}, "meta": {}})
        for n, it in enumerate(items_gsm):
            ok = n not in gs_wrong.get(cell, set())
            content = f"Answer: {int(it['answer_number'])}" if ok else "Answer: -1"
            rows.append({"job_id": P.job_id("gsm8k", it["id"]), "task": "gsm8k", "item_id": it["id"],
                         "terminal": True, "ok": True, "content": content, "reasoning_content": "thinking",
                         "finish_reason": "stop", "usage": {"prompt_tokens": 90, "completion_tokens": 300},
                         "seconds": 30.0, "score": {"correct": ok}, "meta": {}})
        with open(P.journal_path(*cell, journals=jd), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return jd


def selftest() -> int:
    fails = []

    def check(name, got, want, tol=None):
        ok = (got == want) if tol is None else (got is not None and abs(got - want) <= tol)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got={got!r} want={want!r}"), flush=True)
        if not ok:
            fails.append(name)

    print("selftest: statistics against hand-computed values")
    w = wilson(5, 10)
    check("Wilson 5/10 = 0.2366..0.7634", (round(w["lo"], 4), round(w["hi"], 4)), (0.2366, 0.7634))
    w = wilson(0, 10)
    check("Wilson 0/10 = 0..0.2775", (round(w["lo"], 4), round(w["hi"], 4)), (0.0, 0.2775))
    w = wilson(10, 10)
    check("Wilson 10/10 = 0.7225..1", (round(w["lo"], 4), round(w["hi"], 4)), (0.7225, 1.0))
    check("McNemar b=0,c=5 -> 2*(1/32)", mcnemar_exact(0, 5), 0.0625)
    check("McNemar b=1,c=9 -> 2*11/1024", mcnemar_exact(1, 9), 22 / 1024)
    check("McNemar b=2,c=8 -> 2*56/1024", mcnemar_exact(2, 8), 112 / 1024)
    check("McNemar b=c=3 -> 1", mcnemar_exact(3, 3), 1.0)
    check("McNemar no discordant pairs -> 1", mcnemar_exact(0, 0), 1.0)
    check("t quantile df=1 (tan closed form)", t_ppf(0.975, 1), math.tan(math.pi * (0.975 - 0.5)), 1e-7)
    check("t quantile df=2 (closed form)", t_ppf(0.975, 2), (2 * 0.975 - 1) / math.sqrt(2 * 0.975 * 0.025), 1e-9)
    check("t quantile df=10 = 2.228138852", t_ppf(0.975, 10), 2.228138852, 1e-8)
    check("t quantile df=199 = 1.971956544", t_ppf(0.975, 199), 1.971956544, 1e-7)
    check("t quantile df=1e7 ~ normal", t_ppf(0.975, 1e7), Z975, 1e-5)
    # t CDF against numerical integration of the pdf (independent method)
    for tv, df in ((1.3, 7), (-2.1, 15), (0.4, 199)):
        c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        n, lo = 200000, -60.0
        h = (tv - lo) / n
        s = sum((4 if i % 2 else 2) * c * (1 + ((lo + i * h) ** 2) / df) ** (-(df + 1) / 2) for i in range(1, n))
        s += c * (1 + lo ** 2 / df) ** (-(df + 1) / 2) + c * (1 + tv ** 2 / df) ** (-(df + 1) / 2)
        check(f"t CDF({tv}, df={df}) vs Simpson integration", t_cdf(tv, df), s * h / 3, 1e-9)
    check("Holm (0.01, 0.04)", holm({"a": 0.01, "b": 0.04}), {"a": 0.02, "b": 0.04})
    check("Holm (0.03, 0.04) keeps monotonicity", holm({"a": 0.03, "b": 0.04}), {"a": 0.06, "b": 0.06})
    # exact enumeration for a small sign-flip test: x = [1,1,1] -> p = 2/8
    check("sign-flip p ~ exact 0.25 for [1,1,1]", signflip_p([1, 1, 1], draws=100000, seed=7)["p"], 0.25, 0.01)
    check("sign-flip p = 1 when every contrast is 0", signflip_p([0] * 10)["p"], 1.0)
    check("TOST of zeros with sd=0 is equivalent", tost([0.0] * 10)["equivalent"], True)

    print("selftest: full pipeline on synthetic journals")
    items_lc = P.load_longctx()
    items_gsm = P.load_gsm8k()
    with tempfile.TemporaryDirectory(dir=P.WORK) as tmp:
        jd = _synthetic(tmp, items_lc, items_gsm)
        a = analyse(jd, P.LONGCTX_ITEMS, PERM_DRAWS, PERM_SEED, DELTA_POINTS)
        with open(os.path.join(tmp, "analysis.json"), "w") as f:
            json.dump(a, f, indent=1)
        md = markdown(a)
        with open(os.path.join(tmp, "summary.md"), "w") as f:
            f.write(md)
        e = a["estimands"]["longctx"]
        check("D_hybrid(Q2) = 0", round(e["D"][P.HYBRID]["Q2_K_XL"]["mean_pts"], 9), 0.0)
        check("D_GQA(Q2) = 20 points", round(e["D"][P.GQA]["Q2_K_XL"]["mean_pts"], 9), 20.0)
        check("D_GQA(Q4) = 5 points", round(e["D"][P.GQA]["Q4_K_XL"]["mean_pts"], 9), 5.0)
        check("I_hybrid = 0", round(e["I"][P.HYBRID]["mean_pts"], 9), 0.0)
        check("I_GQA = 15 points", round(e["I"][P.GQA]["mean_pts"], 9), 15.0)
        check("M = -15 points", round(e["M"]["mean_pts"], 9), -15.0)
        check("M uses all 200 items", e["M"]["n"], 200)
        check("M permutation p = 1/(1+draws) (30 x -1, never exceeded)",
              e["M"]["permutation"]["p"], 1 / (1 + PERM_DRAWS))
        t = a["co_primary"]["2_equivalence"]["tost"]
        # I_hybrid: -1 on 10 items, +1 on 10, 0 on 180 -> sd = sqrt(20/199), se = sd/sqrt(200)
        se = math.sqrt(20 / 199) / math.sqrt(200)
        check("TOST se (points)", t["se_pts"], 100 * se, 1e-9)
        check("TOST 90% CI half width", t["ci90_hi_pts"], 100 * t_ppf(0.95, 199) * se, 1e-9)
        check("TOST p = P(T_199 >= 0.10/se)", t["p_tost"], t_sf(0.10 / se, 199), 1e-12)
        check("TOST equivalent", t["equivalent"], True)
        check("Holm adjusts the smaller p by 2",
              round(a["co_primary"]["holm"]["co_primary_2_I_hybrid_equivalence"], 12),
              round(min(1.0, 2 * t["p_tost"]), 12))
        mc = {(r["model"], r["tier"], r["kv"]): r for r in a["secondary"]["longctx"]["kv_vs_f16"]}
        r = mc[(P.GQA, "Q2_K_XL", "q4_0")]
        check("McNemar GQA Q2 q4_0 vs f16: 40 lost, 0 gained", (r["lost"], r["gained"]), (40, 0))
        check("its exact p = 2^-39", r["mcnemar_p"], 2.0 ** -39, 1e-18)
        check("complete cases on the cell with a dropped item", mc[(P.HYBRID, "IQ3_XXS", "k8v4")]["n"], 199)
        check("missing item flagged", any("IQ3_XXS__k8v4" in f and "missing" in f for f in a["flags"]), True)
        eq = a["equal_memory"][P.HYBRID]
        check("hybrid Q4 cells excluded from the equal-memory table",
              sorted(r["cell"].split("__")[1] for r in eq["excluded_partly_offloaded"]), ["Q4_K_XL"] * 4)
        check("8 hybrid cells fully resident", len(eq["resident"]), 8)
        top = eq["resident"][0]
        check("equal-memory rows carry measured buffers and KV at 16k",
              (top["device_mib"]["kv"] is not None, round(top["kv_mib_at_16k"] / top["kv_total_mib"], 4)), (True, 0.8))
        check("GSM8K M = -15 points", round(a["estimands"]["gsm8k"]["M"]["mean_pts"], 9), -15.0)
        check("gsm8k rescoring matches the stored score (no mismatch flags)",
              [f for f in a["flags"] if "re-score" in f], [])
        check("markdown mentions both co-primaries and the equal-memory table",
              all(s in md for s in ("Co-primary outcomes", "Equal-memory ranking", "by needle depth")), True)
        print(f"    (synthetic analysis.json {os.path.getsize(os.path.join(tmp, 'analysis.json'))} bytes, "
              f"summary.md {len(md)} chars)")
    if fails:
        print(f"\nSELFTEST FAILED: {fails}")
        return 1
    print("\nSELFTEST PASSED (all checks)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal-dir", default=P.JOURNALS)
    ap.add_argument("--items", default=P.LONGCTX_ITEMS)
    ap.add_argument("--out-dir", default=os.path.join(P.WORK, "analysis"))
    ap.add_argument("--draws", type=int, default=PERM_DRAWS)
    ap.add_argument("--perm-seed", type=int, default=PERM_SEED)
    ap.add_argument("--delta-points", type=float, default=DELTA_POINTS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    res = analyse(a.journal_dir, a.items, a.draws, a.perm_seed, a.delta_points)
    payload = json.dumps(res, indent=1, allow_nan=False)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "analysis.json"), "w") as f:
        f.write(payload)
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write(markdown(res))
    print(markdown(res))
    print(f"\nwrote {a.out_dir}/analysis.json and {a.out_dir}/summary.md")


if __name__ == "__main__":
    main()
