#!/usr/bin/env python3
"""Recompute every Paper 2 (P2R1) number from the public aggregate claims file.

Standard library only; no model, network, GPU, or private data. The claims file
holds aggregate counts only: per-cell correct counts, paired discordance counts,
and, for each of the 12 interaction contrasts, a histogram of source-conversation
clusters by (items in cluster, sum of item contrasts). That is enough to
recompute, exactly and without any item identifiers:

* every Wilson accuracy interval, Newcombe method-10 paired interval, and exact
  McNemar test;
* every interaction estimate, CR2 (Bell--McCaffrey) cluster-robust variance,
  Satterthwaite reference degrees of freedom, t statistic, two-sided p-value,
  pointwise 95% interval, and the secondary exhaustive sign-flip p-value;
* the Holm step-down family over all 12 contrasts and the study-level result
  branch; and
* every headline range and every numeric macro printed in the paper
  (`paper/derived/claims.tex` is re-emitted and compared macro by macro).

It also checks the within-deployment orderings the paper states in words.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT / "generated" / "data"
PAPER_TEX = ROOT / "paper" / "derived" / "claims.tex"
EXPECTED_HASHES = {
    "claims.json": "5819a341c047474fa919b3b189cfb8c860d3873e335890f85c59ce2b82c2a7bd",
}
EXPECTED_CLAIMS_TEX_SHA256 = "786a4b78a352588f1b19efe02c4f7f2fca4d3dd1d49f644a96258b5459c3e6e8"
EXPECTED_PROMOTION_SHA256 = "454b9bd0a19003957efbda9b3d6a47918142fd98103cdcd7580bc1e4cf57b711"
Z_95 = statistics.NormalDist().inv_cdf(0.975)
CORPORA = ("Tsi", "Wc")
TIERS = ("Qf", "Ex", "Iq")
CONDITIONS = ("O", "L", "A", "E")
TOLERANCE = 1e-9


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, what: str, tol: float = TOLERANCE) -> None:
    require(
        abs(float(actual) - float(expected)) <= tol,
        f"{what}: recomputed {actual!r} != recorded {expected!r}",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- intervals --
def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    require(n > 0 and 0 <= k <= n, f"invalid binomial count {k}/{n}")
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_paired(both: int, low_only: int, high_only: int, neither: int,
                    z: float = Z_95) -> tuple[float, float, float, float]:
    """Newcombe method-10 interval for p(high) - p(low) on paired binary data."""
    n = both + low_only + high_only + neither
    high, low = both + high_only, both + low_only
    p_high, p_low = high / n, low / n
    difference = p_high - p_low
    high_lo, high_hi = wilson_interval(high, n, z)
    low_lo, low_hi = wilson_interval(low, n, z)
    denominator = math.sqrt(p_high * (1 - p_high) * p_low * (1 - p_low))
    phi = ((both / n) - p_high * p_low) / denominator if denominator else 0.0
    lower = ((p_high - high_lo) ** 2 + (low_hi - p_low) ** 2
             - 2.0 * phi * (p_high - high_lo) * (low_hi - p_low))
    upper = ((high_hi - p_high) ** 2 + (p_low - low_lo) ** 2
             - 2.0 * phi * (high_hi - p_high) * (p_low - low_lo))
    return (difference, difference - math.sqrt(max(0.0, lower)),
            difference + math.sqrt(max(0.0, upper)), phi)


def mcnemar_exact(low_only: int, high_only: int) -> float:
    discordant = low_only + high_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(low_only, high_only) + 1))
    return float(min(Fraction(1, 1), Fraction(2 * tail, 2 ** discordant)))


# ---------------------------------------------------------- Student t / CR2 --
def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise AssertionError("incomplete-beta continued fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if x in (0.0, 1.0):
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided(t_value: float, df: float) -> float:
    x = df / (df + t_value * t_value)
    return min(1.0, max(0.0, regularized_beta(x, df / 2.0, 0.5)))


def t_critical_975(df: float) -> float:
    lo, hi = 0.0, 2.0
    while student_t_two_sided(hi, df) > 0.05:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if student_t_two_sided(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cr2_satterthwaite_df(sizes: list[int]) -> float:
    n = sum(sizes)
    leverages = [size / n for size in sizes]
    trace = sum(h / n for h in leverages)
    squared = 0.0
    for g, hg in enumerate(leverages):
        for h, hh in enumerate(leverages):
            element = hg / n if g == h else -hg * hh / (n * math.sqrt((1 - hg) * (1 - hh)))
            squared += element * element
    return trace * trace / squared


def sign_flip_p(cluster_sums: list[int]) -> float:
    values = [abs(v) for v in cluster_sums if v]
    if not values:
        return 1.0
    observed = abs(sum(cluster_sums))
    distribution = {0: 1}
    for value in values:
        updated: dict[int, int] = {}
        for total, count in distribution.items():
            updated[total + value] = updated.get(total + value, 0) + count
            updated[total - value] = updated.get(total - value, 0) + count
        distribution = updated
    extreme = sum(c for total, c in distribution.items() if abs(total) >= observed)
    return extreme / (2 ** len(values))


def interaction_from_histogram(entry: dict) -> dict:
    """CR2 intercept-only inference from the aggregate cluster histogram."""
    sizes: list[int] = []
    sums: list[int] = []
    for record in entry["cluster_n_items_raw_z_sum_histogram"]:
        require(set(record) == {"cluster_n_items", "raw_z_sum", "multiplicity"},
                "histogram record fields changed")
        sizes += [record["cluster_n_items"]] * record["multiplicity"]
        sums += [record["raw_z_sum"]] * record["multiplicity"]
    n = sum(sizes)
    estimate = sum(sums) / n
    variance = sum((s - size * estimate) ** 2 / (1.0 - size / n)
                   for size, s in zip(sizes, sums)) / n ** 2
    se = math.sqrt(variance)
    df = cr2_satterthwaite_df(sizes)
    t_value = estimate / se if se else 0.0
    tcrit = t_critical_975(df)
    return {
        "n_items": n, "n_source_clusters": len(sizes), "estimate": estimate,
        "variance": variance, "se": se, "df": df, "t": t_value,
        "p": student_t_two_sided(t_value, df), "tcrit": tcrit,
        "ci": [estimate - tcrit * se, estimate + tcrit * se],
        "p_sign": sign_flip_p(sums),
        "nonzero_clusters": sum(v != 0 for v in sums),
        "size_counts": {str(k): v for k, v in sorted(Counter(sizes).items())},
    }


def holm_family(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    running, still = 0.0, True
    result: dict[str, dict] = {}
    for index, (key, p) in enumerate(ordered):
        remaining = m - index
        running = max(running, min(1.0, remaining * p))
        threshold = alpha / remaining
        reject = still and p <= threshold
        still = still and reject
        result[key] = {"p_raw": p, "p_holm": running,
                       "holm_threshold": threshold, "reject_fwer_005": reject}
    return result


# ----------------------------------------------------------- TeX re-emission --
def fmt_p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def emit_tex(claims: dict) -> str:
    """Byte-identical copy of the promoted builder's macro emitter (paper2/scripts/build_claims.py)."""
    lines = [
        "% Generated by paper2/scripts/build_claims.py. DO NOT EDIT.",
        "% Every macro is derived from the frozen grid JSONLs, the window",
        "% files' character counts, GGUF headers, or run-log tallies.",
    ]
    seen: set[str] = set()

    def cmd(name: str, value) -> None:
        require(name not in seen, f"duplicate macro {name}")
        seen.add(name)
        lines.append(rf"\newcommand{{\Ptwo{name}}}{{{value}}}")

    a = claims["artifacts"]
    cmd("QfGiB", f"{a['Qf']['file_bytes'] / 2**30:.1f}")
    cmd("QfBpw", f"{a['Qf']['effective_file_bpw']:.2f}")
    cmd("IqGiB", f"{a['Iq']['file_bytes'] / 2**30:.1f}")
    cmd("IqBpw", f"{a['Iq']['effective_file_bpw']:.2f}")
    cmd("ExGiB", f"{a['Ex']['weight_safetensors_bytes'] / 2**30:.1f}")
    cmd("ParamsB", f"{a['Qf']['stored_parameters_from_gguf_tensor_shapes'] / 1e9:.1f}")
    for ctag, c in claims["corpora"].items():
        cmd(f"{ctag}Windows", c["windows"])
        cmd(f"{ctag}QKept", c["questions_kept"])
        cmd(f"{ctag}RejPct",
            f"{c['candidate_pair_rejection_rate_after_successful_requests'] * 100:.0f}")
        for dtag, r in c["char_retention_vs_O"].items():
            cmd(f"Ret{dtag}{ctag}", f"{r * 100:.0f}")
    for ctag, g in claims["grid"].items():
        cmd(f"{ctag}Items", g["n_items"])
        cmd(f"{ctag}SourceWindows", g["n_source_windows"])
        cmd(f"{ctag}SourceClusters", g["n_source_clusters"])
        for stratum in ("S-nocode", "S-code", "L-nocode", "L-code"):
            suffix = stratum.replace("-", "").replace("nocode", "NoCode").replace("code", "Code")
            cmd(f"{ctag}{suffix}Items", g["item_strata_counts"].get(stratum, 0))
        for cell_key, cell in g["cells"].items():
            t, d = cell_key.split("-")
            base = f"{t}{ctag}{d}"
            cmd(f"Acc{base}", f"{cell['acc'] * 100:.0f}")
            cmd(f"AccLen{base}", f"{cell['acc_lenient'] * 100:.0f}")
            cmd(f"Lo{base}", f"{cell['wilson'][0] * 100:.0f}")
            cmd(f"Hi{base}", f"{cell['wilson'][1] * 100:.0f}")
            cmd(f"Wall{base}", f"{cell['mean_wall_s']:.1f}")
            cmd(f"Apm{base}", f"{cell['acc_per_min']:.1f}")
            if "paired_vs_O" in cell:
                pv = cell["paired_vs_O"]
                cmd(f"Del{base}", f"{pv['delta'] * 100:.0f}")
                cmd(f"Dlo{base}", f"{pv['ci'][0] * 100:.0f}")
                cmd(f"Dhi{base}", f"{pv['ci'][1] * 100:.0f}")
                cmd(f"P{base}", fmt_p(pv["p_mcnemar"]))
    for ctag, inter in claims["interaction"].items():
        for key, v in inter.items():
            d, t = key.split("-")
            cmd(f"Int{ctag}{d}{t}", f"{v['delta_low_minus_q4_reference'] * 100:.1f}")
            cmd(f"IntLo{ctag}{d}{t}", f"{v['ci_cluster_95'][0] * 100:.1f}")
            cmd(f"IntHi{ctag}{d}{t}", f"{v['ci_cluster_95'][1] * 100:.1f}")
            cmd(f"IntP{ctag}{d}{t}", f"{v['p_cluster_cr2_satterthwaite']:.3f}")
            cmd(f"IntPSignSym{ctag}{d}{t}", f"{v['p_sign_symmetry_sensitivity']:.3f}")
            cmd(f"IntPHolm{ctag}{d}{t}", f"{v['familywise']['p_holm']:.3f}")
            cmd(f"IntHolmReject{ctag}{d}{t}", int(v["familywise"]["reject_fwer_005"]))
    for ctag, sc in claims["survival"].items():
        for dtag, entry in sc.items():
            cmd(f"Surv{ctag}{dtag}N", entry["gold_survives"])
            cmd(f"Surv{ctag}{dtag}Tot", entry["n"])
            cmd(f"Surv{ctag}{dtag}Pct", f"{entry['gold_survives'] / entry['n'] * 100:.0f}")
            for t, tv in entry["tiers"].items():
                cmd(f"Surv{ctag}{dtag}{t}K", tv["surv_correct"])
                cmd(f"Dest{ctag}{dtag}{t}K", tv["dest_correct"])
            cmd(f"Dest{ctag}{dtag}N", entry["n"] - entry["gold_survives"])
    h = claims["headline"]
    cmd("NaiveDelMin", f"{h['naive_delta_pp_min']:.0f}")
    cmd("NaiveDelMax", f"{h['naive_delta_pp_max']:.0f}")
    cmd("LDelMin", f"{h['l2_delta_pp_min']:.0f}")
    cmd("LDelMax", f"{h['l2_delta_pp_max']:.0f}")
    cmd("InterPMin", f"{h['interaction_p_min']:.2f}")
    cmd("InterPMax", f"{h['interaction_p_max']:.2f}")
    cmd("InterDeltaMin", f"{h['interaction_delta_pp_range'][0]:.1f}")
    cmd("InterDeltaMax", f"{h['interaction_delta_pp_range'][1]:.1f}")
    cmd("InterCiLowMin", f"{h['interaction_ci_pp_envelope'][0]:.1f}")
    cmd("InterCiHighMax", f"{h['interaction_ci_pp_envelope'][1]:.1f}")
    cmd("InterResolution", f"{h['interaction_max_abs_ci_endpoint_pp']:.1f}")
    cmd("InterHolmDetected", h["interaction_holm_rejections_fwer_005"])
    cmd("InterPointwiseCiDetected", h["interaction_pointwise_ci_excluding_zero"])
    cmd("InterRawSignflipDetected", h["interaction_unadjusted_signflip_p_below_005"])
    cmd("InterRawClusterTDetected", h["interaction_unadjusted_cluster_t_p_below_005"])
    cmd("InterTests", h["n_interaction_tests"])
    cmd("NCells", h["n_cells"])
    cmd("NRequests", h["n_scored_requests"])
    cmd("EvalRequestErrors", h["evaluation_request_errors"])
    cmd("GenRequestErrors", h["question_generation_request_errors"])
    cmd("OAccMin", f"{h['o_acc_range'][0] * 100:.0f}")
    cmd("OAccMax", f"{h['o_acc_range'][1] * 100:.0f}")
    cmd("LAccMin", f"{h['l_acc_range'][0] * 100:.0f}")
    cmd("LAccMax", f"{h['l_acc_range'][1] * 100:.0f}")
    cmd("ApmOMin", f"{h['apm_o_range'][0]:.1f}")
    cmd("ApmOMax", f"{h['apm_o_range'][1]:.1f}")
    cmd("ApmLMin", f"{h['apm_l_range'][0]:.1f}")
    cmd("ApmLMax", f"{h['apm_l_range'][1]:.1f}")
    cmd("ApmNaiveMin", f"{h['apm_naive_range'][0]:.1f}")
    cmd("ApmNaiveMax", f"{h['apm_naive_range'][1]:.1f}")
    cmd("WallOMin", f"{h['wall_o_range'][0]:.1f}")
    cmd("WallOMax", f"{h['wall_o_range'][1]:.1f}")
    cmd("WallNaiveMin", f"{h['wall_naive_range'][0]:.1f}")
    cmd("WallNaiveMax", f"{h['wall_naive_range'][1]:.1f}")
    cmd("TierSpreadMax", f"{h['tier_delta_spread_pp_max']:.0f}")
    for tag, values in (("SurvAccL", h["surv_acc_l_range"]), ("SurvAccE", h["surv_acc_e_range"])):
        cmd(f"{tag}Min", "N/A" if values is None else f"{values[0] * 100:.0f}")
        cmd(f"{tag}Max", "N/A" if values is None else f"{values[1] * 100:.0f}")
    cmd("DestTrials", h["destroyed_trials"])
    cmd("DestPairs", h["destroyed_item_condition_pairs"])
    cmd("DestRecovered", h["destroyed_recovered"])
    return "\n".join(lines) + "\n"


def macro_map(tex: str) -> dict[str, str]:
    """Parse `\\newcommand{\\PtwoX}{value}` lines into an exact name -> value map."""
    out: dict[str, str] = {}
    for line in tex.splitlines():
        if not line.startswith("\\newcommand{\\Ptwo"):
            continue
        name, _, rest = line[len("\\newcommand{\\"):].partition("}{")
        require(rest.endswith("}"), f"malformed macro line: {line}")
        require(name not in out, f"duplicate macro {name}")
        out[name] = rest[:-1]
    return out


TIER_NAMES = {"Qf": "Q4_K_XL", "Ex": "EXL3-2.0bpw", "Iq": "IQ2_S"}
CORPUS_NAMES = {"Tsi": "private", "Wc": "WildChat"}
COND_NAMES = {"O": "original", "L": "LLMLingua-2", "A": "stopword-rung", "E": "full-ladder"}


def flat_tables(claims: dict) -> tuple[str, str]:
    """The exporter's browsing tables; re-derived here and compared byte for byte."""
    cell_rows = ["corpus,deployment,condition,n,correct_exact,accuracy,wilson_low,wilson_high,"
                 "paired_cost_vs_original,paired_cost_low,paired_cost_high,mcnemar_p,"
                 "mean_request_s,correct_per_minute"]
    for c in sorted(claims["grid"]):
        for t in ("Qf", "Ex", "Iq"):
            for d in ("O", "L", "A", "E"):
                cell = claims["grid"][c]["cells"][f"{t}-{d}"]
                pv = cell.get("paired_vs_O")
                paired = (["", "", "", ""] if pv is None else
                          [f"{pv['delta']:.6f}", f"{pv['ci'][0]:.6f}", f"{pv['ci'][1]:.6f}",
                           f"{pv['p_mcnemar']:.6g}"])
                cell_rows.append(",".join([
                    CORPUS_NAMES[c], TIER_NAMES[t], COND_NAMES[d], str(cell["n"]),
                    str(cell["k_exact"]), f"{cell['acc']:.6f}", f"{cell['wilson'][0]:.6f}",
                    f"{cell['wilson'][1]:.6f}", *paired, f"{cell['mean_wall_s']:.3f}",
                    f"{cell['acc_per_min']:.6f}"]))
    inter_rows = ["corpus,condition,low_deployment,interaction_estimate,ci_low,ci_high,"
                  "cr2_se,satterthwaite_df,p_cr2,p_holm,holm_reject,p_sign_flip,source_clusters"]
    for c in sorted(claims["interaction"]):
        for key in sorted(claims["interaction"][c]):
            v = claims["interaction"][c][key]
            d, t = key.split("-")
            inter_rows.append(",".join([
                CORPUS_NAMES[c], COND_NAMES[d], TIER_NAMES[t],
                f"{v['delta_low_minus_q4_reference']:.6f}", f"{v['ci_cluster_95'][0]:.6f}",
                f"{v['ci_cluster_95'][1]:.6f}", f"{v['cluster_se_cr2']:.6f}",
                f"{v['cluster_reference_df_satterthwaite']:.3f}",
                f"{v['p_cluster_cr2_satterthwaite']:.6g}", f"{v['familywise']['p_holm']:.6g}",
                str(v["familywise"]["reject_fwer_005"]).lower(),
                f"{v['p_sign_symmetry_sensitivity']:.6g}", str(v["n_source_clusters"])]))
    return "\n".join(cell_rows) + "\n", "\n".join(inter_rows) + "\n"


# --------------------------------------------------------------------- main --
def main() -> int:
    claims_path = DATA / "claims.json"
    require(sha256(claims_path) == EXPECTED_HASHES["claims.json"],
            "changed or missing evidence: claims.json")
    export = json.loads((DATA / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    require(export.get("schema") == "paper2-p2r1-public-data-export-v1", "wrong export schema")
    require(export.get("no_inference") is True, "export incorrectly claims inference")
    require(export.get("files", {}).get("claims.json", {}).get("sha256")
            == EXPECTED_HASHES["claims.json"], "export hash record changed")
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    require(claims.get("attempt_id") == "p2r1-primary-v5", "not the official P2R1 attempt")
    require(claims.get("promotion_sha256") == EXPECTED_PROMOTION_SHA256, "promotion binding changed")
    require(set(claims["grid"]) == set(CORPORA), "corpora changed")

    checked = Counter()
    # --- cells: accuracy, Wilson, paired Newcombe interval, McNemar
    for ctag in CORPORA:
        g = claims["grid"][ctag]
        n_items = g["n_items"]
        require(sum(g["item_strata_counts"].values()) == n_items, f"{ctag} strata do not sum")
        for t in TIERS:
            o_cell = g["cells"][f"{t}-O"]
            for d in CONDITIONS:
                cell = g["cells"][f"{t}-{d}"]
                what = f"{ctag}/{t}/{d}"
                require(cell["n"] == n_items, f"{what}: n changed")
                close(cell["acc"], cell["k_exact"] / cell["n"], f"{what} accuracy")
                close(cell["acc_lenient"], cell["k_lenient"] / cell["n"], f"{what} lenient accuracy")
                lo, hi = wilson_interval(cell["k_exact"], cell["n"])
                close(cell["wilson"][0], lo, f"{what} Wilson low")
                close(cell["wilson"][1], hi, f"{what} Wilson high")
                close(cell["acc_per_min"], cell["acc"] / (cell["mean_wall_s"] / 60.0), f"{what} acc/min")
                outcomes = cell["request_outcomes"]
                require(outcomes["total"] == cell["n"]
                        and outcomes["success"] + outcomes["identity_preserved_request_errors"] == cell["n"],
                        f"{what}: request outcomes incomplete")
                checked["cells"] += 1
                if d == "O":
                    continue
                pv = cell["paired_vs_O"]
                b, c = pv["O_only"], pv["cond_only"]
                both = o_cell["k_exact"] - b
                require(both + c == cell["k_exact"], f"{what}: paired table inconsistent with counts")
                neither = cell["n"] - (both + b + c)
                require(neither >= 0, f"{what}: negative paired cell")
                diff, dlo, dhi, phi = newcombe_paired(both, c, b, neither)
                close(pv["delta"], diff, f"{what} paired delta")
                close(pv["ci"][0], dlo, f"{what} Newcombe low")
                close(pv["ci"][1], dhi, f"{what} Newcombe high")
                close(pv["phi"], phi, f"{what} phi")
                close(pv["p_mcnemar"], mcnemar_exact(b, c), f"{what} McNemar p")
                checked["paired"] += 1

    # --- interaction: CR2 / Satterthwaite / t / p / CI / sign flip, then Holm
    pvalues: dict[str, float] = {}
    for ctag in CORPORA:
        for key, entry in claims["interaction"][ctag].items():
            what = f"{ctag}/{key}"
            r = interaction_from_histogram(entry)
            require(r["n_items"] == entry["n_items"] == claims["grid"][ctag]["n_items"], f"{what}: n changed")
            require(r["n_source_clusters"] == entry["n_source_clusters"], f"{what}: cluster count changed")
            require(r["size_counts"] == entry["cluster_size_counts"], f"{what}: cluster sizes changed")
            require(r["nonzero_clusters"] == entry["n_nonzero_source_clusters"], f"{what}: nonzero clusters changed")
            values = entry["item_value_counts"]
            require(sum(values.values()) == entry["n_items"], f"{what}: item values do not sum")
            close(sum(int(v) * k for v, k in values.items()) / entry["n_items"], r["estimate"],
                  f"{what} estimate from item values")
            close(entry["delta_low_minus_q4_reference"], r["estimate"], f"{what} estimate")
            close(entry["variance_cr2"], r["variance"], f"{what} CR2 variance", 1e-12)
            close(entry["cluster_se_cr2"], r["se"], f"{what} CR2 SE", 1e-12)
            close(entry["cluster_reference_df_satterthwaite"], r["df"], f"{what} Satterthwaite df", 1e-7)
            require(r["df"] >= 5 - 1e-12, f"{what}: df below the predeclared minimum")
            close(entry["t_cluster_cr2"], r["t"], f"{what} t")
            close(entry["t_critical_975"], r["tcrit"], f"{what} t critical", 1e-7)
            close(entry["ci_cluster_95"][0], r["ci"][0], f"{what} CI low", 1e-7)
            close(entry["ci_cluster_95"][1], r["ci"][1], f"{what} CI high", 1e-7)
            close(entry["p_cluster_cr2_satterthwaite"], r["p"], f"{what} p", 1e-9)
            close(entry["p_sign_symmetry_sensitivity"], r["p_sign"], f"{what} sign-flip p")
            pvalues[f"{ctag}:{key}"] = r["p"]
            checked["interaction"] += 1
    require(len(pvalues) == 12, "the interaction family is not 12 tests")
    family = holm_family(pvalues)
    for hypothesis, fam in family.items():
        ctag, key = hypothesis.split(":", 1)
        recorded = claims["interaction"][ctag][key]["familywise"]
        require(recorded["reject_fwer_005"] == fam["reject_fwer_005"], f"{hypothesis}: Holm decision changed")
        close(recorded["p_holm"], fam["p_holm"], f"{hypothesis} Holm p", 1e-9)
        close(recorded["holm_threshold"], fam["holm_threshold"], f"{hypothesis} Holm threshold")
    rejections = sum(f["reject_fwer_005"] for f in family.values())
    branch = "interaction-detected" if rejections else "no-interaction-detected"
    h = claims["headline"]
    require(h["interaction_result_branch"] == branch, "result branch changed")
    require(h["interaction_holm_rejections_fwer_005"] == rejections, "Holm rejection count changed")

    # --- headline ranges
    def cells(dtags):
        return [claims["grid"][c]["cells"][f"{t}-{d}"] for c in CORPORA for t in TIERS for d in dtags]
    naive = [x["paired_vs_O"]["delta"] for x in cells(("A", "E"))]
    l2 = [x["paired_vs_O"]["delta"] for x in cells(("L",))]
    inter = [v for c in CORPORA for v in claims["interaction"][c].values()]
    close(h["naive_delta_pp_min"], min(naive) * 100, "naive delta min")
    close(h["naive_delta_pp_max"], max(naive) * 100, "naive delta max")
    close(h["l2_delta_pp_min"], min(l2) * 100, "L2 delta min")
    close(h["l2_delta_pp_max"], max(l2) * 100, "L2 delta max")
    close(h["interaction_p_min"], min(v["p_cluster_cr2_satterthwaite"] for v in inter), "interaction p min")
    close(h["interaction_p_max"], max(v["p_cluster_cr2_satterthwaite"] for v in inter), "interaction p max")
    deltas = [v["delta_low_minus_q4_reference"] * 100 for v in inter]
    close(h["interaction_delta_pp_range"][0], min(deltas), "interaction delta min")
    close(h["interaction_delta_pp_range"][1], max(deltas), "interaction delta max")
    lows = [v["ci_cluster_95"][0] * 100 for v in inter]
    highs = [v["ci_cluster_95"][1] * 100 for v in inter]
    close(h["interaction_ci_pp_envelope"][0], min(lows), "CI envelope low")
    close(h["interaction_ci_pp_envelope"][1], max(highs), "CI envelope high")
    close(h["interaction_max_abs_ci_endpoint_pp"], max(abs(min(lows)), abs(max(highs))), "resolution")
    require(h["interaction_pointwise_ci_excluding_zero"]
            == sum(lo > 0 or hi < 0 for lo, hi in zip(lows, highs)), "pointwise count changed")
    require(h["n_cells"] == 24 and h["n_interaction_tests"] == 12, "grid size changed")
    require(h["n_scored_requests"] == sum(x["n"] for x in cells(CONDITIONS)), "request total changed")
    require(h["evaluation_request_errors"]
            == sum(x["request_outcomes"]["identity_preserved_request_errors"] for x in cells(CONDITIONS)),
            "evaluation error total changed")
    require(h["question_generation_request_errors"]
            == sum(claims["corpora"][c]["question_generation"]["identity_preserved_request_errors"] for c in CORPORA),
            "generation error total changed")
    qa = h["qualitative_assertions"]
    require(qa["llmlingua_cost_below_every_naive_cost"] == (max(l2) < min(naive)), "L2-vs-naive assertion changed")

    # --- the within-deployment orderings stated in words in the paper
    for c in CORPORA:
        for t in TIERS:
            g = claims["grid"][c]["cells"]
            o, l, a, e = (g[f"{t}-{d}"] for d in CONDITIONS)
            require(o["acc"] > l["acc"] > max(a["acc"], e["acc"]),
                    f"{c}/{t}: O > L2 > both naive rungs no longer holds")
            require(all(o["acc_per_min"] > x["acc_per_min"] for x in (l, a, e)),
                    f"{c}/{t}: original context no longer leads accuracy per minute")
            require(all(x["mean_wall_s"] > o["mean_wall_s"] for x in (l, a, e)),
                    f"{c}/{t}: a compressed condition is no longer slower than the original")
            checked["orderings"] += 1

    # --- every numeric macro in the paper
    # The builder emitted macros in its in-memory order; the public claims file
    # is key-sorted, so compare the complete name -> value mapping exactly.
    fresh = macro_map(emit_tex(claims))
    if PAPER_TEX.is_file():
        require(sha256(PAPER_TEX) == EXPECTED_CLAIMS_TEX_SHA256, "changed paper/derived/claims.tex")
        shipped = macro_map(PAPER_TEX.read_text(encoding="utf-8"))
        missing = sorted(set(shipped) ^ set(fresh))
        require(not missing, f"macro sets differ: {missing[:5]}")
        wrong = sorted(k for k in fresh if fresh[k] != shipped[k])
        require(not wrong, f"paper macros differ from claims.json: {wrong[:5]}")
    checked["macros"] = len(fresh)

    cells_csv, interaction_csv = flat_tables(claims)
    for name, text in (("cells.csv", cells_csv), ("interaction.csv", interaction_csv)):
        require((DATA / name).read_text(encoding="utf-8") == text,
                f"data/{name} is not the exact derivation of claims.json")
    print(f"cells {checked['cells']}, paired comparisons {checked['paired']}, "
          f"interaction tests {checked['interaction']}, within-deployment orderings "
          f"{checked['orderings']}, paper macros {checked['macros']}")
    print(f"result branch: {branch} (Holm rejects {rejections} of 12)")
    print("ALL PAPER 2 P2R1 PUBLIC CLAIM CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"PAPER 2 PUBLIC CLAIM CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
