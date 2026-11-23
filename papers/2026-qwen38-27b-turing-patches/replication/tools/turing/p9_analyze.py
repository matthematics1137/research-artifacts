#!/usr/bin/env python3
"""P9 phase 1 analysis (research/prereg/2026-11-05-p9-phase1.md, sections 3 and 4): the registered estimands and the
verdicts on H1-H16 and the limiter rule, from one desktop run directory and one laptop run directory (p9_run_*.sh).
Smoke rows are skipped. Writes OUT/report.md and OUT/report.json.
  python3 p9_analyze.py --desktop testsuite/results/turing/p9/STAMP --laptop testsuite/results/ada/p9/STAMP --out DIR
"""
import argparse
import bisect
import datetime as dt
import json
import math
import os
import re
import statistics as st

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
MODEL = ["attn_kv", "gdn_z", "gdn_out", "gdn_qkv", "attn_q", "mlp_gate", "mlp_down"]
GRID_SHAPES = [(f"g{k}x{n}", k, n) for k in (2048, 5120, 10240, 17408) for n in (1024, 4096, 10240, 17408)]   # p9_common.GRID
LARGEST4 = ["gdn_qkv", "attn_q", "mlp_gate", "mlp_down"]
BUNDLES = {"fused_gate_up": [("mlp_gate", 2)], "fused_qkv_z": [("gdn_qkv", 1), ("gdn_z", 1)],
           "fused_q_k_v": [("attn_q", 1), ("attn_kv", 2)]}
CENSUS = {"desktop": "testsuite/results/turing/census/d0-w12p/int8_by_bits.md",
          "laptop": "testsuite/results/ada/census/d0-stock-mgemmn0-boost/int8_by_bits.md"}
ADA_CENSUS_C1 = {"80 W census": (0.712, 1515.0), "175 W census": (0.481, 2310.0)}   # (b at K = 1, median busy SM MHz)
ADA_CENSUS_A2_KCYC = (5.5 * 2.310, 9.7 * 1.515, 10.0 * 1.515)                    # thousand cycles, three censuses


SMOKE = False   # --include-smoke, for testing this script on smoke rows only


def rows(path, kind=None):
    if not os.path.exists(path):
        return []
    out = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in out if (SMOKE or not r.get("smoke")) and (kind is None or r.get("kind") == kind)]


def clocks(run):
    """Sorted (epoch, sm MHz, mem MHz, power W, temp C) from the sampler, timestamps read in the run's UTC offset."""
    off = open(os.path.join(run, "utc_offset.txt")).read().strip()
    tz = dt.timezone(dt.timedelta(hours=int(off[:3]), minutes=int(off[0] + off[3:])))
    out = []
    for line in open(os.path.join(run, "clocks.csv")):
        p = [x.strip() for x in line.split(",")]
        try:
            t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
            out.append((t, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        except (ValueError, IndexError):
            continue
    out.sort()
    return out


def clock_at(ck, t0, t1, warm=0.15):
    """Median SM and memory clock over the timing and the warm-up replays just before it."""
    ts = [c[0] for c in ck]
    i, j = bisect.bisect_left(ts, t0 - warm), bisect.bisect_right(ts, t1)
    sel = ck[i:j] or ck[max(0, i - 1):i + 1]
    if not sel:
        return None, None
    return st.median(c[1] for c in sel), st.median(c[2] for c in sel)


def ols(pts):
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    return my - b * mx, b


def t_interval(vals):
    n = len(vals)
    m = st.mean(vals)
    if n < 2:
        return m, None
    return m, T975.get(n - 1, 1.96) * st.stdev(vals) / math.sqrt(n)


def fits(timing, arm, K, shapes, wkey="mweights"):
    """Fit on per-shape medians over rounds, and the t interval over per-round fits."""
    sel = [r for r in timing if r.get("arm", r.get("tag")) == arm and r["K"] == K and r["shape"] in shapes]
    by_shape, by_round = {}, {}
    for r in sel:
        by_shape.setdefault(r["shape"], []).append(r)
        by_round.setdefault(r["round"], []).append(r)
    if len(by_shape) < 3:
        return None
    med = {s: st.median(x["us"] for x in v) for s, v in by_shape.items()}
    w = {s: v[0][wkey] for s, v in by_shape.items()}
    a, b = ols([(w[s], med[s]) for s in med])
    per_round = [ols([(x[wkey], x["us"]) for x in v]) for v in by_round.values() if len(v) >= 3]
    am, ah = t_interval([p[0] for p in per_round])
    bm, bh = t_interval([p[1] for p in per_round])
    return {"a": a, "b": b, "a_half": ah, "b_half": bh, "rounds": len(per_round), "medians": med, "w": w}


def census_calls(path):
    """[(calls per token, weights, K)] from a census's int8_by_bits.md (launches over the head's launch count)."""
    out, head = [], None
    rx = re.compile(r"^\| (\w+) \| (\d) \| (\d+) \| [\d.]+ \| ([\d.]+) \|")
    lines = [rx.match(l) for l in open(path)]
    for m in lines:
        if m and m[1] == "head":
            head = int(m[3])
    for m in lines:
        if m:
            K = int(m[2])
            out.append((int(m[3]) / head, float(m[4]) * 8e6 / K, K))
    return out


def fixed_share(fitK, calls, rate_gbs):
    excess = fixed = 0.0
    for n, w, K in calls:
        f = fitK.get(K)
        if not f:
            continue
        t = f["a"] + f["b"] * w / 1e6
        excess += n * (t - w * K / 8 / rate_gbs / 1e3)
        fixed += n * f["a"]
    return fixed / excess if excess > 0 else None, fixed, excess


FIT_LOG_RE = re.compile(r"EXL3_INT8_SQ_FIT k=(\d+) n=(\d+) K=(\d+) M=1 residual=0: kernel rule rows_per=(\d+) "
                        r"ksplit=(\d+) grid=(\d+) -> host rows_per=(\d+) ksplit=(\d+) grid=(\d+)")


def decomps(run, mach, arm, m2rows):
    """{(k, n, K): (rows_per, ksplit, grid)} as the engine ran it: the desktop's W9 host log (M1's stderr), else the
    laptop's M2 decomposition rows (kernel rule; our build's for our arms, stock89's for the stock arms)."""
    out = {}
    if mach == "desktop":
        for f in os.listdir(run):
            if f.startswith("m1.") and f.endswith(".stderr"):
                for m in FIT_LOG_RE.finditer(open(os.path.join(run, f)).read()):
                    k, n, K, rp, ks, g = int(m[1]), int(m[2]), int(m[3]), int(m[7]), int(m[8]), int(m[9])
                    if rp:
                        out.setdefault((k, n, K), (rp, ks, g))
    else:
        tag = "w13sm89" if arm.startswith("w13") else "stock89"
        for r in m2rows:
            if r.get("kind") == "decomp" and r.get("tag") == tag:
                out[(r["k"], r["n"], r["K"])] = (r["rows_per"], r["ksplit"], r["grid"])
    return out


def kernel_rule_units(k, n, grid):
    """Units and waves of a shape under the kernel's own rule at a grid (the laptop's fused shapes; no M2 row)."""
    rows_total, nb = k // 16, n // 256
    r = -(-rows_total * nb // grid)
    rp = min(max((max(r, min(2 * r, 32)) + 7) & ~7, 16), 512, (rows_total + 7) & ~7)
    ks = -(-rows_total // rp)
    return rp, ks, grid


def lstsq3(rows):
    """Least squares t = a + b w + c x over (w, x, t) rows, by the normal equations (3x3)."""
    import itertools
    X = [(1.0, w, x) for w, x, _ in rows]
    y = [t for *_, t in rows]
    A = [[sum(xi[i] * xi[j] for xi in X) for j in range(3)] for i in range(3)]
    v = [sum(xi[i] * yi for xi, yi in zip(X, y)) for i in range(3)]
    # Gaussian elimination
    M = [A[i] + [v[i]] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r_: abs(M[r_][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r_ in range(3):
            if r_ != c:
                f = M[r_][c] / M[c][c]
                M[r_] = [a - f * b for a, b in zip(M[r_], M[c])]
    return [M[i][3] / M[i][i] for i in range(3)]


def verdict(ok):
    return "holds" if ok else "fails"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desktop", required=True)
    ap.add_argument("--laptop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-smoke", action="store_true", help="testing only: analyse smoke rows")
    ap.add_argument("--desktop-1b", default="", help="phase 1b run directory on the desktop (L2 against DRAM)")
    ap.add_argument("--laptop-1b", default="", help="phase 1b run directory on the laptop (L2 against DRAM)")
    a = ap.parse_args()
    global SMOKE
    SMOKE = a.include_smoke
    os.makedirs(a.out, exist_ok=True)
    R, md = {}, []

    def say(s=""):
        md.append(s)

    H = {}

    def hyp(name, ok, text):
        H[name] = {"verdict": verdict(ok) if ok is not None else "not tested", "detail": text}
        say(f"- **{name}** {H[name]['verdict']}: {text}")

    for mach in ("desktop", "laptop"):
        run = getattr(a, mach)
        ck = clocks(run)
        m1 = rows(os.path.join(run, "m1.jsonl"), "time")
        for r in m1:
            r["sm_mhz"], r["mem_mhz"] = clock_at(ck, r["t0"], r["t1"])
        m2 = rows(os.path.join(run, "m2.jsonl"))
        m2t = [r for r in m2 if r["kind"] == "time"]
        for r in m2t:
            r["sm_mhz"], r["mem_mhz"] = clock_at(ck, r["t0"], r["t1"])
        sass = {}
        sd = os.path.join(run, "sass")
        for f in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
            if f.endswith(".json"):
                j = json.load(open(os.path.join(sd, f)))
                sass[(j["label"], j["arch"])] = j["kernels"]
        R[mach] = {"m1": m1, "m2": m2, "m2t": m2t, "sass": sass, "ck": ck}

    out = {"fits": {}, "hypotheses": H}
    sms = {"desktop": 68, "laptop": 58}

    # ---- M1 fits per arm and K, with clocks
    say("# P9 phase 1 report")
    say()
    say("## M1: engine calls, unprofiled, cache-rotated")
    say()
    say("| machine | arm | K | a (us) | +- | b (us/Mw) | +- | SM MHz | mem MHz |")
    say("|---|---|---|---|---|---|---|---|---|")
    F = {}
    for mach in ("desktop", "laptop"):
        m1 = R[mach]["m1"]
        for arm in sorted({r["arm"] for r in m1}):
            for K in (1, 2, 3, 4):
                f = fits(m1, arm, K, MODEL)
                if not f:
                    continue
                sel = [r for r in m1 if r["arm"] == arm and r["K"] == K and r["shape"] in MODEL and r["sm_mhz"]]
                f["sm_mhz"] = st.median(r["sm_mhz"] for r in sel) if sel else None
                f["mem_mhz"] = st.median(r["mem_mhz"] for r in sel) if sel else None
                F[(mach, arm, K)] = f
                say(f"| {mach} | {arm} | {K} | {f['a']:.2f} | {f['a_half'] or 0:.2f} | {f['b']:.4f} | "
                    f"{f['b_half'] or 0:.4f} | {f['sm_mhz'] or 0:.0f} | {f['mem_mhz'] or 0:.0f} |")
    out["fits"] = {f"{k[0]}/{k[1]}/K{k[2]}": {x: y for x, y in v.items() if x not in ("medians", "w")} for k, v in F.items()}
    out["laptop_cooled"] = sum(len(rows(os.path.join(a.laptop, f), "cooled")) for f in ("m1.jsonl", "m2.jsonl"))
    spread = []
    for K in (1, 2):
        sms_ = [F[k]["sm_mhz"] for k in F if k[0] == "laptop" and k[2] == K and F[k]["sm_mhz"]]
        if sms_:
            spread.append(max(sms_) / min(sms_) - 1)
    out["laptop_sm_spread"] = max(spread) if spread else None
    out["medians"] = {f"{k[0]}/{k[1]}/K{k[2]}": v["medians"] for k, v in F.items()}

    def med(mach, arm, K, shape):
        sel = [r["us"] for r in R[mach]["m1"] if r["arm"] == arm and r["K"] == K and r["shape"] == shape]
        return st.median(sel) if sel else None

    for mach in ("desktop", "laptop"):
        for arm, K, shape in sorted({(r["arm"], r["K"], r["shape"]) for r in R[mach]["m1"]}):
            out.setdefault("shape_medians", {}).setdefault(f"{mach}/{arm}/K{K}", {})[shape] = med(mach, arm, K, shape)

    # Secondary (registered, section 3): the grid fit with ksplit as a covariate; and descriptive extras the
    # cold review asked for (research/raw/2026-11-05-p9-reviews/cold-significance-review.md)
    say()
    say("## M1 secondary: grid fit with ksplit (t = a + b w + c ksplit, 16 grid shapes)")
    say()
    out["grid_fit"] = {}
    for mach in ("desktop", "laptop"):
        run = getattr(a, mach)
        for arm in sorted({r["arm"] for r in R[mach]["m1"]}):
            dec = decomps(run, mach, arm, R[mach]["m2"])
            for K in (1, 2):
                pts = []
                for name, k, n in GRID_SHAPES:
                    v = [r["us"] for r in R[mach]["m1"] if r["arm"] == arm and r["K"] == K and r["shape"] == name]
                    d = dec.get((k, n, K))
                    if v and d:
                        pts.append((k * n / 1e6, d[1], st.median(v)))
                if len(pts) >= 6:
                    sol = lstsq3(pts)
                    two = ols([(w, t) for w, _, t in pts])
                    out["grid_fit"][f"{mach}/{arm}/K{K}"] = {"a": sol[0], "b": sol[1], "c_ksplit": sol[2],
                                                             "a_no_ksplit": two[0], "b_no_ksplit": two[1], "n": len(pts)}
                    say(f"- {mach} {arm} K={K}: a {sol[0]:.2f} us, b {sol[1]:.4f} us/Mw, {sol[2]:+.3f} us per slice "
                        f"(without ksplit: a {two[0]:.2f}, b {two[1]:.4f}; {len(pts)} shapes)")

    say()
    say("## Fusion by card, build and bit width, with the merged call's waves (descriptive)")
    say()
    bundles = {"fused_gate_up": [("mlp_gate", 2)], "fused_qkv_z": [("gdn_qkv", 1), ("gdn_z", 1)],
               "fused_q_k_v": [("attn_q", 1), ("attn_kv", 2)]}
    fshape = {"fused_gate_up": (5120, 34816), "fused_qkv_z": (5120, 16384), "fused_q_k_v": (5120, 14336)}
    out["fusion_all"] = {}
    for mach, arms_ in (("desktop", ["w13"]), ("laptop", ["w13sm89", "stock89", "stock"])):
        run = getattr(a, mach)
        for arm in arms_:
            dec = decomps(run, mach, arm, R[mach]["m2"])
            for K in (1, 2):
                f = F.get((mach, arm, K))
                if not f:
                    continue
                for fu, parts in bundles.items():
                    tf = med(mach, arm, K, fu) if 'med' in dir() else None
                    vals = [r["us"] for r in R[mach]["m1"] if r["arm"] == arm and r["K"] == K and r["shape"] == fu]
                    sep = []
                    for sname, cnt in parts:
                        v = [r["us"] for r in R[mach]["m1"] if r["arm"] == arm and r["K"] == K and r["shape"] == sname]
                        sep.append(cnt * st.median(v) if v else None)
                    if not vals or None in sep:
                        continue
                    k, n = fshape[fu]
                    d = dec.get((k, n, K)) or kernel_rule_units(k, n, 290 if mach == "laptop" else 272)
                    grid = d[2]
                    units = (n // 256) * d[1]
                    nparts = sum(cnt for _, cnt in parts)
                    saving = (sum(sep) - st.median(vals)) / (nparts - 1) / f["a"]
                    out["fusion_all"][f"{mach}/{arm}/K{K}/{fu}"] = {"saving_in_a": saving, "units": units,
                                                                    "grid": grid, "waves": -(-units // grid)}
                    say(f"- {mach} {arm} K={K} {fu}: {saving:+.2f} of a fixed part per merged call; {units} units on "
                        f"{grid} blocks ({-(-units // grid)} wave(s))")

    say()
    say("## Registered predictions")
    say()
    # ---- 2080 Ti
    d2 = F.get(("desktop", "w13", 2))
    if d2:
        hyp("H1", 9 <= d2["a"] <= 13 and (d2["a_half"] or 99) <= 1.0,
            f"a(K=2) = {d2['a']:.2f} us, interval half-width {d2['a_half'] or float('nan'):.2f} (need 9-13, <= 1.0)")
        fitK = {K: F[("desktop", "w13", K)] for K in (1, 2, 3, 4) if ("desktop", "w13", K) in F}
        share, fx, ex = fixed_share(fitK, census_calls(CENSUS["desktop"]), 550)
        out["h2"] = {"share": share, "fixed_ms": fx / 1e3, "excess_ms": ex / 1e3,
                     "calls_per_token": sum(n for n, _, _ in census_calls(CENSUS["desktop"]))}
        sh616, _, ex616 = fixed_share(fitK, census_calls(CENSUS["desktop"]), 616)
        out["h2"]["share_rated_616"] = sh616   # descriptive: at the card's rated rate
        hyp("H2", share is not None and share >= 0.70,
            f"fixed parts {fx / 1e3:.2f} ms of a {ex / 1e3:.2f} ms excess per token: {100 * (share or 0):.1f}% (need >= 70%)")
        tiny = [med("desktop", "w13", K, "tiny") for K in (1, 2)]
        null = [r["us"] for r in R["desktop"]["m2t"] if r["variant"] == 16]
        nf = None
        for K in (1, 2):
            f = fits([r for r in R["desktop"]["m2t"] if r["variant"] == 16 and r["decomp_K"] == K], "w13", K, MODEL)
            nf = max(nf or 0, f["a"]) if f else nf
        hyp("H3", all(t is not None and t <= 6 for t in tiny) and nf is not None and nf <= 3,
            f"near-empty call {', '.join(f'{t:.2f}' for t in tiny if t)} us (need <= 6); null kernel intercept "
            f"{nf if nf is None else round(nf, 2)} us (need <= 3)")
        ratios = [med("desktop", "w13", 1, s) / med("desktop", "w13", 2, s) for s in MODEL
                  if med("desktop", "w13", 1, s) and med("desktop", "w13", 2, s)]
        g = math.exp(st.mean(math.log(x) for x in ratios)) if ratios else None
        out["one_over_two"] = {s: med("desktop", "w13", 1, s) / med("desktop", "w13", 2, s) for s in MODEL
                               if med("desktop", "w13", 1, s) and med("desktop", "w13", 2, s)}
        out["one_over_two_geomean"] = g
        out["tiny"] = {f"K{K}": med("desktop", "w13", K, "tiny") for K in (1, 2)}
        hyp("H4", g is not None and 0.95 <= g <= 1.20,
            f"1-bit over 2-bit call time, geometric mean over {len(ratios)} shapes: {g if g is None else round(g, 3)} "
            f"(range {min(ratios):.3f}-{max(ratios):.3f}; need 0.95-1.20)")
        sv = []
        for fused, parts in BUNDLES.items():
            tf = med("desktop", "w13", 2, fused)
            ts = sum(n * (med("desktop", "w13", 2, s) or float("nan")) for s, n in parts)
            nparts = sum(n for _, n in parts)
            sv.append((fused, (ts - tf) / (nparts - 1) / d2["a"] if tf else None))
        hyp("H5", all(x is not None and x < 0.5 for _, x in sv),
            "saving per merged call in units of a: " + ", ".join(f"{f} {x:.2f}" for f, x in sv if x is not None)
            + f" (need < 0.5 each; refuted with fusion the lever if >= 0.8 in two: "
            f"{sum(1 for _, x in sv if x is not None and x >= 0.8)} of 3)")
        out["fusion_saving_in_a"] = dict(sv)

    # ---- M2 phases
    def phase_fit(mach, tag, K, v, dK=None):
        rr = [r for r in R[mach]["m2t"] if r["tag"] == tag and r["K"] == K and r["variant"] == v
              and r["decomp_K"] == (dK or K)]
        return fits(rr, tag, K, MODEL)

    def multiwave(mach, tag):
        bad = set()
        for r in R[mach]["m2"]:
            if r["kind"] == "decomp" and r["tag"] == tag:
                for v, w in r["waves"].items():
                    if w > 1:
                        bad.add((r["shape"], r["K"], int(v)))
        return bad

    say()
    say("## M2: phase split (intercepts over the model shapes)")
    say()
    say("| machine | build | K | full | no stage | no loop | no epilogue | loop only | stage only | epilogue only | counter only | null |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    P = {}
    for mach, tags in (("desktop", ["w13"]), ("laptop", ["stock89", "w13sm89"])):
        for tag in tags:
            # Cases whose full variant did not reproduce the engine's output are not the engine's kernel: excluded
            bad = {(r["shape"], r["K"]) for r in R[mach]["m2"] if r["kind"] == "decomp" and r["tag"] == tag
                   and not r["bit_identical_to_engine"]}
            if bad:
                say(f"(cases not bit-identical to the engine, excluded for {mach}/{tag}: {sorted(bad)})")
                R[mach]["m2t"] = [r for r in R[mach]["m2t"] if not (r["tag"] == tag and (r["shape"], r["decomp_K"]) in bad)]
            mw = multiwave(mach, tag)
            if mw:
                say(f"(multi-wave variant cases excluded for {mach}/{tag}: {sorted(mw)})")
                R[mach]["m2t"] = [r for r in R[mach]["m2t"] if not (r["tag"] == tag and (r["shape"], r["K"], r["variant"]) in mw)]
            for K in (1, 2):
                vals = {v: phase_fit(mach, tag, K, v) for v in (7, 6, 5, 3, 2, 1, 4, 0, 16)}
                P[(mach, tag, K)] = vals
                out.setdefault("phases", {})[f"{mach}/{tag}/K{K}"] = {
                    str(v): ({"a": f["a"], "a_half": f["a_half"], "b": f["b"], "b_half": f["b_half"]} if f else None)
                    for v, f in vals.items()}
                say(f"| {mach} | {tag} | {K} | " + " | ".join(
                    f"{vals[v]['a']:.2f}" if vals[v] else "-" for v in (7, 6, 5, 3, 2, 1, 4, 0, 16)) + " |")
            ident = [r["bit_identical_to_engine"] for r in R[mach]["m2"] if r["kind"] == "decomp" and r["tag"] == tag]
            say(f"(full variant bit-identical to the engine in {sum(ident)} of {len(ident)} cases for {mach}/{tag})")
    say()
    pa_ = P.get(("laptop", "w13sm89", 2))
    if pa_ and pa_[7] and pa_[6] and pa_[5] and pa_[3]:
        fa = pa_[7]["a"]
        out["phase_contrib_laptop_K2"] = {"full": fa, "loop": fa - pa_[5]["a"], "stage": fa - pa_[6]["a"],
                                          "epilogue": fa - pa_[3]["a"], "null": pa_[16]["a"] if pa_[16] else None,
                                          "counter": pa_[0]["a"] if pa_[0] else None}
    pd = P.get(("desktop", "w13", 2))
    if pd and pd[7]:
        full = pd[7]["a"]
        loop_c = full - pd[5]["a"] if pd[5] else None
        st_c = full - pd[6]["a"] if pd[6] else None
        ep_c = full - pd[3]["a"] if pd[3] else None
        out["phase_contrib_desktop_K2"] = {"full": full, "loop": loop_c, "stage": st_c, "epilogue": ep_c,
                                           "null": pd[16]["a"] if pd[16] else None,
                                           "counter": pd[0]["a"] if pd[0] else None}
        hyp("H6", loop_c is not None and st_c is not None and ep_c is not None and loop_c >= 0.5 * full
            and st_c + ep_c <= 0.4 * full,
            f"full intercept {full:.2f} us: loop contributes {loop_c:.2f} ({100 * loop_c / full:.0f}%), stage "
            f"{st_c:.2f} and epilogue {ep_c:.2f} ({100 * (st_c + ep_c) / full:.0f}% together; need loop >= 50%, "
            f"stage + epilogue <= 40%)")
        rr = []
        for s in LARGEST4:
            x = [r["us"] for r in R["desktop"]["m2t"] if r["tag"] == "w13" and r["shape"] == s and r["variant"] == 2
                 and r["K"] == 1 and r["decomp_K"] == 2]
            y = [r["us"] for r in R["desktop"]["m2t"] if r["tag"] == "w13" and r["shape"] == s and r["variant"] == 2
                 and r["K"] == 2 and r["decomp_K"] == 2]
            if x and y:
                rr.append((s, st.median(x) / st.median(y)))
        out["h7_ratios"] = dict(rr)
        kv = {}
        for K, dK in ((1, 1), (1, 2), (2, 2)):
            x = [r["us"] for r in R["desktop"]["m2t"] if r["tag"] == "w13" and r["shape"] == "attn_kv" and r["variant"] == 2
                 and r["K"] == K and r["decomp_K"] == dK]
            kv[f"K{K}_at_K{dK}"] = st.median(x) if x else None
        out["attn_kv_loop"] = kv
        hyp("H7", len(rr) == 4 and all(v >= 0.9 for _, v in rr),
            "1-bit loop at the 2-bit decomposition over the 2-bit loop: " + ", ".join(f"{s} {v:.3f}" for s, v in rr)
            + " (need >= 0.90 on all four)")

    # ---- M3 machine code
    def ipw(mach, label, arch, K):
        k = R[mach]["sass"].get((label, arch), {}).get(str(K)) or R[mach]["sass"].get((label, arch), {}).get(K)
        return k.get("instructions_per_weight") if k and k.get("loop") else None

    pairs = [("desktop", "w13", "sm_75"), ("laptop", "w13sm89", "sm_89"), ("desktop", "w11b", "sm_75"),
             ("laptop", "stock", "sm_75"), ("laptop", "stock", "sm_89"), ("laptop", "stock89", "sm_89")]
    say()
    say("## M3: main loop instructions per weight")
    say()
    for mach, lab, arch in pairs:
        say(f"- {lab} {arch}: K=1 {ipw(mach, lab, arch, 1)}, K=2 {ipw(mach, lab, arch, 2)}")
        out.setdefault("sass", {})[f"{lab}/{arch}"] = {f"K{K}": ipw(mach, lab, arch, K) for K in (1, 2)}
    r8 = [(lab, arch, ipw(m, lab, arch, 1) / ipw(m, lab, arch, 2)) for m, lab, arch in pairs[:2]
          if ipw(m, lab, arch, 1) and ipw(m, lab, arch, 2)]
    say()
    hyp("H8", (all(x >= 0.9 for *_, x in r8)) if len(r8) == 2 else None,
        "K=1 over K=2 instructions per weight: " + ", ".join(f"{l} {a} {x:.3f}" for l, a, x in r8) + " (need >= 0.9)")
    # sm_75: w11b (the lab's toolchain, W12p absent) against w13 (Amendment 1); the official wheel is reported above
    s75, w75 = ipw("desktop", "w11b", "sm_75", 1), ipw("desktop", "w13", "sm_75", 1)
    s89, w89 = ipw("laptop", "stock89", "sm_89", 1), ipw("laptop", "w13sm89", "sm_89", 1)
    red75 = 1 - w75 / s75 if s75 and w75 else None
    red89 = 1 - w89 / s89 if s89 and w89 else None
    out["w12p_reduction"] = {"sm_75": red75, "sm_89": red89}
    hyp("H9", (red75 >= 0.05 and red89 >= 0.05) if red75 is not None and red89 is not None else None,
        f"W12p's reduction at K=1: sm_75 {100 * (red75 or 0):.1f}%, sm_89 {100 * (red89 or 0):.1f}% (need >= 5% on both)")

    # ---- laptop
    def ratio_all(arm1, arm2, K):
        out_ = []
        for s in MODEL:
            x, y = med("laptop", arm1, K, s), med("laptop", arm2, K, s)
            if x and y:
                out_.append(x / y)
        return out_

    tol = [abs(x - 1) for K in (1, 2) for x in ratio_all("stock89", "stock", K)]
    out["h10_maxdiff"] = max(tol) if tol else None
    hyp("H10", (max(tol) <= 0.03) if len(tol) == 2 * len(MODEL) else None, f"stock89 against the official wheel: largest difference "
        f"{100 * max(tol) if tol else float('nan'):.1f}% over {len(tol)} shape-K pairs (need <= 3%)")
    s1, w1 = F.get(("laptop", "stock89", 1)), F.get(("laptop", "w13sm89", 1))
    s2, w2 = F.get(("laptop", "stock89", 2)), F.get(("laptop", "w13sm89", 2))
    if s1 and w1 and s2 and w2:
        drop1 = 1 - w1["b"] / s1["b"]
        ch2 = w2["b"] / s2["b"] - 1
        out["h11"] = {"b1_drop": drop1, "b2_change": ch2}
        hyp("H11", red89 is not None and drop1 >= red89 / 2 and abs(ch2) <= 0.05,
            f"w13sm89 against stock89: b(K=1) {100 * drop1:+.1f}% lower (need >= half of {100 * (red89 or 0):.1f}%), "
            f"b(K=2) {100 * ch2:+.1f}% (need within 5%)")
        q = {arm: F[("laptop", arm, 1)]["b"] / F[("laptop", arm, 2)]["b"] for arm in ("stock89", "w13sm89")}
        out["b1_over_b2_ada"] = q
        hyp("H12", all(v <= 0.85 for v in q.values()), "b(K=1)/b(K=2): " + ", ".join(f"{k} {v:.3f}" for k, v in q.items())
            + " (need <= 0.85)")
    # Descriptive: the laptop's fixed share over its own census's calls, at P8's primary and rated read rates
    fitA = {K: F[("laptop", "w13sm89", K)] for K in (1, 2, 3, 4) if ("laptop", "w13sm89", K) in F}
    if fitA:
        out["laptop_share"] = {}
        for label, rate in (("primary 386", 432 * 550 / 616), ("rated 432", 432.0)):
            sh, fx, ex = fixed_share(fitA, census_calls(CENSUS["laptop"]), rate)
            out["laptop_share"][label] = {"share": sh, "fixed_ms": fx / 1e3, "excess_ms": ex / 1e3}
            say(f"- laptop (w13sm89) fixed share at {label} GB/s: {100 * (sh or 0):.0f}% ({fx / 1e3:.2f} of {ex / 1e3:.2f} ms)")
    st1 = F.get(("laptop", "stock", 1))
    if st1 and st1["sm_mhz"]:
        c = {k: b * 1e-12 * 58 * f * 1e6 for k, (b, f) in ADA_CENSUS_C1.items()}
        c["this bench"] = st1["b"] * 1e-12 * 58 * st1["sm_mhz"] * 1e6
        out["c1_ada_stock"] = c
        hyp("H13", max(c.values()) / min(c.values()) <= 1.10,
            "1-bit SM-cycles per weight (stock): " + ", ".join(f"{k} {v:.4f}" for k, v in c.items())
            + f" (spread {100 * (max(c.values()) / min(c.values()) - 1):.1f}%, need <= 10%)")
    st2 = F.get(("laptop", "stock", 2))
    if st2 and st2["sm_mhz"]:
        kc = st2["a"] * st2["sm_mhz"] / 1e3
        out["a2_kcycles_ada_stock"] = kc
        lo, hi = min(ADA_CENSUS_A2_KCYC) / 1.2, max(ADA_CENSUS_A2_KCYC) * 1.2
        hyp("H14", lo <= kc <= hi, f"a(K=2) x SM clock = {kc:.2f} thousand cycles (need {lo:.1f}-{hi:.1f})")
    dw1 = F.get(("desktop", "w13", 1))
    if dw1 and w1 and dw1["sm_mhz"] and w1["sm_mhz"]:
        cT = dw1["b"] * 1e-12 * 68 * dw1["sm_mhz"] * 1e6
        cA = w1["b"] * 1e-12 * 58 * w1["sm_mhz"] * 1e6
        out["c1_cards"] = {"desktop": cT, "laptop": cA}
        hyp("H15", 0.9 <= cT / cA <= 1.4, f"1-bit cycles per weight, 2080 Ti {cT:.4f} over laptop {cA:.4f} = "
            f"{cT / cA:.3f} (descriptive band 0.9-1.4)")
    wf = F.get(("laptop", "w13sm89fit", 2))
    if wf and w2:
        out["h16_delta"] = wf["a"] - w2["a"]
        hyp("H16", abs(wf["a"] - w2["a"]) < 1.5, f"W9 on changes a(K=2) by {wf['a'] - w2['a']:+.2f} us (need < 1.5)")

    # ---- descriptive extras (the cold review's requests)
    ex = {}
    d1, d2, d4 = F.get(("desktop", "w13", 1)), F.get(("desktop", "w13", 2)), F.get(("desktop", "w13", 4))
    if d1 and d2:
        ex["desktop_b1_over_b2"] = d1["b"] / d2["b"]
        rr = [med("desktop", "w13", 1, s_) / med("desktop", "w13", 2, s_) for s_ in MODEL if s_ != "attn_kv"]
        ex["desktop_call_ratio_without_attn_kv"] = math.exp(st.mean(math.log(x) for x in rr))
        ex["desktop_lower_c_bound"] = 1 - (0.25e6 / 550e9 * 1e6) / d2["b"]   # 2-bit read at 550 over the 2-bit slope
    if d4:
        ex["desktop_rate_from_4bit_gbs"] = 0.5 / (d4["b"] * 1e-12) / 1e9
    for arm in ("w13sm89", "stock89", "stock"):
        rr = [med("laptop", arm, 1, s_) / med("laptop", arm, 2, s_) for s_ in MODEL
              if med("laptop", arm, 1, s_) and med("laptop", arm, 2, s_)]
        if rr:
            ex[f"laptop_call_ratio_{arm}"] = math.exp(st.mean(math.log(x) for x in rr))
    for key in (("desktop", "w13", 2), ("laptop", "w13sm89", 2), ("laptop", "stock89", 2), ("laptop", "stock", 2)):
        f = F.get(key)
        if f and f["sm_mhz"]:
            ex[f"kcycles_a2_{key[0]}_{key[1]}"] = f["a"] * f["sm_mhz"] / 1e3
    fit3 = F.get(("laptop", "w13sm89fit", 1))
    if fit3 and fit3["sm_mhz"]:
        ex["laptop_fit_c1"] = fit3["b"] * 1e-12 * 58 * fit3["sm_mhz"] * 1e6
    # the census's one-row calls per token (P8's d0-w12p census), by role and bits per weight
    tab = []
    rx = re.compile(r"^\| (\w+) \| (\d) \| (\d+) \| [\d.]+ \| ([\d.]+) \|")
    lines_ = [rx.match(l) for l in open(CENSUS["desktop"])]
    head = next(int(m_[3]) for m_ in lines_ if m_ and m_[1] == "head")
    for m_ in lines_:
        if m_:
            K_ = int(m_[2])
            tab.append({"role": m_[1], "K": K_, "calls_per_token": int(m_[3]) / head,
                        "mweights": float(m_[4]) * 8 / K_})
    ex["census_calls"] = tab
    out["extras"] = ex
    say()
    say("## Descriptive extras")
    say()
    for k_, v_ in ex.items():
        if k_ != "census_calls":
            say(f"- {k_}: {v_:.4f}")
    say(f"- census: {sum(t['calls_per_token'] for t in tab):.0f} one-row calls per token over {len(tab)} role-bit rows")

    # ---- limiter of the 1-bit main loop
    say()
    say("## Limiter of the 1-bit main loop")
    say()
    for mach, arm, lab, arch in (("desktop", "w13", "w13", "sm_75"), ("laptop", "w13sm89", "w13sm89", "sm_89")):
        f = F.get((mach, arm, 1))
        k = R[mach]["sass"].get((lab, arch), {}).get("1")
        if not f or not k or not k.get("loop") or not f["sm_mhz"]:
            say(f"- {mach} {arm}: missing inputs")
            continue
        hz = f["sm_mhz"] * 1e6
        issue = k["instructions_per_weight"] * 1e12 / (128 * sms[mach] * hz)
        pipes = {c: n * 1e12 / (k["class_throughput_per_clock_per_sm"][c] * sms[mach] * hz)
                 for c, n in k["classes_per_weight"].items() if c in k["class_throughput_per_clock_per_sm"]}
        top = max(pipes.items(), key=lambda x: x[1]) if pipes else ("none", 0)
        kind = ("issue-bound" if issue >= 0.8 * f["b"] else
                "pipe-bound" if top[1] >= 0.9 * f["b"] and top[1] > issue else
                "latency-bound (pending M4's stall reasons)" if max(issue, top[1]) < 0.6 * f["b"] else "unresolved")
        say(f"- {mach} {arm}: b(K=1) {f['b']:.4f} us/Mw at {f['sm_mhz']:.0f} MHz; issue floor {issue:.4f} "
            f"({100 * issue / f['b']:.0f}% of b); highest class floor {top[0]} {top[1]:.4f} "
            f"({100 * top[1] / f['b']:.0f}%); unassigned per weight "
            f"{k['classes_per_weight'].get('unassigned', 0):.3f}: **{kind}**")
        out.setdefault("limiter", {})[f"{mach}/{arm}"] = {"b": f["b"], "issue_floor": issue, "top_class": top,
                                                          "verdict": kind}

    # ---- phase 1b (registered addendum): weights held in L2 against read from DRAM
    if a.laptop_1b or a.desktop_1b:
        say()
        say("## Phase 1b: weights in L2 against DRAM")
        say()
        out["phase1b"] = {}
        for mach, run in (("laptop", a.laptop_1b), ("desktop", a.desktop_1b)):
            if not run:
                continue
            ck = clocks(run)
            rows_ = rows(os.path.join(run, "m1b.jsonl"), "time")
            for r_ in rows_:
                r_["sm_mhz"], r_["mem_mhz"] = clock_at(ck, r_["t0"], r_["t1"])
            arms_ = sorted({r_["arm"] for r_ in rows_})
            for K in (1, 2):
                res = {}
                for arm in arms_:
                    f = fits(rows_, arm, K, MODEL)
                    if f:
                        res[arm] = f
                l2 = next((f for arm, f in res.items() if arm.endswith("-l2")), None)
                dr = next((f for arm, f in res.items() if arm.endswith("-dram")), None)
                if l2 and dr:
                    ratio = l2["b"] / dr["b"]
                    # ratio's interval from the per-round ratios of the two fits (each round times both)
                    per_round = []
                    l2_arm = next(a_ for a_ in res if a_.endswith("-l2"))
                    dram_arm = next(a_ for a_ in res if a_.endswith("-dram"))
                    for rnd in sorted({r_["round"] for r_ in rows_}):
                        sel = [r_ for r_ in rows_ if r_["round"] == rnd]
                        fl, fd = fits(sel, l2_arm, K, MODEL), fits(sel, dram_arm, K, MODEL)
                        if fl and fd:
                            per_round.append(fl["b"] / fd["b"])
                    ratio_mean, ratio_half = t_interval(per_round) if len(per_round) > 1 else (ratio, None)
                    out["phase1b"][f"{mach}/K{K}"] = {"b_l2": l2["b"], "b_dram": dr["b"], "a_l2": l2["a"],
                                                      "a_dram": dr["a"], "ratio": ratio, "ratio_round_mean": ratio_mean,
                                                      "ratio_half": ratio_half, "rounds": len(per_round)}
                    if ratio_half is not None:
                        say(f"  per-round ratio {ratio_mean:.3f} +- {ratio_half:.3f} ({len(per_round)} rounds)")
                    say(f"- {mach} K={K}: b L2 {l2['b']:.4f} against DRAM {dr['b']:.4f} us/Mw (ratio {ratio:.3f}); "
                        f"a L2 {l2['a']:.2f} against DRAM {dr['a']:.2f} us")
                # per-shape ratios (the desktop's few shapes that fit its L2)
                for s_ in MODEL:
                    v1 = [r_["us"] for r_ in rows_ if r_["K"] == K and r_["shape"] == s_ and r_["arm"].endswith("-l2")]
                    v2 = [r_["us"] for r_ in rows_ if r_["K"] == K and r_["shape"] == s_ and r_["arm"].endswith("-dram")]
                    if v1 and v2:
                        out["phase1b"][f"{mach}/K{K}/{s_}"] = st.median(v1) / st.median(v2)
        p1, p2 = out["phase1b"].get("laptop/K1"), out["phase1b"].get("laptop/K2")
        if p2:
            hyp("H17", p2["ratio"] <= 0.70, f"laptop 2 bits, b(L2)/b(DRAM) = {p2['ratio']:.3f} (need <= 0.70)")
        if p1:
            v = p1["ratio"]
            H["H18"] = {"verdict": "holds" if v >= 0.85 else ("fails" if v <= 0.70 else "unresolved"),
                        "detail": f"laptop 1 bit, b(L2)/b(DRAM) = {v:.3f} (holds >= 0.85; refuted <= 0.70)"}
            say(f"- **H18** {H['H18']['verdict']}: {H['H18']['detail']}")
        dk = [out["phase1b"].get(f"desktop/K1/{s_}") for s_ in ("attn_kv", "gdn_z", "gdn_out")]
        if all(x is not None for x in dk):
            hyp("H19", all(abs(x - 1) <= 0.10 for x in dk), "2080 Ti, 1 bit, L2 over DRAM per shape: "
                + ", ".join(f"{s_} {x:.3f}" for s_, x in zip(("attn_kv", "gdn_z", "gdn_out"), dk)) + " (need within 10%)")

    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
