#!/usr/bin/env python3
"""H38 hardening analysis: paired, temperature-0 comparison of two eval labels on one suite.
Reports accuracy with Wilson 95% CIs, discordant items, exact McNemar p, output identity (sha256 of reasoning+answer),
token-count identity, length/wall ratios, and the verdict against the pre-registered H38a / H38b criteria
(research/efficiency-hypotheses-2026-09-06.md).
usage: hardening_analyze.py <suite> <labelA> <labelB> [--limit N] [--hyp h38a|h38b] [--json out.json]
  A = reference side (undrafted for H38a, Q4_K_XL for H38b); B = the drafted 2.00-bpw EXL3 run.
  --limit N restricts to the first N items in dataset order (H38b pairs the first 200 GSM8K-500 items)."""
import argparse, json, math, os, statistics

RES = "/path/to/lab-repo/testsuite/evals/results"
DS = "/path/to/lab-repo/testsuite/evals/datasets"
FILES = {"gsm8k_500": "gsm8k_500.jsonl", "mmlu_500": "mmlu_500.jsonl", "math500": "math500.jsonl",
         "gsm8k": "gsm8k_100.jsonl", "mmlu": "mmlu_200.jsonl", "math25": "math25.jsonl"}
CRITERIA = {  # pre-registered
    "h38a": {"identity_min": 0.95, "identity_falsify": 0.90, "dacc_max": 1.0, "dacc_falsify": 1.5},
    "h38b": {"identity_min": None, "identity_falsify": None, "dacc_max": 2.0, "dacc_falsify": 3.0},
}


def load(label, suite):
    p = os.path.join(RES, label, f"{suite}.jsonl")
    out = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); out[r["id"]] = r
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("suite"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--hyp", default=None); ap.add_argument("--json", default=None)
    args = ap.parse_args()
    order = [json.loads(l)["id"] for l in open(os.path.join(DS, FILES[args.suite])) if l.strip()]
    if args.limit: order = order[:args.limit]
    A, B = load(args.a, args.suite), load(args.b, args.suite)
    ids = [i for i in order if i in A and i in B]
    missing = len(order) - len(ids)
    ok = [i for i in ids if not A[i].get("error") and not B[i].get("error")]
    errs = len(ids) - len(ok)
    n = len(ok)
    if n == 0: print("no paired items"); return
    accA = sum(1 for i in ok if A[i]["correct"]); accB = sum(1 for i in ok if B[i]["correct"])
    b = sum(1 for i in ok if A[i]["correct"] and not B[i]["correct"]); c = sum(1 for i in ok if B[i]["correct"] and not A[i]["correct"])
    p = mcnemar(b, c)
    same = [i for i in ok if A[i].get("content_sha256") and A[i].get("content_sha256") == B[i].get("content_sha256")]
    have_sha = sum(1 for i in ok if A[i].get("content_sha256") and B[i].get("content_sha256"))
    same_tok = sum(1 for i in ok if A[i].get("completion_tokens") and A[i].get("completion_tokens") == B[i].get("completion_tokens"))
    tokA = [A[i]["completion_tokens"] for i in ok if A[i].get("completion_tokens")]; tokB = [B[i]["completion_tokens"] for i in ok if B[i].get("completion_tokens")]
    wA = [A[i]["wall_s"] for i in ok]; wB = [B[i]["wall_s"] for i in ok]
    trA = sum(1 for i in ok if A[i].get("truncated")); trB = sum(1 for i in ok if B[i].get("truncated"))
    dacc = 100 * (accB - accA) / n
    loA, hiA = wilson(accA, n); loB, hiB = wilson(accB, n)
    out = {"suite": args.suite, "A": args.a, "B": args.b, "n": n, "errors": errs, "missing": missing,
           "accA": round(100 * accA / n, 2), "accA_ci": [round(100 * loA, 1), round(100 * hiA, 1)],
           "accB": round(100 * accB / n, 2), "accB_ci": [round(100 * loB, 1), round(100 * hiB, 1)],
           "delta_acc_pts": round(dacc, 2), "A_only_correct": b, "B_only_correct": c, "mcnemar_p": round(p, 4),
           "identity_rate": round(len(same) / have_sha, 4) if have_sha else None, "identity_ci": [round(x, 3) for x in wilson(len(same), have_sha)] if have_sha else None,
           "token_identity_rate": round(same_tok / n, 4), "mean_tok_A": round(statistics.mean(tokA), 1) if tokA else None,
           "mean_tok_B": round(statistics.mean(tokB), 1) if tokB else None, "mean_wall_A": round(statistics.mean(wA), 2), "mean_wall_B": round(statistics.mean(wB), 2),
           "wall_ratio_A_over_B": round(statistics.mean(wA) / statistics.mean(wB), 3) if statistics.mean(wB) > 0 else None, "truncated_A": trA, "truncated_B": trB}
    print(f"{args.suite}: n={n} (errors {errs}, missing {missing})   A={args.a}  B={args.b}")
    print(f"  accuracy A {out['accA']}% {out['accA_ci']}   B {out['accB']}% {out['accB_ci']}   delta(B-A) {dacc:+.2f} pts")
    print(f"  discordant: A-only-correct {b}, B-only-correct {c}   exact McNemar p = {p:.4f}")
    if have_sha: print(f"  output identity (sha256): {len(same)}/{have_sha} = {out['identity_rate']:.3f} {out['identity_ci']}   token-count identity {out['token_identity_rate']:.3f}")
    print(f"  mean tokens A {out['mean_tok_A']} B {out['mean_tok_B']}   mean wall A {out['mean_wall_A']} s  B {out['mean_wall_B']} s  (A/B {out['wall_ratio_A_over_B']})   truncated A {trA} B {trB}")
    if args.hyp in CRITERIA:
        cr = CRITERIA[args.hyp]; verdict = []
        if cr["identity_min"] is not None and have_sha:
            r = len(same) / have_sha
            verdict.append("identity SUPPORTED" if r >= cr["identity_min"] else ("identity FALSIFIED (<%.2f)" % cr["identity_falsify"] if r < cr["identity_falsify"] else "identity WEAK (accuracy-lossless at best)"))
        if abs(dacc) > cr["dacc_falsify"] and p < 0.05 and dacc < 0: verdict.append("accuracy FALSIFIED (reference ahead beyond the falsifier)")
        elif abs(dacc) <= cr["dacc_max"] and p >= 0.05: verdict.append("accuracy SUPPORTED (within pre-registered band)")
        else: verdict.append("accuracy INCONCLUSIVE (outside the band but not the falsifier; report as measured)")
        out["verdict"] = verdict; print(f"  {args.hyp.upper()} verdict: " + "; ".join(verdict))
    if args.json:
        with open(args.json, "w") as f: json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
