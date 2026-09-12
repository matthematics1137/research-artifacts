#!/usr/bin/env python3
"""H38a follow-up (no GPU): what do the non-identical temperature-0 pairs look like? From the per-item eval rows
(hash, token count, correctness, last-200-char excerpt) of two labels on one suite: how many non-identical pairs still
end in the same final answer, how their lengths differ (same length = token swaps; large deltas = a different
reasoning path), and whether disagreement clusters in long traces.
usage: identity_anatomy.py <suite> <labelA> <labelB> [--json out.json]"""
import argparse, json, os, re, statistics

RES = "/path/to/lab-repo/testsuite/evals/results"
ANS = re.compile(r"Answer:\s*(.+?)\s*$", re.I | re.M)


def load(label, suite):
    p = os.path.join(RES, label, f"{suite}.jsonl")
    return {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def final_answer(r):
    m = ANS.findall(r.get("raw_answer_excerpt") or "")
    return m[-1].strip().rstrip(".").replace("*", "").replace("$", "").replace(",", "") if m else None


def bucket(d):
    d = abs(d)
    return "0" if d == 0 else "1-5" if d <= 5 else "6-50" if d <= 50 else ">50"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("suite"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    A, B = load(a.a, a.suite), load(a.b, a.suite)
    ids = [i for i in A if i in B and not A[i].get("error") and not B[i].get("error") and A[i].get("content_sha256") and B[i].get("content_sha256")]
    same = [i for i in ids if A[i]["content_sha256"] == B[i]["content_sha256"]]
    diff = [i for i in ids if i not in set(same)]
    out = {"suite": a.suite, "A": a.a, "B": a.b, "n": len(ids), "identical": len(same), "non_identical": len(diff)}
    if diff:
        fa = [(final_answer(A[i]), final_answer(B[i])) for i in diff]
        both = [(x, y) for x, y in fa if x is not None and y is not None]
        out["non_identical_same_final_answer"] = sum(1 for x, y in both if x == y); out["non_identical_with_both_answers_parsed"] = len(both)
        out["non_identical_correctness_agree"] = sum(1 for i in diff if A[i]["correct"] == B[i]["correct"])
        dt = [(B[i]["completion_tokens"] or 0) - (A[i]["completion_tokens"] or 0) for i in diff]
        out["token_delta_buckets"] = {k: sum(1 for d in dt if bucket(d) == k) for k in ("0", "1-5", "6-50", ">50")}
        out["token_delta_median"] = statistics.median(dt); out["token_delta_mean"] = round(statistics.mean(dt), 1)
        lenA_same = [A[i]["completion_tokens"] for i in same if A[i].get("completion_tokens")]
        lenA_diff = [A[i]["completion_tokens"] for i in diff if A[i].get("completion_tokens")]
        out["ref_len_median_identical"] = statistics.median(lenA_same) if lenA_same else None
        out["ref_len_median_non_identical"] = statistics.median(lenA_diff) if lenA_diff else None
        # identity rate by reference-length tercile
        allA = sorted((A[i]["completion_tokens"], i) for i in ids if A[i].get("completion_tokens"))
        t = len(allA) // 3; terc = {"short": allA[:t], "mid": allA[t:2 * t], "long": allA[2 * t:]}
        out["identity_by_ref_length_tercile"] = {k: {"n": len(v), "max_tokens": v[-1][0] if v else None, "identity": round(sum(1 for _, i in v if i in set(same)) / len(v), 3) if v else None} for k, v in terc.items()}
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
