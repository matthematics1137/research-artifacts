#!/usr/bin/env python3
"""Paper 8 output quality: pair the 2080 Ti (Turing) eval rows with the laptop's stored rows, item by item.

Pair statistics are hardening_analyze.py's (Paper 7, E47), computed the same way on the dataset order with error
rows excluded pairwise: accuracy with Wilson 95% CIs, discordant items with the exact two-sided McNemar p, byte
identity of content_sha256 (reasoning + answer) with its Wilson CI, token-count identity, mean tokens, mean wall
per item and its A/B ratio, truncations. check_fidelity.py proves the equality by reproducing every stored
hardening_analyze.py output (testsuite/results/efficiency/rows/h38*.json) from the laptop rows.

Added on top, for the "quality of the token output" question:
  - identity_anatomy.py's breakdown of the non-identical pairs (same final answer, token-count deltas, identity by
    reference-length tercile); same definitions, also reproduced by check_fidelity.py;
  - same final answer over all pairs, the discordant item ids, thinking presence and chat_template_kwargs drops on
    each side (a mismatch there means the two sides did not run the same request semantics);
  - token counts on a matched basis: the laptop's EXL3 rows counted tokens with TabbyAPI's /v1/token/encode
    (the lab TabbyAPI returned usage: null) while upstream TabbyAPI reports usage; the Turing sidecar
    (<suite>.full.jsonl from run_eval_turing.py) carries the encode count so the two can be matched. When the
    two sides count differently, the anatomy's token deltas and the mean tokens use the matched counts too, and
    the raw (hardening_analyze) token fields are marked "mixed basis" in the markdown;
  - response shape: E47's server returned the reasoning inline in content ('...\\n</think>\\n\\nAnswer'); upstream
    TabbyAPI and llama-server split it into reasoning_content and drop the tag, which changes content_sha256 for
    the same tokens and the scoring of items truncated inside the reasoning. When one side is inline and the
    other split, byte identity and the anatomy use the sidecar's inline fingerprint, the inline-path scores are
    reported, and a CHECK line says so (run_quality.sh serves E47's shape on every lane, so this is a fallback);
  - a cross-lane table on Turing: accuracy, accuracy on the items every stock lane completed (TURING-<lane>
    labels only: reruns and variants are scored on that set but never shrink it), mean tokens, truncations,
    wall per item, accuracy per minute (run_eval.py's definition: accuracy / (mean wall / 60)), the response
    shape, and, from the server timings in the sidecar, the median decode tok/s and the pooled draft acceptance.

usage: compare_quality.py [--turing-dir D] [--laptop-dir D] [--datasets-dir D] [--runs F]
                          [--suites gsm8k_500,mmlu_500] [--pair SUITE:A:B[:LIMIT]]... [--no-default-pairs]
                          [--md OUT.md] [--json OUT.json]
  A is the reference side (the laptop, or the undrafted lane), B the compared side; delta = B - A.
  Labels starting with TURING- are looked up in --turing-dir first, all others in --laptop-dir first.
  Without --md the markdown goes to stdout. A rows file may hold several rows per id (resumed runs): the latest
  row wins, as in hardening_analyze.py.
Stdlib only.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LAB = os.environ.get("TURING_LAB", "/opt/p8")
FILES = {"gsm8k_500": "gsm8k_500.jsonl", "mmlu_500": "mmlu_500.jsonl", "math500": "math500.jsonl",
         "gsm8k": "gsm8k_100.jsonl", "mmlu": "mmlu_200.jsonl", "math25": "math25.jsonl"}
SUITES = ["gsm8k_500", "mmlu_500"]
LANES = ["exl3-mtp", "exl3-nodraft", "lcpp-iq2xxs", "lcpp-iq2s", "lcpp-iq2s-mtp", "lcpp-iq2xxs-dflash2"]
LCPP_LANES = [x for x in LANES if x.startswith("lcpp-")]
ANS = re.compile(r"Answer:\s*(.+?)\s*$", re.I | re.M)   # identity_anatomy.py's final-answer pattern
# How the laptop's stored rows counted completion_tokens (rows carry no marker; from the campaign record):
# the lab TabbyAPI returned usage: null, so run_eval.py re-encoded the text; llama-server reports usage.
KNOWN_TOKEN_BASIS = {
    "EFF-h38a_mtp": "encode", "EFF-h38a_nodraft": "encode", "EFF-h38c_mtp_rerun": "encode",
    "EFF-h38c_nodraft_rerun": "encode", "EFF-h38d_mtp_clean1": "encode", "EFF-h38d_nodraft_clean1": "encode",
    "EFF-h38d_nodraft_clean2": "encode", "EFF-h38e_nodraft_dev1": "encode", "EFF-h38e_nodraft_dev2": "encode",
    "EFF-h38f_nodraft_rel1": "encode", "EFF-h38f_nodraft_rel2": "encode", "EFF-h38b_q4kxl_mtp": "usage",
}
# How the laptop's stored rows' servers returned the reasoning (rows carry no marker). The lab TabbyAPI (4a4f9f4,
# reasoning parser off by default, no config set it) returned it inline in content (rows/traces_h38c_nodraft.jsonl:
# reasoning "" and content '...\n</think>\n\n...'); llama-server v0.4.0 splits it (--reasoning-format deepseek, its
# default, not overridden by the laptop's scripts).
KNOWN_RESPONSE_SHAPE = {lab: ("split" if b == "usage" else "inline") for lab, b in KNOWN_TOKEN_BASIS.items()}
# Facts about the laptop side of the default pairs (testsuite/results/EFFICIENCY-CAMPAIGN.md E47-E52).
LAPTOP_NOTES = [
    "Laptop E47 rows (EFF-h38a_*): RTX 4080 Laptop 12 GB at 80 W, exllamav3 1.4.7 with the lab patches, lab "
    "TabbyAPI, daily config (32K cache, Q8, chunk 1024, max_batch_size 1, MTP draft 6 dynamic), mixed CUDA "
    "library set (E50), 2026-09-09.",
    "exllamav3 1.4.7's greedy decoding is not bit-reproducible on this model: the same undrafted config agreed "
    "with itself on 50-64 of 100 GSM8K items (E49/E50). Byte identity against E47 is therefore bounded by the "
    "laptop's own run-to-run identity and measures numerics, not correctness; the replication test is the "
    "paired accuracy (McNemar) and the same-final-answer rate.",
    "exllamav3 1.5.0 made greedy decoding reproducible on the laptop (E52: 100/100). EFF-h38f_nodraft_rel1 is "
    "that engine (first 100 GSM8K-500 items), the closest laptop reference for byte identity with 1.6.5.",
    "Token counts: the laptop's TabbyAPI rows were re-encoded with /v1/token/encode (the text plus BOS); upstream "
    "TabbyAPI and llama-server report usage (generated tokens). Where the two sides differ, token identity, mean "
    "tokens and the anatomy's token deltas use the Turing sidecar's encode counts, so both sides share one basis; "
    "the raw hardening_analyze values are shown as 'mixed basis'.",
    "Response shape: the laptop's TabbyAPI returned the reasoning inline in content, with the '</think>' tag. "
    "run_quality.sh serves the same shape on every Turing lane (TabbyAPI reasoning parser and tool parsing off, "
    "llama-server --reasoning-format none). If a lane split the reasoning anyway, byte identity uses the sidecar's "
    "inline fingerprint and the pair gets a CHECK line; without that, a split side shows identity near 0 whatever "
    "it generated. The byte-identity bound by the laptop's run-to-run identity applies only when the shapes match.",
    "EFF-h38b_q4kxl_mtp: UD-Q4_K_XL GGUF, llama.cpp v0.4.0-mtpshared, 30 of 64 layers on the laptop GPU, shared "
    "MTP sidecar, 8K context; first 200 GSM8K-500 items (E48). A 4-bit reference, not a replication.",
]


def first_existing(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return cands[0]


# ------------------------------------------------------------------ statistics (verbatim from hardening_analyze.py)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


# ------------------------------------------------------------------ loading

class Store:
    """Rows and sidecars by label; TURING-* labels resolve in the Turing dir first, others in the laptop dir."""

    def __init__(self, turing_dir, laptop_dir):
        self.turing_dir, self.laptop_dir = turing_dir, laptop_dir
        self._rows, self._full, self.meta = {}, {}, {}

    def path(self, label, suite, ext=".jsonl"):
        dirs = [self.turing_dir, self.laptop_dir] if label.startswith("TURING-") else [self.laptop_dir, self.turing_dir]
        for d in dirs:
            p = os.path.join(d, label, suite + ext)
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _read(path):
        out, lines, torn = {}, 0, 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                lines += 1
                try:
                    r = json.loads(line)
                except ValueError:
                    torn += 1
                    continue
                if isinstance(r, dict) and "id" in r:
                    out[r["id"]] = r
        return out, lines, torn

    def rows(self, label, suite):
        key = (label, suite)
        if key not in self._rows:
            p = self.path(label, suite)
            if p is None:
                self._rows[key] = None
            else:
                r, lines, torn = self._read(p)
                self._rows[key] = r
                self.meta[key] = {"path": p, "lines": lines, "unique_ids": len(r), "torn_lines": torn}
        return self._rows[key]

    def full(self, label, suite):
        key = (label, suite)
        if key not in self._full:
            p = self.path(label, suite, ".full.jsonl")
            self._full[key] = self._read(p)[0] if p else {}
        return self._full[key]

    def turing_labels(self):
        if not os.path.isdir(self.turing_dir):
            return []
        return sorted(d for d in os.listdir(self.turing_dir)
                      if d.startswith("TURING-") and os.path.isdir(os.path.join(self.turing_dir, d)))


def dataset_order(datasets_dir, suite):
    with open(os.path.join(datasets_dir, FILES[suite]), encoding="utf-8") as f:
        return [json.loads(l)["id"] for l in f if l.strip()]


# ------------------------------------------------------------------ pair statistics

def final_answer(r):
    """identity_anatomy.py's final answer: last 'Answer: ...' line of the stored excerpt, lightly normalized."""
    m = ANS.findall(r.get("raw_answer_excerpt") or "")
    return m[-1].strip().rstrip(".").replace("*", "").replace("$", "").replace(",", "") if m else None


def bucket(d):
    d = abs(d)
    return "0" if d == 0 else "1-5" if d <= 5 else "6-50" if d <= 50 else ">50"


def anatomy(A, B, ids, fpA=None, fpB=None, tokA=None, tokB=None):
    """identity_anatomy.py on the given ids (error-free pairs where both sides carry content_sha256).

    With the defaults this is identity_anatomy.py exactly (check_fidelity.py reproduces its stored outputs). fpA/fpB
    (id -> fingerprint) replace content_sha256 when the two sides returned the text in different shapes; tokA/tokB
    (id -> token count) replace completion_tokens when the two sides counted tokens differently: deltas and
    reference lengths then use only ids present in both maps, and the output says which basis it used."""
    if fpA is None:
        same = [i for i in ids if A[i]["content_sha256"] == B[i]["content_sha256"]]
    else:
        same = [i for i in ids if fpA[i] == fpB[i]]
    sset = set(same)
    diff = [i for i in ids if i not in sset]
    out = {"n": len(ids), "identical": len(same), "non_identical": len(diff)}
    if fpA is not None:
        out["identity_basis"] = "inline (E47 shape)"
    if tokA is not None:
        out["token_basis"] = "encode (matched)"
    if diff:
        fa = [(final_answer(A[i]), final_answer(B[i])) for i in diff]
        both = [(x, y) for x, y in fa if x is not None and y is not None]
        out["non_identical_same_final_answer"] = sum(1 for x, y in both if x == y)
        out["non_identical_with_both_answers_parsed"] = len(both)
        out["non_identical_correctness_agree"] = sum(1 for i in diff if A[i]["correct"] == B[i]["correct"])
        if tokA is None:
            dt = [(B[i]["completion_tokens"] or 0) - (A[i]["completion_tokens"] or 0) for i in diff]
            lenA = {i: A[i]["completion_tokens"] for i in ids if A[i].get("completion_tokens")}
        else:
            dt = [tokB[i] - tokA[i] for i in diff if i in tokA and i in tokB]
            lenA = {i: tokA[i] for i in ids if i in tokA and i in tokB and tokA[i]}
            out["token_delta_n"] = len(dt)
        out["token_delta_buckets"] = {k: sum(1 for d in dt if bucket(d) == k) for k in ("0", "1-5", "6-50", ">50")}
        out["token_delta_median"] = statistics.median(dt) if dt else None
        out["token_delta_mean"] = round(statistics.mean(dt), 1) if dt else None
        lenA_same = [lenA[i] for i in same if i in lenA]
        lenA_diff = [lenA[i] for i in diff if i in lenA]
        out["ref_len_median_identical"] = statistics.median(lenA_same) if lenA_same else None
        out["ref_len_median_non_identical"] = statistics.median(lenA_diff) if lenA_diff else None
        allA = sorted((lenA[i], i) for i in ids if i in lenA)
        t = len(allA) // 3; terc = {"short": allA[:t], "mid": allA[t:2 * t], "long": allA[2 * t:]}
        out["identity_by_ref_length_tercile"] = {k: {"n": len(v), "max_tokens": v[-1][0] if v else None, "identity": round(sum(1 for _, i in v if i in sset) / len(v), 3) if v else None} for k, v in terc.items()}
    return out


def token_basis(label, full, ids):
    """'usage' or 'encode': how this side's row completion_tokens were counted ('unknown' if not recorded)."""
    src = [full[i].get("completion_tokens_source") for i in ids if i in full]
    src = [s for s in src if s]
    if src:
        return max(sorted(set(src)), key=src.count)
    return KNOWN_TOKEN_BASIS.get(label, "unknown")


def _sidecar_shape(f):
    return "split" if f.get("reasoning_content") else "inline"


def encode_counts(rows, full, ids, basis):
    """Per-id token counts on the laptop's basis: /v1/token/encode of the inline text E47's server returned.

    A split response's sidecar carries inline_encode_tokens (the reconstructed inline text); its encode_tokens
    counts the text without the '</think>' tag, one token short, so it is not used. An inline sidecar's
    encode_tokens, or the row itself when it was counted with encode, is the laptop's basis already."""
    out = {}
    for i in ids:
        f = full.get(i) or {}
        if f.get("inline_encode_tokens"):
            out[i] = f["inline_encode_tokens"]
        elif f and _sidecar_shape(f) == "split":
            continue
        elif f.get("encode_tokens"):
            out[i] = f["encode_tokens"]
        elif basis == "encode" and rows[i].get("completion_tokens"):
            out[i] = rows[i]["completion_tokens"]
    return out


def response_shape(label, full, ids):
    """'inline' (reasoning inline in content, as E47's server returned it), 'split' (reasoning_content separate)
    or 'unknown' (no sidecar and not a known laptop label). From the sidecar where there is one."""
    fs = [full[i] for i in ids if i in full and "error" not in full[i]]
    if fs:
        return "split" if any(_sidecar_shape(f) == "split" for f in fs) else "inline"
    return KNOWN_RESPONSE_SHAPE.get(label, "unknown")


def inline_fingerprints(rows, full, ids, shape):
    """Per-id sha256 of the text in E47's (inline) shape: the sidecar's inline_sha256 where present, else the row's
    content_sha256 when this side is inline. Ids with neither are left out."""
    out = {}
    for i in ids:
        f = full.get(i) or {}
        if f.get("inline_sha256"):
            out[i] = f["inline_sha256"]
        elif shape == "inline" and rows[i].get("content_sha256"):
            out[i] = rows[i]["content_sha256"]
    return out


def tool_call_items(full, ids):
    return sum(1 for i in ids if (full.get(i) or {}).get("tool_calls") or (full.get(i) or {}).get("finish_reason") == "tool_calls")


def pair_stats(suite, label_a, label_b, A, B, order, limit=0, FA=None, FB=None):
    """hardening_analyze.py's output dict (same keys, same rounding) plus an 'extra' dict; None if nothing pairs."""
    if limit: order = order[:limit]
    ids = [i for i in order if i in A and i in B]
    missing = len(order) - len(ids)
    ok = [i for i in ids if not A[i].get("error") and not B[i].get("error")]
    errs = len(ids) - len(ok)
    n = len(ok)
    if n == 0: return None
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
    out = {"suite": suite, "A": label_a, "B": label_b, "n": n, "errors": errs, "missing": missing,
           "accA": round(100 * accA / n, 2), "accA_ci": [round(100 * loA, 1), round(100 * hiA, 1)],
           "accB": round(100 * accB / n, 2), "accB_ci": [round(100 * loB, 1), round(100 * hiB, 1)],
           "delta_acc_pts": round(dacc, 2), "A_only_correct": b, "B_only_correct": c, "mcnemar_p": round(p, 4),
           "identity_rate": round(len(same) / have_sha, 4) if have_sha else None, "identity_ci": [round(x, 3) for x in wilson(len(same), have_sha)] if have_sha else None,
           "token_identity_rate": round(same_tok / n, 4), "mean_tok_A": round(statistics.mean(tokA), 1) if tokA else None,
           "mean_tok_B": round(statistics.mean(tokB), 1) if tokB else None, "mean_wall_A": round(statistics.mean(wA), 2), "mean_wall_B": round(statistics.mean(wB), 2),
           "wall_ratio_A_over_B": round(statistics.mean(wA) / statistics.mean(wB), 3) if statistics.mean(wB) > 0 else None, "truncated_A": trA, "truncated_B": trB}

    # ---- additions (not in hardening_analyze.py)
    FA, FB = FA or {}, FB or {}
    fa = {i: (final_answer(A[i]), final_answer(B[i])) for i in ok}
    both = [i for i in ok if fa[i][0] is not None and fa[i][1] is not None]
    sha_ids = [i for i in ok if A[i].get("content_sha256") and B[i].get("content_sha256")]
    basis_a, basis_b = token_basis(label_a, FA, ok), token_basis(label_b, FB, ok)
    shape_a, shape_b = response_shape(label_a, FA, ok), response_shape(label_b, FB, ok)
    ex = {
        "limit": limit or None,
        "pairs_total": len(ids),
        "A_only_ids": [i for i in ok if A[i]["correct"] and not B[i]["correct"]],
        "B_only_ids": [i for i in ok if B[i]["correct"] and not A[i]["correct"]],
        "same_final_answer": sum(1 for i in both if fa[i][0] == fa[i][1]),
        "both_answers_parsed": len(both),
        "same_final_answer_rate": round(sum(1 for i in both if fa[i][0] == fa[i][1]) / len(both), 4) if both else None,
        "thinking_share_A": round(sum(1 for i in ok if (A[i].get("thinking_tokens_estimate") or 0) > 0) / n, 4),
        "thinking_share_B": round(sum(1 for i in ok if (B[i].get("thinking_tokens_estimate") or 0) > 0) / n, 4),
        "ctk_dropped_A": sum(1 for i in ok if A[i].get("chat_template_kwargs_dropped")),
        "ctk_dropped_B": sum(1 for i in ok if B[i].get("chat_template_kwargs_dropped")),
        "token_basis_A": basis_a, "token_basis_B": basis_b,
        "response_shape_A": shape_a, "response_shape_B": shape_b,
        "tool_call_items_A": tool_call_items(FA, ok), "tool_call_items_B": tool_call_items(FB, ok),
    }

    # Response shape: one side inline (E47's server), the other split -> compare the inline text on both sides.
    fpA = fpB = None
    an_ids = sha_ids
    if {shape_a, shape_b} == {"inline", "split"}:
        ia, ib = inline_fingerprints(A, FA, ok, shape_a), inline_fingerprints(B, FB, ok, shape_b)
        common = [i for i in ok if i in ia and i in ib]
        ex["shape_matched_n"] = len(common)
        if common:
            k = sum(1 for i in common if ia[i] == ib[i])
            ex["identity_basis"] = "inline (E47 shape)"
            ex["identity_rate_shape_matched"] = round(k / len(common), 4)
            ex["identity_ci_shape_matched"] = [round(x, 3) for x in wilson(k, len(common))]
            fpA, fpB, an_ids = ia, ib, common
        # Scoring: the split side's rows scored its content alone; E47 scored the inline text (an item truncated
        # inside the reasoning has no content when split, but E47's fallbacks scored its reasoning text).
        cor = {"A": {i: bool(A[i]["correct"]) for i in ok}, "B": {i: bool(B[i]["correct"]) for i in ok}}
        complete = True
        for side, F, shp in (("A", FA, shape_a), ("B", FB, shape_b)):
            if shp != "split":
                continue
            ic = {i: F[i]["inline_correct"] for i in ok if i in F and F[i].get("inline_correct") is not None}
            ex[f"inline_scored_{side}"] = len(ic)
            ex[f"inline_path_differs_{side}"] = [i for i in ok if i in ic and ic[i] != cor[side][i]]
            complete = complete and len(ic) == n
            cor[side].update(ic)
        if complete:
            b2 = sum(1 for i in ok if cor["A"][i] and not cor["B"][i]); c2 = sum(1 for i in ok if cor["B"][i] and not cor["A"][i])
            ka, kb = sum(cor["A"].values()), sum(cor["B"].values())
            ex["inline_path"] = {"accA": round(100 * ka / n, 2), "accB": round(100 * kb / n, 2),
                                 "delta_acc_pts": round(100 * (kb - ka) / n, 2), "A_only_correct": b2,
                                 "B_only_correct": c2, "mcnemar_p": round(mcnemar(b2, c2), 4)}

    # Token counts: one basis for both sides, also for the anatomy's deltas and reference lengths.
    tA = tB = None
    if basis_a == basis_b and basis_a != "unknown":
        ex["token_identity_rate_matched"] = out["token_identity_rate"]
        ex["matched_basis"] = basis_a
    else:
        ea, eb = encode_counts(A, FA, ok, basis_a), encode_counts(B, FB, ok, basis_b)
        common = [i for i in ok if i in ea and i in eb]
        if common:
            ex["matched_basis"] = "encode"
            ex["matched_n"] = len(common)
            ex["token_identity_rate_matched"] = round(sum(1 for i in common if ea[i] == eb[i]) / len(common), 4)
            ex["mean_tok_A_matched"] = round(statistics.mean(ea[i] for i in common), 1)
            ex["mean_tok_B_matched"] = round(statistics.mean(eb[i] for i in common), 1)
            tA, tB = ea, eb
        else:
            ex["matched_basis"] = None
    ex["anatomy"] = anatomy(A, B, an_ids, fpA, fpB, tA, tB)
    if basis_a != basis_b and tA is None:
        ex["anatomy"]["token_basis"] = f"mixed ({basis_a} vs {basis_b})"
    out["extra"] = ex
    return out


# ------------------------------------------------------------------ Turing lanes

def _median(xs):
    return round(statistics.median(xs), 1) if xs else None


def lane_summary(label, rows, full, order):
    present = [i for i in order if i in rows]
    ok = [i for i in present if not rows[i].get("error")]
    n = len(ok)
    s = {"label": label, "planned": len(order), "rows": len(present), "errors": len(present) - n, "n": n}
    if n == 0:
        return s
    k = sum(1 for i in ok if rows[i]["correct"])
    lo, hi = wilson(k, n)
    toks = [rows[i]["completion_tokens"] for i in ok if rows[i].get("completion_tokens")]
    walls = [rows[i]["wall_s"] for i in ok]
    mw = statistics.mean(walls)
    s.update({
        "correct": k, "acc": round(100 * k / n, 2), "acc_ci": [round(100 * lo, 1), round(100 * hi, 1)],
        "mean_tok": round(statistics.mean(toks), 1) if toks else None,
        "truncated": sum(1 for i in ok if rows[i].get("truncated")),
        "mean_wall": round(mw, 2), "total_wall_h": round(sum(walls) / 3600, 2),
        "acc_per_min": round((k / n) / (mw / 60), 4) if mw > 0 else None,
        "thinking_share": round(sum(1 for i in ok if (rows[i].get("thinking_tokens_estimate") or 0) > 0) / n, 4),
        "ctk_dropped": sum(1 for i in ok if rows[i].get("chat_template_kwargs_dropped")),
    })
    tm = [(full.get(i) or {}).get("timings") or {} for i in ok]
    dec = [t["predicted_per_second"] for t in tm if t.get("predicted_per_second")]
    pre = [t["prompt_per_second"] for t in tm if t.get("prompt_per_second")]
    dn = sum(t.get("draft_n") or 0 for t in tm); da = sum(t.get("draft_n_accepted") or 0 for t in tm)
    s.update({"decode_tps_median": _median(dec), "prefill_tps_median": _median(pre),
              "draft_acceptance": round(da / dn, 4) if dn else None, "timed_items": len(dec),
              "token_basis": token_basis(label, full, ok), "response_shape": response_shape(label, full, ok),
              "tool_call_items": tool_call_items(full, ok)})
    return s


def lane_table(store, suite, labels, order):
    """One summary per label, plus accuracy on the common items: the items every stock lane (TURING-<lane>, no
    variant or rerun suffix) completed without error. Reruns and variants (often partial, e.g. the first 100 items)
    are scored on their overlap with that set and never shrink it."""
    out = []
    oks = {}
    for lab in labels:
        rows = store.rows(lab, suite)
        if rows is None:
            continue
        s = lane_summary(lab, rows, store.full(lab, suite), order)
        out.append(s)
        oks[lab] = {i for i in order if i in rows and not rows[i].get("error")}
    stock = []
    for lab in oks:
        lane, var = split_label(lab)
        if lane and var == "" and oks[lab]:
            stock.append(lab)
    if len(stock) > 1:
        common = set.intersection(*(oks[lab] for lab in stock))
        limited = [lab for lab in stock if len(oks[lab]) < len(order)]
        for s in out:
            rows = store.rows(s["label"], suite)
            mine = common & oks[s["label"]]
            s["common_set_n"] = len(common)
            s["common_n"] = len(mine)
            s["in_common_set"] = s["label"] in stock
            s["common_limited_by"] = limited
            s["acc_common"] = round(100 * sum(1 for i in mine if rows[i]["correct"]) / len(mine), 2) if mine else None
    return out


LAPTOP_TWIN = {"exl3-mtp": "EFF-h38a_mtp", "exl3-nodraft": "EFF-h38a_nodraft"}   # the E47 rows each lane mirrors


def split_label(lab):
    """TURING-<lane>[-<variant>] -> (lane, variant), longest known lane first; (None, None) if no lane matches."""
    if not lab.startswith("TURING-"):
        return None, None
    rest = lab[len("TURING-"):]
    if rest in LANES:
        return rest, ""
    for ln in sorted(LANES, key=len, reverse=True):
        if rest.startswith(ln + "-"):
            return ln, rest[len(ln) + 1:]
    return None, None


def lane_order(labels):
    def key(lab):
        lane, var = split_label(lab)
        return (LANES.index(lane) if lane else len(LANES), lane or lab, var or "")
    return sorted(labels, key=key)


# ------------------------------------------------------------------ the default comparison plan

def default_pairs(turing_labels, suites):
    P = []

    def add(group, suite, a, b, limit=0, note=""):
        if suite in suites:
            P.append({"group": group, "suite": suite, "A": a, "B": b, "limit": limit, "note": note})

    for s in SUITES:
        add("replication", s, "EFF-h38a_mtp", "TURING-exl3-mtp", 0,
            "E47 drafted daily config: laptop 4080 (exllamav3 1.4.7-lab) vs 2080 Ti (1.6.5)")
        add("replication", s, "EFF-h38a_nodraft", "TURING-exl3-nodraft", 0, "E47 undrafted control: laptop vs 2080 Ti")
    add("engine-reference", "gsm8k_500", "EFF-h38f_nodraft_rel1", "TURING-exl3-nodraft", 100,
        "laptop exllamav3 1.5.0 (reproducible greedy, E52) vs 2080 Ti 1.6.5; first 100 items")
    add("engine-reference", "gsm8k_500", "EFF-h38f_nodraft_rel1", "TURING-exl3-mtp", 100,
        "laptop 1.5.0 undrafted vs 2080 Ti 1.6.5 drafted; first 100 items")
    for s in SUITES:
        add("turing-internal", s, "TURING-exl3-nodraft", "TURING-exl3-mtp", 0,
            "H38a's design on the 2080 Ti: is the MTP drafter lossless (and sample-identical) on 1.6.5?")
        add("turing-internal", s, "TURING-lcpp-iq2s", "TURING-lcpp-iq2s-mtp", 0, "llama.cpp shared MTP sidecar: lossless?")
        add("turing-internal", s, "TURING-lcpp-iq2xxs", "TURING-lcpp-iq2xxs-dflash2", 0, "DFlash2 drafter: lossless?")
        add("turing-internal", s, "TURING-exl3-nodraft", "TURING-lcpp-iq2s", 0, "EXL3 2.0 bpw vs UD-IQ2_S, undrafted, same card")
        add("turing-internal", s, "TURING-exl3-nodraft", "TURING-lcpp-iq2xxs", 0, "EXL3 2.0 bpw vs UD-IQ2_XXS, undrafted, same card")
        add("turing-internal", s, "TURING-exl3-mtp", "TURING-lcpp-iq2s-mtp", 0, "best drafted lanes: EXL3+MTP vs IQ2_S+MTP")
        add("turing-internal", s, "TURING-exl3-mtp", "TURING-lcpp-iq2xxs-dflash2", 0, "EXL3+MTP vs IQ2_XXS+DFlash2")
    for lane in LCPP_LANES:
        add("reference", "gsm8k_500", "EFF-h38b_q4kxl_mtp", f"TURING-{lane}", 200,
            "laptop UD-Q4_K_XL + shared MTP (E48), first 200 items: a 4-bit reference, not a replication")
    for lab in turing_labels:
        m = re.fullmatch(r"(TURING-.+)-r(\d+)", lab)
        if m and m.group(1) in turing_labels:
            for s in SUITES:
                add("reproducibility", s, m.group(1), lab, 0, "same lane, fresh server: greedy run-to-run identity on sm_75")
    for lab in turing_labels:   # variants, e.g. TURING-exl3-mtp-fp32 (TURING_LANE_ENV=TABBY_MODEL_RECURRENT_STATE_FP32=true)
        lane, var = split_label(lab)
        if not var or re.fullmatch(r"r\d+", var) or re.search(r"-r\d+$", var):
            continue
        for s in SUITES:
            add("variant", s, f"TURING-{lane}", lab, 0, f"variant '{var}' against the stock lane, same card")
            if lane in LAPTOP_TWIN:
                add("replication", s, LAPTOP_TWIN[lane], lab, 0,
                    f"E47 replication with variant '{var}' (plan 6.2 item 6: fp32 GDN state for laptop comparisons)")
    return P


def run_pairs(store, plan, orders):
    out = []
    for pr in plan:
        rec = dict(pr)
        s = pr["suite"]
        A, B = store.rows(pr["A"], s), store.rows(pr["B"], s)
        if A is None or B is None:
            rec["skipped"] = "no rows for " + " and ".join(x for x, r in ((pr["A"], A), (pr["B"], B)) if r is None)
        else:
            st = pair_stats(s, pr["A"], pr["B"], A, B, orders[s], pr["limit"], store.full(pr["A"], s), store.full(pr["B"], s))
            if st is None:
                rec["skipped"] = "no error-free paired items"
            else:
                rec["stats"] = st
        out.append(rec)
    return out


# ------------------------------------------------------------------ rendering

def _z(x):
    """Display only: hardening_analyze's Wilson bound can round to -0.0 at k = 0; show it as 0.0."""
    return x + 0.0 if isinstance(x, float) else x


def _ci(v, ci):
    return f"{v:.1f}% [{_z(ci[0])}, {_z(ci[1])}]" if v is not None else "—"


def _f(v, nd=1, suffix=""):
    if v is None:
        return "—"
    return f"{v:.{nd}f}{suffix}" if isinstance(v, float) else f"{v}{suffix}"


GROUP_TITLES = {
    "replication": "Replication of E47: 2080 Ti vs laptop, same config and suite",
    "engine-reference": "Engine reference: laptop exllamav3 1.5.0 (reproducible greedy) vs 2080 Ti",
    "turing-internal": "Pairs on the 2080 Ti itself",
    "reproducibility": "Reproducibility on the 2080 Ti (same lane, fresh server)",
    "variant": "Lane variants on the 2080 Ti (one setting changed against the stock lane)",
    "reference": "Reference: laptop 4-bit GGUF (E48) vs 2080 Ti lanes",
    "custom": "Requested pairs",
}


def render_md(res):
    L = []
    w = L.append
    w(f"# Paper 8 output quality: 2080 Ti lanes vs the laptop (generated {res['generated']})")
    w("")
    inp = res["inputs"]
    w(f"Turing rows: `{inp['turing_dir']}`. Laptop rows: `{inp['laptop_dir']}`. Datasets: `{inp['datasets_dir']}`.")
    w("Statistics as in `testsuite/results/efficiency/tools/hardening_analyze.py` (Wilson 95% CIs, exact two-sided "
      "McNemar on discordant pairs, sha256 identity of reasoning + answer); error rows excluded pairwise; latest row "
      "per id. A is the reference side, delta = B - A, wall ratio = A / B (above 1: B is faster). Where the two sides "
      "returned the text in different shapes (inline vs split reasoning), byte identity is computed on E47's inline "
      "text; where they counted tokens differently (encode vs usage), token identity and mean tokens are shown on the "
      "matched encode basis and the raw values are marked 'mixed basis'.")
    w("")
    for s, lanes in res["lanes"].items():
        w(f"## Turing lanes on {s}")
        w("")
        if not lanes:
            w("No Turing rows yet.")
            w("")
            continue
        cn = next((x.get("common_set_n") for x in lanes if x.get("common_set_n") is not None), None)
        w(f"| label | rows (err) | accuracy [95% CI] | acc. on common items (n={cn if cn is not None else '—'}) | "
          "mean tokens | truncated | wall/item s | acc./min | decode tok/s (median) | draft acceptance | thinking | "
          "ctk dropped | shape |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in lanes:
            if not x.get("n"):
                w(f"| {x['label']} | {x['rows']}/{x['planned']} ({x['errors']}) | — | — | — | — | — | — | — | — | — | — | — |")
                continue
            accc = _f(x.get("acc_common"), 1, "%")
            if x.get("common_n") is not None and x.get("common_n") != cn:
                accc += f" (n={x['common_n']})"
            shape = x.get("response_shape", "—") + (f", {x['tool_call_items']} tool calls" if x.get("tool_call_items") else "")
            w(f"| {x['label']} | {x['rows']}/{x['planned']} ({x['errors']}) | {_ci(x['acc'], x['acc_ci'])} | "
              f"{accc} | {_f(x['mean_tok'])} | {x['truncated']} | {_f(x['mean_wall'], 2)} | "
              f"{_f(x['acc_per_min'], 3)} | {_f(x['decode_tps_median'])} | {_f(x['draft_acceptance'], 3)} | "
              f"{_f(x['thinking_share'], 3)} | {x['ctk_dropped']} | {shape} |")
        w("")
        if cn is not None:
            lim = next((x.get("common_limited_by") for x in lanes if x.get("common_limited_by") is not None), [])
            w("Common items: those every stock lane (`TURING-<lane>`) completed without error; reruns and variants are "
              "scored on their overlap with that set (n shown when smaller) and do not shrink it."
              + (f" Stock lanes without an error-free row for every planned item, which limit the set: "
                 f"{', '.join(lim)}." if lim else ""))
            w("")
    groups = []
    for pr in res["pairs"]:
        if pr["group"] not in groups:
            groups.append(pr["group"])
    for g in groups:
        prs = [p for p in res["pairs"] if p["group"] == g]
        done = [p for p in prs if "stats" in p]
        w(f"## {GROUP_TITLES.get(g, g)}")
        w("")
        if done:
            w("| suite | A (reference) | B | n (err, missing) | acc. A [95% CI] | acc. B [95% CI] | delta pts | A-only / B-only | "
              "McNemar p | byte-identical [95% CI] | same final answer | token-identical (matched) | mean tokens A / B | "
              "wall/item A / B s (A/B) | truncated A / B |")
            w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for p in done:
                st, ex = p["stats"], p["stats"]["extra"]
                lim = f" first {p['limit']}" if p["limit"] else ""
                ident = (f"{st['identity_rate']:.3f} [{_z(st['identity_ci'][0])}, {_z(st['identity_ci'][1])}]"
                         if st["identity_rate"] is not None else "—")
                if ex.get("identity_rate_shape_matched") is not None:
                    ci = ex["identity_ci_shape_matched"]
                    ident = (f"{ex['identity_rate_shape_matched']:.3f} [{_z(ci[0])}, {_z(ci[1])}] inline basis, "
                             f"n={ex['shape_matched_n']} (raw {_f(st['identity_rate'], 3)}, shapes differ)")
                elif {ex.get("response_shape_A"), ex.get("response_shape_B")} == {"inline", "split"}:
                    ident += " (shapes differ: not comparable)"
                sfa = (f"{ex['same_final_answer']}/{ex['both_answers_parsed']}" if ex["both_answers_parsed"] else "—")
                tim = ex.get("token_identity_rate_matched")
                mixed = ex["token_basis_A"] != ex["token_basis_B"]
                mt = f"{_f(st['mean_tok_A'])} / {_f(st['mean_tok_B'])}"
                if not mixed:
                    tok = f"{st['token_identity_rate']:.3f}" + (f" ({ex['matched_basis']})" if ex.get("matched_basis") else "")
                elif tim is not None:
                    tok = (f"{tim:.3f} (encode, n={ex['matched_n']}; raw {st['token_identity_rate']:.3f} mixed basis)")
                    mt = (f"{ex['mean_tok_A_matched']} / {ex['mean_tok_B_matched']} (encode, matched; raw {mt} "
                          "mixed basis)")
                else:
                    tok = f"{st['token_identity_rate']:.3f} (mixed basis: {ex['token_basis_A']} vs {ex['token_basis_B']})"
                    mt += " (mixed basis)"
                w(f"| {st['suite']}{lim} | {st['A']} | {st['B']} | {st['n']} ({st['errors']}, {st['missing']}) | "
                  f"{_ci(st['accA'], st['accA_ci'])} | {_ci(st['accB'], st['accB_ci'])} | {st['delta_acc_pts']:+.2f} | "
                  f"{st['A_only_correct']} / {st['B_only_correct']} | {st['mcnemar_p']:.4f} | {ident} | {sfa} | {tok} | {mt} | "
                  f"{st['mean_wall_A']} / {st['mean_wall_B']} ({_f(st['wall_ratio_A_over_B'], 3)}) | {st['truncated_A']} / {st['truncated_B']} |")
            w("")
            for p in done:
                ex = p["stats"]["extra"]
                flags = []
                if abs(ex["thinking_share_A"] - ex["thinking_share_B"]) > 0.05:
                    flags.append(f"thinking share differs ({ex['thinking_share_A']} vs {ex['thinking_share_B']}): "
                                 "the two sides may not have run the same reasoning mode")
                if ex["ctk_dropped_A"] or ex["ctk_dropped_B"]:
                    flags.append(f"chat_template_kwargs dropped on {ex['ctk_dropped_A']} / {ex['ctk_dropped_B']} items")
                if ex["token_basis_A"] != ex["token_basis_B"] and ex.get("matched_basis") is None:
                    flags.append(f"token counts on different bases ({ex['token_basis_A']} vs {ex['token_basis_B']}) "
                                 "and no sidecar encode counts to match them")
                sa, sb = ex.get("response_shape_A"), ex.get("response_shape_B")
                if {sa, sb} == {"inline", "split"}:
                    m = (f"response shapes differ (A {sa}, B {sb}): the split side's server moved the reasoning to "
                         "reasoning_content and dropped '</think>', so its raw content_sha256 cannot match")
                    if ex.get("identity_rate_shape_matched") is not None:
                        m += (f"; byte identity and the anatomy use the inline text from the sidecar "
                              f"(n={ex['shape_matched_n']})")
                    else:
                        m += "; the split side's sidecar has no inline fingerprints, so byte identity is not comparable"
                    dif = ex.get("inline_path_differs_A", []) + ex.get("inline_path_differs_B", [])
                    ip = ex.get("inline_path")
                    if ip:
                        m += (f"; scored on E47's inline path: acc. A {ip['accA']}% / B {ip['accB']}% "
                              f"({ip['delta_acc_pts']:+.2f} pts, A-only / B-only {ip['A_only_correct']} / "
                              f"{ip['B_only_correct']}, McNemar p {ip['mcnemar_p']:.4f})")
                    if dif:
                        m += f"; {len(dif)} rows score differently on that path ({', '.join(dif[:8])}{' ...' if len(dif) > 8 else ''})"
                    if (p["A"] if sa == "split" else p["B"]).startswith("TURING-"):
                        m += "; rerun the split side with run_quality.sh's E47 shape for a clean comparison"
                    flags.append(m)
                elif "unknown" in (sa, sb):
                    side = " and ".join(x for x, s_ in (("A", sa), ("B", sb)) if s_ == "unknown")
                    flags.append(f"response shape unknown on {side} (no sidecar, not a known laptop label): byte "
                                 "identity may compare inline with split text")
                if ex.get("tool_call_items_A") or ex.get("tool_call_items_B"):
                    flags.append(f"the server parsed tool calls out of the text on {ex.get('tool_call_items_A')} / "
                                 f"{ex.get('tool_call_items_B')} items (E47's server parsed none)")
                note = f"- {p['suite']} {p['A']} vs {p['B']}: {p['note']}"
                if flags:
                    note += ". CHECK: " + "; ".join(flags)
                w(note)
            w("")
        skipped = [p for p in prs if "skipped" in p]
        if skipped:
            w("Not yet computable: " + "; ".join(f"{p['suite']} {p['A']} vs {p['B']} ({p['skipped']})" for p in skipped) + ".")
            w("")
    w("## Notes on the laptop side")
    w("")
    for n in LAPTOP_NOTES:
        w(f"- {n}")
    w("")
    if res.get("runs"):
        w("## Turing run records")
        w("")
        w("| label | suite | status | start | duration h | engine | VRAM after load | GPU (max temp, mean power, mean SM clock) |")
        w("|---|---|---|---|---|---|---|---|")
        for r in res["runs"]:
            eng = r.get("engine") or {}
            if not eng:
                engs = "—"
            elif str(r.get("lane", "")).startswith("exl3"):
                engs = f"exllamav3 {eng.get('exllamav3')}, TabbyAPI {str(eng.get('tabbyAPI'))[:12]}"
            else:
                engs = (f"llama.cpp {str(eng.get('llama_cpp'))[:12]} / "
                        f"mtpshared {str(eng.get('llama_cpp_v040_mtpshared'))[:12]}")
            g = r.get("gpu_samples") or {}
            gl = str(r.get("gpu_loaded") or "").split(",")
            vram = gl[3].strip() if len(gl) > 3 else "—"
            dur = r.get("duration_s")
            gpu = (f"{g.get('temp_max_c')} C, {g.get('power_mean_w')} W, {g.get('sm_clock_mean_mhz')} MHz"
                   if g.get("samples") else "—")
            w(f"| {r.get('label')} | {r.get('suite')} | {r.get('status')} | {r.get('t_start')} | "
              f"{round(dur / 3600, 2) if isinstance(dur, (int, float)) else '—'} | {engs} | {vram} | {gpu} |")
        w("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def parse_pair(s):
    parts = s.split(":")
    if len(parts) not in (3, 4) or parts[0] not in FILES:
        raise argparse.ArgumentTypeError("--pair wants SUITE:A:B[:LIMIT]")
    return {"group": "custom", "suite": parts[0], "A": parts[1], "B": parts[2],
            "limit": int(parts[3]) if len(parts) == 4 else 0, "note": "requested pair"}


def load_runs(path):
    out = []
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turing-dir", default=first_existing(os.path.join(LAB, "evals", "results"),
                                                           os.path.join(REPO, "testsuite", "results", "turing", "evals")))
    ap.add_argument("--laptop-dir", default=first_existing(os.path.join(REPO, "testsuite", "evals", "results"),
                                                           os.path.join(LAB, "evals", "laptop_results")))
    ap.add_argument("--datasets-dir", default=first_existing(os.path.join(REPO, "testsuite", "evals", "datasets"),
                                                             os.path.join(LAB, "evals", "datasets")))
    ap.add_argument("--runs", default=first_existing(os.path.join(LAB, "runs", "quality_runs.jsonl"),
                                                     os.path.join(REPO, "testsuite", "results", "turing", "quality_runs.jsonl")))
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--pair", action="append", type=parse_pair, default=[])
    ap.add_argument("--no-default-pairs", action="store_true")
    ap.add_argument("--md", default=None, help="write the markdown report here (default: stdout)")
    ap.add_argument("--json", default=None, help="write the full results as JSON here")
    args = ap.parse_args(argv)

    suites = [s for s in args.suites.split(",") if s]
    for s in suites:
        if s not in FILES:
            ap.error(f"unknown suite {s}")
    store = Store(args.turing_dir, args.laptop_dir)
    orders = {s: dataset_order(args.datasets_dir, s) for s in set(suites) | {p["suite"] for p in args.pair}}
    tl = lane_order(store.turing_labels())
    plan = ([] if args.no_default_pairs else default_pairs(tl, suites)) + args.pair
    res = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {"turing_dir": args.turing_dir, "laptop_dir": args.laptop_dir, "datasets_dir": args.datasets_dir,
                   "runs": args.runs if os.path.exists(args.runs) else None},
        "lanes": {s: lane_table(store, s, tl, orders[s]) for s in suites},
        "pairs": run_pairs(store, plan, orders),
        "runs": load_runs(args.runs),
        "row_files": [{"label": k[0], "suite": k[1], **v} for k, v in sorted(store.meta.items())],
    }
    md = render_md(res)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"markdown -> {args.md}")
    else:
        sys.stdout.write(md)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print(f"json -> {args.json}", file=sys.stderr if not args.md else sys.stdout)
    return res


if __name__ == "__main__":
    main()
