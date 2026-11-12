#!/usr/bin/env python3
"""Turing numerics check (plan W4(b)): does exllamav3's kernel dispatch on this GPU change the model's output
distribution, independent of task accuracy?

One process, one model load. Fixed token streams are teacher-forced through the model in up to three dispatch modes,
and the per-position next-token distributions are compared:

  default  upstream dispatch as served: graph-captured C++ decode blocks (attention, gated delta net, MLP), fused
           multi-matrix GEMMs (exl3_mgemm) and whichever EXL3 GEMV/GEMM kernels the wheel picks for this GPU
  unfused  every EXL3 linear still runs its default kernel, but one linear at a time from Python: no C++ decode
           blocks, no fused multi-matrix GEMMs (the paths upstream's EXL3_BC_ATTN=0 / EXL3_BC_GDN=0 switches select).
           The torch GDN path's qkv projection comes out in fp32, as the C++ GDN block produces it (--gdn-proj)
  ref      as unfused, but every EXL3 linear goes through an oracle: reconstruct its weights to fp16, multiply with
           cuBLAS at fp32 accumulation (ext.hgemm, CUBLAS_COMPUTE_32F), output Hadamard in fp32, then cast to the
           dtype the module asks for. Nothing in the installed wheel is edited: the switch is in-process attribute
           toggles plus a replacement LinearEXL3.forward, and a probe verifies that every EXL3 linear is called
           exactly once per decode step through the oracle before any long work starts

Every pass prefills its prompt in one fixed mode (--prefill-mode, default: default), so all passes enter the scored
forwards with the same cache and recurrent state and only the scored decode-shape forwards differ between modes.

KL(unfused||ref) is the plan's E2 measure: the EXL3 linear kernels against the same-card oracle with everything else
shared (mean <= 2e-4 and top-1 agreement >= 99% is the analysis criterion). KL(default||unfused) is the effect of the
fused decode blocks; KL(default||ref) is the end-to-end difference of the served dispatch.

Streams. --rows takes per-item eval rows (JSONL with "id"; E47 rows hold no full response text, only a 200-character
excerpt). Prompts are rebuilt from the eval datasets exactly as testsuite/evals/run_eval.py builds them and rendered
with the model's chat template (reasoning effort medium by default). The response stream is then generated greedily
in default mode at one token per forward, which is also the default-mode measurement. Rows that do carry
"response_ids" or "response_text" are teacher-forced as given. --streams reuses the exact token streams of an earlier
run (<tag>.streams.jsonl), which is how a later run on the laptop's Ada GPU gets identical inputs. Target log-probs
(NLL, target log-prob differences) are summarised only on streams no mode on this machine picked: on a stream the
run generated itself they favour the generating mode (selection bias), so those positions are left out.

Cross-run. A --streams run also loads the reference run file beside the streams file (<tag>.npz; --xref) and scores
every local mode against the reference's stored top-k at every position. This run's log-probs are exact for every
token, so the KL on the partition {reference top-k tokens, rest} is a lower bound of the exact cross-run KL in both
directions. --full-logits-stride K adds exact KL on every K-th position.

Outputs, all under --out-dir (default <lab>/runs/numerics):
  <tag>/run.json, <tag>/census.json, <tag>/item_NNNNN.npz   per-item shards (a stopped run resumes with --resume)
  <tag>.npz             per-position top-k log-probs, tail mass, target log-prob, entropy, top-1, top-2 margin per
                        mode, per-position KL for every mode pair, cross-run bounds, the token streams and meta_json
  <tag>.streams.jsonl   token ids per item (and who generated them), for --streams
  summary.jsonl         one JSON row per finished run (appended)

Other actions (CPU only, numpy only, no model):
  --combine TAG                 rebuild <tag>.npz and a summary row from the shards
  --compare A.npz B.npz         cross-run/cross-device comparison on identical streams (--label-a/--label-b)
  --selftest                    metric and prompt checks on synthetic data
"""
import argparse
import glob
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

LAB = os.environ.get("TURING_LAB", "/opt/p8")  # the desktop's ssd2 lab, never the main drive
CACHE_ENV = (("XDG_CACHE_HOME", "cache"), ("TRITON_HOME", "cache/triton"), ("TRITON_CACHE_DIR", "cache/triton/cache"),
             ("HF_HOME", "cache/huggingface"), ("PIP_CACHE_DIR", "cache/pip"))
MODES = ("default", "unfused", "ref")
# (a, b): KL(a||b) and KL(b||a) are stored. unfused||ref is E2 (linear kernels), default||ref end-to-end,
# default||unfused the fused decode blocks
PAIRS = (("unfused", "ref"), ("default", "ref"), ("default", "unfused"), ("default", "default_repeat"),
         ("default_s1", "default"), ("default_s1", "ref"))
MARGINS = (0.1, 0.5, 1.0)          # nats; "confident" positions are those where the second mode's top-2 gap exceeds it
BUCKETS = (0, 64, 256, 1024, 4096, 1 << 30)
ROW_CHUNK = 8                      # rows per float64 block in the metric code (~16 MB per mode at a 248k vocab)
E2_KL_MEAN, E2_TOP1 = 2e-4, 0.99   # plan section 6.5, E2: analysis criterion, never a launch gate
FULL_LOGITS_MAX_GIB = 6.0          # full-logit samples finish_run holds in RAM (one copy); the launcher caps at 20 GiB
TARGET_LP_NOTE = ("target log-prob statistics (NLL, target_lp differences) cover only positions whose stream no mode "
                  "on the compared machine(s) generated: on a self-generated stream the generating mode's argmax "
                  "picked the token, so on noise alone it scores higher there than any other mode (selection bias); "
                  "KL, flip rates and top-1 agreement are symmetric and use every position")

# ------------------------------------------------------------------ prompts (as testsuite/evals/run_eval.py)

HARDENING_SUITES = {"gsm8k_500": "gsm8k", "mmlu_500": "mmlu", "math500": "math25"}
DATASET_FILES = {"gsm8k": "gsm8k_100.jsonl", "mmlu": "mmlu_200.jsonl", "humaneval": "humaneval_20.jsonl",
                 "humaneval_plus": "humaneval_plus_20.jsonl", "math25": "math25.jsonl", "needle": "needle.jsonl",
                 "gsm8k_500": "gsm8k_500.jsonl", "mmlu_500": "mmlu_500.jsonl", "math500": "math500.jsonl"}


def build_prompt(suite, item):
    """Copied from run_eval.build_prompt (2026-10-30) so the desktop needs no eval kit."""
    suite = HARDENING_SUITES.get(suite, suite)
    if suite == "gsm8k":
        return item["question"] + "\n\nEnd your response with: Answer: <number>"
    if suite == "mmlu":
        letters = "ABCD"
        ch = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
        return (f"{item['question']}\n{ch}\n\n"
                "End your response with: Answer: <letter>")
    if suite in ("humaneval", "humaneval_plus"):
        return ("Complete the following Python function. Return the complete, "
                "runnable function (including the signature and any imports it "
                "needs) in a single ```python code block and nothing else "
                "after it.\n\n```python\n" + item["prompt"] + "\n```")
    if suite == "math25":
        return (item["problem"] +
                "\n\nEnd your response with: Answer: <final answer>")
    if suite == "needle":
        return item["context"] + "\n\n" + item["question"]
    raise ValueError(suite)


def visible_tail(text):
    """run_eval.visible_content: the answer after any </think>, stripped."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def suite_of(path):
    return os.path.basename(path).rsplit(".", 1)[0]


def load_row_items(rows_paths, datasets_dir, n):
    """Items from per-item eval rows, interleaved across files (row 0 of each file, then row 1, ...), first n."""
    per_file = []
    for path in rows_paths:
        suite = suite_of(path)
        ds_name = DATASET_FILES.get(suite)
        ds_path = os.path.join(datasets_dir, ds_name) if ds_name else None
        if not ds_path or not os.path.exists(ds_path):
            sys.exit(f"no dataset for rows file {path} (suite '{suite}', looked for {ds_path}); keep the rows file named "
                     f"after its suite (e.g. gsm8k_500.jsonl) and copy the datasets to --datasets")
        dataset = {d["id"]: d for d in read_jsonl(ds_path)}
        items = []
        for row in read_jsonl(path):
            d = dataset.get(row.get("id"))
            if d is None:
                print(f" !! {row.get('id')}: not in {ds_path}, skipped", flush=True)
                continue
            it = {"id": row["id"], "suite": suite, "prompt": build_prompt(suite, d),
                  "excerpt": row.get("raw_answer_excerpt"), "row_correct": row.get("correct")}
            if row.get("response_ids"):
                it["response_ids"] = [int(t) for t in row["response_ids"]]
            elif isinstance(row.get("response_text"), str):
                it["response_text"] = row["response_text"]
            items.append(it)
        per_file.append(items)
    out = []
    for i in range(max((len(x) for x in per_file), default=0)):
        for items in per_file:
            if i < len(items):
                out.append(items[i])
    return out[:n]


def render_chat(model_dir, prompt, effort):
    """The model's chat template, rendered as TabbyAPI does (one user turn, generation prompt, reasoning_effort)."""
    path = os.path.join(model_dir, "chat_template.jinja")
    src = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            src = f.read()
    else:
        tc = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(tc):
            with open(tc, encoding="utf-8") as f:
                src = json.load(f).get("chat_template")
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        jinja2 = None
    if src and jinja2 is not None:
        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        kw = {"reasoning_effort": effort} if effort else {}
        text = env.from_string(src).render(messages=[{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, **kw)
        return text, "jinja2:" + ("chat_template.jinja" if os.path.exists(path) else "tokenizer_config")
    if effort not in ("medium", None, ""):
        sys.exit("jinja2 unavailable: the literal fallback only matches reasoning effort medium")
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n", "literal-chatml"

# ------------------------------------------------------------------ metrics (numpy, float64)


def item_metrics(logits, targets, pairs, topk):
    """logits: label -> (n, V) float16/float32 raw logits; targets: (n,) next-token ids.
    Per label: top-k ids and log-probs (float64, sorted, ties by id), the log of the probability mass outside the
    top-k (tail_lp, summed directly, no cancellation), target log-prob, entropy, log-sum-exp, top-1 (first max, as
    torch/np argmax) and top-2 margin (nats). Per pair (a, b): KL(a||b) and KL(b||a) per position, float64."""
    labels = list(logits)
    n = len(targets)
    out = {}
    for lab in labels:
        out[f"{lab}__top_ids"] = np.empty((n, topk), np.int32)
        out[f"{lab}__top_lp"] = np.empty((n, topk), np.float64)
        for k, dt in (("target_lp", np.float64), ("entropy", np.float64), ("lse", np.float64),
                      ("top1", np.int32), ("margin", np.float64), ("tail_lp", np.float64)):
            out[f"{lab}__{k}"] = np.empty((n,), dt)
    for a, b in pairs:
        out[f"kl__{a}__{b}"] = np.empty((n,), np.float64)
        out[f"kl__{b}__{a}"] = np.empty((n,), np.float64)
    tg = np.asarray(targets, np.int64)
    for i0 in range(0, n, ROW_CHUNK):
        i1 = min(n, i0 + ROW_CHUNK)
        r = np.arange(i1 - i0)
        lp, p = {}, {}
        for lab in labels:
            x = logits[lab][i0:i1].astype(np.float64)
            m = x.max(axis=1, keepdims=True)
            e = np.exp(x - m)
            s = e.sum(axis=1, keepdims=True)
            lse = m + np.log(s)
            lp[lab] = x - lse
            p[lab] = e / s
            out[f"{lab}__lse"][i0:i1] = lse[:, 0]
            out[f"{lab}__target_lp"][i0:i1] = lp[lab][r, tg[i0:i1]]
            out[f"{lab}__entropy"][i0:i1] = -(p[lab] * lp[lab]).sum(axis=1)
            out[f"{lab}__top1"][i0:i1] = np.argmax(x, axis=1)
            part = np.argpartition(-x, topk - 1, axis=1)[:, :topk]
            vals = np.take_along_axis(lp[lab], part, axis=1)
            order = np.lexsort((part, -vals), axis=1)
            ids = np.take_along_axis(part, order, axis=1)
            vals = np.take_along_axis(vals, order, axis=1)
            out[f"{lab}__top_ids"][i0:i1] = ids
            out[f"{lab}__top_lp"][i0:i1] = vals
            out[f"{lab}__margin"][i0:i1] = vals[:, 0] - vals[:, 1] if topk > 1 else np.inf
            np.put_along_axis(e, part, 0.0, axis=1)          # e is not used again: sum what lies outside the top-k
            with np.errstate(divide="ignore"):
                out[f"{lab}__tail_lp"][i0:i1] = np.log(e.sum(axis=1)) - np.log(s[:, 0])
        for a, b in pairs:
            out[f"kl__{a}__{b}"][i0:i1] = (p[a] * (lp[a] - lp[b])).sum(axis=1)
            out[f"kl__{b}__{a}"][i0:i1] = (p[b] * (lp[b] - lp[a])).sum(axis=1)
    return out


def _plogq(p, q):
    """p * log(p / q) elementwise, 0 where p == 0 (q floored at 1e-300 against rounding to zero)."""
    p = np.asarray(p, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300)))
    return np.where(p > 0, v, 0.0)


def topk_bound_kl(x, refs):
    """x: (n, V) raw logits of this run. refs: list of (ids (n, k), lp (n, k), tail_lp (n,) or None), the stored
    top-k (and tail mass) of reference distributions at the same positions. Returns one (KL(ref||this),
    KL(this||ref)) pair of (n,) float64 arrays per reference: the exact KL of both distributions coarse-grained onto
    {the reference's k tokens, everything else}. This run's log-probs are exact for every token and the reference's
    k log-probs and tail mass are stored exactly, so coarse-graining makes both values lower bounds of the exact KL
    (never above it), tight when the reference's tail mass is small. Without a stored tail_lp (older run files) the
    tail is 1 - sum(p), which is only as exact as the stored log-probs."""
    n = x.shape[0]
    out = [(np.empty(n), np.empty(n)) for _ in refs]
    for i0 in range(0, n, ROW_CHUNK):
        i1 = min(n, i0 + ROW_CHUNK)
        xx = x[i0:i1].astype(np.float64)
        mx = xx.max(axis=1, keepdims=True)
        e = np.exp(xx - mx)
        s = e.sum(axis=1, keepdims=True)
        for (ids, lp, tail_lp), (fw, rv) in zip(refs, out):
            idx = np.asarray(ids[i0:i1], np.int64)
            la = np.asarray(lp[i0:i1], np.float64)
            lb = np.take_along_axis(xx, idx, axis=1) - mx - np.log(s)
            rest = e.copy()
            np.put_along_axis(rest, idx, 0.0, axis=1)
            tb = rest.sum(axis=1) / s[:, 0]
            pa, pb = np.exp(la), np.exp(lb)
            if tail_lp is not None:
                ta = np.exp(np.asarray(tail_lp[i0:i1], np.float64))
            else:
                ta = np.clip(1.0 - pa.sum(axis=1), 0.0, None)
            fw[i0:i1] = (pa * (la - lb)).sum(axis=1) + _plogq(ta, tb)
            rv[i0:i1] = (pb * (lb - la)).sum(axis=1) + _plogq(tb, ta)
    return out


def dist_stats(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [50, 90, 99, 99.9])
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "p999": float(q[3]), "max": float(v.max()), "min": float(v.min())}


def stream_self_generated(meta, hosts):
    """True when an item's stream was picked by argmax on one of these hosts (this run's own greedy pass, or a
    --streams file generated there). Target log-probs on such a stream favour the generating mode."""
    g = meta.get("stream_generator")
    if isinstance(g, dict) and g.get("host") in hosts:
        return True
    return bool(meta.get("lp_selection_biased"))


def position_mask(metas, offsets, hosts):
    """Per-position True where the target log-prob is free of selection bias (see stream_self_generated)."""
    parts = [np.full(int(offsets[i + 1] - offsets[i]), not stream_self_generated(m, hosts))
             for i, m in enumerate(metas)]
    return np.concatenate(parts) if parts else np.zeros(0, bool)


def pair_stats(arr, a, b, pos, lp_ok=None):
    """Statistics for KL(a||b); confident positions use b's top-2 margin (b is the reference in every listed pair).
    Target log-prob differences only over lp_ok positions (streams no compared mode picked)."""
    kl, klr = arr[f"kl__{a}__{b}"], arr[f"kl__{b}__{a}"]
    t1a, t1b = arr[f"{a}__top1"], arr[f"{b}__top1"]
    diff = t1a != t1b
    s = {"kl": dist_stats(kl), "kl_reverse_mean": float(np.mean(klr)) if klr.size else None,
         "top1_agree": float(np.mean(~diff)) if diff.size else None, "top1_flips": int(diff.sum())}
    for m in MARGINS:
        for who, mg in (("ref", arr[f"{b}__margin"]), ("either", np.maximum(arr[f"{a}__margin"], arr[f"{b}__margin"]))):
            conf = mg > m
            s[f"flip_rate_{who}_margin_gt_{m}"] = float(np.mean(diff[conf])) if conf.any() else None
            s[f"confident_{who}_margin_gt_{m}"] = int(conf.sum())
            s[f"flips_{who}_margin_gt_{m}"] = int((diff & conf).sum())
    ok = np.ones(kl.size, bool) if lp_ok is None else np.asarray(lp_ok, bool)
    dlp = (arr[f"{a}__target_lp"] - arr[f"{b}__target_lp"])[ok]
    s["target_lp_diff_mean"] = float(np.mean(dlp)) if dlp.size else None
    s["target_lp_absdiff_mean"] = float(np.mean(np.abs(dlp))) if dlp.size else None
    s["target_lp_positions"] = int(dlp.size)
    by_pos = {}
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        sel = (pos >= lo) & (pos < hi)
        if sel.any():
            by_pos[f"[{lo},{hi})" if hi < (1 << 30) else f"[{lo},inf)"] = {"kl_mean": float(kl[sel].mean()),
                                                                          "n": int(sel.sum())}
    s["kl_mean_by_response_position"] = by_pos
    return s


def _criterion(pairs, pair):
    h = pairs.get(pair)
    if not h:
        return {"pair": pair, "available": False}
    km = h["kl"].get("mean")
    return {"pair": pair, "available": True, "kl_mean": km, "kl_p99": h["kl"].get("p99"), "top1_agree": h["top1_agree"],
            "met": bool(km is not None and km <= E2_KL_MEAN and (h["top1_agree"] or 0.0) >= E2_TOP1)}


def cross_run_stats(arr, labels):
    """Summary of the in-process cross-run bounds (xref arrays from xref_item_arrays), per reference and local label."""
    out = {}
    for xl in sorted({k.split("__", 1)[1] for k in arr if k.startswith("xtop1__")}):
        t1x, mx = arr[f"xtop1__{xl}"], arr[f"xmargin__{xl}"]
        ok = t1x >= 0
        for lab in labels:
            f, r = arr.get(f"xkl__{xl}__{lab}"), arr.get(f"xklr__{xl}__{lab}")
            if f is None or r is None or not ok.any():
                continue
            diff = t1x[ok] != arr[f"{lab}__top1"][ok]
            s = {"positions": int(ok.sum()), "positions_without_reference": int((~ok).sum()),
                 "kl_lower_bound": dist_stats(f[ok]), "kl_reverse_lower_bound_mean": float(np.mean(r[ok])),
                 "top1_agree": float(np.mean(~diff)), "top1_flips": int(diff.sum())}
            for m in MARGINS:
                conf = mx[ok] > m
                s[f"flip_rate_xref_margin_gt_{m}"] = float(np.mean(diff[conf])) if conf.any() else None
                s[f"confident_xref_margin_gt_{m}"] = int(conf.sum())
            out[f"{xl}@xref||{lab}"] = s
    return out


def summarize(arr, labels, item_ids, offsets, lp_ok=None):
    pos = arr["pos_in_response"]
    ok = np.ones(pos.size, bool) if lp_ok is None else np.asarray(lp_ok, bool)
    out = {"n_items": len(item_ids), "n_positions": int(pos.size), "label_stats": {}, "pairs": {},
           "target_lp_scope": {"positions_used": int(ok.sum()), "positions_excluded_self_generated": int((~ok).sum()),
                               "note": TARGET_LP_NOTE}}
    for lab in labels:
        tl = arr[f"{lab}__target_lp"]
        out["label_stats"][lab] = {"nll_mean": float(-tl[ok].mean()) if ok.any() else None,
                                   "nll_positions": int(ok.sum()),
                                   "entropy_mean": float(arr[f"{lab}__entropy"].mean()) if tl.size else None,
                                   "stream_token_is_top1": float(np.mean(arr[f"{lab}__top1"] == arr["targets"]))
                                   if tl.size else None}
    for a, b in PAIRS:
        if a in labels and b in labels:
            s = pair_stats(arr, a, b, pos, ok)
            kl = arr[f"kl__{a}__{b}"]
            per_item = [(float(kl[offsets[i]:offsets[i + 1]].mean()), item_ids[i]) for i in range(len(item_ids))
                        if offsets[i + 1] > offsets[i]]
            s["worst_items"] = [{"id": i, "kl_mean": k} for k, i in sorted(per_item, reverse=True)[:5]]
            out["pairs"][f"{a}||{b}"] = s
    out["e2_analysis_criterion"] = {
        "rule": (f"KL(unfused||ref) mean <= {E2_KL_MEAN} and top-1 agreement >= {E2_TOP1}: the EXL3 linear kernels "
                 "against the same-card oracle with attention, GDN conv/scan, norms and dtypes shared (plan 6.5 E2; "
                 "a miss is treated as a kernel bug; analysis only, never a gate)"),
        **_criterion(out["pairs"], "unfused||ref")}
    out["end_to_end"] = {
        "rule": ("the same thresholds on KL(default||ref): the served dispatch against the oracle. It also contains "
                 "the fused blocks' own arithmetic (C++ attention, GDN conv/scan, graph capture, multi-matrix GEMMs) "
                 "against upstream's eager paths, so a miss here with E2 met is not a linear-kernel bug"),
        **_criterion(out["pairs"], "default||ref")}
    out["fused_blocks"] = {"rule": "KL(default||unfused): the fused decode blocks and multi-matrix GEMM bundles alone",
                           **_criterion(out["pairs"], "default||unfused")}
    out["fused_blocks"].pop("met", None)
    cross = cross_run_stats(arr, labels)
    if cross:
        out["cross_run"] = {"pairs": cross, "note": (
            "KL(reference label || local label) at every position, on {reference top-k tokens, rest}: a lower bound "
            "of the exact KL in both directions (this run's log-probs are exact for every token)")}
    return out

# ------------------------------------------------------------------ shards -> run file


def combine_shards(shards, labels):
    """shards: list of dicts (arrays + 'meta'); returns (arrays for the run .npz, per-item metas, item ids, offsets).
    Full-logit arrays present in the shards are concatenated too (finish_run loads them separately instead)."""
    keys = [k for k in shards[0] if k not in ("meta", "prompt_ids", "response_ids") and not k.startswith("full__")
            and k != "full_rows"] if shards else []
    arr, metas, ids, offsets = {}, [], [], [0]
    for k in keys:
        arr[k] = np.concatenate([s[k] for s in shards if k in s]) if all(k in s for s in shards) else None
    arr = {k: v for k, v in arr.items() if v is not None}
    full = {}
    base = 0
    for s in shards:
        meta = json.loads(str(s["meta"]))
        metas.append(meta)
        ids.append(meta["id"])
        n = int(s["targets"].size)
        offsets.append(offsets[-1] + n)
        if "full_rows" in s:
            full.setdefault("full_index", []).append(s["full_rows"].astype(np.int64) + base)
            for lab in labels:
                if f"full__{lab}" in s:
                    full.setdefault(f"full__{lab}", []).append(s[f"full__{lab}"])
        base += n
    for k, v in full.items():
        arr[k] = np.concatenate(v)
    for name in ("prompt_ids", "response_ids"):
        arr[name.replace("_ids", "_tokens")] = np.concatenate([np.asarray(s[name], np.int32) for s in shards]) \
            if shards else np.zeros(0, np.int32)
        arr[name.replace("_ids", "_offsets")] = np.cumsum([0] + [int(np.asarray(s[name]).size) for s in shards])
    arr["item_ids"] = np.array(ids)
    arr["item_offsets"] = np.asarray(offsets, np.int64)
    return arr, metas, ids, offsets


def fill_full_logits(shard_maps, keys, rows_per_shard):
    """Full-logit samples of a run, copied shard by shard into one preallocated array per key, so combining a run
    holds a single copy of them in RAM. shard_maps yields mappings (open npz files or dicts) in shard order."""
    total = int(sum(rows_per_shard))
    out, row = {}, 0
    for z, n in zip(shard_maps, rows_per_shard):
        for k in keys:
            a = z[k]
            if k not in out:
                out[k] = np.empty((total,) + tuple(a.shape[1:]), a.dtype)
            out[k][row:row + n] = a
        row += n
    return out


def _open_shards(paths):
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            yield z


def full_logits_gib(n_items, positions, stride, vocab, n_labels):
    """RAM for the full-logit samples (fp16) of a run: n_items x ceil(positions / stride) rows per label."""
    if stride <= 0:
        return 0.0
    return n_items * -(-positions // stride) * vocab * 2 * n_labels / 2 ** 30


def unique_path(path):
    stem, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(path):
        path = f"{stem}.c{k}{ext}"
        k += 1
    return path


def _brief(pairs, p):
    h = pairs.get(p)
    return h and {"kl_mean": h["kl"].get("mean"), "kl_p99": h["kl"].get("p99"), "top1_agree": h["top1_agree"],
                  "flip_rate_ref_margin_gt_0.1": h.get("flip_rate_ref_margin_gt_0.1")}


def finish_run(out_dir, tag, extra=None):
    """Read shards of a run, write <tag>.npz + <tag>.streams.jsonl (never over an existing file), append summary."""
    tag_dir = os.path.join(out_dir, tag)
    with open(os.path.join(tag_dir, "run.json"), encoding="utf-8") as f:
        run = json.load(f)
    census = None
    cpath = os.path.join(tag_dir, "census.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            census = json.load(f)
    paths = sorted(glob.glob(os.path.join(tag_dir, "item_*.npz")))
    if not paths:
        sys.exit(f"no shards in {tag_dir}")
    shards, full_keys = [], None                   # first pass: everything except the full logits
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            shards.append({k: z[k] for k in z.files if not k.startswith("full__")})
            fk = {k for k in z.files if k.startswith("full__")}
        full_keys = fk if full_keys is None else full_keys & fk
    labels = sorted({k.split("__")[0] for k in shards[0] if k.count("__") == 1 and k.endswith("__top1")})
    arr, metas, ids, offsets = combine_shards(shards, labels)
    labels = [lab for lab in labels if f"{lab}__top1" in arr]
    if full_keys and "full_index" in arr:          # second pass: one preallocated copy of the full logits
        rows = [int(s["full_rows"].size) if "full_rows" in s else 0 for s in shards]
        arr.update(fill_full_logits(_open_shards(paths), sorted(full_keys), rows))
    lp_ok = position_mask(metas, offsets, {run.get("host")})
    summary = summarize(arr, labels, ids, offsets, lp_ok=lp_ok)
    timing = {}
    for m in metas:
        for lab, t in m.get("timing", {}).items():
            agg = timing.setdefault(lab, {"s": 0.0, "positions": 0, "prefill_s": 0.0})
            agg["s"] += t["decode_s"]
            agg["prefill_s"] += t["prefill_s"]
            agg["positions"] += t["positions"]
    for t in timing.values():
        t["s_per_position"] = t["s"] / max(t["positions"], 1)
    excerpt = [m["excerpt_match"] for m in metas if m.get("excerpt_match") is not None]
    row = {"kind": "turing_numerics_w4b", "tag": tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "host": run.get("host"), "device": run.get("device"), "versions": run.get("versions"),
           "dispatch_info": run.get("dispatch_info"), "env": run.get("env"), "config": run.get("config"),
           "census": census, "labels": labels, "timing": timing,
           "greedy_excerpt_vs_rows": {"complete": len(excerpt), "match": int(sum(excerpt))},
           "gpu_peak_mib": max((m.get("torch_peak_mib", 0) for m in metas), default=0), **summary}
    if extra:
        row.update(extra)
    npz_path = unique_path(os.path.join(out_dir, f"{tag}.npz"))
    meta_all = {"run": run, "census": census, "items": metas, "summary": row}
    np.savez_compressed(npz_path, meta_json=np.array(json.dumps(meta_all)), **arr)
    streams_path = unique_path(os.path.join(out_dir, f"{tag}.streams.jsonl"))
    with open(streams_path, "w", encoding="utf-8") as f:
        for s, m in zip(shards, metas):
            f.write(json.dumps({"id": m["id"], "suite": m.get("suite"), "excerpt": m.get("excerpt"),
                                "stream_generator": m.get("stream_generator"),
                                "prompt_ids": np.asarray(s["prompt_ids"]).tolist(),
                                "response_ids": np.asarray(s["response_ids"]).tolist()}) + "\n")
    row["files"] = {"npz": npz_path, "streams": streams_path, "shards": tag_dir}
    with open(os.path.join(out_dir, "summary.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    cross = (row.get("cross_run") or {}).get("pairs", {})
    print(json.dumps({"tag": tag, "n_positions": row["n_positions"],
                      "e2_unfused_ref": row.get("e2_analysis_criterion"),
                      "end_to_end_default_ref": _brief(row["pairs"], "default||ref"),
                      "fused_blocks_default_unfused": _brief(row["pairs"], "default||unfused"),
                      "floor_default_repeat": _brief(row["pairs"], "default||default_repeat"),
                      "cross_run": {k: {"kl_lower_bound_mean": v["kl_lower_bound"].get("mean"),
                                        "top1_agree": v["top1_agree"]} for k, v in cross.items()} or None,
                      "npz": npz_path}), flush=True)
    return row

# ------------------------------------------------------------------ cross-run / cross-device comparison


def topk_overlap_kl(ids_a, lp_a, ids_b, lp_b):
    """Lower bounds of KL(A||B) and KL(B||A) from two stored top-k lists alone: both distributions coarse-grained
    onto {each token in both lists, everything else}, where every probability is known exactly. Never above the
    exact KL; far below it where the lists differ (0 when they are disjoint). Returns (fwd, rev, overlap fraction)."""
    n, k = ids_a.shape
    fw, rv, ov = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        _, ia, ib = np.intersect1d(ids_a[i], ids_b[i], assume_unique=True, return_indices=True)
        la = np.asarray(lp_a[i][ia], np.float64)
        lb = np.asarray(lp_b[i][ib], np.float64)
        pa, pb = np.exp(la), np.exp(lb)
        ra, rb = max(1.0 - float(pa.sum()), 0.0), max(1.0 - float(pb.sum()), 0.0)
        fw[i] = float((pa * (la - lb)).sum() + _plogq(ra, rb))
        rv[i] = float((pb * (lb - la)).sum() + _plogq(rb, ra))
        ov[i] = ia.size / k
    return fw, rv, ov


def full_kl(xa, xb):
    a = xa.astype(np.float64)
    b = xb.astype(np.float64)
    la = a - (a.max(1, keepdims=True) + np.log(np.exp(a - a.max(1, keepdims=True)).sum(1, keepdims=True)))
    lb = b - (b.max(1, keepdims=True) + np.log(np.exp(b - b.max(1, keepdims=True)).sum(1, keepdims=True)))
    return (np.exp(la) * (la - lb)).sum(1)


def run_meta(D):
    """The meta_json record of a run file (dict of arrays), or {} when absent."""
    mj = D.get("meta_json") if isinstance(D, dict) else None
    if mj is None:
        return {}
    try:
        return json.loads(str(mj))
    except ValueError:
        return {}


def compare_arrays(A, B, label_a, label_b):
    """A, B: dicts of run-file arrays. Aligns items by id, requires identical prompt and response streams."""
    ma, mb = run_meta(A), run_meta(B)
    hosts = {h for h in ((ma.get("run") or {}).get("host"), (mb.get("run") or {}).get("host")) if h}
    item_metas = {}
    for m in (ma.get("items") or []) + (mb.get("items") or []):
        item_metas.setdefault(str(m.get("id")), []).append(m)
    ida = [str(x) for x in A["item_ids"]]
    idb = {str(x): j for j, x in enumerate(B["item_ids"])}
    rows_a, rows_b, lp_flags, mismatched = [], [], [], []
    for i, iid in enumerate(ida):
        j = idb.get(iid)
        if j is None:
            continue
        pa = A["prompt_tokens"][A["prompt_offsets"][i]:A["prompt_offsets"][i + 1]]
        pb = B["prompt_tokens"][B["prompt_offsets"][j]:B["prompt_offsets"][j + 1]]
        sa = A["item_offsets"][i], A["item_offsets"][i + 1]
        sb = B["item_offsets"][j], B["item_offsets"][j + 1]
        n = min(sa[1] - sa[0], sb[1] - sb[0])
        if not np.array_equal(pa, pb) or not np.array_equal(A["targets"][sa[0]:sa[0] + n], B["targets"][sb[0]:sb[0] + n]):
            mismatched.append(iid)
            continue
        rows_a.append(np.arange(sa[0], sa[0] + n))
        rows_b.append(np.arange(sb[0], sb[0] + n))
        biased = any(stream_self_generated(m, hosts) for m in item_metas.get(iid, []))
        lp_flags.append(np.full(n, not biased))
    if not rows_a:
        return {"error": "no items with identical streams", "mismatched": mismatched[:10]}
    ra, rb = np.concatenate(rows_a), np.concatenate(rows_b)
    ok = np.concatenate(lp_flags)

    def g(D, lab, key, rows):
        return D[f"{lab}__{key}"][rows]
    t1a, t1b = g(A, label_a, "top1", ra), g(B, label_b, "top1", rb)
    diff = t1a != t1b
    mb_ = g(B, label_b, "margin", rb)
    tla, tlb = g(A, label_a, "target_lp", ra)[ok], g(B, label_b, "target_lp", rb)[ok]
    dlp = tla - tlb
    fw, rv, ov = topk_overlap_kl(g(A, label_a, "top_ids", ra), g(A, label_a, "top_lp", ra),
                                 g(B, label_b, "top_ids", rb), g(B, label_b, "top_lp", rb))
    res = {"label_a": label_a, "label_b": label_b, "items": len(rows_a), "items_stream_mismatch": len(mismatched),
           "positions": int(ra.size), "top1_agree": float(np.mean(~diff)),
           "target_lp_diff_mean": float(dlp.mean()) if dlp.size else None,
           "target_lp_absdiff_mean": float(np.abs(dlp).mean()) if dlp.size else None,
           "nll_a": float(-tla.mean()) if tla.size else None, "nll_b": float(-tlb.mean()) if tlb.size else None,
           "target_lp_scope": {"positions_used": int(ok.sum()), "positions_excluded_self_generated": int((~ok).sum()),
                               "hosts": sorted(hosts), "note": TARGET_LP_NOTE},
           "topk_kl_lower_bound": dist_stats(fw), "topk_kl_reverse_lower_bound_mean": float(rv.mean()),
           "topk_overlap_mean": float(ov.mean()),
           "topk_note": ("KL of both runs coarse-grained onto {tokens in both top-k lists, rest}: never above the exact "
                         "KL, far below it where the lists differ (0 when disjoint). Tighter: full_kl (both runs with "
                         "--full-logits-stride) and the cross_run bound in the summary of a --streams run")}
    for m in MARGINS:
        conf = mb_ > m
        res[f"flip_rate_b_margin_gt_{m}"] = float(np.mean(diff[conf])) if conf.any() else None
        res[f"confident_b_margin_gt_{m}"] = int(conf.sum())
    fa, fb = A.get(f"full__{label_a}"), B.get(f"full__{label_b}")
    if fa is not None and fb is not None:
        ia = {int(r): k for k, r in enumerate(A["full_index"])}
        ib = {int(r): k for k, r in enumerate(B["full_index"])}
        pairs = [(ia[int(x)], ib[int(y)]) for x, y in zip(ra, rb) if int(x) in ia and int(y) in ib]
        if pairs:
            ka, kb = zip(*pairs)
            res["full_kl"] = dist_stats(full_kl(fa[list(ka)], fb[list(kb)]))
            res["full_kl_reverse_mean"] = float(full_kl(fb[list(kb)], fa[list(ka)]).mean())
    return res


def load_run(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


XREF_BASE = ("item_ids", "item_offsets", "targets", "prompt_tokens", "prompt_offsets")
XREF_LABEL_KEYS = ("top_ids", "top_lp", "top1", "margin", "tail_lp")


def xref_from_arrays(D, labels):
    """The reference-run arrays xref_item_arrays needs, from a dict of run-file arrays; returns (X, labels found)."""
    X = {k: np.asarray(D[k]) for k in XREF_BASE}
    labs = [lab for lab in labels if all(f"{lab}__{k}" in D for k in XREF_LABEL_KEYS[:4])]
    for lab in labs:
        for k in XREF_LABEL_KEYS:
            if f"{lab}__{k}" in D:
                X[f"{lab}__{k}"] = np.asarray(D[f"{lab}__{k}"])
    X["_index"] = {str(x): j for j, x in enumerate(X["item_ids"])}
    return X, labs


def load_xref(path, labels):
    """Reference run file -> (X, labels found, run meta). Reads only the arrays needed, never the full logits."""
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        want = [k for k in names if k in XREF_BASE or k == "meta_json" or
                (k.count("__") == 1 and k.split("__")[0] in labels and k.split("__")[1] in XREF_LABEL_KEYS)]
        D = {k: z[k] for k in want}
    missing = [k for k in XREF_BASE if k not in D]
    if missing:
        sys.exit(f"--xref {path}: not a numerics run file (missing {missing})")
    X, labs = xref_from_arrays(D, labels)
    return X, labs, run_meta(D)


def xref_item_arrays(X, xlabels, item_id, prompt_ids, targets, logits):
    """Per-position cross-run arrays of one item against reference run X (from xref_from_arrays): for every
    reference label R and local label L, xkl__R__L = lower bound of KL(R||L) and xklr__R__L = lower bound of
    KL(L||R) (topk_bound_kl), plus R's top-1 (xtop1__R) and top-2 margin (xmargin__R). Positions the reference does
    not cover, and items whose prompt or stream differ, are NaN (top-1 -1). Returns (arrays, status)."""
    n = int(len(targets))
    j = X["_index"].get(str(item_id))
    m, r0, why = 0, 0, "not in the reference run"
    if j is not None:
        po = X["prompt_offsets"]
        if not np.array_equal(X["prompt_tokens"][po[j]:po[j + 1]], np.asarray(prompt_ids)):
            why = "prompt differs"
        else:
            r0, r1 = int(X["item_offsets"][j]), int(X["item_offsets"][j + 1])
            m = min(n, r1 - r0)
            if not np.array_equal(X["targets"][r0:r0 + m], np.asarray(targets)[:m]):
                m, why = 0, "stream differs"
            else:
                why = None
    out, refs = {}, []
    for xl in xlabels:
        top1 = np.full(n, -1, np.int32)
        margin = np.full(n, np.nan)
        if m:
            top1[:m] = X[f"{xl}__top1"][r0:r0 + m]
            margin[:m] = X[f"{xl}__margin"][r0:r0 + m]
            tail = X.get(f"{xl}__tail_lp")
            refs.append((X[f"{xl}__top_ids"][r0:r0 + m], X[f"{xl}__top_lp"][r0:r0 + m],
                         None if tail is None else tail[r0:r0 + m]))
        out[f"xtop1__{xl}"] = top1
        out[f"xmargin__{xl}"] = margin
    for lab, lg in logits.items():
        res = topk_bound_kl(lg[:m], refs) if m else [None] * len(xlabels)
        for xl, r in zip(xlabels, res):
            fw, rv = np.full(n, np.nan), np.full(n, np.nan)
            if r is not None:
                fw[:m], rv[:m] = r
            out[f"xkl__{xl}__{lab}"] = fw
            out[f"xklr__{xl}__{lab}"] = rv
    return out, {"matched_positions": m, "reason": why}

# ------------------------------------------------------------------ GPU run


def gpu_state():
    q = ("name,driver_version,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
         "clocks_throttle_reasons.active")
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0].split(", ")
    except Exception as e:  # noqa: BLE001 - logging only
        return {"error": f"{type(e).__name__}: {e}"}
    keys = ["name", "driver", "mem_used_mib", "mem_total_mib", "temp_c", "power_w", "sm_mhz", "mem_mhz", "throttle"]
    return dict(zip(keys, out))


def free_gib_at(path):
    probe = os.path.abspath(path)
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    st = os.statvfs(probe)
    return st.f_bavail * st.f_frsize / 2 ** 30, probe


def gpu_main(argv):
    if not os.path.isdir(LAB):
        sys.exit(f"lab directory {LAB} not found; set TURING_LAB to this machine's lab folder")
    for k, sub in CACHE_ENV:                       # caches stay inside the lab even under desktop_run.sh cmd
        os.environ.setdefault(k, os.path.join(LAB, sub))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch
    from exllamav3 import model_init, Config, Tokenizer
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant import exl3 as exl3_mod
    from exllamav3.modules.quant.exl3 import LinearEXL3
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    model_init.add_args(parser, default_cache_size=0)
    parser.add_argument("--rows", nargs="+", default=[], help="per-item eval rows (JSONL named after the suite)")
    parser.add_argument("--datasets", default=os.path.join(LAB, "evals", "datasets"), help="eval datasets directory")
    parser.add_argument("--streams", default=None, help="token streams from an earlier run (<tag>.streams.jsonl)")
    parser.add_argument("--num-items", type=int, default=50, help="items, interleaved across --rows files (default 50)")
    parser.add_argument("--limit", type=int, default=0, help="smoke runs: first K items only")
    parser.add_argument("--max-response-tokens", type=int, default=256, help="scored positions per item (default 256)")
    parser.add_argument("--gen-max-tokens", type=int, default=0,
                        help="greedy generation length when the stream is generated (default: --max-response-tokens; "
                             "4096 generates complete answers for the excerpt check, unscored beyond the scored part)")
    parser.add_argument("--step", type=int, default=1, help="tokens per teacher-forced forward (1 = decode; 5 = MTP "
                                                            "draft-4 verify shape)")
    parser.add_argument("--modes", default="default,unfused,ref", help="comma list of default, unfused, ref")
    parser.add_argument("--repeat-default", action="store_true", help="second default pass: run-to-run floor")
    parser.add_argument("--prefill-mode", choices=("default", "ref", "own"), default="default",
                        help="mode every pass prefills its prompt in: default (the served path; every pass enters the "
                             "scored forwards with the same state, so only decode-shape dispatch is compared), ref "
                             "(the oracle; the greedy stream then no longer matches serving) or own (each pass in its "
                             "own mode: end-to-end, mixes prefill-shape kernel error into every scored position)")
    parser.add_argument("--gdn-proj", choices=("fp32", "upstream"), default="fp32",
                        help="qkv projection dtype of the torch GDN path in unfused and ref modes: fp32 (default; what "
                             "the C++ GDN decode block produces, so the oracle is never coarser than the served path "
                             "and unfused||ref differ only in the linear kernels) or upstream (EXL3_GDN_PROJ_FP32: "
                             "fp16 unless set)")
    parser.add_argument("--reasoning-effort", default="medium", help="chat template reasoning_effort ('' for none)")
    parser.add_argument("--ref-epilogue", choices=("fp32", "fp16"), default="fp32",
                        help="oracle GEMM output and output Hadamard in fp32 (default) or fp16 as upstream's "
                             "reconstruct path does")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--full-logits-stride", type=int, default=0,
                        help="also keep full fp16 logits of every K-th scored position (exact cross-run KL on a "
                             "sample; 16 is about 0.4 GiB per mode for 50 x 256 positions). The projected size is "
                             f"printed, and a run above {FULL_LOGITS_MAX_GIB:g} GiB or the free disk is refused")
    parser.add_argument("--xref", default="auto",
                        help="reference run file (<tag>.npz) for the in-process cross-run KL bound on identical "
                             "streams: auto (default) takes the .npz beside --streams when it exists, none disables")
    parser.add_argument("--xref-labels", default="default,ref", help="reference-run labels to compare against")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-dir", default=os.path.join(LAB, "runs", "numerics"))
    parser.add_argument("--resume", action="store_true", help="continue a stopped run with the same --tag")
    parser.add_argument("--allow-incomplete-ref", action="store_true",
                        help="continue even if the probe finds EXL3 linears that bypass the oracle")
    args = parser.parse_args(argv)

    modes = [m for m in args.modes.split(",") if m]
    if any(m not in MODES for m in modes) or not modes:
        sys.exit(f"--modes takes a comma list of {MODES}")
    if args.step < 1 or args.max_response_tokens < 1 or args.topk < 2 or args.full_logits_stride < 0:
        sys.exit("--step and --max-response-tokens must be >= 1, --topk >= 2, --full-logits-stride >= 0")
    R = args.max_response_tokens
    G = max(args.gen_max_tokens or R, R)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S") + "-" + socket.gethostname()
    tag_dir = os.path.join(args.out_dir, tag)
    host = socket.gethostname()
    if os.path.exists(tag_dir) and not args.resume:
        sys.exit(f"{tag_dir} exists; pass --resume to continue it or choose another --tag")
    if args.resume and not os.path.exists(os.path.join(tag_dir, "run.json")):
        sys.exit(f"--resume: no run.json in {tag_dir}")

    def plan(have_stream):
        """(label, mode, step, generate) per pass; without a given stream the first pass generates it greedily."""
        passes = []
        if not have_stream:
            passes.append(("default" if args.step == 1 else "default_s1", "default", 1, True))
        for m in modes:
            if not (m == "default" and not have_stream and args.step == 1):
                passes.append((m, m, args.step, False))
        if args.repeat_default:
            passes.append(("default_repeat", "default", args.step, False))
        return passes

    # Items and tokenization (tokenizer only, before the model load, so the cache can be sized to the streams)
    pre_config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(pre_config)
    n_items = args.limit or args.num_items
    if args.streams:
        items = read_jsonl(args.streams)[:n_items]
        render = "streams-file"
    else:
        if not args.rows:
            sys.exit("give --rows (eval rows) or --streams")
        items = load_row_items(args.rows, args.datasets, n_items)
        render = None
        for it in items:
            text, render = render_chat(args.model_dir, it["prompt"], args.reasoning_effort)
            it["prompt_ids"] = tokenizer.encode(text, add_bos=False, encode_special_tokens=True)[0].tolist()
            if "response_text" in it:
                it["response_ids"] = tokenizer.encode(it.pop("response_text"), add_bos=False,
                                                      encode_special_tokens=True)[0].tolist()
    if not items:
        sys.exit("no items")
    longest = max(len(it["prompt_ids"]) + (min(len(it["response_ids"]), R) if it.get("response_ids") else G)
                  for it in items)
    if not args.cache_size:
        # at least one load chunk: upstream's autosplit load runs a measuring forward of (1, chunk_size) against the
        # cache and asserts it fits (attn.py prepare_flash_attn: "Cache too small for batch shape"); chunk_size
        # defaults to 4096 (model_init.add_args), so a 2048-token cache failed at load on the desktop (2026-10-30)
        args.cache_size = max(2048, -(-(longest + 256) // 256) * 256, -(-args.chunk_size // 256) * 256)
    if args.cache_size < args.chunk_size:
        sys.exit(f"--cache_size {args.cache_size} is smaller than --chunk_size {args.chunk_size}: upstream's model load "
                 f"needs one chunk to fit the cache; raise -cs or lower -chunk_size")
    stop_ids = {tokenizer.single_id("<|im_end|>")} | set(pre_config.eos_token_id_list or [])
    V = tokenizer.actual_vocab_size

    # Full-logit samples: projected RAM (finish_run holds one copy) and disk (shards plus run file)
    n_labels = len({p[0] for have in (False, True) for p in plan(have)})
    if args.full_logits_stride > 0:
        gib = full_logits_gib(len(items), R, args.full_logits_stride, V, n_labels)
        free, where = free_gib_at(args.out_dir)
        print(f" -- full logits: every {args.full_logits_stride}th scored position, at most "
              f"{len(items) * -(-R // args.full_logits_stride)} rows x {n_labels} modes x {V} vocab fp16 = {gib:.2f} GiB "
              f"in RAM when the run is combined, about {2 * gib:.2f} GiB on disk (shards plus run file; fp16 logits "
              f"barely compress); {free:.0f} GiB free at {where}", flush=True)
        if gib > FULL_LOGITS_MAX_GIB:
            ok_stride = args.full_logits_stride
            while full_logits_gib(len(items), R, ok_stride, V, n_labels) > FULL_LOGITS_MAX_GIB:
                ok_stride += 1
            sys.exit(f"--full-logits-stride {args.full_logits_stride}: {gib:.1f} GiB of full logits exceeds the "
                     f"{FULL_LOGITS_MAX_GIB:g} GiB budget (the run would be OOM-killed when combining, after the GPU "
                     f"work); use a stride of at least {ok_stride}")
        if 2.2 * gib > free:
            sys.exit(f"--full-logits-stride {args.full_logits_stride}: needs about {2.2 * gib:.1f} GiB of disk, "
                     f"{free:.1f} GiB free at {where}")

    # Reference run for the in-process cross-run bound
    xref_path = None
    if args.xref == "auto":
        suffix = ".streams.jsonl"
        if args.streams and args.streams.endswith(suffix) and os.path.exists(args.streams[:-len(suffix)] + ".npz"):
            xref_path = args.streams[:-len(suffix)] + ".npz"
    elif args.xref != "none":
        xref_path = args.xref
        if not os.path.exists(xref_path):
            sys.exit(f"--xref {xref_path}: not found")
    X, xlabels, xmeta = None, [], {}
    if xref_path:
        X, xlabels, xmeta = load_xref(xref_path, [lab for lab in args.xref_labels.split(",") if lab])
        if not xlabels:
            sys.exit(f"--xref {xref_path}: none of the labels {args.xref_labels} are in it")
        if any(X[f"{xl}__top_ids"].size and int(X[f"{xl}__top_ids"].max()) >= V for xl in xlabels):
            sys.exit(f"--xref {xref_path}: its token ids exceed this model's vocabulary ({V})")
        print(f" -- xref: {xref_path} labels {xlabels} (run {(xmeta.get('run') or {}).get('host')}, "
              f"{((xmeta.get('run') or {}).get('device') or {}).get('name')})", flush=True)
    xref_run = {k: (xmeta.get("run") or {}).get(k) for k in ("host", "time")} if xref_path else None
    if xref_run is not None:
        xref_run["device"] = ((xmeta.get("run") or {}).get("device") or {}).get("name")

    config_rec = {"model_dir": os.path.abspath(args.model_dir), "cache_size": args.cache_size,
                  "cache_quant": args.cache_quant, "recurrent_state_fp32": bool(args.recurrent_state_fp32),
                  "chunk_size": args.chunk_size, "no_warmup": bool(args.no_warmup), "max_response_tokens": R,
                  "gen_max_tokens": G, "step": args.step, "modes": modes, "repeat_default": args.repeat_default,
                  "prefill_mode": args.prefill_mode, "gdn_proj": args.gdn_proj,
                  "reasoning_effort": args.reasoning_effort, "prompt_render": render, "ref_epilogue": args.ref_epilogue,
                  "topk": args.topk, "full_logits_stride": args.full_logits_stride, "vocab_scored": V,
                  "rows": [os.path.abspath(p) for p in args.rows], "streams": args.streams,
                  "xref": os.path.abspath(xref_path) if xref_path else None, "xref_labels": xlabels,
                  "xref_run": xref_run, "item_ids": [it["id"] for it in items]}
    os.makedirs(tag_dir, exist_ok=True)
    run_path = os.path.join(tag_dir, "run.json")
    if args.resume:
        with open(run_path, encoding="utf-8") as f:
            old = json.load(f)["config"]
        for k in ("model_dir", "max_response_tokens", "gen_max_tokens", "step", "modes", "repeat_default",
                  "prefill_mode", "gdn_proj", "reasoning_effort", "ref_epilogue", "item_ids", "cache_quant",
                  "recurrent_state_fp32", "topk", "full_logits_stride", "xref", "xref_labels"):
            if old.get(k) != config_rec.get(k):
                sys.exit(f"--resume: '{k}' differs from the stored run ({old.get(k)!r} vs {config_rec.get(k)!r})")

    print(f" -- {len(items)} items, scored {R} positions each, cache {args.cache_size}, modes {modes}, "
          f"step {args.step}, prefill {args.prefill_mode}, tag {tag}", flush=True)
    t_load = time.time()
    model, config, cache, tokenizer = model_init.init(args, progress=False)
    load_s = time.time() - t_load
    dev = next((m.device for m in model if getattr(m, "device", None) is not None
                and str(m.device).startswith("cuda")), "cuda:0")
    dev_idx = torch.device(dev).index or 0
    if longest > cache.max_num_tokens // 256 * 256:
        sys.exit(f"longest stream ({longest} tokens) exceeds the cache ({cache.max_num_tokens}); raise -cs")

    # ---- dispatch modes
    attn_mods = []
    for name in ("exllamav3.modules.attn", "exllamav3.modules.sliding_attn", "exllamav3.modules.mla_attn"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for flag in ("_bc_attn_enable", "_bc_mla_enable"):
            if hasattr(mod, flag):
                attn_mods.append((mod, flag, getattr(mod, flag)))
    # The torch GDN path asks for the qkv projection in _proj_dtype (fp16 unless EXL3_GDN_PROJ_FP32), the C++ GDN
    # block and the sliced qkvz bundle produce it in fp32; the global is read at call time
    gdn_mods = []
    try:
        mod = importlib.import_module("exllamav3.modules.gated_delta_net")
        if hasattr(mod, "_proj_dtype"):
            gdn_mods.append((mod, mod._proj_dtype))
    except ImportError:
        pass
    toggles = []                                    # (object, attribute, default value, unfused/ref value)
    for m in model:
        for attr in ("multi_qkv", "multi_qg", "multi_kv", "multi_qkvz"):
            if getattr(m, attr, None) is not None:
                toggles.append((m, attr, getattr(m, attr), None))
        gu = getattr(m, "multi_gu", None)
        if isinstance(gu, list) and any(x is not None for x in gu):
            toggles.append((m, "multi_gu", list(gu), [None] * len(gu)))
        if type(m).__name__ in ("GatedMLP", "MLP") and getattr(m, "bc", None) is not None:
            toggles.append((m, "bc", m.bc, None))
        if getattr(m, "bc_split", False) is True:
            toggles.append((m, "bc_split", True, False))
    exl3_all = [(m.key, m.inner) for m in model
                if getattr(m, "quant_type", None) == "exl3" and isinstance(m.inner, LinearEXL3)]
    # The checkpoint also carries a vision tower and the MTP head; neither runs in a text forward
    off_path = ("model.visual", "visual.", "mtp.", "model.mtp")
    exl3 = [(k, i) for k, i in exl3_all if not str(k).startswith(off_path)]
    if not exl3:
        sys.exit("no EXL3 linears found in the loaded model")
    slice_n = exl3_mod.MAX_RECONSTRUCT_SLICE_N
    scratch_numel = max(inner.in_features * min(inner.out_features, slice_n) for _, inner in exl3)

    class Census:
        active = False
        calls = {}
        casts = 0

    def hit(inner):
        if Census.active:
            Census.calls[id(inner)] = Census.calls.get(id(inner), 0) + 1

    orig_forward = LinearEXL3.forward

    def counting_forward(self, x, params, out_dtype=None):
        hit(self)
        return orig_forward(self, x, params, out_dtype)

    scratch = {}
    epilogue32 = args.ref_epilogue == "fp32"

    def ref_forward(self, x, params, out_dtype=None):
        if "ovr" in params:                         # tensor overrides keep upstream semantics
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)
        hit(self)
        k, n = self.in_features, self.out_features
        rows = x.numel() // x.shape[-1]
        x2 = x.reshape(rows, k)
        if x2.dtype == torch.half:
            xh = torch.empty_like(x2)
            ext.had_r_128(x2, xh, self.suh, None, 1.0)
        else:                                       # input Hadamard in fp32, one rounding to fp16 for the GEMM
            Census.casts += 1
            x32 = x2.float().contiguous()
            xh32 = torch.empty_like(x32)
            ext.had_r_128(x32, xh32, self.suh, None, 1.0)
            xh = xh32.half()
        acc = torch.empty((rows, n), dtype=torch.float if epilogue32 else torch.half, device=x.device)
        buf = scratch.get("w")
        if buf is None:
            buf = scratch["w"] = torch.empty((scratch_numel,), dtype=torch.half, device=x.device)
        for n0 in range(0, n, slice_n):
            n1 = min(n0 + slice_n, n)
            w = buf[: k * (n1 - n0)].view(k, n1 - n0)
            self._reconstruct_inner(w, n0)          # rotated-basis fp16 weights, exactly the trellis values
            ext.hgemm(xh, w, acc[:, n0:n1])         # cuBLAS GemmEx, CUBLAS_COMPUTE_32F
        ext.had_r_128(acc, acc, None, self.svh, 1.0)
        if self.bias is not None:
            acc += self.bias.to(acc.dtype)
        dtype = out_dtype or self.default_out_dtype
        y = acc if acc.dtype == dtype else acc.to(dtype)
        return y.view(*x.shape[:-1], n)

    def set_mode(mode):
        fast = mode == "default"
        for obj, attr, fv, sv in toggles:
            setattr(obj, attr, (list(fv) if isinstance(fv, list) else fv) if fast else
                    (list(sv) if isinstance(sv, list) else sv))
        for mod, flag, orig in attn_mods:
            setattr(mod, flag, orig if fast else False)
        for mod, orig in gdn_mods:
            mod._proj_dtype = orig if fast or args.gdn_proj == "upstream" else torch.float
        LinearEXL3.forward = ref_forward if mode == "ref" else counting_forward

    batch_shape = (1, cache.max_num_tokens // 256 * 256)

    def fwd(ids, pos, rs, prefill=False):
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos, "batch_shape": batch_shape,
                  "recurrent_states": rs}
        t = torch.tensor([ids], dtype=torch.long)
        return model.prefill(t, params) if prefill else model.forward(t, params)

    # inference mode, as upstream's generator runs: the cache's recurrent-state tensors are created at load under
    # inference mode, and get_new_state() zeroes them in place, which plain no-grad code may not do (desktop,
    # 2026-10-30: "Inplace update to inference tensor outside InferenceMode is not allowed")
    @torch.inference_mode()
    def run_pass(mode, prompt_ids, resp, step, generate, probe=None):
        """Prefill prompt[:-1] in chunks (in --prefill-mode), then feed prompt[-1] and the stream in this pass's mode;
        returns (logits (n, V), stream, timing)."""
        pre = mode if args.prefill_mode == "own" else args.prefill_mode
        set_mode(pre)
        torch.cuda.reset_peak_memory_stats()
        rs = [cache.get_new_state()]
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            pos = 0
            P = len(prompt_ids)
            while pos < P - 1:
                end = min(P - 1, pos + args.chunk_size)
                fwd(prompt_ids[pos:end], pos, rs, prefill=True)
                pos = end
            set_mode(mode)
            torch.cuda.synchronize()
            t1 = time.time()
            store = None
            n_score = R if generate else min(R, len(resp))
            seq = None if generate else [prompt_ids[-1]] + resp[:n_score - 1]
            j = 0
            cur = prompt_ids[-1]
            gen = []
            while True:
                if generate:
                    if j >= G:
                        break
                    ids = [cur]
                else:
                    if j >= n_score:
                        break
                    ids = seq[j:j + step]
                if probe is not None and j == 0:
                    Census.calls, Census.active = {}, True
                logits = fwd(ids, pos, rs)
                if probe is not None and j == 0:
                    Census.active = False
                    probe.update(dict(Census.calls))
                pos += len(ids)
                rows = logits[0, :, :V]
                if store is None:
                    store = np.empty((n_score, V), np.float16 if rows.dtype == torch.half else np.float32)
                if j < n_score:
                    take = min(len(ids), n_score - j)
                    store[j:j + take] = rows[:take].float().cpu().numpy() if rows.dtype not in (torch.half, torch.float) \
                        else rows[:take].cpu().numpy()
                if generate:
                    tok = int(torch.argmax(rows[-1]))
                    gen.append(tok)
                    cur = tok
                    j += 1
                    if tok in stop_ids:
                        break
                else:
                    j += len(ids)
            torch.cuda.synchronize()
            t2 = time.time()
        finally:
            for r in rs:
                r.free()
        stream = gen if generate else resp
        n = min(n_score, len(stream))
        timing = {"prefill_s": t1 - t0, "decode_s": t2 - t1, "positions": n, "forwards_after_prefill": j,
                  "prefill_mode": pre, "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
        return store[:n], stream, timing

    # ---- run record, dispatch facts, coverage probe
    def safe(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            return f"n/a ({type(e).__name__})"
    backend = importlib.import_module("exllamav3.util.backend")
    dispatch_info = {"int8_gemv_mode": safe(lambda: backend.INT8_GEMV_MODE), "qkv_slice": safe(lambda: backend.QKV_SLICE),
                     "int8_gemv_max_k": safe(lambda: int(ext.exl3_gemv_int8_max_k(dev_idx))),
                     "hgemm_f16acc_status": safe(lambda: int(ext.hgemm_f16acc_status(dev_idx))),
                     "auto_reconstruct_threshold": exl3_mod.AUTO_RECONSTRUCT_THRESHOLD,
                     "exl3_linears": len(exl3), "exl3_linears_off_text_path": len(exl3_all) - len(exl3),
                     "toggles": len(toggles),
                     "toggle_kinds": sorted({f"{type(o).__name__}.{a}" for o, a, _, _ in toggles}),
                     "attn_flags": [f"{m.__name__}.{f}={o}" for m, f, o in attn_mods],
                     "gdn_proj_dtype": {"upstream": [str(o) for _, o in gdn_mods],
                                        "unfused_ref": "torch.float32" if args.gdn_proj == "fp32" else "upstream"},
                     "prefill_mode": args.prefill_mode}
    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                "exllamav3": safe(lambda: importlib.import_module("exllamav3.version").__version__),
                "exllamav3_ext": safe(lambda: ext.__file__), "numpy": np.__version__,
                "triton": safe(lambda: importlib.import_module("triton").__version__)}
    device = {"name": torch.cuda.get_device_name(dev_idx), "cc": list(torch.cuda.get_device_capability(dev_idx)),
              "start": gpu_state()}
    env = {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "EXLLAMA", "CUDA_", "PYTORCH_", "TRITON"))}
    if not args.resume:
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump({"host": host, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load_s": load_s,
                       "device": device, "versions": versions, "dispatch_info": dispatch_info, "env": env,
                       "config": config_rec}, f, indent=1)
    print(" -- dispatch:", json.dumps(dispatch_info), flush=True)

    census_path = os.path.join(tag_dir, "census.json")
    census = {}
    if os.path.exists(census_path):
        with open(census_path, encoding="utf-8") as f:
            census = json.load(f)
    else:
        # One prefill of up to 16 prompt tokens (in --prefill-mode), then one teacher-forced forward of --step tokens
        # per mode; the calls of that forward are counted. Also pays each mode's first-use compiles before the
        # timed items
        pp = list(items[0]["prompt_ids"])
        if len(pp) < 4:
            sys.exit("the first item's prompt is too short for the dispatch probe")
        s = max(1, min(args.step, len(pp) - 2))
        head = pp[:len(pp) - s][:16]
        tail = pp[len(head):len(head) + s]
        known = {id(inner): key for key, inner in exl3}
        for mode in modes:
            calls = {}
            run_pass(mode, head, tail, args.step, False, probe=calls)
            missing = [known[i] for i in known if i not in calls]
            census[mode] = {"python_dispatched_linears": sum(1 for i in calls if i in known),
                            "exl3_linears": len(known), "called_once": sum(1 for i in known if calls.get(i) == 1),
                            "missing": missing[:20], "n_missing": len(missing),
                            "called_more_than_once": [known[i] for i in known if calls.get(i, 0) > 1][:20],
                            "unknown_callers": sum(1 for i in calls if i not in known)}
        census["ref_input_casts"] = Census.casts
        census["note"] = ("python_dispatched_linears counts EXL3 linears that ran through LinearEXL3.forward in the "
                          "first decode forward; the rest ran inside C++ decode blocks or fused multi-matrix GEMMs")
        with open(census_path, "w", encoding="utf-8") as f:
            json.dump(census, f, indent=1)
    print(" -- census:", json.dumps({m: {k: v for k, v in c.items() if k != "missing"} if isinstance(c, dict) else c
                                     for m, c in census.items()}), flush=True)
    for m in ("ref", "unfused"):
        c = census.get(m)
        if c and (c["n_missing"] or c["called_once"] != c["exl3_linears"]):
            msg = f" !! {m}: {c['n_missing']} EXL3 linears bypass Python dispatch (first: {c['missing'][:5]})"
            if m == "ref" and not args.allow_incomplete_ref:
                sys.exit(msg + "; the oracle would be incomplete. Stopping (--allow-incomplete-ref overrides)")
            print(msg, flush=True)

    # ---- items
    t_run = time.time()
    done_items = 0
    for idx, it in enumerate(items):
        shard = os.path.join(tag_dir, f"item_{idx:05d}.npz")
        if os.path.exists(shard):
            continue
        prompt_ids = list(it["prompt_ids"])
        given = list(it["response_ids"]) if it.get("response_ids") else None
        logits, timing, stream = {}, {}, given
        for label, mode, step, generate in plan(given is not None):
            lg, st, tm = run_pass(mode, prompt_ids, stream, step, generate)
            if generate:
                stream = st
            logits[label], timing[label] = lg, tm
        n = min(x.shape[0] for x in logits.values())
        logits = {k: v[:n] for k, v in logits.items()}
        targets = np.asarray(stream[:n], np.int32)
        pairs = [(a, b) for a, b in PAIRS if a in logits and b in logits]
        arrays = item_metrics(logits, targets, pairs, args.topk)
        arrays["targets"] = targets
        arrays["pos_in_response"] = np.arange(n, dtype=np.int32)
        xstatus = None
        if X is not None:
            xa, xstatus = xref_item_arrays(X, xlabels, it["id"], prompt_ids, targets, logits)
            arrays.update(xa)
        if args.full_logits_stride > 0:
            sel = np.arange(0, n, args.full_logits_stride)
            arrays["full_rows"] = sel.astype(np.int32)
            for lab, lg in logits.items():
                arrays[f"full__{lab}"] = lg[sel].astype(np.float16)
        if given is None:
            generator = {"host": host, "gpu": device["name"], "tag": tag, "label": plan(False)[0][0]}
            source = "greedy"
        else:
            generator = it.get("stream_generator")
            source = "streams-file" if args.streams else "given"
        complete = bool(stream) and stream[-1] in stop_ids
        match = None
        if complete and it.get("excerpt") is not None:
            text = tokenizer.decode(torch.tensor([stream[:-1]], dtype=torch.long), decode_special_tokens=True)
            text = text[0] if isinstance(text, list) else text
            match = visible_tail(text)[-200:] == it["excerpt"]

        def kl_mean(a, b):
            k = f"kl__{a}__{b}"
            return float(arrays[k].mean()) if k in arrays and arrays[k].size else None
        meta = {"id": it["id"], "suite": it.get("suite"), "excerpt": it.get("excerpt"), "prompt_tokens": len(prompt_ids),
                "stream_tokens": len(stream), "scored": n, "stream_source": source, "stream_generator": generator,
                "lp_selection_biased": stream_self_generated({"stream_generator": generator}, {host}),
                "complete": complete, "excerpt_match": match, "timing": timing, "xref": xstatus,
                "torch_peak_mib": max(t["torch_peak_mib"] for t in timing.values()),
                "kl_unfused_ref_mean": kl_mean("unfused", "ref"), "kl_default_ref_mean": kl_mean("default", "ref")}
        np.savez_compressed(shard, meta=np.array(json.dumps(meta)), prompt_ids=np.asarray(prompt_ids, np.int32),
                            response_ids=np.asarray(stream, np.int32), **arrays)
        done_items += 1
        left = len(items) - idx - 1
        rate = (time.time() - t_run) / done_items
        print(json.dumps({"item": idx, "id": it["id"], "P": len(prompt_ids), "n": n, "complete": complete,
                          "excerpt_match": match, "kl_unfused_ref": meta["kl_unfused_ref_mean"],
                          "kl_default_ref": meta["kl_default_ref_mean"],
                          "xref_positions": xstatus and xstatus["matched_positions"],
                          "s": {k: round(t["prefill_s"] + t["decode_s"], 1) for k, t in timing.items()},
                          "eta_min": round(rate * left / 60, 1), "gpu": gpu_state().get("sm_mhz")}), flush=True)

    set_mode("default")
    LinearEXL3.forward = orig_forward
    finish_run(args.out_dir, tag, extra={"device_end": gpu_state(), "load_s": load_s})

# ------------------------------------------------------------------ self-test (CPU, numpy only)


def selftest():
    rng = np.random.default_rng(0)
    fails = []

    def check(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    n, V, K = 37, 1000, 20
    a = (rng.standard_normal((n, V)) * 3).astype(np.float32)
    b = (a + rng.standard_normal((n, V)).astype(np.float32) * 0.01).astype(np.float32)
    tg = rng.integers(0, V, n)
    m = item_metrics({"default": a, "ref": b, "default_repeat": a.copy()}, tg,
                     [("default", "ref"), ("default", "default_repeat")], K)

    def ref_kl(x, y):
        x, y = x.astype(np.float64), y.astype(np.float64)
        lx = x - np.logaddexp.reduce(x)
        ly = y - np.logaddexp.reduce(y)
        return float(np.sum(np.exp(lx) * (lx - ly)))
    exact_f = np.array([ref_kl(a[i], b[i]) for i in range(n)])
    exact_r = np.array([ref_kl(b[i], a[i]) for i in range(n)])
    check("KL matches an independent logaddexp formula",
          all(abs(m["kl__default__ref"][i] - exact_f[i]) < 1e-12 for i in range(n)))
    check("reverse KL stored", abs(m["kl__ref__default"][3] - exact_r[3]) < 1e-12)
    check("KL of identical logits is exactly 0", np.all(m["kl__default__default_repeat"] == 0.0))
    check("small perturbation gives small positive KL", 0 < m["kl__default__ref"].mean() < 1e-2)
    check("top1 equals argmax", np.array_equal(m["default__top1"], np.argmax(a, axis=1)))
    lp0 = a[0].astype(np.float64) - np.logaddexp.reduce(a[0].astype(np.float64))
    check("top-k sorted and exact", np.allclose(m["default__top_lp"][0], np.sort(lp0)[::-1][:K], atol=1e-12)
          and np.array_equal(m["default__top_ids"][0], np.argsort(-lp0, kind="stable")[:K]))
    check("tail mass + top-k mass = 1", np.all(np.abs(np.exp(m["default__tail_lp"])
                                                      + np.exp(m["default__top_lp"]).sum(axis=1) - 1) < 1e-12))
    check("tail mass exact", abs(m["default__tail_lp"][0] - np.logaddexp.reduce(np.sort(lp0)[::-1][K:])) < 1e-9)
    check("target log-prob", abs(m["default__target_lp"][0] - lp0[tg[0]]) < 1e-12)
    check("entropy", abs(m["default__entropy"][0] + float(np.sum(np.exp(lp0) * lp0))) < 1e-9)
    # ties: fp16 logits can tie exactly; top1 is the first index, margin 0
    t = np.zeros((1, 50), np.float16)
    t[0, 7] = t[0, 3] = 5.0
    mt = item_metrics({"x": t}, np.array([3]), [], 5)
    check("tie: first max index, zero margin, ids ordered", mt["x__top1"][0] == 3 and mt["x__margin"][0] == 0.0
          and list(mt["x__top_ids"][0][:2]) == [3, 7])
    # flip accounting: three positions, known margins
    lg = {"default": np.array([[2.0, 0.0, 0.0], [1.0, 1.05, 0.0], [0.0, 3.0, 0.0]], np.float32),
          "ref": np.array([[2.0, 0.0, 0.0], [1.05, 1.0, 0.0], [3.0, 0.0, 0.0]], np.float32)}
    mm = item_metrics(lg, np.array([0, 0, 1]), [("default", "ref")], 2)
    mm["targets"] = np.array([0, 0, 1], np.int32)
    mm["pos_in_response"] = np.arange(3, dtype=np.int32)
    s = summarize(mm, ["default", "ref"], ["x"], [0, 3])["pairs"]["default||ref"]
    check("flip counts: 2 flips, 1 at a near tie", s["top1_flips"] == 2 and s["confident_ref_margin_gt_0.1"] == 2
          and s["flips_ref_margin_gt_0.1"] == 1 and abs(s["flip_rate_ref_margin_gt_0.1"] - 0.5) < 1e-12)
    # E2 on unfused||ref, default||ref end-to-end, default||unfused fused blocks
    m3 = item_metrics({"default": a, "unfused": b, "ref": b.copy()}, tg,
                      [("unfused", "ref"), ("default", "ref"), ("default", "unfused")], K)
    m3["targets"] = tg.astype(np.int32)
    m3["pos_in_response"] = np.arange(n, dtype=np.int32)
    s3 = summarize(m3, ["default", "unfused", "ref"], ["x"], [0, n])
    check("E2 is unfused||ref; default||ref is end-to-end; fused blocks reported",
          s3["e2_analysis_criterion"]["pair"] == "unfused||ref" and s3["e2_analysis_criterion"]["met"] is True
          and s3["e2_analysis_criterion"]["kl_mean"] == 0.0 and s3["end_to_end"]["pair"] == "default||ref"
          and s3["fused_blocks"]["kl_mean"] > 0 and "met" not in s3["fused_blocks"])
    # target log-probs: left out on self-generated streams
    s_none = summarize(m3, ["default", "unfused", "ref"], ["x"], [0, n], lp_ok=np.zeros(n, bool))
    s_half = summarize(m3, ["default", "unfused", "ref"], ["x"], [0, n], lp_ok=np.arange(n) < 10)
    check("self-generated positions: no NLL / target-lp stats, KL kept",
          s_none["label_stats"]["default"]["nll_mean"] is None
          and s_none["pairs"]["default||ref"]["target_lp_diff_mean"] is None
          and s_none["pairs"]["default||ref"]["kl"]["n"] == n
          and s_half["pairs"]["default||ref"]["target_lp_positions"] == 10
          and abs(s_half["label_stats"]["ref"]["nll_mean"] + m3["ref__target_lp"][:10].mean()) < 1e-12)
    check("stream provenance", stream_self_generated({"stream_generator": {"host": "desk"}}, {"desk"})
          and not stream_self_generated({"stream_generator": {"host": "desk"}}, {"lap"})
          and not stream_self_generated({"stream_generator": None}, {"lap"})
          and list(position_mask([{"stream_generator": {"host": "desk"}}, {}], [0, 2, 5], {"desk"}))
          == [False, False, True, True, True])
    # in-process cross-run bound: reference top-k against this run's exact logits
    (fw, rv), = topk_bound_kl(b, [(m["default__top_ids"], m["default__top_lp"], m["default__tail_lp"])])
    check("xref bound <= exact KL both ways, >= 0, close to exact for a small tail",
          np.all(fw <= exact_f + 1e-12) and np.all(rv <= exact_r + 1e-12) and fw.min() >= -1e-15
          and fw.mean() > 0.5 * exact_f.mean())
    mfull = item_metrics({"default": a}, tg, [], V)
    (fw2, rv2), = topk_bound_kl(b, [(mfull["default__top_ids"], mfull["default__top_lp"], mfull["default__tail_lp"])])
    check("xref bound with k = V equals the exact KL", np.max(np.abs(fw2 - exact_f)) < 1e-12
          and np.max(np.abs(rv2 - exact_r)) < 1e-12)
    da = np.zeros((1, 1000), np.float32)
    da[0, :20] = 10.0
    db = np.zeros((1, 1000), np.float32)
    db[0, 20:40] = 10.0
    md = item_metrics({"a": da, "b": db}, np.array([0]), [("a", "b")], 20)
    (dfw, drv), = topk_bound_kl(db, [(md["a__top_ids"], md["a__top_lp"], md["a__tail_lp"])])
    ofw, orv, oov = topk_overlap_kl(md["a__top_ids"], md["a__top_lp"], md["b__top_ids"], md["b__top_lp"])
    check("disjoint top-20: xref bound near the exact KL, overlap bound 0, both <= exact",
          0.99 * md["kl__a__b"][0] <= dfw[0] <= md["kl__a__b"][0] + 1e-9 and drv[0] <= md["kl__b__a"][0] + 1e-9
          and abs(ofw[0]) < 1e-12 and oov[0] == 0.0)
    ofw, orv, oov = topk_overlap_kl(m["default__top_ids"], m["default__top_lp"], m["ref__top_ids"], m["ref__top_lp"])
    check("overlap bound <= exact KL both ways", np.all(ofw <= exact_f + 1e-12) and np.all(orv <= exact_r + 1e-12)
          and ofw.mean() > 0 and oov.mean() > 0.8)
    # combine + compare round trip on synthetic shards
    shards = []
    for i, (lo, hi) in enumerate(((0, 20), (20, n))):
        sub = {k: v[lo:hi] for k, v in m.items()}
        sub["targets"] = tg[lo:hi].astype(np.int32)
        sub["pos_in_response"] = np.arange(hi - lo, dtype=np.int32)
        sub["prompt_ids"] = np.array([1, 2, 3 + i], np.int32)
        sub["response_ids"] = tg[lo:hi].astype(np.int32)
        sub["full_rows"] = np.array([0, 5], np.int32)
        sub["full__default"] = a[lo:hi][[0, 5]].astype(np.float16)
        sub["meta"] = np.array(json.dumps({"id": f"item{i}", "timing": {}}))
        shards.append(sub)
    arr, metas, ids, offs = combine_shards(shards, ["default", "ref", "default_repeat"])
    check("combine offsets", list(arr["item_offsets"]) == [0, 20, n] and ids == ["item0", "item1"])
    check("combine full index", list(arr["full_index"]) == [0, 5, 20, 25])
    filled = fill_full_logits(iter(shards), ["full__default"], [2, 2])
    check("full logits filled shard by shard", np.array_equal(filled["full__default"], arr["full__default"]))
    summ = summarize(arr, ["default", "ref", "default_repeat"], ids, offs)
    check("summary pair present", "default||ref" in summ["pairs"] and summ["n_positions"] == n)
    same = compare_arrays(arr, arr, "default", "default")
    check("compare identical runs", same["top1_agree"] == 1.0 and abs(same["topk_kl_lower_bound"]["max"]) < 1e-12
          and same["full_kl"]["max"] == 0.0)
    diff = compare_arrays(arr, arr, "default", "ref")
    check("compare different labels", diff["positions"] == n and diff["topk_kl_lower_bound"]["mean"] > 0
          and diff["target_lp_scope"]["positions_used"] == n)
    arr_gen = dict(arr)
    arr_gen["meta_json"] = np.array(json.dumps({"run": {"host": "desk"}, "items": [
        {"id": "item0", "stream_generator": {"host": "desk"}}, {"id": "item1", "stream_generator": None}]}))
    gen = compare_arrays(arr_gen, arr, "default", "ref")
    check("compare leaves out self-generated streams for target log-probs",
          gen["target_lp_scope"]["positions_used"] == n - 20 and gen["target_lp_scope"]["hosts"] == ["desk"])
    # in-process xref against a reference run on the same streams
    X, xl = xref_from_arrays(arr, ["default", "nope"])
    xa, xs = xref_item_arrays(X, xl, "item1", [1, 2, 4], tg[20:n], {"ref": b[20:n]})
    (efw, erv), = topk_bound_kl(b[20:n], [(m["default__top_ids"][20:n], m["default__top_lp"][20:n],
                                           m["default__tail_lp"][20:n])])
    check("xref item arrays", xl == ["default"] and xs["matched_positions"] == n - 20
          and np.array_equal(xa["xkl__default__ref"], efw) and np.array_equal(xa["xklr__default__ref"], erv)
          and np.array_equal(xa["xtop1__default"], m["default__top1"][20:n]))
    xb, xsb = xref_item_arrays(X, xl, "item1", [1, 2, 3], tg[20:n], {"ref": b[20:n]})
    check("xref item with another prompt is left out", xsb["matched_positions"] == 0
          and np.all(np.isnan(xb["xkl__default__ref"])) and np.all(xb["xtop1__default"] == -1))
    xsh = {k: v[20:n] for k, v in m.items()}
    xsh.update(xa)
    xsh["targets"] = tg[20:n].astype(np.int32)
    xsh["pos_in_response"] = np.arange(n - 20, dtype=np.int32)
    xsum = summarize(xsh, ["default", "ref"], ["item1"], [0, n - 20])
    cr = xsum["cross_run"]["pairs"]["default@xref||ref"]
    check("cross-run summary", cr["positions"] == n - 20 and abs(cr["kl_lower_bound"]["mean"] - efw.mean()) < 1e-15
          and cr["top1_agree"] == float(np.mean(m["default__top1"][20:n] == m["ref__top1"][20:n])))
    # RAM projection for full logits
    check("full-logit projection: stride 1 refused, stride 16 allowed",
          full_logits_gib(50, 256, 1, 248320, 5) > 25 and full_logits_gib(50, 256, 16, 248320, 5) < FULL_LOGITS_MAX_GIB
          and full_logits_gib(50, 256, 0, 248320, 5) == 0.0)
    # prompts identical to run_eval.build_prompt
    check("gsm8k prompt", build_prompt("gsm8k_500", {"question": "Q?"}) == "Q?\n\nEnd your response with: Answer: <number>")
    check("mmlu prompt", build_prompt("mmlu_500", {"question": "Q", "choices": ["w", "x", "y", "z"]})
          == "Q\nA. w\nB. x\nC. y\nD. z\n\nEnd your response with: Answer: <letter>")
    check("visible tail", visible_tail("think a</think>\n\n answer \n") == "answer")
    print(f"selftest: {len(fails)} failure(s)")
    return 1 if fails else 0


def main():
    argv = sys.argv[1:]
    if "--selftest" in argv:
        sys.exit(selftest())
    if "--combine" in argv or "--compare" in argv:
        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument("--combine", default=None, metavar="TAG")
        p.add_argument("--compare", nargs=2, default=None, metavar=("A.npz", "B.npz"))
        p.add_argument("--label-a", default="default")
        p.add_argument("--label-b", default="default")
        p.add_argument("--out-dir", default=os.path.join(LAB, "runs", "numerics"))
        a = p.parse_args(argv)
        if a.combine:
            finish_run(a.out_dir, a.combine)
        if a.compare:
            print(json.dumps(compare_arrays(load_run(a.compare[0]), load_run(a.compare[1]), a.label_a, a.label_b),
                             indent=1))
        return
    gpu_main(argv)


if __name__ == "__main__":
    main()
