#!/usr/bin/env python3
"""Strict, descriptive P5 follow-up analysis. Never infer outcomes for unrun jobs.

LoCoMo output contains aggregates only. Token F1 uses the primary normalizer;
paired F1 uncertainty resamples whole conversations, not independent questions.
This is a local analysis stage, not an independent public release verifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import random
import re
import statistics
import string

from followups import load_module
from workspace import (LEGS, ROOT, Workspace, canonical, digest_text, integer,
                       probability, read_json, sha, terminal_rows)

STATS = load_module("p5_followup_statistics", ROOT / "paper4/scripts/verify_claims.py")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260920


def visible_answer(text):
    if text is None:
        return "", False
    if not isinstance(text, str):
        raise ValueError("generation text must be a string or null")
    if "</think>" not in text:
        return "", False
    return text.split("</think>", 1)[1].strip(), True


def normalize(text):
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(answer, gold):
    a, g = normalize(answer).split(), normalize(gold).split()
    if not a or not g:
        return float(a == g)
    same = sum((Counter(a) & Counter(g)).values())
    return 2 * same / (len(a) + len(g)) if same else 0.0


def accuracy(labels):
    n, correct = len(labels), sum(labels)
    if not n:
        raise ValueError("empty accuracy denominator")
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "wilson95": list(STATS.wilson_interval(correct, n))}


def paired(arm, reference):
    if arm.keys() != reference.keys() or not arm:
        raise ValueError("paired comparison requires identical nonempty item IDs")
    both = sum(arm[qid] and reference[qid] for qid in arm)
    gain = sum(arm[qid] and not reference[qid] for qid in arm)
    loss = sum(reference[qid] and not arm[qid] for qid in arm)
    neither = len(arm) - both - gain - loss
    diff, low, high, phi = STATS.newcombe_paired_risk_difference_interval(both, loss, gain, neither)
    return {"n": len(arm), "both": both, "arm_only": gain, "reference_only": loss,
            "neither": neither, "difference": diff, "newcombe95": [low, high],
            "mcnemar_p_unadjusted": STATS.mcnemar_exact_two_sided(gain, loss),
            "interpretation": "descriptive, pointwise; outside the primary Holm family"}


def quantile(values, p):
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def clustered_f1(values, clusters, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    if not values or values.keys() != clusters.keys():
        raise ValueError("cluster bootstrap needs a cluster for every paired item")
    integer(draws, "bootstrap draws", 100)
    integer(seed, "bootstrap seed")
    sums = defaultdict(float)
    counts = Counter()
    for qid, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("F1 values/differences must be finite and within [-1, 1]")
        cluster = clusters[qid]
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("missing conversation cluster")
        sums[cluster] += value
        counts[cluster] += 1
    names = sorted(counts)
    if len(names) < 2:
        raise ValueError("cluster bootstrap needs at least two conversations")
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        chosen = rng.choices(names, k=len(names))
        samples.append(sum(sums[c] for c in chosen) / sum(counts[c] for c in chosen))
    samples.sort()
    return {"n": len(values), "conversations": len(names), "mean": statistics.mean(values.values()),
            "percentile95": [quantile(samples, .025), quantile(samples, .975)],
            "draws": draws, "seed": seed, "resampling_unit": "conversation",
            "estimand": "item-weighted mean; ratio recomputed in each conversation resample",
            "interpretation": "descriptive, pointwise percentile interval"}


def judge_labels(records, generations, judge):
    selected = {}
    for jid, row in records.items():
        meta = row.get("meta") or {}
        gid = meta.get("gen_job_id")
        if gid not in generations or meta.get("judge") != judge:
            raise ValueError("judge record belongs to another generation/judge")
        base = gid + "|judge=" + judge
        if jid not in (base, base + "|retry"):
            raise ValueError("unexpected judge job identity")
        if meta.get("gen_record_sha256") != digest_text(canonical(generations[gid])):
            raise ValueError("judge label is bound to a different generation record")
        if row["ok"] and row.get("text") is not None and not isinstance(row["text"], str):
            raise ValueError("judge text must be a string or null")
        selected.setdefault(gid, {})["retry" if jid.endswith("|retry") else "base"] = row
    labels, failed = {}, set()
    for gid, generation in generations.items():
        trials = selected.get(gid, {})
        if generation["ok"] and "base" not in trials:
            raise ValueError(f"{judge}: judging incomplete")
        if "retry" in trials:
            base = trials.get("base")
            if judge != "gptoss" or base is None or not base["ok"] or (base.get("text") or "").strip():
                raise ValueError("judge retry does not follow an empty successful reply")
        base = trials.get("base", {})
        if judge == "gptoss" and base.get("ok") and not (base.get("text") or "").strip() and "retry" not in trials:
            raise ValueError("empty judge reply still needs its declared retry")
        retained = trials.get("retry", base)
        content = retained.get("text") if retained.get("ok") else None
        if not (content or "").strip():
            failed.add(gid)
        labels[gid] = bool(generation["ok"] and "yes" in (content or "").lower())
    return labels, failed


def analyze_records(leg, arms, items, prompts, generations, judges, manifest_sha256,
                    *, bootstrap_draws=BOOTSTRAP_DRAWS):
    by_item = {it["question_id"]: it for it in items}
    if len(by_item) != len(items) or not items:
        raise ValueError("invalid item inventory")
    expected = {p["job_id"]: p for p in prompts}
    if len(expected) != len(prompts) or set(generations) != set(expected):
        raise ValueError("missing, duplicate or unexpected follow-up generation jobs")
    required_judges = {"gptoss", "self"} if leg == "cross-model" else {"gptoss"}
    if set(judges) != required_judges:
        raise ValueError("follow-up is missing its declared judge journals")
    grouped = {arm: {} for arm in arms}
    answers, unclosed = {}, set()
    for gid, p in expected.items():
        g = generations[gid]
        if type(g.get("ok")) is not bool:
            raise ValueError("generation outcome not explicit")
        meta = g.get("meta") or {}
        if meta.get("leg") != leg or meta.get("model") != LEGS[leg]["model"] or meta.get("prompt_manifest_sha256") != manifest_sha256:
            raise ValueError("generation identity/freeze does not match this follow-up")
        if meta.get("prompt_sha256") != p["prompt_sha256"]:
            raise ValueError("generation attempt is bound to another prompt")
        if g["ok"] and (g.get("prompt_sha256") != p["prompt_sha256"] or g.get("server_prompt_tokens") != p["n_prompt"]):
            raise ValueError("generation prompt content/token parity mismatch")
        qid, arm = p["question_id"], p["arm"]
        if arm not in grouped or qid not in by_item or qid in grouped[arm]:
            raise ValueError("unexpected or duplicated item/arm")
        grouped[arm][qid] = gid
        answers[gid], closed = visible_answer(g.get("text") if g["ok"] else None)
        if g["ok"] and not closed:
            unclosed.add(gid)
    for arm, ids in grouped.items():
        if ids.keys() != by_item.keys():
            raise ValueError("unpaired or incomplete follow-up cell: " + arm)
    scored = {judge: judge_labels(rows, generations, judge) for judge, rows in judges.items()}
    primary, failures = scored["gptoss"]
    labels = {arm: {qid: primary[gid] for qid, gid in ids.items()} for arm, ids in grouped.items()}
    cells = {}
    for arm, ids in grouped.items():
        cells[arm] = accuracy(list(labels[arm].values()))
        cells[arm].update(failed_generations=sum(not generations[gid]["ok"] for gid in ids.values()),
                          judge_failures=sum(gid in failures and generations[gid]["ok"] for gid in ids.values()),
                          unterminated_thinking=sum(gid in unclosed for gid in ids.values()))
        if "self" in scored:
            second = {qid: scored["self"][0][gid] for qid, gid in ids.items()}
            successful = [qid for qid, gid in ids.items() if generations[gid]["ok"]]
            cells[arm]["secondary_judge"] = accuracy(list(second.values()))
            cells[arm]["judge_agreement"] = {"n": len(successful), "raw":
                sum(labels[arm][qid] == second[qid] for qid in successful) / len(successful) if successful else None}
    result = {"schema": "paper5-followup-analysis-v1", "leg": leg, "model": LEGS[leg]["model"],
              "budget": 4096, "cells": cells,
              "contrasts_vs_T": {arm: paired(labels[arm], labels["T"]) for arm in arms if arm != "T"},
              "confirmatory": False, "publication": "aggregates-only" if leg == "locomo" else "curated-item-level"}
    if leg == "locomo":
        clusters = {qid: it["conversation_id"] for qid, it in by_item.items()}
        f1 = {arm: {qid: token_f1(answers[gid], by_item[qid]["answer"]) if generations[gid]["ok"] else 0.0
                    for qid, gid in ids.items()} for arm, ids in grouped.items()}
        result["token_f1"] = {arm: clustered_f1(scores, clusters, draws=bootstrap_draws) for arm, scores in f1.items()}
        result["token_f1_differences_vs_T"] = {
            arm: clustered_f1({qid: f1[arm][qid] - f1["T"][qid] for qid in by_item}, clusters, draws=bootstrap_draws)
            for arm in arms if arm != "T"}
        result["accuracy_interval_caution"] = "Wilson/Newcombe/McNemar treat questions as independent; shared-conversation dependence limits that reading. F1 intervals resample conversations."
    return result


def analyze(work):
    _, items = work.load()
    prompts = work.frozen_prompts()
    manifest_path = work.path("PROMPTS_MANIFEST.json")
    gen_path = work.path("journals/grid.jsonl")
    labels = ("gptoss", "self") if work.leg == "cross-model" else ("gptoss",)
    judge_paths = {label: work.path(f"journals/judge_{label}__grid.jsonl") for label in labels}
    paths = [manifest_path, gen_path, *judge_paths.values()]
    generations = terminal_rows(gen_path)
    judges = {label: terminal_rows(path) for label, path in judge_paths.items()}
    stages = {"generate": (gen_path, generations)}
    stages.update({"judge-" + label: (judge_paths[label], judges[label]) for label in labels})
    for stage, (journal, records) in stages.items():
        signature = work.path(f"execution/{stage}.json")
        finished = work.path(f"execution/{stage}-finished.json")
        receipt = read_json(finished)
        identity = read_json(signature)
        if identity.get("schema") != "paper5-followup-execution-v1" or identity.get("leg") != work.leg or identity.get("stage") != stage:
            raise ValueError("execution receipt belongs to another follow-up stage")
        if identity.get("workspace_sha256") != sha(work.path("WORKSPACE.json")) or identity.get("prompt_manifest_sha256") != sha(manifest_path):
            raise ValueError("execution receipt belongs to another input freeze")
        if receipt.get("complete") is not True or receipt.get("journal_sha256") != sha(journal) or receipt.get("execution_fingerprint_sha256") != sha(signature):
            raise ValueError("stage completion receipt is absent or stale")
        if any(row.get("execution_fingerprint_sha256") != sha(signature) for row in records.values()):
            raise ValueError("journal mixes incompatible execution identities")
        paths.extend([signature, finished])
    inputs = {str(path): sha(path) for path in paths}
    result = analyze_records(work.leg, work.expected_arms(), items, prompts, generations, judges, sha(manifest_path))
    work.frozen_prompts()
    if inputs != {str(path): sha(path) for path in paths}:
        raise ValueError("follow-up evidence changed during analysis")
    result["inputs"] = inputs
    result["analysis_code_sha256"] = sha(__file__)
    work.write_json_once("analysis/analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", choices=tuple(LEGS))
    args = parser.parse_args()
    try:
        result = analyze(Workspace(args.leg))
        print(json.dumps({"leg": args.leg, "cells": result["cells"], "confirmatory": False}, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"P5 follow-up analysis not ready: {exc}\n")


if __name__ == "__main__":
    main()
