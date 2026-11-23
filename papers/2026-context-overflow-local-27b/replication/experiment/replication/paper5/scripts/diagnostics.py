"""Read-only P5 scorer diagnostics, separate from the frozen inferential analysis.

Primary accuracy keeps the registered 'yes' substring rule and counts explicit
failures as incorrect. Agreement is additionally reported on valid paired
verdicts, so two failed judges cannot masquerade as agreement. No new hypothesis
test or alternative primary label is introduced here.
"""
from collections import Counter, defaultdict
import re
import string


def require(condition, message):
    if not condition:
        raise ValueError(message)


def visible(text):
    require(text is None or isinstance(text, str), "response text must be a string or null")
    if text is None or "</think>" not in text:
        return "", False
    return text.split("</think>", 1)[1].strip(), True


def normalize(text):
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def agreement(pairs):
    n = len(pairs)
    if not n:
        return {"n": 0, "raw": None, "kappa": None}
    same = sum(a == b for a, b in pairs) / n
    px, py = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    chance = px * py + (1 - px) * (1 - py)
    return {"n": n, "raw": same, "kappa": None if chance == 1 else (same - chance) / (1 - chance)}


def verdicts(generations, records, judge):
    trials = defaultdict(dict)
    for jid, row in records.items():
        meta = row.get("meta") or {}
        gid = meta.get("gen_job_id")
        require(gid in generations and meta.get("judge") == judge, "foreign generation/judge in verdict journal")
        base = gid + "|judge=" + judge
        require(jid in (base, base + "|retry"), "unexpected judgment job ID")
        require(type(row.get("ok")) is bool, "judge outcome must be explicit")
        require(row.get("text") is None or isinstance(row["text"], str), "invalid judge text")
        which = "retry" if jid.endswith("|retry") else "base"
        require(which not in trials[gid], "duplicate judgment attempt")
        trials[gid][which] = row
    out = {}
    for gid, gen in generations.items():
        require(type(gen.get("ok")) is bool, "generation outcome must be explicit")
        rows = trials.get(gid, {})
        require(not gen["ok"] or "base" in rows, "required judge pass has not completed")
        base = rows.get("base")
        empty_base = bool(base and base["ok"] and not (base.get("text") or "").strip())
        if "retry" in rows:
            require(judge == "gptoss" and empty_base, "retry does not follow the registered empty-reply rule")
        require(not (judge == "gptoss" and empty_base) or "retry" in rows, "empty primary verdict still needs retry")
        retained = rows.get("retry", base)
        text = (retained.get("text") or "").strip() if retained and retained["ok"] else ""
        valid = bool(gen["ok"] and text)
        out[gid] = {"label": bool(gen["ok"] and "yes" in text.lower()), "valid": valid,
                    "failed": bool(gen["ok"] and not text),
                    "noncanonical": valid and text.lower() not in ("yes", "no", "yes.", "no.")}
    return out


def summarize(items, prompts, generations, judges):
    """Returns aggregates only, with explicit pair-specific denominators."""
    by_id = {it["question_id"]: it for it in items}
    require(len(by_id) == len(items), "duplicate diagnostic item ID")
    expected = {p["job_id"]: p for p in prompts}
    require(len(expected) == len(prompts) and expected.keys() == generations.keys(), "incomplete diagnostic generation inventory")
    require(set(judges) in ({"gptoss"}, {"gptoss", "self"}), "wrong diagnostic judge inventory")
    scored = {name: verdicts(generations, records, name) for name, records in judges.items()}
    cells = defaultdict(list)
    for gid, p in expected.items():
        require(p["question_id"] in by_id, "diagnostic item missing from corpus")
        cells[f"{p['arm']}@{p['budget']}"].append((gid, by_id[p["question_id"]]))
    result = {}
    for cell, records in cells.items():
        pairs = defaultdict(list)
        failed = Counter()
        noncanonical = Counter()
        types = defaultdict(lambda: [0, 0])
        n_generated = n_unclosed = n_empty = n_empty_credited = 0
        abstention = [0, 0]
        for gid, item in records:
            gen, first = generations[gid], scored["gptoss"][gid]
            ans, closed = visible(gen.get("text") if gen["ok"] else None)
            n_generated += gen["ok"]
            n_unclosed += gen["ok"] and not closed
            n_empty += gen["ok"] and not ans
            n_empty_credited += gen["ok"] and not ans and first["label"]
            types[item["question_type"]][0] += first["label"]
            types[item["question_type"]][1] += 1
            if "_abs" in item["question_id"]:
                abstention[0] += first["label"]
                abstention[1] += 1
            available = {}
            for name in scored:
                verdict = scored[name][gid]
                failed[name] += verdict["failed"]
                noncanonical[name] += verdict["noncanonical"]
                if verdict["valid"]:
                    available[name] = verdict["label"]
            short = (not item["question_id"].startswith("locomo-") and "_abs" not in item["question_id"]
                     and item["question_type"] != "single-session-preference")
            if gen["ok"] and short:
                gold = normalize(item["answer"])
                available["containment"] = bool(gold) and gold in normalize(ans)
            for x, y in (("gptoss", "self"), ("gptoss", "containment"), ("self", "containment")):
                if x in available and y in available:
                    pairs[x + "_vs_" + y].append((available[x], available[y]))
        result[cell] = {
            "n": len(records), "generated": n_generated, "generation_failures": len(records) - n_generated,
            "judge_failures_on_generated": {k: failed[k] for k in scored},
            "noncanonical_verdicts": {k: noncanonical[k] for k in scored},
            "unterminated_thinking": n_unclosed, "empty_visible_answers": n_empty,
            "empty_visible_answers_credited": n_empty_credited,
            "valid_pair_agreement": {key: agreement(pairs[key]) for key in (
                ("gptoss_vs_self", "gptoss_vs_containment", "self_vs_containment") if "self" in scored
                else ("gptoss_vs_containment",))},
            "by_type": {key: {"correct": k, "n": n, "accuracy": k / n} for key, (k, n) in sorted(types.items())},
            "abstention": {"correct": abstention[0], "n": abstention[1],
                           "accuracy": abstention[0] / abstention[1] if abstention[1] else None},
        }
    return {"schema": "p5-valid-verdict-diagnostics-v1", "cells": result,
            "agreement_denominator": "successful generation and nonempty successful verdict from each compared judge; containment is LongMemEval-only and excludes preference/abstention items",
            "interpretation": "agreement is not correctness validation; accuracy still counts explicit failures as incorrect"}
