#!/usr/bin/env python3
"""Standalone P5 numerical replay from sanitized scores and LoCoMo aggregates.

This checks scored records, not the withheld text or live execution. LoCoMo
F1 intervals are archived, privately verified aggregates; they are NOT publicly
recomputed from question/conversation-level scores.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from types import ModuleType

TEXT_MODULES = {"draft_support": "support_core.py", "reporting": "reporting_core.py", "followup_results": "followup_text.py",
                "result_sections": "sections_text.py", "manuscript": "manuscript_text.py", "p5_macros": "macros_core.py"}
FILES = {"observations.json", "results.json", "statistics_core.py", "primary_core.py", "check_results.py", "README.md",
         *TEXT_MODULES.values(), "figure_core.py", "rebuild_results.py", "claims.tex"}
ARMS = ("T", "S", "C", "M")
TYPES = {"single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"}
LOC_TYPES = {"locomo-cat1", "locomo-cat2", "locomo-cat4"}
DIAG_INT = {"messages_kept", "messages_total", "rounds_kept", "rounds_total", "sessions_kept", "sessions_total",
            "sessions", "percentile", "fallback_chunks_dropped", "chunks", "history_chars_kept", "window_tokens"}
DIAG_FRACTION = {"evidence_turns_kept", "evidence_sessions_summarized", "evidence_word_recall"}
DIAG_SECONDS = {"policy_seconds", "selection_seconds"}
DIAG_NAMES = DIAG_INT | DIAG_FRACTION | DIAG_SECONDS | {"fallback"}
DIAGNOSTIC_SCOPE = "successful generation and nonempty successful verdict from each compared judge; containment is LongMemEval-only and excludes preference/abstention items"
DIAGNOSTIC_LIMIT = "agreement is not correctness validation; accuracy still counts explicit failures as incorrect"
PAIR_LIMIT = "descriptive, pointwise; outside the primary Holm family"
F1_LIMIT = "descriptive, pointwise percentile interval"
F1_ESTIMAND = "item-weighted mean; ratio recomputed in each conversation resample"
LOC_CAUTION = "Wilson/Newcombe/McNemar treat questions as independent; shared-conversation dependence limits that reading. F1 intervals resample conversations."
SCOPE = "scored-record replay; no text rescoring, execution attestation, or public LoCoMo bootstrap reconstruction"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(value, keys):
    require(type(value) is dict and value.keys() == set(keys), "unexpected or missing fields")


def integer(value, low=0, high=None):
    require(type(value) is int and low <= value and (high is None or value <= high), "invalid integer")
    return value


def number(value, low=0., high=None):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value and
            (high is None or value <= high), "invalid finite number")
    return value


def boolean(value):
    require(type(value) is bool, "explicit boolean required")
    return value


def digest(value):
    require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None, "invalid SHA256")
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError("non-finite JSON value: " + value)
    def floating(value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else nonfinite(value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=floating)


def load_source(path, namespace=None):
    module = ModuleType(Path(path).stem)
    if namespace:
        module.__dict__.update(namespace)
    exec(compile(Path(path).read_text(), Path(path).name, "exec"), module.__dict__)
    return module


@contextmanager
def text_modules(sources):
    """Use only the bundled pure modules, even inside an existing repo process."""
    absent = object()
    old = {name: sys.modules.get(name, absent) for name in TEXT_MODULES}
    loaded = {}
    try:
        for name, filename in TEXT_MODULES.items():
            module = ModuleType(name)
            exec(compile(sources[filename], filename, "exec"), module.__dict__)
            sys.modules[name] = loaded[name] = module
        yield loaded
    finally:
        for name, module in old.items():
            if module is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def claims_text(data, results, sources):
    # Canonical JSON order matches the real analyzer's saved claims, and makes
    # definition order independent of a caller's in-memory dictionary order.
    results = json.loads(canonical(results))
    primary = results["primary"]["analysis"]
    evidence = {"cells": {key: value["n"] for key, value in primary["cells"].items()},
                "completion_cap": data["completion_cap"]}
    with text_modules(sources) as modules:
        narrative = modules["manuscript"].render(primary, evidence,
            followups={leg: results[leg] for leg in ("cross-model", "locomo")},
            diagnostics=results["primary"]["diagnostics"])
        macros = modules["p5_macros"].macros(primary)
    source_sha = hashlib.sha256((canonical(data) + "\n").encode()).hexdigest()
    return ("% Checked scored-data replay; not publication approval.\n"
            f"% canonical observations SHA-256: {source_sha}\n" + "\n".join(macros) + "\n" + narrative)


def compare(actual, expected, path="results"):
    if type(expected) is dict:
        exact(actual, expected)
        for key in expected:
            compare(actual[key], expected[key], path + "." + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "different list at " + path)
        for i, value in enumerate(expected):
            compare(actual[i], value, path + f"[{i}]")
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual), "invalid result number at " + path)
        # Nonzero exact p-values must never pass as zero via absolute tolerance.
        tolerance = 0. if path.endswith((".mcnemar_p", ".mcnemar_p_unadjusted", ".holm_p")) else 2e-12
        require(math.isclose(actual, expected, rel_tol=5e-10, abs_tol=tolerance), "different number at " + path)
    else:
        require(type(actual) is type(expected) and actual == expected, "different value at " + path)


def validate_records(records, items, leg, cap):
    require(type(records) is list, "observations must be a list")
    cells = defaultdict(dict)
    allowed = {f"{a}@{b}" for a in ("T", "S", "C", "M", "Cw") for b in (1024, 4096)}
    allowed |= {"Mbm25@4096", "O@4096", "E@None"}
    if leg == "cross-model":
        allowed = {a + "@4096" for a in ARMS}
    for row in records:
        exact(row, {"item_id", "arm", "budget", "generated", "closed_thinking", "empty_visible_answer",
                    "cap_hit", "n_prompt", "completion_tokens", "completion_tokens_harness", "seconds",
                    "gptoss", "self", "containment", "diag", "prompt_sha256", "response_sha256"})
        qid = row["item_id"]
        require(type(qid) is str and qid in items, "unknown item")
        require(type(row["arm"]) is str and (row["budget"] is None or type(row["budget"]) is int), "invalid cell")
        key = f"{row['arm']}@{row['budget']}"
        require(key in allowed and qid not in cells[key], "unexpected or duplicated cell/item")
        item = items[qid]
        for name in ("generated", "closed_thinking", "empty_visible_answer", "cap_hit"):
            boolean(row[name])
        ok = row["generated"]
        require(ok or (not row["closed_thinking"] and row["empty_visible_answer"] and not row["cap_hit"]),
                "failed generation has successful-response fields")
        require(row["closed_thinking"] or row["empty_visible_answer"], "unclosed thinking cannot have visible answer")
        integer(row["n_prompt"], 1, row["budget"])
        # The cap bounds the server's count. The harness count re-encodes the decoded text and may pass the cap
        # only on a capped reply, by at most 1% (as in followup_results.validate_generation_requests)
        if row["completion_tokens"] is not None:
            integer(row["completion_tokens"], 0, cap)
        if row["completion_tokens_harness"] is not None:
            harness = integer(row["completion_tokens_harness"], 0, cap + cap // 100)
            require(harness <= cap or (row["cap_hit"] and row["completion_tokens"] in (None, cap)),
                    "completion count above the cap on a reply that was not capped")
        if row["seconds"] is not None:
            number(row["seconds"])
        require(ok or all(row[x] is None for x in ("completion_tokens", "completion_tokens_harness", "seconds")),
                "failed terminal contains undeclared measurements")
        if ok:
            integer(row["completion_tokens_harness"], 0, cap + cap // 100)
        for name in ("gptoss", "self"):
            v = row[name]
            exact(v, {"label", "valid", "failed", "noncanonical"})
            for value in v.values():
                boolean(value)
            require(v["failed"] is (ok and not v["valid"]), "judge failure denominator changed")
            require(ok or not any(v.values()), "judgment on failed generation")
            require(v["valid"] or not (v["label"] or v["noncanonical"]), "invalid verdict credited")
        eligible = ok and not item["abstention"] and item["question_type"] != "single-session-preference"
        if eligible:
            boolean(row["containment"])
            require(not row["empty_visible_answer"] or row["containment"] is False, "empty answer containment")
        else:
            require(row["containment"] is None, "ineligible containment score")
        require(type(row["diag"]) is dict and row["diag"].keys() <= DIAG_NAMES, "unexpected policy diagnostic")
        for name, value in row["diag"].items():
            if name in DIAG_INT:
                integer(value, 0, 99 if name == "percentile" else None)
            elif name in DIAG_FRACTION:
                if value is not None:
                    number(value, 0., 1.)
            elif name in DIAG_SECONDS:
                number(value)
            else:
                boolean(value)
        for name in ("prompt_sha256", "response_sha256"):
            digest(row[name])
        cells[key][qid] = row
    require(cells.keys() == allowed, "incomplete cell inventory")
    for key, rows in cells.items():
        require(bool(rows) and (rows.keys() <= items.keys() if key == "O@4096" else rows.keys() == items.keys()),
                "missing/unpaired observations")
    return cells


def diagnostic_agreement(pairs):
    n = len(pairs)
    if not n:
        return {"n": 0, "raw": None, "kappa": None}
    raw = sum(a == b for a, b in pairs) / n
    px, py = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    chance = px*py + (1-px)*(1-py)
    return {"n": n, "raw": raw, "kappa": None if chance == 1 else (raw-chance)/(1-chance)}


def diagnostics(cells, items):
    out = {}
    for key, records in cells.items():
        pairs, types = defaultdict(list), defaultdict(list)
        abstention, empty_credited = [], 0
        for qid, row in records.items():
            item = items[qid]
            types[item["question_type"]].append(row["gptoss"]["label"])
            if item["abstention"]:
                abstention.append(row["gptoss"]["label"])
            empty_credited += row["generated"] and row["empty_visible_answer"] and row["gptoss"]["label"]
            available = {name: row[name]["label"] for name in ("gptoss", "self") if row[name]["valid"]}
            if row["containment"] is not None:
                available["containment"] = row["containment"]
            for x, y in (("gptoss", "self"), ("gptoss", "containment"), ("self", "containment")):
                if x in available and y in available:
                    pairs[x + "_vs_" + y].append((available[x], available[y]))
        rows = list(records.values())
        generated = sum(r["generated"] for r in rows)
        out[key] = {
            "n": len(rows), "generated": generated, "generation_failures": len(rows)-generated,
            "judge_failures_on_generated": {name: sum(r[name]["failed"] for r in rows) for name in ("gptoss", "self")},
            "noncanonical_verdicts": {name: sum(r[name]["noncanonical"] for r in rows) for name in ("gptoss", "self")},
            "unterminated_thinking": sum(r["generated"] and not r["closed_thinking"] for r in rows),
            "empty_visible_answers": sum(r["generated"] and r["empty_visible_answer"] for r in rows),
            "empty_visible_answers_credited": empty_credited,
            "valid_pair_agreement": {name: diagnostic_agreement(pairs[name]) for name in
                                     ("gptoss_vs_self", "gptoss_vs_containment", "self_vs_containment")},
            "by_type": {t: {"correct": sum(v), "n": len(v), "accuracy": sum(v)/len(v)} for t, v in sorted(types.items())},
            "abstention": {"correct": sum(abstention), "n": len(abstention),
                           "accuracy": sum(abstention)/len(abstention) if abstention else None}}
    return {"schema": "p5-valid-verdict-diagnostics-v1", "cells": out,
            "agreement_denominator": DIAGNOSTIC_SCOPE, "interpretation": DIAGNOSTIC_LIMIT}


def primary(cells, items, core):
    recs = defaultdict(dict)
    for key, rows in cells.items():
        arm, budget = key.split("@")
        for qid, row in rows.items():
            recs[(arm, None if budget == "None" else int(budget))][qid] = {
                "generated": row["generated"], "closed_thinking": row["closed_thinking"],
                "judge": row["gptoss"]["label"], "judge_failed": not row["gptoss"]["valid"],
                "self_judge": row["self"]["label"], "containment": row["containment"],
                "n_prompt": row["n_prompt"], "completion_tokens": row["completion_tokens"],
                "finish_reason": "length" if row["cap_hit"] else None, "seconds": row["seconds"],
                "type": items[qid]["question_type"], "abstention": items[qid]["abstention"], "diag": row["diag"]}
    # JSON normalizes numeric ranking keys in the frozen analyzer's output.
    return json.loads(json.dumps(core.aggregate(recs), allow_nan=False))


def accuracy(k, n, stats):
    return {"n": n, "correct": k, "accuracy": k/n, "wilson95": list(stats.wilson_interval(k, n))}


def paired_counts(counts, stats):
    exact(counts, {"both", "arm_only", "reference_only", "neither"})
    a, b, c, d = [integer(counts[k], 0, 500) for k in ("both", "arm_only", "reference_only", "neither")]
    diff, lo, hi, _ = stats.newcombe_paired_risk_difference_interval(a, c, b, d)
    return {"n": a+b+c+d, **counts, "difference": diff, "newcombe95": [lo, hi],
            "mcnemar_p_unadjusted": stats.mcnemar_exact_two_sided(b, c), "interpretation": PAIR_LIMIT}


def cross_model(cells, stats):
    out = {"schema": "paper5-followup-analysis-v1", "leg": "cross-model", "model": "qwen3-8b",
           "budget": 4096, "confirmatory": False, "publication": "curated-item-level", "cells": {}, "contrasts_vs_T": {}}
    for arm in ARMS:
        rows = list(cells[arm + "@4096"].values())
        ok = [r for r in rows if r["generated"]]
        out["cells"][arm] = {
            **accuracy(sum(r["gptoss"]["label"] for r in rows), len(rows), stats),
            "failed_generations": len(rows)-len(ok), "judge_failures": sum(r["gptoss"]["failed"] for r in rows),
            "unterminated_thinking": sum(not r["closed_thinking"] for r in ok),
            "secondary_judge": accuracy(sum(r["self"]["label"] for r in rows), len(rows), stats),
            "judge_agreement": {"n": len(ok), "raw": sum(r["gptoss"]["label"] == r["self"]["label"] for r in ok)/len(ok) if ok else None}}
        if arm != "T":
            counts = Counter()
            for qid, r in cells[arm + "@4096"].items():
                x, y = r["gptoss"]["label"], cells["T@4096"][qid]["gptoss"]["label"]
                counts["both" if x and y else "arm_only" if x else "reference_only" if y else "neither"] += 1
            out["contrasts_vs_T"][arm] = paired_counts({k: counts[k] for k in ("both", "arm_only", "reference_only", "neither")}, stats)
    return out


def validate_f1(row, difference=False):
    exact(row, {"n", "conversations", "mean", "percentile95", "draws", "seed", "resampling_unit", "estimand", "interpretation"})
    integer(row["n"], 200, 200)
    integer(row["conversations"], 2, 200)
    integer(row["draws"], 10000, 10000)
    integer(row["seed"], 20260920, 20260920)
    require(row["resampling_unit"] == "conversation" and row["estimand"] == F1_ESTIMAND and
            row["interpretation"] == F1_LIMIT, "changed F1 bootstrap description")
    low = -1. if difference else 0.
    number(row["mean"], low, 1.)
    require(type(row["percentile95"]) is list and len(row["percentile95"]) == 2, "invalid F1 interval")
    lo, hi = [number(x, low, 1.) for x in row["percentile95"]]
    require(lo <= hi, "reversed F1 interval")


def validate_locomo_diagnostics(value, analysis):
    exact(value, {"schema", "cells", "agreement_denominator", "interpretation"})
    require(value["schema"] == "p5-valid-verdict-diagnostics-v1" and value["agreement_denominator"] == DIAGNOSTIC_SCOPE
            and value["interpretation"] == DIAGNOSTIC_LIMIT, "changed diagnostic description")
    exact(value["cells"], {a + "@4096" for a in analysis["cells"]})
    for arm, cell in analysis["cells"].items():
        d = value["cells"][arm + "@4096"]
        exact(d, {"n", "generated", "generation_failures", "judge_failures_on_generated", "noncanonical_verdicts",
                  "unterminated_thinking", "empty_visible_answers", "empty_visible_answers_credited",
                  "valid_pair_agreement", "by_type", "abstention"})
        for key in ("n", "generated", "generation_failures", "unterminated_thinking", "empty_visible_answers", "empty_visible_answers_credited"):
            integer(d[key], 0, 200)
        require(d["n"] == 200 and d["generation_failures"] == cell["failed_generations"]
                and d["generated"] == 200-d["generation_failures"]
                and d["unterminated_thinking"] == cell["unterminated_thinking"]
                and d["unterminated_thinking"] <= d["empty_visible_answers"] <= d["generated"]
                and d["empty_visible_answers_credited"] <= min(d["empty_visible_answers"], cell["correct"]), "inconsistent LoCoMo diagnostics")
        for key in ("judge_failures_on_generated", "noncanonical_verdicts"):
            exact(d[key], {"gptoss"})
            integer(d[key]["gptoss"], 0, d["generated"])
        require(d["judge_failures_on_generated"]["gptoss"] == cell["judge_failures"] and
                d["noncanonical_verdicts"]["gptoss"] <= d["generated"]-cell["judge_failures"], "wrong verdict counts")
        compare(d["valid_pair_agreement"], {"gptoss_vs_containment": {"n": 0, "raw": None, "kappa": None}})
        require(type(d["by_type"]) is dict and d["by_type"].keys() == LOC_TYPES, "unexpected LoCoMo categories")
        for category, n in (("locomo-cat1", 70), ("locomo-cat2", 70), ("locomo-cat4", 60)):
            r = d["by_type"][category]
            exact(r, {"correct", "n", "accuracy"})
            integer(r["n"], n, n)
            integer(r["correct"], 0, n)
            compare(r["accuracy"], r["correct"]/n)
        require(sum(r["correct"] for r in d["by_type"].values()) == cell["correct"], "inconsistent category totals")
        compare(d["abstention"], {"correct": 0, "n": 0, "accuracy": None})


def locomo(value, primary_analysis, stats):
    exact(value, {"selected_policy", "cells", "paired_counts", "token_f1", "token_f1_differences_vs_T", "diagnostics"})
    selected = max(("S", "M", "C"), key=lambda a: primary_analysis["cells"][a + "@4096"]["correct"])
    require(value["selected_policy"] == selected, "LoCoMo selection/tie rule changed")
    exact(value["cells"], {"T", selected})
    exact(value["paired_counts"], {selected})
    analysis = {"schema": "paper5-followup-analysis-v1", "leg": "locomo", "model": "qwen38-exl3",
                "budget": 4096, "confirmatory": False, "publication": "aggregates-only", "cells": {},
                "contrasts_vs_T": {selected: paired_counts(value["paired_counts"][selected], stats)},
                "token_f1": value["token_f1"], "token_f1_differences_vs_T": value["token_f1_differences_vs_T"],
                "accuracy_interval_caution": LOC_CAUTION}
    for arm, cell in value["cells"].items():
        exact(cell, {"n", "correct", "failed_generations", "judge_failures", "unterminated_thinking"})
        integer(cell["n"], 200, 200)
        for count in cell.values():
            integer(count, 0, 200)
        require(cell["correct"] <= 200-cell["failed_generations"]-cell["judge_failures"]
                and cell["unterminated_thinking"] <= 200-cell["failed_generations"], "inconsistent LoCoMo counts")
        analysis["cells"][arm] = {**cell, **accuracy(cell["correct"], 200, stats)}
    c = analysis["contrasts_vs_T"][selected]
    require(c["n"] == 200 and c["both"]+c["arm_only"] == analysis["cells"][selected]["correct"]
            and c["both"]+c["reference_only"] == analysis["cells"]["T"]["correct"], "LoCoMo paired counts differ")
    exact(value["token_f1"], {"T", selected})
    exact(value["token_f1_differences_vs_T"], {selected})
    for row in value["token_f1"].values():
        validate_f1(row)
    validate_f1(value["token_f1_differences_vs_T"][selected], True)
    rows = [*value["token_f1"].values(), value["token_f1_differences_vs_T"][selected]]
    require(len({r["conversations"] for r in rows}) == 1, "F1 cluster denominators differ")
    compare(rows[-1]["mean"], value["token_f1"][selected]["mean"]-value["token_f1"]["T"]["mean"])
    validate_locomo_diagnostics(value["diagnostics"], analysis)
    return {"analysis": analysis, "diagnostics": value["diagnostics"]}


def rebuild(data, stats, core):
    exact(data, {"schema", "scope", "completion_cap", "items", "primary", "cross-model", "locomo_aggregates", "source_hashes"})
    require(data["schema"] == "p5-public-observations-v1" and data["scope"] == SCOPE, "wrong public data identity/scope")
    require(type(data["completion_cap"]) is int and data["completion_cap"] in (8192, 16384), "wrong pilot completion cap")
    require(type(data["source_hashes"]) is dict and bool(data["source_hashes"]), "missing source binding")
    for key, value in data["source_hashes"].items():
        digest(key)
        digest(value)
    require(type(data["items"]) is list and len(data["items"]) == 500, "expected 500 LongMemEval items")
    items, sources = {}, set()
    for i, row in enumerate(data["items"]):
        exact(row, {"id", "source_id_sha256", "question_type", "abstention"})
        require(row["id"] == f"lme-{i:03}" and type(row["question_type"]) is str and row["question_type"] in TYPES,
                "invalid item metadata/order")
        boolean(row["abstention"])
        digest(row["source_id_sha256"])
        require(row["source_id_sha256"] not in sources, "duplicate source item")
        sources.add(row["source_id_sha256"])
        items[row["id"]] = row
    primary_cells = validate_records(data["primary"], items, "primary", data["completion_cap"])
    cross_cells = validate_records(data["cross-model"], items, "cross-model", data["completion_cap"])
    p = primary(primary_cells, items, core)
    return {"schema": "p5-public-results-v1", "primary": {"analysis": p, "diagnostics": diagnostics(primary_cells, items)},
            "cross-model": {"analysis": cross_model(cross_cells, stats), "diagnostics": diagnostics(cross_cells, items)},
            "locomo": locomo(data["locomo_aggregates"], p, stats)}


def check(bundle):
    bundle = Path(bundle)
    require(not any(p.is_symlink() for p in (bundle, *bundle.parents)), "bundle path contains a symlink")
    require({p.name for p in bundle.iterdir()} == FILES | {"MANIFEST.json"}, "candidate file inventory differs")
    require(all(p.is_file() and not p.is_symlink() for p in bundle.iterdir()), "candidate contains nonregular files")
    manifest = read_json(bundle / "MANIFEST.json")
    exact(manifest, {"schema", "files", "author_approval", "scope"})
    require(manifest["schema"] == "p5-analysis-data-candidate-v1" and manifest["author_approval"] is False and
            manifest["scope"] == SCOPE, "candidate is not release/author approval")
    exact(manifest["files"], FILES)
    before = {name: sha(bundle / name) for name in FILES | {"MANIFEST.json"}}
    for name, expected in manifest["files"].items():
        require(before[name] == digest(expected), "candidate checksum mismatch: " + name)
    stats = load_source(bundle / "statistics_core.py")
    core = load_source(bundle / "primary_core.py", vars(stats))
    result = rebuild(read_json(bundle / "observations.json"), stats, core)
    compare(read_json(bundle / "results.json"), result)
    sources = {filename: (bundle / filename).read_text() for filename in TEXT_MODULES.values()}
    require(claims_text(read_json(bundle / "observations.json"), result, sources) == (bundle / "claims.tex").read_text(),
            "results text differs from checked scores")
    require(before == {name: sha(bundle / name) for name in before}, "candidate changed while checking")
    return {"schema": "p5-public-check-v1", "primary_observations": sum(c["n"] for c in result["primary"]["analysis"]["cells"].values()),
            "cross_model_observations": 2000, "locomo_questions": 200, "scope": SCOPE, "author_approval": False}


if __name__ == "__main__":
    try:
        print(canonical(check(Path(__file__).resolve().parent)))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit("REFUSED: " + str(exc))
