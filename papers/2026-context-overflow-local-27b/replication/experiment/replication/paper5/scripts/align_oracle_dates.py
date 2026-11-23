#!/usr/bin/env python3
"""Prospective P5 oracle timestamp alignment; no inference or frozen-code edit.

Default: compute/inspect only. --apply preserves the original O prompt file
and installs the aligned one, but only before the primary prompt freeze and
before any primary generation journal exists. Run in the installed exl3-env
so tokenization uses its existing tokenizers package; nothing is downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from evidence import ROOT
from draft_support import atomic_text, read_json, sha
from followup_results import source_modules
from migrate_followup_preparation import once

DATA_PINS = {
    "longmemeval_s_cleaned.json": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    "longmemeval_oracle.json": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(records):
    # Exact serialization used by the original build_prompts.write.
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)


def compute_rows(items, oracle, builder):
    by_id = {it["question_id"]: it for it in oracle}
    require(len(by_id) == len(oracle) == len(items) and by_id.keys() == {it["question_id"] for it in items},
            "oracle and primary item IDs differ")
    original, aligned, changed_dates = [], [], 0
    for item in items:
        source = by_id[item["question_id"]]
        for key in ("question", "question_type", "answer"):
            require(source[key] == item[key], "oracle differs in more than timestamp: " + key)
        require(isinstance(source["question_date"], str) and isinstance(item["question_date"], str), "invalid question date")
        changed_dates += source["question_date"] != item["question_date"]
        old, new = builder(source), builder(dict(source, question_date=item["question_date"]))
        if old is not None:
            original.append(old)
        if new is not None:
            aligned.append(new)
    return original, aligned, changed_dates


def require_unfrozen(root):
    require(not (root / "testsuite/paper5/PROMPTS_MANIFEST.json").exists(), "primary prompt manifest already exists")
    require(not (root / "testsuite/paper5/work/journals/grid.jsonl").exists(), "primary generation journal already exists")


def run(apply=False, root=ROOT):
    root = Path(root).resolve()
    require_unfrozen(root)
    work = root / "testsuite/paper5/work"
    output = work / "prompts/O_4096.jsonl"
    archive = work / "oracle-date-alignment-20260920"
    receipt_path = archive / "ALIGNMENT.json"
    require(not output.is_symlink() and not archive.is_symlink(), "symlink in correction output/archive")
    before = sha(output)
    sources = [Path(__file__).resolve(), root / "testsuite/paper5/p5common.py", root / "testsuite/paper5/build_prompts.py",
               root / "testsuite/paper5_followups/legacy.py", root / "research/design-overflow-policies.md"]
    source_hashes = {str(path.relative_to(root)): sha(path) for path in sources}
    inputs = {}
    for name, expected in DATA_PINS.items():
        path = root / "testsuite/paper5_data/longmemeval" / name
        require(sha(path) == expected, "source dataset differs from frozen pin")
        inputs[str(path)] = expected
    with source_modules():
        from legacy import algorithms
        with algorithms(SimpleNamespace(root=root, base=work, leg="locomo")) as a:
            items = a.C.load_json_verified(a.C.LME_S)
            oracle = a.C.load_json_verified(a.C.LME_ORACLE)
            tokenizer_path = Path(a.C.MODEL_DIR) / "tokenizer.json"
            inputs[str(tokenizer_path)] = sha(tokenizer_path)
            old, new, changed = compute_rows(items, oracle, a.BP.build_O)
    old_text, new_text = encode(old), encode(new)
    current = output.read_text()
    require(current in (old_text, new_text), "existing oracle prompts do not match either reproducible version")
    new_sha = hashlib.sha256(new_text.encode()).hexdigest()
    report = {"schema": "p5-oracle-date-alignment-v1", "reason": "use the cleaned primary question timestamp with oracle evidence",
              "source_items": len(items), "source_timestamp_mismatches": changed,
              "original_fit_items": len(old), "aligned_fit_items": len(new),
              "original_prompt_sha256": hashlib.sha256(old_text.encode()).hexdigest(),
              "aligned_prompt_sha256": new_sha,
              "aligned_token_range": [min(p["n_prompt"] for p in new), max(p["n_prompt"] for p in new)],
              "fit_subset_changed": {p["question_id"] for p in old} != {p["question_id"] for p in new},
              "inference_started": False, "inputs": inputs}
    report["source_sha256"] = source_hashes
    require(all(sha(root / name) == value for name, value in source_hashes.items()), "source code changed during alignment computation")
    require(all(sha(path) == value for path, value in inputs.items()), "input changed during alignment computation")
    require_unfrozen(root)
    require(sha(output) == before, "oracle output changed during alignment computation")
    if receipt_path.exists():
        require(read_json(receipt_path) == report, "existing alignment receipt differs")
        require((archive / "original-O_4096.jsonl").read_text() == old_text, "original prompt backup changed")
        if not apply:
            require(current == new_text, "alignment receipt exists but installation has not completed")
    elif current == new_text and old_text != new_text:
        raise ValueError("aligned file lacks its original preservation receipt")
    if apply:
        once(archive / "original-O_4096.jsonl", old_text)
        for path in sources:
            once(archive / "sources" / path.relative_to(root), path.read_bytes())
        once(receipt_path, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        require_unfrozen(root)
        if current != new_text:
            atomic_text(output, new_text)
        require(sha(output) == new_sha, "aligned prompts failed final readback")
        require_unfrozen(root)
    return {k: v for k, v in report.items() if k not in ("inputs", "source_sha256")} | {
        "installed": apply or (receipt_path.exists() and current == new_text), "receipt": str(receipt_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.apply), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"P5 oracle alignment stopped: {exc}\n")
