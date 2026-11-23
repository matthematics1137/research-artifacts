#!/usr/bin/env python3
"""Prepare and inspect isolated P5 follow-ups. No command here starts inference.

Model-backed prompt preparation and execution are separate stages. In
particular, init/freeze/select cannot silently start a GPU service.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from legacy import algorithms
from workspace import LEGS, ROOT, Workspace, canonical, read_json, select_policy, sha

PRIMARY_FILES = ("p5common.py", "p5run.py", "build_prompts.py", "summarize.py",
                 "judge.py", "run_grid.py", "lingua_probs.py", "analyze.py", "tabby-p5.yml")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verified_primary(root=ROOT):
    """Require complete primary generations and both graders before selecting."""
    evidence = load_module("p5_followup_primary_evidence", root / "paper5/scripts/evidence.py")
    checked = evidence.verify(root)
    claims_path = root / "paper5/derived/claims.json"
    claims_sha = sha(claims_path)
    claims = read_json(claims_path)
    # This function can also be called from the secondary prompt-preparation
    # process. Give the read-only analyzer fresh PRIMARY module globals; do
    # not accidentally inherit a secondary C.WORK from an enclosing bridge.
    primary = SimpleNamespace(root=root, base=root / "testsuite/paper5/work", leg="locomo")
    with algorithms(primary):
        analyzer = load_module("p5_followup_primary_analysis", root / "testsuite/paper5/analyze.py")
        if canonical(analyzer.analyze()) != canonical(claims):
            raise ValueError("primary claims are stale relative to the complete evidence")
    if evidence.verify(root) != checked or sha(claims_path) != claims_sha:
        raise ValueError("primary evidence or claims changed during selection")
    bindings = {str(root / name): digest for name, digest in checked["inputs"].items()}
    bindings[str(claims_path)] = claims_sha
    return claims, bindings


def initialize(work):
    with algorithms(work) as a:
        if work.leg == "cross-model":
            dataset = Path(a.C.LME_S)
            items = a.C.load_json_verified(a.C.LME_S)
            for item in items:
                item["answer"] = str(item["answer"])
        else:
            dataset = Path(a.C.LOCOMO)
            items = a.C.locomo_subset(a.C.locomo_items())
            for item in items:
                item["conversation_id"] = item["question_id"].rsplit("-", 1)[0]
                # The legacy adapter left this diagnostic empty. Recover the
                # session IDs from the dataset's explicit evidence-turn flags.
                item["answer_session_ids"] = [sid for sid, turns in zip(item["haystack_session_ids"], item["haystack_sessions"])
                                                if any(turn.get("has_answer") for turn in turns)]
        cap_path = work.root / "testsuite/paper5/work/cap.json"
        paths = [dataset, cap_path, work.root / "research/design-overflow-policies.md",
                 work.root / "paper4/scripts/verify_claims.py"]
        paths += [work.root / "testsuite/paper5" / name for name in PRIMARY_FILES]
        paths += sorted((work.root / "testsuite/paper5_followups").glob("*.py"))
        model_dir = Path(a.C.model_dir(LEGS[work.leg]["model"]))
        paths += [model_dir / "tokenizer.json", model_dir / "tokenizer_config.json", model_dir / "config.json"]
        # Qwen3-8B stores its template inside tokenizer_config.json; the 27B
        # artifact also has a standalone Jinja file. Bind whichever is present.
        if (model_dir / "chat_template.jinja").is_file():
            paths.append(model_dir / "chat_template.jinja")
        bindings = {str(path): sha(path) for path in paths}
        return work.initialize(items, read_json(cap_path)["completion_cap"], bindings)


def select(work):
    if work.leg != "locomo":
        raise ValueError("only LoCoMo selects a winning policy")
    work.load()
    claims, inputs = verified_primary(work.root)
    selection = {"schema": "paper5-followup-selection-v1", "policy": select_policy(claims),
                 "rule": "primary gptoss accuracy at 4096; ties S, M, C",
                 "primary_claims": {"schema": claims["schema"], "primary_judge": claims["primary_judge"],
                                    "cells": {arm + "@4096": claims["cells"][arm + "@4096"] for arm in ("S", "M", "C")}},
                 "primary_inputs": inputs}
    work.write_json_once("SELECTION.json", selection)
    return {"policy": selection["policy"], "selection": str(work.path("SELECTION.json"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "init", "select", "freeze", "verify"))
    parser.add_argument("leg", choices=tuple(LEGS))
    args = parser.parse_args()
    work = Workspace(args.leg)
    try:
        if args.command == "plan":
            result = {"leg": args.leg, **LEGS[args.leg], "budget": 4096,
                      "workspace": str(work.base), "inference_started": False,
                      "policy": "T/S/C/M" if args.leg == "cross-model" else "T versus selected S/M/C; selection pending complete primary evidence"}
        elif args.command == "init":
            result = initialize(work)
            result = {key: result[key] for key in ("leg", "model", "items", "publication")}
        elif args.command == "select":
            result = select(work)
        elif args.command == "freeze":
            result = work.freeze()
        else:
            result = {"leg": args.leg, "prompts": len(work.frozen_prompts()), "inference_started": False}
        print(json.dumps(result, sort_keys=True, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"P5 follow-up not ready: {exc}\n")


if __name__ == "__main__":
    main()
