#!/usr/bin/env python3
"""Check/freeze a NEW P5 workspace's inputs; never launch inference.

Default is a read-only check. --freeze creates the manifest and then a private
input-check receipt; --verify checks an existing pair. This does not attest
pilot execution, loaded runtime/model identity, or authorize measurement.
Real tokenization can be substantial: check live compute headroom first.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import replication_check as A

SCHEMA = "paper5-rerun-input-freeze-v1"
MANIFEST = "testsuite/paper5/PROMPTS_MANIFEST.json"
RECEIPT = "testsuite/paper5/work/RERUN_INPUT_FREEZE.json"
MODULES = ("draft_support", "diagnostics", "evidence", "primary_audit", "primary_requests",
           "followup_results", "migrate_followup_preparation", "align_oracle_dates", "legacy")
INPUT_ASSETS = {("models", "qwen38-exl3-2.0/tokenizer.json"),
                ("repo", "testsuite/paper5_data/longmemeval/longmemeval_s_cleaned.json"),
                ("repo", "testsuite/paper5_data/longmemeval/longmemeval_oracle.json"),
                ("repo", "testsuite/paper5_data/locomo/locomo10.json")}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(root, relative):
    path = root / A.relative(relative)
    A.require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink in rerun input/output path")
    A.require(not path.exists() or path.is_file() or path.is_dir(), "non-regular rerun input/output path")
    return path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    return module


def layout(root):
    A.require(root.is_absolute() and ".." not in root.parts and not any(p.is_symlink() for p in (root, *root.parents)),
              "use an absolute, non-symlink new workspace")
    # No module from the experiment is imported before setup verification.
    setup = load("p5_rerun_setup_checker", safe(root, "verify_setup.py"))
    checked = setup.verify_setup(root)
    record = A.read_json(root / "RERUN_SETUP.json")
    A.require(record["files"].get("freeze_inputs.py") == sha(Path(__file__)), "input-freeze checker differs from setup")
    return record, checked


@contextmanager
def algorithms(root):
    """Import only the new workspace's source, restoring generic module names."""
    absent = object()
    saved = {name: sys.modules.get(name, absent) for name in MODULES}
    paths, bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        for name in MODULES:
            sys.modules.pop(name, None)
        sys.path[:0] = [str(root / folder) for folder in ("publication", "paper5/scripts", "testsuite/paper5_followups")]
        for name in ("draft_support", "diagnostics", "evidence", "primary_audit", "primary_requests", "legacy"):
            folder = "publication" if name == "draft_support" else "testsuite/paper5_followups" if name == "legacy" else "paper5/scripts"
            sys.modules[name] = load(name, root / folder / (name + ".py"))
        with sys.modules["legacy"].algorithms(SimpleNamespace(root=root, base=root / "testsuite/paper5/work", leg="locomo")) as a:
            yield a, sys.modules["primary_requests"]
    finally:
        for name, old in saved.items():
            if old is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        sys.path[:], sys.dont_write_bytecode = paths, bytecode


def input_assets(root, record):
    catalog = A.validate(A.read_json(root / "ASSETS.json"))
    selected = [row for row in catalog["assets"] if (row["root"], row["path"]) in INPUT_ASSETS]
    A.require(len(selected) == 4, "missing pinned datasets/tokenizer")
    roots = {"models": record["paths"]["models"], "repo": record["paths"]["repo_assets"]}
    checked = A.inspect(catalog, roots, verify=True, asset_ids=[row["id"] for row in selected])
    A.require(checked["ok"], "dataset/tokenizer content check failed: " + canonical(checked))
    return {str(A.locate(row, roots)): row["sha256"] for row in selected}


def candidate(root, q):
    directory = safe(root, "testsuite/paper5/work/prompts")
    A.require({p.name for p in directory.glob("*.jsonl")} == set(q.ARM_FILES), "incomplete/unexpected thirteen-cell prompt inventory")
    manifest = {"schema": "paper5-prompts-v1", "files": {}, "harness": {}}
    for name in q.ARM_FILES:
        path = safe(root, "testsuite/paper5/work/prompts/" + name)
        before = sha(path)
        batch = list(q.rows(path))
        A.require(batch, "empty primary prompt cell")
        for row in batch:
            q.integer(row.get("n_prompt"), "prompt token count", 1)
        counts = [row["n_prompt"] for row in batch]
        A.require(sha(path) == before, "prompt changed while constructing manifest")
        manifest["files"][name] = {"sha256": before, "prompts": len(batch),
                                    "tokens_mean": round(sum(counts) / len(counts), 1), "tokens_max": max(counts)}
    for name in sorted(q.FROZEN_HARNESS):
        manifest["harness"][name] = sha(safe(root, "testsuite/paper5/" + name))[:16]
    return manifest


@contextmanager
def candidate_view(q, root, manifest):
    """Validate before writing, using the unchanged scientific input reader.

    Only the not-yet-written manifest has an in-memory view. All source, data,
    prompts, pilot and correction evidence are read from actual files. No result
    from this view is an on-disk freeze receipt. After creation, the reader runs
    again with its normal file readers before a completion receipt is written.
    """
    original_read, original_sha = q.read_json, q.sha
    target = root / MANIFEST
    try:
        q.read_json = lambda path: manifest if Path(path) == target else original_read(path)
        q.sha = lambda path: digest(canonical(manifest).encode()) if Path(path) == target else original_sha(path)
        yield
    finally:
        q.read_json, q.sha = original_read, original_sha


def pilot(root, a, q, manifest):
    """Audit all twenty actual pilot requests, not merely a 'pilot|' ID prefix."""
    path = safe(root, "testsuite/paper5/work/journals/pilot.jsonl")
    before = sha(path)
    prompts = a.grid.build_pilot()  # CPU-only; fixed excluded category-3 cohort
    expected_ids = [p["job_id"] for p in prompts]
    A.require(len(expected_ids) == len(set(expected_ids)) == 20, "pilot cohort must contain twenty unique excluded items")
    records = q.P.journal_attempts(list(q.rows(path)))
    A.require(list(records) == expected_ids and all(row["ok"] for row in records.values()),
              "pilot outcomes/cohort/order are incomplete or different")
    harness = {name: manifest["harness"][name] for name in q.P.RUNNER_FILES}
    for prompt in prompts:
        q.integer(prompt["n_prompt"], "pilot prompt count", 1)
        A.require(prompt["arm"] == "T" and prompt["budget"] == 4096 and prompt["n_prompt"] <= 4096
                  and prompt["question_type"] == "locomo-cat3" and a.C.count_tokens(prompt["prompt"]) == prompt["n_prompt"],
                  "pilot prompt/tokenizer differs")
        expected = q.P.generation_spec(prompt, 16384)
        expected["job_id"] = "pilot|" + expected["job_id"]
        A.require(q.P.same(q.spec(a.grid.to_job(prompt, 16384)), expected), "pilot request constructor differs")
        q.P.validate_row(records[prompt["job_id"]], expected, harness, stage="generate", count_tokens=a.C.count_tokens)
    A.require(sha(path) == before, "pilot changed during validation")
    return {"items": 20, "job_ids_sha256": digest(canonical(expected_ids).encode()),
            "journal_sha256": before, "request_records_verified": True, "execution_verified": False}


def stable(bindings):
    A.require(all(sha(path) == expected for path, expected in bindings.items()), "rerun inputs changed during validation")


def before_measurements(root):
    journals = safe(root, "testsuite/paper5/work/journals")
    A.require(not any((journals / name).exists() for name in
                     ("grid.jsonl", "judge_gptoss__grid.jsonl", "judge_self__grid.jsonl")),
              "cannot establish an input freeze after primary measurement journals exist")
    execution = safe(root, "testsuite/paper5/work/execution")
    A.require(not list(execution.glob("pre-primary-*")) and not (execution / "primary").exists(),
              "original-study execution evidence must not be transplanted into a new campaign")


def once(path, payload):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def check(root, mode="check"):
    A.require(mode in ("check", "freeze", "verify"), "unknown input-freeze mode")
    root = Path(root)
    record, setup = layout(root)
    manifest_path, receipt_path = safe(root, MANIFEST), safe(root, RECEIPT)
    if mode == "verify":
        A.require(manifest_path.is_file() and receipt_path.is_file(), "complete input freeze does not exist")
    else:
        before_measurements(root)
        A.require(not receipt_path.exists(), "input freeze already exists; use --verify")
    bindings = input_assets(root, record)
    with algorithms(root) as (a, q):
        proposed = candidate(root, q)
        payload = canonical(proposed).encode()
        if manifest_path.exists():
            A.require(manifest_path.read_bytes() == payload, "existing prompt manifest differs; never overwrite it")
        archive = "testsuite/paper5/work/oracle-date-alignment-20260920/"
        archive_files = ["ALIGNMENT.json", "original-O_4096.jsonl"] + ["sources/" + name for name in
            ("paper5/scripts/align_oracle_dates.py", "testsuite/paper5/p5common.py", "testsuite/paper5/build_prompts.py",
             "testsuite/paper5_followups/legacy.py", "research/design-overflow-policies.md")]
        for relative in ("testsuite/paper5/work/cap.json", "testsuite/paper5/work/journals/pilot.jsonl",
                         *(archive + name for name in archive_files)):
            safe(root, relative)
        with candidate_view(q, root, proposed):
            inputs = q.frozen_inputs(root, a)
        pilot_check = pilot(root, a, q, proposed)
        A.require(all(path not in bindings or bindings[path] == value for path, value in inputs["bindings"].items()),
                  "scientific inputs differ from checked dataset/tokenizer pins")
        bindings.update(inputs["bindings"])
        bindings.pop(str(manifest_path))  # virtual manifest is not file evidence
        bindings.update({str(root / name): value for name, value in record["files"].items()})
        bindings[str(root / "RERUN_SETUP.json")] = setup["setup_sha256"]
        stable(bindings)
        report = {"schema": SCHEMA, "paper": "paper5", "status": "validated-inputs-not-execution-evidence",
                  "author_approval": False, "inference_started": False, "pilot_execution_verified": False,
                  "new_execution_baseline_created": False, "historical_smoke_adopted": False,
                  "setup_sha256": setup["setup_sha256"], "manifest_sha256": digest(payload),
                  "completion_cap": inputs["cap"], "primary_requests": len(inputs["prompts"]),
                  "primary_job_ids_sha256": digest(canonical([p["job_id"] for p in inputs["prompts"]]).encode()),
                  "cells": {name: data["prompts"] for name, data in proposed["files"].items()},
                  "pilot": pilot_check, "files": bindings,
                  "remaining": "prospective pilot/runtime/model/hardware/ownership evidence and a separate guarded new-campaign execution path"}
        if mode == "verify":
            saved = A.read_json(receipt_path)
            stamp = saved.pop("created_at", None)
            A.require(isinstance(stamp, str) and datetime.fromisoformat(stamp).utcoffset() is not None,
                      "input-freeze timestamp missing or unzoned")
            A.require(canonical(saved) == canonical(report), "saved input-freeze receipt differs")
            # Recheck actual on-disk manifest; never use candidate_view here.
            q.frozen_inputs(root, a)
            stable(bindings)
        elif mode == "freeze":
            lock = safe(root, "testsuite/paper5/work/.input-freeze.lock")
            # This ordinary local writer lock is not a compute allocation.
            with lock.open("a") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError("another input-freeze writer is active") from exc
                before_measurements(root)
                A.require(not receipt_path.exists(), "input freeze was concurrently completed")
                stable(bindings)
                if manifest_path.exists():
                    A.require(manifest_path.read_bytes() == payload, "existing prompt manifest changed")
                else:
                    once(manifest_path, payload)
                q.frozen_inputs(root, a)
                stable(bindings)
                before_measurements(root)
                A.require(manifest_path.read_bytes() == payload, "manifest changed before receipt creation")
                once(receipt_path, canonical({**report, "created_at": datetime.now(timezone.utc).isoformat()}).encode())
        return {"schema": SCHEMA, "mode": mode, "inputs_checked": True,
                "freeze_complete": mode in ("freeze", "verify"), "inference_started": False, "author_approval": False,
                "pilot_execution_verified": False, "primary_requests": report["primary_requests"],
                "cells": report["cells"], "completion_cap": report["completion_cap"], "setup_sha256": report["setup_sha256"],
                "manifest_sha256": report["manifest_sha256"], "remaining": report["remaining"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        print(canonical(check(args.workspace.absolute(), "freeze" if args.freeze else "verify" if args.verify else "check")), end="")
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(2, "P5 rerun input check refused (any partial freeze is retained): " + str(exc) + "\n")


if __name__ == "__main__":
    main()
