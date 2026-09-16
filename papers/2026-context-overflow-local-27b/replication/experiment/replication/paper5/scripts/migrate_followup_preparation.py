#!/usr/bin/env python3
"""Preserve prepared P5 prompts across the pre-measurement JSON-loader correction.

This one-off migration is deliberately outside the frozen execution harness.
Checkpoint first, edit the two permitted source files, then adopt. Checkpoint
moves the original, unmeasured workspaces into a private archive; adoption
initializes fresh workspaces and imports byte-identical prompts with explicit
historical provenance. It never rewrites an old receipt or runs a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from evidence import ROOT
from draft_support import read_json, sha
from followup_results import source_modules

IDENTIFIER = "primary-json-roundtrip-20260920"
LEGS = {"cross-model": ("T", "C", "M"), "locomo": ("T",)}
LOADER = "testsuite/paper5_followups/followups.py"
TEST = "testsuite/paper5_followups/test_workspace.py"
OLD_IMPORT = "from workspace import LEGS, ROOT, Workspace, read_json, select_policy, sha"
NEW_IMPORT = "from workspace import LEGS, ROOT, Workspace, canonical, read_json, select_policy, sha"
OLD_COMPARE = "if analyzer.analyze() != claims:"
NEW_COMPARE = "if canonical(analyzer.analyze()) != canonical(claims):"
LOADER_REPLACEMENTS = (
    (OLD_IMPORT, NEW_IMPORT), (OLD_COMPARE, NEW_COMPARE),
    ("    claims = read_json(claims_path)", "    claims_sha = sha(claims_path)\n    claims = read_json(claims_path)"),
    ("    if evidence.verify(root) != checked:", "    if evidence.verify(root) != checked or sha(claims_path) != claims_sha:"),
    ('raise ValueError("primary evidence changed during selection")',
     'raise ValueError("primary evidence or claims changed during selection")'),
    ("    bindings[str(claims_path)] = sha(claims_path)", "    bindings[str(claims_path)] = claims_sha"),
)


def corrected_loader(old):
    for before, after in LOADER_REPLACEMENTS:
        require(old.count(before) == 1, "unexpected original loader")
        old = old.replace(before, after)
    return old


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def safe_path(base, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts, "unsafe migration path")
    dest = base / rel
    require(not any(p.is_symlink() for p in (dest, *dest.parents)), "symlink in migration path")
    return dest


def once(path, payload):
    data = payload if isinstance(payload, bytes) else payload.encode()
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"refusing to replace {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".migration-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def files_at(base):
    require(base.is_dir(), f"missing workspace: {base}")
    result = {}
    for path in sorted(base.rglob("*")):
        require(not path.is_symlink(), "symlink in archived workspace")
        if path.is_file():
            result[str(path.relative_to(base))] = sha(path)
        else:
            require(path.is_dir(), "non-regular workspace artifact")
    return result


def archive_path(root):
    return safe_path(root, "testsuite/paper5_followups/work/_archive/" + IDENTIFIER)


def expected_files(arms, migrated=False):
    files = {"WORKSPACE.json", "items.json"}
    for arm in arms:
        files.update((f"prompts/{arm}_4096.jsonl", f"preparation/{arm}.json", f"preparation/{arm}-timing.json"))
    if migrated:
        files.add("PREPARATION_MIGRATION.json")
    return files


def inspect_work(work, W, arms):
    info, items = work.load()
    files = files_at(work.base)
    require(set(files) == expected_files(arms), "migration requires only the complete declared preparation; no selection, freeze or measurements")
    for arm in arms:
        path = work.path(f"prompts/{arm}_4096.jsonl")
        records = list(W.read_rows(path))
        W.validate_prompts(work.leg, arm, records, items)
        prep = read_json(work.path(f"preparation/{arm}.json"))
        timing = read_json(work.path(f"preparation/{arm}-timing.json"))
        require(prep.get("schema") == "paper5-followup-preparation-v1" and prep.get("arm") == arm
                and prep.get("model") == info["model"] and prep.get("prompts_sha256") == sha(path), "invalid preparation receipt")
        W.verify_inputs(prep["inputs"])
        require(prep["inputs"].get(str(work.path("WORKSPACE.json"))) == sha(work.path("WORKSPACE.json")), "preparation is not bound to this workspace")
        require(timing.get("schema") == "paper5-followup-preparation-timing-v1" and timing.get("arm") == arm
                and timing.get("model") == info["model"] and timing.get("prompts_sha256") == sha(path)
                and timing.get("inference_started") is False, "missing or stale preparation timing")
    return info, files


def checkpoint(root=ROOT):
    root = Path(root).resolve()
    archive = archive_path(root)
    manifest_path = safe_path(archive, "CHECKPOINT.json")
    with source_modules() as modules:
        W = modules.W
        if manifest_path.exists():
            saved = read_json(manifest_path)
            require(saved.get("root") == str(root) and saved.get("identifier") == IDENTIFIER, "foreign migration checkpoint")
            # Resume only the archive moves; never invent a new inventory after
            # interruption or accept source changes before the checkpoint ends.
            for name, digest in saved["inputs"].items():
                require(sha(name) == digest, "source/input changed before checkpoint completed")
        else:
            saved = {"schema": "p5-preparation-checkpoint-v1", "root": str(root), "identifier": IDENTIFIER,
                     "inference_started": False, "workspaces": {}, "inputs": {}, "source_copies": {}}
            for leg, arms in LEGS.items():
                work = W.Workspace(leg, root)
                info, files = inspect_work(work, W, arms)
                saved["workspaces"][leg] = files
                for name, digest in info["inputs"].items():
                    require(name not in saved["inputs"] or saved["inputs"][name] == digest, "inconsistent input bindings")
                    saved["inputs"][name] = digest
                # Copy the small, mutable repository sources before any edit.
                for name, digest in info["inputs"].items():
                    path = Path(name)
                    if path.is_relative_to(root) and path.suffix in (".py", ".md", ".yml"):
                        relative = str(path.relative_to(root))
                        once(safe_path(archive, "sources/" + relative), path.read_bytes())
                        saved["source_copies"][relative] = digest
            require(LOADER in saved["source_copies"] and TEST in saved["source_copies"], "original loader/test sources not bound")
            once(manifest_path, canonical(saved))
        for leg in LEGS:
            current = W.Workspace(leg, root).base
            destination = safe_path(archive, leg)
            if destination.exists():
                require(not current.exists(), "both old canonical and archive exist; inspect before proceeding")
            else:
                require(files_at(current) == saved["workspaces"][leg], "preparation changed during checkpoint")
                current.rename(destination)
            require(files_at(destination) == saved["workspaces"][leg], "archived bytes differ from checkpoint")
        for relative, digest in saved["source_copies"].items():
            require(sha(safe_path(archive, "sources/" + relative)) == digest, "source archive changed")
    return {"archive": str(archive), "original_bytes_preserved": True, "inference_started": False}


def verify_archive(root, archive, saved):
    require(saved.get("schema") == "p5-preparation-checkpoint-v1" and saved.get("root") == str(root)
            and saved.get("identifier") == IDENTIFIER and saved.get("inference_started") is False,
            "wrong preparation checkpoint")
    for leg in LEGS:
        require(files_at(safe_path(archive, leg)) == saved["workspaces"][leg], "archived workspace changed")
    for relative, digest in saved["source_copies"].items():
        require(sha(safe_path(archive, "sources/" + relative)) == digest, "archived source changed")
    old = safe_path(archive, "sources/" + LOADER).read_text()
    corrected = corrected_loader(old)
    require((root / LOADER).read_text() == corrected, "only the exact JSON-roundtrip correction is allowed")
    for name, digest in saved["inputs"].items():
        if name not in {str(root / LOADER), str(root / TEST)}:
            require(sha(name) == digest, "unrelated frozen input changed")


def adopt(root=ROOT, initialize=None):
    root = Path(root).resolve()
    archive = archive_path(root)
    checkpoint_file = safe_path(archive, "CHECKPOINT.json")
    saved = read_json(checkpoint_file)
    verify_archive(root, archive, saved)
    adopted = {}
    with source_modules() as modules:
        W = modules.W
        if initialize is None:
            from followups import initialize
        for leg, arms in LEGS.items():
            old = safe_path(archive, leg)
            info = read_json(old / "WORKSPACE.json")
            work = W.Workspace(leg, root)
            new = initialize(work)
            expected = {**info, "inputs": {name: sha(name) for name in info["inputs"]}}
            require(new == expected, "initialization changed more than the two permitted source fingerprints")
            require(sha(work.path("items.json")) == saved["workspaces"][leg]["items.json"], "frozen sample changed")
            migration = {"schema": "p5-preparation-adoption-v1", "reason": "JSON serialization converts numeric budget keys to strings",
                         "checkpoint": str(checkpoint_file), "checkpoint_sha256": sha(checkpoint_file),
                         "original_workspace": str(old / "WORKSPACE.json"), "original_workspace_sha256": sha(old / "WORKSPACE.json"),
                         "current_workspace_sha256": sha(work.path("WORKSPACE.json")),
                         "measurements_reused": False, "prompt_bytes_changed": False, "timings": "original preparation measurements, not migration time",
                         "source_changes": {name: {"before": saved["inputs"][str(root / name)], "after": sha(root / name)}
                                            for name in (LOADER, TEST)}}
            receipt = work.write_json_once("PREPARATION_MIGRATION.json", migration)
            for arm in arms:
                prep_path = old / f"preparation/{arm}.json"
                timing_path = old / f"preparation/{arm}-timing.json"
                prompt_path = old / f"prompts/{arm}_4096.jsonl"
                previous = read_json(prep_path)
                inputs = dict(previous["inputs"])
                # The current workspace is new; original receipts are immutable
                # historical evidence, not relabelled as new preparation runs.
                workspace_key = str(work.path("WORKSPACE.json"))
                require(inputs.pop(workspace_key, None) == sha(old / "WORKSPACE.json"), "original preparation/workspace binding differs")
                require(all(not Path(name).is_relative_to(work.base) for name in inputs), "unexpected local preparation dependency")
                if inputs:  # T depends only on the workspace; C/M also bind caches or weights.
                    W.verify_inputs(inputs)
                inputs[workspace_key] = sha(work.path("WORKSPACE.json"))
                for path in (receipt, checkpoint_file, old / "WORKSPACE.json", prep_path, timing_path, prompt_path):
                    inputs[str(path)] = sha(path)
                rows = list(W.read_rows(prompt_path))
                serialized = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows)
                require(hashlib.sha256(serialized.encode()).hexdigest() == sha(prompt_path), "adoption would change prompt serialization")
                work.write_prompts(arm, rows, inputs)
                work.write_once(f"preparation/{arm}-timing.json", timing_path.read_bytes())
                require(sha(work.path(f"prompts/{arm}_4096.jsonl")) == sha(prompt_path), "adopted prompt bytes changed")
                W.verify_inputs(read_json(work.path(f"preparation/{arm}.json"))["inputs"])
            current = files_at(work.base)
            require(set(current) == expected_files(arms, migrated=True), "unexpected new-workspace artifacts")
            work.load()
            adopted[leg] = current
    verify_archive(root, archive, saved)
    report = {"schema": "p5-preparation-migration-complete-v1", "checkpoint_sha256": sha(checkpoint_file),
              "canonical_workspaces": adopted, "original_bytes_preserved": True, "inference_started": False}
    once(safe_path(archive, "ADOPTION.json"), canonical(report))
    return {"archive": str(archive), "adopted": {leg: list(arms) for leg, arms in LEGS.items()},
            "original_bytes_preserved": True, "inference_started": False}


def require_idle():
    processes = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    programs = tuple("paper5_followups/" + p + ".py" for p in ("prepare", "preprocess", "run", "followups"))
    require(not any(any(program in line for program in programs) for line in processes.splitlines()),
            "a follow-up process may still be active; inspect before migrating")


def verify_current(root=ROOT):
    """Read-only verification; later, additional arms/results need their own audits."""
    root = Path(root).resolve()
    archive = archive_path(root)
    checkpoint_file = safe_path(archive, "CHECKPOINT.json")
    saved = read_json(checkpoint_file)
    verify_archive(root, archive, saved)
    adopted = read_json(safe_path(archive, "ADOPTION.json"))
    require(adopted.get("schema") == "p5-preparation-migration-complete-v1"
            and adopted.get("checkpoint_sha256") == sha(checkpoint_file)
            and adopted.get("original_bytes_preserved") is True and adopted.get("inference_started") is False,
            "invalid migration-completion receipt")
    with source_modules() as modules:
        W = modules.W
        for leg, arms in LEGS.items():
            work = W.Workspace(leg, root)
            work.load()
            for relative, digest in adopted["canonical_workspaces"][leg].items():
                require(sha(work.path(relative)) == digest, "adopted preparation artifact changed")
            for arm in arms:
                require(sha(work.path(f"prompts/{arm}_4096.jsonl")) == saved["workspaces"][leg][f"prompts/{arm}_4096.jsonl"],
                        "adopted prompt differs from original")
                W.verify_inputs(read_json(work.path(f"preparation/{arm}.json"))["inputs"])
    return {"archive": str(archive), "preserved": {leg: list(arms) for leg, arms in LEGS.items()},
            "read_only": True, "measurements_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("checkpoint", "adopt", "verify"))
    args = parser.parse_args()
    try:
        if args.command != "verify":
            require_idle()
        print(json.dumps({"checkpoint": checkpoint, "adopt": adopt, "verify": verify_current}[args.command](), indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"P5 preparation migration stopped safely: {exc}\n")


if __name__ == "__main__":
    main()
