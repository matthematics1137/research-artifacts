#!/usr/bin/env python3
"""Freeze or verify the committed-clean prospective P4R1 campaign plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
PUBLIC = HERE.parent
INFERENCE_SOURCES = (
    PUBLIC / "P4R1_PROTOCOL.md",
    PUBLIC / "FULL_RERUN.md",
    HERE / "prepare_campaign_plan.py",
    HERE / "approve_campaign_plan.py",
    HERE / "prepare_model_manifest.py",
    HERE / "prepare_gsm8k.py",
    HERE / "server_guard.py",
    HERE / "run_gsm8k.py",
    HERE / "run_one.sh",
    HERE / "run_matrix.sh",
)
ANALYSIS_SOURCES = (
    HERE / "promote_validation.py",
    HERE / "check_p4r1_evidence.py",
    HERE / "test_p4r1_protocol.py",
    HERE / "test_run_one.py",
    HERE / "test_p4r1_promotion.py",
    WORKSPACE / "paper4/scripts/build_p4r1_claims.py",
    WORKSPACE / "paper4/scripts/verify_claims.py",
    WORKSPACE / "paper4/scripts/check_p4r1_manuscript.py",
    WORKSPACE / "paper4/scripts/approve_p4r1_narrative.py",
    WORKSPACE / "paper4/scripts/make_figures.py",
    WORKSPACE / "paper4/Makefile",
)
EXPECTED_DATASET_SHA256 = "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37"
EXPECTED_ARTIFACT_SHA256 = (
    "7e8d75c23bd4de55582ac91084a0575f081c1de374b693f5ea7ffa6fa66a2e3c",
    "db62b07756f50d2f046b0a00dc594f1aa73dba82ebdc182d9f55c96474655f03",
    "7f2f98f301a74b645d0a4c2bd64c98e9811f2dfa26dbae4787c280eb1183632c",
    "d527a854db2a1582a3ce746a17b1f42d860334ece18d385ede9e2e395058b39e",
    "aa0b04ac05f71aa94b542b074500b3de956608365e74f0e1d97ae0d6a6380269",
    "6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0",
)


class PlanError(RuntimeError):
    pass


def command(*arguments: str) -> str:
    result = subprocess.run(
        arguments, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=30, check=False,
    )
    if result.returncode != 0:
        raise PlanError(result.stderr.strip() or f"command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def relative(path: Path) -> str:
    return path.resolve(strict=True).relative_to(WORKSPACE.resolve(strict=True)).as_posix()


def committed_clean_inventory() -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    repository = Path(command("git", "rev-parse", "--show-toplevel")).resolve(strict=True)
    if repository != WORKSPACE.resolve(strict=True):
        raise PlanError(f"P4R1 sources are not rooted in the expected repository: {repository}")
    all_sources = (*INFERENCE_SOURCES, *ANALYSIS_SOURCES)
    if len(set(all_sources)) != len(all_sources):
        raise PlanError("prospective inference/analysis source lists overlap")
    for path in all_sources:
        if path.is_symlink() or not path.is_file():
            raise PlanError(f"missing or unsafe prospective source: {path}")
    paths = [relative(path) for path in all_sources]
    tracked = set(command("git", "ls-files", "--", *paths).splitlines())
    if tracked != set(paths):
        raise PlanError(f"prospective sources are not all committed/tracked: {sorted(set(paths) - tracked)}")
    status = command("git", "status", "--porcelain=v1", "--untracked-files=all", "--", *paths)
    if status:
        raise PlanError(f"prospective P4R1 sources have staged/unstaged/untracked changes:\n{status}")
    commit = command("git", "rev-parse", "HEAD")
    def inventory_for(sources: tuple[Path, ...]) -> list[dict[str, Any]]:
        return [
        {
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob": command("git", "rev-parse", f"HEAD:{relative(path)}"),
        }
        for path in sources
        ]
    return commit, inventory_for(INFERENCE_SOURCES), inventory_for(ANALYSIS_SOURCES)


def build_plan() -> dict[str, Any]:
    commit, inference_inventory, analysis_inventory = committed_clean_inventory()
    return {
        "schema": "paper4-p4r1-prospective-plan-v1",
        "status": "prospectively frozen corrective exploratory validation; not independent",
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "git_state": "all inference/scoring and result-interpreting sources tracked and clean in index and worktree",
        "protocol": "publication/papers/moe-quantization-granularity/public/P4R1_PROTOCOL.md",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "artifact_sha256": list(EXPECTED_ARTIFACT_SHA256),
        "inference_source_inventory": inference_inventory,
        "analysis_source_inventory": analysis_inventory,
    }


def verify_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"missing or unsafe plan receipt: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PlanError("prospective plan receipt SHA-256 differs")
    plan = json.loads(raw)
    if (
        not isinstance(plan, dict)
        or set(plan)
        != {
            "schema", "status", "frozen_at_utc", "git_commit", "git_state",
            "protocol", "dataset_sha256", "artifact_sha256",
            "inference_source_inventory", "analysis_source_inventory",
        }
        or plan.get("schema") != "paper4-p4r1-prospective-plan-v1"
        or plan.get("status")
        != "prospectively frozen corrective exploratory validation; not independent"
        or plan.get("git_state")
        != "all inference/scoring and result-interpreting sources tracked and clean in index and worktree"
        or plan.get("protocol")
        != "publication/papers/moe-quantization-granularity/public/P4R1_PROTOCOL.md"
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00",
            str(plan.get("frozen_at_utc")),
        )
        is None
        or plan.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or plan.get("artifact_sha256") != list(EXPECTED_ARTIFACT_SHA256)
    ):
        raise PlanError("prospective plan receipt has the wrong identity or status")
    commit, inference_inventory, analysis_inventory = committed_clean_inventory()
    if (
        plan.get("git_commit") != commit
        or plan.get("inference_source_inventory") != inference_inventory
        or plan.get("analysis_source_inventory") != analysis_inventory
    ):
        raise PlanError("current committed-clean source tree differs from the frozen plan")
    return plan


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.stat().st_mode & 0o077:
        raise PlanError(f"plan directory must be owner-only (0700): {path.parent}")
    if path.exists() or path.is_symlink():
        raise PlanError(f"refusing to overwrite plan receipt: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if (args.output is None) == (args.verify is None):
        parser.error("choose exactly one of --output or --verify")
    if args.output is not None:
        plan = build_plan()
        payload = canonical_bytes(plan)
        write_exclusive(args.output.resolve(), payload)
        print(json.dumps({"plan": str(args.output.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}, indent=2))
    else:
        if not args.sha256 or len(args.sha256) != 64:
            parser.error("--verify requires --sha256")
        plan = verify_plan(args.verify.resolve(strict=True), args.sha256)
        print(json.dumps({"status": "pass", "git_commit": plan["git_commit"], "sha256": args.sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
