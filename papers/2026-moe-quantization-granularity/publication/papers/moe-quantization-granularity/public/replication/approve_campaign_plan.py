#!/usr/bin/env python3
"""Create an interactive author approval bound to one frozen P4R1 plan hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite plan approval: {path}")
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


def validate_approval(value: Any, plan: dict[str, Any], plan_sha256: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema", "operation", "actor", "plan_sha256", "plan_git_commit",
            "approved_at_utc", "method",
        }
        or value.get("schema") != "paper4-p4r1-plan-approval-v1"
        or value.get("operation") != "run-p4r1-validation"
        or not isinstance(value.get("actor"), str)
        or not value["actor"].strip()
        or value.get("plan_sha256") != plan_sha256
        or value.get("plan_git_commit") != plan.get("git_commit")
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00",
            str(value.get("approved_at_utc")),
        )
        is None
        or value.get("method") != "interactive-exact-challenge"
    ):
        raise RuntimeError("approval does not authorize this exact P4R1 plan")


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--actor", default="Matthew Schwartz")
    args = parser.parse_args()
    if (args.output is None) == (args.verify is None):
        parser.error("choose exactly one of --output or --verify")
    if SHA256.fullmatch(args.plan_sha256) is None:
        parser.error("--plan-sha256 must be 64 lowercase hexadecimal characters")
    plan_path = args.plan.resolve(strict=True)
    if plan_path.is_symlink() or not plan_path.is_file():
        raise RuntimeError("plan must be a non-symlink regular file")
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.plan_sha256:
        raise RuntimeError("plan bytes differ from --plan-sha256")
    plan = json.loads(raw)
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "paper4-p4r1-prospective-plan-v1"
        or plan.get("status")
        != "prospectively frozen corrective exploratory validation; not independent"
        or re.fullmatch(r"[0-9a-f]{40}", str(plan.get("git_commit"))) is None
    ):
        raise RuntimeError("plan is not the frozen P4R1 corrective-validation plan")
    if args.verify is not None:
        approval_path = args.verify.resolve(strict=True)
        if approval_path.is_symlink() or not approval_path.is_file():
            raise RuntimeError("approval must be a non-symlink regular file")
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        validate_approval(approval, plan, args.plan_sha256)
        print(
            json.dumps(
                {
                    "status": "pass", "approval": str(approval_path),
                    "approval_sha256": hashlib.sha256(approval_path.read_bytes()).hexdigest(),
                    "plan_sha256": args.plan_sha256,
                },
                indent=2,
            )
        )
        return 0
    challenge = f"AUTHORIZE p4r1-validation {args.plan_sha256[:12]}"
    print("Exact effects:")
    print("  - run six P4R1 inference cells under the frozen corrective protocol")
    print("  - preserve private responses, identity receipts, and immutable per-cell logs")
    print("  - do not publish or overwrite historical evidence")
    print(f"\nType exactly: {challenge}")
    entered = input("> ")
    if entered != challenge:
        raise RuntimeError("authorization challenge did not match exactly")
    record = {
        "schema": "paper4-p4r1-plan-approval-v1",
        "operation": "run-p4r1-validation",
        "actor": args.actor,
        "plan_sha256": args.plan_sha256,
        "plan_git_commit": plan["git_commit"],
        "approved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "interactive-exact-challenge",
    }
    validate_approval(record, plan, args.plan_sha256)
    payload = canonical_bytes(record)
    assert args.output is not None
    write_exclusive(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "approval": str(args.output.resolve()),
                "approval_sha256": hashlib.sha256(payload).hexdigest(),
                **record,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
