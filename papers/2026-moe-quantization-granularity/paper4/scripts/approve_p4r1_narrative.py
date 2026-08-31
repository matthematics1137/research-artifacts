#!/usr/bin/env python3
"""Record author review of exact P4R1 claims and exact manuscript source bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parent.parent
WORKSPACE = PAPER.parent
PRIVATE_ROOT = WORKSPACE / ".publication" / "private" / "p4r1"
SOURCES = (PAPER / "main.tex", PAPER / "explainer.tex", PAPER / "brief.tex")
AUTHOR = "Matthew Schwartz"
PROFILE_MARKER = "% P4R1-PROTOCOL-PROFILE: corrective-exploratory-v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    if ".." in path.parts:
        raise RuntimeError(f"path contains parent-directory traversal: {path}")
    return Path(os.path.abspath(path if path.is_absolute() else Path.cwd() / path))


def _reject_symlink_chain(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise RuntimeError(f"{label} traverses a symlink: {current}")


def read_workspace_file(path: Path, label: str) -> tuple[Path, bytes]:
    candidate = _lexical_absolute(path)
    if not candidate.is_relative_to(WORKSPACE):
        raise RuntimeError(f"{label} must be inside the workspace")
    _reject_symlink_chain(candidate, label)
    try:
        before = candidate.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"missing {label}: {candidate}") from error
    if candidate.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} is not a regular non-symlink file")
    payload = candidate.read_bytes()
    after = candidate.lstat()
    if (
        candidate.is_symlink()
        or not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while it was read")
    return candidate, payload


def prepare_private_output(path: Path) -> Path:
    candidate = _lexical_absolute(path)
    root = _lexical_absolute(PRIVATE_ROOT)
    if candidate.suffix != ".json":
        raise RuntimeError("narrative-review output must have a .json suffix")
    if not candidate.is_relative_to(root) or candidate == root:
        raise RuntimeError(
            f"narrative-review output must be beneath the private P4R1 root: {root}"
        )
    _reject_symlink_chain(root, "private P4R1 root")
    try:
        root_metadata = root.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"private P4R1 root is absent: {root}") from error
    if (
        root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_mode & 0o777 != 0o700
    ):
        raise RuntimeError("private P4R1 root is not an owner-only directory")

    current = root
    for component in candidate.parent.relative_to(root).parts:
        current /= component
        if os.path.lexists(current):
            metadata = current.lstat()
            if (
                current.is_symlink()
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o777 != 0o700
            ):
                raise RuntimeError(f"unsafe narrative-review output parent: {current}")
        else:
            os.mkdir(current, mode=0o700)
    _reject_symlink_chain(candidate.parent, "narrative-review output parent")
    if os.path.lexists(candidate):
        raise RuntimeError(f"refusing to overwrite narrative review: {candidate}")
    return candidate


def require_real_terminal() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("author narrative approval requires a real interactive terminal")
    if os.environ.get("SUDO_UID") or os.environ.get("SUDO_USER"):
        raise RuntimeError("author narrative approval must not run through sudo")


def write_exclusive(path: Path, payload: bytes) -> None:
    parent_metadata = path.parent.lstat()
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or parent_metadata.st_mode & 0o777 != 0o700
    ):
        raise RuntimeError("narrative-review output parent is not owner-only")
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to overwrite narrative review: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--claims-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviewer", default=AUTHOR)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{64}", args.claims_sha256) is None:
        parser.error("--claims-sha256 must be lowercase SHA-256")
    if args.reviewer != AUTHOR:
        parser.error(f"--reviewer must be the responsible author: {AUTHOR}")
    claims_path, claims_payload = read_workspace_file(args.claims, "claims file")
    if hashlib.sha256(claims_payload).hexdigest() != args.claims_sha256:
        raise RuntimeError("claims bytes differ from the review hash")
    claims = json.loads(claims_payload)
    signature = claims.get("outcome_signature")
    if claims.get("schema_version") != "paper4-p4r1-claims-v1" or not isinstance(signature, dict):
        raise RuntimeError("claims are not the clean P4R1 outcome schema")
    source_records = []
    source_payloads: dict[Path, bytes] = {}
    for source in SOURCES:
        source_path, source_payload = read_workspace_file(source, source.name)
        try:
            text = source_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"{source.name} is not UTF-8") from error
        if text.splitlines()[:1] != [PROFILE_MARKER]:
            raise RuntimeError(f"{source.name} has not entered the P4R1 prose profile")
        source_payloads[source_path] = source_payload
        source_records.append(
            {
                "path": source.name,
                "bytes": len(source_payload),
                "sha256": hashlib.sha256(source_payload).hexdigest(),
            }
        )
    outcome_sha = canonical_sha256(signature)
    challenge = f"AUTHORIZE p4r1-narrative {args.claims_sha256[:8]} {outcome_sha[:8]}"
    print("This attests that the exact paper, explainer, and brief were reviewed after seeing P4R1 outcomes.")
    print("It does not authorize publication.")
    print(f"Type exactly: {challenge}")
    require_real_terminal()
    if input("> ") != challenge:
        raise RuntimeError("narrative-review challenge did not match exactly")
    if read_workspace_file(claims_path, "claims file")[1] != claims_payload:
        raise RuntimeError("claims changed during narrative approval")
    for source_path, expected_payload in source_payloads.items():
        if read_workspace_file(source_path, source_path.name)[1] != expected_payload:
            raise RuntimeError(f"{source_path.name} changed during narrative approval")
    review = {
        "schema": "paper4-p4r1-narrative-review-v1",
        "status": "author-reviewed-after-p4r1-results",
        "reviewer": AUTHOR,
        "claims_sha256": args.claims_sha256,
        "outcome_signature_sha256": outcome_sha,
        "sources": source_records,
        "attestations": {
            "historical_provenance_failure_disclosed_in_abstract_methods_and_limitations": True,
            "corrective_exploratory_not_independent_status_disclosed": True,
            "qualitative_claims_reviewed_against_exact_outcome_signature": True,
            "explainer_and_brief_reviewed": True,
        },
        "reviewed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "interactive-exact-challenge",
    }
    payload = (
        json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    output = prepare_private_output(args.output)
    write_exclusive(output, payload)
    print(json.dumps({"review": str(output), "sha256": hashlib.sha256(payload).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
