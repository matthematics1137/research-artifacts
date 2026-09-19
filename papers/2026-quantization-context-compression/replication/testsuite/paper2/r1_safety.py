#!/usr/bin/env python3
"""Append-only and input-freeze helpers for the P2R1 recovery."""

from __future__ import annotations

import json
import hashlib
import fcntl
import os
import re
import uuid
from pathlib import Path
from typing import Any

from identity_guard import sha256_file, utc_now


LABEL_RE = re.compile(r"^P2R1-(Q4KXL|EXL3|IQ2S)-(TSI|WC)$")
ATTEMPT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{5,79}$")


class RunSafetyError(RuntimeError):
    pass


class TornTailInvalidation(RunSafetyError):
    """An official append-only series suffered an ambiguous partial append."""

    pass


class EvidenceLossError(RunSafetyError):
    """A request may have completed but its exact bounded evidence was lost."""

    pass


def require_r1_label(label: str) -> None:
    if not LABEL_RE.fullmatch(label):
        raise RunSafetyError(
            f"refusing non-R1 or malformed label {label!r}; expected "
            "P2R1-{Q4KXL,EXL3,IQ2S}-{TSI,WC}"
        )


def require_attempt_id(attempt_id: str) -> None:
    if not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise RunSafetyError(
            "P2R1 attempt ID must be 6-80 lowercase ASCII letters/digits/dot/"
            "underscore/hyphen and must not contain a path separator"
        )


def attempt_invalidation_path(attempt_root: Path) -> Path:
    return attempt_root / "ATTEMPT_INVALIDATED.json"


def require_attempt_eligible(attempt_root: Path) -> None:
    marker = attempt_invalidation_path(attempt_root)
    if marker.exists():
        try:
            value = json.loads(marker.read_text())
            reason = value.get("reason", "unknown")
        except (OSError, json.JSONDecodeError):
            reason = "unreadable invalidation marker"
        raise RunSafetyError(
            f"P2R1 attempt namespace is permanently ineligible ({reason}): "
            f"{attempt_root}; preserve it for forensics and require a committed "
            "protocol amendment before any replacement attempt"
        )


def invalidate_attempt(
    attempt_root: Path, *, attempt_id: str, reason: str, details: dict[str, Any]
) -> Path:
    """Permanently mark one isolated attempt as forensic-only.

    The receipt is intentionally deterministic and contains no timestamp: a
    repeated startup after the same crash state verifies the same bytes rather
    than creating mutable provenance.
    """
    require_attempt_id(attempt_id)
    marker = attempt_invalidation_path(attempt_root)
    value = {
        "schema": "paper2-r1-attempt-invalidation-v1",
        "attempt_id": attempt_id,
        "promotion_eligible": False,
        "reason": reason,
        "details": details,
        "action": (
            "preserve this namespace for forensics; do not resume or promote it; "
            "a replacement requires a prospective committed protocol amendment"
        ),
    }
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior.get("schema") != value["schema"] or prior.get("attempt_id") != attempt_id:
            raise RunSafetyError(f"malformed attempt invalidation marker: {marker}")
        return marker
    ensure_immutable_json(marker, value)
    return marker


def attempt_ambiguity_error(
    attempt_root: Path, *, attempt_id: str, reason: str, error: BaseException
) -> RunSafetyError:
    """Persist the permanent marker before propagating an evidence ambiguity."""
    marker = invalidate_attempt(
        attempt_root,
        attempt_id=attempt_id,
        reason=reason,
        details={
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return RunSafetyError(
        f"{reason}; attempt {attempt_id} is permanently ineligible ({marker})"
    )


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_exclusive(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    payload = canonical_bytes(value)
    temp = path.with_name(f".{path.name}.install.{uuid.uuid4().hex}")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise RunSafetyError(f"refusing to overwrite {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp.exists():
            temp.unlink()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": stat.st_size,
    }


def ensure_run_contract(outdir: Path, contract: dict[str, Any]) -> Path:
    """Create once, or require byte-for-byte semantic equality on resume."""
    outdir.mkdir(parents=True, exist_ok=True)
    outdir.chmod(0o700)
    path = outdir / "run_contract.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != contract:
            raise RunSafetyError(
                f"existing output has a different frozen contract: {path}; "
                "use a new result-series label instead of overwriting"
            )
    else:
        write_exclusive(path, contract)
    return path


def ensure_immutable_json(path: Path, value: dict[str, Any]) -> Path:
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != value:
            raise RunSafetyError(
                f"immutable contract mismatch at {path}; start a new named attempt"
            )
    else:
        write_exclusive(path, value)
    return path


JSONL_CHECKSUM_FIELD = "_jsonl_sha256"
JSONL_PREVIOUS_FIELD = "_jsonl_previous_sha256"
JSONL_ZERO_SHA256 = "0" * 64


def checked_jsonl_head_path(path: Path) -> Path:
    return path.with_name(path.name + ".head.json")


def _load_checked_jsonl_head(path: Path, *, byte_count: int) -> dict[str, Any] | None:
    head_path = checked_jsonl_head_path(path)
    if not head_path.exists():
        if byte_count:
            raise RunSafetyError(f"nonempty checked JSONL lacks its head receipt: {path}")
        return None
    try:
        head = json.loads(head_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSafetyError(f"checked JSONL head receipt is unreadable: {head_path}") from exc
    if (
        not isinstance(head, dict)
        or head.get("schema") != "paper2-r1-jsonl-head-v1"
        or head.get("path") != str(path.resolve())
        or type(head.get("event_count")) is not int
        or head["event_count"] < 1
        or type(head.get("byte_count")) is not int
        or not re.fullmatch(r"[0-9a-f]{64}", str(head.get("head_sha256", "")))
        or head["byte_count"] != byte_count
    ):
        raise RunSafetyError(f"checked JSONL head receipt changed or is stale: {head_path}")
    return head


def _event_digest(value: dict[str, Any]) -> str:
    clean = dict(value)
    clean.pop(JSONL_CHECKSUM_FIELD, None)
    payload = json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checked_event(value: dict[str, Any], *, previous_sha256: str) -> dict[str, Any]:
    if JSONL_CHECKSUM_FIELD in value or JSONL_PREVIOUS_FIELD in value:
        raise RunSafetyError("caller may not set reserved checked-JSONL fields")
    stored = dict(value)
    stored[JSONL_PREVIOUS_FIELD] = previous_sha256
    stored[JSONL_CHECKSUM_FIELD] = _event_digest(stored)
    return stored


def append_checked_jsonl(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Append one checksummed event; a killed writer may leave only a tail.

    A startup read may preserve an unterminated tail in a private quarantine
    and restore the verified prefix for forensic inspection. That recovery
    permanently invalidates the official series: it may not be resumed or
    promoted because the partial event could have followed a completed model
    response. Complete lines are never silently dropped or replaced.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        assert_no_torn_tail_recovery(path)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            initial_bytes = os.fstat(fd).st_size
            head = _load_checked_jsonl_head(path, byte_count=initial_bytes)
            previous = head["head_sha256"] if head else JSONL_ZERO_SHA256
            event_count = head["event_count"] if head else 0
            stored = checked_event(value, previous_sha256=previous)
            payload = (json.dumps(stored, sort_keys=True) + "\n").encode("utf-8")
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise RunSafetyError(f"short append to {path}")
                offset += written
            os.fsync(fd)
            atomic_summary(checked_jsonl_head_path(path), {
                "schema": "paper2-r1-jsonl-head-v1",
                "path": str(path.resolve()),
                "event_count": event_count + 1,
                "byte_count": initial_bytes + len(payload),
                "head_sha256": stored[JSONL_CHECKSUM_FIELD],
            })
        finally:
            os.close(fd)
        return stored
    except RunSafetyError:
        raise
    except OSError as exc:
        raise RunSafetyError(f"I/O failure while appending checked JSONL {path}") from exc


def torn_tail_recovery_receipts(path: Path) -> list[Path]:
    """Return persistent invalidation receipts for one append-only stream."""
    directory = path.parent / "torn_tail_quarantine"
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{path.name}.*.recovery.json"))


def assert_no_torn_tail_recovery(path: Path) -> None:
    receipts = torn_tail_recovery_receipts(path)
    if receipts:
        raise TornTailInvalidation(
            f"official append-only series was tail-recovered and is permanently "
            f"ineligible; preserve it for forensics and start a new named campaign: "
            f"{path} ({receipts[0]})"
        )


def _quarantine_torn_tail(
    path: Path, original: bytes, prefix: bytes, tail: bytes
) -> Path:
    original_sha = hashlib.sha256(original).hexdigest()
    prefix_sha = hashlib.sha256(prefix).hexdigest()
    tail_sha = hashlib.sha256(tail).hexdigest()
    directory = path.parent / "torn_tail_quarantine"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    quarantine = directory / f"{path.name}.{original_sha}.torn"
    receipt = directory / f"{path.name}.{original_sha}.recovery.json"
    if quarantine.exists():
        if quarantine.read_bytes() != original:
            raise RunSafetyError(f"torn-tail quarantine collision: {quarantine}")
    else:
        os.link(path, quarantine)
        quarantine.chmod(0o600)
    ensure_immutable_json(receipt, {
        "schema": "paper2-r1-jsonl-tail-recovery-v1",
        "promotion_eligible": False,
        "source": str(path.resolve()),
        "quarantine": str(quarantine.resolve()),
        "original_sha256": original_sha,
        "verified_prefix_bytes": len(prefix),
        "verified_prefix_sha256": prefix_sha,
        "torn_tail_bytes": len(tail),
        "torn_tail_sha256": tail_sha,
        "action": (
            "preserve original; replace source with checksummed complete-line prefix "
            "for forensic inspection only; invalidate official attempt and require a "
            "new named campaign"
        ),
    })
    temp = path.with_name(f"{path.name}.repair.{uuid.uuid4().hex}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(prefix):
            written = os.write(fd, prefix[offset:])
            if written <= 0:
                raise RunSafetyError(f"short repair write for {path}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temp, path)
    path.chmod(0o600)
    return receipt


def read_checked_jsonl(
    path: Path, *, recover_torn_tail: bool = False
) -> list[dict[str, Any]]:
    # A prior recovery is an enduring scientific invalidation, even though the
    # on-disk source now contains only a syntactically valid prefix.
    assert_no_torn_tail_recovery(path)
    if not path.exists():
        if checked_jsonl_head_path(path).exists():
            raise RunSafetyError(f"checked JSONL was deleted but its head remains: {path}")
        return []
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if recover_torn_tail else fcntl.LOCK_SH)
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read()
        rows: list[dict[str, Any]] = []
        prefix_end = 0
        previous = JSONL_ZERO_SHA256
        parts = data.splitlines(keepends=True)
        for index, line in enumerate(parts):
            if not line.endswith(b"\n"):
                if index != len(parts) - 1:
                    raise RunSafetyError(f"non-final torn JSONL fragment: {path}")
                break
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RunSafetyError(f"corrupt complete JSONL line in {path}") from exc
            if not isinstance(value, dict):
                raise RunSafetyError(f"non-object JSONL event in {path}")
            expected = value.get(JSONL_CHECKSUM_FIELD)
            if (
                value.get(JSONL_PREVIOUS_FIELD) != previous
                or not isinstance(expected, str)
                or expected != _event_digest(value)
            ):
                raise RunSafetyError(f"JSONL event checksum mismatch in {path}")
            rows.append(value)
            previous = expected
            prefix_end += len(line)
        if prefix_end != len(data):
            if not recover_torn_tail:
                raise RunSafetyError(f"unterminated JSONL tail in {path}")
            receipt = _quarantine_torn_tail(
                path, data, data[:prefix_end], data[prefix_end:]
            )
            raise TornTailInvalidation(
                f"quarantined an unterminated append from {path}; the official "
                f"series must not resume or promote and requires a new named "
                f"campaign ({receipt})"
            )
        if rows:
            head = _load_checked_jsonl_head(path, byte_count=len(data))
            if (
                head is None
                or head["event_count"] != len(rows)
                or head["head_sha256"] != previous
            ):
                raise RunSafetyError(f"checked JSONL chain/head mismatch in {path}")
        elif checked_jsonl_head_path(path).exists():
            raise RunSafetyError(f"empty checked JSONL has a stale head receipt: {path}")
        return rows
    finally:
        os.close(fd)


def append_session(outdir: Path, session: dict[str, Any]) -> None:
    row = dict(session)
    row.setdefault("started_at", utc_now())
    path = outdir / "run_sessions.jsonl"
    append_checked_jsonl(path, row)


def atomic_summary(path: Path, value: Any) -> None:
    """Replace a derived summary; raw request rows remain append-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    tmp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()
