#!/usr/bin/env python3
"""Fail-closed endpoint identity receipts for the Paper 2 recovery.

Health is deliberately insufficient. A valid receipt binds an endpoint to the
newly launched process, the exact model ID reported by the API, an engine
revision, and an artifact checksum. The module uses only the Python standard
library and performs no inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "paper2-endpoint-identity-v1"
CONTRACT = "EMPIRICAL_RESEARCH_LOOP.md@1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE_MANIFEST = (
    REPO_ROOT
    / "publication/papers/qwen38_27b_12gb/public/environment/model_artifacts.json"
)
MODEL_SOURCE_MANIFEST_SHA256 = (
    "6d2a262b60774b58641611427cdec7b1e270ac6c69e8dd1fd55d9b77551a94db"
)


class IdentityError(RuntimeError):
    """The live service does not prove the identity required by the run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> dict[str, Any]:
    """Hash a file or a directory tree, including relative file names."""
    path = path.resolve()
    if not path.exists():
        raise IdentityError(f"artifact does not exist: {path}")
    if path.is_file():
        stat = path.stat()
        return {
            "kind": "file",
            "path": str(path),
            "sha256": sha256_file(path),
            "file_count": 1,
            "total_bytes": stat.st_size,
            "max_mtime_ns": stat.st_mtime_ns,
        }
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise IdentityError(f"artifact directory is empty: {path}")
    digest = hashlib.sha256()
    total = 0
    max_mtime = 0
    for file_path in files:
        rel = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        stat = file_path.stat()
        total += stat.st_size
        max_mtime = max(max_mtime, stat.st_mtime_ns)
    return {
        "kind": "directory",
        "path": str(path),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
        "max_mtime_ns": max_mtime,
    }


def artifact_fast_state(path: Path) -> tuple[int, int, int]:
    path = path.resolve()
    if path.is_file():
        stat = path.stat()
        return 1, stat.st_size, stat.st_mtime_ns
    files = [p for p in path.rglob("*") if p.is_file()]
    stats = [p.stat() for p in files]
    return len(stats), sum(s.st_size for s in stats), max(s.st_mtime_ns for s in stats)


def source_provenance(
    artifact_path: Path, source_manifest_path: Path, artifact_id: str,
    *, verify_source_files: bool,
) -> dict[str, Any]:
    """Bind a local deployment to the immutable upstream artifact inventory.

    EXL3's two weight shards are enumerated by the Paper 1 source manifest;
    the separate local-tree hash also covers its config/tokenizer/calibration
    files. We do not imply that every non-weight local file is an independently
    enumerated upstream download.
    """
    manifest_path = source_manifest_path.resolve()
    if (
        manifest_path != MODEL_SOURCE_MANIFEST.resolve()
        or sha256_file(manifest_path) != MODEL_SOURCE_MANIFEST_SHA256
    ):
        raise IdentityError("model source manifest is not the frozen Paper 1 inventory")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "qwen38-model-artifacts-v1":
        raise IdentityError("model source manifest has the wrong schema")
    entries = [
        row for row in manifest.get("artifacts", [])
        if isinstance(row, dict) and row.get("artifact_id") == artifact_id
    ]
    if not entries:
        raise IdentityError(f"model source manifest lacks artifact {artifact_id}")
    repos = {row.get("upstream_repo") for row in entries}
    revisions = {row.get("revision") for row in entries}
    licenses = {row.get("license") for row in entries}
    if len(repos) != 1 or len(revisions) != 1 or len(licenses) != 1:
        raise IdentityError(f"inconsistent upstream provenance for {artifact_id}")
    root = artifact_path.resolve()
    if root.is_file():
        if len(entries) != 1 or root.name != entries[0].get("filename"):
            raise IdentityError(f"local filename does not match source manifest: {root}")
        if root.stat().st_size != entries[0].get("bytes"):
            raise IdentityError(f"local artifact size differs from source manifest: {root}")
        if verify_source_files and sha256_file(root) != entries[0].get("sha256"):
            raise IdentityError(f"local artifact hash differs from source manifest: {root}")
        coverage = "exact upstream file"
    else:
        for entry in entries:
            source_file = root / str(entry.get("filename", ""))
            if not source_file.is_file() or source_file.stat().st_size != entry.get("bytes"):
                raise IdentityError(
                    f"local EXL3 source shard differs from manifest: {source_file}"
                )
            if verify_source_files and sha256_file(source_file) != entry.get("sha256"):
                raise IdentityError(
                    f"local EXL3 source shard hash differs from manifest: {source_file}"
                )
        coverage = (
            "upstream weight shards enumerated exactly; the artifact record's "
            "separate full local-tree hash covers all additional local files"
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": MODEL_SOURCE_MANIFEST_SHA256,
        "artifact_id": artifact_id,
        "upstream_repo": next(iter(repos)),
        "revision": next(iter(revisions)),
        "license": next(iter(licenses)),
        "source_entries": sorted(entries, key=lambda row: row["filename"]),
        "coverage": coverage,
    }


def load_artifact_record(path: Path, *, full_hash: bool = False) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record.get("schema") != "paper2-artifact-identity-v1":
        raise IdentityError(f"unsupported artifact record schema in {path}")
    artifact = record.get("artifact")
    if not isinstance(artifact, dict):
        raise IdentityError(f"artifact record is malformed: {path}")
    artifact_path = Path(artifact["path"])
    provenance = record.get("source_provenance")
    if not isinstance(provenance, dict):
        raise IdentityError(f"artifact record lacks source provenance: {path}")
    expected_provenance = source_provenance(
        artifact_path,
        Path(provenance.get("manifest", "")),
        str(provenance.get("artifact_id", "")),
        verify_source_files=False,
    )
    if provenance != expected_provenance:
        raise IdentityError(f"artifact source provenance changed: {path}")
    current_fast = artifact_fast_state(artifact_path)
    recorded_fast = (
        artifact["file_count"], artifact["total_bytes"], artifact["max_mtime_ns"]
    )
    if current_fast != recorded_fast:
        raise IdentityError("artifact size/mtime state no longer matches frozen record")
    if full_hash and sha256_path(artifact_path) != artifact:
        raise IdentityError("artifact contents no longer match frozen checksum")
    return record


def create_artifact_record(
    artifact_path: Path, output_path: Path, *, source_manifest_path: Path,
    artifact_id: str,
) -> dict[str, Any]:
    artifact = sha256_path(artifact_path)
    provenance = source_provenance(
        artifact_path, source_manifest_path, artifact_id,
        verify_source_files=True,
    )
    if artifact["kind"] == "file" and artifact["sha256"] != provenance["source_entries"][0]["sha256"]:
        raise IdentityError("local file hash and upstream source hash disagree")
    record = {
        "schema": "paper2-artifact-identity-v1",
        "created_at": utc_now(),
        "artifact": artifact,
        "source_provenance": provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    try:
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise IdentityError(f"refusing to overwrite artifact record: {output_path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return record


def parse_endpoint(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise IdentityError("P2R1 permits only a loopback http endpoint")
    if parsed.port is None:
        raise IdentityError("base URL must contain an explicit port")
    return parsed.hostname, parsed.port


def fetch_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "p2r1-identity/1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise IdentityError(f"non-object response from {url}")
    return value


def listening_socket_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.exists():
            continue
        with table.open() as handle:
            next(handle, None)
            for line in handle:
                fields = line.split()
                if len(fields) < 10 or fields[3] != "0A":
                    continue
                local_port = int(fields[1].split(":")[1], 16)
                if local_port == port:
                    inodes.add(fields[9])
    return inodes


def listening_pids(port: int) -> set[int]:
    inodes = listening_socket_inodes(port)
    if not inodes:
        return set()
    wanted = {f"socket:[{inode}]" for inode in inodes}
    owners: set[int] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.readlink(fd) in wanted:
                        owners.add(int(proc.name))
                        break
                except (FileNotFoundError, PermissionError, OSError):
                    continue
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return owners


def process_alive(pid: int) -> bool:
    proc = Path("/proc") / str(pid)
    try:
        state = (proc / "stat").read_text().rsplit(")", 1)[1].strip().split()[0]
    except (FileNotFoundError, PermissionError, OSError, IndexError):
        return False
    return state != "Z"


def parent_pid(pid: int) -> int | None:
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def descends_from(pid: int, root_pid: int) -> bool:
    seen: set[int] = set()
    current: int | None = pid
    while current and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        current = parent_pid(current)
    return False


def process_cmdline(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError) as exc:
        raise IdentityError(f"cannot read launched PID {pid}: {exc}") from exc
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def process_fingerprint(pid: int) -> dict[str, Any]:
    """Identity fields that do not survive PID reuse."""
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text()
        # comm is parenthesized and may contain spaces; fields after the final
        # ')' start at field 3. starttime is field 22, hence offset 19 here.
        after_comm = stat_text.rsplit(")", 1)[1].strip().split()
        starttime_ticks = int(after_comm[19])
        exe = (proc / "exe").resolve(strict=True)
        exe_stat = exe.stat()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError) as exc:
        raise IdentityError(f"cannot fingerprint launched PID {pid}: {exc}") from exc
    return {
        "pid": pid,
        "boot_id": boot_id,
        "starttime_ticks": starttime_ticks,
        "executable": str(exe),
        "executable_device": exe_stat.st_dev,
        "executable_inode": exe_stat.st_ino,
    }


def assert_process_fingerprint(pid: int, expected: dict[str, Any]) -> None:
    current = process_fingerprint(pid)
    if current != expected:
        raise IdentityError(
            f"PID {pid} was reused or executable identity changed; "
            "recorded process is no longer live"
        )


def create_process_record(pid: int, output_path: Path) -> dict[str, Any]:
    record = {
        "schema": "paper2-launched-process-v1",
        "captured_at": utc_now(),
        "process": process_fingerprint(pid),
        "cmdline": process_cmdline(pid),
        "cwd": str((Path("/proc") / str(pid) / "cwd").resolve(strict=True)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    try:
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise IdentityError(f"refusing to overwrite process record: {output_path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return record


def load_process_record(path: Path, expected_pid: int) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record.get("schema") != "paper2-launched-process-v1":
        raise IdentityError(f"unsupported process record: {path}")
    if record.get("process", {}).get("pid") != expected_pid:
        raise IdentityError("process record PID does not match launcher PID")
    assert_process_fingerprint(expected_pid, record["process"])
    return record


def assert_port_owned_by_process(port: int, root_pid: int) -> list[int]:
    owners = sorted(listening_pids(port))
    if not owners:
        raise IdentityError(f"no listening process found on port {port}")
    foreign = [pid for pid in owners if not descends_from(pid, root_pid)]
    if foreign:
        raise IdentityError(
            f"port {port} is owned by PID(s) {owners}, not launched PID tree {root_pid}"
        )
    return owners


def assert_port_free(host: str, port: int) -> None:
    if listening_socket_inodes(port):
        raise IdentityError(f"required port {host}:{port} already has a listener")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        # Permit a recently closed test/server socket in TIME_WAIT; that state
        # is not a listener and does not represent the orphan-service hazard.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise IdentityError(f"required port {host}:{port} is not free: {exc}") from exc
    finally:
        sock.close()


def read_log_prefix(path: Path, prefix_bytes: int | None = None) -> tuple[int, str, str]:
    if not path.exists():
        data = b""
    elif prefix_bytes is None:
        data = path.read_bytes()
    else:
        with path.open("rb") as handle:
            data = handle.read(prefix_bytes)
    return len(data), hashlib.sha256(data).hexdigest(), data.decode("utf-8", "replace")


def live_identity(
    *,
    base_url: str,
    expected_pid: int,
    expected_model_id: str,
    expected_owned_by: str,
    cmd_fragments: list[str],
    server_log: Path,
    forbidden_log_patterns: list[str],
    expected_process: dict[str, Any] | None = None,
    expected_cwd: str | None = None,
    config_file: Path | None = None,
    config_sha256: str | None = None,
    required_config_lines: list[str] | None = None,
    required_log_patterns: list[str] | None = None,
    expected_log_prefix_bytes: int | None = None,
    expected_log_prefix_sha256: str | None = None,
) -> dict[str, Any]:
    host, port = parse_endpoint(base_url)
    if not process_alive(expected_pid):
        raise IdentityError(f"launched PID {expected_pid} is no longer alive")
    current_process = process_fingerprint(expected_pid)
    if expected_process is not None and current_process != expected_process:
        raise IdentityError(f"launched PID {expected_pid} no longer matches its start identity")
    cmdline = process_cmdline(expected_pid)
    joined = "\0".join(cmdline)
    for fragment in cmd_fragments:
        if fragment not in joined:
            raise IdentityError(f"launched command lacks required fragment: {fragment}")
    cwd = str((Path("/proc") / str(expected_pid) / "cwd").resolve(strict=True))
    if expected_cwd is not None and cwd != str(Path(expected_cwd).resolve()):
        raise IdentityError(f"process cwd mismatch: expected {expected_cwd}, got {cwd}")
    if config_file is not None:
        if sha256_file(config_file) != config_sha256:
            raise IdentityError("dedicated server config bytes changed after launch")
        config_text = config_file.read_text()
        for line in required_config_lines or []:
            if line not in {entry.strip() for entry in config_text.splitlines()}:
                raise IdentityError(f"dedicated config lacks frozen line: {line}")
    owners = assert_port_owned_by_process(port, expected_pid)
    health = fetch_json(base_url.rstrip("/") + "/health")
    if health.get("status") not in {"ok", "healthy"}:
        # llama.cpp commonly returns {"status":"ok"}; tabbyAPI returns healthy.
        raise IdentityError(f"endpoint is not healthy: {health!r}")
    models = fetch_json(base_url.rstrip("/") + "/v1/models")
    cards = models.get("data")
    if not isinstance(cards, list) or not cards:
        raise IdentityError("/v1/models returned no loaded model")
    matching = [card for card in cards if card.get("id") == expected_model_id]
    if len(matching) != 1:
        ids = [card.get("id") for card in cards]
        raise IdentityError(f"expected model ID {expected_model_id!r}; endpoint reported {ids!r}")
    owner = matching[0].get("owned_by")
    if owner != expected_owned_by:
        raise IdentityError(f"expected API owner {expected_owned_by!r}; got {owner!r}")
    log_bytes, log_hash, log_text = read_log_prefix(
        server_log, expected_log_prefix_bytes
    )
    if expected_log_prefix_bytes is not None and (
        log_bytes != expected_log_prefix_bytes
        or log_hash != expected_log_prefix_sha256
    ):
        raise IdentityError("server log identity prefix changed after receipt creation")
    for pattern in forbidden_log_patterns:
        if pattern.casefold() in log_text.casefold():
            raise IdentityError(f"server log contains forbidden pattern: {pattern}")
    normalized_log = " ".join(log_text.split())
    for pattern in required_log_patterns or []:
        if " ".join(pattern.split()) not in normalized_log:
            raise IdentityError(f"server log lacks required identity pattern: {pattern}")
    return {
        "host": host,
        "port": port,
        "listener_pids": owners,
        "process_fingerprint": current_process,
        "launch_cmdline": cmdline,
        "cwd": cwd,
        "health_readback": health,
        "models_readback": models,
        "server_log": str(server_log.resolve()),
        "server_log_prefix_bytes": log_bytes,
        "server_log_prefix_sha256": log_hash,
    }


def official_llama_launch_contract(
    *, cmdline: list[str], process: dict[str, Any], artifact_path: str,
    expected_model_id: str, base_url: str,
) -> dict[str, Any] | None:
    """Require the complete frozen llama.cpp argv for official P2R1 tiers."""
    if re.fullmatch(r"P2R1-Q4KXL-[0-9a-f]{12}", expected_model_id):
        ngl = "33"
    elif re.fullmatch(r"P2R1-IQ2S-[0-9a-f]{12}", expected_model_id):
        ngl = "99"
    else:
        return None
    _host, port = parse_endpoint(base_url)
    if not cmdline:
        raise IdentityError("llama.cpp launch command is empty")
    try:
        executable = str(Path(cmdline[0]).resolve(strict=True))
    except OSError as exc:
        raise IdentityError("cannot resolve llama.cpp launch executable") from exc
    if executable != process.get("executable"):
        raise IdentityError("llama.cpp argv[0] differs from launched executable inode")
    expected = [
        cmdline[0], "-m", artifact_path, "-ngl", ngl, "-c", "8192",
        "-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on",
        "--jinja", "--host", "127.0.0.1", "--port", str(port),
        "-t", "16", "--alias", expected_model_id, "-lv", "4",
    ]
    if cmdline != expected:
        raise IdentityError(
            f"official llama.cpp argv differs from frozen launch contract: {cmdline!r}"
        )
    return {
        "schema": "paper2-r1-llama-launch-contract-v1",
        "argv": expected,
        "options": {
            "model": artifact_path, "gpu_layers": int(ngl),
            "context_tokens": 8192, "cache_key": "q8_0",
            "cache_value": "q8_0", "flash_attention": "on",
            "jinja": True, "host": "127.0.0.1", "port": port,
            "threads": 16, "alias": expected_model_id,
            "log_verbosity": 4,
        },
    }


def create_receipt(
    *,
    base_url: str,
    expected_pid: int,
    expected_model_id: str,
    expected_owned_by: str,
    engine: str,
    engine_revision: str,
    artifact_record_path: Path,
    process_record_path: Path,
    server_log: Path,
    receipt_path: Path,
    cmd_fragments: list[str],
    forbidden_log_patterns: list[str],
    expected_cwd: str | None,
    config_file: Path | None,
    required_config_lines: list[str],
    required_log_patterns: list[str],
    wait_seconds: float,
) -> dict[str, Any]:
    launch_record = load_process_record(process_record_path, expected_pid)
    frozen_process = launch_record["process"]
    config_hash = sha256_file(config_file) if config_file is not None else None
    deadline = time.monotonic() + wait_seconds
    last_error: Exception | None = None
    while time.monotonic() <= deadline:
        if not process_alive(expected_pid):
            raise IdentityError(f"launched PID {expected_pid} exited before identity was proven")
        assert_process_fingerprint(expected_pid, frozen_process)
        try:
            live = live_identity(
                base_url=base_url,
                expected_pid=expected_pid,
                expected_model_id=expected_model_id,
                expected_owned_by=expected_owned_by,
                cmd_fragments=cmd_fragments,
                server_log=server_log,
                forbidden_log_patterns=forbidden_log_patterns,
                expected_process=frozen_process,
                expected_cwd=expected_cwd,
                config_file=config_file,
                config_sha256=config_hash,
                required_config_lines=required_config_lines,
                required_log_patterns=required_log_patterns,
            )
            break
        except Exception as exc:  # model load/readback may not be ready yet
            last_error = exc
            time.sleep(0.5)
    else:
        raise IdentityError(f"identity timeout: {last_error}")

    # Deliberately do not read weight contents here. Hashing a multi-GiB mmap
    # artifact immediately before evaluation would perturb the host page cache
    # and therefore the timing measurement. The content hash is frozen in a
    # separately generated pre-campaign record; live stages check only its
    # file count, byte size, and mtimes.
    artifact_record = load_artifact_record(artifact_record_path, full_hash=False)
    artifact = artifact_record["artifact"]
    llama_launch_contract = None
    if engine == "llama.cpp":
        llama_launch_contract = official_llama_launch_contract(
            cmdline=live["launch_cmdline"], process=frozen_process,
            artifact_path=artifact["path"], expected_model_id=expected_model_id,
            base_url=base_url,
        )
    receipt = {
        "schema": SCHEMA,
        "research_contract": CONTRACT,
        "created_at": utc_now(),
        "base_url": base_url.rstrip("/"),
        "launched_pid": expected_pid,
        "launch_process_record": str(process_record_path.resolve()),
        "launch_process_record_sha256": sha256_file(process_record_path),
        "expected_model_id": expected_model_id,
        "expected_owned_by": expected_owned_by,
        "engine": engine,
        "engine_revision": engine_revision,
        "artifact": artifact,
        "artifact_record": str(artifact_record_path.resolve()),
        "artifact_record_sha256": sha256_file(artifact_record_path),
        "cmd_fragments": cmd_fragments,
        "forbidden_log_patterns": forbidden_log_patterns,
        "required_log_patterns": required_log_patterns,
        "expected_cwd": str(Path(expected_cwd).resolve()) if expected_cwd else None,
        "config_file": str(config_file.resolve()) if config_file else None,
        "config_sha256": config_hash,
        "required_config_lines": required_config_lines,
        "identity": live,
        "llama_launch_contract": llama_launch_contract,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    try:
        fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise IdentityError(f"refusing to overwrite identity receipt: {receipt_path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return receipt


def validate_receipt(path: Path, *, full_artifact_hash: bool = False) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != SCHEMA:
        raise IdentityError(f"unsupported identity receipt schema in {path}")
    artifact = receipt["artifact"]
    artifact_path = Path(artifact["path"])
    if full_artifact_hash:
        current = sha256_path(artifact_path)
        if current != artifact:
            raise IdentityError("artifact checksum/state no longer matches identity receipt")
    else:
        current_fast = artifact_fast_state(artifact_path)
        recorded_fast = (
            artifact["file_count"], artifact["total_bytes"], artifact["max_mtime_ns"]
        )
        if current_fast != recorded_fast:
            raise IdentityError("artifact size/mtime state no longer matches identity receipt")
    live = live_identity(
        base_url=receipt["base_url"],
        expected_pid=int(receipt["launched_pid"]),
        expected_model_id=receipt["expected_model_id"],
        expected_owned_by=receipt["expected_owned_by"],
        cmd_fragments=list(receipt.get("cmd_fragments", [])),
        server_log=Path(receipt["identity"]["server_log"]),
        forbidden_log_patterns=list(receipt.get("forbidden_log_patterns", [])),
        expected_process=receipt["identity"]["process_fingerprint"],
        expected_cwd=receipt.get("expected_cwd"),
        config_file=Path(receipt["config_file"]) if receipt.get("config_file") else None,
        config_sha256=receipt.get("config_sha256"),
        required_config_lines=list(receipt.get("required_config_lines", [])),
        required_log_patterns=list(receipt.get("required_log_patterns", [])),
        expected_log_prefix_bytes=receipt["identity"]["server_log_prefix_bytes"],
        expected_log_prefix_sha256=receipt["identity"]["server_log_prefix_sha256"],
    )
    if receipt.get("engine") == "llama.cpp":
        expected_contract = official_llama_launch_contract(
            cmdline=live["launch_cmdline"],
            process=receipt["identity"]["process_fingerprint"],
            artifact_path=artifact["path"],
            expected_model_id=receipt["expected_model_id"],
            base_url=receipt["base_url"],
        )
        if receipt.get("llama_launch_contract") != expected_contract:
            raise IdentityError("llama.cpp launch contract changed after receipt creation")
    return receipt


def receipt_sha256(path: Path) -> str:
    return sha256_file(path)


def record_shutdown(receipt_path: Path, output_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    pid = int(receipt["launched_pid"])
    host, port = parse_endpoint(receipt["base_url"])
    if process_alive(pid):
        current = process_fingerprint(pid)
        if current == receipt["identity"]["process_fingerprint"]:
            raise IdentityError(f"launched PID {pid} is still alive")
    assert_port_free(host, port)
    record = {
        "schema": "paper2-endpoint-shutdown-v1",
        "closed_at": utc_now(),
        "identity_receipt": str(receipt_path.resolve()),
        "identity_receipt_sha256": receipt_sha256(receipt_path),
        "launched_pid": pid,
        "host": host,
        "port": port,
        "verified_pid_absent": True,
        "verified_port_free": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    try:
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise IdentityError(f"refusing to overwrite shutdown record: {output_path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return record


def terminate_process_record(process_record_path: Path, wait_seconds: float = 30) -> None:
    """Stop only the frozen process group; never pattern-match unrelated jobs."""
    record = json.loads(process_record_path.read_text())
    if record.get("schema") != "paper2-launched-process-v1":
        raise IdentityError(f"unsupported process record: {process_record_path}")
    expected = record["process"]
    pid = int(expected["pid"])
    if not process_alive(pid):
        return
    assert_process_fingerprint(pid, expected)
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    if pgid != pid:
        raise IdentityError(
            f"refusing shutdown: launched PID {pid} is not its own process-group leader"
        )
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + wait_seconds
    while process_alive(pid) and time.monotonic() < deadline:
        try:
            assert_process_fingerprint(pid, expected)
        except IdentityError:
            return
        time.sleep(0.2)
    if process_alive(pid):
        assert_process_fingerprint(pid, expected)
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
    if process_alive(pid):
        try:
            assert_process_fingerprint(pid, expected)
        except IdentityError:
            return
        raise IdentityError(f"launched process group {pgid} did not terminate")


def terminate_receipt(
    receipt_path: Path, shutdown_path: Path, wait_seconds: float = 30
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCHEMA:
        raise IdentityError(f"unsupported identity receipt: {receipt_path}")
    terminate_process_record(Path(receipt["launch_process_record"]), wait_seconds)
    return record_shutdown(receipt_path, shutdown_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    free = sub.add_parser("port-free")
    free.add_argument("--host", default="127.0.0.1")
    free.add_argument("--port", required=True, action="append", type=int)

    create = sub.add_parser("create")
    create.add_argument("--base-url", required=True)
    create.add_argument("--expected-pid", required=True, type=int)
    create.add_argument("--expected-model-id", required=True)
    create.add_argument("--expected-owned-by", required=True)
    create.add_argument("--engine", required=True)
    create.add_argument("--engine-revision", required=True)
    create.add_argument("--artifact-record", required=True, type=Path)
    create.add_argument("--process-record", required=True, type=Path)
    create.add_argument("--server-log", required=True, type=Path)
    create.add_argument("--receipt", required=True, type=Path)
    create.add_argument("--cmd-fragment", action="append", default=[])
    create.add_argument("--forbidden-log-pattern", action="append", default=[])
    create.add_argument("--required-log-pattern", action="append", default=[])
    create.add_argument("--expected-cwd")
    create.add_argument("--config-file", type=Path)
    create.add_argument("--required-config-line", action="append", default=[])
    create.add_argument("--wait-seconds", type=float, default=600)

    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument("--full-artifact-hash", action="store_true")

    fingerprint = sub.add_parser("fingerprint-artifact")
    fingerprint.add_argument("--artifact", required=True, type=Path)
    fingerprint.add_argument("--output", required=True, type=Path)
    fingerprint.add_argument(
        "--source-manifest", required=True, type=Path,
        help="frozen Paper 1 model-artifact source inventory",
    )
    fingerprint.add_argument("--artifact-id", required=True)

    verify_artifact = sub.add_parser("verify-artifact")
    verify_artifact.add_argument("--record", required=True, type=Path)
    verify_artifact.add_argument("--full", action="store_true")

    capture = sub.add_parser("capture-process")
    capture.add_argument("--pid", required=True, type=int)
    capture.add_argument("--output", required=True, type=Path)

    close = sub.add_parser("record-shutdown")
    close.add_argument("--receipt", required=True, type=Path)
    close.add_argument("--output", required=True, type=Path)
    terminate = sub.add_parser("terminate")
    terminate.add_argument("--receipt", required=True, type=Path)
    terminate.add_argument("--shutdown", required=True, type=Path)
    terminate.add_argument("--wait-seconds", type=float, default=30)

    terminate_process = sub.add_parser("terminate-process")
    terminate_process.add_argument("--process-record", required=True, type=Path)
    terminate_process.add_argument("--wait-seconds", type=float, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "port-free":
        for port in args.port:
            assert_port_free(args.host, port)
        print("identity preflight: requested port(s) are free")
    elif args.command == "create":
        receipt = create_receipt(
            base_url=args.base_url,
            expected_pid=args.expected_pid,
            expected_model_id=args.expected_model_id,
            expected_owned_by=args.expected_owned_by,
            engine=args.engine,
            engine_revision=args.engine_revision,
            artifact_record_path=args.artifact_record,
            process_record_path=args.process_record,
            server_log=args.server_log,
            receipt_path=args.receipt,
            cmd_fragments=args.cmd_fragment,
            forbidden_log_patterns=args.forbidden_log_pattern,
            expected_cwd=args.expected_cwd,
            config_file=args.config_file,
            required_config_lines=args.required_config_line,
            required_log_patterns=args.required_log_pattern,
            wait_seconds=args.wait_seconds,
        )
        print(json.dumps({
            "receipt": str(args.receipt),
            "model": receipt["expected_model_id"],
            "artifact_sha256": receipt["artifact"]["sha256"],
        }, sort_keys=True))
    elif args.command == "verify":
        receipt = validate_receipt(args.receipt, full_artifact_hash=args.full_artifact_hash)
        print(json.dumps({"status": "verified", "model": receipt["expected_model_id"]}))
    elif args.command == "fingerprint-artifact":
        record = create_artifact_record(
            args.artifact, args.output,
            source_manifest_path=args.source_manifest,
            artifact_id=args.artifact_id,
        )
        print(json.dumps({
            "record": str(args.output),
            "artifact_sha256": record["artifact"]["sha256"],
        }, sort_keys=True))
    elif args.command == "verify-artifact":
        record = load_artifact_record(args.record, full_hash=args.full)
        print(json.dumps({
            "status": "verified",
            "mode": "full" if args.full else "fast",
            "artifact_sha256": record["artifact"]["sha256"],
        }, sort_keys=True))
    elif args.command == "capture-process":
        record = create_process_record(args.pid, args.output)
        print(json.dumps({
            "record": str(args.output),
            "pid": record["process"]["pid"],
            "starttime_ticks": record["process"]["starttime_ticks"],
        }, sort_keys=True))
    elif args.command == "record-shutdown":
        record_shutdown(args.receipt, args.output)
        print(json.dumps({"status": "closed", "record": str(args.output)}))
    elif args.command == "terminate":
        terminate_receipt(args.receipt, args.shutdown, args.wait_seconds)
        print(json.dumps({"status": "closed", "record": str(args.shutdown)}))
    elif args.command == "terminate-process":
        terminate_process_record(args.process_record, args.wait_seconds)
        print(json.dumps({"status": "closed", "record": str(args.process_record)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IdentityError as exc:
        raise SystemExit(f"IDENTITY FAILURE: {exc}") from exc
