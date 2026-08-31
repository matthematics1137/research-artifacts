#!/usr/bin/env python3
"""Hash all six GGUFs once, before the validation campaign.

The live runner consumes this immutable manifest using stat-only checks. It
does not hash weight contents immediately before a timed cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SPECS = (
    ("P4R1-MIXTRAL-IQ1M", "mixtral", "Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf", 10847178368, "7e8d75c23bd4de55582ac91084a0575f081c1de374b693f5ea7ffa6fa66a2e3c"),
    ("P4R1-MIXTRAL-IQ2XXS", "mixtral", "Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf", 12556357248, "db62b07756f50d2f046b0a00dc594f1aa73dba82ebdc182d9f55c96474655f03"),
    ("P4R1-MIXTRAL-Q4KM", "mixtral", "Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf", 28448468608, "7f2f98f301a74b645d0a4c2bd64c98e9811f2dfa26dbae4787c280eb1183632c"),
    ("P4R1-QWEN3MOE-IQ1M", "qwen", "Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf", 9685733792, "d527a854db2a1582a3ce746a17b1f42d860334ece18d385ede9e2e395058b39e"),
    ("P4R1-QWEN3MOE-IQ2XXS", "qwen", "Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf", 10341814688, "aa0b04ac05f71aa94b542b074500b3de956608365e74f0e1d97ae0d6a6380269"),
    ("P4R1-QWEN3MOE-Q4KM", "qwen", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 18556686752, "6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0"),
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--mixtral-dir", required=True, type=Path)
    parser.add_argument("--qwen-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    directories = {
        "mixtral": args.mixtral_dir.resolve(strict=True),
        "qwen": args.qwen_dir.resolve(strict=True),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.parent.stat().st_mode & 0o077:
        raise SystemExit(
            f"model-manifest directory must be owner-only (0700): {output.parent}"
        )
    checksum_path = output.with_name(output.name + ".sha256")
    if output.exists() or checksum_path.exists():
        raise SystemExit(f"refusing to overwrite {output} or {checksum_path}")

    artifacts: list[dict[str, Any]] = []
    for alias, family, filename, expected_bytes, expected_sha256 in SPECS:
        if not SHA256.fullmatch(expected_sha256):
            raise RuntimeError(f"invalid pinned SHA-256 for {alias}")
        candidate = directories[family] / filename
        if candidate.is_symlink():
            raise RuntimeError(f"{alias}: model path must not be a symlink")
        path = candidate.resolve(strict=True)
        if path.parent != directories[family] or not path.is_file():
            raise RuntimeError(f"{alias}: expected one regular file directly under {directories[family]}")
        stat = path.stat()
        if stat.st_size != expected_bytes:
            raise RuntimeError(f"{alias}: expected {expected_bytes} bytes, got {stat.st_size}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{alias}: SHA-256 mismatch: {actual_sha256}")
        artifacts.append(
            {
                "alias": alias,
                "family": family,
                "filename": filename,
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": actual_sha256,
                "stat": {
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "mtime_ns": stat.st_mtime_ns,
                },
            }
        )
        print(f"verified {alias}: {expected_bytes} bytes {expected_sha256}", flush=True)

    manifest = {
        "schema": "paper4-validation-model-manifest-v1",
        "content_verification": "full SHA-256 completed before the campaign; live cells use stat-only checks",
        "artifact_count": len(artifacts),
        "total_bytes": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }
    payload = canonical_bytes(manifest)
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    write_exclusive(output, payload)
    write_exclusive(checksum_path, f"{manifest_sha256}  {output.name}\n".encode("utf-8"))
    print(json.dumps({"manifest": str(output), "sha256": manifest_sha256, "total_bytes": manifest["total_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
