#!/usr/bin/env python3
"""Generate the exact private tabbyAPI config used by a P2R1 stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from identity_guard import sha256_file, utc_now  # noqa: E402
from r1_safety import RunSafetyError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    runtime_root = (HERE / "r1" / "private_inputs" / "runtime").resolve()
    for path in (args.out, args.manifest):
        try:
            path.resolve().relative_to(runtime_root)
        except ValueError as exc:
            raise RunSafetyError(f"runtime config must stay below {runtime_root}") from exc
    model_dir = args.model_dir.resolve()
    artifact = model_dir / "qwen38-exl3-2.0"
    if not artifact.is_dir():
        raise RunSafetyError(f"EXL3 artifact directory is missing: {artifact}")
    if args.out.exists() or args.manifest.exists():
        raise RunSafetyError("refusing to overwrite a frozen runtime config")
    template_path = HERE / "tabby_p2r1_config.template.yml"
    rendered = template_path.read_text().replace("__MODEL_DIR__", str(model_dir)).replace(
        "__PORT__", str(args.port)
    )
    if "__" in rendered:
        raise RunSafetyError("unresolved runtime-config placeholder")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.chmod(0o700)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "schema": "paper2-r1-tabby-config-v1",
        "created_at": utc_now(),
        "template_sha256": sha256_file(template_path),
        "config_sha256": sha256_file(args.out),
        "config": str(args.out.resolve()),
        "model_dir": str(model_dir),
        "model_name": "qwen38-exl3-2.0",
        "artifact": str(artifact.resolve()),
        "port": args.port,
        "max_seq_len": 8192,
        "cache_size": 8192,
        "cache_mode": "Q8",
        "security_and_logging": {
            "host": "127.0.0.1",
            "disable_auth": True,
            "disable_fetch_requests": True,
            "log_prompt": False,
            "log_generation_params": False,
            "log_requests": False,
            "log_chat_completion_requests": False,
            "backend": "exllamav3",
        },
    }
    fd = os.open(args.manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"config": str(args.out), "sha256": manifest["config_sha256"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 TABBY CONFIG ABORTED: {exc}") from exc
