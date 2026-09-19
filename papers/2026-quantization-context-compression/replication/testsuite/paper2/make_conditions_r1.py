#!/usr/bin/env python3
"""Create frozen A/L2 conditions for the pinned P2R1 WildChat windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ladder  # noqa: E402
from identity_guard import sha256_file, utc_now  # noqa: E402
from r1_safety import RunSafetyError  # noqa: E402


MODEL_REVISION = "ebaba9b0e874dadd3003ffcff828e4397e568089"
MODEL_HASHES = {
    "model.safetensors": "a33a153b2493bff6be06af6921e69de9c0d0bb6ff06fe5bbb68670ba8d980ae2",
    "tokenizer.json": "f59925fcb90c92b894cb93e51bb9b4a6105c5c249fe54ce1c704420ac39b81af",
    "tokenizer_config.json": "f90024142df07163e5e6c5b9a6ad7c8c68b22a9112af11e3db4559a9ff90f737",
    "config.json": "a3fdcf4e63057797101ecc84412bc654f9adb4e8d4ed3c91c5afb28eda327706",
    "special_tokens_map.json": "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
}
REQUIRED_PACKAGES = {
    "llmlingua": "0.2.2",
    "torch": "2.13.0",
    "transformers": "5.16.1",
    "tokenizers": "0.23.1",
    "numpy": "2.5.2",
}

STRATA = ("S-nocode", "S-code", "L-nocode", "L-code")
EXPECTED_FRAMES = {
    "TSI": {"S-nocode": 40, "S-code": 40, "L-nocode": 6, "L-code": 40},
    "WC": {"S-nocode": 40, "S-code": 40, "L-nocode": 40, "L-code": 40},
}


def stratum_key(row: dict) -> str:
    return f"{row['stratum']}-{'code' if int(row['has_code']) else 'nocode'}"


def condition_digest(rows: list[dict], field: str) -> str:
    """Bind one condition to ordered window IDs and exact UTF-8 text."""
    digest = hashlib.sha256()
    for row in rows:
        window_id = row["window_id"].encode("utf-8")
        value = row[field].encode("utf-8")
        digest.update(len(window_id).to_bytes(8, "big"))
        digest.update(window_id)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=["TSI", "WC"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots" / MODEL_REVISION,
    )
    args = parser.parse_args()
    corpus_dir = "tsi" if args.corpus == "TSI" else "wildchat"
    private_root = (HERE / "r1" / "private_inputs" / corpus_dir).resolve()
    for path in (args.out, args.manifest):
        try:
            path.resolve().relative_to(private_root)
        except ValueError as exc:
            raise RunSafetyError(f"WildChat condition output must stay below {private_root}") from exc
    if args.out.exists() or args.manifest.exists():
        raise RunSafetyError("refusing to overwrite frozen WildChat conditions")
    source_manifest = json.loads(args.source_manifest.read_text())
    expected_source_schema = (
        "paper2-r1-tsi-source-v1" if args.corpus == "TSI"
        else "paper2-r1-wildchat-source-v1"
    )
    if source_manifest.get("schema") != expected_source_schema:
        raise RunSafetyError(f"wrong pinned {args.corpus} source manifest")
    if source_manifest.get("output_sha256") != sha256_file(args.input):
        raise RunSafetyError("WildChat base windows do not match source manifest")
    observed_packages = {
        package: importlib.metadata.version(package) for package in REQUIRED_PACKAGES
    }
    if observed_packages != REQUIRED_PACKAGES:
        raise RunSafetyError(
            f"P2R1 compression environment changed: {observed_packages}; "
            f"expected {REQUIRED_PACKAGES}"
        )
    for name, expected in MODEL_HASHES.items():
        path = args.model_path / name
        if sha256_file(path) != expected:
            raise RunSafetyError(f"LLMLingua model hash mismatch: {name}")

    with args.input.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    window_ids = [row.get("window_id") for row in rows]
    if any(not value for value in window_ids) or len(window_ids) != len(set(window_ids)):
        raise RunSafetyError("base-window IDs are missing or duplicated")
    try:
        frame_counts = {
            key: sum(stratum_key(row) == key for row in rows) for key in STRATA
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RunSafetyError("base windows contain invalid stratum/code metadata") from exc
    if frame_counts != EXPECTED_FRAMES[args.corpus]:
        raise RunSafetyError(
            f"{args.corpus} frame changed: {frame_counts}; "
            f"expected {EXPECTED_FRAMES[args.corpus]}"
        )
    if any(not row.get("source_cluster_id") for row in rows):
        raise RunSafetyError("every base window needs an opaque source_cluster_id")
    from llmlingua import PromptCompressor
    compressor = PromptCompressor(
        model_name=str(args.model_path), use_llmlingua2=True, device_map="cpu"
    )
    for index, row in enumerate(rows):
        row["text_A"] = ladder.rung_A(row["text_O"])
        row["chars_A"] = len(row["text_A"])
        rate = max(0.15, min(0.9, row["chars_E"] / max(1, row["chars_O"])))
        result = compressor.compress_prompt(row["text_O"], rate=rate)
        row["text_L2"] = result["compressed_prompt"]
        row["chars_L2"] = len(row["text_L2"])
        if not row["text_L2"]:
            raise RunSafetyError(f"LLMLingua produced empty output for {row['window_id']}")
        if (index + 1) % 10 == 0:
            print(f"[{index + 1}/{len(rows)}] conditions", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.chmod(0o700)
    temp = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    temp_fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(temp_fd, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, args.out)
    manifest = {
        "schema": (
            "paper2-r1-tsi-input-v1" if args.corpus == "TSI"
            else "paper2-r1-wildchat-input-v1"
        ),
        "corpus": args.corpus,
        "created_at": utc_now(),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_revision": source_manifest.get("revision"),
        "source_shard": source_manifest.get("shard"),
        "builder_source_sha256": source_manifest.get("builder_source_sha256"),
        "ladder_source_sha256": sha256_file(HERE / "ladder.py"),
        "conditions_source_sha256": sha256_file(Path(__file__)),
        "output_sha256": sha256_file(args.out),
        "windows": len(rows),
        "window_ids_sha256": hashlib.sha256(
            "\n".join(window_ids).encode("utf-8")
        ).hexdigest(),
        "frame_counts": frame_counts,
        "condition_text_sha256": {
            condition: condition_digest(rows, f"text_{condition}")
            for condition in ("O", "A", "L2", "E")
        },
        "llmlingua": {
            "package_version": "0.2.2",
            "model": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            "model_revision": MODEL_REVISION,
            "file_sha256": MODEL_HASHES,
            "device": "cpu",
            "rate_rule": "clamp(chars_E/chars_O, 0.15, 0.90) per window"
        },
        "python_packages": observed_packages,
        "requirements_sha256": sha256_file(HERE / "requirements-r1.txt"),
        "release_boundary": source_manifest["release_boundary"]
    }
    manifest_temp = args.manifest.with_name(args.manifest.name + f".tmp.{os.getpid()}")
    manifest_fd = os.open(
        manifest_temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(manifest_fd, "w") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(manifest_temp, args.manifest)
    print(json.dumps({"windows": len(rows), "output_sha256": manifest["output_sha256"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 CONDITION BUILD ABORTED: {exc}") from exc
