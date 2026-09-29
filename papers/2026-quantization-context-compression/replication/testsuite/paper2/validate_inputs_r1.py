#!/usr/bin/env python3
"""Analysis-only validation of frozen P2R1 private inputs and runtime config."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from identity_guard import sha256_file  # noqa: E402
from make_conditions_r1 import (  # noqa: E402
    EXPECTED_FRAMES,
    MODEL_HASHES,
    MODEL_REVISION,
    REQUIRED_PACKAGES,
    condition_digest,
)
import ladder  # noqa: E402
from r1_safety import RunSafetyError  # noqa: E402


WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
WILDCHAT_SHARD_SHA = "abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RunSafetyError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_corpus(corpus: str, directory: Path) -> None:
    base = directory / "windows_base.jsonl"
    source_path = directory / "source_manifest.json"
    windows_path = directory / "windows.jsonl"
    input_path = directory / "input_manifest.json"
    for path in (base, source_path, windows_path, input_path):
        if not path.is_file():
            raise RunSafetyError(f"missing frozen {corpus} input: {path}")
    source = load_json(source_path)
    manifest = load_json(input_path)
    expected_source_schema = (
        "paper2-r1-tsi-source-v1" if corpus == "TSI"
        else "paper2-r1-wildchat-source-v1"
    )
    expected_input_schema = (
        "paper2-r1-tsi-input-v1" if corpus == "TSI"
        else "paper2-r1-wildchat-input-v1"
    )
    if source.get("schema") != expected_source_schema or manifest.get("schema") != expected_input_schema:
        raise RunSafetyError(f"wrong frozen manifest schema for {corpus}")
    if source.get("output_sha256") != sha256_file(base):
        raise RunSafetyError(f"{corpus} source-window hash changed")
    if manifest.get("output_sha256") != sha256_file(windows_path):
        raise RunSafetyError(f"{corpus} final-window hash changed")
    if manifest.get("source_manifest_sha256") != sha256_file(source_path):
        raise RunSafetyError(f"{corpus} source-manifest binding changed")
    if manifest.get("requirements_sha256") != sha256_file(HERE / "requirements-r1.txt"):
        raise RunSafetyError(f"{corpus} requirements binding changed")
    if source.get("ladder_source_sha256") != sha256_file(HERE / "ladder.py"):
        raise RunSafetyError(f"{corpus} source ladder binding changed")
    expected_builder = HERE / (
        "freeze_tsi_r1.py" if corpus == "TSI" else "build_wildchat_r1.py"
    )
    if source.get("builder_source_sha256") != sha256_file(expected_builder):
        raise RunSafetyError(f"{corpus} source-builder binding changed")
    if (
        manifest.get("conditions_source_sha256")
        != sha256_file(HERE / "make_conditions_r1.py")
        or manifest.get("ladder_source_sha256") != sha256_file(HERE / "ladder.py")
    ):
        raise RunSafetyError(f"{corpus} condition-builder binding changed")
    if corpus == "TSI" and (
        source.get("e_recomputed_from_text_o") is not True
        or source.get("legacy_derived_condition_fields_discarded")
        != ["A", "L2", "E"]
        or source.get("legacy_e_exact_match_count") != 0
        or source.get("legacy_e_mismatch_count") != 126
    ):
        raise RunSafetyError("TSI E condition lacks a pinned clean recomputation assertion")
    if manifest.get("python_packages") != REQUIRED_PACKAGES:
        raise RunSafetyError(f"{corpus} compression environment differs from protocol")
    llm = manifest.get("llmlingua", {})
    if (
        llm.get("package_version") != "0.2.2"
        or llm.get("model_revision") != MODEL_REVISION
        or llm.get("file_sha256") != MODEL_HASHES
    ):
        raise RunSafetyError(f"{corpus} LLMLingua provenance changed")
    base_rows = read_jsonl(base)
    rows = read_jsonl(windows_path)
    expected_n = sum(EXPECTED_FRAMES[corpus].values())
    if len(base_rows) != expected_n or len(rows) != expected_n:
        raise RunSafetyError(f"{corpus} frame length changed")
    if [row["window_id"] for row in base_rows] != [row["window_id"] for row in rows]:
        raise RunSafetyError(f"{corpus} condition build changed window order")
    ids = [row["window_id"] for row in rows]
    clusters = [row.get("source_cluster_id") for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in clusters):
        raise RunSafetyError(f"{corpus} IDs/clusters are incomplete")
    cluster_pattern = r"ctsi\d{5}" if corpus == "TSI" else r"cwc\d{5}"
    if any(not re.fullmatch(cluster_pattern, str(value)) for value in clusters):
        raise RunSafetyError(f"{corpus} contains a non-opaque source cluster ID")
    frame = {
        key: sum(
            f"{row['stratum']}-{'code' if int(row['has_code']) else 'nocode'}" == key
            for row in rows
        )
        for key in EXPECTED_FRAMES[corpus]
    }
    if frame != EXPECTED_FRAMES[corpus] or manifest.get("frame_counts") != frame:
        raise RunSafetyError(f"{corpus} stratum frame changed")
    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if manifest.get("window_ids_sha256") != ids_hash:
        raise RunSafetyError(f"{corpus} window-ID hash changed")
    for row in rows:
        if any(key in row for key in ("conversation_id", "conversation_hash", "source_row_index")):
            raise RunSafetyError(f"{corpus} final windows expose upstream identifiers")
        for condition in ("O", "A", "L2", "E"):
            text = row.get(f"text_{condition}")
            if not isinstance(text, str) or not text:
                raise RunSafetyError(f"{corpus}/{row['window_id']} lacks {condition}")
            if row.get(f"chars_{condition}") != len(text):
                raise RunSafetyError(f"{corpus}/{row['window_id']} has wrong {condition} length")
        if ladder.rung_E(row["text_O"]) != row["text_E"]:
            raise RunSafetyError(f"{corpus}/{row['window_id']} E differs from pinned ladder")
        if row["text_O"] != next(
            value["text_O"] for value in base_rows if value["window_id"] == row["window_id"]
        ) or row["text_E"] != next(
            value["text_E"] for value in base_rows if value["window_id"] == row["window_id"]
        ):
            raise RunSafetyError(f"{corpus} condition build changed O/E source text")
    expected_digests = {
        condition: condition_digest(rows, f"text_{condition}")
        for condition in ("O", "A", "L2", "E")
    }
    if manifest.get("condition_text_sha256") != expected_digests:
        raise RunSafetyError(f"{corpus} condition hashes changed")
    if corpus == "WC":
        if (
            source.get("revision") != WILDCHAT_REVISION
            or source.get("shard", {}).get("sha256") != WILDCHAT_SHARD_SHA
            or source.get("rows_scanned") != 6000
            or source.get("selection_seed") != 7
            or source.get("per_cell") != 40
            or len(source.get("private_window_provenance", [])) != expected_n
        ):
            raise RunSafetyError("WildChat immutable snapshot/builder parameters changed")


def validate_tabby(model_root: Path) -> None:
    runtime = HERE / "r1" / "private_inputs" / "runtime"
    config_path = runtime / "tabby.yml"
    manifest_path = runtime / "tabby_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise RunSafetyError("private tabbyAPI config/manifest is missing")
    manifest = load_json(manifest_path)
    template_path = HERE / "tabby_p2r1_config.template.yml"
    expected = template_path.read_text().replace("__MODEL_DIR__", str(model_root.resolve())).replace(
        "__PORT__", "18090"
    )
    if config_path.read_text() != expected:
        raise RunSafetyError("tabbyAPI config is not the exact rendered security template")
    exact = {
        "schema": "paper2-r1-tabby-config-v1",
        "template_sha256": sha256_file(template_path),
        "config_sha256": sha256_file(config_path),
        "config": str(config_path.resolve()),
        "model_dir": str(model_root.resolve()),
        "model_name": "qwen38-exl3-2.0",
        "artifact": str((model_root / "qwen38-exl3-2.0").resolve()),
        "port": 18090,
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
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise RunSafetyError(f"tabbyAPI runtime manifest changed: {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", action="store_true")
    parser.add_argument("--tabby-config", action="store_true")
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    if not args.inputs and not args.tabby_config:
        raise RunSafetyError("select --inputs and/or --tabby-config")
    if args.inputs:
        validate_corpus("TSI", HERE / "r1" / "private_inputs" / "tsi")
        validate_corpus("WC", HERE / "r1" / "private_inputs" / "wildchat")
    if args.tabby_config:
        if args.model_root is None:
            raise RunSafetyError("--tabby-config requires --model-root")
        validate_tabby(args.model_root)
    print("P2R1 frozen input/config validation passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 INPUT VALIDATION ABORTED: {exc}") from exc
