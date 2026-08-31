#!/usr/bin/env python3
"""Fail-closed lifecycle checks for the public llama.cpp replication runner."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SOCKET_LINK = re.compile(r"socket:\[(\d+)\]")
SHA256 = re.compile(r"[0-9a-f]{64}")
P4R1_DATASET_SHA256 = "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37"
P4R1_SEED_BASE = 4_200_000
LLAMA_CPP_SOURCE_REVISION = "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3"
LLAMA_CPP_BUILD_ID = "b1-035e227"
EXPECTED_CUDA_RUNTIME_LIBRARIES = {
    "libcudart.so.12": {
        "resolved_filename": "libcudart.so.12.6.77",
        "bytes": 716128,
        "sha256": "095296727cfb5f51e5b3ae69a3cb88cbd7b6592f314ace0fdfc4e1d758c2aa85",
    },
    "libcublas.so.12": {
        "resolved_filename": "libcublas.so.12.6.4.1",
        "bytes": 108244960,
        "sha256": "d9343511e1d2ed51e4d65fa94a25cb8f73b0ebc5ae5487da794e18e6950c98dc",
    },
    "libcublasLt.so.12": {
        "resolved_filename": "libcublasLt.so.12.6.4.1",
        "bytes": 491106832,
        "sha256": "a4ddaed1410acc604815f0e84cdd6c4a70c61cfd2b95dc96360f0799169f0374",
    },
}
P4R1_EXPECTED_IDS = tuple(
    f"gsm8k_{index}"
    for index in sorted(random.Random(42).sample(range(1319), 100))
)
NUMBER = re.compile(r"[-+]?\$?[\d][\d,]*(?:\.\d+)?")
PRIVATE_ROW_KEYS = frozenset(
    {
        "id", "label", "correct", "wall_s", "identity_guard_wall_s",
        "completion_tokens", "thinking_tokens_estimate", "truncated",
        "raw_response", "scoring_content", "parsed_answer_number",
        "expected_answer_number", "api_error", "chat_template_retry_error",
        "chat_template_kwargs_dropped",
        "model_id", "model_manifest_sha256", "startup_identity_sha256",
        "sampler_seed", "dataset_sha256", "harness_sha256",
        "server_guard_sha256", "sampler_contract_sha256",
    }
)
P4R1_MATRIX = {
    "P4R1-MIXTRAL-IQ1M": ("Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf", "7e8d75c23bd4de55582ac91084a0575f081c1de374b693f5ea7ffa6fa66a2e3c", "29", 4096),
    "P4R1-MIXTRAL-IQ2XXS": ("Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf", "db62b07756f50d2f046b0a00dc594f1aa73dba82ebdc182d9f55c96474655f03", "26", 4096),
    "P4R1-MIXTRAL-Q4KM": ("Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf", "7f2f98f301a74b645d0a4c2bd64c98e9811f2dfa26dbae4787c280eb1183632c", "12", 4096),
    "P4R1-QWEN3MOE-IQ1M": ("Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf", "d527a854db2a1582a3ce746a17b1f42d860334ece18d385ede9e2e395058b39e", "auto", 4096),
    "P4R1-QWEN3MOE-IQ2XXS": ("Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf", "aa0b04ac05f71aa94b542b074500b3de956608365e74f0e1d97ae0d6a6380269", "auto", 4096),
    "P4R1-QWEN3MOE-Q4KM": ("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", "6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0", "auto", 4096),
}
P4R1_BYTES = {
    "P4R1-MIXTRAL-IQ1M": 10847178368,
    "P4R1-MIXTRAL-IQ2XXS": 12556357248,
    "P4R1-MIXTRAL-Q4KM": 28448468608,
    "P4R1-QWEN3MOE-IQ1M": 9685733792,
    "P4R1-QWEN3MOE-IQ2XXS": 10341814688,
    "P4R1-QWEN3MOE-Q4KM": 18556686752,
}
EXPECTED_HOST = {
    "cpu_model": "Intel(R) Core(TM) i9-14900HX",
    "logical_cpus": 32,
    "physical_cores": 24,
    "sockets": 1,
    "mem_total_kib": 65445316,
    "kernel": "Linux 7.0.0-29-generic x86_64",
    "os_pretty_name": "Ubuntu 24.04.3 LTS",
    "gpu_name": "NVIDIA GeForce RTX 4080 Laptop GPU",
    "gpu_memory_mib": 12282,
    "gpu_power_limit_w": 80.0,
    "driver_version": "580.167.08",
    "filesystem": "ext4",
    "storage_model": "WD PC SN740 SDDPNQE-2T00-1102",
    "storage_transport": "nvme",
    "storage_rotational": 0,
}


def numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def require_finite_json(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{context} contains a non-finite number")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise RuntimeError(f"{context} contains a non-string object key")
        for child in value.values():
            require_finite_json(child, context)
    elif isinstance(value, list):
        for child in value:
            require_finite_json(child, context)


def validate_error_telemetry(value: Any, context: str) -> None:
    keys = {
        "schema", "exception_type", "message", "http_status", "http_reason",
        "url", "headers", "body_bytes", "body_sha256", "body_base64",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError(f"{context} has an invalid exact-telemetry schema")
    if (
        value.get("schema") != "paper4-p4r1-api-error-v1"
        or not isinstance(value.get("exception_type"), str)
        or not value["exception_type"]
        or not isinstance(value.get("message"), str)
        or not (
            value.get("http_status") is None
            or isinstance(value["http_status"], int)
            and not isinstance(value["http_status"], bool)
            and 100 <= value["http_status"] <= 599
        )
        or not (value.get("http_reason") is None or isinstance(value["http_reason"], str))
        or not (value.get("url") is None or isinstance(value["url"], str))
        or not isinstance(value.get("headers"), list)
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
            for item in value["headers"]
        )
        or not isinstance(value.get("body_bytes"), int)
        or isinstance(value["body_bytes"], bool)
        or not 0 <= value["body_bytes"] <= 1024 * 1024
        or not isinstance(value.get("body_sha256"), str)
        or SHA256.fullmatch(value["body_sha256"]) is None
        or not isinstance(value.get("body_base64"), str)
    ):
        raise RuntimeError(f"{context} contains an invalid exact-telemetry value")
    try:
        body = base64.b64decode(value["body_base64"], validate=True)
    except (ValueError, binascii.Error) as error:
        raise RuntimeError(f"{context} contains invalid base64") from error
    if len(body) != value["body_bytes"] or hashlib.sha256(body).hexdigest() != value["body_sha256"]:
        raise RuntimeError(f"{context} body length/hash differs from preserved bytes")
    if value["http_status"] is None and (
        value["http_reason"] is not None
        or value["url"] is not None
        or value["headers"]
        or body
    ):
        raise RuntimeError(f"{context} non-HTTP exception contains HTTP-only fields")


def to_float(token: str) -> float | None:
    try:
        return float(token.strip().replace("$", "").replace(",", "").rstrip("."))
    except ValueError:
        return None


def parse_gsm8k_answer(content: str) -> float | None:
    markers = list(re.finditer(r"[Aa]nswer\s*:", content))
    candidate: float | None = None
    if markers:
        match = NUMBER.search(content[markers[-1].end() :][:120])
        if match:
            candidate = to_float(match.group(0))
    if candidate is None:
        lines = [line for line in content.strip().splitlines() if line.strip()]
        if lines:
            numbers = NUMBER.findall(lines[-1])
            if numbers:
                candidate = to_float(numbers[-1])
    return candidate


def score_gsm8k(content: str, expected: float) -> bool:
    candidate = parse_gsm8k_answer(content)
    if candidate is None:
        return False
    return abs(candidate - expected) <= 1e-4 or (
        expected != 0 and abs(candidate - expected) / abs(expected) <= 1e-4
    )


def response_scoring_content(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("successful row has no first raw response choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content") or "", str):
        raise RuntimeError("successful row has no textual raw response message")
    content = message.get("content") or ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    return content.strip(), choices[0]


def thinking_tokens(message: dict[str, Any]) -> int:
    reasoning = message.get("reasoning_content")
    if not reasoning:
        content = message.get("content") or ""
        reasoning = content.split("</think>", 1)[0] if "</think>" in content else ""
    return int(len(reasoning.split()) * 1.3) if reasoning else 0


def validate_result_row(
    row: dict[str, Any], alias: str, expected_gold: float,
    dataset_sha256: str, harness_sha256: str, guard_sha256: str,
    startup_identity_sha256: str, model_manifest_sha256: str,
    sampler_contract_sha256: str,
) -> None:
    if set(row) != PRIVATE_ROW_KEYS:
        raise RuntimeError("cell row schema differs from the exact P4R1 private schema")
    match = re.fullmatch(r"gsm8k_(\d+)", row["id"]) if isinstance(row["id"], str) else None
    if (
        row["label"] != alias
        or row["model_id"] != alias
        or not isinstance(row["correct"], bool)
        or not numeric(row["wall_s"]) or row["wall_s"] < 0
        or not numeric(row["identity_guard_wall_s"]) or row["identity_guard_wall_s"] < 0
        or not (
            row["completion_tokens"] is None
            or isinstance(row["completion_tokens"], int)
            and not isinstance(row["completion_tokens"], bool)
            and row["completion_tokens"] >= 0
        )
        or not isinstance(row["thinking_tokens_estimate"], int)
        or isinstance(row["thinking_tokens_estimate"], bool)
        or row["thinking_tokens_estimate"] < 0
        or not isinstance(row["truncated"], bool)
        or not isinstance(row["scoring_content"], str)
        or not (
            row["parsed_answer_number"] is None or numeric(row["parsed_answer_number"])
        )
        or not numeric(row["expected_answer_number"])
        or not isinstance(row["chat_template_kwargs_dropped"], bool)
        or not (row["api_error"] is None or isinstance(row["api_error"], dict))
        or not (
            row["chat_template_retry_error"] is None
            or isinstance(row["chat_template_retry_error"], dict)
        )
        or not isinstance(row["sampler_seed"], int)
        or isinstance(row["sampler_seed"], bool)
        or match is None
    ):
        raise RuntimeError("cell row has an invalid value or type")
    if any(
        not isinstance(row[key], str) or SHA256.fullmatch(row[key]) is None
        for key in (
            "model_manifest_sha256", "startup_identity_sha256", "dataset_sha256",
            "harness_sha256", "server_guard_sha256", "sampler_contract_sha256",
        )
    ):
        raise RuntimeError("cell row has a malformed SHA-256 field")
    if (
        row["model_manifest_sha256"] != model_manifest_sha256
        or row["startup_identity_sha256"] != startup_identity_sha256
        or row["dataset_sha256"] != dataset_sha256
        or row["harness_sha256"] != harness_sha256
        or row["server_guard_sha256"] != guard_sha256
        or row["sampler_contract_sha256"] != sampler_contract_sha256
        or row["sampler_seed"] != P4R1_SEED_BASE + int(match.group(1))
        or float(row["expected_answer_number"]) != expected_gold
    ):
        raise RuntimeError("cell row identity, source, sampler, or pinned gold differs")
    if row["api_error"] is None:
        if row["chat_template_retry_error"] is not None:
            validate_error_telemetry(
                row["chat_template_retry_error"], "chat-template retry error"
            )
        if row["chat_template_kwargs_dropped"] != (
            row["chat_template_retry_error"] is not None
        ):
            raise RuntimeError("chat-template retry flag differs from telemetry")
        raw_response = row["raw_response"]
        if not isinstance(raw_response, dict) or raw_response.get("model") != alias:
            raise RuntimeError("successful cell row lacks an alias-bound raw response")
        require_finite_json(raw_response, "raw response")
        content, choice = response_scoring_content(raw_response)
        message = choice["message"]
        usage = raw_response.get("usage") or {}
        if not isinstance(usage, dict):
            raise RuntimeError("successful cell row raw usage is not an object")
        if (
            content != row["scoring_content"]
            or parse_gsm8k_answer(content) != row["parsed_answer_number"]
            or (choice.get("finish_reason") == "length") != row["truncated"]
            or usage.get("completion_tokens") != row["completion_tokens"]
            or thinking_tokens(message) != row["thinking_tokens_estimate"]
            or score_gsm8k(content, expected_gold) != row["correct"]
        ):
            raise RuntimeError("cell row fields or score differ from raw response and pinned gold")
    else:
        validate_error_telemetry(row["api_error"], "API error")
        if row["chat_template_retry_error"] is not None:
            validate_error_telemetry(
                row["chat_template_retry_error"], "chat-template retry error"
            )
        if (
            row["chat_template_kwargs_dropped"]
            != (row["chat_template_retry_error"] is not None)
            or row["raw_response"] is not None
            and not isinstance(row["raw_response"], dict)
            or row["scoring_content"] != ""
            or row["parsed_answer_number"] is not None
            or row["correct"]
        ):
            raise RuntimeError("failed request row is not an exact incorrect/error record")
        if isinstance(row["raw_response"], dict):
            require_finite_json(row["raw_response"], "failed raw response")
            if row["raw_response"].get("model") != alias:
                raise RuntimeError("failed-row raw response model identity differs")


def process_stat(pid: int) -> tuple[str, str]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError("malformed process stat record")
    fields = raw[closing + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError("incomplete process stat record")
    return fields[0], fields[19]


def process_start_time(pid: int) -> str:
    return process_stat(pid)[1]


def process_cmdline(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part]


def process_socket_inodes(pid: int) -> set[str]:
    result: set[str] = set()
    for descriptor in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        match = SOCKET_LINK.fullmatch(target)
        if match:
            result.add(match.group(1))
    return result


def listener_inodes(port: int) -> set[str]:
    result: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.is_file():
            continue
        for line in table.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                result.add(fields[9])
    return result


def option_values(argv: list[str], options: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for index, argument in enumerate(argv):
        if argument in options:
            if index + 1 >= len(argv):
                raise RuntimeError(f"process argv ends after {argument}")
            values.append(argv[index + 1])
            continue
        for option in options:
            prefix = option + "="
            if argument.startswith(prefix):
                values.append(argument[len(prefix) :])
    return values


def launcher_matches(pid: int, launcher: Path, argv: list[str]) -> bool:
    expected = launcher.resolve()
    try:
        executable = Path(f"/proc/{pid}/exe").resolve(strict=True)
    except (FileNotFoundError, PermissionError, OSError):
        return False
    if executable == expected:
        return True
    # A script launcher appears in cmdline while /proc/PID/exe names its
    # interpreter. This branch exists for the isolated lifecycle regression.
    return any(Path(argument).resolve() == expected for argument in argv if argument.startswith("/"))


def same_process(pid: int, start_time: str) -> bool:
    try:
        os.kill(pid, 0)
        state, observed_start = process_stat(pid)
        return state != "Z" and observed_start == start_time
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, RuntimeError):
        return False


def verify_process(args: argparse.Namespace) -> None:
    if not same_process(args.pid, args.start_time):
        raise RuntimeError("launched process is no longer alive with the recorded identity")
    argv = process_cmdline(args.pid)
    if not launcher_matches(args.pid, args.launcher, argv):
        raise RuntimeError("listener process does not match the launched executable")
    expected_options = (
        (("-m", "--model"), str(args.model.resolve()), "model"),
        (("--alias",), args.alias, "alias"),
        (("--host",), args.host, "host"),
        (("--port",), str(args.port), "port"),
        (("-c", "--ctx-size"), str(args.context), "context"),
        (("-lv", "--verbosity", "--log-verbosity"), "4", "log verbosity"),
    )
    for options, expected, description in expected_options:
        if option_values(argv, options) != [expected]:
            raise RuntimeError(
                f"listener process does not retain exactly one {description} value {expected!r}"
            )
    ngl_values = option_values(argv, ("-ngl", "--n-gpu-layers", "--gpu-layers"))
    if args.ngl == "auto":
        if ngl_values:
            raise RuntimeError("automatic-fit cell contains an explicit GPU-layer override")
    elif ngl_values != [args.ngl]:
        raise RuntimeError(
            f"listener process does not retain exactly one GPU-layer value {args.ngl!r}"
        )
    if not (process_socket_inodes(args.pid) & listener_inodes(args.port)):
        raise RuntimeError("launched PID does not own the requested listening port")


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=2) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}")
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise RuntimeError("server identity response is unexpectedly large")
    return json.loads(raw)


def verify_endpoints(base_url: str, expected_alias: str) -> None:
    fetch_json(base_url.rstrip("/") + "/health")
    models = fetch_json(base_url.rstrip("/") + "/v1/models")
    data = models.get("data") if isinstance(models, dict) else None
    if not isinstance(data, list) or len(data) != 1:
        raise RuntimeError("/v1/models did not return exactly one model")
    model = data[0]
    if not isinstance(model, dict) or model.get("id") != expected_alias:
        raise RuntimeError("/v1/models did not return the exact launched alias")


def prove_port_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        probe.listen(1)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model_entry(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    raw = args.manifest.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if not SHA256.fullmatch(args.manifest_sha256):
        raise RuntimeError("model-manifest SHA-256 is malformed")
    if actual_manifest_sha256 != args.manifest_sha256:
        raise RuntimeError(
            f"model-manifest SHA-256 mismatch: expected {args.manifest_sha256}, got {actual_manifest_sha256}"
        )
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {"schema", "content_verification", "artifact_count", "total_bytes", "artifacts"}
        or manifest.get("schema") != "paper4-validation-model-manifest-v1"
        or manifest.get("content_verification")
        != "full SHA-256 completed before the campaign; live cells use stat-only checks"
        or manifest.get("artifact_count") != 6
        or manifest.get("total_bytes") != sum(P4R1_BYTES.values())
    ):
        raise RuntimeError("model manifest has an unknown schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 6:
        raise RuntimeError("model manifest must contain exactly six artifacts")
    aliases: list[str] = []
    paths: list[str] = []
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {
            "alias", "family", "filename", "path", "bytes", "sha256", "stat"
        }:
            raise RuntimeError("model manifest artifact has an unknown schema")
        alias = item.get("alias")
        expected = P4R1_MATRIX.get(alias)
        stat_record = item.get("stat")
        if (
            expected is None
            or (item.get("filename"), item.get("sha256")) != expected[:2]
            or item.get("bytes") != P4R1_BYTES[alias]
            or not isinstance(item.get("family"), str)
            or not isinstance(item.get("path"), str)
            or not Path(item["path"]).is_absolute()
            or Path(item["path"]).name != item["filename"]
            or not isinstance(stat_record, dict)
            or set(stat_record) != {"device", "inode", "mtime_ns"}
            or any(
                not isinstance(stat_record[key], int)
                or isinstance(stat_record[key], bool)
                or stat_record[key] < 0
                for key in stat_record
            )
        ):
            raise RuntimeError("model manifest artifact identity/stat differs from P4R1")
        aliases.append(alias)
        paths.append(item["path"])
    if set(aliases) != set(P4R1_MATRIX) or len(set(aliases)) != 6 or len(set(paths)) != 6:
        raise RuntimeError("model manifest contains duplicate, missing, or extra artifacts")
    matches = [item for item in artifacts if isinstance(item, dict) and item.get("alias") == args.alias]
    if len(matches) != 1:
        raise RuntimeError(f"model manifest does not contain exactly one {args.alias!r} entry")
    entry = matches[0]
    expected = P4R1_MATRIX.get(args.alias)
    if expected is not None and (entry.get("filename"), entry.get("sha256")) != expected[:2]:
        raise RuntimeError("model-manifest artifact identity differs from the P4R1 matrix")
    expected_path = entry.get("path")
    if not isinstance(expected_path, str) or Path(expected_path).resolve() != args.model.resolve():
        raise RuntimeError("live model path differs from the pre-campaign identity manifest")
    if not isinstance(entry.get("sha256"), str) or not SHA256.fullmatch(entry["sha256"]):
        raise RuntimeError("model manifest entry has a malformed content SHA-256")
    stat = args.model.stat()
    expected_stat = entry.get("stat")
    if not isinstance(expected_stat, dict):
        raise RuntimeError("model manifest entry has no stat identity")
    observed = {"device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size != entry.get("bytes") or observed != expected_stat:
        raise RuntimeError("live model size/stat identity differs from the pre-campaign manifest")
    return entry, actual_manifest_sha256


def verify_model_entry(args: argparse.Namespace) -> None:
    entry, manifest_sha256 = load_model_entry(args)
    if args.output is not None:
        write_json_exclusive(
            args.output,
            {
                "schema": "paper4-validation-live-model-identity-v1",
                "alias": args.alias,
                "model_manifest_sha256": manifest_sha256,
                "model": entry,
                "verification": "manifest SHA plus exact path, size, device, inode, and mtime; weight contents were not reread",
            },
        )


def verify_manifest_content(args: argparse.Namespace) -> None:
    artifacts = []
    manifest_sha256: str | None = None
    for alias, (filename, expected_sha256, _ngl, _context) in P4R1_MATRIX.items():
        raw_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        matches = [
            item for item in raw_manifest.get("artifacts", [])
            if isinstance(item, dict) and item.get("alias") == alias
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
            raise RuntimeError(f"model manifest lacks one exact post-campaign path for {alias}")
        model = Path(matches[0]["path"])
        entry, observed_manifest_sha256 = load_model_entry(
            argparse.Namespace(
                manifest=args.manifest,
                manifest_sha256=args.manifest_sha256,
                alias=alias,
                model=model,
            )
        )
        if manifest_sha256 is None:
            manifest_sha256 = observed_manifest_sha256
        elif manifest_sha256 != observed_manifest_sha256:
            raise RuntimeError("model manifest changed during post-campaign verification")
        actual_sha256 = sha256_file(model)
        post = model.stat()
        post_stat = {
            "device": post.st_dev, "inode": post.st_ino, "mtime_ns": post.st_mtime_ns,
        }
        if (
            actual_sha256 != expected_sha256
            or actual_sha256 != entry["sha256"]
            or post.st_size != entry["bytes"]
            or post_stat != entry["stat"]
            or model.name != filename
        ):
            raise RuntimeError(f"{alias}: artifact bytes/stat changed during the campaign")
        artifacts.append(
            {
                "alias": alias,
                "filename": filename,
                "bytes": post.st_size,
                "sha256": actual_sha256,
                "pre_campaign_stat": entry["stat"],
                "post_campaign_stat": post_stat,
            }
        )
    if manifest_sha256 is None:
        raise RuntimeError("post-campaign verification did not inspect any artifacts")
    toolchain_raw = args.toolchain_receipt.read_bytes()
    toolchain_sha256 = hashlib.sha256(toolchain_raw).hexdigest()
    toolchain = json.loads(toolchain_raw)
    executable = toolchain.get("executable") if isinstance(toolchain, dict) else None
    cmake_cache = toolchain.get("observed_cmake_cache") if isinstance(toolchain, dict) else None
    if (
        not isinstance(executable, dict)
        or not isinstance(cmake_cache, dict)
        or not isinstance(executable.get("path"), str)
        or not isinstance(executable.get("sha256"), str)
        or not isinstance(cmake_cache.get("path"), str)
        or not isinstance(cmake_cache.get("sha256"), str)
    ):
        raise RuntimeError("pre-campaign toolchain receipt is malformed")
    launcher = Path(executable["path"])
    cache = Path(cmake_cache["path"])
    if (
        launcher.is_symlink()
        or not launcher.is_file()
        or sha256_file(launcher) != executable["sha256"]
        or cache.is_symlink()
        or not cache.is_file()
        or sha256_file(cache) != cmake_cache["sha256"]
    ):
        raise RuntimeError("executable or CMake cache changed during the campaign")
    cache_values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key, _type = key_and_type.split(":", 1)
        cache_values[key] = value
    post_cuda_runtime = cuda_runtime_resolution(launcher.resolve(strict=True), cache_values)
    pre_cuda_runtime = toolchain.get("observed_cuda_runtime")
    if cuda_runtime_stable_identity(post_cuda_runtime) != cuda_runtime_stable_identity(
        pre_cuda_runtime
    ):
        raise RuntimeError("loader-resolved CUDA runtime changed during the campaign")
    write_json_exclusive(
        args.output,
        {
            "schema": "paper4-p4r1-post-campaign-model-verification-v1",
            "model_manifest_sha256": manifest_sha256,
            "toolchain_receipt_sha256": toolchain_sha256,
            "artifact_count": 6,
            "total_bytes": sum(P4R1_BYTES.values()),
            "artifacts": artifacts,
            "post_campaign_cuda_runtime": post_cuda_runtime,
            "verification": "all six artifact contents and the loader-resolved CUDA runtime were fully SHA-256-verified after all timed cells and shutdowns; model size/device/inode/mtime and stable CUDA runtime identity remained equal to their pre-campaign receipts",
        },
    )


def validate_post_model_verification(
    receipt: dict[str, Any], manifest_sha256: str, manifest: dict[str, Any],
    toolchain_sha256: str, toolchain: dict[str, Any],
) -> None:
    artifacts = receipt.get("artifacts")
    if (
        set(receipt)
        != {
            "schema", "model_manifest_sha256", "artifact_count", "total_bytes",
            "artifacts", "toolchain_receipt_sha256", "post_campaign_cuda_runtime",
            "verification",
        }
        or receipt.get("schema")
        != "paper4-p4r1-post-campaign-model-verification-v1"
        or receipt.get("model_manifest_sha256") != manifest_sha256
        or receipt.get("toolchain_receipt_sha256") != toolchain_sha256
        or receipt.get("artifact_count") != 6
        or receipt.get("total_bytes") != sum(P4R1_BYTES.values())
        or not isinstance(artifacts, list)
        or len(artifacts) != 6
    ):
        raise RuntimeError("post-campaign model verification has the wrong schema/count")
    if cuda_runtime_stable_identity(
        receipt.get("post_campaign_cuda_runtime")
    ) != cuda_runtime_stable_identity(toolchain.get("observed_cuda_runtime")):
        raise RuntimeError("post-campaign CUDA runtime differs from the pre-campaign toolchain")
    by_alias = {
        item.get("alias"): item for item in artifacts if isinstance(item, dict)
    }
    if len(by_alias) != 6 or set(by_alias) != set(P4R1_MATRIX):
        raise RuntimeError("post-campaign model verification aliases differ from P4R1")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, list):
        raise RuntimeError("model manifest has no artifact list for post-campaign verification")
    manifest_by_alias = {
        item.get("alias"): item
        for item in manifest_artifacts if isinstance(item, dict)
    }
    if len(manifest_by_alias) != 6 or set(manifest_by_alias) != set(P4R1_MATRIX):
        raise RuntimeError("model manifest aliases differ from post-campaign verification")
    for alias, (filename, expected_sha256, _ngl, _context) in P4R1_MATRIX.items():
        item = by_alias[alias]
        if (
            set(item)
            != {
                "alias", "filename", "bytes", "sha256", "pre_campaign_stat",
                "post_campaign_stat",
            }
            or item.get("filename") != filename
            or item.get("bytes") != P4R1_BYTES[alias]
            or item.get("sha256") != expected_sha256
            or item.get("pre_campaign_stat") != manifest_by_alias[alias].get("stat")
            or item.get("post_campaign_stat") != manifest_by_alias[alias].get("stat")
            or not isinstance(item.get("pre_campaign_stat"), dict)
            or set(item["pre_campaign_stat"]) != {"device", "inode", "mtime_ns"}
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in item["pre_campaign_stat"].values()
            )
        ):
            raise RuntimeError(f"{alias}: malformed post-campaign content verification")


def record_ready(args: argparse.Namespace) -> None:
    expected = P4R1_MATRIX.get(args.alias)
    if expected is None or (args.ngl, args.context) != (expected[2], expected[3]):
        raise RuntimeError("requested placement/context differs from the exact P4R1 matrix")
    verify_process(args)
    verify_endpoints(args.base_url, args.alias)
    model_identity = json.loads(args.model_identity.read_text(encoding="utf-8"))
    if model_identity.get("alias") != args.alias:
        raise RuntimeError("live model-identity receipt alias differs")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required to record observed GPU placement")
    snapshot_result = subprocess.run(
        [nvidia_smi, "--query-compute-apps=pid,used_gpu_memory,gpu_uuid", "--format=csv,noheader,nounits"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=10, check=False,
    )
    if snapshot_result.returncode != 0:
        raise RuntimeError("nvidia-smi placement snapshot failed")
    matching_gpu_rows = []
    for line in snapshot_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0] == str(args.pid):
            matching_gpu_rows.append(
                {"pid": args.pid, "used_gpu_memory_mib": int(parts[1]), "gpu_uuid": parts[2]}
            )
    if len(matching_gpu_rows) != 1:
        raise RuntimeError("nvidia-smi did not report exactly one GPU placement row for launched PID")
    record = {
        "schema": "paper4-validation-server-identity-v1",
        "phase": args.phase,
        "alias": args.alias,
        "pid": args.pid,
        "process_start_time_ticks": args.start_time,
        "launcher": {
            "path": str(args.launcher.resolve()),
            "sha256": sha256_file(args.launcher),
        },
        "model_identity_sha256": sha256_file(args.model_identity),
        "model_manifest_sha256": model_identity["model_manifest_sha256"],
        "endpoint": {"host": args.host, "port": args.port, "base_url": args.base_url, "model_id": args.alias},
        "request": {"context_tokens": args.context, "gpu_layers": args.ngl},
        "observed_gpu_process": matching_gpu_rows[0],
        "verification": "PID/start-time, executable, exact argv, listener ownership, health, and sole /v1/models ID",
    }
    write_json_exclusive(args.output, record)


def cuda_runtime_resolution(launcher: Path, cache_values: dict[str, str]) -> dict[str, Any]:
    configured_raw = cache_values.get("CUDAToolkit_ROOT", "")
    if not configured_raw or not Path(configured_raw).is_absolute():
        raise RuntimeError("CMakeCache has no absolute CUDAToolkit_ROOT")
    configured_root = Path(configured_raw).resolve(strict=True)
    if not configured_root.is_dir():
        raise RuntimeError("configured CUDA toolkit root is not a directory")
    configured_lib64 = configured_root / "lib64"
    resolved_lib64 = configured_lib64.resolve(strict=True)
    if not resolved_lib64.is_dir() or not resolved_lib64.is_relative_to(configured_root):
        raise RuntimeError("configured CUDA toolkit lib64 is missing or escapes its toolkit")
    search_entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    if not search_entries:
        raise RuntimeError("LD_LIBRARY_PATH does not select the frozen CUDA runtime")
    try:
        first_search_entry = Path(search_entries[0]).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("first LD_LIBRARY_PATH entry is missing") from error
    if first_search_entry != resolved_lib64:
        raise RuntimeError("frozen CUDA toolkit lib64 must be first in LD_LIBRARY_PATH")

    ldd_raw = shutil.which("ldd")
    if ldd_raw is None:
        raise RuntimeError("ldd is required for CUDA runtime provenance")
    ldd = Path(ldd_raw).resolve(strict=True)
    result = subprocess.run(
        [str(ldd), str(launcher)], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=30, check=False,
    )
    if (
        result.returncode != 0
        or len(result.stdout.encode("utf-8")) > 1024 * 1024
        or len(result.stderr.encode("utf-8")) > 65536
        or "not found" in result.stdout
    ):
        raise RuntimeError("ldd failed or reported an unresolved dependency")
    observed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(", line)
        if match and match.group(1) in EXPECTED_CUDA_RUNTIME_LIBRARIES:
            soname, path = match.groups()
            if soname in observed:
                raise RuntimeError(f"ldd reported duplicate CUDA dependency {soname}")
            observed[soname] = path
    if set(observed) != set(EXPECTED_CUDA_RUNTIME_LIBRARIES):
        raise RuntimeError("ldd did not resolve the three frozen CUDA runtime libraries")

    libraries = []
    for soname, expected in EXPECTED_CUDA_RUNTIME_LIBRARIES.items():
        loader_path = Path(observed[soname])
        if not loader_path.is_absolute():
            raise RuntimeError(f"ldd returned a non-absolute path for {soname}")
        resolved = loader_path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(configured_root):
            raise RuntimeError(f"{soname} resolved outside the configured CUDA toolkit")
        stat_before = resolved.stat()
        digest = sha256_file(resolved)
        stat_after = resolved.stat()
        if (
            stat_before != stat_after
            or resolved.name != expected["resolved_filename"]
            or stat_after.st_size != expected["bytes"]
            or digest != expected["sha256"]
        ):
            raise RuntimeError(f"{soname} differs from the frozen CUDA 12.6 runtime")
        libraries.append(
            {
                "soname": soname,
                "loader_path": str(loader_path),
                "resolved_path": str(resolved),
                "resolved_filename": resolved.name,
                "bytes": stat_after.st_size,
                "sha256": digest,
            }
        )
    configured_cudart = Path(cache_values.get("CUDA_CUDART", ""))
    if not configured_cudart.is_absolute() or configured_cudart.resolve(strict=True) != Path(
        next(item["resolved_path"] for item in libraries if item["soname"] == "libcudart.so.12")
    ):
        raise RuntimeError("CMake CUDA_CUDART and loader-resolved libcudart differ")
    return {
        "schema": "paper4-p4r1-cuda-runtime-resolution-v1",
        "configured_cudatoolkit_root": str(configured_root),
        "configured_lib64": str(configured_lib64),
        "resolved_lib64": str(resolved_lib64),
        "ld_library_path_first": str(Path(search_entries[0])),
        "loader_probe": {
            "tool_path": str(ldd), "tool_sha256": sha256_file(ldd),
            "returncode": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        },
        "libraries": libraries,
        "verification": "ldd resolved the exact hashed CUDA 12.6 runtime libraries beneath the CMake-configured toolkit root",
    }


def cuda_runtime_stable_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema", "configured_cudatoolkit_root", "configured_lib64",
        "resolved_lib64", "ld_library_path_first", "loader_probe",
        "libraries", "verification",
    }:
        raise RuntimeError("CUDA runtime receipt has an unknown schema")
    if (
        value.get("schema") != "paper4-p4r1-cuda-runtime-resolution-v1"
        or value.get("verification")
        != "ldd resolved the exact hashed CUDA 12.6 runtime libraries beneath the CMake-configured toolkit root"
    ):
        raise RuntimeError("CUDA runtime receipt has an unknown identity statement")
    toolkit_root = Path(str(value.get("configured_cudatoolkit_root")))
    configured_lib64 = Path(str(value.get("configured_lib64")))
    resolved_lib64 = Path(str(value.get("resolved_lib64")))
    if (
        not toolkit_root.is_absolute()
        or not configured_lib64.is_absolute()
        or not resolved_lib64.is_absolute()
        or not configured_lib64.is_relative_to(toolkit_root)
        or not resolved_lib64.is_relative_to(toolkit_root)
        or value.get("ld_library_path_first") != str(configured_lib64)
    ):
        raise RuntimeError("CUDA runtime receipt paths are malformed")
    probe = value.get("loader_probe")
    if (
        not isinstance(probe, dict)
        or set(probe)
        != {"tool_path", "tool_sha256", "returncode", "stdout_sha256", "stderr_sha256"}
        or not Path(str(probe.get("tool_path"))).is_absolute()
        or probe.get("returncode") != 0
        or any(
            not isinstance(probe.get(key), str) or SHA256.fullmatch(probe[key]) is None
            for key in ("tool_sha256", "stdout_sha256", "stderr_sha256")
        )
    ):
        raise RuntimeError("CUDA loader probe is malformed")
    libraries = value.get("libraries")
    by_soname = {
        item.get("soname"): item for item in libraries if isinstance(item, dict)
    } if isinstance(libraries, list) else {}
    if len(by_soname) != 3 or set(by_soname) != set(EXPECTED_CUDA_RUNTIME_LIBRARIES):
        raise RuntimeError("CUDA runtime library inventory is incomplete or duplicated")
    for soname, expected in EXPECTED_CUDA_RUNTIME_LIBRARIES.items():
        item = by_soname[soname]
        loader_path = Path(str(item.get("loader_path")))
        resolved_path = Path(str(item.get("resolved_path")))
        if (
            set(item)
            != {"soname", "loader_path", "resolved_path", "resolved_filename", "bytes", "sha256"}
            or not loader_path.is_absolute()
            or not resolved_path.is_absolute()
            or not loader_path.is_relative_to(toolkit_root)
            or not resolved_path.is_relative_to(toolkit_root)
            or item.get("resolved_filename") != expected["resolved_filename"]
            or resolved_path.name != expected["resolved_filename"]
            or item.get("bytes") != expected["bytes"]
            or item.get("sha256") != expected["sha256"]
        ):
            raise RuntimeError(f"{soname}: CUDA runtime receipt differs from the frozen library")
    # ldd includes randomized load addresses in stdout. Everything that can
    # affect library selection is stable; only that diagnostic hash may vary.
    stable_probe = dict(probe)
    stable_probe.pop("stdout_sha256")
    return {
        "schema": value["schema"],
        "configured_cudatoolkit_root": str(toolkit_root),
        "configured_lib64": str(configured_lib64),
        "resolved_lib64": str(resolved_lib64),
        "ld_library_path_first": value["ld_library_path_first"],
        "loader_probe": stable_probe,
        "libraries": [by_soname[soname] for soname in EXPECTED_CUDA_RUNTIME_LIBRARIES],
        "verification": value["verification"],
    }


def record_toolchain(args: argparse.Namespace) -> None:
    launcher = args.launcher.resolve(strict=True)
    if args.launcher.is_symlink() or not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise RuntimeError("llama-server must be a non-symlink executable regular file")
    result = subprocess.run(
        [str(launcher), "--version"], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"llama-server --version failed with status {result.returncode}")
    if len(result.stdout) > 65536 or len(result.stderr) > 65536:
        raise RuntimeError("llama-server --version output is unexpectedly large")
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    combined = stdout + "\n" + stderr
    if "035e227" not in combined:
        raise RuntimeError("llama-server --version does not report the frozen 035e227 revision")
    checkout_result = subprocess.run(
        ["git", "-C", str(launcher.parent), "rev-parse", "--show-toplevel"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    if checkout_result.returncode != 0:
        raise RuntimeError("llama-server is not inside a Git checkout")
    checkout = Path(checkout_result.stdout.strip()).resolve(strict=True)
    if not launcher.is_relative_to(checkout) or launcher.relative_to(checkout).as_posix() != "build/bin/llama-server":
        raise RuntimeError("llama-server is not the frozen checkout's build/bin/llama-server")
    revision_result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    status_result = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    if (
        revision_result.returncode != 0
        or revision_result.stdout.strip() != LLAMA_CPP_SOURCE_REVISION
        or status_result.returncode != 0
        or status_result.stdout.strip()
    ):
        raise RuntimeError("llama.cpp checkout is not at the exact clean frozen revision")
    cache = checkout / "build/CMakeCache.txt"
    if cache.is_symlink() or not cache.is_file():
        raise RuntimeError("frozen llama.cpp build has no regular CMakeCache.txt")
    cache_values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key, _type = key_and_type.split(":", 1)
        cache_values[key] = value
    expected_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "89",
        "GGML_BLAS": "OFF",
        "GGML_CUDA": "ON",
        "GGML_NATIVE": "ON",
        "GGML_OPENMP": "ON",
        "LLAMA_CURL": "OFF",
    }
    if any(cache_values.get(key) != value for key, value in expected_cache.items()):
        raise RuntimeError("CMakeCache build type, architecture, or feature flags differ")
    configured_tool_paths = {
        "cmake": Path(cache_values.get("CMAKE_COMMAND", "")),
        "cxx": Path(cache_values.get("CMAKE_CXX_COMPILER", "")),
        "nvcc": Path(cache_values.get("CMAKE_CUDA_COMPILER", "")),
    }
    try:
        tool_paths = {name: path.resolve(strict=True) for name, path in configured_tool_paths.items()}
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError(f"CMakeCache compiler/tool path is missing: {error}") from error
    if any(not path.is_file() for path in tool_paths.values()):
        raise RuntimeError("CMakeCache compiler/tool target is not a regular file")
    tool_versions: dict[str, str] = {}
    for name, path in tool_paths.items():
        version = subprocess.run(
            [str(path), "--version"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=30, check=False,
        )
        if version.returncode != 0 or len(version.stdout.encode("utf-8")) > 65536:
            raise RuntimeError(f"{name} --version failed or returned oversized output")
        tool_versions[name] = version.stdout
    if (
        "cmake version 3.28.3" not in tool_versions["cmake"].lower()
        or "13.3.0" not in tool_versions["cxx"]
        or "12.6" not in tool_versions["nvcc"]
        or "12.6.85" not in tool_versions["nvcc"]
    ):
        raise RuntimeError("observed CMake/C++/CUDA toolkit versions differ from the frozen build")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for driver provenance")
    driver_result = subprocess.run(
        [nvidia_smi, "--query-gpu=driver_version,name", "--format=csv,noheader"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    driver_rows = [line.strip() for line in driver_result.stdout.splitlines() if line.strip()]
    if driver_result.returncode != 0 or len(driver_rows) != 1:
        raise RuntimeError("nvidia-smi did not report exactly one GPU driver row")
    driver_version, separator, gpu_name = driver_rows[0].partition(",")
    if not separator or driver_version.strip() != "580.167.08":
        raise RuntimeError("NVIDIA driver differs from frozen P4R1 version 580.167.08")
    compatibility_result = subprocess.run(
        [nvidia_smi], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, timeout=30, check=False,
    )
    compatibility_match = re.search(r"CUDA Version:\s*([0-9.]+)", compatibility_result.stdout)
    if (
        compatibility_result.returncode != 0
        or compatibility_match is None
        or compatibility_match.group(1) != "13.0"
    ):
        raise RuntimeError("driver-reported CUDA compatibility differs from frozen 13.0")
    cuda_runtime = cuda_runtime_resolution(launcher, cache_values)
    write_json_exclusive(
        args.output,
        {
            "schema": "paper4-p4r1-toolchain-receipt-v1",
            "executable": {
                "path": str(launcher),
                "filename": launcher.name,
                "bytes": launcher.stat().st_size,
                "sha256": sha256_file(launcher),
            },
            "version_command": {
                "argv": [launcher.name, "--version"],
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "combined_sha256": hashlib.sha256(result.stdout + b"\n" + result.stderr).hexdigest(),
            },
            "observed_build_checkout": {
                "path": str(checkout),
                "git_commit": revision_result.stdout.strip(),
                "git_status": "clean including untracked files",
            },
            "observed_cmake_cache": {
                "path": str(cache),
                "sha256": sha256_file(cache),
                "settings": expected_cache,
                "configured_tool_paths": {
                    name: str(path) for name, path in configured_tool_paths.items()
                },
                "resolved_tool_paths": {name: str(path) for name, path in tool_paths.items()},
                "tool_sha256": {name: sha256_file(path) for name, path in tool_paths.items()},
                "tool_versions": tool_versions,
            },
            "observed_driver": {
                "driver_version": driver_version.strip(),
                "gpu_name": gpu_name.strip(),
                "driver_reported_cuda_compatibility": compatibility_match.group(1),
            },
            "observed_cuda_runtime": cuda_runtime,
            "historical_build_association_reference": {
                "llama_cpp_commit": LLAMA_CPP_SOURCE_REVISION,
                "campaign_build_id": LLAMA_CPP_BUILD_ID,
                "meaning": "historical logs printed the short build ID; P4R1 independently read back the clean checkout and build cache above",
            },
            "verification": "executable hash/version, clean Git checkout, CMake cache, compiler/toolkit versions, exact loader-resolved CUDA runtime hashes, feature flags, architecture, and driver read back before inference",
        },
    )


def run_text(command: list[str], context: str) -> str:
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, timeout=30, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{context} failed: {result.stderr.strip()[-500:]}")
    return result.stdout


def record_host(args: argparse.Namespace) -> None:
    cpu_models = {
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("model name") and ":" in line
    }
    if cpu_models != {EXPECTED_HOST["cpu_model"]}:
        raise RuntimeError(f"CPU model differs from P4R1: {sorted(cpu_models)}")
    topology_rows = [
        line for line in run_text(["lscpu", "-p=CPU,CORE,SOCKET"], "lscpu").splitlines()
        if line and not line.startswith("#")
    ]
    topology = [tuple(int(value) for value in line.split(",")) for line in topology_rows]
    logical_cpus = len({row[0] for row in topology})
    physical_cores = len({(row[2], row[1]) for row in topology})
    sockets = len({row[2] for row in topology})
    mem_match = re.search(
        r"^MemTotal:\s+(\d+)\s+kB$",
        Path("/proc/meminfo").read_text(encoding="utf-8"), re.MULTILINE,
    )
    if mem_match is None:
        raise RuntimeError("could not read MemTotal from /proc/meminfo")
    mem_total_kib = int(mem_match.group(1))
    uname = os.uname()
    kernel = f"{uname.sysname} {uname.release} {uname.machine}"
    os_release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')
    gpu_rows = [
        [part.strip() for part in line.split(",")]
        for line in run_text(
            [
                "nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            "nvidia-smi host query",
        ).splitlines()
        if line.strip()
    ]
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 4:
        raise RuntimeError("host receipt requires exactly one GPU")
    gpu_name, gpu_uuid, gpu_memory_raw, driver_version = gpu_rows[0]
    power_text = run_text(["nvidia-smi", "-q", "-d", "POWER"], "nvidia-smi power query")
    power_limits = {}
    for key, label in (
        ("current_w", "Current Power Limit"),
        ("requested_w", "Requested Power Limit"),
        ("default_w", "Default Power Limit"),
    ):
        match = re.search(
            rf"^\s*{re.escape(label)}\s*:\s*([0-9.]+) W$", power_text, re.MULTILINE
        )
        if match is None:
            raise RuntimeError(f"could not read {label} from nvidia-smi")
        power_limits[key] = float(match.group(1))
    path_records = []
    mount_sources: set[tuple[str, str]] = set()
    for role, path in (
        ("workspace", args.workspace),
        ("mixtral_models", args.mixtral_dir),
        ("qwen_models", args.qwen_dir),
    ):
        resolved = path.resolve(strict=True)
        mount_fields = run_text(
            ["findmnt", "-T", str(resolved), "-no", "SOURCE,FSTYPE"], f"findmnt {role}"
        ).strip().split()
        if len(mount_fields) != 2:
            raise RuntimeError(f"could not resolve storage mount for {role}")
        source, filesystem = mount_fields
        mount_sources.add((source, filesystem))
        path_records.append(
            {"role": role, "path": str(resolved), "source": source, "filesystem": filesystem}
        )
    if len(mount_sources) != 1:
        raise RuntimeError("workspace and model artifacts are not on one P4R1 storage device")
    storage_source, filesystem = next(iter(mount_sources))
    parent_name = run_text(["lsblk", "-no", "PKNAME", storage_source], "lsblk parent").strip()
    if not parent_name:
        parent_name = Path(storage_source).name
    storage_json = json.loads(
        run_text(
            ["lsblk", "-J", "-b", "-o", "NAME,MODEL,TRAN,ROTA,TYPE", f"/dev/{parent_name}"],
            "lsblk storage identity",
        )
    )
    devices = storage_json.get("blockdevices") if isinstance(storage_json, dict) else None
    if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], dict):
        raise RuntimeError("could not read one exact P4R1 storage identity")
    device = devices[0]
    observed = {
        "cpu_model": next(iter(cpu_models)), "logical_cpus": logical_cpus,
        "physical_cores": physical_cores, "sockets": sockets,
        "mem_total_kib": mem_total_kib, "kernel": kernel,
        "os_pretty_name": os_release.get("PRETTY_NAME"), "gpu_name": gpu_name,
        "gpu_memory_mib": int(gpu_memory_raw), "gpu_power_limit_w": power_limits["current_w"],
        "driver_version": driver_version, "filesystem": filesystem,
        "storage_model": device.get("model"), "storage_transport": device.get("tran"),
        "storage_rotational": int(bool(device.get("rota"))),
    }
    if observed != EXPECTED_HOST or any(value != 80.0 for value in power_limits.values()):
        raise RuntimeError(f"observed host differs from exact P4R1 host: {observed}")
    if (
        device.get("type") != "disk"
        or device.get("name") != parent_name
        or not re.fullmatch(r"GPU-[0-9A-Fa-f-]{16,}", gpu_uuid)
    ):
        raise RuntimeError("storage/GPU device identity is malformed")
    write_json_exclusive(
        args.output,
        {
            "schema": "paper4-p4r1-host-receipt-v1",
            "cpu": {
                "model": observed["cpu_model"], "logical_cpus": logical_cpus,
                "physical_cores": physical_cores, "sockets": sockets,
            },
            "memory": {"mem_total_kib": mem_total_kib},
            "operating_system": {"pretty_name": observed["os_pretty_name"], "kernel": kernel},
            "gpu": {
                "name": gpu_name, "uuid": gpu_uuid, "memory_total_mib": observed["gpu_memory_mib"],
                "power_limits": power_limits, "driver_version": driver_version,
            },
            "storage": {
                "verified_paths": path_records, "source": storage_source,
                "filesystem": filesystem, "device": f"/dev/{parent_name}",
                "model": device["model"], "transport": device["tran"],
                "rotational": int(bool(device["rota"])),
            },
            "verification": "exact CPU/topology, MemTotal, OS/kernel, single GPU/UUID/VRAM/power/driver, and shared NVMe filesystem read back before inference",
        },
    )


def validate_host_receipt(value: dict[str, Any]) -> None:
    cpu = value.get("cpu")
    memory = value.get("memory")
    operating_system = value.get("operating_system")
    gpu = value.get("gpu")
    storage = value.get("storage")
    if (
        set(value)
        != {"schema", "cpu", "memory", "operating_system", "gpu", "storage", "verification"}
        or value.get("schema") != "paper4-p4r1-host-receipt-v1"
        or cpu
        != {
            "model": EXPECTED_HOST["cpu_model"], "logical_cpus": 32,
            "physical_cores": 24, "sockets": 1,
        }
        or memory != {"mem_total_kib": EXPECTED_HOST["mem_total_kib"]}
        or operating_system
        != {
            "pretty_name": EXPECTED_HOST["os_pretty_name"],
            "kernel": EXPECTED_HOST["kernel"],
        }
        or not isinstance(gpu, dict)
        or set(gpu) != {"name", "uuid", "memory_total_mib", "power_limits", "driver_version"}
        or gpu.get("name") != EXPECTED_HOST["gpu_name"]
        or re.fullmatch(r"GPU-[0-9A-Fa-f-]{16,}", str(gpu.get("uuid"))) is None
        or gpu.get("memory_total_mib") != EXPECTED_HOST["gpu_memory_mib"]
        or gpu.get("power_limits")
        != {"current_w": 80.0, "requested_w": 80.0, "default_w": 80.0}
        or gpu.get("driver_version") != EXPECTED_HOST["driver_version"]
        or not isinstance(storage, dict)
        or set(storage)
        != {"verified_paths", "source", "filesystem", "device", "model", "transport", "rotational"}
        or storage.get("filesystem") != EXPECTED_HOST["filesystem"]
        or storage.get("model") != EXPECTED_HOST["storage_model"]
        or storage.get("transport") != EXPECTED_HOST["storage_transport"]
        or storage.get("rotational") != EXPECTED_HOST["storage_rotational"]
        or not isinstance(storage.get("verified_paths"), list)
    ):
        raise RuntimeError("host receipt differs from the exact P4R1 machine")
    paths = storage["verified_paths"]
    if (
        len(paths) != 3
        or {item.get("role") for item in paths if isinstance(item, dict)}
        != {"workspace", "mixtral_models", "qwen_models"}
        or any(
            not isinstance(item, dict)
            or set(item) != {"role", "path", "source", "filesystem"}
            or not isinstance(item.get("path"), str)
            or not Path(item["path"]).is_absolute()
            or item.get("source") != storage.get("source")
            or item.get("filesystem") != storage.get("filesystem")
            for item in paths
        )
    ):
        raise RuntimeError("host receipt path/storage binding differs")


PLACEMENT_LINE = re.compile(
    r"load_model: loading model|model loaded|initializing, n_slots|offload|n_gpu_layers|CUDA|ROCm|Vulkan|buffer size|load_tensors",
    re.IGNORECASE,
)
OFFLOADED_LAYERS = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", re.IGNORECASE)


def parse_placement_logs(
    stdout: Path, stderr: Path, ngl: str,
) -> tuple[list[str], int, int]:
    """Read one exact llama.cpp offload result from the preserved cell logs."""
    placement_lines: list[str] = []
    offloaded_matches: list[tuple[int, int]] = []
    for log_path in (stdout, stderr):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if PLACEMENT_LINE.search(line):
                placement_lines.append(line[:1000])
            offloaded_matches.extend(
                (int(gpu), int(total))
                for gpu, total in OFFLOADED_LAYERS.findall(line)
            )
    if not placement_lines:
        raise RuntimeError("server logs contain no observable placement record")
    unique_offloaded = set(offloaded_matches)
    if len(unique_offloaded) != 1:
        raise RuntimeError(
            f"server logs do not contain one unambiguous offloaded-layer readback: {sorted(unique_offloaded)}"
        )
    offloaded_gpu, offloaded_total = next(iter(unique_offloaded))
    if ngl != "auto" and offloaded_gpu != int(ngl):
        raise RuntimeError(
            f"observed GPU layer count {offloaded_gpu} differs from requested {ngl}"
        )
    return placement_lines, offloaded_gpu, offloaded_total


def verify_placement_log(args: argparse.Namespace) -> None:
    expected = P4R1_MATRIX.get(args.alias)
    if expected is None or args.ngl != expected[2]:
        raise RuntimeError("placement-log probe differs from the exact P4R1 matrix")
    parse_placement_logs(args.stdout, args.stderr, args.ngl)


READY_RECEIPT_KEYS = frozenset(
    {
        "schema", "phase", "alias", "pid", "process_start_time_ticks",
        "launcher", "model_identity_sha256", "model_manifest_sha256",
        "endpoint", "request", "observed_gpu_process", "verification",
    }
)


def read_regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing or unsafe lifecycle receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"lifecycle receipt is not a JSON object: {path}")
    return value


def validate_ready_pair(
    startup: dict[str, Any], completion: dict[str, Any], alias: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for receipt, phase in ((startup, "startup"), (completion, "completion")):
        if (
            set(receipt) != READY_RECEIPT_KEYS
            or receipt.get("schema") != "paper4-validation-server-identity-v1"
            or receipt.get("phase") != phase
            or receipt.get("alias") != alias
            or not isinstance(receipt.get("pid"), int)
            or isinstance(receipt.get("pid"), bool)
            or receipt["pid"] <= 0
            or not isinstance(receipt.get("process_start_time_ticks"), str)
            or not receipt["process_start_time_ticks"].isdigit()
        ):
            raise RuntimeError(f"{phase} identity receipt schema/process fields differ")
        launcher = receipt.get("launcher")
        endpoint = receipt.get("endpoint")
        request = receipt.get("request")
        gpu = receipt.get("observed_gpu_process")
        if (
            not isinstance(launcher, dict)
            or set(launcher) != {"path", "sha256"}
            or not isinstance(launcher.get("path"), str)
            or not Path(launcher["path"]).is_absolute()
            or not isinstance(launcher.get("sha256"), str)
            or SHA256.fullmatch(launcher["sha256"]) is None
            or not isinstance(endpoint, dict)
            or set(endpoint) != {"host", "port", "base_url", "model_id"}
            or endpoint.get("host") != "127.0.0.1"
            or not isinstance(endpoint.get("port"), int)
            or isinstance(endpoint.get("port"), bool)
            or not 1024 <= endpoint["port"] <= 65535
            or endpoint.get("base_url") != f"http://127.0.0.1:{endpoint['port']}"
            or endpoint.get("model_id") != alias
            or not isinstance(request, dict)
            or set(request) != {"context_tokens", "gpu_layers"}
            or not isinstance(request.get("context_tokens"), int)
            or isinstance(request.get("context_tokens"), bool)
            or not isinstance(request.get("gpu_layers"), str)
            or not isinstance(gpu, dict)
            or set(gpu) != {"pid", "used_gpu_memory_mib", "gpu_uuid"}
            or gpu.get("pid") != receipt["pid"]
            or not isinstance(gpu.get("used_gpu_memory_mib"), int)
            or isinstance(gpu.get("used_gpu_memory_mib"), bool)
            or gpu["used_gpu_memory_mib"] < 0
            or not isinstance(gpu.get("gpu_uuid"), str)
            or re.fullmatch(r"GPU-[0-9A-Fa-f-]{16,}", gpu["gpu_uuid"]) is None
            or not isinstance(receipt.get("model_identity_sha256"), str)
            or SHA256.fullmatch(receipt["model_identity_sha256"]) is None
            or not isinstance(receipt.get("model_manifest_sha256"), str)
            or SHA256.fullmatch(receipt["model_manifest_sha256"]) is None
        ):
            raise RuntimeError(f"{phase} identity receipt has malformed lifecycle metadata")
    shared = (
        "pid", "process_start_time_ticks", "launcher", "model_identity_sha256",
        "model_manifest_sha256", "endpoint", "request",
    )
    if any(startup[key] != completion[key] for key in shared):
        raise RuntimeError("startup and completion identities do not describe one lifecycle")
    return startup["endpoint"], startup["request"]


def validate_lifecycle_receipts(
    args: argparse.Namespace, expected: tuple[str, str, str, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    startup = read_regular_json(args.startup_identity)
    completion = read_regular_json(args.completion_identity)
    shutdown = read_regular_json(args.shutdown_identity)
    model_identity = read_regular_json(args.model_identity)
    endpoint, request = validate_ready_pair(startup, completion, args.alias)
    if request != {"context_tokens": args.context, "gpu_layers": args.ngl}:
        raise RuntimeError("lifecycle request differs from the exact cell matrix")
    if (
        set(model_identity)
        != {"schema", "alias", "model_manifest_sha256", "model", "verification"}
        or model_identity.get("schema") != "paper4-validation-live-model-identity-v1"
        or model_identity.get("alias") != args.alias
        or model_identity.get("model_manifest_sha256") != startup["model_manifest_sha256"]
        or sha256_file(args.model_identity) != startup["model_identity_sha256"]
    ):
        raise RuntimeError("model-identity receipt does not cross-bind the server lifecycle")
    model = model_identity.get("model")
    if (
        not isinstance(model, dict)
        or model.get("alias") != args.alias
        or (model.get("filename"), model.get("sha256")) != expected[:2]
        or not isinstance(model.get("path"), str)
        or not Path(model["path"]).is_absolute()
        or not isinstance(model.get("bytes"), int)
        or isinstance(model.get("bytes"), bool)
        or model["bytes"] <= 0
        or not isinstance(model.get("stat"), dict)
    ):
        raise RuntimeError("model-identity artifact differs from the exact P4R1 matrix")
    launcher = startup["launcher"]
    launcher_path = Path(launcher["path"])
    if launcher_path.is_symlink() or not launcher_path.is_file() or sha256_file(launcher_path) != launcher["sha256"]:
        raise RuntimeError("llama-server executable changed during the cell lifecycle")
    expected_shutdown_keys = {
        "schema", "phase", "alias", "pid", "process_start_time_ticks", "host",
        "port", "same_process_alive", "port_free", "startup_identity_sha256",
        "completion_identity_sha256", "verification",
    }
    if (
        set(shutdown) != expected_shutdown_keys
        or shutdown.get("schema") != "paper4-validation-server-shutdown-v1"
        or shutdown.get("phase") != "shutdown"
        or shutdown.get("alias") != args.alias
        or shutdown.get("pid") != startup["pid"]
        or shutdown.get("process_start_time_ticks") != startup["process_start_time_ticks"]
        or shutdown.get("host") != endpoint["host"]
        or shutdown.get("port") != endpoint["port"]
        or shutdown.get("same_process_alive") is not False
        or shutdown.get("port_free") is not True
        or shutdown.get("startup_identity_sha256") != sha256_file(args.startup_identity)
        or shutdown.get("completion_identity_sha256") != sha256_file(args.completion_identity)
    ):
        raise RuntimeError("shutdown receipt does not cross-bind the complete server lifecycle")
    return startup, completion, shutdown


def record_cell(args: argparse.Namespace) -> None:
    expected = P4R1_MATRIX.get(args.alias)
    if expected is None or (args.ngl, args.context) != (expected[2], expected[3]):
        raise RuntimeError("cell placement/context differs from the exact P4R1 matrix")
    rows = [json.loads(line) for line in args.result.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 100 or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("cell result must contain exactly 100 JSON-object rows")
    ids = [row.get("id") for row in rows]
    if ids != list(P4R1_EXPECTED_IDS) or len(set(ids)) != 100:
        raise RuntimeError("cell result IDs/order differ from the pinned seed-42 subset")
    dataset_sha256 = sha256_file(args.dataset)
    harness_sha256 = sha256_file(args.harness)
    guard_sha256 = sha256_file(args.guard_source)
    startup_identity_sha256 = sha256_file(args.startup_identity)
    host_receipt = read_regular_json(args.host_receipt)
    validate_host_receipt(host_receipt)
    host_receipt_sha256 = sha256_file(args.host_receipt)
    if host_receipt_sha256 != args.host_receipt_sha256:
        raise RuntimeError("cell host receipt differs from the pre-campaign hash")
    if dataset_sha256 != P4R1_DATASET_SHA256:
        raise RuntimeError("cell dataset differs from the pinned seed-42 subset")
    dataset_rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(dataset_rows) != 100
        or any(not isinstance(row, dict) for row in dataset_rows)
        or [row.get("id") for row in dataset_rows] != list(P4R1_EXPECTED_IDS)
    ):
        raise RuntimeError("cell dataset rows/order differ from the pinned P4R1 contract")
    gold_by_id: dict[str, float] = {}
    for item in dataset_rows:
        if (
            not isinstance(item.get("question"), str)
            or not item["question"]
            or not numeric(item.get("answer_number"))
        ):
            raise RuntimeError("cell dataset has an invalid item schema")
        gold_by_id[item["id"]] = float(item["answer_number"])
    sampler_hashes = {row.get("sampler_contract_sha256") for row in rows}
    if len(sampler_hashes) != 1 or not SHA256.fullmatch(next(iter(sampler_hashes), "")):
        raise RuntimeError("cell rows do not bind one sampler contract")
    startup, completion, shutdown = validate_lifecycle_receipts(args, expected)
    sampler_hash = next(iter(sampler_hashes))
    for row in rows:
        validate_result_row(
            row, args.alias, gold_by_id[row["id"]], dataset_sha256,
            harness_sha256, guard_sha256, startup_identity_sha256,
            startup.get("model_manifest_sha256"), sampler_hash,
        )
    placement_lines, offloaded_gpu, offloaded_total = parse_placement_logs(
        args.stdout, args.stderr, args.ngl,
    )
    inputs = {
        name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in (
            ("result", args.result),
            ("stdout", args.stdout),
            ("stderr", args.stderr),
            ("model_identity", args.model_identity),
            ("startup_identity", args.startup_identity),
            ("completion_identity", args.completion_identity),
            ("shutdown_identity", args.shutdown_identity),
        )
    }
    if any(
        row.get("model_manifest_sha256") != startup.get("model_manifest_sha256")
        for row in rows
    ):
        raise RuntimeError("cell rows do not bind the startup model manifest")
    if not isinstance(startup.get("observed_gpu_process"), dict) or not isinstance(
        completion.get("observed_gpu_process"), dict
    ):
        raise RuntimeError("startup/completion receipts lack observed GPU placement")
    write_json_exclusive(
        args.output,
        {
            "schema": "paper4-validation-cell-receipt-v1",
            "alias": args.alias,
            "row_count": 100,
            "model_manifest_sha256": startup["model_manifest_sha256"],
            "dataset_sha256": dataset_sha256,
            "sampler_contract_sha256": next(iter(sampler_hashes)),
            "host_receipt_sha256": host_receipt_sha256,
            "scores_recomputed_from_raw_response_and_pinned_gold": True,
            "api_error_telemetry_preserved_and_validated": True,
            "shutdown_verified": True,
            "engine_executable": startup["launcher"],
            "requested": {"context_tokens": args.context, "gpu_layers": args.ngl},
            "observed_placement_log_lines": placement_lines,
            "observed_offloaded_layers": {
                "gpu": offloaded_gpu,
                "total": offloaded_total,
                "readback": f"{offloaded_gpu}/{offloaded_total}",
            },
            "observed_gpu_process": {
                "startup": startup["observed_gpu_process"],
                "completion": completion["observed_gpu_process"],
            },
            "protocol_sources": {
                "harness_sha256": harness_sha256,
                "server_guard_sha256": guard_sha256,
                "run_one_sha256": sha256_file(args.runner),
                "run_matrix_sha256": sha256_file(args.matrix),
            },
            "files": inputs,
        },
    )


def record_shutdown(args: argparse.Namespace) -> None:
    startup_sha256 = sha256_file(args.startup_identity)
    completion_sha256 = sha256_file(args.completion_identity)
    startup = read_regular_json(args.startup_identity)
    completion = read_regular_json(args.completion_identity)
    endpoint, _request = validate_ready_pair(startup, completion, args.alias)
    if (
        startup.get("pid") != args.pid
        or startup.get("process_start_time_ticks") != args.start_time
        or endpoint.get("host") != args.host
        or endpoint.get("port") != args.port
    ):
        raise RuntimeError("startup/completion identities do not match the stopped process")
    process_alive = same_process(args.pid, args.start_time)
    if process_alive:
        raise RuntimeError("launched server process is still alive after shutdown")
    prove_port_free(args.host, args.port)
    write_json_exclusive(
        args.output,
        {
            "schema": "paper4-validation-server-shutdown-v1",
            "phase": "shutdown",
            "alias": args.alias,
            "pid": args.pid,
            "process_start_time_ticks": args.start_time,
            "host": args.host,
            "port": args.port,
            "same_process_alive": False,
            "port_free": True,
            "startup_identity_sha256": startup_sha256,
            "completion_identity_sha256": completion_sha256,
            "verification": "recorded PID/start-time no longer alive and exact campaign port bind/listen probe succeeded",
        },
    )


def record_campaign(args: argparse.Namespace) -> None:
    raw_manifest = args.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    if manifest_sha256 != args.manifest_sha256:
        raise RuntimeError("campaign model-manifest SHA-256 differs from the approved value")
    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, dict):
        raise RuntimeError("campaign model manifest is not a JSON object")
    toolchain = read_regular_json(args.toolchain_receipt)
    toolchain_sha256 = sha256_file(args.toolchain_receipt)
    post_model_verification = read_regular_json(args.post_model_verification)
    validate_post_model_verification(
        post_model_verification, manifest_sha256, manifest,
        toolchain_sha256, toolchain,
    )
    post_model_verification_sha256 = sha256_file(args.post_model_verification)
    plan = read_regular_json(args.protocol_plan)
    if (
        sha256_file(args.protocol_plan) != args.protocol_plan_sha256
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
        or plan.get("dataset_sha256") != P4R1_DATASET_SHA256
        or plan.get("artifact_sha256") != [value[1] for value in P4R1_MATRIX.values()]
        or re.fullmatch(r"[0-9a-f]{40}", str(plan.get("git_commit"))) is None
    ):
        raise RuntimeError("campaign protocol plan is not the frozen P4R1 prospective design")
    approval = read_regular_json(args.protocol_plan_approval)
    if (
        set(approval)
        != {
            "schema", "operation", "actor", "plan_sha256", "plan_git_commit",
            "approved_at_utc", "method",
        }
        or approval.get("schema") != "paper4-p4r1-plan-approval-v1"
        or approval.get("operation") != "run-p4r1-validation"
        or not isinstance(approval.get("actor"), str)
        or not approval["actor"].strip()
        or approval.get("plan_sha256") != args.protocol_plan_sha256
        or approval.get("plan_git_commit") != plan.get("git_commit")
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00",
            str(approval.get("approved_at_utc")),
        )
        is None
        or approval.get("method") != "interactive-exact-challenge"
    ):
        raise RuntimeError("campaign does not have an exact operator-approved plan")
    host_receipt = read_regular_json(args.host_receipt)
    validate_host_receipt(host_receipt)
    host_receipt_sha256 = sha256_file(args.host_receipt)
    toolchain_executable = toolchain.get("executable")
    if (
        toolchain.get("schema") != "paper4-p4r1-toolchain-receipt-v1"
        or toolchain.get("observed_build_checkout", {}).get("git_commit")
        != LLAMA_CPP_SOURCE_REVISION
        or not isinstance(toolchain_executable, dict)
        or not isinstance(toolchain_executable.get("sha256"), str)
        or SHA256.fullmatch(toolchain_executable["sha256"]) is None
    ):
        raise RuntimeError("campaign toolchain receipt differs from the frozen llama.cpp association")
    if len(args.cell_receipt) != 6:
        raise RuntimeError("campaign requires exactly six cell receipts")
    receipts = []
    aliases: set[str] = set()
    dataset_hashes: set[str] = set()
    sampler_hashes: set[str] = set()
    protocol_sources: list[dict[str, str]] = []
    for path in args.cell_receipt:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        alias = receipt.get("alias")
        if (
            receipt.get("schema") != "paper4-validation-cell-receipt-v1"
            or not isinstance(alias, str)
            or receipt.get("scores_recomputed_from_raw_response_and_pinned_gold") is not True
            or receipt.get("api_error_telemetry_preserved_and_validated") is not True
            or receipt.get("shutdown_verified") is not True
            or receipt.get("host_receipt_sha256") != host_receipt_sha256
        ):
            raise RuntimeError(f"invalid cell receipt: {path}")
        if receipt.get("engine_executable") != {
            "path": toolchain_executable.get("path"),
            "sha256": toolchain_executable.get("sha256"),
        }:
            raise RuntimeError("cell executable identity differs from the campaign toolchain receipt")
        if receipt.get("model_manifest_sha256") != manifest_sha256 or alias in aliases:
            raise RuntimeError("cell receipts do not uniquely bind the campaign model manifest")
        aliases.add(alias)
        dataset_hashes.add(str(receipt.get("dataset_sha256")))
        sampler_hashes.add(str(receipt.get("sampler_contract_sha256")))
        sources = receipt.get("protocol_sources")
        if not isinstance(sources, dict):
            raise RuntimeError("cell receipt has no protocol-source binding")
        protocol_sources.append(sources)
        receipts.append({"alias": alias, "path": f"{path.parent.name}/{path.name}", "sha256": sha256_file(path)})
    if aliases != set(P4R1_MATRIX):
        raise RuntimeError("campaign cell aliases differ from the six-cell validation matrix")
    if dataset_hashes != {P4R1_DATASET_SHA256} or len(sampler_hashes) != 1:
        raise RuntimeError("campaign cells do not share the pinned dataset and sampler contract")
    if any(item != protocol_sources[0] for item in protocol_sources[1:]):
        raise RuntimeError("campaign cells do not share one harness/source identity")
    plan_inventory = plan.get("inference_source_inventory")
    if not isinstance(plan_inventory, list):
        raise RuntimeError("prospective plan has no inference-source inventory")
    plan_by_path = {
        item.get("path"): item
        for item in plan_inventory
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if len(plan_by_path) != len(plan_inventory):
        raise RuntimeError("prospective plan inference-source inventory has duplicate paths")
    source_binding = {
        "harness_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_gsm8k.py",
        "server_guard_sha256": "publication/papers/moe-quantization-granularity/public/replication/server_guard.py",
        "run_one_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_one.sh",
        "run_matrix_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_matrix.sh",
    }
    if any(
        not isinstance(plan_by_path.get(path), dict)
        or plan_by_path[path].get("sha256") != protocol_sources[0].get(key)
        for key, path in source_binding.items()
    ):
        raise RuntimeError("cell protocol source hashes differ from the frozen prospective plan")
    for path in args.cell_receipt:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        expected = P4R1_MATRIX[receipt["alias"]]
        if receipt.get("requested") != {
            "context_tokens": expected[3], "gpu_layers": expected[2]
        }:
            raise RuntimeError("campaign receipt contains a cell outside the exact P4R1 matrix")
    write_json_exclusive(
        args.output,
        {
            "schema": "paper4-validation-campaign-receipt-v1",
            "model_manifest_sha256": manifest_sha256,
            "cell_count": 6,
            "row_count": 600,
            "dataset_sha256": P4R1_DATASET_SHA256,
            "sampler_contract_sha256": next(iter(sampler_hashes)),
            "protocol_sources": protocol_sources[0],
            "prospective_plan": {
                "path": args.protocol_plan.name,
                "sha256": args.protocol_plan_sha256,
                "git_commit": plan.get("git_commit"),
            },
            "prospective_plan_approval": {
                "path": args.protocol_plan_approval.name,
                "sha256": sha256_file(args.protocol_plan_approval),
                "actor": approval["actor"],
                "plan_sha256": approval["plan_sha256"],
            },
            "host_receipt": {
                "path": args.host_receipt.name,
                "sha256": host_receipt_sha256,
            },
            "post_model_verification": {
                "path": args.post_model_verification.name,
                "sha256": post_model_verification_sha256,
            },
            "toolchain_receipt": {
                "path": args.toolchain_receipt.name,
                "sha256": toolchain_sha256,
                "executable_sha256": toolchain_executable["sha256"],
                "llama_cpp_commit": LLAMA_CPP_SOURCE_REVISION,
            },
            "cells": sorted(receipts, key=lambda item: item["alias"]),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    free = subparsers.add_parser("port-free")
    free.add_argument("--host", required=True)
    free.add_argument("--port", required=True, type=int)

    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--pid", required=True, type=int)

    same = subparsers.add_parser("same-process")
    same.add_argument("--pid", required=True, type=int)
    same.add_argument("--start-time", required=True)

    ready = subparsers.add_parser("ready")
    ready.add_argument("--pid", required=True, type=int)
    ready.add_argument("--start-time", required=True)
    ready.add_argument("--launcher", required=True, type=Path)
    ready.add_argument("--model", required=True, type=Path)
    ready.add_argument("--alias", required=True)
    ready.add_argument("--host", required=True)
    ready.add_argument("--port", required=True, type=int)
    ready.add_argument("--base-url", required=True)
    ready.add_argument("--context", required=True, type=int)
    ready.add_argument("--ngl", required=True)

    model = subparsers.add_parser("verify-model-entry")
    model.add_argument("--manifest", required=True, type=Path)
    model.add_argument("--manifest-sha256", required=True)
    model.add_argument("--model", required=True, type=Path)
    model.add_argument("--alias", required=True)
    model.add_argument("--output", type=Path)

    post_models = subparsers.add_parser("verify-manifest-content")
    post_models.add_argument("--manifest", required=True, type=Path)
    post_models.add_argument("--manifest-sha256", required=True)
    post_models.add_argument("--toolchain-receipt", required=True, type=Path)
    post_models.add_argument("--output", required=True, type=Path)

    toolchain = subparsers.add_parser("record-toolchain")
    toolchain.add_argument("--launcher", required=True, type=Path)
    toolchain.add_argument("--output", required=True, type=Path)

    host_receipt = subparsers.add_parser("record-host")
    host_receipt.add_argument("--workspace", required=True, type=Path)
    host_receipt.add_argument("--mixtral-dir", required=True, type=Path)
    host_receipt.add_argument("--qwen-dir", required=True, type=Path)
    host_receipt.add_argument("--output", required=True, type=Path)

    record = subparsers.add_parser("record-ready")
    record.add_argument("--phase", required=True, choices=("startup", "completion"))
    record.add_argument("--pid", required=True, type=int)
    record.add_argument("--start-time", required=True)
    record.add_argument("--launcher", required=True, type=Path)
    record.add_argument("--model", required=True, type=Path)
    record.add_argument("--model-identity", required=True, type=Path)
    record.add_argument("--alias", required=True)
    record.add_argument("--host", required=True)
    record.add_argument("--port", required=True, type=int)
    record.add_argument("--base-url", required=True)
    record.add_argument("--context", required=True, type=int)
    record.add_argument("--ngl", required=True)
    record.add_argument("--output", required=True, type=Path)

    placement = subparsers.add_parser("verify-placement-log")
    placement.add_argument("--alias", required=True)
    placement.add_argument("--ngl", required=True)
    placement.add_argument("--stdout", required=True, type=Path)
    placement.add_argument("--stderr", required=True, type=Path)

    shutdown = subparsers.add_parser("record-shutdown")
    shutdown.add_argument("--pid", required=True, type=int)
    shutdown.add_argument("--start-time", required=True)
    shutdown.add_argument("--alias", required=True)
    shutdown.add_argument("--host", required=True)
    shutdown.add_argument("--port", required=True, type=int)
    shutdown.add_argument("--startup-identity", required=True, type=Path)
    shutdown.add_argument("--completion-identity", required=True, type=Path)
    shutdown.add_argument("--output", required=True, type=Path)

    cell = subparsers.add_parser("record-cell")
    cell.add_argument("--alias", required=True)
    cell.add_argument("--context", required=True, type=int)
    cell.add_argument("--ngl", required=True)
    cell.add_argument("--result", required=True, type=Path)
    cell.add_argument("--stdout", required=True, type=Path)
    cell.add_argument("--stderr", required=True, type=Path)
    cell.add_argument("--model-identity", required=True, type=Path)
    cell.add_argument("--startup-identity", required=True, type=Path)
    cell.add_argument("--completion-identity", required=True, type=Path)
    cell.add_argument("--shutdown-identity", required=True, type=Path)
    cell.add_argument("--dataset", required=True, type=Path)
    cell.add_argument("--harness", required=True, type=Path)
    cell.add_argument("--guard-source", required=True, type=Path)
    cell.add_argument("--host-receipt", required=True, type=Path)
    cell.add_argument("--host-receipt-sha256", required=True)
    cell.add_argument("--runner", required=True, type=Path)
    cell.add_argument("--matrix", required=True, type=Path)
    cell.add_argument("--output", required=True, type=Path)

    campaign = subparsers.add_parser("record-campaign")
    campaign.add_argument("--manifest", required=True, type=Path)
    campaign.add_argument("--manifest-sha256", required=True)
    campaign.add_argument("--protocol-plan", required=True, type=Path)
    campaign.add_argument("--protocol-plan-sha256", required=True)
    campaign.add_argument("--protocol-plan-approval", required=True, type=Path)
    campaign.add_argument("--host-receipt", required=True, type=Path)
    campaign.add_argument("--post-model-verification", required=True, type=Path)
    campaign.add_argument("--toolchain-receipt", required=True, type=Path)
    campaign.add_argument("--cell-receipt", required=True, action="append", type=Path)
    campaign.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "port-free":
            prove_port_free(args.host, args.port)
        elif args.command == "fingerprint":
            print(process_start_time(args.pid))
        elif args.command == "same-process":
            if not same_process(args.pid, args.start_time):
                return 1
        elif args.command == "ready":
            verify_process(args)
            verify_endpoints(args.base_url, args.alias)
        elif args.command == "verify-model-entry":
            verify_model_entry(args)
        elif args.command == "verify-manifest-content":
            verify_manifest_content(args)
        elif args.command == "record-toolchain":
            record_toolchain(args)
        elif args.command == "record-host":
            record_host(args)
        elif args.command == "record-ready":
            record_ready(args)
        elif args.command == "verify-placement-log":
            verify_placement_log(args)
        elif args.command == "record-shutdown":
            record_shutdown(args)
        elif args.command == "record-cell":
            record_cell(args)
        elif args.command == "record-campaign":
            record_campaign(args)
        else:  # pragma: no cover - argparse prevents this
            raise RuntimeError(f"unknown command: {args.command}")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
        print(f"server guard: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
