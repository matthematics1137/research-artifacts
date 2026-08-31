#!/usr/bin/env python3
"""Validate one P4R1 campaign and promote it without touching historical rows.

The destination contains a private canonical evidence tree plus a separately
sanitized public candidate. The command refuses every existing destination.
It does not edit manuscripts, derived claims, historical evidence, or releases.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import getpass
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATASET_SHA256 = "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37"
SEED_BASE = 4_200_000
EXPECTED_IDS = [
    f"gsm8k_{index}"
    for index in sorted(random.Random(42).sample(range(1319), 100))
]
MATRIX = {
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
CUDA_RUNTIME_LIBRARIES = {
    "libcudart.so.12": ("libcudart.so.12.6.77", 716128, "095296727cfb5f51e5b3ae69a3cb88cbd7b6592f314ace0fdfc4e1d758c2aa85"),
    "libcublas.so.12": ("libcublas.so.12.6.4.1", 108244960, "d9343511e1d2ed51e4d65fa94a25cb8f73b0ebc5ae5487da794e18e6950c98dc"),
    "libcublasLt.so.12": ("libcublasLt.so.12.6.4.1", 491106832, "a4ddaed1410acc604815f0e84cdd6c4a70c61cfd2b95dc96360f0799169f0374"),
}
SOURCE_PATHS = {
    "harness_sha256": HERE / "run_gsm8k.py",
    "server_guard_sha256": HERE / "server_guard.py",
    "run_one_sha256": HERE / "run_one.sh",
    "run_matrix_sha256": HERE / "run_matrix.sh",
}
PLAN_SOURCE_PATHS = frozenset(
    {
        "publication/papers/moe-quantization-granularity/public/P4R1_PROTOCOL.md",
        "publication/papers/moe-quantization-granularity/public/FULL_RERUN.md",
        "publication/papers/moe-quantization-granularity/public/replication/prepare_campaign_plan.py",
        "publication/papers/moe-quantization-granularity/public/replication/approve_campaign_plan.py",
        "publication/papers/moe-quantization-granularity/public/replication/prepare_model_manifest.py",
        "publication/papers/moe-quantization-granularity/public/replication/prepare_gsm8k.py",
        "publication/papers/moe-quantization-granularity/public/replication/server_guard.py",
        "publication/papers/moe-quantization-granularity/public/replication/run_gsm8k.py",
        "publication/papers/moe-quantization-granularity/public/replication/run_one.sh",
        "publication/papers/moe-quantization-granularity/public/replication/run_matrix.sh",
    }
)
PLAN_ANALYSIS_PATHS = frozenset(
    {
        "publication/papers/moe-quantization-granularity/public/replication/promote_validation.py",
        "publication/papers/moe-quantization-granularity/public/replication/check_p4r1_evidence.py",
        "publication/papers/moe-quantization-granularity/public/replication/test_p4r1_protocol.py",
        "publication/papers/moe-quantization-granularity/public/replication/test_run_one.py",
        "publication/papers/moe-quantization-granularity/public/replication/test_p4r1_promotion.py",
        "paper4/scripts/build_p4r1_claims.py",
        "paper4/scripts/verify_claims.py",
        "paper4/scripts/check_p4r1_manuscript.py",
        "paper4/scripts/approve_p4r1_narrative.py",
        "paper4/scripts/make_figures.py",
        "paper4/Makefile",
    }
)
WORKSPACE = HERE.parents[4]
PROMOTION_TOOL_PATHS = {
    "promoter": Path(__file__).resolve(),
    "public_checker": HERE / "check_p4r1_evidence.py",
    "claims_builder": WORKSPACE / "paper4/scripts/build_p4r1_claims.py",
    "manuscript_gate": WORKSPACE / "paper4/scripts/check_p4r1_manuscript.py",
    "claims_analysis": WORKSPACE / "paper4/scripts/verify_claims.py",
    "paper_makefile": WORKSPACE / "paper4/Makefile",
}
PRIVATE_FILENAMES = (
    "model-identity.json", "startup-identity.json", "completion-identity.json",
    "shutdown-identity.json", "stdout.log", "stderr.log", "cell-receipt.json",
)
FORBIDDEN_PUBLIC = (
    re.compile(rb"/(?:home|Users)/[^/\s]+"),
    re.compile(rb"file://", re.IGNORECASE),
    re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    re.compile(rb"(?:token|secret|password)\s*[:=]\s*[^\s]{8,}", re.IGNORECASE),
    re.compile(rb"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    re.compile(rb"(?<![0-9])(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}(?![0-9])"),
    re.compile(rb"GPU-[0-9A-F-]{16,}", re.IGNORECASE),
    re.compile(rb'"(?:email|gpu_uuid|host|hostname|user|username)"\s*:', re.IGNORECASE),
)
SHA256 = re.compile(r"[0-9a-f]{64}")
NUMBER = re.compile(r"[-+]?\$?[\d][\d,]*(?:\.\d+)?")
PRIVATE_ROW_KEYS = frozenset(
    {
        "id", "label", "correct", "wall_s", "identity_guard_wall_s",
        "completion_tokens", "thinking_tokens_estimate", "truncated",
        "raw_response", "scoring_content", "parsed_answer_number",
        "expected_answer_number",
        "api_error", "chat_template_retry_error",
        "chat_template_kwargs_dropped", "model_id",
        "model_manifest_sha256", "startup_identity_sha256", "sampler_seed",
        "dataset_sha256", "harness_sha256", "server_guard_sha256",
        "sampler_contract_sha256",
    }
)
PUBLIC_ROW_KEYS = frozenset(
    {
        "id", "label", "correct", "wall_s", "identity_guard_wall_s",
        "completion_tokens", "thinking_tokens_estimate", "truncated",
        "model_id", "model_manifest_sha256", "startup_identity_sha256",
        "sampler_seed", "dataset_sha256", "harness_sha256",
        "server_guard_sha256", "sampler_contract_sha256",
        "chat_template_kwargs_dropped", "api_error_recorded",
    }
)
EXPECTED_HOST = {
    "cpu": {
        "model": "Intel(R) Core(TM) i9-14900HX", "logical_cpus": 32,
        "physical_cores": 24, "sockets": 1,
    },
    "memory": {"mem_total_kib": 65445316},
    "operating_system": {
        "pretty_name": "Ubuntu 24.04.3 LTS", "kernel": "Linux 7.0.0-29-generic x86_64",
    },
}


class PromotionError(RuntimeError):
    pass


def numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def require_finite_json(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PromotionError(f"{context} contains a non-finite number")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise PromotionError(f"{context} contains a non-string object key")
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
        raise PromotionError(f"{context} has an invalid exact-telemetry schema")
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
        raise PromotionError(f"{context} contains an invalid exact-telemetry value")
    try:
        body = base64.b64decode(value["body_base64"], validate=True)
    except (ValueError, binascii.Error) as error:
        raise PromotionError(f"{context} contains invalid base64") from error
    if len(body) != value["body_bytes"] or sha256_bytes(body) != value["body_sha256"]:
        raise PromotionError(f"{context} body length/hash differs from the preserved bytes")
    if value["http_status"] is None and (
        value["http_reason"] is not None
        or value["url"] is not None
        or value["headers"]
        or body
    ):
        raise PromotionError(f"{context} non-HTTP exception contains HTTP-only fields")


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


def response_scoring_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise PromotionError("successful private row has no first response choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content") or "", str):
        raise PromotionError("successful private row has no textual response message")
    content = message.get("content") or ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    return content.strip()


def response_thinking_tokens(response: dict[str, Any]) -> int:
    choices = response.get("choices")
    message = choices[0]["message"]
    reasoning = message.get("reasoning_content")
    if not reasoning:
        content = message.get("content") or ""
        reasoning = content.split("</think>", 1)[0] if "</think>" in content else ""
    return int(len(reasoning.split()) * 1.3) if reasoning else 0


def validate_dataset(path: Path) -> dict[str, float]:
    raw = regular(path).read_bytes()
    if sha256_bytes(raw) != DATASET_SHA256:
        raise PromotionError("promotion dataset differs from the pinned P4R1 subset")
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"promotion dataset is malformed: {error}") from error
    if len(rows) != 100 or [row.get("id") for row in rows if isinstance(row, dict)] != EXPECTED_IDS:
        raise PromotionError("promotion dataset IDs/order differ from P4R1")
    result: dict[str, float] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not isinstance(row.get("question"), str)
            or not row["question"]
            or not numeric(row.get("answer_number"))
        ):
            raise PromotionError("promotion dataset row has an invalid schema")
        result[row["id"]] = float(row["answer_number"])
    return result


def validate_private_row(
    row: dict[str, Any], alias: str, expected_gold: float,
    manifest_sha256: str, sampler_hash: str, startup_identity_sha256: str,
    protocol_sources: dict[str, Any],
) -> None:
    if set(row) != PRIVATE_ROW_KEYS:
        missing = sorted(PRIVATE_ROW_KEYS - set(row))
        extra = sorted(set(row) - PRIVATE_ROW_KEYS)
        raise PromotionError(f"{alias}: private row keys differ; missing={missing}, extra={extra}")
    match = re.fullmatch(r"gsm8k_(\d+)", row["id"]) if isinstance(row["id"], str) else None
    scalar_types_ok = (
        row["label"] == alias
        and row["model_id"] == alias
        and isinstance(row["correct"], bool)
        and numeric(row["wall_s"]) and row["wall_s"] >= 0
        and numeric(row["identity_guard_wall_s"]) and row["identity_guard_wall_s"] >= 0
        and (row["completion_tokens"] is None or (
            isinstance(row["completion_tokens"], int)
            and not isinstance(row["completion_tokens"], bool)
            and row["completion_tokens"] >= 0
        ))
        and isinstance(row["thinking_tokens_estimate"], int)
        and not isinstance(row["thinking_tokens_estimate"], bool)
        and row["thinking_tokens_estimate"] >= 0
        and isinstance(row["truncated"], bool)
        and isinstance(row["scoring_content"], str)
        and (row["parsed_answer_number"] is None or numeric(row["parsed_answer_number"]))
        and numeric(row["expected_answer_number"])
        and isinstance(row["chat_template_kwargs_dropped"], bool)
        and (row["api_error"] is None or isinstance(row["api_error"], dict))
        and (
            row["chat_template_retry_error"] is None
            or isinstance(row["chat_template_retry_error"], dict)
        )
        and match is not None
        and isinstance(row["sampler_seed"], int)
        and not isinstance(row["sampler_seed"], bool)
    )
    if not scalar_types_ok:
        raise PromotionError(f"{alias}: private row has an invalid value or type")
    hashes = (
        "model_manifest_sha256", "startup_identity_sha256", "dataset_sha256",
        "harness_sha256", "server_guard_sha256", "sampler_contract_sha256",
    )
    if any(not isinstance(row[key], str) or SHA256.fullmatch(row[key]) is None for key in hashes):
        raise PromotionError(f"{alias}: private row has a malformed SHA-256 field")
    if (
        row["model_manifest_sha256"] != manifest_sha256
        or row["startup_identity_sha256"] != startup_identity_sha256
        or row["dataset_sha256"] != DATASET_SHA256
        or row["sampler_contract_sha256"] != sampler_hash
        or row["harness_sha256"] != protocol_sources["harness_sha256"]
        or row["server_guard_sha256"] != protocol_sources["server_guard_sha256"]
        or row["sampler_seed"] != SEED_BASE + int(match.group(1))
        or float(row["expected_answer_number"]) != expected_gold
    ):
        raise PromotionError(f"{alias}: private row identity, source, seed, or gold binding differs")
    if row["api_error"] is None:
        if row["chat_template_retry_error"] is not None:
            validate_error_telemetry(
                row["chat_template_retry_error"], f"{alias}: chat-template retry error"
            )
        if row["chat_template_kwargs_dropped"] != (
            row["chat_template_retry_error"] is not None
        ):
            raise PromotionError(f"{alias}: chat-template retry flag differs from telemetry")
        if not isinstance(row["raw_response"], dict):
            raise PromotionError(f"{alias}: successful private row lacks its raw response")
        require_finite_json(row["raw_response"], f"{alias}: raw response")
        if row["raw_response"].get("model") != alias:
            raise PromotionError(f"{alias}: raw response model identity differs")
        content = response_scoring_content(row["raw_response"])
        choices = row["raw_response"]["choices"]
        usage = row["raw_response"].get("usage") or {}
        if not isinstance(usage, dict):
            raise PromotionError(f"{alias}: raw response usage is not an object")
        if (
            content != row["scoring_content"]
            or parse_gsm8k_answer(content) != row["parsed_answer_number"]
            or (choices[0].get("finish_reason") == "length") != row["truncated"]
            or usage.get("completion_tokens") != row["completion_tokens"]
            or response_thinking_tokens(row["raw_response"])
            != row["thinking_tokens_estimate"]
        ):
            raise PromotionError(f"{alias}: scored fields differ from the preserved raw response")
        recomputed = score_gsm8k(content, expected_gold)
    else:
        validate_error_telemetry(row["api_error"], f"{alias}: API error")
        if row["chat_template_retry_error"] is not None:
            validate_error_telemetry(
                row["chat_template_retry_error"], f"{alias}: chat-template retry error"
            )
        if row["chat_template_kwargs_dropped"] != (
            row["chat_template_retry_error"] is not None
        ):
            raise PromotionError(f"{alias}: chat-template retry flag differs from telemetry")
        if row["raw_response"] is not None:
            if not isinstance(row["raw_response"], dict):
                raise PromotionError(f"{alias}: failed request raw response is not an object/null")
            require_finite_json(row["raw_response"], f"{alias}: failed raw response")
            if row["raw_response"].get("model") != alias:
                raise PromotionError(f"{alias}: failed-row raw response model identity differs")
        if row["scoring_content"] != "" or row["parsed_answer_number"] is not None or row["correct"]:
            raise PromotionError(f"{alias}: failed request is not represented as an incorrect empty-parser row")
        recomputed = False
    if recomputed != row["correct"]:
        raise PromotionError(f"{alias}: stored score differs from raw-response/gold recomputation")


def validate_public_row(row: dict[str, Any]) -> None:
    if set(row) != PUBLIC_ROW_KEYS:
        raise PromotionError("public projection is missing a required key or contains an extra key")
    if (
        not isinstance(row["correct"], bool)
        or not isinstance(row["truncated"], bool)
        or not isinstance(row["chat_template_kwargs_dropped"], bool)
        or not isinstance(row["api_error_recorded"], bool)
        or not numeric(row["wall_s"]) or row["wall_s"] < 0
        or not numeric(row["identity_guard_wall_s"]) or row["identity_guard_wall_s"] < 0
        or not isinstance(row["thinking_tokens_estimate"], int)
        or isinstance(row["thinking_tokens_estimate"], bool)
        or row["thinking_tokens_estimate"] < 0
        or not (
            row["completion_tokens"] is None
            or isinstance(row["completion_tokens"], int)
            and not isinstance(row["completion_tokens"], bool)
            and row["completion_tokens"] >= 0
        )
        or not isinstance(row["id"], str)
        or re.fullmatch(r"gsm8k_\d+", row["id"]) is None
        or not isinstance(row["label"], str)
        or not isinstance(row["model_id"], str)
        or row["model_id"] != row["label"]
        or not isinstance(row["sampler_seed"], int)
        or isinstance(row["sampler_seed"], bool)
        or any(
            not isinstance(row[key], str) or SHA256.fullmatch(row[key]) is None
            for key in (
                "model_manifest_sha256", "startup_identity_sha256",
                "dataset_sha256", "harness_sha256", "server_guard_sha256",
                "sampler_contract_sha256",
            )
        )
    ):
        raise PromotionError("public projection contains an invalid value or type")


def validate_private_host(value: dict[str, Any]) -> None:
    gpu = value.get("gpu")
    storage = value.get("storage")
    if (
        set(value)
        != {"schema", "cpu", "memory", "operating_system", "gpu", "storage", "verification"}
        or value.get("schema") != "paper4-p4r1-host-receipt-v1"
        or value.get("cpu") != EXPECTED_HOST["cpu"]
        or value.get("memory") != EXPECTED_HOST["memory"]
        or value.get("operating_system") != EXPECTED_HOST["operating_system"]
        or not isinstance(gpu, dict)
        or set(gpu) != {"name", "uuid", "memory_total_mib", "power_limits", "driver_version"}
        or gpu.get("name") != "NVIDIA GeForce RTX 4080 Laptop GPU"
        or re.fullmatch(r"GPU-[0-9A-Fa-f-]{16,}", str(gpu.get("uuid"))) is None
        or gpu.get("memory_total_mib") != 12282
        or gpu.get("power_limits")
        != {"current_w": 80.0, "requested_w": 80.0, "default_w": 80.0}
        or gpu.get("driver_version") != "580.167.08"
        or not isinstance(storage, dict)
        or set(storage)
        != {"verified_paths", "source", "filesystem", "device", "model", "transport", "rotational"}
        or storage.get("filesystem") != "ext4"
        or storage.get("model") != "WD PC SN740 SDDPNQE-2T00-1102"
        or storage.get("transport") != "nvme"
        or storage.get("rotational") != 0
        or not isinstance(storage.get("verified_paths"), list)
    ):
        raise PromotionError("private host receipt differs from the exact P4R1 host")
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
        raise PromotionError("private host receipt paths/storage are malformed")


def cuda_runtime_stable_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema", "configured_cudatoolkit_root", "configured_lib64",
        "resolved_lib64", "ld_library_path_first", "loader_probe",
        "libraries", "verification",
    }:
        raise PromotionError("CUDA runtime receipt has an unknown schema")
    if (
        value.get("schema") != "paper4-p4r1-cuda-runtime-resolution-v1"
        or value.get("verification")
        != "ldd resolved the exact hashed CUDA 12.6 runtime libraries beneath the CMake-configured toolkit root"
    ):
        raise PromotionError("CUDA runtime receipt has an unknown identity statement")
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
        raise PromotionError("CUDA runtime receipt paths are malformed")
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
        raise PromotionError("CUDA loader probe is malformed")
    libraries = value.get("libraries")
    by_soname = {
        item.get("soname"): item for item in libraries if isinstance(item, dict)
    } if isinstance(libraries, list) else {}
    if len(by_soname) != 3 or set(by_soname) != set(CUDA_RUNTIME_LIBRARIES):
        raise PromotionError("CUDA runtime library inventory is incomplete or duplicated")
    for soname, (filename, byte_count, digest) in CUDA_RUNTIME_LIBRARIES.items():
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
            or item.get("resolved_filename") != filename
            or resolved_path.name != filename
            or item.get("bytes") != byte_count
            or item.get("sha256") != digest
        ):
            raise PromotionError(f"{soname}: CUDA runtime receipt differs from the frozen library")
    stable_probe = dict(probe)
    stable_probe.pop("stdout_sha256")
    return {
        "schema": value["schema"],
        "configured_cudatoolkit_root": str(toolkit_root),
        "configured_lib64": str(configured_lib64),
        "resolved_lib64": str(resolved_lib64),
        "ld_library_path_first": value["ld_library_path_first"],
        "loader_probe": stable_probe,
        "libraries": [by_soname[soname] for soname in CUDA_RUNTIME_LIBRARIES],
        "verification": value["verification"],
    }


def validate_private_post_model_verification(
    value: dict[str, Any], manifest_sha256: str,
    artifact_by_alias: dict[str, dict[str, Any]], toolchain_sha256: str,
    toolchain: dict[str, Any],
) -> None:
    artifacts = value.get("artifacts")
    if (
        set(value)
        != {
            "schema", "model_manifest_sha256", "artifact_count", "total_bytes",
            "artifacts", "toolchain_receipt_sha256", "post_campaign_cuda_runtime",
            "verification",
        }
        or value.get("schema")
        != "paper4-p4r1-post-campaign-model-verification-v1"
        or value.get("model_manifest_sha256") != manifest_sha256
        or value.get("toolchain_receipt_sha256") != toolchain_sha256
        or value.get("artifact_count") != 6
        or value.get("total_bytes") != sum(P4R1_BYTES.values())
        or not isinstance(artifacts, list)
        or len(artifacts) != 6
    ):
        raise PromotionError("post-campaign model verification has the wrong schema/count")
    if cuda_runtime_stable_identity(
        value.get("post_campaign_cuda_runtime")
    ) != cuda_runtime_stable_identity(toolchain.get("observed_cuda_runtime")):
        raise PromotionError("post-campaign CUDA runtime differs from the pre-campaign toolchain")
    by_alias = {
        item.get("alias"): item for item in artifacts if isinstance(item, dict)
    }
    if len(by_alias) != 6 or set(by_alias) != set(MATRIX):
        raise PromotionError("post-campaign model verification aliases differ from P4R1")
    for alias, expected in MATRIX.items():
        item = by_alias[alias]
        manifest_item = artifact_by_alias[alias]
        if (
            set(item)
            != {
                "alias", "filename", "bytes", "sha256", "pre_campaign_stat",
                "post_campaign_stat",
            }
            or item.get("filename") != expected[0]
            or item.get("bytes") != P4R1_BYTES[alias]
            or item.get("sha256") != expected[1]
            or item.get("pre_campaign_stat") != manifest_item.get("stat")
            or item.get("post_campaign_stat") != manifest_item.get("stat")
        ):
            raise PromotionError(f"{alias}: post-campaign artifact content/stat differs")


def validate_plan_and_toolchain(
    campaign: Path, campaign_receipt: dict[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    plan_record = campaign_receipt.get("prospective_plan")
    approval_record = campaign_receipt.get("prospective_plan_approval")
    toolchain_record = campaign_receipt.get("toolchain_receipt")
    if (
        not isinstance(plan_record, dict)
        or set(plan_record) != {"path", "sha256", "git_commit"}
        or plan_record.get("path") != "protocol-plan.json"
        or not isinstance(plan_record.get("sha256"), str)
        or SHA256.fullmatch(plan_record["sha256"]) is None
        or not isinstance(approval_record, dict)
        or set(approval_record) != {"path", "sha256", "actor", "plan_sha256"}
        or approval_record.get("path") != "protocol-plan-approval.json"
        or not isinstance(approval_record.get("sha256"), str)
        or SHA256.fullmatch(approval_record["sha256"]) is None
        or approval_record.get("plan_sha256") != plan_record.get("sha256")
        or not isinstance(approval_record.get("actor"), str)
        or not approval_record["actor"].strip()
        or not isinstance(toolchain_record, dict)
        or set(toolchain_record)
        != {"path", "sha256", "executable_sha256", "llama_cpp_commit"}
        or toolchain_record.get("path") != "toolchain-receipt.json"
        or any(
            not isinstance(toolchain_record.get(key), str)
            or SHA256.fullmatch(toolchain_record[key]) is None
            for key in ("sha256", "executable_sha256")
        )
    ):
        raise PromotionError("campaign has no exact prospective-plan/toolchain binding")
    plan_path = private_file(campaign / "protocol-plan.json")
    approval_path = private_file(campaign / "protocol-plan-approval.json")
    toolchain_path = private_file(campaign / "toolchain-receipt.json")
    if (
        sha256_file(plan_path) != plan_record["sha256"]
        or sha256_file(approval_path) != approval_record["sha256"]
        or sha256_file(toolchain_path) != toolchain_record["sha256"]
    ):
        raise PromotionError("campaign plan/toolchain bytes differ from the campaign receipt")
    plan = read_json(plan_path)
    approval = read_json(approval_path)
    if (
        set(plan)
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
        or plan.get("dataset_sha256") != DATASET_SHA256
        or plan.get("git_commit") != plan_record["git_commit"]
        or re.fullmatch(r"[0-9a-f]{40}", str(plan.get("git_commit"))) is None
        or not isinstance(plan.get("inference_source_inventory"), list)
        or not isinstance(plan.get("analysis_source_inventory"), list)
        or plan.get("artifact_sha256") != [value[1] for value in MATRIX.values()]
    ):
        raise PromotionError("prospective plan identity/status/source inventory differs from P4R1")
    if (
        set(approval)
        != {
            "schema", "operation", "actor", "plan_sha256", "plan_git_commit",
            "approved_at_utc", "method",
        }
        or approval.get("schema") != "paper4-p4r1-plan-approval-v1"
        or approval.get("operation") != "run-p4r1-validation"
        or approval.get("actor") != approval_record["actor"]
        or approval.get("plan_sha256") != plan_record["sha256"]
        or approval.get("plan_git_commit") != plan.get("git_commit")
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00",
            str(approval.get("approved_at_utc")),
        )
        is None
        or approval.get("method") != "interactive-exact-challenge"
    ):
        raise PromotionError("campaign approval does not authorize the exact prospective plan")
    for item in plan["inference_source_inventory"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256", "git_blob"}
            or not isinstance(item.get("path"), str)
            or item["path"].startswith("/")
            or not isinstance(item.get("bytes"), int)
            or isinstance(item.get("bytes"), bool)
            or item["bytes"] <= 0
            or not isinstance(item.get("sha256"), str)
            or SHA256.fullmatch(item["sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(item.get("git_blob"))) is None
        ):
            raise PromotionError("prospective inference-source inventory is malformed")
    plan_by_path = {
        item["path"]: item for item in plan["inference_source_inventory"]
    }
    if set(plan_by_path) != PLAN_SOURCE_PATHS or len(plan_by_path) != len(
        plan["inference_source_inventory"]
    ):
        raise PromotionError("prospective plan source inventory is incomplete or duplicated")
    analysis_by_path: dict[str, dict[str, Any]] = {}
    for item in plan["analysis_source_inventory"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256", "git_blob"}
            or not isinstance(item.get("path"), str)
            or item["path"].startswith("/")
            or not isinstance(item.get("bytes"), int)
            or isinstance(item["bytes"], bool)
            or item["bytes"] <= 0
            or not isinstance(item.get("sha256"), str)
            or SHA256.fullmatch(item["sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(item.get("git_blob"))) is None
        ):
            raise PromotionError("prospective analysis-source inventory is malformed")
        if item["path"] in analysis_by_path:
            raise PromotionError("prospective analysis-source inventory contains a duplicate")
        analysis_by_path[item["path"]] = item
    if set(analysis_by_path) != PLAN_ANALYSIS_PATHS:
        raise PromotionError("prospective analysis-source inventory is incomplete or extra")
    verify_current_analysis_freeze(analysis_by_path)
    source_binding = {
        "harness_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_gsm8k.py",
        "server_guard_sha256": "publication/papers/moe-quantization-granularity/public/replication/server_guard.py",
        "run_one_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_one.sh",
        "run_matrix_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_matrix.sh",
    }
    protocol_sources = campaign_receipt.get("protocol_sources")
    if not isinstance(protocol_sources, dict) or any(
        plan_by_path[path]["sha256"] != protocol_sources.get(key)
        for key, path in source_binding.items()
    ):
        raise PromotionError("campaign source hashes differ from the prospective plan")
    toolchain = read_json(toolchain_path)
    executable = toolchain.get("executable")
    checkout = toolchain.get("observed_build_checkout")
    cache = toolchain.get("observed_cmake_cache")
    driver = toolchain.get("observed_driver")
    cuda_runtime = toolchain.get("observed_cuda_runtime")
    version = toolchain.get("version_command")
    if (
        toolchain.get("schema") != "paper4-p4r1-toolchain-receipt-v1"
        or not isinstance(executable, dict)
        or set(executable) != {"path", "filename", "bytes", "sha256"}
        or executable.get("sha256") != toolchain_record["executable_sha256"]
        or not isinstance(executable.get("path"), str)
        or not Path(executable["path"]).is_absolute()
        or Path(executable["path"]).name != executable.get("filename")
        or not isinstance(executable.get("bytes"), int)
        or isinstance(executable.get("bytes"), bool)
        or executable["bytes"] <= 0
        or not isinstance(checkout, dict)
        or checkout.get("git_commit") != toolchain_record.get("llama_cpp_commit")
        or checkout.get("git_status") != "clean including untracked files"
        or not isinstance(cache, dict)
        or cache.get("settings")
        != {
            "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "89",
            "GGML_BLAS": "OFF", "GGML_CUDA": "ON", "GGML_NATIVE": "ON",
            "GGML_OPENMP": "ON", "LLAMA_CURL": "OFF",
        }
        or not isinstance(driver, dict)
        or driver.get("driver_version") != "580.167.08"
        or driver.get("driver_reported_cuda_compatibility") != "13.0"
        or not isinstance(cuda_runtime, dict)
        or set(cuda_runtime)
        != {
            "schema", "configured_cudatoolkit_root", "configured_lib64",
            "resolved_lib64", "ld_library_path_first", "loader_probe",
            "libraries", "verification",
        }
        or cuda_runtime.get("schema")
        != "paper4-p4r1-cuda-runtime-resolution-v1"
        or not isinstance(cuda_runtime.get("libraries"), list)
        or len(cuda_runtime["libraries"]) != 3
        or not isinstance(version, dict)
        or version.get("returncode") != 0
        or "035e227" not in (str(version.get("stdout")) + str(version.get("stderr")))
    ):
        raise PromotionError("toolchain receipt does not prove the frozen clean build/readbacks")
    toolkit_root = Path(str(cuda_runtime["configured_cudatoolkit_root"]))
    configured_lib64 = Path(str(cuda_runtime["configured_lib64"]))
    resolved_lib64 = Path(str(cuda_runtime["resolved_lib64"]))
    loader_probe = cuda_runtime.get("loader_probe")
    if (
        not toolkit_root.is_absolute()
        or not configured_lib64.is_absolute()
        or not resolved_lib64.is_absolute()
        or not configured_lib64.is_relative_to(toolkit_root)
        or not resolved_lib64.is_relative_to(toolkit_root)
        or cuda_runtime.get("ld_library_path_first") != str(configured_lib64)
        or not isinstance(loader_probe, dict)
        or set(loader_probe)
        != {"tool_path", "tool_sha256", "returncode", "stdout_sha256", "stderr_sha256"}
        or not Path(str(loader_probe.get("tool_path"))).is_absolute()
        or loader_probe.get("returncode") != 0
        or any(
            not isinstance(loader_probe.get(key), str)
            or SHA256.fullmatch(loader_probe[key]) is None
            for key in ("tool_sha256", "stdout_sha256", "stderr_sha256")
        )
    ):
        raise PromotionError("CUDA runtime loader provenance is malformed")
    runtime_by_soname = {
        item.get("soname"): item
        for item in cuda_runtime["libraries"]
        if isinstance(item, dict)
    }
    if len(runtime_by_soname) != 3 or set(runtime_by_soname) != set(CUDA_RUNTIME_LIBRARIES):
        raise PromotionError("CUDA runtime library inventory is incomplete or duplicated")
    for soname, (filename, byte_count, digest) in CUDA_RUNTIME_LIBRARIES.items():
        item = runtime_by_soname[soname]
        loader_path = Path(str(item.get("loader_path")))
        resolved_path = Path(str(item.get("resolved_path")))
        if (
            set(item)
            != {"soname", "loader_path", "resolved_path", "resolved_filename", "bytes", "sha256"}
            or not loader_path.is_absolute()
            or not resolved_path.is_absolute()
            or not loader_path.is_relative_to(toolkit_root)
            or not resolved_path.is_relative_to(toolkit_root)
            or item.get("resolved_filename") != filename
            or resolved_path.name != filename
            or item.get("bytes") != byte_count
            or item.get("sha256") != digest
        ):
            raise PromotionError(f"{soname}: CUDA runtime receipt differs from the frozen library")
    executable_path = Path(executable["path"])
    if (
        executable_path.is_symlink()
        or not executable_path.is_file()
        or executable_path.stat().st_size != executable["bytes"]
        or sha256_file(executable_path) != executable["sha256"]
    ):
        raise PromotionError("llama-server executable changed between campaign and promotion")
    return plan_path, plan, approval_path, approval, toolchain_path, toolchain


def validate_private_lifecycle(
    log_dir: Path, alias: str, expected: tuple[str, str, str, int],
    manifest_sha256: str, toolchain: dict[str, Any],
) -> dict[str, Any]:
    model_identity_path = log_dir / "model-identity.json"
    startup_path = log_dir / "startup-identity.json"
    completion_path = log_dir / "completion-identity.json"
    shutdown_path = log_dir / "shutdown-identity.json"
    model_identity = read_json(model_identity_path)
    startup = read_json(startup_path)
    completion = read_json(completion_path)
    shutdown = read_json(shutdown_path)
    ready_keys = {
        "schema", "phase", "alias", "pid", "process_start_time_ticks",
        "launcher", "model_identity_sha256", "model_manifest_sha256",
        "endpoint", "request", "observed_gpu_process", "verification",
    }
    for receipt, phase in ((startup, "startup"), (completion, "completion")):
        if (
            set(receipt) != ready_keys
            or receipt.get("schema") != "paper4-validation-server-identity-v1"
            or receipt.get("phase") != phase
            or receipt.get("alias") != alias
        ):
            raise PromotionError(f"{alias}: malformed {phase} lifecycle receipt")
    for key in (
        "pid", "process_start_time_ticks", "launcher", "model_identity_sha256",
        "model_manifest_sha256", "endpoint", "request",
    ):
        if startup.get(key) != completion.get(key):
            raise PromotionError(f"{alias}: startup/completion lifecycle differs at {key}")
    executable = toolchain["executable"]
    if startup.get("launcher") != {"path": executable["path"], "sha256": executable["sha256"]}:
        raise PromotionError(f"{alias}: lifecycle executable differs from toolchain")
    endpoint = startup.get("endpoint")
    request = startup.get("request")
    if (
        startup.get("model_manifest_sha256") != manifest_sha256
        or not isinstance(startup.get("pid"), int)
        or isinstance(startup.get("pid"), bool)
        or startup["pid"] <= 0
        or not isinstance(startup.get("process_start_time_ticks"), str)
        or not startup["process_start_time_ticks"].isdigit()
        or not isinstance(endpoint, dict)
        or endpoint
        != {
            "host": "127.0.0.1", "port": endpoint.get("port") if isinstance(endpoint, dict) else None,
            "base_url": f"http://127.0.0.1:{endpoint.get('port')}" if isinstance(endpoint, dict) else None,
            "model_id": alias,
        }
        or not isinstance(endpoint.get("port"), int)
        or isinstance(endpoint.get("port"), bool)
        or request != {"context_tokens": expected[3], "gpu_layers": expected[2]}
        or startup.get("model_identity_sha256") != sha256_file(model_identity_path)
    ):
        raise PromotionError(f"{alias}: lifecycle endpoint/request/model binding differs")
    for receipt in (startup, completion):
        gpu = receipt.get("observed_gpu_process")
        if (
            not isinstance(gpu, dict)
            or set(gpu) != {"pid", "used_gpu_memory_mib", "gpu_uuid"}
            or gpu.get("pid") != startup["pid"]
            or not isinstance(gpu.get("used_gpu_memory_mib"), int)
            or isinstance(gpu.get("used_gpu_memory_mib"), bool)
            or not isinstance(gpu.get("gpu_uuid"), str)
        ):
            raise PromotionError(f"{alias}: malformed lifecycle GPU snapshot")
    model = model_identity.get("model")
    if (
        set(model_identity)
        != {"schema", "alias", "model_manifest_sha256", "model", "verification"}
        or model_identity.get("schema") != "paper4-validation-live-model-identity-v1"
        or model_identity.get("alias") != alias
        or model_identity.get("model_manifest_sha256") != manifest_sha256
        or not isinstance(model, dict)
        or (model.get("filename"), model.get("sha256")) != expected[:2]
    ):
        raise PromotionError(f"{alias}: live model receipt differs from the artifact manifest")
    if (
        shutdown.get("schema") != "paper4-validation-server-shutdown-v1"
        or shutdown.get("phase") != "shutdown"
        or shutdown.get("alias") != alias
        or shutdown.get("pid") != startup["pid"]
        or shutdown.get("process_start_time_ticks") != startup["process_start_time_ticks"]
        or shutdown.get("host") != endpoint["host"]
        or shutdown.get("port") != endpoint["port"]
        or shutdown.get("same_process_alive") is not False
        or shutdown.get("port_free") is not True
        or shutdown.get("startup_identity_sha256") != sha256_file(startup_path)
        or shutdown.get("completion_identity_sha256") != sha256_file(completion_path)
    ):
        raise PromotionError(f"{alias}: shutdown receipt does not close the same lifecycle")
    return startup


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def promotion_tooling_receipt() -> dict[str, Any]:
    paths = [path.resolve(strict=True) for path in PROMOTION_TOOL_PATHS.values()]
    relative = [path.relative_to(WORKSPACE.resolve(strict=True)).as_posix() for path in paths]
    tracked = subprocess.run(
        ["git", "ls-files", "--", *relative], cwd=WORKSPACE,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    if tracked.returncode != 0 or set(tracked.stdout.splitlines()) != set(relative):
        raise PromotionError("promotion/checker/builder sources are not all committed and tracked")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *relative],
        cwd=WORKSPACE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, timeout=30, check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise PromotionError(
            "promotion/checker/builder sources must be committed-clean before promotion"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False,
    )
    if commit.returncode != 0:
        raise PromotionError("cannot resolve promotion-tooling Git commit")
    inventory = []
    for (name, _), path, relative_path in zip(PROMOTION_TOOL_PATHS.items(), paths, relative):
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative_path}"], cwd=WORKSPACE,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30, check=False,
        )
        if blob.returncode != 0:
            raise PromotionError(f"cannot resolve committed blob for {relative_path}")
        inventory.append(
            {
                "role": name,
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "git_blob": blob.stdout.strip(),
            }
        )
    return {
        "schema": "paper4-p4r1-promotion-tooling-v1",
        "git_commit": commit.stdout.strip(),
        "git_state": "all promotion/checker/builder sources tracked and clean",
        "inventory": inventory,
    }


def regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"missing or unsafe file: {path}")
    return path


def git_text(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=WORKSPACE, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=30, check=False,
    )
    if result.returncode != 0:
        raise PromotionError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def verify_current_analysis_freeze(
    analysis_by_path: dict[str, dict[str, Any]],
) -> None:
    relative_paths = sorted(analysis_by_path)
    tracked = set(git_text("ls-files", "--", *relative_paths).splitlines())
    if tracked != set(relative_paths):
        raise PromotionError(
            "result-interpreting sources are not all committed/tracked at promotion"
        )
    status = git_text(
        "status", "--porcelain=v1", "--untracked-files=all", "--", *relative_paths
    )
    if status:
        raise PromotionError(
            f"result-interpreting sources are not committed-clean at promotion:\n{status}"
        )
    for relative, item in analysis_by_path.items():
        current = regular(WORKSPACE / relative)
        if (
            current.stat().st_size != item["bytes"]
            or sha256_file(current) != item["sha256"]
            or git_text("rev-parse", f"HEAD:{relative}") != item["git_blob"]
        ):
            raise PromotionError(
                f"result-interpreting source changed after the prospective freeze: {relative}"
            )


def private_file(path: Path) -> Path:
    regular(path)
    if path.stat().st_mode & 0o777 != 0o600:
        raise PromotionError(f"private evidence file must have mode 0600: {path}")
    return path


def private_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir() or path.stat().st_mode & 0o777 != 0o700:
        raise PromotionError(f"private evidence directory must have mode 0700: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(regular(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PromotionError(f"expected JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in regular(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise PromotionError(f"non-object row: {path}")
    return rows


def sanitize_line(line: str, private_paths: list[str]) -> str:
    sanitized = line
    for private_path in private_paths:
        sanitized = sanitized.replace(private_path, Path(private_path).name)
    return sanitized


def local_private_identifiers() -> tuple[bytes, ...]:
    values: set[str] = set()
    for candidate in (
        getpass.getuser(), socket.gethostname(), os.uname().nodename,
        Path.home().name, os.environ.get("USER"), os.environ.get("LOGNAME"),
    ):
        if candidate and candidate.lower() not in {"root", "user", "unknown"}:
            values.add(candidate)
    return tuple(sorted((value.encode("utf-8").lower() for value in values), key=len, reverse=True))


def scan_public_payload(payload: bytes, description: str) -> None:
    for pattern in FORBIDDEN_PUBLIC:
        if pattern.search(payload):
            raise PromotionError(f"public privacy scan failed for {description}: {pattern.pattern!r}")
    lowered = payload.lower()
    for identifier in local_private_identifiers():
        if identifier in lowered:
            raise PromotionError(
                f"public privacy scan found a local username/hostname identifier in {description}"
            )


def public_scan(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise PromotionError(f"unsafe public output: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        scan_public_payload(relative, f"output name {path.relative_to(root)}")
        if not path.is_file():
            continue
        payload = path.read_bytes()
        scan_public_payload(payload, str(path.relative_to(root)))


def copy_private(source: Path, destination: Path) -> None:
    regular(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def validate_and_promote(
    campaign: Path, model_manifest: Path, dataset: Path, destination: Path,
) -> dict[str, Any]:
    campaign = private_directory(campaign.resolve(strict=True))
    model_manifest = private_file(model_manifest.resolve(strict=True))
    private_file(dataset.resolve(strict=True))
    gold_by_id = validate_dataset(dataset.resolve(strict=True))
    promotion_tooling = promotion_tooling_receipt()
    if destination.exists() or destination.is_symlink():
        raise PromotionError(f"refusing to overwrite destination: {destination}")
    campaign_receipt_path = private_file(campaign / "campaign-receipt.json")
    campaign_receipt = read_json(campaign_receipt_path)
    if set(campaign_receipt) != {
        "schema", "model_manifest_sha256", "cell_count", "row_count",
        "dataset_sha256", "sampler_contract_sha256", "protocol_sources",
        "prospective_plan", "prospective_plan_approval", "host_receipt",
        "post_model_verification",
        "toolchain_receipt", "cells",
    }:
        raise PromotionError("private campaign receipt keys differ from the exact P4R1 schema")
    (
        plan_path, prospective_plan, plan_approval_path, plan_approval,
        toolchain_path, toolchain,
    ) = validate_plan_and_toolchain(campaign, campaign_receipt)
    host_record = campaign_receipt.get("host_receipt")
    if (
        not isinstance(host_record, dict)
        or set(host_record) != {"path", "sha256"}
        or host_record.get("path") != "host-receipt.json"
        or not isinstance(host_record.get("sha256"), str)
        or SHA256.fullmatch(host_record["sha256"]) is None
    ):
        raise PromotionError("campaign has no exact host-receipt binding")
    host_path = private_file(campaign / "host-receipt.json")
    if sha256_file(host_path) != host_record["sha256"]:
        raise PromotionError("campaign host receipt bytes differ")
    private_host = read_json(host_path)
    validate_private_host(private_host)
    post_record = campaign_receipt.get("post_model_verification")
    if (
        not isinstance(post_record, dict)
        or set(post_record) != {"path", "sha256"}
        or post_record.get("path") != "post-model-verification.json"
        or not isinstance(post_record.get("sha256"), str)
        or SHA256.fullmatch(post_record["sha256"]) is None
    ):
        raise PromotionError("campaign has no exact post-campaign model verification binding")
    post_path = private_file(campaign / "post-model-verification.json")
    if sha256_file(post_path) != post_record["sha256"]:
        raise PromotionError("campaign post-model verification bytes differ")
    private_post_model_verification = read_json(post_path)
    analysis_by_path = {
        item["path"]: item for item in prospective_plan["analysis_source_inventory"]
    }
    for item in promotion_tooling["inventory"]:
        frozen = analysis_by_path.get(item["path"])
        if frozen is None or any(
            frozen[key] != item[key] for key in ("bytes", "sha256", "git_blob")
        ):
            raise PromotionError(
                f"promotion source {item['path']} differs from the prospective analysis freeze"
            )
    manifest_raw = model_manifest.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_raw)
    manifest = json.loads(manifest_raw)
    if (
        campaign_receipt.get("schema") != "paper4-validation-campaign-receipt-v1"
        or campaign_receipt.get("model_manifest_sha256") != manifest_sha256
        or campaign_receipt.get("dataset_sha256") != DATASET_SHA256
        or campaign_receipt.get("cell_count") != 6
        or campaign_receipt.get("row_count") != 600
    ):
        raise PromotionError("campaign receipt identity/counts do not match P4R1")
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
        raise PromotionError("private model manifest has an unknown schema or total")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != 6
        or any(
            not isinstance(item, dict)
            or set(item)
            != {"alias", "family", "filename", "path", "bytes", "sha256", "stat"}
            for item in artifacts
        )
    ):
        raise PromotionError("private model manifest artifact inventory is malformed")
    artifact_by_alias = {
        item.get("alias"): item for item in artifacts or [] if isinstance(item, dict)
    }
    if set(artifact_by_alias) != set(MATRIX):
        raise PromotionError("private model manifest differs from the P4R1 six-artifact matrix")
    if len(artifact_by_alias) != len(artifacts):
        raise PromotionError("private model manifest contains a duplicate artifact alias")
    validate_private_post_model_verification(
        private_post_model_verification, manifest_sha256, artifact_by_alias,
        sha256_file(toolchain_path), toolchain,
    )
    private_paths: list[str] = []
    for alias, expected in MATRIX.items():
        item = artifact_by_alias[alias]
        stat_record = item.get("stat")
        if (
            (item.get("filename"), item.get("sha256")) != expected[:2]
            or item.get("bytes") != P4R1_BYTES[alias]
            or not isinstance(item.get("family"), str)
            or not isinstance(stat_record, dict)
            or set(stat_record) != {"device", "inode", "mtime_ns"}
            or any(
                not isinstance(stat_record[key], int)
                or isinstance(stat_record[key], bool)
                or stat_record[key] < 0
                for key in stat_record
            )
        ):
            raise PromotionError(f"{alias}: artifact filename/hash differs")
        if (
            not isinstance(item.get("path"), str)
            or not Path(item["path"]).is_absolute()
            or Path(item["path"]).name != item["filename"]
        ):
            raise PromotionError(f"{alias}: private manifest has no exact model path")
        private_paths.append(item["path"])
    if len(set(private_paths)) != 6:
        raise PromotionError("private model manifest contains duplicate artifact paths")

    declared_cells = campaign_receipt.get("cells")
    if not isinstance(declared_cells, list) or len(declared_cells) != 6:
        raise PromotionError("campaign receipt has no six-cell inventory")
    declared_by_alias = {item.get("alias"): item for item in declared_cells if isinstance(item, dict)}
    if set(declared_by_alias) != set(MATRIX):
        raise PromotionError("campaign receipt aliases differ from P4R1")
    protocol_sources = campaign_receipt.get("protocol_sources")
    if not isinstance(protocol_sources, dict):
        raise PromotionError("campaign receipt has no protocol source hashes")
    for key, path in SOURCE_PATHS.items():
        if protocol_sources.get(key) != sha256_file(path):
            raise PromotionError(f"campaign was not produced by current {path.name}")

    temporary_parent = destination.parent.resolve()
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=temporary_parent))
    try:
        private_root = temporary / "private"
        public_root = temporary / "public-candidate"
        private_root.mkdir()
        public_root.mkdir()
        copy_private(model_manifest, private_root / "model-manifest.json")
        copy_private(campaign_receipt_path, private_root / "campaign-receipt.json")
        copy_private(plan_path, private_root / "protocol-plan.json")
        copy_private(plan_approval_path, private_root / "protocol-plan-approval.json")
        copy_private(host_path, private_root / "host-receipt.json")
        copy_private(post_path, private_root / "post-model-verification.json")
        copy_private(toolchain_path, private_root / "toolchain-receipt.json")
        public_rows: list[dict[str, Any]] = []
        public_cells: list[dict[str, Any]] = []

        for alias, expected in MATRIX.items():
            log_dir = campaign / f"{alias}-server"
            private_directory(log_dir)
            lifecycle_startup = validate_private_lifecycle(
                log_dir, alias, expected, manifest_sha256, toolchain
            )
            cell_receipt_path = private_file(log_dir / "cell-receipt.json")
            declared = declared_by_alias[alias]
            if declared.get("path") != f"{alias}-server/cell-receipt.json" or declared.get("sha256") != sha256_file(cell_receipt_path):
                raise PromotionError(f"{alias}: campaign receipt hash/path differs")
            receipt = read_json(cell_receipt_path)
            if (
                set(receipt)
                != {
                    "schema", "alias", "row_count", "model_manifest_sha256",
                    "dataset_sha256", "sampler_contract_sha256",
                    "scores_recomputed_from_raw_response_and_pinned_gold",
                    "api_error_telemetry_preserved_and_validated",
                    "host_receipt_sha256",
                    "shutdown_verified", "engine_executable", "requested",
                    "observed_placement_log_lines", "observed_offloaded_layers",
                    "observed_gpu_process", "protocol_sources", "files",
                }
                or
                receipt.get("schema") != "paper4-validation-cell-receipt-v1"
                or receipt.get("alias") != alias
                or receipt.get("row_count") != 100
                or receipt.get("model_manifest_sha256") != manifest_sha256
                or receipt.get("dataset_sha256") != DATASET_SHA256
                or receipt.get("sampler_contract_sha256")
                != campaign_receipt["sampler_contract_sha256"]
                or receipt.get("requested") != {"context_tokens": expected[3], "gpu_layers": expected[2]}
                or receipt.get("protocol_sources") != protocol_sources
                or receipt.get("engine_executable") != lifecycle_startup["launcher"]
                or receipt.get("scores_recomputed_from_raw_response_and_pinned_gold") is not True
                or receipt.get("api_error_telemetry_preserved_and_validated") is not True
                or receipt.get("host_receipt_sha256") != host_record["sha256"]
                or receipt.get("shutdown_verified") is not True
            ):
                raise PromotionError(f"{alias}: cell receipt differs from P4R1")
            observed_layers = receipt.get("observed_offloaded_layers")
            if (
                not isinstance(observed_layers, dict)
                or not isinstance(observed_layers.get("gpu"), int)
                or isinstance(observed_layers.get("gpu"), bool)
                or not isinstance(observed_layers.get("total"), int)
                or isinstance(observed_layers.get("total"), bool)
                or observed_layers.get("readback")
                != f"{observed_layers.get('gpu')}/{observed_layers.get('total')}"
                or (expected[2] != "auto" and observed_layers.get("gpu") != int(expected[2]))
            ):
                raise PromotionError(f"{alias}: cell receipt lacks an exact placement readback")
            result_path = regular(campaign / f"{alias}.jsonl")
            files = receipt.get("files")
            if not isinstance(files, dict):
                raise PromotionError(f"{alias}: receipt has no file inventory")
            source_by_key = {
                "result": result_path,
                "stdout": log_dir / "stdout.log",
                "stderr": log_dir / "stderr.log",
                "model_identity": log_dir / "model-identity.json",
                "startup_identity": log_dir / "startup-identity.json",
                "completion_identity": log_dir / "completion-identity.json",
                "shutdown_identity": log_dir / "shutdown-identity.json",
            }
            for key, source in source_by_key.items():
                private_file(source)
                record = files.get(key)
                if not isinstance(record, dict) or record.get("sha256") != sha256_file(source) or record.get("bytes") != source.stat().st_size:
                    raise PromotionError(f"{alias}: {key} differs from cell receipt")
            rows = read_rows(result_path)
            if len(rows) != 100 or [row.get("id") for row in rows] != EXPECTED_IDS:
                raise PromotionError(f"{alias}: rows do not have the pinned ID order")
            sampler_hash = receipt.get("sampler_contract_sha256")
            for row in rows:
                validate_private_row(
                    row, alias, gold_by_id[row["id"]], manifest_sha256,
                    sampler_hash, sha256_file(log_dir / "startup-identity.json"),
                    protocol_sources,
                )
            cell_private = private_root / "cells" / alias
            copy_private(result_path, cell_private / "gsm8k.jsonl")
            for filename in PRIVATE_FILENAMES:
                copy_private(log_dir / filename, cell_private / filename)

            for row in rows:
                public_row: dict[str, Any] = {
                    key: row[key]
                    for key in (
                        "id", "label", "correct", "wall_s", "identity_guard_wall_s",
                        "completion_tokens", "thinking_tokens_estimate", "truncated",
                        "model_id", "model_manifest_sha256", "sampler_seed",
                        "dataset_sha256", "harness_sha256", "server_guard_sha256",
                        "sampler_contract_sha256", "startup_identity_sha256",
                        "chat_template_kwargs_dropped",
                    )
                }
                public_row["api_error_recorded"] = row["api_error"] is not None
                validate_public_row(public_row)
                public_rows.append(public_row)
            gpu = receipt.get("observed_gpu_process")
            if not isinstance(gpu, dict) or set(gpu) != {"startup", "completion"}:
                raise PromotionError(f"{alias}: receipt lacks two GPU placement snapshots")
            for phase in ("startup", "completion"):
                snapshot = gpu[phase]
                if (
                    not isinstance(snapshot, dict)
                    or not isinstance(snapshot.get("used_gpu_memory_mib"), int)
                    or isinstance(snapshot.get("used_gpu_memory_mib"), bool)
                    or snapshot["used_gpu_memory_mib"] < 0
                    or not isinstance(snapshot.get("gpu_uuid"), str)
                ):
                    raise PromotionError(f"{alias}: malformed private GPU placement snapshot")
            public_cells.append(
                {
                    "alias": alias,
                    "model": {
                        "filename": artifact_by_alias[alias]["filename"],
                        "bytes": artifact_by_alias[alias]["bytes"],
                        "sha256": artifact_by_alias[alias]["sha256"],
                    },
                    "requested": receipt["requested"],
                    "engine_executable_sha256": lifecycle_startup["launcher"]["sha256"],
                    "observed_offloaded_layers": observed_layers,
                    "observed_gpu_memory_mib": {
                        phase: gpu[phase]["used_gpu_memory_mib"]
                        for phase in ("startup", "completion")
                    },
                    "host_receipt_sha256": host_record["sha256"],
                    "n": 100,
                    "correct": sum(bool(row.get("correct")) for row in rows),
                    "truncated": sum(bool(row.get("truncated")) for row in rows),
                    "api_errors": sum(row["api_error"] is not None for row in rows),
                    "private_scores_recomputed": True,
                    "private_api_error_telemetry_validated": True,
                    "private_cell_receipt_sha256": sha256_file(cell_receipt_path),
                }
            )

        public_cell_inventory = []
        for cell in sorted(public_cells, key=lambda item: item["alias"]):
            cell_path = public_root / "cells" / cell["alias"] / "receipt.json"
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            cell_payload = canonical_bytes(
                {"schema": "paper4-p4r1-public-cell-receipt-v1", **cell}
            )
            cell_path.write_bytes(cell_payload)
            public_cell_inventory.append(
                {
                    "alias": cell["alias"],
                    "path": f"cells/{cell['alias']}/receipt.json",
                    "sha256": sha256_bytes(cell_payload),
                }
            )
        public_rows_payload = b"".join(canonical_bytes(row) for row in public_rows)
        (public_root / "primary_gsm8k.jsonl").write_bytes(public_rows_payload)
        public_plan_payload = canonical_bytes(prospective_plan)
        (public_root / "protocol-plan.json").write_bytes(public_plan_payload)
        public_approval_payload = canonical_bytes(plan_approval)
        (public_root / "protocol-plan-approval.json").write_bytes(public_approval_payload)
        public_post_model_verification = {
            "schema": "paper4-p4r1-public-post-campaign-model-verification-v1",
            "model_manifest_sha256": manifest_sha256,
            "artifact_count": 6,
            "total_bytes": sum(P4R1_BYTES.values()),
            "artifacts": [
                {
                    "alias": alias,
                    "filename": artifact_by_alias[alias]["filename"],
                    "bytes": artifact_by_alias[alias]["bytes"],
                    "sha256": artifact_by_alias[alias]["sha256"],
                }
                for alias in MATRIX
            ],
            "cuda_runtime": {
                "schema": "paper4-p4r1-public-post-campaign-cuda-runtime-v1",
                "toolkit_version": "12.6.85",
                "stable_loader_identity_matches_pre_campaign": True,
                "libraries": [
                    {
                        "soname": item["soname"],
                        "bytes": item["bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in private_post_model_verification[
                        "post_campaign_cuda_runtime"
                    ]["libraries"]
                ],
                "verification": "privacy-sanitized post-campaign loader re-resolution and full CUDA runtime library rehash; local paths omitted",
            },
            "private_post_verification_sha256": post_record["sha256"],
            "verification": "all six artifact contents and the loader-resolved CUDA runtime fully SHA-256-verified after all timed cells and shutdowns; private filesystem stat identities and local paths omitted",
        }
        public_post_payload = canonical_bytes(public_post_model_verification)
        (public_root / "post-model-verification.json").write_bytes(public_post_payload)
        public_host = {
            "schema": "paper4-p4r1-public-host-receipt-v1",
            "cpu": private_host["cpu"],
            "memory": private_host["memory"],
            "operating_system": private_host["operating_system"],
            "gpu": {
                key: private_host["gpu"][key]
                for key in ("name", "memory_total_mib", "power_limits", "driver_version")
            },
            "storage": {
                "path_roles_on_same_device": sorted(
                    item["role"] for item in private_host["storage"]["verified_paths"]
                ),
                "filesystem": private_host["storage"]["filesystem"],
                "model": private_host["storage"]["model"],
                "transport": private_host["storage"]["transport"],
                "rotational": private_host["storage"]["rotational"],
            },
            "private_host_receipt_sha256": host_record["sha256"],
            "verification": "privacy-sanitized exact pre-campaign host readback; GPU UUID and local paths omitted",
        }
        public_host_payload = canonical_bytes(public_host)
        (public_root / "host-receipt.json").write_bytes(public_host_payload)
        private_executable = toolchain["executable"]
        private_cache = toolchain["observed_cmake_cache"]
        private_cuda_runtime = toolchain["observed_cuda_runtime"]
        public_toolchain = {
            "schema": "paper4-p4r1-public-toolchain-receipt-v1",
            "executable": {
                "filename": private_executable["filename"],
                "bytes": private_executable["bytes"],
                "sha256": private_executable["sha256"],
            },
            "version_command": toolchain["version_command"],
            "observed_build_checkout": {
                "git_commit": toolchain["observed_build_checkout"]["git_commit"],
                "git_status": toolchain["observed_build_checkout"]["git_status"],
            },
            "observed_cmake_cache": {
                "sha256": private_cache["sha256"],
                "settings": private_cache["settings"],
                "tool_versions": private_cache["tool_versions"],
                "tool_sha256": private_cache["tool_sha256"],
            },
            "observed_driver": toolchain["observed_driver"],
            "observed_cuda_runtime": {
                "schema": "paper4-p4r1-public-cuda-runtime-resolution-v1",
                "toolkit_version": "12.6.85",
                "ld_library_path_priority_verified": True,
                "libraries": [
                    {
                        "soname": item["soname"],
                        "bytes": item["bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in private_cuda_runtime["libraries"]
                ],
                "verification": "privacy-sanitized exact loader-resolved CUDA runtime hashes; local paths omitted",
            },
            "historical_build_association_reference": toolchain[
                "historical_build_association_reference"
            ],
            "verification": toolchain["verification"],
        }
        public_toolchain_payload = canonical_bytes(public_toolchain)
        (public_root / "toolchain-receipt.json").write_bytes(public_toolchain_payload)
        public_validation = {
            "schema": "paper4-p4r1-public-validation-v1",
            "scope": "privacy-sanitized identity and outcome receipt; exact local paths, GPU device identifiers, model output text, and server logs omitted",
            "model_manifest_sha256": manifest_sha256,
            "dataset_sha256": DATASET_SHA256,
            "sampler_contract_sha256": campaign_receipt["sampler_contract_sha256"],
            "protocol_sources": protocol_sources,
            "row_count": 600,
            "cells": public_cell_inventory,
            "primary_rows_sha256": sha256_bytes(public_rows_payload),
            "private_scores_recomputed_from_preserved_output_and_pinned_gold": True,
            "private_api_error_telemetry_preserved_and_validated": True,
            "promotion_source_sha256": sha256_file(Path(__file__)),
            "promotion_tooling": promotion_tooling,
            "prospective_plan": {
                "path": "protocol-plan.json",
                "sha256": sha256_bytes(public_plan_payload),
                "git_commit": prospective_plan["git_commit"],
            },
            "prospective_plan_approval": {
                "path": "protocol-plan-approval.json",
                "sha256": sha256_bytes(public_approval_payload),
                "actor": plan_approval["actor"],
                "plan_sha256": sha256_bytes(public_plan_payload),
            },
            "host_receipt": {
                "path": "host-receipt.json",
                "sha256": sha256_bytes(public_host_payload),
                "private_receipt_sha256": host_record["sha256"],
            },
            "post_model_verification": {
                "path": "post-model-verification.json",
                "sha256": sha256_bytes(public_post_payload),
                "private_receipt_sha256": post_record["sha256"],
            },
            "toolchain_receipt": {
                "path": "toolchain-receipt.json",
                "sha256": sha256_bytes(public_toolchain_payload),
                "executable_sha256": private_executable["sha256"],
                "llama_cpp_commit": toolchain["observed_build_checkout"]["git_commit"],
            },
        }
        (public_root / "validation-receipt.json").write_bytes(canonical_bytes(public_validation))
        public_scan(public_root)

        inventory = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                inventory.append(
                    {
                        "path": path.relative_to(temporary).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        promotion = {
            "schema": "paper4-p4r1-promotion-receipt-v1",
            "source_campaign_receipt_sha256": sha256_file(campaign_receipt_path),
            "model_manifest_sha256": manifest_sha256,
            "private_evidence_preserved": True,
            "historical_files_overwritten": False,
            "public_privacy_scan": "pass",
            "promotion_tooling": promotion_tooling,
            "files": inventory,
        }
        promotion_payload = canonical_bytes(promotion)
        (temporary / "promotion-receipt.json").write_bytes(promotion_payload)
        # Private evidence never becomes group/world-readable. The public
        # candidate is readable only through an explicit later export because
        # the enclosing promotion directory remains owner-only.
        for path in sorted(private_root.rglob("*"), reverse=True):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        os.chmod(private_root, 0o700)
        for path in sorted(public_root.rglob("*"), reverse=True):
            os.chmod(path, 0o755 if path.is_dir() else 0o444)
        os.chmod(public_root, 0o755)
        os.chmod(temporary / "promotion-receipt.json", 0o600)
        os.chmod(temporary, 0o700)
        os.replace(temporary, destination)
        return {
            "destination": str(destination),
            "model_manifest_sha256": manifest_sha256,
            "public_rows_sha256": public_validation["primary_rows_sha256"],
            "promotion_receipt_sha256": sha256_bytes(promotion_payload),
            "file_count": len(inventory) + 1,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    result = validate_and_promote(
        args.campaign, args.model_manifest, args.dataset, args.destination.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
