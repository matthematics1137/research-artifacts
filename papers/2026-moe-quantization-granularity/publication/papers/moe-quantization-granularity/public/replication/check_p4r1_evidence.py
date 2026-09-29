#!/usr/bin/env python3
"""Verify a privacy-sanitized promoted P4R1 candidate (no inference)."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import random
import re
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASET_SHA256 = "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37"
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
SEED_BASE = 4_200_000
EXPECTED_IDS = [f"gsm8k_{index}" for index in sorted(random.Random(42).sample(range(1319), 100))]
MATRIX = {
    "P4R1-MIXTRAL-IQ1M": ("Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf", 10847178368, "7e8d75c23bd4de55582ac91084a0575f081c1de374b693f5ea7ffa6fa66a2e3c", "29"),
    "P4R1-MIXTRAL-IQ2XXS": ("Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf", 12556357248, "db62b07756f50d2f046b0a00dc594f1aa73dba82ebdc182d9f55c96474655f03", "26"),
    "P4R1-MIXTRAL-Q4KM": ("Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf", 28448468608, "7f2f98f301a74b645d0a4c2bd64c98e9811f2dfa26dbae4787c280eb1183632c", "12"),
    "P4R1-QWEN3MOE-IQ1M": ("Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf", 9685733792, "d527a854db2a1582a3ce746a17b1f42d860334ece18d385ede9e2e395058b39e", "auto"),
    "P4R1-QWEN3MOE-IQ2XXS": ("Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf", 10341814688, "aa0b04ac05f71aa94b542b074500b3de956608365e74f0e1d97ae0d6a6380269", "auto"),
    "P4R1-QWEN3MOE-Q4KM": ("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 18556686752, "6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0", "auto"),
}
CUDA_RUNTIME_LIBRARIES = {
    "libcudart.so.12": ("libcudart.so.12.6.77", 716128, "095296727cfb5f51e5b3ae69a3cb88cbd7b6592f314ace0fdfc4e1d758c2aa85"),
    "libcublas.so.12": ("libcublas.so.12.6.4.1", 108244960, "d9343511e1d2ed51e4d65fa94a25cb8f73b0ebc5ae5487da794e18e6950c98dc"),
    "libcublasLt.so.12": ("libcublasLt.so.12.6.4.1", 491106832, "a4ddaed1410acc604815f0e84cdd6c4a70c61cfd2b95dc96360f0799169f0374"),
}
COMPARISONS = {
    "mixtral_iq1_vs_q4": ("P4R1-MIXTRAL-IQ1M", "P4R1-MIXTRAL-Q4KM"),
    "mixtral_iq2_vs_q4": ("P4R1-MIXTRAL-IQ2XXS", "P4R1-MIXTRAL-Q4KM"),
    "qwen_iq1_vs_q4": ("P4R1-QWEN3MOE-IQ1M", "P4R1-QWEN3MOE-Q4KM"),
    "qwen_iq2_vs_q4": ("P4R1-QWEN3MOE-IQ2XXS", "P4R1-QWEN3MOE-Q4KM"),
}
FORBIDDEN = (
    rb"/(?:home|Users)/[^/\s]+", rb"file://", rb"raw_(?:answer|response)",
    rb"scoring_content", rb"expected_answer_number", rb"parsed_answer_number",
    rb"gpu_uuid", rb"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    rb"(?<![0-9])(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}(?![0-9])",
    rb"GPU-[0-9A-F-]{16,}",
    rb'"(?:email|gpu_uuid|host|hostname|user|username)"\s*:',
)
SHA256 = re.compile(r"[0-9a-f]{64}")
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def local_private_identifiers() -> tuple[bytes, ...]:
    values: set[str] = set()
    for candidate in (
        getpass.getuser(), socket.gethostname(), os.uname().nodename,
        Path.home().name, os.environ.get("USER"), os.environ.get("LOGNAME"),
    ):
        if candidate and candidate.lower() not in {"root", "user", "unknown"}:
            values.add(candidate)
    return tuple(value.encode("utf-8").lower() for value in values)


def scan(payload: bytes, description: str) -> None:
    for raw in FORBIDDEN:
        require(re.search(raw, payload, re.IGNORECASE) is None, f"privacy leak in {description}: {raw!r}")
    lowered = payload.lower()
    for identifier in local_private_identifiers():
        require(identifier not in lowered, f"local username/hostname leak in {description}")


def validate_public_row(row: dict[str, Any]) -> None:
    require(set(row) == PUBLIC_ROW_KEYS, "public row keys differ from the exact projection schema")
    require(isinstance(row["id"], str) and re.fullmatch(r"gsm8k_\d+", row["id"]) is not None, "bad row ID")
    require(isinstance(row["label"], str) and row["label"] in MATRIX, "bad row label")
    require(row["model_id"] == row["label"], "row model identity mismatch")
    require(isinstance(row["correct"], bool), "row correct is not Boolean")
    require(isinstance(row["truncated"], bool), "row truncated is not Boolean")
    require(isinstance(row["chat_template_kwargs_dropped"], bool), "row retry marker is not Boolean")
    require(isinstance(row["api_error_recorded"], bool), "row API-error marker is not Boolean")
    require(numeric(row["wall_s"]) and row["wall_s"] >= 0, "row request time is invalid")
    require(
        numeric(row["identity_guard_wall_s"]) and row["identity_guard_wall_s"] >= 0,
        "row guard time is invalid",
    )
    require(
        isinstance(row["thinking_tokens_estimate"], int)
        and not isinstance(row["thinking_tokens_estimate"], bool)
        and row["thinking_tokens_estimate"] >= 0,
        "row thinking-token estimate is invalid",
    )
    require(
        row["completion_tokens"] is None
        or isinstance(row["completion_tokens"], int)
        and not isinstance(row["completion_tokens"], bool)
        and row["completion_tokens"] >= 0,
        "row completion-token count is invalid",
    )
    require(
        isinstance(row["sampler_seed"], int) and not isinstance(row["sampler_seed"], bool),
        "row sampler seed is invalid",
    )
    for key in (
        "model_manifest_sha256", "startup_identity_sha256", "dataset_sha256",
        "harness_sha256", "server_guard_sha256", "sampler_contract_sha256",
    ):
        require(isinstance(row[key], str) and SHA256.fullmatch(row[key]) is not None, f"bad {key}")


def wilson(correct: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = correct / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    result: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * values[key]))
        result[key] = running
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def paired_table(low: list[dict[str, Any]], high: list[dict[str, Any]]) -> dict[str, int]:
    low_map = {row["id"]: bool(row["correct"]) for row in low}
    high_map = {row["id"]: bool(row["correct"]) for row in high}
    require(low_map.keys() == high_map.keys(), "paired item sets differ")
    return {
        "both_correct": sum(low_map[key] and high_map[key] for key in low_map),
        "low_only_correct": sum(low_map[key] and not high_map[key] for key in low_map),
        "high_only_correct": sum(not low_map[key] and high_map[key] for key in low_map),
        "both_wrong": sum(not low_map[key] and not high_map[key] for key in low_map),
    }


def mcnemar(table: dict[str, int]) -> float:
    a, b = table["low_only_correct"], table["high_only_correct"]
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1))
    return min(1.0, 2 * tail / (2**n))


def check(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    expected_files = {
        "validation-receipt.json",
        "primary_gsm8k.jsonl",
        "protocol-plan.json",
        "protocol-plan-approval.json",
        "host-receipt.json",
        "post-model-verification.json",
        "toolchain-receipt.json",
        *{f"cells/{alias}/receipt.json" for alias in MATRIX},
    }
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    require(actual_files == expected_files, "public candidate file inventory differs")
    expected_directories = {"cells", *{f"cells/{alias}" for alias in MATRIX}}
    actual_directories = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    }
    require(actual_directories == expected_directories, "public candidate directory inventory differs")
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"symlink in public candidate: {path}")
        scan(path.relative_to(root).as_posix().encode("utf-8"), f"output name {path.relative_to(root)}")
        if path.is_file():
            payload = path.read_bytes()
            scan(payload, str(path.relative_to(root)))
    campaign = read_json(root / "validation-receipt.json")
    require(
        set(campaign)
        == {
            "schema", "scope", "model_manifest_sha256", "dataset_sha256",
            "sampler_contract_sha256", "protocol_sources", "row_count", "cells",
            "primary_rows_sha256",
            "private_scores_recomputed_from_preserved_output_and_pinned_gold",
            "private_api_error_telemetry_preserved_and_validated",
            "promotion_source_sha256", "promotion_tooling", "prospective_plan",
            "prospective_plan_approval", "host_receipt",
            "post_model_verification", "toolchain_receipt",
        },
        "public campaign receipt keys differ from the exact schema",
    )
    require(campaign.get("schema") == "paper4-p4r1-public-validation-v1", "wrong public campaign schema")
    require(campaign.get("dataset_sha256") == DATASET_SHA256, "wrong dataset hash")
    require(campaign.get("row_count") == 600, "wrong campaign row count")
    require(
        campaign.get("private_scores_recomputed_from_preserved_output_and_pinned_gold") is True,
        "private score/gold recomputation was not attested",
    )
    require(
        campaign.get("private_api_error_telemetry_preserved_and_validated") is True,
        "private exact API-error telemetry validation was not attested",
    )
    for key in (
        "model_manifest_sha256", "sampler_contract_sha256", "primary_rows_sha256",
        "promotion_source_sha256",
    ):
        require(
            isinstance(campaign.get(key), str) and SHA256.fullmatch(campaign[key]) is not None,
            f"malformed campaign {key}",
        )
    protocol_sources = campaign.get("protocol_sources")
    require(
        isinstance(protocol_sources, dict)
        and set(protocol_sources)
        == {"harness_sha256", "server_guard_sha256", "run_one_sha256", "run_matrix_sha256"},
        "campaign protocol-source inventory differs",
    )
    require(
        all(isinstance(value, str) and SHA256.fullmatch(value) is not None for value in protocol_sources.values()),
        "campaign protocol-source hash is malformed",
    )
    tooling = campaign.get("promotion_tooling")
    require(
        isinstance(tooling, dict)
        and set(tooling) == {"schema", "git_commit", "git_state", "inventory"}
        and tooling.get("schema") == "paper4-p4r1-promotion-tooling-v1"
        and re.fullmatch(r"[0-9a-f]{40}", str(tooling.get("git_commit"))) is not None
        and tooling.get("git_state")
        == "all promotion/checker/builder sources tracked and clean"
        and isinstance(tooling.get("inventory"), list),
        "promotion-tooling receipt is malformed",
    )
    tooling_by_role = {
        item.get("role"): item for item in tooling["inventory"] if isinstance(item, dict)
    }
    require(
        set(tooling_by_role)
        == {"promoter", "public_checker", "claims_builder", "manuscript_gate", "claims_analysis", "paper_makefile"}
        and len(tooling_by_role) == len(tooling["inventory"]),
        "promotion-tooling role inventory differs",
    )
    for role, item in tooling_by_role.items():
        require(
            set(item) == {"role", "path", "bytes", "sha256", "git_blob"}
            and isinstance(item.get("path"), str)
            and not item["path"].startswith("/")
            and isinstance(item.get("bytes"), int)
            and not isinstance(item.get("bytes"), bool)
            and item["bytes"] > 0
            and isinstance(item.get("sha256"), str)
            and SHA256.fullmatch(item["sha256"]) is not None
            and re.fullmatch(r"[0-9a-f]{40}", str(item.get("git_blob"))) is not None,
            f"promotion-tooling item is malformed: {role}",
        )
    require(
        tooling_by_role["promoter"]["sha256"] == campaign["promotion_source_sha256"],
        "promotion source hash differs from tooling inventory",
    )
    require(
        tooling_by_role["public_checker"]["sha256"] == sha256_file(Path(__file__)),
        "public checker bytes differ from the promotion tooling inventory",
    )
    plan_record = campaign.get("prospective_plan")
    host_record = campaign.get("host_receipt")
    post_record = campaign.get("post_model_verification")
    toolchain_record = campaign.get("toolchain_receipt")
    require(
        isinstance(plan_record, dict)
        and set(plan_record) == {"path", "sha256", "git_commit"}
        and plan_record.get("path") == "protocol-plan.json"
        and isinstance(plan_record.get("sha256"), str)
        and SHA256.fullmatch(plan_record["sha256"]) is not None
        and re.fullmatch(r"[0-9a-f]{40}", str(plan_record.get("git_commit"))) is not None,
        "prospective-plan inventory record is malformed",
    )
    plan_path = root / "protocol-plan.json"
    require(sha256_file(plan_path) == plan_record["sha256"], "prospective-plan hash mismatch")
    plan = read_json(plan_path)
    require(
        set(plan)
        == {
            "schema", "status", "frozen_at_utc", "git_commit", "git_state",
            "protocol", "dataset_sha256", "artifact_sha256",
            "inference_source_inventory", "analysis_source_inventory",
        }
        and plan.get("schema") == "paper4-p4r1-prospective-plan-v1"
        and plan.get("status")
        == "prospectively frozen corrective exploratory validation; not independent"
        and plan.get("git_state")
        == "all inference/scoring and result-interpreting sources tracked and clean in index and worktree"
        and plan.get("protocol")
        == "publication/papers/moe-quantization-granularity/public/P4R1_PROTOCOL.md"
        and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00",
            str(plan.get("frozen_at_utc")),
        )
        is not None
        and plan.get("git_commit") == plan_record["git_commit"]
        and plan.get("dataset_sha256") == DATASET_SHA256
        and plan.get("artifact_sha256") == [value[2] for value in MATRIX.values()]
        and isinstance(plan.get("inference_source_inventory"), list),
        "prospective plan identity/status differs",
    )
    inference_inventory = plan.get("inference_source_inventory")
    analysis_inventory = plan.get("analysis_source_inventory")
    require(
        isinstance(inference_inventory, list) and isinstance(analysis_inventory, list),
        "prospective plan lacks inference/analysis inventories",
    )
    for inventory, expected_paths, label in (
        (inference_inventory, PLAN_SOURCE_PATHS, "inference"),
        (analysis_inventory, PLAN_ANALYSIS_PATHS, "analysis"),
    ):
        by_path = {
            item.get("path"): item for item in inventory if isinstance(item, dict)
        }
        require(
            len(by_path) == len(inventory) and set(by_path) == expected_paths,
            f"prospective {label}-source inventory differs",
        )
        for path, item in by_path.items():
            require(
                set(item) == {"path", "bytes", "sha256", "git_blob"}
                and isinstance(item.get("bytes"), int)
                and not isinstance(item.get("bytes"), bool)
                and item["bytes"] > 0
                and isinstance(item.get("sha256"), str)
                and SHA256.fullmatch(item["sha256"]) is not None
                and re.fullmatch(r"[0-9a-f]{40}", str(item.get("git_blob"))) is not None,
                f"malformed prospective {label} source: {path}",
            )
    inference_by_path = {item["path"]: item for item in inference_inventory}
    source_binding = {
        "harness_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_gsm8k.py",
        "server_guard_sha256": "publication/papers/moe-quantization-granularity/public/replication/server_guard.py",
        "run_one_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_one.sh",
        "run_matrix_sha256": "publication/papers/moe-quantization-granularity/public/replication/run_matrix.sh",
    }
    require(
        all(
            inference_by_path[path]["sha256"] == protocol_sources[key]
            for key, path in source_binding.items()
        ),
        "prospective inference inventory differs from campaign/cell source hashes",
    )
    analysis_by_path = {item["path"]: item for item in analysis_inventory}
    for role, item in tooling_by_role.items():
        frozen = analysis_by_path.get(item["path"])
        require(
            frozen is not None
            and all(frozen[key] == item[key] for key in ("bytes", "sha256", "git_blob")),
            f"promotion tool differs from prospective analysis freeze: {role}",
        )
    approval_record = campaign.get("prospective_plan_approval")
    require(
        isinstance(approval_record, dict)
        and set(approval_record) == {"path", "sha256", "actor", "plan_sha256"}
        and approval_record.get("path") == "protocol-plan-approval.json"
        and isinstance(approval_record.get("sha256"), str)
        and SHA256.fullmatch(approval_record["sha256"]) is not None
        and isinstance(approval_record.get("actor"), str)
        and bool(approval_record["actor"].strip())
        and approval_record.get("plan_sha256") == plan_record["sha256"],
        "prospective-plan approval inventory is malformed",
    )
    approval_path = root / "protocol-plan-approval.json"
    require(sha256_file(approval_path) == approval_record["sha256"], "plan-approval hash mismatch")
    approval = read_json(approval_path)
    require(
        set(approval)
        == {
            "schema", "operation", "actor", "plan_sha256", "plan_git_commit",
            "approved_at_utc", "method",
        }
        and approval.get("schema") == "paper4-p4r1-plan-approval-v1"
        and approval.get("operation") == "run-p4r1-validation"
        and approval.get("actor") == approval_record["actor"]
        and approval.get("plan_sha256") == plan_record["sha256"]
        and approval.get("plan_git_commit") == plan_record["git_commit"]
        and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00",
            str(approval.get("approved_at_utc")),
        )
        is not None
        and approval.get("method") == "interactive-exact-challenge",
        "prospective-plan approval does not bind the plan",
    )
    require(
        isinstance(host_record, dict)
        and set(host_record) == {"path", "sha256", "private_receipt_sha256"}
        and host_record.get("path") == "host-receipt.json"
        and all(
            isinstance(host_record.get(key), str)
            and SHA256.fullmatch(host_record[key]) is not None
            for key in ("sha256", "private_receipt_sha256")
        ),
        "host inventory record is malformed",
    )
    host_path = root / "host-receipt.json"
    require(sha256_file(host_path) == host_record["sha256"], "host receipt hash mismatch")
    host = read_json(host_path)
    host_gpu = host.get("gpu")
    host_storage = host.get("storage")
    require(
        set(host)
        == {
            "schema", "cpu", "memory", "operating_system", "gpu", "storage",
            "private_host_receipt_sha256", "verification",
        }
        and host.get("schema") == "paper4-p4r1-public-host-receipt-v1"
        and host.get("cpu")
        == {
            "model": "Intel(R) Core(TM) i9-14900HX", "logical_cpus": 32,
            "physical_cores": 24, "sockets": 1,
        }
        and host.get("memory") == {"mem_total_kib": 65445316}
        and host.get("operating_system")
        == {
            "pretty_name": "Ubuntu 24.04.3 LTS",
            "kernel": "Linux 7.0.0-29-generic x86_64",
        }
        and host_gpu
        == {
            "name": "NVIDIA GeForce RTX 4080 Laptop GPU",
            "memory_total_mib": 12282,
            "power_limits": {
                "current_w": 80.0, "requested_w": 80.0, "default_w": 80.0,
            },
            "driver_version": "580.167.08",
        }
        and host_storage
        == {
            "path_roles_on_same_device": [
                "mixtral_models", "qwen_models", "workspace",
            ],
            "filesystem": "ext4",
            "model": "WD PC SN740 SDDPNQE-2T00-1102",
            "transport": "nvme",
            "rotational": 0,
        }
        and host.get("private_host_receipt_sha256")
        == host_record["private_receipt_sha256"]
        and host.get("verification")
        == "privacy-sanitized exact pre-campaign host readback; GPU UUID and local paths omitted",
        "public host receipt does not prove the exact privacy-sanitized P4R1 host",
    )
    require(
        isinstance(post_record, dict)
        and set(post_record) == {"path", "sha256", "private_receipt_sha256"}
        and post_record.get("path") == "post-model-verification.json"
        and all(
            isinstance(post_record.get(key), str)
            and SHA256.fullmatch(post_record[key]) is not None
            for key in ("sha256", "private_receipt_sha256")
        ),
        "post-campaign model verification inventory is malformed",
    )
    post_path = root / "post-model-verification.json"
    require(
        sha256_file(post_path) == post_record["sha256"],
        "post-campaign model verification hash mismatch",
    )
    post = read_json(post_path)
    post_artifacts = post.get("artifacts")
    require(
        set(post)
        == {
            "schema", "model_manifest_sha256", "artifact_count", "total_bytes",
            "artifacts", "cuda_runtime", "private_post_verification_sha256",
            "verification",
        }
        and post.get("schema")
        == "paper4-p4r1-public-post-campaign-model-verification-v1"
        and post.get("model_manifest_sha256") == campaign.get("model_manifest_sha256")
        and post.get("artifact_count") == 6
        and post.get("total_bytes") == sum(value[1] for value in MATRIX.values())
        and post.get("private_post_verification_sha256")
        == post_record["private_receipt_sha256"]
        and post.get("verification")
        == "all six artifact contents and the loader-resolved CUDA runtime fully SHA-256-verified after all timed cells and shutdowns; private filesystem stat identities and local paths omitted"
        and isinstance(post_artifacts, list)
        and len(post_artifacts) == 6,
        "public post-campaign model verification schema/count differs",
    )
    post_by_alias = {
        item.get("alias"): item for item in post_artifacts if isinstance(item, dict)
    }
    require(
        len(post_by_alias) == 6 and set(post_by_alias) == set(MATRIX),
        "public post-campaign artifact aliases differ",
    )
    for alias, expected in MATRIX.items():
        require(
            post_by_alias[alias]
            == {
                "alias": alias, "filename": expected[0], "bytes": expected[1],
                "sha256": expected[2],
            },
            f"{alias}: post-campaign artifact identity differs",
        )
    post_cuda = post.get("cuda_runtime")
    require(
        isinstance(post_cuda, dict)
        and set(post_cuda)
        == {
            "schema", "toolkit_version", "stable_loader_identity_matches_pre_campaign",
            "libraries", "verification",
        }
        and post_cuda.get("schema")
        == "paper4-p4r1-public-post-campaign-cuda-runtime-v1"
        and post_cuda.get("toolkit_version") == "12.6.85"
        and post_cuda.get("stable_loader_identity_matches_pre_campaign") is True
        and post_cuda.get("verification")
        == "privacy-sanitized post-campaign loader re-resolution and full CUDA runtime library rehash; local paths omitted",
        "public post-campaign CUDA runtime receipt is malformed",
    )
    post_runtime_libraries = post_cuda.get("libraries")
    post_runtime_by_soname = {
        item.get("soname"): item
        for item in post_runtime_libraries
        if isinstance(item, dict)
    } if isinstance(post_runtime_libraries, list) else {}
    require(
        len(post_runtime_by_soname) == 3
        and set(post_runtime_by_soname) == set(CUDA_RUNTIME_LIBRARIES),
        "public post-campaign CUDA runtime inventory is incomplete or duplicated",
    )
    for soname, (_filename, byte_count, digest) in CUDA_RUNTIME_LIBRARIES.items():
        require(
            post_runtime_by_soname[soname]
            == {"soname": soname, "bytes": byte_count, "sha256": digest},
            f"{soname}: post-campaign CUDA runtime identity differs",
        )
    require(
        isinstance(toolchain_record, dict)
        and set(toolchain_record)
        == {"path", "sha256", "executable_sha256", "llama_cpp_commit"}
        and toolchain_record.get("path") == "toolchain-receipt.json"
        and all(
            isinstance(toolchain_record.get(key), str)
            and SHA256.fullmatch(toolchain_record[key]) is not None
            for key in ("sha256", "executable_sha256")
        )
        and toolchain_record.get("llama_cpp_commit")
        == "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3",
        "toolchain inventory record is malformed",
    )
    toolchain_path = root / "toolchain-receipt.json"
    require(sha256_file(toolchain_path) == toolchain_record["sha256"], "toolchain hash mismatch")
    toolchain = read_json(toolchain_path)
    executable = toolchain.get("executable")
    cache = toolchain.get("observed_cmake_cache")
    driver = toolchain.get("observed_driver")
    cuda_runtime = toolchain.get("observed_cuda_runtime")
    require(
        toolchain.get("schema") == "paper4-p4r1-public-toolchain-receipt-v1"
        and isinstance(executable, dict)
        and set(executable) == {"filename", "bytes", "sha256"}
        and executable.get("sha256") == toolchain_record["executable_sha256"]
        and isinstance(executable.get("bytes"), int)
        and not isinstance(executable.get("bytes"), bool)
        and executable["bytes"] > 0
        and toolchain.get("observed_build_checkout")
        == {
            "git_commit": toolchain_record["llama_cpp_commit"],
            "git_status": "clean including untracked files",
        }
        and isinstance(cache, dict)
        and cache.get("settings")
        == {
            "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "89",
            "GGML_BLAS": "OFF", "GGML_CUDA": "ON", "GGML_NATIVE": "ON",
            "GGML_OPENMP": "ON", "LLAMA_CURL": "OFF",
        }
        and isinstance(driver, dict)
        and driver.get("driver_version") == "580.167.08"
        and driver.get("driver_reported_cuda_compatibility") == "13.0"
        and isinstance(cuda_runtime, dict)
        and set(cuda_runtime)
        == {
            "schema", "toolkit_version", "ld_library_path_priority_verified",
            "libraries", "verification",
        }
        and cuda_runtime.get("schema")
        == "paper4-p4r1-public-cuda-runtime-resolution-v1"
        and cuda_runtime.get("toolkit_version") == "12.6.85"
        and cuda_runtime.get("ld_library_path_priority_verified") is True
        and cuda_runtime.get("verification")
        == "privacy-sanitized exact loader-resolved CUDA runtime hashes; local paths omitted"
        and "035e227" in (
            str(toolchain.get("version_command", {}).get("stdout"))
            + str(toolchain.get("version_command", {}).get("stderr"))
        ),
        "public toolchain receipt does not prove the frozen build/readbacks",
    )
    runtime_libraries = cuda_runtime.get("libraries")
    runtime_by_soname = {
        item.get("soname"): item
        for item in runtime_libraries
        if isinstance(item, dict)
    } if isinstance(runtime_libraries, list) else {}
    require(
        len(runtime_by_soname) == 3
        and set(runtime_by_soname) == set(CUDA_RUNTIME_LIBRARIES),
        "public CUDA runtime library inventory is incomplete or duplicated",
    )
    for soname, (_filename, byte_count, digest) in CUDA_RUNTIME_LIBRARIES.items():
        require(
            runtime_by_soname[soname]
            == {
                "soname": soname, "bytes": byte_count, "sha256": digest,
            },
            f"{soname}: public CUDA runtime identity differs",
        )
    inventory = campaign.get("cells")
    require(isinstance(inventory, list) and len(inventory) == 6, "expected six public cell receipts")
    receipts: dict[str, dict[str, Any]] = {}
    for item in inventory:
        require(
            isinstance(item, dict)
            and set(item) == {"alias", "path", "sha256"}
            and item.get("alias") in MATRIX,
            "unknown or malformed cell inventory item",
        )
        require(isinstance(item.get("sha256"), str) and SHA256.fullmatch(item["sha256"]) is not None, "bad cell hash")
        path = root / str(item["path"])
        require(path.resolve().is_relative_to(root), "unsafe cell receipt path")
        require(sha256_file(path) == item.get("sha256"), "cell receipt hash mismatch")
        receipt = read_json(path)
        alias = item["alias"]
        expected = MATRIX[alias]
        require(
            set(receipt)
            == {
                "schema", "alias", "model", "requested",
                "engine_executable_sha256",
                "observed_offloaded_layers", "observed_gpu_memory_mib",
                "host_receipt_sha256", "n", "correct", "truncated", "api_errors",
                "private_cell_receipt_sha256", "private_scores_recomputed",
                "private_api_error_telemetry_validated",
            },
            "public cell receipt keys differ from the exact schema",
        )
        require(receipt.get("schema") == "paper4-p4r1-public-cell-receipt-v1", "wrong cell schema")
        require(receipt.get("alias") == alias, "cell alias mismatch")
        require(
            receipt.get("model")
            == {"filename": expected[0], "bytes": expected[1], "sha256": expected[2]},
            "model artifact identity mismatch",
        )
        require(receipt.get("requested") == {"context_tokens": 4096, "gpu_layers": expected[3]}, "matrix mismatch")
        require(
            receipt.get("engine_executable_sha256") == toolchain_record["executable_sha256"],
            "cell engine identity differs from toolchain receipt",
        )
        require(
            receipt.get("host_receipt_sha256") == host_record["private_receipt_sha256"],
            "cell host identity differs from the campaign host receipt",
        )
        observed = receipt.get("observed_offloaded_layers")
        require(
            isinstance(observed, dict)
            and set(observed) == {"gpu", "total", "readback"}
            and isinstance(observed.get("gpu"), int)
            and not isinstance(observed.get("gpu"), bool)
            and isinstance(observed.get("total"), int)
            and not isinstance(observed.get("total"), bool)
            and observed["gpu"] >= 0
            and observed["total"] >= observed["gpu"]
            and observed["readback"] == f"{observed['gpu']}/{observed['total']}"
            and (expected[3] == "auto" or observed["gpu"] == int(expected[3])),
            "observed layer placement is malformed or differs from the explicit request",
        )
        gpu_memory = receipt.get("observed_gpu_memory_mib")
        require(
            isinstance(gpu_memory, dict)
            and set(gpu_memory) == {"startup", "completion"}
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in gpu_memory.values()
            ),
            "GPU-memory placement snapshots are malformed",
        )
        require(
            receipt.get("n") == 100
            and isinstance(receipt.get("correct"), int)
            and not isinstance(receipt.get("correct"), bool)
            and 0 <= receipt["correct"] <= 100
            and isinstance(receipt.get("truncated"), int)
            and not isinstance(receipt.get("truncated"), bool)
            and 0 <= receipt["truncated"] <= 100
            and isinstance(receipt.get("api_errors"), int)
            and not isinstance(receipt.get("api_errors"), bool)
            and 0 <= receipt["api_errors"] <= 100,
            "cell count fields are malformed",
        )
        require(receipt.get("private_scores_recomputed") is True, "cell score recomputation was not attested")
        require(
            receipt.get("private_api_error_telemetry_validated") is True,
            "cell exact API-error telemetry validation was not attested",
        )
        require(
            isinstance(receipt.get("private_cell_receipt_sha256"), str)
            and SHA256.fullmatch(receipt["private_cell_receipt_sha256"]) is not None,
            "private cell receipt hash is malformed",
        )
        receipts[alias] = receipt
    require(set(receipts) == set(MATRIX), "cell receipt set differs from P4R1")

    rows_path = root / "primary_gsm8k.jsonl"
    require(sha256_file(rows_path) == campaign.get("primary_rows_sha256"), "primary row hash mismatch")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 600, "expected 600 public rows")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        require(isinstance(row, dict), "public JSONL contains a non-object row")
        validate_public_row(row)
        alias = row.get("label")
        require(alias in MATRIX and row.get("model_id") == alias, "row model identity mismatch")
        require(row.get("dataset_sha256") == DATASET_SHA256, "row dataset binding mismatch")
        require(row.get("model_manifest_sha256") == campaign.get("model_manifest_sha256"), "row manifest binding mismatch")
        require(row.get("sampler_contract_sha256") == campaign.get("sampler_contract_sha256"), "row sampler binding mismatch")
        require(row.get("harness_sha256") == protocol_sources["harness_sha256"], "row harness binding mismatch")
        require(row.get("server_guard_sha256") == protocol_sources["server_guard_sha256"], "row guard binding mismatch")
        match = re.fullmatch(r"gsm8k_(\d+)", str(row.get("id")))
        require(match is not None and row.get("sampler_seed") == SEED_BASE + int(match.group(1)), "row seed mismatch")
        grouped[alias].append(row)
    require(set(grouped) == set(MATRIX), "row alias set differs from P4R1")
    for alias, values in grouped.items():
        require([row["id"] for row in values] == EXPECTED_IDS, f"{alias}: item order mismatch")
        receipt = receipts[alias]
        require(sum(bool(row.get("correct")) for row in values) == receipt["correct"], f"{alias}: score mismatch")
        require(sum(bool(row.get("truncated")) for row in values) == receipt["truncated"], f"{alias}: truncation mismatch")
        require(sum(bool(row.get("api_error_recorded")) for row in values) == receipt["api_errors"], f"{alias}: API error mismatch")
    paired = {
        name: {"table": (table := paired_table(grouped[low], grouped[high])), "mcnemar_exact_two_sided": mcnemar(table)}
        for name, (low, high) in COMPARISONS.items()
    }
    adjusted = holm({name: value["mcnemar_exact_two_sided"] for name, value in paired.items()})
    for name, value in adjusted.items():
        paired[name]["holm_adjusted_mcnemar_p"] = value
    return {
        "status": "pass",
        "cells": {
            alias: {
                "correct": receipts[alias]["correct"],
                "n": 100,
                "wilson_95": wilson(receipts[alias]["correct"], 100),
                "offloaded_layers": receipts[alias]["observed_offloaded_layers"]["readback"],
            }
            for alias in sorted(receipts)
        },
        "row_count": len(rows),
        "paired": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.public_candidate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
