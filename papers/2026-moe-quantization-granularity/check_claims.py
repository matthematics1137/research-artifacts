#!/usr/bin/env python3
"""Recompute Paper 4's P4R1 paired results from sanitized public rows."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT / "generated" / "data"
ENVIRONMENT = ROOT / "environment"
PLACEMENT_EXCERPT_SHA256 = "d4b31ecddc6a6b2189a5c2d2c598fff211ec2423fd33e7ef1d1b82cb6a5946af"
EXPECTED_HASHES = {
    "claims.json": "4c25a8dab6dd84889f9639c1e0a78f2a5c4562800ffcfb018fde31489d0f9b8b",
    "primary_gsm8k.jsonl": "1305d1ccc4df54302696b92dcee1664520022c0136af91a1d0ccf9906c8245fb",
}
EXPECTED_AUTHORITY = {
    "claims_sha256": EXPECTED_HASHES["claims.json"],
    "promotion_receipt_sha256": "cfdce45819a8647c5c646c42e4561b3abd38c20b1b31afcceab33130adfe766d",
    "public_checker_sha256": "975fd31b980f57632b4bbb71c72cb43c6c418d8cff0e329a082d9bc0891bb5db",
    "promoted_public_rows_sha256": EXPECTED_HASHES["primary_gsm8k.jsonl"],
}
Z95 = NormalDist().inv_cdf(0.975)
PRIMARY_MODELS = {
    "P4R1-MIXTRAL-IQ1M": ("mixtral_iq1_m", "MIXTRAL-IQ1M-n100/gsm8k"),
    "P4R1-MIXTRAL-IQ2XXS": ("mixtral_iq2_xxs", "MIXTRAL-IQ2XXS-n100/gsm8k"),
    "P4R1-MIXTRAL-Q4KM": ("mixtral_q4_k_m", "MIXTRAL-Q4KM-n100/gsm8k"),
    "P4R1-QWEN3MOE-IQ1M": ("qwen3moe_iq1_m", "QWEN3MOE-IQ1M/gsm8k"),
    "P4R1-QWEN3MOE-IQ2XXS": ("qwen3moe_iq2_xxs", "QWEN3MOE-IQ2XXS/gsm8k"),
    "P4R1-QWEN3MOE-Q4KM": ("qwen3moe_q4_k_m", "QWEN3MOE-Q4KM/gsm8k"),
}
EXPECTED_SCORES = {
    "P4R1-MIXTRAL-IQ1M": 26,
    "P4R1-MIXTRAL-IQ2XXS": 46,
    "P4R1-MIXTRAL-Q4KM": 68,
    "P4R1-QWEN3MOE-IQ1M": 96,
    "P4R1-QWEN3MOE-IQ2XXS": 97,
    "P4R1-QWEN3MOE-Q4KM": 91,
}
EXPECTED_TRUNCATIONS = {
    "P4R1-MIXTRAL-IQ1M": 1,
    "P4R1-MIXTRAL-IQ2XXS": 0,
    "P4R1-MIXTRAL-Q4KM": 0,
    "P4R1-QWEN3MOE-IQ1M": 1,
    "P4R1-QWEN3MOE-IQ2XXS": 1,
    "P4R1-QWEN3MOE-Q4KM": 1,
}
ROW_FIELDS = {
    "api_error_recorded", "chat_template_kwargs_dropped", "completion_tokens",
    "correct", "dataset_sha256", "harness_sha256", "id",
    "identity_guard_wall_s", "label", "model_id", "model_manifest_sha256",
    "sampler_contract_sha256", "sampler_seed", "server_guard_sha256",
    "startup_identity_sha256", "thinking_tokens_estimate", "truncated", "wall_s",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    require(abs(actual - expected) <= tolerance, f"{actual} != {expected} +/- {tolerance}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denominator = 1 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denominator
    return center - half, center + half


def mcnemar(low_only: int, q4_only: int) -> float:
    discordant = low_only + q4_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(low_only, q4_only) + 1))
    return float(min(Fraction(1, 1), Fraction(2 * tail, 2**discordant)))


def paired_interval(table: dict[str, int]) -> tuple[float, float, float, float]:
    both = table["both_correct"]
    low_only = table["low_only_correct"]
    q4_only = table["healthy_only_correct"]
    neither = table["both_wrong"]
    n = both + low_only + q4_only + neither
    q4_p, low_p = (both + q4_only) / n, (both + low_only) / n
    q4_lo, q4_hi = wilson(both + q4_only, n)
    low_lo, low_hi = wilson(both + low_only, n)
    denominator = math.sqrt(q4_p * (1 - q4_p) * low_p * (1 - low_p))
    phi = ((both / n) - q4_p * low_p) / denominator if denominator else 0.0
    difference = q4_p - low_p
    lower_term = (q4_p - q4_lo) ** 2 + (low_hi - low_p) ** 2 - 2 * phi * (q4_p - q4_lo) * (low_hi - low_p)
    upper_term = (q4_hi - q4_p) ** 2 + (low_p - low_lo) ** 2 - 2 * phi * (q4_hi - q4_p) * (low_p - low_lo)
    return difference, difference - math.sqrt(max(0, lower_term)), difference + math.sqrt(max(0, upper_term)), phi


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda pair: (pair[1], pair[0]))
    answer: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        answer[name] = running
    return answer


def main() -> int:
    require(
        sha256(ENVIRONMENT / "placement_readback_excerpt.json") == PLACEMENT_EXCERPT_SHA256,
        "changed or missing source-bound Qwen placement excerpt",
    )
    for name, expected in EXPECTED_HASHES.items():
        require(sha256(DATA / name) == expected, f"changed or missing evidence: data/{name}")
    export = json.loads((DATA / "EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
    require(export.get("schema") == "paper4-p4r1-public-data-export-v1", "wrong export schema")
    require(export.get("status") == "unpublished-local-package-source", "export status changed")
    require(export.get("no_inference") is True, "export incorrectly claims inference")
    require(export.get("authority") == EXPECTED_AUTHORITY, "export authority changed")
    require(export.get("outcome_signature") == EXPECTED_SCORES, "export outcome changed")
    for name, expected in EXPECTED_HASHES.items():
        record = export.get("files", {}).get(name, {})
        require(record.get("sha256") == expected, f"export hash record changed for {name}")
        require(record.get("bytes") == (DATA / name).stat().st_size, f"export byte record changed for {name}")

    claims = json.loads((DATA / "claims.json").read_text(encoding="utf-8"))
    require(claims.get("schema_version") == "paper4-p4r1-claims-v1", "not a P4R1 claims record")
    require(claims.get("outcome_signature", {}).get("cell_correct") == EXPECTED_SCORES, "claims score signature changed")
    require(claims.get("p4r1_validation", {}).get("promotion_receipt_sha256") == EXPECTED_AUTHORITY["promotion_receipt_sha256"], "promotion binding changed")
    require(claims.get("p4r1_validation", {}).get("public_checker_sha256") == EXPECTED_AUTHORITY["public_checker_sha256"], "public-checker binding changed")

    rows = load_jsonl(DATA / "primary_gsm8k.jsonl")
    require(len(rows) == 600, "primary row count changed")
    grouped: dict[str, dict[str, bool]] = {}
    truncations: dict[str, int] = {}
    for row in rows:
        require(set(row) == ROW_FIELDS, f"unexpected public row fields: {sorted(row)}")
        label = row["label"]
        require(label in PRIMARY_MODELS and row["model_id"] == label, f"unexpected primary label: {label}")
        match = re.fullmatch(r"gsm8k_(\d+)", row["id"])
        require(match is not None, f"malformed item ID: {row['id']}")
        require(type(row["correct"]) is bool and type(row["truncated"]) is bool, "non-Boolean outcome")
        require(type(row["api_error_recorded"]) is bool and not row["api_error_recorded"], "API error marker changed")
        require(type(row["chat_template_kwargs_dropped"]) is bool and not row["chat_template_kwargs_dropped"], "chat-template exception changed")
        require(isinstance(row["wall_s"], (int, float)) and row["wall_s"] >= 0, "invalid wall time")
        require(isinstance(row["identity_guard_wall_s"], (int, float)) and row["identity_guard_wall_s"] >= 0, "invalid guard time")
        require(type(row["completion_tokens"]) is int and row["completion_tokens"] > 0, "invalid completion-token count")
        require(type(row["thinking_tokens_estimate"]) is int and row["thinking_tokens_estimate"] >= 0, "invalid thinking-token estimate")
        require(row["sampler_seed"] == 4_200_000 + int(match.group(1)), "per-item seed changed")
        for key in ("dataset_sha256", "harness_sha256", "model_manifest_sha256", "sampler_contract_sha256", "server_guard_sha256", "startup_identity_sha256"):
            require(isinstance(row[key], str) and SHA256.fullmatch(row[key]) is not None, f"malformed {key}")
        require(row["dataset_sha256"] == claims["dataset_provenance"]["subset_sha256"], "dataset binding changed")
        grouped.setdefault(label, {})[row["id"]] = row["correct"]
        truncations[label] = truncations.get(label, 0) + int(row["truncated"])

    require(set(grouped) == set(PRIMARY_MODELS), "primary labels changed")
    require(all(len(values) == 100 for values in grouped.values()), "a primary cell is incomplete")
    expected_ids = {f"gsm8k_{index}" for index in sorted(random.Random(42).sample(range(1319), 100))}
    require(all(set(values) == expected_ids for values in grouped.values()), "paired item IDs changed")
    scores = {label: sum(values.values()) for label, values in grouped.items()}
    require(scores == EXPECTED_SCORES, f"primary scores changed: {scores}")
    require(truncations == EXPECTED_TRUNCATIONS, f"truncation counts changed: {truncations}")

    for label, correct in scores.items():
        model_id, cell_id = PRIMARY_MODELS[label]
        cell = claims["eval_cells"][cell_id]
        require((cell["label"], cell["correct"], cell["n"], cell["truncated"]) == (label, correct, 100, truncations[label]), f"claims cell changed for {label}")
        close(cell["accuracy"], correct / 100)
        interval = wilson(correct, 100)
        close(interval[0], cell["wilson_95"][0])
        close(interval[1], cell["wilson_95"][1])
        model = claims["models"][model_id]
        close(model["effective_file_bpw"], model["total_model_file_bytes"] * 8 / model["stored_parameters_from_gguf_tensor_shapes"])

    specs = {
        "mixtral_iq1_vs_q4": ("P4R1-MIXTRAL-IQ1M", "P4R1-MIXTRAL-Q4KM"),
        "mixtral_iq2_vs_q4": ("P4R1-MIXTRAL-IQ2XXS", "P4R1-MIXTRAL-Q4KM"),
        "qwen3moe_iq1_vs_q4": ("P4R1-QWEN3MOE-IQ1M", "P4R1-QWEN3MOE-Q4KM"),
        "qwen3moe_iq2_vs_q4": ("P4R1-QWEN3MOE-IQ2XXS", "P4R1-QWEN3MOE-Q4KM"),
    }
    p_values: dict[str, float] = {}
    for name, (low_label, q4_label) in specs.items():
        low, q4 = grouped[low_label], grouped[q4_label]
        table = {
            "both_correct": sum(low[key] and q4[key] for key in low),
            "low_only_correct": sum(low[key] and not q4[key] for key in low),
            "healthy_only_correct": sum(q4[key] and not low[key] for key in low),
            "both_wrong": sum(not low[key] and not q4[key] for key in low),
        }
        recorded = claims["comparisons"][name]["paired"]
        require(table == recorded["table"], f"paired table changed for {name}: {table}")
        p_values[name] = mcnemar(table["low_only_correct"], table["healthy_only_correct"])
        close(p_values[name], recorded["mcnemar_exact_two_sided"])
        difference, lower, upper, phi = paired_interval(table)
        close(difference, recorded["risk_difference_healthy_minus_low"])
        close(lower, recorded["risk_difference_95_newcombe_method_10"][0])
        close(upper, recorded["risk_difference_95_newcombe_method_10"][1])
        close(phi, recorded["phi_correlation"])

    adjusted = holm(p_values)
    for name, value in adjusted.items():
        close(value, claims["comparisons"][name]["paired"]["holm_adjusted_mcnemar_p"])

    artifacts = json.loads((ENVIRONMENT / "model_artifacts.json").read_text(encoding="utf-8"))
    artifact_rows = {row["artifact_id"]: row for row in artifacts["artifacts"]}
    require(artifacts.get("schema") == "paper4-p4r1-model-artifacts-v1", "model-artifact schema changed")
    require(set(artifact_rows) == {value[0] for value in PRIMARY_MODELS.values()}, "model-artifact inventory changed")
    require(artifacts["total_bytes"] == 90_436_239_456, "six-file byte total changed")
    for model_id in artifact_rows:
        public = artifact_rows[model_id]
        model = claims["models"][model_id]
        identity = model["p4r1_artifact_identity"]
        require({key: public[key] for key in ("filename", "bytes", "sha256")} == identity, f"artifact identity changed for {model_id}")
        close(public["effective_file_bpw"], model["effective_file_bpw"], 1e-9)

    software = json.loads((ENVIRONMENT / "software.json").read_text(encoding="utf-8"))
    require(software.get("schema") == "paper4-p4r1-software-v1", "software schema changed")
    require(software["evaluation"]["context_tokens_all_cells"] == 4096, "context contract changed")
    require(software["evaluation"]["per_item_sampler_seed"] == "4200000 + integer GSM8K item suffix", "seed contract changed")
    require(software["evaluation"]["observed_gpu_layers"] == {
        "P4R1-MIXTRAL-IQ1M": "29/33", "P4R1-MIXTRAL-IQ2XXS": "26/33",
        "P4R1-MIXTRAL-Q4KM": "12/33", "P4R1-QWEN3MOE-IQ1M": "49/49",
        "P4R1-QWEN3MOE-IQ2XXS": "49/49", "P4R1-QWEN3MOE-Q4KM": "49/49",
    }, "placement record changed")

    print("ALL PAPER 4 P4R1 PUBLIC CLAIM CHECKS PASSED")
    print("six complete paired cells; 600 sanitized outcomes; exact P4R1 provenance binding")
    print("Mixtral Q4/IQ2/IQ1: 68/46/26; Qwen Q4/IQ2/IQ1: 91/97/96")
    print("Qwen non-rejection after Holm correction is not evidence of improvement or equivalence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
