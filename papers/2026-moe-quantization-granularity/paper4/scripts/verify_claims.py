#!/usr/bin/env python3
"""Rebuild Paper 4's quantitative claims from frozen local evidence.

This program is intentionally read-only with respect to experiments.  It reads
saved evaluation JSONLs, the canonical MATH-25 rescorer, campaign launch scripts,
and GGUF headers.  It never starts an inference engine or modifies raw data.

Outputs (deterministic; no timestamps or absolute home-directory paths):
  paper4/derived/claims.json
  paper4/derived/claims.tex
  paper4/derived/REPORT.md
"""

from __future__ import annotations

import argparse
import ast
import contextlib
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import runpy
import statistics
import struct
from typing import Any, Iterable
import warnings


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
RESULTS = ROOT / "testsuite" / "evals" / "results"
SUMMARY = RESULTS / "summary.jsonl"
RESCORER = ROOT / "testsuite" / "evals" / "rescore_math25.py"
DATASET_PREP = ROOT / "testsuite" / "evals" / "prepare_datasets.py"
FROZEN_GSM8K = ROOT / "testsuite" / "evals" / "datasets" / "gsm8k_100.jsonl"
RUN_EVAL = ROOT / "testsuite" / "evals" / "run_eval.py"
DEFAULT_OUTPUT = PAPER / "derived" / "claims.json"
DEFAULT_TEX = PAPER / "derived" / "claims.tex"
DEFAULT_REPORT = PAPER / "derived" / "REPORT.md"
HARDWARE_RECORD = (
    ROOT / "publication" / "papers" / "qwen38_27b_12gb"
    / "public" / "environment" / "hardware.json"
)

Z_95 = statistics.NormalDist().inv_cdf(0.975)


class VerificationError(RuntimeError):
    """Frozen evidence is missing, malformed, or internally inconsistent."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise VerificationError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"invalid binomial count {k}/{n}")
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Fisher's two-sided exact p, using probability ordering and exact rationals."""
    if not (0 <= k1 <= n1 and 0 <= k2 <= n2):
        raise ValueError("invalid Fisher table")
    successes = k1 + k2
    total = n1 + n2
    denominator = math.comb(total, n1)

    def probability(x: int) -> Fraction:
        return Fraction(
            math.comb(successes, x) * math.comb(total - successes, n1 - x),
            denominator,
        )

    lower = max(0, n1 - (total - successes))
    upper = min(n1, successes)
    observed = probability(k1)
    return float(sum((probability(x) for x in range(lower, upper + 1)
                      if probability(x) <= observed), Fraction(0, 1)))


def mcnemar_exact_two_sided(low_only: int, healthy_only: int) -> float:
    """Exact two-sided McNemar p (binomial test on discordant pairs)."""
    if low_only < 0 or healthy_only < 0:
        raise ValueError("negative discordance count")
    discordant = low_only + healthy_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(low_only, healthy_only) + 1))
    return float(min(Fraction(1, 1), Fraction(2 * tail, 2 ** discordant)))


def newcombe_paired_risk_difference_interval(
    both_correct: int,
    low_only: int,
    healthy_only: int,
    both_wrong: int,
    z: float = Z_95,
) -> tuple[float, float, float, float]:
    """Newcombe method-10 hybrid-score CI for healthy-minus-low accuracy.

    The two marginal Wilson intervals are combined with the observed phi
    correlation.  This is deterministic, respects pairing, and behaves much
    better than the paired Wald interval at the near-ceiling cells in this study.
    Returns (risk_difference, lower, upper, phi).
    """
    counts = (both_correct, low_only, healthy_only, both_wrong)
    if any(value < 0 for value in counts):
        raise ValueError("negative paired-table count")
    n = sum(counts)
    if n <= 0:
        raise ValueError("empty paired table")
    healthy_correct = both_correct + healthy_only
    low_correct = both_correct + low_only
    p_healthy = healthy_correct / n
    p_low = low_correct / n
    difference = p_healthy - p_low
    healthy_lo, healthy_hi = wilson_interval(healthy_correct, n, z)
    low_lo, low_hi = wilson_interval(low_correct, n, z)
    phi_denominator = math.sqrt(
        p_healthy * (1.0 - p_healthy) * p_low * (1.0 - p_low)
    )
    phi = ((both_correct / n) - p_healthy * p_low) / phi_denominator \
        if phi_denominator else 0.0
    lower_term = (
        (p_healthy - healthy_lo) ** 2
        + (low_hi - p_low) ** 2
        - 2.0 * phi * (p_healthy - healthy_lo) * (low_hi - p_low)
    )
    upper_term = (
        (healthy_hi - p_healthy) ** 2
        + (p_low - low_lo) ** 2
        - 2.0 * phi * (healthy_hi - p_healthy) * (p_low - low_lo)
    )
    lower = max(-1.0, difference - math.sqrt(max(0.0, lower_term)))
    upper = min(1.0, difference + math.sqrt(max(0.0, upper_term)))
    return difference, lower, upper, phi


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down family-wise-error adjustment."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


# Minimal, streaming GGUF-v3 header reader.  Unlike GGUFReader, this does not
# mmap tens of GiB of tensor data merely to count stored parameters.
_FIXED_VALUE_BYTES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
_STRUCT_FORMAT = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}
_GGUF_METADATA_KEYS = {
    "general.architecture", "general.name", "general.size_label",
    "general.file_type", "general.quantized_by", "split.no", "split.count",
    "split.tensors.count",
}


def _unpack(handle: Any, fmt: str) -> Any:
    size = struct.calcsize("<" + fmt)
    payload = handle.read(size)
    if len(payload) != size:
        raise VerificationError("unexpected EOF in GGUF header")
    return struct.unpack("<" + fmt, payload)[0]


def _gguf_string(handle: Any) -> str:
    length = _unpack(handle, "Q")
    payload = handle.read(length)
    if len(payload) != length:
        raise VerificationError("unexpected EOF in GGUF string")
    return payload.decode("utf-8")


def _skip_value(handle: Any, value_type: int) -> None:
    if value_type in _FIXED_VALUE_BYTES:
        handle.seek(_FIXED_VALUE_BYTES[value_type], io.SEEK_CUR)
    elif value_type == 8:  # STRING
        handle.seek(_unpack(handle, "Q"), io.SEEK_CUR)
    elif value_type == 9:  # ARRAY
        element_type = _unpack(handle, "I")
        count = _unpack(handle, "Q")
        if element_type in _FIXED_VALUE_BYTES:
            handle.seek(_FIXED_VALUE_BYTES[element_type] * count, io.SEEK_CUR)
        elif element_type == 8:
            for _ in range(count):
                handle.seek(_unpack(handle, "Q"), io.SEEK_CUR)
        else:
            raise VerificationError(f"unsupported nested GGUF array type {element_type}")
    else:
        raise VerificationError(f"unsupported GGUF value type {value_type}")


def _read_scalar_value(handle: Any, value_type: int) -> Any:
    if value_type == 8:
        return _gguf_string(handle)
    fmt = _STRUCT_FORMAT.get(value_type)
    if fmt is None:
        raise VerificationError(f"GGUF metadata value is not scalar: {value_type}")
    return _unpack(handle, fmt)


def read_gguf_header(path: Path) -> dict[str, Any]:
    structure_hash = hashlib.sha256()
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise VerificationError(f"{path}: not a little-endian GGUF file")
        version = _unpack(handle, "I")
        if version != 3:
            raise VerificationError(f"{path}: unsupported GGUF version {version}")
        tensor_count = _unpack(handle, "Q")
        metadata_count = _unpack(handle, "Q")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _gguf_string(handle)
            value_type = _unpack(handle, "I")
            keep = (
                key in _GGUF_METADATA_KEYS
                or key.endswith(".block_count")
                or key.endswith(".expert_count")
                or key.endswith(".expert_used_count")
            )
            if keep and value_type != 9:
                metadata[key] = _read_scalar_value(handle, value_type)
            else:
                _skip_value(handle, value_type)

        stored_parameters = 0
        descriptors: list[tuple[str, list[int], int, int]] = []
        for _ in range(tensor_count):
            name = _gguf_string(handle)
            dimensions = _unpack(handle, "I")
            shape = [_unpack(handle, "Q") for _ in range(dimensions)]
            tensor_type = _unpack(handle, "I")
            offset = _unpack(handle, "Q")
            elements = math.prod(shape)
            stored_parameters += elements
            descriptors.append((name, shape, tensor_type, offset))
        structure_hash.update(json.dumps(
            {"metadata": metadata, "tensors": descriptors},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8"))
        return {
            "version": version,
            "tensor_count": tensor_count,
            "metadata_count": metadata_count,
            "stored_parameters": stored_parameters,
            "metadata": metadata,
            "header_end_offset": handle.tell(),
            "structure_sha256": structure_hash.hexdigest(),
        }


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "dense_iq2_xxs": {
        "files": ["qwen38/Qwen3.8-27B-UD-IQ2_XXS.gguf"],
        "auxiliary_files": ["qwen38/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"],
        "role": "dense low tier",
    },
    "dense_q4_k_xl": {
        "files": ["qwen38/Qwen3.8-27B-UD-Q4_K_XL.gguf"],
        "role": "dense healthy tier",
    },
    "flash_iq1_s": {
        "files": [
            "qwen38-flash-next/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
            "qwen38-flash-next/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf",
            "qwen38-flash-next/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf",
        ],
        "role": "512-expert observed tier",
    },
    "mixtral_iq1_m": {
        "files": ["mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.i1-IQ1_M.gguf"],
        "role": "8-expert low tier",
    },
    "mixtral_iq2_xxs": {
        "files": ["mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf"],
        "role": "8-expert intermediate tier",
    },
    "mixtral_q4_k_m": {
        "files": ["mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.i1-Q4_K_M.gguf"],
        "role": "8-expert healthy tier",
    },
    "qwen3moe_iq1_m": {
        "files": ["qwen3-30b-a3b/Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf"],
        "role": "128-expert low tier",
    },
    "qwen3moe_iq2_xxs": {
        "files": ["qwen3-30b-a3b/Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf"],
        "role": "128-expert intermediate tier",
    },
    "qwen3moe_q4_k_m": {
        "files": ["qwen3-30b-a3b/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"],
        "role": "128-expert healthy tier",
    },
    "gptoss_mxfp4": {
        "files": ["gpt-oss-120b/gpt-oss-120b-MXFP4.gguf"],
        "role": "128-expert QAT healthy anchor",
    },
}


# Expected Hugging Face cache bindings for the six artifacts in the primary
# paired analysis. A huggingface_hub download record contains three lines:
# source revision, LFS etag/OID, and download timestamp. The timestamp is not a
# scientific identifier, so the verifier binds only the first two values plus
# the exact downloaded basename.
HF_PROVENANCE_SPECS: dict[str, dict[str, str]] = {
    "mixtral_iq1_m": {
        "source_revision": "0e8d21e9cc0d28b59173d329348f60f8628d1c05",
        "lfs_oid_sha256": "7e8d75c23bd4de55582ac91084a0575f081c1de374b693f5ea7ffa6fa66a2e3c",
    },
    "mixtral_iq2_xxs": {
        "source_revision": "0e8d21e9cc0d28b59173d329348f60f8628d1c05",
        "lfs_oid_sha256": "db62b07756f50d2f046b0a00dc594f1aa73dba82ebdc182d9f55c96474655f03",
    },
    "mixtral_q4_k_m": {
        "source_revision": "0e8d21e9cc0d28b59173d329348f60f8628d1c05",
        "lfs_oid_sha256": "7f2f98f301a74b645d0a4c2bd64c98e9811f2dfa26dbae4787c280eb1183632c",
    },
    "qwen3moe_q4_k_m": {
        "source_revision": "eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
        "lfs_oid_sha256": "6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0",
    },
    "qwen3moe_iq1_m": {
        "source_revision": "eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
        "lfs_oid_sha256": "d527a854db2a1582a3ce746a17b1f42d860334ece18d385ede9e2e395058b39e",
    },
    "qwen3moe_iq2_xxs": {
        "source_revision": "eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
        "lfs_oid_sha256": "aa0b04ac05f71aa94b542b074500b3de956608365e74f0e1d97ae0d6a6380269",
    },
}


EVAL_CELLS: tuple[tuple[str, str], ...] = (
    ("UD-IQ2_XXS-mtpgpu-effmed", "gsm8k"),
    ("UD-Q4_K_XL-ngl33-effmed", "gsm8k"),
    ("FLASH-IQ1S-effmed", "gsm8k"),
    ("FLASH-IQ1S-effmed", "math25"),
    ("FLASH-IQ1S-effmed", "humaneval_plus"),
    ("GPTOSS-MXFP4-effmed", "gsm8k"),
    ("GPTOSS-MXFP4-effmed", "math25"),
    ("GPTOSS-MXFP4-effmed", "humaneval_plus"),
    ("MIXTRAL-IQ1M", "gsm8k"),
    ("MIXTRAL-IQ1M", "math25"),
    ("MIXTRAL-IQ1M", "humaneval_plus"),
    ("MIXTRAL-IQ2XXS", "gsm8k"),
    ("MIXTRAL-IQ2XXS", "math25"),
    ("MIXTRAL-IQ2XXS", "humaneval_plus"),
    ("MIXTRAL-Q4KM", "gsm8k"),
    ("MIXTRAL-Q4KM", "math25"),
    ("MIXTRAL-Q4KM", "humaneval_plus"),
    ("MIXTRAL-IQ1M-n100", "gsm8k"),
    ("MIXTRAL-IQ2XXS-n100", "gsm8k"),
    ("MIXTRAL-Q4KM-n100", "gsm8k"),
    ("MIXTRAL-Q4KM-t02", "humaneval_plus"),
    ("QWEN3MOE-Q4KM", "gsm8k"),
    ("QWEN3MOE-Q4KM", "math25"),
    ("QWEN3MOE-Q4KM", "humaneval_plus"),
    ("QWEN3MOE-IQ1M", "gsm8k"),
    ("QWEN3MOE-IQ2XXS", "gsm8k"),
)


PROTOCOLS: dict[str, dict[str, Any]] = {
    "dense": {
        "labels": ["UD-IQ2_XXS-mtpgpu-effmed", "UD-Q4_K_XL-ngl33-effmed"],
        "source": "testsuite/run_phase2a.sh", "context_tokens": 8192,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "concurrency": 1,
        "reasoning_effort_request": "medium", "gsm8k_max_tokens": 4096,
        "gsm8k_n": 50,
        "differences": "The low tier used an external MTP draft GGUF; the healthy tier did not.",
    },
    "flash": {
        "labels": ["FLASH-IQ1S-effmed"], "source": "testsuite/flash_quality.sh",
        "context_tokens": 8192, "temperature": 1.0, "top_p": 0.95,
        "top_k": 20, "concurrency": 1, "reasoning_effort_request": "medium",
        "max_tokens": {"gsm8k": 4096, "math25": 8192, "humaneval_plus": 4096},
        "n": {"gsm8k": 50, "math25": 25, "humaneval_plus": 20},
        "differences": "Expert weights were CPU-mmap streamed; one public quantization tier was evaluated.",
    },
    "mixtral_pilot": {
        "labels": ["MIXTRAL-IQ1M", "MIXTRAL-IQ2XXS", "MIXTRAL-Q4KM"],
        "source": "testsuite/mixtral_granularity.sh", "context_tokens": 4096,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "concurrency": 1,
        "reasoning_effort_request": "medium (harness default despite omission on CLI)",
        "max_tokens": {"gsm8k": 2048, "math25": 2048, "humaneval_plus": 2048},
        "n": {"gsm8k": 50, "math25": 25, "humaneval_plus": 20},
        "gpu_layers_by_label": {
            "MIXTRAL-IQ1M": 29, "MIXTRAL-IQ2XXS": 26, "MIXTRAL-Q4KM": 12,
        },
        "differences": "No --reasoning-effort flag was written, but the harness default was medium and was sent unless server retry dropped it.",
    },
    "mixtral_replication": {
        "labels": ["MIXTRAL-IQ1M-n100", "MIXTRAL-IQ2XXS-n100", "MIXTRAL-Q4KM-n100"],
        "source": "testsuite/overnight_20260827.sh", "context_tokens": 4096,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "concurrency": 1,
        "reasoning_effort_request": "medium (harness default despite omission on CLI)",
        "gsm8k_max_tokens": 2048, "gsm8k_n": 100,
        "gpu_layers_by_label": {
            "MIXTRAL-IQ1M-n100": 29, "MIXTRAL-IQ2XXS-n100": 26,
            "MIXTRAL-Q4KM-n100": 12,
        },
        "differences": "Same frozen 100-item GSM8K prefix; one IQ1_M request retried without chat-template kwargs.",
    },
    "mixtral_temperature_control": {
        "labels": ["MIXTRAL-Q4KM-t02"], "source": "testsuite/overnight_20260827.sh",
        "context_tokens": 4096, "temperature": 0.2, "top_p": 0.95,
        "top_k": 20, "concurrency": 1,
        "reasoning_effort_request": "medium (harness default despite omission on CLI)",
        "humaneval_plus_max_tokens": 2048, "humaneval_plus_n": 20,
        "differences": "Temperature 0.2 instead of 1.0; otherwise the same Q4 artifact and 20 items.",
    },
    "qwen3moe": {
        "labels": ["QWEN3MOE-Q4KM", "QWEN3MOE-IQ1M", "QWEN3MOE-IQ2XXS"],
        "source": "testsuite/qwen3moe_battery.sh", "temperature": 1.0,
        "top_p": 0.95, "top_k": 20, "concurrency": 1,
        "reasoning_effort_request": "medium (harness default)",
        "healthy_context_tokens": 8192, "low_context_tokens": 4096,
        "max_tokens": {"gsm8k": 2048, "healthy_math25": 4096,
                       "healthy_humaneval_plus": 2048},
        "n": {"gsm8k": 100, "healthy_math25": 25, "healthy_humaneval_plus": 20},
        "placement": {
            "mode": "llama.cpp automatic fit; no explicit -ngl",
            "exact_outcomes_preserved": False,
            "auditable": False,
        },
        "differences": "Healthy Q4 used 8192-token context; low GSM8K tiers used 4096. Placement was auto-fit, but the exact per-tier outcomes were not preserved; artifact tier and placement are therefore confounded.",
    },
    "gptoss": {
        "labels": ["GPTOSS-MXFP4-effmed"], "source": "testsuite/overnight_20260827.sh",
        "context_tokens": 8192, "temperature": 1.0, "top_p": 0.95,
        "top_k": 20, "concurrency": 1,
        "reasoning_effort_request": "medium (harness default)",
        "max_tokens": {"gsm8k": 4096, "math25": 8192, "humaneval_plus": 4096},
        "n": {"gsm8k": 50, "math25": 25, "humaneval_plus": 20},
        "differences": "Native MXFP4/QAT anchor with expert weights CPU-mmap streamed; no low tier.",
    },
}


COMPARISON_SPECS = {
    "mixtral_iq1_vs_q4": ("MIXTRAL-IQ1M-n100/gsm8k", "MIXTRAL-Q4KM-n100/gsm8k"),
    "mixtral_iq2_vs_q4": ("MIXTRAL-IQ2XXS-n100/gsm8k", "MIXTRAL-Q4KM-n100/gsm8k"),
    "qwen3moe_iq1_vs_q4": ("QWEN3MOE-IQ1M/gsm8k", "QWEN3MOE-Q4KM/gsm8k"),
    "qwen3moe_iq2_vs_q4": ("QWEN3MOE-IQ2XXS/gsm8k", "QWEN3MOE-Q4KM/gsm8k"),
}


def huggingface_metadata_path(model_root: Path, artifact_relative: str) -> Path:
    """Return huggingface_hub's metadata path for a downloaded artifact."""
    relative = Path(artifact_relative)
    if len(relative.parts) < 2:
        raise VerificationError(
            f"model artifact must include its download directory: {artifact_relative}"
        )
    download_relative = Path(*relative.parts[1:])
    return (
        model_root / relative.parts[0] / ".cache" / "huggingface" / "download"
        / download_relative.parent / f"{download_relative.name}.metadata"
    )


def read_huggingface_metadata(
    model_root: Path,
    artifact_relative: str,
    expected_revision: str,
    expected_lfs_oid: str,
) -> dict[str, Any]:
    """Parse and verify a local huggingface_hub download metadata record."""
    metadata_path = huggingface_metadata_path(model_root, artifact_relative)
    if not metadata_path.is_file():
        raise VerificationError(f"missing Hugging Face metadata: {metadata_path}")
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise VerificationError(f"malformed Hugging Face metadata: {metadata_path}")
    revision, lfs_oid = lines[:2]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise VerificationError(
            f"invalid Hugging Face source revision in {metadata_path}: {revision!r}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", lfs_oid):
        raise VerificationError(
            f"invalid Hugging Face LFS OID in {metadata_path}: {lfs_oid!r}"
        )
    if revision != expected_revision:
        raise VerificationError(
            f"Hugging Face revision mismatch for {artifact_relative}: "
            f"expected {expected_revision}, got {revision}"
        )
    if lfs_oid != expected_lfs_oid:
        raise VerificationError(
            f"Hugging Face LFS OID mismatch for {artifact_relative}: "
            f"expected {expected_lfs_oid}, got {lfs_oid}"
        )
    relative_metadata = metadata_path.relative_to(model_root)
    binding = {
        "artifact_basename": Path(artifact_relative).name,
        "source_revision": revision,
        "lfs_oid_sha256": lfs_oid,
    }
    return {
        **binding,
        "metadata_path_relative_to_model_root": str(relative_metadata),
        # The third .metadata line is a download timestamp and is deliberately
        # excluded so an identical source download verifies on another machine.
        "binding_sha256": sha256_json(binding),
    }


def build_model_claims(model_root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_id, spec in MODEL_SPECS.items():
        shard_claims: list[dict[str, Any]] = []
        total_bytes = 0
        total_parameters = 0
        total_tensors = 0
        metadata: dict[str, Any] = {}
        structural = hashlib.sha256()
        for relative in spec["files"]:
            path = model_root / relative
            if not path.is_file():
                raise VerificationError(f"missing GGUF: {path}")
            header = read_gguf_header(path)
            size = path.stat().st_size
            total_bytes += size
            total_parameters += header["stored_parameters"]
            total_tensors += header["tensor_count"]
            metadata.update({key: value for key, value in header["metadata"].items()
                             if key not in metadata})
            structural.update(header["structure_sha256"].encode("ascii"))
            shard_claims.append({
                "path_relative_to_model_root": relative,
                "file_bytes": size,
                "gguf_tensor_count": header["tensor_count"],
                "stored_parameters_from_tensor_shapes": header["stored_parameters"],
                "gguf_header_end_offset": header["header_end_offset"],
                "gguf_structure_sha256": header["structure_sha256"],
            })
        if total_parameters <= 0:
            raise VerificationError(f"{model_id}: no tensor parameters in GGUF shards")
        declared_tensors = metadata.get("split.tensors.count")
        if declared_tensors is not None and declared_tensors != total_tensors:
            raise VerificationError(
                f"{model_id}: split declares {declared_tensors} tensors, read {total_tensors}"
            )
        auxiliary: list[dict[str, Any]] = []
        auxiliary_bytes = 0
        for relative in spec.get("auxiliary_files", []):
            path = model_root / relative
            if not path.is_file():
                raise VerificationError(f"missing auxiliary GGUF: {path}")
            size = path.stat().st_size
            auxiliary_bytes += size
            auxiliary.append({"path_relative_to_model_root": relative, "file_bytes": size})
        source_provenance = None
        if model_id in HF_PROVENANCE_SPECS:
            if len(spec["files"]) != 1:
                raise VerificationError(
                    f"{model_id}: expected one artifact for Hugging Face provenance"
                )
            expected = HF_PROVENANCE_SPECS[model_id]
            source_provenance = read_huggingface_metadata(
                model_root,
                spec["files"][0],
                expected["source_revision"],
                expected["lfs_oid_sha256"],
            )
        models[model_id] = {
            "role": spec["role"],
            "shards": shard_claims,
            "shard_count": len(shard_claims),
            "total_model_file_bytes": total_bytes,
            "total_model_file_gib": total_bytes / 2 ** 30,
            "stored_parameters_from_gguf_tensor_shapes": total_parameters,
            "effective_file_bpw": total_bytes * 8.0 / total_parameters,
            "gguf_tensor_count": total_tensors,
            "metadata": metadata,
            "aggregate_gguf_structure_sha256": structural.hexdigest(),
            "auxiliary_weight_files": auxiliary,
            "auxiliary_weight_file_bytes": auxiliary_bytes,
            "deployment_weight_file_bytes": total_bytes + auxiliary_bytes,
            "deployment_file_bits_per_primary_stored_parameter":
                (total_bytes + auxiliary_bytes) * 8.0 / total_parameters,
        }
        if source_provenance is not None:
            models[model_id]["huggingface_source"] = source_provenance
    return models


def load_math_rescorer() -> tuple[Any, dict[str, str]]:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", ResourceWarning)
        namespace = runpy.run_path(str(RESCORER))
    return namespace["eq"], namespace["TRUTH"]


def build_eval_claims() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summaries = read_jsonl(SUMMARY)
    eq, truth = load_math_rescorer()
    claims: dict[str, Any] = {}
    raw_by_key: dict[str, list[dict[str, Any]]] = {}
    for label, suite in EVAL_CELLS:
        key = f"{label}/{suite}"
        path = RESULTS / label / f"{suite}.jsonl"
        summary_matches = [row for row in summaries
                           if row.get("label") == label and row.get("suite") == suite
                           and not row.get("partial")]
        if not summary_matches:
            raise VerificationError(f"missing complete summary row for {key}")
        summary = summary_matches[-1]
        if path.is_file():
            rows = read_jsonl(path)
            ids = [row.get("id") for row in rows]
            if any(not isinstance(item_id, str) for item_id in ids):
                raise VerificationError(f"{path}: every row needs a string id")
            if len(ids) != len(set(ids)):
                raise VerificationError(f"{path}: duplicate item ids")
            raw_correct = sum(bool(row.get("correct")) for row in rows)
            raw_n = len(rows)
            summary_n = int(summary["n"])
            summary_correct_float = float(summary["accuracy"]) * summary_n
            summary_correct = round(summary_correct_float)
            if not math.isclose(summary_correct_float, summary_correct, abs_tol=1e-8):
                raise VerificationError(f"cannot recover integer summary count for {key}")
            raw_complete = raw_n == summary_n and raw_correct == summary_correct
            if raw_n > summary_n or raw_correct > summary_correct:
                raise VerificationError(f"raw/summary contradiction for {key}")
            n = summary_n
            correct = summary_correct
            if raw_complete:
                raw_by_key[key] = rows
            source: dict[str, Any] = {
                "raw_available": True,
                "raw_complete": raw_complete,
                "raw_rows_available": raw_n,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            if not raw_complete:
                source["limitation"] = (
                    f"Only {raw_n}/{summary_n} streamed item rows survive; "
                    "aggregate correct/n comes from the completed summary."
                )
        else:
            n = int(summary["n"])
            correct_float = float(summary["accuracy"]) * n
            correct = round(correct_float)
            if not math.isclose(correct_float, correct, abs_tol=1e-8):
                raise VerificationError(f"cannot recover integer correct count for {key}")
            source = {
                "raw_available": False,
                "raw_complete": False,
                "raw_rows_available": 0,
                "path": None,
                "sha256": None,
                "limitation": "Only the aggregate summary survives in the active evidence tree.",
            }
            rows = []
        lo, hi = wilson_interval(correct, n)
        cell: dict[str, Any] = {
            "label": label, "suite": suite, "n": n, "correct": correct,
            "accuracy": correct / n, "wilson_95": [lo, hi], "source": source,
            "truncated": sum(bool(row.get("truncated")) for row in rows)
                if rows and source["raw_complete"] else None,
            "truncated_observed_in_surviving_rows":
                sum(bool(row.get("truncated")) for row in rows) if rows else None,
            "errors": sum("error" in row for row in rows)
                if rows and source["raw_complete"] else None,
            "errors_observed_in_surviving_rows":
                sum("error" in row for row in rows) if rows else None,
            "completion_tokens_missing":
                sum(row.get("completion_tokens") is None for row in rows) if rows else None,
            "chat_template_kwargs_dropped":
                sum(bool(row.get("chat_template_kwargs_dropped")) for row in rows)
                if rows else None,
            "item_id_set_sha256": hashlib.sha256(
                "\n".join(sorted(row["id"] for row in rows)).encode("utf-8")
            ).hexdigest() if rows else None,
        }
        if suite == "math25" and rows:
            lenient_correct = 0
            format_artifacts = 0
            for row in rows:
                expected = row.get("expected_answer")
                if expected in (None, "None"):
                    expected = truth.get(row["id"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    warnings.simplefilter("ignore", ResourceWarning)
                    lenient_ok = bool(row.get("correct")) or eq(
                        row.get("predicted_answer"), expected
                    )
                lenient_correct += int(lenient_ok)
                format_artifacts += int(lenient_ok and not row.get("correct"))
            lenient_lo, lenient_hi = wilson_interval(lenient_correct, n)
            cell["lenient"] = {
                "correct": lenient_correct, "accuracy": lenient_correct / n,
                "wilson_95": [lenient_lo, lenient_hi],
                "format_artifacts": format_artifacts,
                "normalizer_source": str(RESCORER.relative_to(ROOT)),
                "normalizer_sha256": sha256_file(RESCORER),
            }
        claims[key] = cell
    return claims, raw_by_key


def build_comparison(
    comparison_id: str,
    low_key: str,
    healthy_key: str,
    cells: dict[str, Any],
    raw_by_key: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    low = cells[low_key]
    healthy = cells[healthy_key]
    result: dict[str, Any] = {
        "low_cell": low_key, "healthy_cell": healthy_key,
        "delta_healthy_minus_low": healthy["accuracy"] - low["accuracy"],
        "delta_pp": 100.0 * (healthy["accuracy"] - low["accuracy"]),
        "fisher_exact_two_sided": fisher_exact_two_sided(
            low["correct"], low["n"], healthy["correct"], healthy["n"]
        ),
    }
    if low_key not in raw_by_key or healthy_key not in raw_by_key:
        result["paired"] = {
            "available": False,
            "reason": "one or both per-item JSONLs are absent",
        }
        return result
    low_rows = {row["id"]: bool(row.get("correct")) for row in raw_by_key[low_key]}
    healthy_rows = {
        row["id"]: bool(row.get("correct")) for row in raw_by_key[healthy_key]
    }
    if low_rows.keys() != healthy_rows.keys():
        raise VerificationError(
            f"{comparison_id}: paired item sets differ: "
            f"{len(low_rows.keys() ^ healthy_rows.keys())} unmatched ids"
        )
    both_correct = sum(low_rows[item] and healthy_rows[item] for item in low_rows)
    low_only = sum(low_rows[item] and not healthy_rows[item] for item in low_rows)
    healthy_only = sum(not low_rows[item] and healthy_rows[item] for item in low_rows)
    both_wrong = sum(not low_rows[item] and not healthy_rows[item] for item in low_rows)
    difference, interval_lo, interval_hi, phi = newcombe_paired_risk_difference_interval(
        both_correct, low_only, healthy_only, both_wrong
    )
    result["paired"] = {
        "available": True,
        "n": len(low_rows),
        "table": {
            "both_correct": both_correct,
            "low_only_correct": low_only,
            "healthy_only_correct": healthy_only,
            "both_wrong": both_wrong,
        },
        "discordant_pairs": low_only + healthy_only,
        "mcnemar_exact_two_sided": mcnemar_exact_two_sided(low_only, healthy_only),
        "risk_difference_healthy_minus_low": difference,
        "risk_difference_95_newcombe_method_10": [interval_lo, interval_hi],
        "phi_correlation": phi,
    }
    return result


def truncation_inventory(cells: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "n": cell["n"], "truncated": cell["truncated"],
            "truncated_observed_in_surviving_rows":
                cell["truncated_observed_in_surviving_rows"],
            "errors": cell["errors"],
            "errors_observed_in_surviving_rows":
                cell["errors_observed_in_surviving_rows"],
            "completion_tokens_missing": cell["completion_tokens_missing"],
            "chat_template_kwargs_dropped": cell["chat_template_kwargs_dropped"],
            "raw_available": cell["source"]["raw_available"],
            "raw_complete": cell["source"]["raw_complete"],
            "raw_rows_available": cell["source"]["raw_rows_available"],
        }
        for key, cell in cells.items()
    }


def _python_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = node.value
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise VerificationError(f"{name} is not a literal") from exc
    raise VerificationError(f"missing module constant {name}")


def _argparse_defaults(tree: ast.Module) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            continue
        flag = node.args[0].value
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[flag] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError) as exc:
                    raise VerificationError(
                        f"nonliteral argparse default for {flag}"
                    ) from exc
    return defaults


def _chat_request_contract(tree: ast.Module) -> dict[str, Any]:
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "chat_completion"
    ]
    if len(functions) != 1:
        raise VerificationError("expected one chat_completion function")
    function = functions[0]
    body_fields: dict[str, Any] | None = None
    reasoning_bound = False
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "body"
                    for target in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            body_fields = {}
            for key, value in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                try:
                    body_fields[key.value] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    body_fields[key.value] = ast.unparse(value)
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "body"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "chat_template_kwargs"
            ):
                continue
            values = {
                key.value: ast.unparse(value)
                for key, value in zip(node.value.keys, node.value.values)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            reasoning_bound = values.get("reasoning_effort") == "args.reasoning_effort"
    if body_fields is None:
        raise VerificationError("chat_completion request body was not found")
    return {"body_fields": body_fields, "reasoning_effort_bound": reasoning_bound}


def _gsm8k_scorer_tolerances(tree: ast.Module) -> dict[str, float]:
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "score_gsm8k"
    ]
    if len(functions) != 1:
        raise VerificationError("expected one score_gsm8k function")
    tolerances: dict[str, float] = {}
    for node in ast.walk(functions[0]):
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.LtE)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, (int, float))
        ):
            continue
        kind = "relative" if isinstance(node.left, ast.BinOp) \
            and isinstance(node.left.op, ast.Div) else "absolute"
        tolerances[kind] = float(node.comparators[0].value)
    if tolerances != {"absolute": 1e-4, "relative": 1e-4}:
        raise VerificationError(
            "GSM8K numeric scoring tolerance changed: "
            f"expected absolute=relative=1e-4, got {tolerances}"
        )
    return tolerances


def _shell_function(text: str, name: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(name)}\(\)\s*\{{.*?^\}}\s*$", text)
    if not match:
        raise VerificationError(f"shell function not found: {name}")
    return match.group(0)


def build_dataset_provenance() -> dict[str, Any]:
    """Verify the seed-42 selection and fingerprint the frozen GSM8K subset."""
    prep_tree = _python_tree(DATASET_PREP)
    seed = _module_literal(prep_tree, "SEED")
    if seed != 42:
        raise VerificationError(f"GSM8K preparation seed changed: expected 42, got {seed}")
    rows = read_jsonl(FROZEN_GSM8K)
    if len(rows) != 100:
        raise VerificationError(f"frozen GSM8K subset has {len(rows)} rows, expected 100")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids) or any(
        not isinstance(item_id, str) or not re.fullmatch(r"gsm8k_\d+", item_id)
        for item_id in ids
    ):
        raise VerificationError("frozen GSM8K subset has malformed or duplicate IDs")
    actual_indices = [int(item_id.removeprefix("gsm8k_")) for item_id in ids]
    expected_indices = sorted(random.Random(seed).sample(range(1319), 100))
    if actual_indices != expected_indices:
        raise VerificationError(
            "frozen GSM8K IDs do not match the seed-42 sample of the 1,319-row test set"
        )
    return {
        "preparation_script": str(DATASET_PREP.relative_to(ROOT)),
        "preparation_script_sha256": sha256_file(DATASET_PREP),
        "source_dataset": "openai/grade-school-math test.jsonl",
        "source_row_count": 1319,
        "sampling_seed": seed,
        "seed_scope": (
            "Dataset item selection only. The inference request did not set a sampler seed."
        ),
        "sampling_algorithm": "sorted(random.Random(seed).sample(range(1319), 100))",
        "selection_matches_seed": True,
        "selected_index_sha256": sha256_json(actual_indices),
        "frozen_dataset": {
            "path": str(FROZEN_GSM8K.relative_to(ROOT)),
            "bytes": FROZEN_GSM8K.stat().st_size,
            "rows": len(rows),
            "sha256": sha256_file(FROZEN_GSM8K),
            "canonical_rows_sha256": sha256_json(rows),
        },
    }


def build_protocol_verification(dataset: dict[str, Any]) -> dict[str, Any]:
    """Bind primary generation and placement settings to their source files."""
    run_tree = _python_tree(RUN_EVAL)
    defaults = _argparse_defaults(run_tree)
    request = _chat_request_contract(run_tree)
    gsm8k_tolerances = _gsm8k_scorer_tolerances(run_tree)
    body = request["body_fields"]
    observed = {
        "dataset_sampling_seed": dataset["sampling_seed"],
        "inference_sampler_seed": body.get("seed"),
        "temperature": defaults.get("--temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "concurrency": defaults.get("--concurrency"),
        "reasoning_effort_request": defaults.get("--reasoning-effort"),
    }
    expected = {
        "dataset_sampling_seed": 42,
        "inference_sampler_seed": None,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "concurrency": 1,
        "reasoning_effort_request": "medium",
    }
    if observed != expected:
        raise VerificationError(
            f"primary generation protocol changed: expected {expected}, got {observed}"
        )
    if body.get("temperature") != "args.temperature":
        raise VerificationError("request temperature is not bound to args.temperature")
    if not request["reasoning_effort_bound"]:
        raise VerificationError("reasoning effort is not bound into chat_template_kwargs")

    overnight_path = ROOT / "testsuite" / "overnight_20260827.sh"
    qwen_path = ROOT / "testsuite" / "qwen3moe_battery.sh"
    pilot_path = ROOT / "testsuite" / "mixtral_granularity.sh"
    overnight = overnight_path.read_text(encoding="utf-8")
    qwen = qwen_path.read_text(encoding="utf-8")
    pilot = pilot_path.read_text(encoding="utf-8")
    primary_blocks = {
        "mixtral_replication": _shell_function(overnight, "stage_B"),
        "qwen_healthy": _shell_function(qwen, "stage_healthy"),
        "qwen_low": _shell_function(qwen, "stage_low"),
    }
    primary_commands = [
        line.strip()
        for block in primary_blocks.values()
        for line in block.splitlines()
        if "./evals/run_eval.py" in line and "--suite gsm8k" in line
    ]
    if len(primary_commands) != 3:
        raise VerificationError(
            f"expected three primary GSM8K command templates, found {len(primary_commands)}"
        )
    for command in primary_commands:
        if "--limit 100" not in command or "--max-tokens 2048" not in command:
            raise VerificationError(f"primary GSM8K command changed: {command}")
        if any(flag in command for flag in (
            "--temperature", "--concurrency", "--reasoning-effort",
        )):
            raise VerificationError(f"primary command overrides a verified default: {command}")

    expected_placement = {"IQ1_M": 29, "IQ2_XXS": 26, "Q4_K_M": 12}
    replication_placement = {
        match.group(1): int(match.group(2))
        for match in re.finditer(
            r'"MIXTRAL-(IQ1M|IQ2XXS|Q4KM)-n100:[^:"]+:(\d+)"',
            primary_blocks["mixtral_replication"],
        )
    }
    replication_placement = {
        {"IQ1M": "IQ1_M", "IQ2XXS": "IQ2_XXS", "Q4KM": "Q4_K_M"}[key]: value
        for key, value in replication_placement.items()
    }
    pilot_placement = {
        {"IQ1M": "IQ1_M", "IQ2XXS": "IQ2_XXS", "Q4KM": "Q4_K_M"}[match.group(1)]:
            int(match.group(2))
        for match in re.finditer(
            r"run_tier\s+MIXTRAL-(IQ1M|IQ2XXS|Q4KM)\s+\S+\s+(\d+)", pilot,
        )
    }
    if replication_placement != expected_placement or pilot_placement != expected_placement:
        raise VerificationError(
            "Mixtral -ngl placement changed: "
            f"pilot={pilot_placement}, replication={replication_placement}"
        )

    qwen_server = _shell_function(qwen, "start_server")
    if re.search(r"(^|\s)-ngl(\s|$)", qwen_server):
        raise VerificationError("Qwen server unexpectedly specifies -ngl")
    shared_log = '2>"results/server_qwen3moe.log"'
    if shared_log not in qwen_server:
        raise VerificationError("Qwen shared server-log behavior changed")

    return {
        "verified": True,
        "generation": {
            **observed,
            "max_tokens": 2048,
            "primary_item_count": 100,
            "request_temperature_binding": "args.temperature",
            "reasoning_effort_transport": "chat_template_kwargs.reasoning_effort",
            "reasoning_effort_caveat": (
                "The request was medium by default; one Mixtral IQ1 request retried "
                "without chat-template kwargs, as retained in the row metadata."
            ),
            "source": str(RUN_EVAL.relative_to(ROOT)),
            "primary_launch_sources": [
                str(overnight_path.relative_to(ROOT)),
                str(qwen_path.relative_to(ROOT)),
            ],
        },
        "scoring": {
            "gsm8k": {
                "numeric_absolute_tolerance": gsm8k_tolerances["absolute"],
                "numeric_relative_tolerance": gsm8k_tolerances["relative"],
                "acceptance_rule": (
                    "absolute error <= 1e-4, or relative error <= 1e-4 "
                    "when the expected answer is nonzero"
                ),
                "source": str(RUN_EVAL.relative_to(ROOT)),
            },
        },
        "placement": {
            "mixtral": {
                "mode": "explicit llama.cpp -ngl",
                "gpu_layers_by_tier": expected_placement,
                "verified_in_pilot_and_replication_scripts": True,
                "sources": [
                    str(pilot_path.relative_to(ROOT)),
                    str(overnight_path.relative_to(ROOT)),
                ],
            },
            "qwen": {
                "mode": "llama.cpp automatic fit; no explicit -ngl",
                "exact_auto_fit_outcomes_preserved": False,
                "auditable": False,
                "shared_overwritten_server_log": "testsuite/results/server_qwen3moe.log",
                "limitation": (
                    "Exact per-tier auto-fit outcomes were not preserved. Artifact tier "
                    "and runtime placement therefore cannot be separated in this study."
                ),
                "source": str(qwen_path.relative_to(ROOT)),
            },
        },
    }


def build_findings(
    models: dict[str, Any],
    cells: dict[str, Any],
    comparisons: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "flash_effective_bpw_not_ultralow",
            "severity": "release_blocking",
            "derived": models["flash_iq1_s"]["effective_file_bpw"],
            "manuscript_characterization": "approximately 1.7--1.8 effective bpw (and ~1.75 in the table)",
            "finding": "All three deployed GGUF shards contain 176,943,899,520 stored parameters and 72,546,461,344 bytes, yielding 3.27998 effective file bpw. IQ1_S/1.56 is a bulk quantization label, not this deployed file's effective bpw.",
        },
        {
            "id": "dense_low_bpw_rounding",
            "severity": "paper_correction",
            "derived": models["dense_iq2_xxs"]["effective_file_bpw"],
            "manuscript_value": 2.13,
            "finding": "The primary GGUF is 2.16123 effective file bpw. Including the deployed MTP draft file raises aggregate weight-file bits per primary stored parameter to 2.56855.",
        },
        {
            "id": "gptoss_effective_bpw",
            "severity": "paper_correction",
            "derived": models["gptoss_mxfp4"]["effective_file_bpw"],
            "manuscript_value": 4.25,
            "finding": "The exact file/parameter ratio is 4.34052 bpw; 4.25 is a nominal-format/model-size approximation.",
        },
        {
            "id": "paired_tests_should_be_mcnemar",
            "severity": "release_blocking",
            "finding": "The same item IDs were used at Q4 and low tiers, so the four central tests should use paired exact McNemar rather than independent-sample Fisher tests. Exact paired tables, Newcombe intervals, and Holm-adjusted p-values are in comparisons.",
        },
        {
            "id": "dense_paired_raw_unavailable",
            "severity": "limitation",
            "finding": "The dense healthy GSM8K per-item JSONL is absent from the active and staged evidence; its 49/50 aggregate remains, but dense McNemar discordance cannot be reconstructed.",
        },
        {
            "id": "mixtral_reasoning_effort_metadata",
            "severity": "paper_correction",
            "finding": "Mixtral launch scripts omitted --reasoning-effort, but run_eval.py defaults to medium and sends that chat-template kwarg. The claim that no reasoning-effort kwarg was used is not literally supported; whether the template ignored it is not logged.",
        },
        {
            "id": "monotonic_claim_not_identified",
            "severity": "release_blocking",
            "finding": "A measured 1 pp change at 128 experts followed by only an upper bound of <=4 pp at 512 experts does not establish monotonic decline. The data support a coarse-versus-finer contrast, not the written 45->1-><=4 monotonic sequence.",
        },
        {
            "id": "truncations_present",
            "severity": "reporting_required",
            "finding": "The parser scored visible content before truncation status was recorded. All eight output-limit rows in the central GSM8K analysis scored incorrect: Mixtral IQ1_M has 1/100, Qwen IQ1_M has 4/100, Qwen IQ2_XXS has 3/100, and both Q4 anchors have zero; see truncation_inventory for pilot and auxiliary cells.",
        },
        {
            "id": "figures_claims_bound",
            "severity": "release_control",
            "finding": "paper4/scripts/make_figures.py now consumes only derived/claims.json and has deterministic smoke tests. Its three supplementary views are restricted to within-model scores and bpw ladders plus the audited paired Newcombe intervals; none is included in the manuscript.",
        },
        {
            "id": "qwen_autofit_placement_unpreserved",
            "severity": "limitation",
            "finding": "The Qwen launch used llama.cpp automatic GPU fitting without explicit -ngl. A shared server log was overwritten across launches, so exact per-tier fit outcomes are not preserved or auditable. Artifact tier and runtime placement are therefore confounded.",
        },
    ]


def source_fingerprints() -> dict[str, dict[str, Any]]:
    paths = {
        PAPER / "main.tex",
        RESCORER,
        DATASET_PREP,
        FROZEN_GSM8K,
        RUN_EVAL,
        HARDWARE_RECORD,
        ROOT / "testsuite" / "results" / "FLASH-CAMPAIGN.md",
        ROOT / "testsuite" / "results" / "GPTOSS-CAMPAIGN.md",
        *(ROOT / protocol["source"] for protocol in PROTOCOLS.values()),
    }
    fingerprints = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths) if path.is_file()
    }
    summary_rows = read_jsonl(SUMMARY)
    selected_summary_rows = []
    for label, suite in EVAL_CELLS:
        matches = [
            row for row in summary_rows
            if row.get("label") == label and row.get("suite") == suite
            and not row.get("partial")
        ]
        if not matches:
            raise VerificationError(f"missing complete summary row for {label}/{suite}")
        selected_summary_rows.append(matches[-1])
    fingerprints[str(SUMMARY.relative_to(ROOT))] = {
        "scope": "only the selected complete rows consumed by Paper 4",
        "selected_rows": len(selected_summary_rows),
        "sha256": sha256_json(selected_summary_rows),
    }
    return dict(sorted(fingerprints.items()))


def build_environment_claims() -> dict[str, Any]:
    hardware = json.loads(HARDWARE_RECORD.read_text(encoding="utf-8"))
    short_commits = set()
    for relative in (
        "testsuite/results/FLASH-CAMPAIGN.md",
        "testsuite/results/GPTOSS-CAMPAIGN.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        match = re.search(r"\bb1-([0-9a-f]{7,40})\b", text)
        if not match:
            raise VerificationError(f"no preserved llama.cpp build id in {relative}")
        short_commits.add(match.group(1))
    if len(short_commits) != 1:
        raise VerificationError(f"campaign build ids disagree: {sorted(short_commits)}")
    return {
        "hardware": hardware,
        "hardware_source": str(HARDWARE_RECORD.relative_to(ROOT)),
        "llama_cpp_build_id_preserved_in_campaign_records": "b1-" + short_commits.pop(),
        "full_commit_provenance_limit": (
            "The campaign records preserve only the short build id; the full hash in "
            "the manuscript is an association with the surviving checkout, not a value "
            "printed by the evaluation logs."
        ),
    }


def build_manuscript_checks() -> dict[str, Any]:
    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    paired_rows = (
        "Mixtral & IQ1 vs Q4",
        "Mixtral & IQ2 vs Q4",
        "Qwen & IQ1 vs Q4",
        "Qwen & IQ2 vs Q4",
    )
    checks = {
        "imports_generated_claim_macros": r"\input{derived/claims}" in text,
        "uses_paired_mcnemar": "exact McNemar" in text and "McNemar tests" in text,
        "reports_holm_adjustment": "Holm-adjusted" in text and "Holm adjustment" in text,
        "removes_monotonic_granularity_claim": "monotonically" not in text and "monotonic" not in text,
        "reports_flash_exact_bpw_as_context": (
            r"\PfourFlashEffectiveBpw\ effective" in text
        ),
        "reports_qwen_qfour_two_decimal_bpw": (
            r"\q{Q4\_K\_M} & \PfourQwenMoeHealthyEffectiveBpw" in text
        ),
        "reports_central_truncations": (
            r"\PfourMixtralIqOneTruncations\ IQ1 response" in text
            and r"\PfourQwenMoeIqOneTruncations\ IQ1 responses" in text
            and r"\PfourQwenMoeIqTwoTruncations\ IQ2 responses" in text
            and r"\PfourQwenMoeHealthyTruncations\ at Q4" in text
        ),
        "has_exactly_four_unique_paired_table_rows": (
            all(text.count(row) == 1 for row in paired_rows)
        ),
        "does_not_include_legacy_figures": r"\includegraphics" not in text,
    }
    return {
        "checks": checks,
        "all_core_numeric_corrections_present": all(
            value for name, value in checks.items()
            if name != "imports_generated_claim_macros"
        ),
        "all_checks_pass": all(checks.values()),
    }
def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def render_report(claims: dict[str, Any]) -> str:
    manuscript = claims["manuscript_checks"]
    dataset = claims["dataset_provenance"]
    generation = claims["protocol_verification"]["generation"]
    placement = claims["protocol_verification"]["placement"]
    failed = [name for name, passed in manuscript["checks"].items() if not passed]
    if manuscript["all_checks_pass"]:
        verdict = "**The current manuscript passes every automated numeric/source-binding check.**"
    elif manuscript["all_core_numeric_corrections_present"]:
        verdict = (
            "**The rewritten manuscript contains the core numeric corrections.** "
            "The remaining pipeline check is: " + ", ".join(failed) + "."
        )
    else:
        verdict = "**The current manuscript still fails numeric checks:** " + ", ".join(failed) + "."
    lines = [
        "# Paper 4 deterministic quantitative audit",
        "",
        "This report is generated by `paper4/scripts/verify_claims.py` from frozen JSONLs, local GGUF headers, Hugging Face download metadata, and protocol source files. No inference or simulation is run.",
        "",
        "## Release verdict",
        "",
        verdict,
        "",
        "The audit invalidated the earlier ~1.75-effective-bpw Flash point and replaced independent-sample tests with paired inference. The current source is checked against those corrections on every rebuild.",
        "",
        "## Exact deployed-file ratios",
        "",
        "| Artifact | Bytes | Stored parameters from GGUF shapes | Effective file bpw | Experts/active |",
        "|---|---:|---:|---:|---:|",
    ]
    model_order = (
        "dense_iq2_xxs", "dense_q4_k_xl", "mixtral_iq1_m", "mixtral_iq2_xxs",
        "mixtral_q4_k_m", "qwen3moe_iq1_m", "qwen3moe_iq2_xxs",
        "qwen3moe_q4_k_m", "flash_iq1_s", "gptoss_mxfp4",
    )
    for model_id in model_order:
        model = claims["models"][model_id]
        metadata = model["metadata"]
        expert = next((v for k, v in metadata.items() if k.endswith(".expert_count")), "—")
        active = next((v for k, v in metadata.items() if k.endswith(".expert_used_count")), "—")
        pair = f"{expert}/{active}" if expert != "—" else "—"
        lines.append(
            f"| `{model_id}` | {model['total_model_file_bytes']:,} | "
            f"{model['stored_parameters_from_gguf_tensor_shapes']:,} | "
            f"{model['effective_file_bpw']:.5f} | {pair} |"
        )

    lines.extend([
        "",
        "The dense IQ2_XXS run also loaded a 1,369,590,656-byte MTP draft GGUF. Counting both deployed weight files gives 2.56855 file bits per primary stored parameter; the table above keeps the conventional primary-checkpoint ratio.",
        "",
        "## Primary artifact-source bindings",
        "",
        "The verifier reads each local `huggingface_hub` `.metadata` record and fails unless its basename, source revision, and full LFS SHA-256 OID match the frozen expectation.",
        "",
        "| Artifact | Exact basename | Source revision | LFS OID (SHA-256) | Binding SHA-256 |",
        "|---|---|---|---|---|",
    ])
    for model_id in (
        "mixtral_iq1_m", "mixtral_iq2_xxs", "mixtral_q4_k_m",
        "qwen3moe_iq1_m", "qwen3moe_iq2_xxs", "qwen3moe_q4_k_m",
    ):
        source = claims["models"][model_id]["huggingface_source"]
        lines.append(
            f"| `{model_id}` | `{source['artifact_basename']}` | "
            f"`{source['source_revision']}` | `{source['lfs_oid_sha256']}` | "
            f"`{source['binding_sha256']}` |"
        )

    lines.extend([
        "",
        "## Four paired Q4-versus-low comparisons",
        "",
        "Orientation is healthy Q4 minus low tier. The interval is Newcombe's paired hybrid-score method 10. Holm adjustment covers exactly these four exploratory McNemar tests.",
        "",
        "| Comparison | Both correct | Low only | Q4 only | Both wrong | Difference (95% CI) | McNemar p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for comparison_id in COMPARISON_SPECS:
        comparison = claims["comparisons"][comparison_id]
        paired = comparison["paired"]
        table = paired["table"]
        lo, hi = paired["risk_difference_95_newcombe_method_10"]
        lines.append(
            f"| `{comparison_id}` | {table['both_correct']} | "
            f"{table['low_only_correct']} | {table['healthy_only_correct']} | "
            f"{table['both_wrong']} | {comparison['delta_pp']:.1f} pp "
            f"[{100*lo:.1f}, {100*hi:.1f}] | "
            f"{paired['mcnemar_exact_two_sided']:.8g} | "
            f"{paired['holm_adjusted_mcnemar_p']:.8g} |"
        )

    lines.extend([
        "",
        "For comparison, the manuscript's independent-sample Fisher p-values are reproducible from the margins, but discard pairing: Mixtral IQ1/Q4 2.48e-10 and IQ2/Q4 0.00548. The paired exact values are 4.36e-12 and 0.002887, respectively.",
        "",
        "## Accuracy and Wilson intervals",
        "",
        "| Cell | Correct/n | Accuracy | Wilson 95% | Truncated | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    central_keys = (
        "UD-IQ2_XXS-mtpgpu-effmed/gsm8k", "UD-Q4_K_XL-ngl33-effmed/gsm8k",
        "FLASH-IQ1S-effmed/gsm8k", "GPTOSS-MXFP4-effmed/gsm8k",
        "MIXTRAL-IQ1M-n100/gsm8k", "MIXTRAL-IQ2XXS-n100/gsm8k",
        "MIXTRAL-Q4KM-n100/gsm8k", "QWEN3MOE-IQ1M/gsm8k",
        "QWEN3MOE-IQ2XXS/gsm8k", "QWEN3MOE-Q4KM/gsm8k",
    )
    for key in central_keys:
        cell = claims["eval_cells"][key]
        lo, hi = cell["wilson_95"]
        trunc = cell["truncated"] if cell["truncated"] is not None else "unknown"
        errors = cell["errors"] if cell["errors"] is not None else "unknown"
        lines.append(
            f"| `{key}` | {cell['correct']}/{cell['n']} | "
            f"{fmt_pct(cell['accuracy'])} | [{fmt_pct(lo)}, {fmt_pct(hi)}] | "
            f"{trunc} | {errors} |"
        )

    lines.extend([
        "",
        "## Protocol and truncation inventory",
        "",
        f"- Frozen machine record: {claims['environment']['hardware']['gpu']['name']}, {claims['environment']['hardware']['gpu']['vram_mib']:,} MiB VRAM, {claims['environment']['hardware']['gpu']['power_limit_w']} W; {claims['environment']['hardware']['cpu']}; {claims['environment']['hardware']['ram']}; NVMe.",
        f"- Campaign records preserve llama.cpp build `{claims['environment']['llama_cpp_build_id_preserved_in_campaign_records']}`. They do not themselves preserve its full 40-character hash.",
        f"- Dataset selection: seed {dataset['sampling_seed']} in `{dataset['preparation_script']}` selected the 100 frozen GSM8K IDs. The frozen file is `{dataset['frozen_dataset']['sha256']}` (SHA-256), and its IDs exactly reproduce the seed-42 sample. This seed applies only to dataset selection; the inference request did not set a sampler seed.",
        f"- Primary generation request: temperature {generation['temperature']}, top-p {generation['top_p']}, top-k {generation['top_k']}, concurrency {generation['concurrency']}, and requested reasoning effort `{generation['reasoning_effort_request']}`. The three primary command templates do not override these verified harness defaults.",
        "- GSM8K numeric scoring accepted an answer when absolute error was at most `1e-4`, or relative error was at most `1e-4` for a nonzero expected answer. The verifier binds both constants directly from `score_gsm8k`.",
        "- Dense: context 8192, GSM8K max 4096, n=50; IQ2_XXS used external MTP while Q4 did not.",
        "- Flash: context 8192; max tokens 4096 GSM8K/HE+ and 8192 MATH; expert weights streamed from mmap.",
        f"- Mixtral pilot/replication: context 4096, max 2048; pilot n=50 and replication n=100. Explicit `-ngl` placement was IQ1_M={placement['mixtral']['gpu_layers_by_tier']['IQ1_M']}, IQ2_XXS={placement['mixtral']['gpu_layers_by_tier']['IQ2_XXS']}, and Q4_K_M={placement['mixtral']['gpu_layers_by_tier']['Q4_K_M']}. Although the CLI omitted `--reasoning-effort`, the harness default requested `medium`.",
        "- One Mixtral IQ1 GSM8K request retried without `chat_template_kwargs`; the row is retained and counted in the generated claims rather than silently treated as an identical request.",
        "- Mixtral temperature control: the same Q4 artifact/items at temperature 0.2 rather than 1.0.",
        f"- Qwen3MoE: Q4 context 8192 versus low-tier context 4096; GSM8K max 2048, n=100. {placement['qwen']['limitation']}",
        "- gpt-oss: context 8192; max 4096 GSM8K/HE+ and 8192 MATH; native MXFP4, streamed experts, no low tier.",
        "",
        "All cells not listed below have complete raw rows with zero truncations and zero request errors. `Observed` is a lower bound when the raw stream is incomplete.",
        "",
        "| Cell | Raw rows | Truncated observed | Errors observed | CTK dropped | Status |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for key, item in claims["truncation_inventory"].items():
        notable = (
            not item["raw_complete"]
            or bool(item["truncated_observed_in_surviving_rows"])
            or bool(item["errors_observed_in_surviving_rows"])
            or bool(item["completion_tokens_missing"])
            or bool(item["chat_template_kwargs_dropped"])
        )
        if not notable:
            continue
        status = "complete" if item["raw_complete"] else "incomplete/aggregate only"
        lines.append(
            f"| `{key}` | {item['raw_rows_available']}/{item['n']} | "
            f"{item['truncated_observed_in_surviving_rows'] if item['truncated_observed_in_surviving_rows'] is not None else 'unknown'} | "
            f"{item['errors_observed_in_surviving_rows'] if item['errors_observed_in_surviving_rows'] is not None else 'unknown'} | "
            f"{item['chat_template_kwargs_dropped'] if item['chat_template_kwargs_dropped'] is not None else 'unknown'} | {status} |"
        )
    lines.extend([
        "",
        "The Flash HumanEval+ stream retains 17/20 rows while its completed summary retains 17/20 correct; item-level truncation/error counts for the missing three rows are therefore unknown. Dense-Q4 GSM8K survives only as a 49/50 summary.",
        "",
        "## Audit findings and release controls",
        "",
        "Severities describe what would happen if a correction were absent. The release verdict above records whether the current manuscript has addressed each automated blocking condition.",
        "",
    ])
    for finding in claims["audit_findings"]:
        lines.append(f"- **{finding['id']} ({finding['severity']}):** {finding['finding']}")
    lines.extend([
        "",
        "## Rebuild",
        "",
        "```bash",
        "python3 paper4/scripts/verify_claims.py --check",
        "```",
        "",
        "`--check` rebuilds in memory and fails if any committed derived file differs.",
        "",
    ])
    return "\n".join(lines)


def _macro(name: str, value: Any) -> str:
    if not name.isalpha():
        raise VerificationError(f"LaTeX macro name must contain letters only: {name}")
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _plain_number(value: float, digits: int = 2) -> str:
    rendered = f"{value:.{digits}f}"
    return rendered.rstrip("0").rstrip(".")


def _latex_p(value: float) -> str:
    if value == 1.0:
        return "1.0"
    if value == 0.0:
        return "0"
    if value < 0.001:
        coefficient, exponent = f"{value:.2e}".split("e")
        coefficient = coefficient.rstrip("0").rstrip(".")
        return rf"\ensuremath{{{coefficient}\times 10^{{{int(exponent)}}}}}"
    return _plain_number(value, 6)


def render_claims_tex(claims: dict[str, Any]) -> str:
    """Stable manuscript macros; all values originate in claims.json."""
    lines = [
        "% Generated by paper4/scripts/verify_claims.py. DO NOT EDIT.",
        "% Every macro is derived from frozen JSONLs or local GGUF headers.",
    ]
    models = claims["models"]
    model_macros = {
        "DenseLow": "dense_iq2_xxs",
        "DenseHealthy": "dense_q4_k_xl",
        "MixtralIqOne": "mixtral_iq1_m",
        "MixtralIqTwo": "mixtral_iq2_xxs",
        "MixtralHealthy": "mixtral_q4_k_m",
        "QwenMoeIqOne": "qwen3moe_iq1_m",
        "QwenMoeIqTwo": "qwen3moe_iq2_xxs",
        "QwenMoeHealthy": "qwen3moe_q4_k_m",
        "Flash": "flash_iq1_s",
        "GptOss": "gptoss_mxfp4",
    }
    lines.append("% Artifact sizes, GGUF-header parameter counts, and exact file bpw")
    for prefix, model_id in model_macros.items():
        model = models[model_id]
        lines.extend([
            _macro(f"Pfour{prefix}FileBytes", model["total_model_file_bytes"]),
            _macro(f"Pfour{prefix}StoredParameters",
                   model["stored_parameters_from_gguf_tensor_shapes"]),
            _macro(f"Pfour{prefix}EffectiveBpw",
                   f"{model['effective_file_bpw']:.2f}"),
            _macro(f"Pfour{prefix}EffectiveBpwExact",
                   f"{model['effective_file_bpw']:.5f}"),
        ])
    mixtral_revisions = {
        models[model_id]["huggingface_source"]["source_revision"]
        for model_id in ("mixtral_iq1_m", "mixtral_iq2_xxs", "mixtral_q4_k_m")
    }
    qwen_revisions = {
        models[model_id]["huggingface_source"]["source_revision"]
        for model_id in ("qwen3moe_iq1_m", "qwen3moe_iq2_xxs", "qwen3moe_q4_k_m")
    }
    if len(mixtral_revisions) != 1 or len(qwen_revisions) != 1:
        raise VerificationError("primary model tiers do not share a source revision")
    lines.extend([
        _macro("PfourMixtralSourceRevision", mixtral_revisions.pop()),
        _macro("PfourQwenMoeSourceRevision", qwen_revisions.pop()),
    ])
    lines.extend([
        _macro("PfourDenseLowDeploymentBpw",
               f"{models['dense_iq2_xxs']['deployment_file_bits_per_primary_stored_parameter']:.2f}"),
        _macro("PfourMixtralExperts", "8"),
        _macro("PfourMixtralActiveExperts", "2"),
        _macro("PfourQwenMoeExperts", "128"),
        _macro("PfourQwenMoeActiveExperts", "8"),
        _macro("PfourFlashExperts", "512"),
        _macro("PfourFlashActiveExperts", "10"),
        _macro("PfourGptOssExperts", "128"),
        _macro("PfourGptOssActiveExperts", "4"),
    ])

    environment = claims["environment"]
    generation = claims["protocol_verification"]["generation"]
    scoring = claims["protocol_verification"]["scoring"]["gsm8k"]
    placement = claims["protocol_verification"]["placement"]
    hardware = environment["hardware"]
    ram_match = re.match(r"(\d+)\s+GiB", hardware["ram"])
    if not ram_match:
        raise VerificationError(f"cannot parse RAM record: {hardware['ram']}")
    lines.extend([
        _macro("PfourGpuVramMib", hardware["gpu"]["vram_mib"]),
        _macro("PfourGpuPowerW", hardware["gpu"]["power_limit_w"]),
        _macro("PfourRamGib", ram_match.group(1)),
        _macro("PfourPrimaryItemCount", claims["study_integrity"]["primary_item_count"]),
        _macro("PfourDatasetSamplingSeed", generation["dataset_sampling_seed"]),
        _macro("PfourInferenceSamplerSeed", "not set"),
        _macro("PfourTemperature", f"{generation['temperature']:.1f}"),
        _macro("PfourTopP", f"{generation['top_p']:.2f}"),
        _macro("PfourTopK", generation["top_k"]),
        _macro("PfourConcurrency", generation["concurrency"]),
        _macro("PfourReasoningEffort", generation["reasoning_effort_request"]),
        _macro("PfourGsmScorerTolerance", r"\ensuremath{10^{-4}}"),
        _macro("PfourPrimaryMaxTokens", generation["max_tokens"]),
        _macro("PfourMixtralContextTokens", "4096"),
        _macro("PfourQwenHealthyContextTokens", "8192"),
        _macro("PfourQwenLowContextTokens", "4096"),
        _macro("PfourMixtralIqOneGpuLayers",
               placement["mixtral"]["gpu_layers_by_tier"]["IQ1_M"]),
        _macro("PfourMixtralIqTwoGpuLayers",
               placement["mixtral"]["gpu_layers_by_tier"]["IQ2_XXS"]),
        _macro("PfourMixtralHealthyGpuLayers",
               placement["mixtral"]["gpu_layers_by_tier"]["Q4_K_M"]),
        _macro("PfourQwenPlacementMode", "automatic fit"),
        _macro("PfourQwenPlacementAuditable", "no"),
    ])
    if not math.isclose(scoring["numeric_absolute_tolerance"], 1e-4) or not math.isclose(
        scoring["numeric_relative_tolerance"], 1e-4
    ):
        raise VerificationError("cannot render unexpected GSM8K scorer tolerances")

    cells = claims["eval_cells"]
    cell_macros = {
        "DenseLow": "UD-IQ2_XXS-mtpgpu-effmed/gsm8k",
        "DenseHealthy": "UD-Q4_K_XL-ngl33-effmed/gsm8k",
        "Flash": "FLASH-IQ1S-effmed/gsm8k",
        "GptOss": "GPTOSS-MXFP4-effmed/gsm8k",
        "MixtralIqOne": "MIXTRAL-IQ1M-n100/gsm8k",
        "MixtralIqTwo": "MIXTRAL-IQ2XXS-n100/gsm8k",
        "MixtralHealthy": "MIXTRAL-Q4KM-n100/gsm8k",
        "QwenMoeIqOne": "QWEN3MOE-IQ1M/gsm8k",
        "QwenMoeIqTwo": "QWEN3MOE-IQ2XXS/gsm8k",
        "QwenMoeHealthy": "QWEN3MOE-Q4KM/gsm8k",
    }
    lines.append("% Central GSM8K cells")
    for prefix, key in cell_macros.items():
        cell = cells[key]
        lo, hi = cell["wilson_95"]
        lines.extend([
            _macro(f"Pfour{prefix}N", cell["n"]),
            _macro(f"Pfour{prefix}Correct", cell["correct"]),
            _macro(f"Pfour{prefix}AccuracyPct", _plain_number(100 * cell["accuracy"], 1)),
            _macro(f"Pfour{prefix}WilsonLowPct", f"{100 * lo:.1f}"),
            _macro(f"Pfour{prefix}WilsonHighPct", f"{100 * hi:.1f}"),
        ])
        if cell["truncated"] is not None:
            lines.append(_macro(f"Pfour{prefix}Truncations", cell["truncated"]))
    lines.extend([
        _macro("PfourDenseDeltaPp", "2"),
        _macro("PfourFlashCeilingBoundPp", "4"),
        _macro(
            "PfourMixtralIqOneChatTemplateKwargsDropped",
            cells["MIXTRAL-IQ1M-n100/gsm8k"]["chat_template_kwargs_dropped"],
        ),
    ])

    comparison_macros = {
        "MixtralIqOne": "mixtral_iq1_vs_q4",
        "MixtralIqTwo": "mixtral_iq2_vs_q4",
        "QwenMoeIqOne": "qwen3moe_iq1_vs_q4",
        "QwenMoeIqTwo": "qwen3moe_iq2_vs_q4",
    }
    lines.append("% Paired Q4-minus-low comparisons")
    for prefix, comparison_id in comparison_macros.items():
        comparison = claims["comparisons"][comparison_id]
        paired = comparison["paired"]
        table = paired["table"]
        ci_lo, ci_hi = paired["risk_difference_95_newcombe_method_10"]
        lines.extend([
            _macro(f"Pfour{prefix}DeltaPp", _plain_number(comparison["delta_pp"], 1)),
            _macro(f"Pfour{prefix}BothCorrect", table["both_correct"]),
            _macro(f"Pfour{prefix}LowOnlyCorrect", table["low_only_correct"]),
            _macro(f"Pfour{prefix}HealthyOnlyCorrect", table["healthy_only_correct"]),
            _macro(f"Pfour{prefix}BothWrong", table["both_wrong"]),
            _macro(f"Pfour{prefix}RiskDiffCiLowPp", f"{100 * ci_lo:.1f}"),
            _macro(f"Pfour{prefix}RiskDiffCiHighPp", f"{100 * ci_hi:.1f}"),
            _macro(f"Pfour{prefix}McNemarP", _latex_p(paired["mcnemar_exact_two_sided"])),
            _macro(f"Pfour{prefix}HolmP", _latex_p(paired["holm_adjusted_mcnemar_p"])),
            _macro(f"Pfour{prefix}LegacyFisherP", _latex_p(comparison["fisher_exact_two_sided"])),
        ])

    supplementary_cells = {
        "QwenMoeHealthyMath": "QWEN3MOE-Q4KM/math25",
        "QwenMoeHealthyCode": "QWEN3MOE-Q4KM/humaneval_plus",
        "GptOssMath": "GPTOSS-MXFP4-effmed/math25",
        "GptOssCode": "GPTOSS-MXFP4-effmed/humaneval_plus",
        "FlashMath": "FLASH-IQ1S-effmed/math25",
        "FlashCode": "FLASH-IQ1S-effmed/humaneval_plus",
        "MixtralTempOneCode": "MIXTRAL-Q4KM/humaneval_plus",
        "MixtralTempPointTwoCode": "MIXTRAL-Q4KM-t02/humaneval_plus",
        "MixtralHealthyMath": "MIXTRAL-Q4KM/math25",
    }
    lines.append("% Suite-validity and healthy-anchor cells")
    for prefix, key in supplementary_cells.items():
        cell = cells[key]
        lines.extend([
            _macro(f"Pfour{prefix}N", cell["n"]),
            _macro(f"Pfour{prefix}Correct", cell["correct"]),
            _macro(f"Pfour{prefix}AccuracyPct", _plain_number(100 * cell["accuracy"], 1)),
        ])
        if "lenient" in cell:
            lines.extend([
                _macro(f"Pfour{prefix}LenientCorrect", cell["lenient"]["correct"]),
                _macro(f"Pfour{prefix}LenientAccuracyPct",
                       _plain_number(100 * cell["lenient"]["accuracy"], 1)),
                _macro(f"Pfour{prefix}FormatArtifacts", cell["lenient"]["format_artifacts"]),
            ])
    temp = claims["supplementary_comparisons"]["mixtral_temperature_control"]
    lines.extend([
        _macro("PfourMixtralTemperatureFisherP", _latex_p(temp["fisher_exact_two_sided"])),
        _macro("PfourMixtralTemperatureMcNemarP",
               _latex_p(temp["paired"]["mcnemar_exact_two_sided"])),
    ])

    pilot = {
        "MixtralPilotIqOne": "MIXTRAL-IQ1M/gsm8k",
        "MixtralPilotIqTwo": "MIXTRAL-IQ2XXS/gsm8k",
        "MixtralPilotHealthy": "MIXTRAL-Q4KM/gsm8k",
    }
    lines.append("% Original n=50 Mixtral pilot")
    for prefix, key in pilot.items():
        cell = cells[key]
        lines.extend([
            _macro(f"Pfour{prefix}N", cell["n"]),
            _macro(f"Pfour{prefix}Correct", cell["correct"]),
            _macro(f"Pfour{prefix}AccuracyPct", _plain_number(100 * cell["accuracy"], 1)),
        ])
    lines.append("")
    return "\n".join(lines)


def build_claims(model_root: Path) -> dict[str, Any]:
    dataset = build_dataset_provenance()
    protocol_verification = build_protocol_verification(dataset)
    models = build_model_claims(model_root)
    cells, raw_by_key = build_eval_claims()
    mixtral_iq1_dropped = cells["MIXTRAL-IQ1M-n100/gsm8k"][
        "chat_template_kwargs_dropped"
    ]
    if mixtral_iq1_dropped != 1:
        raise VerificationError(
            "expected one Mixtral IQ1 row to retry without chat-template kwargs, "
            f"got {mixtral_iq1_dropped}"
        )
    protocol_verification["observed_exceptions"] = {
        "mixtral_iq1_chat_template_kwargs_dropped": {
            "cell": "MIXTRAL-IQ1M-n100/gsm8k",
            "rows": mixtral_iq1_dropped,
            "meaning": (
                "One request retried without chat_template_kwargs after the initial "
                "medium-reasoning request was rejected."
            ),
        },
    }
    comparisons = {
        comparison_id: build_comparison(
            comparison_id, low_key, healthy_key, cells, raw_by_key
        )
        for comparison_id, (low_key, healthy_key) in COMPARISON_SPECS.items()
    }
    adjusted = holm_adjust({
        comparison_id: comparison["paired"]["mcnemar_exact_two_sided"]
        for comparison_id, comparison in comparisons.items()
    })
    for comparison_id, value in adjusted.items():
        comparisons[comparison_id]["paired"]["holm_adjusted_mcnemar_p"] = value

    dense = build_comparison(
        "dense_iq2_vs_q4", "UD-IQ2_XXS-mtpgpu-effmed/gsm8k",
        "UD-Q4_K_XL-ngl33-effmed/gsm8k", cells, raw_by_key,
    )
    temp_control = build_comparison(
        "mixtral_q4_temp_1_vs_0_2", "MIXTRAL-Q4KM/humaneval_plus",
        "MIXTRAL-Q4KM-t02/humaneval_plus", cells, raw_by_key,
    )
    supplementary = {"dense_iq2_vs_q4": dense, "mixtral_temperature_control": temp_control}

    primary_keys = [
        f"{label}/gsm8k" for label in (
            "MIXTRAL-IQ1M-n100", "MIXTRAL-IQ2XXS-n100", "MIXTRAL-Q4KM-n100",
            "QWEN3MOE-IQ1M", "QWEN3MOE-IQ2XXS", "QWEN3MOE-Q4KM",
        )
    ]
    item_hashes = {cells[key]["item_id_set_sha256"] for key in primary_keys}
    if None in item_hashes or len(item_hashes) != 1:
        raise VerificationError("the six primary GSM8K cells do not share one item-id set")

    claims: dict[str, Any] = {
        "schema_version": "1.0.0",
        "scope": "Paper 4 frozen-output quantitative claims; no inference or simulation",
        "methods": {
            "effective_bpw": "sum(deployed GGUF shard bytes)*8 / sum(stored tensor-shape elements)",
            "accuracy_interval": "two-sided 95% Wilson score interval; z=NormalDist().inv_cdf(0.975)",
            "paired_test": "exact two-sided McNemar/binomial test on discordant pairs",
            "paired_risk_difference_interval": "Newcombe method 10 paired hybrid-score 95% interval",
            "multiplicity": "Holm FWER adjustment over four Q4-versus-low exploratory McNemar tests",
            "legacy_margin_test": "two-sided Fisher exact test retained only to reproduce manuscript margins",
            "math_rescore": "canonical testsuite/evals/rescore_math25.py applied to frozen rows",
            "gsm8k_numeric_acceptance": (
                "absolute error <= 1e-4, or relative error <= 1e-4 when expected != 0"
            ),
        },
        "models": models,
        "environment": build_environment_claims(),
        "dataset_provenance": dataset,
        "eval_cells": cells,
        "comparisons": comparisons,
        "supplementary_comparisons": supplementary,
        "protocols": PROTOCOLS,
        "protocol_verification": protocol_verification,
        "study_integrity": {
            "primary_gsm8k_cells": primary_keys,
            "all_six_primary_cells_share_item_ids": True,
            "primary_item_count": 100,
            "primary_item_id_set_sha256": item_hashes.pop(),
        },
        "truncation_inventory": truncation_inventory(cells),
        "source_fingerprints": source_fingerprints(),
        "manuscript_checks": build_manuscript_checks(),
    }
    claims["audit_findings"] = build_findings(models, cells, comparisons)
    return claims


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path.home() / "models")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true",
                        help="fail if committed generated files differ; write nothing")
    args = parser.parse_args(argv)

    claims = build_claims(args.model_root.resolve())
    json_text = canonical_json(claims)
    tex_text = render_claims_tex(claims)
    report_text = render_report(claims)
    if args.check:
        mismatches = []
        for path, expected in (
            (args.output, json_text), (args.tex, tex_text), (args.report, report_text)
        ):
            actual = path.read_text(encoding="utf-8") if path.is_file() else None
            if actual != expected:
                mismatches.append(str(path))
        if mismatches:
            raise VerificationError("stale or missing derived outputs: " + ", ".join(mismatches))
        print("Paper 4 derived claims: VERIFIED (no inference run)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text, encoding="utf-8")
    args.tex.write_text(tex_text, encoding="utf-8")
    args.report.write_text(report_text, encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.tex}")
    print(f"wrote {args.report}")
    print("Paper 4 derived claims rebuilt from frozen evidence; no inference run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
