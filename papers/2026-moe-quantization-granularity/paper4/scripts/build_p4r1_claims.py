#!/usr/bin/env python3
"""Build Paper 4 primary claims from a promoted P4R1 evidence tree.

Historical raw files are never edited or copied into the new claims schema.
The prior audit supplies only hash-anchored static artifact metadata; all host,
results, protocols, placements, provenance, comparisons, and figure inputs are
rebuilt from the promoted P4R1 evidence and receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parent.parent
BASELINE = PAPER / "derived" / "claims.json"
VERIFY_PATH = PAPER / "scripts" / "verify_claims.py"
PUBLIC_CHECKER = (
    PAPER.parent
    / "publication/papers/moe-quantization-granularity/public/replication/check_p4r1_evidence.py"
)
BASELINE_SHA256 = "8c9ca9c703259b19dde68450f301e5bccf9524a23375cc9710af2dea5f201cc2"
VERIFY_SHA256 = "fccf114d57b887aff1350de6ffa0cb147c417c65932ae8c1848cd26812668fb6"
MAPPING = {
    "P4R1-MIXTRAL-IQ1M": "MIXTRAL-IQ1M-n100/gsm8k",
    "P4R1-MIXTRAL-IQ2XXS": "MIXTRAL-IQ2XXS-n100/gsm8k",
    "P4R1-MIXTRAL-Q4KM": "MIXTRAL-Q4KM-n100/gsm8k",
    "P4R1-QWEN3MOE-IQ1M": "QWEN3MOE-IQ1M/gsm8k",
    "P4R1-QWEN3MOE-IQ2XXS": "QWEN3MOE-IQ2XXS/gsm8k",
    "P4R1-QWEN3MOE-Q4KM": "QWEN3MOE-Q4KM/gsm8k",
}
COMPARISONS = {
    "mixtral_iq1_vs_q4": ("MIXTRAL-IQ1M-n100/gsm8k", "MIXTRAL-Q4KM-n100/gsm8k"),
    "mixtral_iq2_vs_q4": ("MIXTRAL-IQ2XXS-n100/gsm8k", "MIXTRAL-Q4KM-n100/gsm8k"),
    "qwen3moe_iq1_vs_q4": ("QWEN3MOE-IQ1M/gsm8k", "QWEN3MOE-Q4KM/gsm8k"),
    "qwen3moe_iq2_vs_q4": ("QWEN3MOE-IQ2XXS/gsm8k", "QWEN3MOE-Q4KM/gsm8k"),
}


def load_verifier() -> Any:
    spec = importlib.util.spec_from_file_location("paper4_verify_claims", VERIFY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Paper 4 claims verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_promotion_inventory(promoted: Path, expected_receipt_sha256: str) -> dict[str, Any]:
    if promoted.is_symlink() or not promoted.is_dir() or promoted.stat().st_mode & 0o777 != 0o700:
        raise RuntimeError("promoted evidence root must be an owner-only 0700 directory")
    receipt_path = promoted / "promotion-receipt.json"
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise RuntimeError("promotion receipt differs from the explicitly anchored SHA-256")
    promotion = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        promotion.get("schema") != "paper4-p4r1-promotion-receipt-v1"
        or promotion.get("public_privacy_scan") != "pass"
        or promotion.get("private_evidence_preserved") is not True
        or promotion.get("historical_files_overwritten") is not False
    ):
        raise RuntimeError("P4R1 evidence has no valid privacy-checked promotion receipt")
    records = promotion.get("files")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise RuntimeError("promotion receipt has no exact file inventory")
    declared: dict[str, dict[str, Any]] = {}
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str) or relative in declared:
            raise RuntimeError("promotion inventory contains an invalid or duplicate path")
        path = promoted / relative
        if not path.resolve().is_relative_to(promoted.resolve()) or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe promotion inventory path: {relative}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"promoted file differs from receipt: {relative}")
        declared[relative] = record
    actual = {
        path.relative_to(promoted).as_posix()
        for path in promoted.rglob("*")
        if path.is_file() and path != receipt_path
    }
    if actual != set(declared):
        raise RuntimeError("promotion inventory does not exactly cover the evidence tree")
    private = promoted / "private"
    for path in [private, *private.rglob("*")]:
        expected_mode = 0o700 if path.is_dir() else 0o600
        if path.is_symlink() or path.stat().st_mode & 0o777 != expected_mode:
            raise RuntimeError(f"private promoted evidence has unsafe permissions: {path}")
    checker = load_module("paper4_p4r1_public_checker", PUBLIC_CHECKER)
    result = checker.check(promoted / "public-candidate")
    if result.get("status") != "pass" or result.get("row_count") != 600:
        raise RuntimeError("public P4R1 checker did not validate the promoted projection")
    return promotion


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replace_macro(tex: str, name: str, old: str, new: str) -> str:
    source = rf"\newcommand{{\{name}}}{{{old}}}"
    target = rf"\newcommand{{\{name}}}{{{new}}}"
    if tex.count(source) != 1:
        raise RuntimeError(f"expected one anchored {name} macro with value {old!r}")
    return tex.replace(source, target)


def tex_p(value: float) -> str:
    if value == 0 or value >= 0.001:
        return f"{value:.6g}"
    coefficient, exponent = f"{value:.3e}".split("e")
    return rf"\ensuremath{{{float(coefficient):g}\times 10^{{{int(exponent)}}}}}"


def render_p4r1_tex(claims: dict[str, Any]) -> str:
    macros: dict[str, Any] = {
        "PfourPrimaryItemCount": 100,
        "PfourDatasetSamplingSeed": 42,
        "PfourInferenceSamplerSeed": "4200000 plus GSM8K item suffix",
        "PfourTemperature": "1.0",
        "PfourTopP": "0.95",
        "PfourTopK": 20,
        "PfourConcurrency": 1,
        "PfourReasoningEffort": "medium",
        "PfourGsmScorerTolerance": r"\ensuremath{10^{-4}}",
        "PfourPrimaryMaxTokens": 2048,
        "PfourMixtralContextTokens": 4096,
        "PfourQwenHealthyContextTokens": 4096,
        "PfourQwenLowContextTokens": 4096,
        "PfourMixtralIqOneGpuLayers": 29,
        "PfourMixtralIqTwoGpuLayers": 26,
        "PfourMixtralHealthyGpuLayers": 12,
        "PfourQwenPlacementMode": "automatic fit with per-cell readback",
        "PfourQwenPlacementAuditable": "yes",
    }
    hardware = claims["environment"]["p4r1_observed_host"]
    power_w = hardware["gpu"]["power_limits"]["current_w"]
    if not isinstance(power_w, (int, float)) or isinstance(power_w, bool):
        raise RuntimeError("P4R1 public host receipt has no numeric GPU power readback")
    macros.update(
        {
            "PfourGpuVramMib": hardware["gpu"]["memory_total_mib"],
            "PfourGpuPowerW": int(power_w) if float(power_w).is_integer() else power_w,
            "PfourRamGib": round(hardware["memory"]["mem_total_kib"] / (1024 * 1024)),
        }
    )
    model_specs = {
        "MixtralIqOne": "mixtral_iq1_m",
        "MixtralIqTwo": "mixtral_iq2_xxs",
        "MixtralHealthy": "mixtral_q4_k_m",
        "QwenMoeIqOne": "qwen3moe_iq1_m",
        "QwenMoeIqTwo": "qwen3moe_iq2_xxs",
        "QwenMoeHealthy": "qwen3moe_q4_k_m",
    }
    for prefix, key in model_specs.items():
        model = claims["models"][key]
        macros[f"Pfour{prefix}FileBytes"] = model["total_model_file_bytes"]
        macros[f"Pfour{prefix}StoredParameters"] = model[
            "stored_parameters_from_gguf_tensor_shapes"
        ]
        macros[f"Pfour{prefix}EffectiveBpw"] = f"{model['effective_file_bpw']:.2f}"
        macros[f"Pfour{prefix}EffectiveBpwExact"] = f"{model['effective_file_bpw']:.5f}"
    macros.update(
        {
            "PfourMixtralExperts": 8,
            "PfourMixtralActiveExperts": 2,
            "PfourQwenMoeExperts": 128,
            "PfourQwenMoeActiveExperts": 8,
            "PfourMixtralSourceRevision": claims["models"]["mixtral_iq1_m"][
                "huggingface_source"
            ]["source_revision"],
            "PfourQwenMoeSourceRevision": claims["models"]["qwen3moe_iq1_m"][
                "huggingface_source"
            ]["source_revision"],
        }
    )
    cell_specs = {
        "MixtralIqOne": "MIXTRAL-IQ1M-n100/gsm8k",
        "MixtralIqTwo": "MIXTRAL-IQ2XXS-n100/gsm8k",
        "MixtralHealthy": "MIXTRAL-Q4KM-n100/gsm8k",
        "QwenMoeIqOne": "QWEN3MOE-IQ1M/gsm8k",
        "QwenMoeIqTwo": "QWEN3MOE-IQ2XXS/gsm8k",
        "QwenMoeHealthy": "QWEN3MOE-Q4KM/gsm8k",
    }
    for prefix, key in cell_specs.items():
        cell = claims["eval_cells"][key]
        macros[f"Pfour{prefix}N"] = cell["n"]
        macros[f"Pfour{prefix}Correct"] = cell["correct"]
        macros[f"Pfour{prefix}AccuracyPct"] = f"{100 * cell['accuracy']:.0f}"
        macros[f"Pfour{prefix}WilsonLowPct"] = f"{100 * cell['wilson_95'][0]:.1f}"
        macros[f"Pfour{prefix}WilsonHighPct"] = f"{100 * cell['wilson_95'][1]:.1f}"
        macros[f"Pfour{prefix}Truncations"] = cell["truncated"]
        macros[f"Pfour{prefix}ChatTemplateKwargsDropped"] = cell[
            "chat_template_kwargs_dropped"
        ]
    comparison_specs = {
        "MixtralIqOne": "mixtral_iq1_vs_q4",
        "MixtralIqTwo": "mixtral_iq2_vs_q4",
        "QwenMoeIqOne": "qwen3moe_iq1_vs_q4",
        "QwenMoeIqTwo": "qwen3moe_iq2_vs_q4",
    }
    for prefix, key in comparison_specs.items():
        comparison = claims["comparisons"][key]
        signed_branch = claims["outcome_signature"]["signed_result_branch"][key]
        paired = comparison["paired"]
        table = paired["table"]
        interval = paired["risk_difference_95_newcombe_method_10"]
        macros[f"Pfour{prefix}DeltaPp"] = f"{comparison['delta_pp']:.0f}"
        macros[f"Pfour{prefix}LowOnlyCorrect"] = table["low_only_correct"]
        macros[f"Pfour{prefix}HealthyOnlyCorrect"] = table["healthy_only_correct"]
        macros[f"Pfour{prefix}RiskDiffCiLowPp"] = f"{100 * interval[0]:.1f}"
        macros[f"Pfour{prefix}RiskDiffCiHighPp"] = f"{100 * interval[1]:.1f}"
        macros[f"Pfour{prefix}McNemarP"] = tex_p(paired["mcnemar_exact_two_sided"])
        macros[f"Pfour{prefix}HolmP"] = tex_p(paired["holm_adjusted_mcnemar_p"])
        macros[f"Pfour{prefix}SignedDirection"] = {
            "q4_higher": "Q4 scored higher",
            "low_tier_higher": "the low tier scored higher",
            "tied": "the two tiers tied",
        }[signed_branch]
    return "".join(
        rf"\newcommand{{\{name}}}{{{value}}}" + "\n"
        for name, value in macros.items()
    )


def build(promoted: Path, promotion_receipt_sha256: str) -> tuple[dict[str, Any], str, str]:
    if sha256_file(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("historical baseline claims changed; audit and update the explicit anchor")
    if sha256_file(VERIFY_PATH) != VERIFY_SHA256:
        raise RuntimeError("claims analysis implementation changed; audit and update its explicit anchor")
    verifier = load_verifier()
    promotion = verify_promotion_inventory(promoted, promotion_receipt_sha256)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    cells: dict[str, Any] = {}
    raw_by_key: dict[str, list[dict[str, Any]]] = {}
    for alias, key in MAPPING.items():
        path = promoted / "private" / "cells" / alias / "gsm8k.jsonl"
        rows = read_rows(path)
        if len(rows) != 100 or any(row.get("label") != alias for row in rows):
            raise RuntimeError(f"{alias}: promoted rows are incomplete or mislabeled")
        correct = sum(bool(row.get("correct")) for row in rows)
        lo, hi = verifier.wilson_interval(correct, 100)
        cells[key] = {
            "label": alias,
            "suite": "gsm8k",
            "n": 100,
            "correct": correct,
            "accuracy": correct / 100,
            "wilson_95": [lo, hi],
            "source": {
                "raw_available": True,
                "raw_complete": True,
                "raw_rows_available": 100,
                "path": f"private/cells/{alias}/gsm8k.jsonl",
                "sha256": verifier.sha256_file(path),
                "validation_protocol": "P4R1",
            },
            "truncated": sum(bool(row.get("truncated")) for row in rows),
            "truncated_observed_in_surviving_rows": sum(bool(row.get("truncated")) for row in rows),
            "errors": sum(row.get("api_error") is not None for row in rows),
            "errors_observed_in_surviving_rows": sum(
                row.get("api_error") is not None for row in rows
            ),
            "completion_tokens_missing": sum(row.get("completion_tokens") is None for row in rows),
            "chat_template_kwargs_dropped": sum(bool(row.get("chat_template_kwargs_dropped")) for row in rows),
            "item_id_set_sha256": hashlib.sha256(
                "\n".join(sorted(row["id"] for row in rows)).encode("utf-8")
            ).hexdigest(),
        }
        raw_by_key[key] = rows

    comparisons = {
        name: verifier.build_comparison(name, low, healthy, cells, raw_by_key)
        for name, (low, healthy) in COMPARISONS.items()
    }
    adjusted = verifier.holm_adjust(
        {name: value["paired"]["mcnemar_exact_two_sided"] for name, value in comparisons.items()}
    )
    for name, value in adjusted.items():
        comparisons[name]["paired"]["holm_adjusted_mcnemar_p"] = value
    public_root = promoted / "public-candidate"
    public_validation = json.loads(
        (public_root / "validation-receipt.json").read_text(encoding="utf-8")
    )
    public_toolchain = json.loads(
        (public_root / "toolchain-receipt.json").read_text(encoding="utf-8")
    )
    public_host = json.loads(
        (public_root / "host-receipt.json").read_text(encoding="utf-8")
    )
    public_post_models = json.loads(
        (public_root / "post-model-verification.json").read_text(encoding="utf-8")
    )
    public_plan = json.loads(
        (public_root / "protocol-plan.json").read_text(encoding="utf-8")
    )
    public_approval = json.loads(
        (public_root / "protocol-plan-approval.json").read_text(encoding="utf-8")
    )
    public_cells = {
        alias: json.loads(
            (public_root / "cells" / alias / "receipt.json").read_text(encoding="utf-8")
        )
        for alias in MAPPING
    }
    model_keys = {
        "P4R1-MIXTRAL-IQ1M": "mixtral_iq1_m",
        "P4R1-MIXTRAL-IQ2XXS": "mixtral_iq2_xxs",
        "P4R1-MIXTRAL-Q4KM": "mixtral_q4_k_m",
        "P4R1-QWEN3MOE-IQ1M": "qwen3moe_iq1_m",
        "P4R1-QWEN3MOE-IQ2XXS": "qwen3moe_iq2_xxs",
        "P4R1-QWEN3MOE-Q4KM": "qwen3moe_q4_k_m",
    }
    models: dict[str, Any] = {}
    for alias, key in model_keys.items():
        model = json.loads(json.dumps(baseline["models"][key]))
        public_model = public_cells[alias]["model"]
        if (
            model["total_model_file_bytes"] != public_model["bytes"]
            or model["huggingface_source"]["artifact_basename"] != public_model["filename"]
            or model["huggingface_source"]["lfs_oid_sha256"] != public_model["sha256"]
        ):
            raise RuntimeError(f"{alias}: audited static artifact metadata differs from P4R1")
        model["p4r1_artifact_identity"] = public_model
        model["static_metadata_source_sha256"] = BASELINE_SHA256
        models[key] = model
    placements = {
        alias: {
            "requested": public_cells[alias]["requested"],
            "observed_offloaded_layers": public_cells[alias]["observed_offloaded_layers"],
            "observed_gpu_memory_mib": public_cells[alias]["observed_gpu_memory_mib"],
        }
        for alias in MAPPING
    }
    outcome_signature = {
        "cell_correct": {alias: cells[key]["correct"] for alias, key in MAPPING.items()},
        "q4_minus_low_delta_pp": {
            name: comparison["delta_pp"] for name, comparison in comparisons.items()
        },
        "signed_result_branch": {
            name: (
                "q4_higher" if comparison["delta_pp"] > 0
                else "low_tier_higher" if comparison["delta_pp"] < 0
                else "tied"
            )
            for name, comparison in comparisons.items()
        },
    }
    claims = {
        "schema_version": "paper4-p4r1-claims-v1",
        "scope": "six-cell P4R1 corrective exploratory GSM8K validation; no inference executed by this builder",
        "historical_audit_only": {
            "baseline_claims_sha256": BASELINE_SHA256,
            "publication_status": "invalidated as primary evidence; forensic context only",
            "historical_cells_copied_into_p4r1": False,
        },
        "study_integrity": {
            "primary_protocol": "P4R1",
            "primary_item_count": 100,
            "primary_gsm8k_cells": list(MAPPING.values()),
            "all_six_primary_cells_share_item_ids": True,
            "independence": "not independent: historical outcomes, item IDs, and hypothesis were known",
        },
        "dataset_provenance": {
            "source_revision": "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc",
            "source_sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
            "subset_sha256": public_validation["dataset_sha256"],
            "sampling_seed": 42,
            "sampling_algorithm": "sorted(random.Random(42).sample(range(1319), 100))",
            "inference_sampler_seed": "4,200,000 plus integer GSM8K item suffix",
        },
        "models": models,
        "eval_cells": cells,
        "comparisons": comparisons,
        "analysis_contract": {
            "paired_tests": "four exact two-sided McNemar tests",
            "interval": "Newcombe paired hybrid-score method 10",
            "multiplicity": "Holm across exactly four comparisons",
            "prospective_analysis_source_inventory": public_plan["analysis_source_inventory"],
        },
        "protocol_verification": {
            "generation": {
                "context_tokens_all_cells": 4096,
                "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "max_tokens": 2048, "reasoning_effort_request": "medium",
                "concurrency": 1,
                "per_item_sampler_seed": "4200000 + integer suffix of gsm8k_<suffix>",
                "source_hashes": public_validation["protocol_sources"],
            },
            "scoring": {
                "numeric_absolute_tolerance": 0.0001,
                "numeric_relative_tolerance": 0.0001,
                "private_output_and_gold_recomputed": True,
                "exact_api_error_telemetry_validated": True,
            },
            "identity": {
                "plan_sha256": public_validation["prospective_plan"]["sha256"],
                "plan_approval_sha256": public_validation[
                    "prospective_plan_approval"
                ]["sha256"],
                "toolchain_sha256": public_validation["toolchain_receipt"]["sha256"],
                "host_sha256": public_validation["host_receipt"]["sha256"],
                "post_campaign_model_verification_sha256": public_validation[
                    "post_model_verification"
                ]["sha256"],
                "post_campaign_artifact_content_verified": (
                    public_post_models.get("artifact_count") == 6
                ),
                "startup_completion_shutdown_receipts": True,
                "placements": placements,
            },
        },
        "environment": {
            "p4r1_observed_host": public_host,
            "p4r1_observed_toolchain": public_toolchain,
        },
        "truncation_inventory": {
            key: {
                "n": cell["n"], "truncated": cell["truncated"],
                "errors": cell["errors"],
                "chat_template_kwargs_dropped": cell["chat_template_kwargs_dropped"],
            }
            for key, cell in cells.items()
        },
        "outcome_signature": outcome_signature,
        "p4r1_validation": {
        "status": "promoted-private-evidence",
        "independence": "not independent: the item set and hypothesis were seen before P4R1",
        "context_tokens_all_cells": 4096,
        "sampler_seed": "4,200,000 plus the integer GSM8K item suffix, shared across artifacts",
        "model_manifest_sha256": promotion["model_manifest_sha256"],
        "promotion_receipt": "promotion-receipt.json",
        "promotion_receipt_sha256": promotion_receipt_sha256,
        "baseline_claims_sha256": BASELINE_SHA256,
        "claims_analysis_sha256": VERIFY_SHA256,
        "public_checker_sha256": sha256_file(PUBLIC_CHECKER),
        },
    }
    tex = render_p4r1_tex(claims)
    report_lines = [
        "# Paper 4 P4R1 claims report",
        "",
        "This report was rebuilt from the privacy-checked promoted P4R1 evidence. The builder ran no inference.",
        "P4R1 is exploratory, not independent: the item set and hypothesis were seen before the validation run.",
        "",
        "| Cell | Correct/n |",
        "|---|---:|",
    ]
    for alias, key in MAPPING.items():
        report_lines.append(f"| `{alias}` | {cells[key]['correct']}/100 |")
    report_lines.extend(["", "All four paired comparisons and Holm adjustments are in `claims.json`.", ""])
    return claims, tex, "\n".join(report_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted", required=True, type=Path)
    parser.add_argument("--promotion-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite P4R1 derived output: {output}")
    if len(args.promotion_receipt_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.promotion_receipt_sha256
    ):
        raise SystemExit("--promotion-receipt-sha256 must be 64 lowercase hexadecimal characters")
    claims, tex, report = build(
        args.promoted.resolve(strict=True), args.promotion_receipt_sha256
    )
    output.mkdir(parents=True)
    (output / "claims.json").write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "claims.tex").write_text(tex, encoding="utf-8")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote P4R1 claims to {output}; no inference or historical overwrite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
