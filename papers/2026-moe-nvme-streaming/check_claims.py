#!/usr/bin/env python3
"""Verify Paper 3's headline values from the sanitized evidence bundle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT / "generated/data"
EXPECTED_HASHES = {
    "claims.json": "9aadfdf99e233cf2469aa0a5b21a557cc62a4fa91d2db145a1b4b8673f1460d5",
    "timing_measurements.jsonl": "11d92e3b50e1439ed98a653492c05a25b8c1ae0acb1687bfe81acb3059376486",
    "quality_outcomes.jsonl": "63fbe07eb4fad81659a73415798b06f5f8840077f0d1211d26b99d5f2f1b52e5",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    require(abs(actual - expected) <= tolerance, f"{actual} != {expected} ± {tolerance}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lenient_norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"</?final[ _]answer>", "", text).replace("$", "").strip()
    text = re.sub(r"^\\\(|\\\)$", "", text).strip()
    for source, destination in (
        ("√", "\\sqrt"), ("π", "\\pi"), ("∪", "\\cup"), ("−", "-"),
        ("≤", "\\le"), ("≥", "\\ge"), ("×", "\\times"),
    ):
        text = text.replace(source, destination)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\displaystyle", "").strip()
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\sqrt(\d|[a-zA-Z])", r"\\sqrt{\1}", text)
    text = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\^\{?\\circ\}?|°", "", text)
    text = re.sub(r"_\{?\d+\}?$", "", text)
    text = re.sub(r"^[a-zA-Z]\s*=\s*", "", text)
    text = re.sub(r"\\text\{[^}]*\}", "", text)
    text = re.sub(r"[\s,]+", "", text)
    return re.sub(r"^\((\d+(?:\.\d+)?)\)/\((\d+(?:\.\d+)?)\)$", r"\1/\2", text)


def numeric(value: str) -> float | None:
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:/([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?", value)
    if match is None:
        return None
    numerator = float(match.group(1))
    if match.group(2) is None:
        return numerator
    denominator = float(match.group(2))
    return None if denominator == 0 else numerator / denominator


def equivalent(predicted: Any, expected: Any) -> bool:
    prediction, truth = lenient_norm(predicted), lenient_norm(expected)
    if prediction and prediction == truth:
        return True
    first, second = numeric(prediction), numeric(truth)
    return first is not None and second is not None and abs(first - second) < 1e-6


def main() -> int:
    for name, expected in EXPECTED_HASHES.items():
        require(sha256(DATA / name) == expected, f"changed or missing evidence: data/{name}")
    claims = json.loads((DATA / "claims.json").read_text(encoding="utf-8"))
    timing = load_jsonl(DATA / "timing_measurements.jsonl")
    quality = load_jsonl(DATA / "quality_outcomes.jsonl")

    expected_rates = {
        "flash_initial_loaded_label": [9.4, 10.4, 10.0],
        "flash_quiet_gate": [14.7, 14.4, 13.9],
        "gptoss_initial_loaded_label": [9.3, 8.7, 9.2],
    }
    for group, expected in expected_rates.items():
        rows = [row for row in timing if row["kind"] == "generation" and row["group"] == group]
        actual = [row["generation_tokens_per_second"] for row in rows]
        require(actual == expected, f"{group} timing changed: {actual}")
        close(sum(actual) / len(actual), claims["generation_runs"][group]["generation_tokens_per_second_summary"]["mean"])
        require(all(row["requested_decode_tokens"] == 256 for row in rows), "decode request changed")

    flash, gptoss = claims["artifacts"]["flash"], claims["artifacts"]["gptoss"]
    require(flash["artifact"]["file_bytes"] == 72_546_461_344, "Flash bytes changed")
    require(gptoss["artifact"]["file_bytes"] == 63_387_346_208, "gpt-oss bytes changed")
    close(
        8 * flash["artifact"]["file_bytes"] / flash["artifact"]["logical_tensor_elements"],
        flash["artifact"]["effective_file_bits_per_logical_element"],
    )
    close(
        8 * gptoss["artifact"]["file_bytes"] / gptoss["artifact"]["logical_tensor_elements"],
        gptoss["artifact"]["effective_file_bits_per_logical_element"],
    )
    close(flash["artifact"]["effective_file_bits_per_logical_element"], 3.2799757)
    close(gptoss["artifact"]["effective_file_bits_per_logical_element"], 4.340515537)
    close(flash["routed_expert_storage"]["gib"], 37.109375)
    close(gptoss["routed_expert_storage"]["gib"], 56.878967285)

    model_manifest = json.loads((ROOT / "environment/model_artifacts.json").read_text(encoding="utf-8"))
    model_rows = model_manifest.get("artifacts", [])
    require(len(model_rows) == 4, "expected four pinned model files")
    expected_models = {
        "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf": (
            10_946_624,
            "88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd",
            "qwen-community-1.0",
        ),
        "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf": (
            49_990_818_368,
            "3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6",
            "qwen-community-1.0",
        ),
        "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf": (
            22_544_696_352,
            "0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a",
            "qwen-community-1.0",
        ),
        "gpt-oss-120b-MXFP4.gguf": (
            63_387_346_208,
            "582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d",
            "apache-2.0",
        ),
    }
    actual_models = {
        row.get("filename"): (row.get("bytes"), row.get("sha256"), row.get("license"))
        for row in model_rows
    }
    require(actual_models == expected_models, "pinned model-file identity changed")
    flash_rows = [row for row in model_rows if row.get("upstream_repo") == "unsloth/Qwen3.8-Flash-Next-GGUF"]
    gptoss_rows = [row for row in model_rows if row.get("upstream_repo") == "ggml-org/gpt-oss-120b-GGUF"]
    require(sum(row["bytes"] for row in flash_rows) == flash["artifact"]["file_bytes"], "Flash shard sum changed")
    require(sum(row["bytes"] for row in gptoss_rows) == gptoss["artifact"]["file_bytes"], "gpt-oss file size changed")

    def rows(label: str, suite: str) -> list[dict[str, Any]]:
        return [row for row in quality if row["label"] == label and row["suite"] == suite]

    expected_raw = {
        ("FLASH-IQ1S-effmed", "math25"): (18, 25),
        ("FLASH-IQ1S-effmed", "gsm8k"): (48, 50),
        ("FLASH-IQ1S-effmed", "humaneval_plus"): (16, 17),
        ("GPTOSS-MXFP4-effmed", "math25"): (12, 25),
        ("GPTOSS-MXFP4-effmed", "gsm8k"): (46, 50),
        ("GPTOSS-MXFP4-effmed", "humaneval_plus"): (18, 20),
    }
    for cell, expected in expected_raw.items():
        selected = rows(*cell)
        actual = (sum(bool(row["correct"]) for row in selected), len(selected))
        require(actual == expected, f"score changed for {cell}: {actual}")

    for label in ("FLASH-IQ1S-effmed", "GPTOSS-MXFP4-effmed"):
        selected = rows(label, "math25")
        require(
            sum(bool(row.get("errored")) for row in selected) == 1,
            f"expected one MATH request error for {label}",
        )

    for label, expected in (("FLASH-IQ1S-effmed", 23), ("GPTOSS-MXFP4-effmed", 20)):
        selected = rows(label, "math25")
        accepted = sum(bool(row["correct"] or equivalent(row.get("predicted_answer"), row.get("expected_answer"))) for row in selected)
        require(accepted == expected, f"lenient MATH score changed for {label}: {accepted}")

    print("ALL PAPER 3 CLAIM CHECKS PASSED")
    print("Flash: 67.564 GiB, 9.4–14.7 tok/s; gpt-oss: 59.034 GiB, 8.7–9.3 tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
