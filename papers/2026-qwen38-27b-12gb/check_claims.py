#!/usr/bin/env python3
"""Verify headline paper values from the frozen public evidence.

This analysis-only program does not contact a model endpoint or write results.
It fails on missing rows, changed denominators, unexpected scores, or changed
statistical conclusions.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "testsuite/evals/results"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    require(abs(actual - expected) <= tolerance, f"{actual} != {expected} ± {tolerance}")


def lenient_norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"</?final[ _]answer>", "", text)
    text = text.replace("$", "").strip()
    text = re.sub(r"^\\\(|\\\)$", "", text).strip()
    for source, destination in (
        ("√", "\\sqrt"), ("π", "\\pi"), ("∪", "\\cup"), ("−", "-"),
        ("≤", "\\le"), ("≥", "\\ge"), ("×", "\\times"),
    ):
        text = text.replace(source, destination)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\sqrt(\d|[a-zA-Z])", r"\\sqrt{\1}", text)
    text = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\^\{?\\circ\}?|°", "", text)
    text = re.sub(r"_\{?\d+\}?$", "", text)
    text = re.sub(r"^[a-zA-Z]\s*=\s*", "", text)
    text = re.sub(r"\\text\{[^}]*\}", "", text)
    text = re.sub(r"[\s,]+", "", text)
    text = re.sub(r"^\((\d+(?:\.\d+)?)\)/\((\d+(?:\.\d+)?)\)$", r"\1/\2", text)
    return text


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
    prediction = lenient_norm(predicted)
    truth = lenient_norm(expected)
    if prediction and prediction == truth:
        return True
    prediction_number = numeric(prediction)
    truth_number = numeric(truth)
    return (
        prediction_number is not None
        and truth_number is not None
        and abs(prediction_number - truth_number) < 1e-6
    )


def math_scores() -> tuple[dict[str, tuple[int, int, int]], dict[str, dict[str, bool]]]:
    truth_rows = load_jsonl(ROOT / "testsuite/evals/datasets/math25.jsonl")
    truth = {row["id"]: row["answer"] for row in truth_rows}
    scores: dict[str, tuple[int, int, int]] = {}
    maps: dict[str, dict[str, bool]] = {}
    for path in sorted(RESULTS.glob("*/math25.jsonl")):
        rows = load_jsonl(path)
        require(len(rows) == 25, f"partial MATH run: {path}")
        raw = sum(bool(row["correct"]) for row in rows)
        alternate = 0
        upgrades = 0
        item_map: dict[str, bool] = {}
        for row in rows:
            expected = row.get("expected_answer")
            if expected in (None, "None"):
                expected = truth[row["id"]]
            accepted = bool(row["correct"] or equivalent(row.get("predicted_answer"), expected))
            alternate += accepted
            upgrades += accepted and not row["correct"]
            item_map[row["id"]] = accepted
        scores[path.parent.name] = (raw, alternate, upgrades)
        maps[path.parent.name] = item_map
    return scores, maps


def fisher_two_sided(success_a: int, total_a: int, success_b: int, total_b: int) -> float:
    successes = success_a + success_b
    total = total_a + total_b

    def probability(value: int) -> float:
        return (
            math.comb(successes, value)
            * math.comb(total - successes, total_a - value)
            / math.comb(total, total_a)
        )

    minimum = max(0, total_a - (total - successes))
    maximum = min(total_a, successes)
    observed = probability(success_a)
    return sum(
        probability(value)
        for value in range(minimum, maximum + 1)
        if probability(value) <= observed + 1e-15
    )


def mcnemar_exact(only_a: int, only_b: int) -> float:
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def only_counts(first: dict[str, bool], second: dict[str, bool]) -> tuple[int, int]:
    require(first.keys() == second.keys(), "paired item IDs differ")
    return (
        sum(first[key] and not second[key] for key in first),
        sum(second[key] and not first[key] for key in first),
    )


def check_phase1() -> None:
    rows = load_jsonl(ROOT / "testsuite/results/phase1.jsonl")
    require(len(rows) == 47, "phase1 row count changed")
    expected = {
        ("Qwen3.8-27B-UD-IQ2_XXS.gguf", 99): 25.48,
        ("Qwen3.8-27B-UD-IQ2_S.gguf", 99): 24.36,
        ("Qwen3.8-27B-UD-Q2_K_XL.gguf", 61): 16.62,
        ("Qwen3.8-27B-IQ2_S.gguf", 59): 13.83,
        ("Qwen3.8-27B-UD-IQ3_XXS.gguf", 54): 11.31,
        ("Qwen3.8-27B-UD-IQ3_S.gguf", 49): 8.17,
        ("Qwen3.8-27B-UD-Q4_K_XL.gguf", 33): 4.87,
        ("Qwen3.8-27B-UD-IQ2_S.gguf", 0): 3.34,
    }
    observed = {
        (row["model"], row["ngl"]): float(row["tok_s"])
        for row in rows
        if row.get("stage") == "bench" and row.get("test") == "tg" and row.get("ok")
    }
    require(observed == expected, f"controlled throughput changed: {observed}")


def check_summaries() -> None:
    summary = load_jsonl(ROOT / "testsuite/evals/results/summary.jsonl")
    require(len(summary) == 33, "filtered summary must contain 33 Qwen rows")
    forbidden = {"FLASH-IQ1S-effmed", "MIXTRAL-IQ1M", "MIXTRAL-IQ2XXS", "MIXTRAL-Q4KM", "SMOKE-IQ2XXS-mtpgpu-effmed"}
    require(not forbidden & {str(row["label"]) for row in summary}, "unrelated campaign rows leaked")
    expected = {
        ("UD-IQ2_S-effmed", "mmlu"): (200, 0.875, 36.41),
        ("EXL3-2.0bpw-effmed", "mmlu"): (200, 0.91, 25.42),
        ("UD-IQ2_S-effmed", "needle"): (6, 0.6667, 628.12),
        ("EXL3-2.0bpw-effmed", "needle"): (6, 1.0, 34.78),
    }
    latest = {(row["label"], row["suite"]): row for row in summary if not row.get("partial")}
    for key, (count, accuracy, seconds) in expected.items():
        row = latest[key]
        require(row["n"] == count, f"denominator changed for {key}")
        close(float(row["accuracy"]), accuracy)
        close(float(row["mean_wall_s"]), seconds, 0.01)


def check_math_and_statistics() -> None:
    scores, maps = math_scores()
    expected = {
        "UD-IQ2_XXS-mtpgpu-effmed": (12, 16, 4),
        "UD-IQ2_S-effmed": (14, 23, 9),
        "UD-IQ2_S-effxhigh": (17, 21, 4),
        "UD-IQ3_XXS-ngl54-effmed": (22, 24, 2),
        "UD-Q4_K_XL-ngl33-effmed": (18, 23, 5),
        "EXL3-2.0bpw-effmed": (20, 24, 4),
        "bart-IQ2_S-ngl59-effmed": (20, 23, 3),
    }
    require(scores == expected, f"MATH scoring changed: {scores}")
    close(fisher_two_sided(16, 25, 23, 25), 0.0366, 0.001)
    close(fisher_two_sided(14, 20, 18, 20), 0.235, 0.003)
    close(fisher_two_sided(30, 45, 41, 45), 0.0088, 0.001)
    close(fisher_two_sided(9, 25, 2, 25), 0.0366, 0.001)
    close(mcnemar_exact(9, 0), 0.00390625, 1e-12)
    x_only, s_only = only_counts(
        maps["UD-IQ2_XXS-mtpgpu-effmed"], maps["UD-IQ2_S-effmed"]
    )
    require((x_only, s_only) == (0, 7), "paired MATH cliff changed")
    close(mcnemar_exact(x_only, s_only), 0.015625, 1e-12)


def check_response_rates() -> None:
    math_correct = {
        "UD-IQ2_XXS-mtpgpu-effmed": (16, 0.6862),
        "UD-IQ2_S-effmed": (23, 0.7274),
        "EXL3-2.0bpw-effmed": (24, 0.8328),
        "bart-IQ2_S-ngl59-effmed": (23, 0.3821),
        "UD-IQ3_XXS-ngl54-effmed": (24, 0.3338),
        "UD-Q4_K_XL-ngl33-effmed": (23, 0.1887),
    }
    for label, (correct, expected_rate) in math_correct.items():
        rows = load_jsonl(RESULTS / label / "math25.jsonl")
        close(60 * correct / sum(float(row["wall_s"]) for row in rows), expected_rate)
    he_expected = {
        "UD-IQ2_XXS-mtpgpu-effmed": 1.9893,
        "UD-IQ2_S-effmed": 1.4009,
        "EXL3-2.0bpw-effmed": 2.2695,
        "bart-IQ2_S-ngl59-effmed": 0.8604,
        "UD-IQ3_XXS-ngl54-effmed": 0.7269,
        "UD-Q4_K_XL-ngl33-effmed": 0.4270,
    }
    for label, expected_rate in he_expected.items():
        rows = load_jsonl(RESULTS / label / "humaneval_plus.jsonl")
        rate = 60 * sum(bool(row["correct"]) for row in rows) / sum(float(row["wall_s"]) for row in rows)
        close(rate, expected_rate)


def check_matched_workload() -> None:
    exl3_text = (ROOT / "testsuite/results/server_EXL3-2.0.log").read_text(encoding="utf-8").replace("\n", " ")
    exl3 = re.findall(r"(\d+) tokens generated in ([\d.]+) seconds\s*\(.*?Generate:\s*([\d.]+)\s*T/s", exl3_text)
    bart = re.findall(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens", (ROOT / "testsuite/results/server_bart_hard.log").read_text(encoding="utf-8"))
    require(len(exl3) == len(bart) == 45, "matched-workload request count changed")

    def exl3_rate(rows: list[tuple[str, str, str]]) -> float:
        return sum(int(tokens) for tokens, _, _ in rows) / sum(int(tokens) / float(rate) for tokens, _, rate in rows)

    def bart_rate(rows: list[tuple[str, str]]) -> float:
        return 1000 * sum(int(tokens) for _, tokens in rows) / sum(float(milliseconds) for milliseconds, _ in rows)

    values = (exl3_rate(exl3[:25]), bart_rate(bart[:25]), exl3_rate(exl3[25:]), bart_rate(bart[25:]))
    for actual, expected in zip(values, (22.4395, 13.7051, 30.6997, 13.8151)):
        close(actual, expected)
    close(values[0] / values[1], 1.6373, 0.001)


def check_validation_pairing() -> None:
    first_rows = load_jsonl(RESULTS / "UD-IQ2_S-effmed/mmlu.jsonl")
    second_rows = load_jsonl(RESULTS / "EXL3-2.0bpw-effmed/mmlu.jsonl")
    first = {row["id"]: bool(row["correct"]) for row in first_rows}
    second = {row["id"]: bool(row["correct"]) for row in second_rows}
    first_only, second_only = only_counts(first, second)
    both = sum(first[key] and second[key] for key in first)
    neither = sum(not first[key] and not second[key] for key in first)
    require((both, first_only, second_only, neither) == (171, 4, 11, 14), "MMLU pairing changed")
    close(mcnemar_exact(first_only, second_only), 0.1185, 0.001)


def check_model_manifest() -> None:
    manifest = json.loads((ROOT / "environment/model_artifacts.json").read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    require(len(records) == 10, "model artifact record count changed")
    require(all(re.fullmatch(r"[0-9a-f]{40}", row["revision"]) for row in records), "model revision is not pinned")
    require(all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in records), "model hash is not pinned")
    exl3_bytes = sum(row["bytes"] for row in records if row["artifact_id"] == "EXL3-SC_2.00bpw_H3")
    require(exl3_bytes == 10196126231, "EXL3 aggregate bytes changed")
    close(exl3_bytes * 8 / 26_895_998_464, 3.032756, 0.000001)


def main() -> int:
    check_phase1()
    check_summaries()
    check_math_and_statistics()
    check_response_rates()
    check_matched_workload()
    check_validation_pairing()
    check_model_manifest()
    print("ALL CLAIM CHECKS PASSED")
    print("Controlled GGUF decode: 25.48 -> 4.87 tok/s across the fit/offload ladder")
    print("MATH cliff: 16/25 vs 23/25 alternate-normalizer; Fisher p=0.037")
    print("Matched footprint MATH throughput: 22.44 vs 13.71 tok/s (1.64x)")
    print("Validation MMLU: 175/200 vs 182/200; paired exact p=0.119")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
