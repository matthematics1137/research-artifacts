#!/usr/bin/env python3
"""Apply Paper 3's disclosed, safe lenient MATH rescore to JSONL results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
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
    return re.sub(
        r"^\((\d+(?:\.\d+)?)\)/\((\d+(?:\.\d+)?)\)$", r"\1/\2", text
    )


def numeric(value: str) -> float | None:
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:/([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?",
        value,
    )
    if match is None:
        return None
    numerator = float(match.group(1))
    if match.group(2) is None:
        return numerator
    denominator = float(match.group(2))
    return None if denominator == 0 else numerator / denominator


def equivalent(predicted: Any, expected: Any) -> bool:
    prediction, truth = normalize(predicted), normalize(expected)
    if prediction and prediction == truth:
        return True
    first, second = numeric(prediction), numeric(truth)
    return first is not None and second is not None and abs(first - second) < 1e-6


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object JSONL row in {path}")
    if any("suite" in row for row in rows):
        rows = [row for row in rows if row.get("suite") == "math25"]
    return rows


def selftest() -> int:
    cases = [
        ("$\\pi$", "\\pi", True),
        ("x = 5", "5", True),
        ("\\frac{1}{2}", "0.5", True),
        ("6", "5", False),
        ("", "5", False),
    ]
    for predicted, expected, wanted in cases:
        actual = equivalent(predicted, expected)
        if actual != wanted:
            raise AssertionError((predicted, expected, actual, wanted))
    print("MATH RESCORER SELFTEST PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="MATH result JSONL files")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.files:
        parser.error("provide at least one result JSONL file, or use --selftest")

    for path in args.files:
        rows = load_rows(path)
        if not rows:
            raise SystemExit(f"no MATH rows found in {path}")
        missing = [index for index, row in enumerate(rows) if "correct" not in row]
        if missing:
            raise SystemExit(f"missing original correctness in {path} rows: {missing[:5]}")
        raw = sum(bool(row["correct"]) for row in rows)
        accepted = sum(
            bool(
                row["correct"]
                or (
                    "predicted_answer" in row
                    and "expected_answer" in row
                    and equivalent(row["predicted_answer"], row["expected_answer"])
                )
            )
            for row in rows
        )
        label = rows[0].get("label") or path.parent.name
        print(json.dumps({
            "file": str(path),
            "label": label,
            "n": len(rows),
            "original_correct": raw,
            "lenient_correct": accepted,
            "upgrades": accepted - raw,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
