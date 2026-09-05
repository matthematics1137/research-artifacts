"""Read-only correction checks; no new measurements or inferred attempt counts."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABEL = "UD-IQ2_XXS-mtpgpu-effmed"


def rows(path: Path) -> list[dict]:
    result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in result):
        raise ValueError(f"Expected JSON objects: {path}")
    return result


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mtp_metrics() -> dict:
    text = (ROOT / f"testsuite/results/server_{LABEL}.log").read_text()
    entries = [(float(ms), int(tokens), float(rate)) for ms, tokens, rate in re.findall(
        r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens \([^\n]*?([\d.]+) tokens per second\)", text
    )]
    results = ROOT / "testsuite/evals/results" / LABEL
    math_rows = rows(results / "math25.jsonl")
    code_rows = rows(results / "humaneval_plus.jsonl")
    check((len(entries), len(math_rows), len(code_rows)) == (45, 25, 20), "MTP denominators changed")
    check([e[1] for e in entries] == [r["completion_tokens"] for r in math_rows + code_rows], "MTP log/result token sequence differs")
    selected = entries[:25]
    check(all(ms > 0 and n > 0 and math.isfinite(rate) for ms, n, rate in entries), "Invalid timing")
    # Rates are server-reported: the engine's count convention can differ by
    # one token from n/time. Aggregate rate explicitly uses retained n/time.
    return {
        "decode_min": min(rate for _, _, rate in selected),
        "decode_max": max(rate for _, _, rate in selected),
        "decode_aggregate": sum(n for _, n, _ in selected) / sum(ms / 1000 for ms, _, _ in selected),
        "response_aggregate": sum(r["completion_tokens"] for r in math_rows) / sum(r["wall_s"] for r in math_rows),
    }


def verify() -> None:
    values = mtp_metrics()
    check({k: round(v, 2) for k, v in values.items()} == {
        "decode_min": 41.09, "decode_max": 50.84,
        "decode_aggregate": 45.35, "response_aggregate": 44.96,
    }, "MTP correction values differ")
    results = ROOT / "testsuite/evals/results"
    q4 = rows(results / "UD-Q4_K_XL-ngl33-effmed/math25.jsonl")
    errors = [r for r in q4 if r.get("error")]
    check({r["id"] for r in errors} == {"math_intermediate_algebra_1411", "math_counting_&_probability_731"}, "Q4 error IDs differ")
    check(all(r["correct"] is False and r["truncated"] is False and r["error"].startswith("TimeoutError:") for r in errors), "Q4 error classification differs")
    check(round(sum(r["wall_s"] for r in errors), 2) == 2404.36, "Q4 retry denominator differs")
    check(round(sum(r["wall_s"] for r in q4), 2) == 7313.90, "Q4 total denominator differs")
    gsm = rows(ROOT / "testsuite/evals/datasets/gsm8k_100.jsonl")[:50]
    expected_ids = [r["id"] for r in gsm]
    check(len(set(expected_ids)) == 50, "Duplicate GSM sample IDs")
    for path in results.glob("*/gsm8k.jsonl"):
        check([r["id"] for r in rows(path)] == expected_ids, "GSM retained selection differs")
    indices = [int(item.removeprefix("gsm8k_")) for item in expected_ids]
    check((min(indices), max(indices)) == (13, 563), "GSM source-index range differs")
    phase1 = rows(ROOT / "testsuite/results/phase1.jsonl")
    tg = {(r["model"], r["ngl"]): r["tok_s"] for r in phase1 if r.get("stage") == "bench" and r.get("test") == "tg" and r.get("ok")}
    servers = [r for r in phase1 if r.get("stage") == "servertest" and r.get("ok") and r.get("mtp") == "off"]
    check(len(servers) == 14, "Draft-off server matrix differs")
    maximum = max(abs(r["tg_tok_s"] - tg[(r["model"], r["ngl"])]) for r in servers)
    check(round(maximum, 2) == 0.73, "Server/tg128 caption bound differs")
    print("CORRECTION CHECKS PASSED: MTP alignment; Q4 errors/retry time; GSM selection; server/tg128 bound")


if __name__ == "__main__":
    verify()
