#!/usr/bin/env python3
"""Issue one declared, unmeasured P2R1 evaluation warmup request."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from identity_guard import receipt_sha256, utc_now, validate_receipt  # noqa: E402
from r1_safety import RunSafetyError, write_exclusive  # noqa: E402
from run_grid import (  # noqa: E402
    PROMPT, chat, transport_evidence, validate_api_evidence, visible_projection,
)


WARMUP_WINDOW = (
    "This is a deterministic unscored warmup passage for the Paper 2 recovery. "
    "Its only extractive answer token is P2R1-WARMUP. " * 32
)
WARMUP_QUESTION = "What is the exact warmup answer token?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raw = json.dumps({
            "model": "P2R1-SYNTHETIC",
            "choices": [{"message": {"content": "ANSWER: P2R1-WARMUP"}}],
            "usage": {"completion_tokens": 3},
        }).encode()
        evidence = {
            **transport_evidence(raw, status=200, reason="OK", headers=[]),
            "response_model": "P2R1-SYNTHETIC",
            "usage": {"completion_tokens": 3},
            "raw_message_content": "ANSWER: P2R1-WARMUP",
            "raw_message_content_sha256": hashlib.sha256(
                b"ANSWER: P2R1-WARMUP"
            ).hexdigest(),
            "visible_response": visible_projection("ANSWER: P2R1-WARMUP"),
            "visible_response_sha256": hashlib.sha256(
                b"ANSWER: P2R1-WARMUP"
            ).hexdigest(),
        }
        response, usage, model = validate_api_evidence(evidence, success=True)
        if response != "ANSWER: P2R1-WARMUP" or usage != {"completion_tokens": 3} \
                or model != "P2R1-SYNTHETIC":
            raise RunSafetyError("warmup evidence self-test failed")
        print("P2R1 warmup static evidence self-test passed")
        return 0
    if args.identity_receipt is None or args.output is None:
        raise RunSafetyError("--identity-receipt and --output are required")
    receipt = validate_receipt(args.identity_receipt, full_artifact_hash=False)
    prompt = PROMPT.format(window=WARMUP_WINDOW, q=WARMUP_QUESTION)
    started = utc_now()
    t0 = time.monotonic()
    api_evidence = chat(
        base_url=receipt["base_url"],
        model_id=receipt["expected_model_id"],
        prompt=prompt,
        max_tokens=2048,
        temperature=0.0,
        seed=42,
        timeout=600,
    )
    response, usage, response_model = validate_api_evidence(
        api_evidence, success=True
    )
    wall = time.monotonic() - t0
    validate_receipt(args.identity_receipt, full_artifact_hash=False)
    if response_model != receipt["expected_model_id"]:
        raise RunSafetyError(
            f"warmup response model mismatch: {response_model!r} != "
            f"{receipt['expected_model_id']!r}"
        )
    if "P2R1-WARMUP" not in response:
        raise RunSafetyError("deterministic warmup response omitted its extractive token")
    write_exclusive(args.output, {
        "schema": "paper2-r1-unmeasured-warmup-v1",
        "started_at": started,
        "finished_at": utc_now(),
        "identity_receipt": str(args.identity_receipt.resolve()),
        "identity_receipt_sha256": receipt_sha256(args.identity_receipt),
        "model_id": response_model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "api_evidence": api_evidence,
        "extractive_token_verified": True,
        "usage": usage,
        "wall_s_unmeasured": round(wall, 4),
        "sampling": {"temperature": 0.0, "seed": 42, "max_tokens": 2048},
        "excluded_from_analysis": True,
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 WARMUP ABORTED: {exc}") from exc
