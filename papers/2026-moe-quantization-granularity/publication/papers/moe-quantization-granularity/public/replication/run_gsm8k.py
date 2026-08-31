#!/usr/bin/env python3
"""Run the post-audit Paper 4 P4R1 GSM8K validation contract.

The harness is standard-library-only, sends one request at a time, proves the
launched server identity around every request, and never executes generated
code. Private rows preserve the exact response and gold value needed to audit
the parser verdict; the promotion step removes both from the public projection.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REQUEST_TIMEOUT_SECONDS = 600
MAX_ERROR_TELEMETRY_BYTES = 1024 * 1024
MAX_SUCCESS_RESPONSE_BYTES = MAX_ERROR_TELEMETRY_BYTES
NUMBER = re.compile(r"[-+]?\$?[\d][\d,]*(?:\.\d+)?")
SAMPLER_SEED_BASE = 4_200_000
DATASET_SHA256 = "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37"
EXPECTED_IDS = tuple(
    f"gsm8k_{index}"
    for index in sorted(random.Random(42).sample(range(1319), 100))
)


class IdentityFailure(RuntimeError):
    """The launched server can no longer be proven to own the endpoint."""


def http_error_telemetry(error: urllib.error.HTTPError) -> dict[str, Any]:
    try:
        body = error.read(MAX_ERROR_TELEMETRY_BYTES + 1)
    except OSError as read_error:
        raise IdentityFailure(f"could not preserve exact HTTP error telemetry: {read_error}") from read_error
    if len(body) > MAX_ERROR_TELEMETRY_BYTES:
        raise IdentityFailure("HTTP error body exceeds exact-telemetry safety limit")
    return {
        "schema": "paper4-p4r1-api-error-v1",
        "exception_type": type(error).__name__,
        "message": str(error),
        "http_status": error.code,
        "http_reason": str(error.reason),
        "url": error.geturl(),
        "headers": list(error.headers.items()) if error.headers is not None else [],
        "body_bytes": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_base64": base64.b64encode(body).decode("ascii"),
    }


def exception_telemetry(error: Exception) -> dict[str, Any]:
    saved = getattr(error, "p4_api_error", None)
    if isinstance(saved, dict):
        return saved
    message = str(error)
    if len(message.encode("utf-8")) > MAX_ERROR_TELEMETRY_BYTES:
        raise IdentityFailure("API/model exception text exceeds exact-telemetry safety limit")
    return {
        "schema": "paper4-p4r1-api-error-v1",
        "exception_type": type(error).__name__,
        "message": message,
        "http_status": None,
        "http_reason": None,
        "url": None,
        "headers": [],
        "body_bytes": 0,
        "body_sha256": hashlib.sha256(b"").hexdigest(),
        "body_base64": "",
    }


def response_parse_telemetry(
    error: Exception, raw: bytes, status: int | None, reason: str | None,
    url: str | None, headers: list[tuple[str, str]],
) -> dict[str, Any]:
    if len(raw) > MAX_SUCCESS_RESPONSE_BYTES:
        raise IdentityFailure("successful response exceeds exact-telemetry safety limit")
    return {
        "schema": "paper4-p4r1-api-error-v1",
        "exception_type": type(error).__name__,
        "message": str(error),
        "http_status": status,
        "http_reason": reason,
        "url": url,
        "headers": headers,
        "body_bytes": len(raw),
        "body_sha256": hashlib.sha256(raw).hexdigest(),
        "body_base64": base64.b64encode(raw).decode("ascii"),
    }


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
    if candidate is None:
        return None
    return candidate


def score_gsm8k(content: str, expected: float) -> bool:
    candidate = parse_gsm8k_answer(content)
    if candidate is None:
        return False
    return abs(candidate - expected) <= 1e-4 or (
        expected != 0 and abs(candidate - expected) / abs(expected) <= 1e-4
    )


def visible_content(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    return content.strip()


def thinking_tokens(message: dict[str, Any]) -> int:
    reasoning = message.get("reasoning_content")
    if not reasoning:
        content = message.get("content") or ""
        reasoning = content.split("</think>", 1)[0] if "</think>" in content else ""
    return int(len(reasoning.split()) * 1.3) if reasoning else 0


def prove_identity(args: argparse.Namespace) -> None:
    command = [
        sys.executable, "-B", str(args.server_guard), "ready",
        "--pid", str(args.server_pid), "--start-time", args.server_start_time,
        "--launcher", str(args.launcher), "--model", str(args.model_path),
        "--alias", args.model_id, "--host", args.host, "--port", str(args.port),
        "--base-url", args.base_url, "--context", str(args.context_tokens),
        "--ngl", args.ngl,
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        args.guard_wall_s += time.monotonic() - started
        raise IdentityFailure(f"server identity check could not run: {error}") from error
    args.guard_wall_s += time.monotonic() - started
    if result.returncode != 0:
        detail = result.stderr.strip()[-500:]
        raise IdentityFailure(f"server identity check failed: {detail}")


def request_completion(
    args: argparse.Namespace, prompt: str, sampler_seed: int,
    use_kwargs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    body: dict[str, Any] = {
        "model": args.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": sampler_seed,
    }
    if use_kwargs and args.reasoning_effort:
        body["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    def send() -> dict[str, Any]:
        prove_identity(args)
        request_started = time.monotonic()
        raw: bytes
        status: int | None = None
        reason: str | None = None
        response_url: str | None = None
        response_headers: list[tuple[str, str]] = []
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status_value = getattr(response, "status", None)
                if status_value is None and hasattr(response, "getcode"):
                    status_value = response.getcode()
                status = (
                    status_value
                    if isinstance(status_value, int) and not isinstance(status_value, bool)
                    else None
                )
                reason_value = getattr(response, "reason", None)
                reason = str(reason_value) if reason_value is not None else None
                if hasattr(response, "geturl"):
                    url_value = response.geturl()
                    response_url = str(url_value) if url_value is not None else None
                headers_value = getattr(response, "headers", None)
                if headers_value is not None:
                    response_headers = [
                        (str(name), str(value)) for name, value in headers_value.items()
                    ]
                raw = response.read(MAX_SUCCESS_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError:
            prove_identity(args)
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            raise IdentityFailure(f"transport loss during completion request: {error}") from error
        finally:
            args.http_wall_s += time.monotonic() - request_started
        prove_identity(args)
        if len(raw) > MAX_SUCCESS_RESPONSE_BYTES:
            raise IdentityFailure("successful response exceeds exact-preservation safety limit")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            setattr(
                error, "p4_api_error",
                response_parse_telemetry(
                    error, raw, status, reason, response_url, response_headers
                ),
            )
            raise
        if not isinstance(payload, dict) or payload.get("model") != args.model_id:
            raise IdentityFailure(
                f"completion response model identity differs from {args.model_id!r}"
            )
        return payload

    try:
        return send(), None
    except urllib.error.HTTPError as error:
        telemetry = http_error_telemetry(error)
        setattr(error, "p4_api_error", telemetry)
        if use_kwargs and args.reasoning_effort and error.code in (400, 422, 500):
            try:
                response, _ = request_completion(args, prompt, sampler_seed, use_kwargs=False)
            except Exception as retry_error:
                setattr(retry_error, "p4_retry_error", telemetry)
                raise
            return response, telemetry
        raise


def run_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = item["question"] + "\n\nEnd your response with: Answer: <number>"
    match = re.fullmatch(r"gsm8k_(\d+)", item["id"])
    if match is None:
        raise IdentityFailure(f"unexpected validation item ID: {item['id']!r}")
    sampler_seed = args.sampler_seed_base + int(match.group(1))
    args.http_wall_s = 0.0
    args.guard_wall_s = 0.0
    expected_answer = float(item["answer_number"])
    response: dict[str, Any] | None = None
    retry_error: dict[str, Any] | None = None
    row: dict[str, Any] = {"id": item["id"], "label": args.label}
    try:
        response, retry_error = request_completion(args, prompt, sampler_seed)
        choice = response["choices"][0]
        message = choice.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("completion choice message is not an object")
        content = visible_content(message)
        usage = response.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("completion usage is not an object")
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None and (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            raise ValueError("completion token count is not a nonnegative integer")
        row.update(
            {
                "correct": score_gsm8k(content, expected_answer),
                "wall_s": round(args.http_wall_s, 3),
                "identity_guard_wall_s": round(args.guard_wall_s, 3),
                "completion_tokens": completion_tokens,
                "thinking_tokens_estimate": thinking_tokens(message),
                "truncated": choice.get("finish_reason") == "length",
                "raw_response": response,
                "scoring_content": content,
                "parsed_answer_number": parse_gsm8k_answer(content),
                "expected_answer_number": expected_answer,
                "api_error": None,
                "chat_template_retry_error": retry_error,
                "chat_template_kwargs_dropped": retry_error is not None,
                "model_id": args.model_id,
                "model_manifest_sha256": args.model_manifest_sha256,
                "startup_identity_sha256": args.startup_identity_sha256,
                "sampler_seed": sampler_seed,
                "dataset_sha256": args.dataset_sha256,
                "harness_sha256": args.harness_sha256,
                "server_guard_sha256": args.server_guard_sha256,
                "sampler_contract_sha256": args.sampler_contract_sha256,
            }
        )
    except IdentityFailure:
        raise
    except Exception as error:  # preserve model/API failures only while identity stays proven
        prove_identity(args)
        retry_error = retry_error or getattr(error, "p4_retry_error", None)
        row.update(
            {
                "correct": False,
                "wall_s": round(args.http_wall_s, 3),
                "identity_guard_wall_s": round(args.guard_wall_s, 3),
                "completion_tokens": None,
                "thinking_tokens_estimate": 0,
                "truncated": False,
                "raw_response": response,
                "scoring_content": "",
                "parsed_answer_number": None,
                "expected_answer_number": expected_answer,
                "api_error": exception_telemetry(error),
                "chat_template_retry_error": retry_error,
                "chat_template_kwargs_dropped": retry_error is not None,
                "model_id": args.model_id,
                "model_manifest_sha256": args.model_manifest_sha256,
                "startup_identity_sha256": args.startup_identity_sha256,
                "sampler_seed": sampler_seed,
                "dataset_sha256": args.dataset_sha256,
                "harness_sha256": args.harness_sha256,
                "server_guard_sha256": args.server_guard_sha256,
                "sampler_contract_sha256": args.sampler_contract_sha256,
            }
        )
    return row


def selftest() -> int:
    cases = [
        ("Answer: $1,234", 1234.0, True),
        ("work\n1234", 1234.0, True),
        ("Answer: 99", 1234.0, False),
        ("no numeric answer", 1234.0, False),
    ]
    for content, expected, verdict in cases:
        if score_gsm8k(content, expected) != verdict:
            raise AssertionError((content, expected, verdict))
    message = {"content": "<think>one two three four</think>\nAnswer: 5"}
    if visible_content(message) != "Answer: 5" or thinking_tokens(message) != 5:
        raise AssertionError("thinking-content handling changed")
    defaults = SimpleNamespace(temperature=1.0, top_p=0.95, top_k=20, max_tokens=2048)
    if (defaults.temperature, defaults.top_p, defaults.top_k, defaults.max_tokens) != (1.0, 0.95, 20, 2048):
        raise AssertionError("protocol defaults changed")
    if SAMPLER_SEED_BASE + 13 != 4_200_013:
        raise AssertionError("per-item sampler-seed function changed")
    print("GSM8K HARNESS SELFTEST PASSED")
    return 0


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--model-id", default="local")
    parser.add_argument("--server-guard", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--server-start-time")
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--startup-identity", type=Path)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sampler-seed-base", type=int, default=SAMPLER_SEED_BASE)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.dataset or not args.label or not args.output:
        parser.error("--dataset, --label, and --output are required")
    if "/" in args.label:
        parser.error("--label must not contain '/'")
    identity_args = (
        args.server_guard, args.server_pid, args.server_start_time, args.launcher,
        args.model_path, args.startup_identity,
    )
    if any(value is None for value in identity_args):
        parser.error("server guard, PID/start-time, launcher, model path, and startup identity are required")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}; choose a new path")
    if args.dataset.is_symlink() or not args.dataset.is_file() or args.dataset.stat().st_mode & 0o077:
        parser.error("dataset must be a private regular file (0600 or stricter)")
    partial = args.output.with_name(args.output.name + ".partial")
    quarantine = args.output.with_name(args.output.name + ".quarantine")
    if partial.exists() or quarantine.exists():
        parser.error(f"refusing existing partial/quarantine output beside {args.output}")
    startup_raw = args.startup_identity.read_bytes()
    startup = json.loads(startup_raw)
    if (
        startup.get("schema") != "paper4-validation-server-identity-v1"
        or startup.get("phase") != "startup"
        or startup.get("alias") != args.model_id
        or startup.get("pid") != args.server_pid
        or startup.get("process_start_time_ticks") != args.server_start_time
    ):
        parser.error("startup identity receipt does not match this evaluation process")
    args.startup_identity_sha256 = hashlib.sha256(startup_raw).hexdigest()
    args.model_manifest_sha256 = startup.get("model_manifest_sha256")
    if not isinstance(args.model_manifest_sha256, str):
        parser.error("startup identity receipt has no model-manifest SHA-256")
    request_contract = startup.get("request")
    if not isinstance(request_contract, dict) or request_contract.get("context_tokens") != 4096:
        parser.error("P4R1 requires verified 4096-token context for every cell")
    args.context_tokens = request_contract["context_tokens"]
    args.ngl = request_contract.get("gpu_layers")
    if not isinstance(args.ngl, str):
        parser.error("startup identity receipt has no GPU-layer request")
    if (
        args.sampler_seed_base != SAMPLER_SEED_BASE
        or args.temperature != 1.0
        or args.top_p != 0.95
        or args.top_k != 20
        or args.max_tokens != 2048
        or args.reasoning_effort != "medium"
    ):
        parser.error("sampler arguments differ from the frozen P4R1 contract")

    dataset_raw = args.dataset.read_bytes()
    args.dataset_sha256 = hashlib.sha256(dataset_raw).hexdigest()
    if args.dataset_sha256 != DATASET_SHA256:
        parser.error(
            f"dataset SHA-256 differs from the pinned P4R1 subset: {args.dataset_sha256}"
        )
    items = [json.loads(line) for line in dataset_raw.decode("utf-8").splitlines() if line.strip()]
    if len(items) != 100:
        parser.error(f"expected 100 dataset rows, found {len(items)}")
    if any(not isinstance(item, dict) for item in items):
        parser.error("dataset contains a non-object row")
    ids = [item.get("id") for item in items]
    if ids != list(EXPECTED_IDS) or len(set(ids)) != 100:
        parser.error("dataset IDs/order differ from the exact seed-42 P4R1 subset")
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "question", "answer_number"}
            or not isinstance(item["question"], str)
            or not item["question"]
            or isinstance(item["answer_number"], bool)
            or not isinstance(item["answer_number"], (int, float))
            or not math.isfinite(item["answer_number"])
        ):
            parser.error("dataset contains a row outside the exact P4R1 schema")
    args.harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.server_guard_sha256 = hashlib.sha256(args.server_guard.read_bytes()).hexdigest()
    sampler_contract = {
        "schema": "paper4-p4r1-sampler-v1",
        "seed_base": args.sampler_seed_base,
        "seed_derivation": "seed_base + integer suffix of gsm8k_<suffix>",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "context_tokens": request_contract["context_tokens"],
        "concurrency": 1,
    }
    args.sampler_contract_sha256 = hashlib.sha256(
        json.dumps(sampler_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.output.parent.stat().st_mode & 0o077:
        parser.error(f"output directory must be owner-only (0700): {args.output.parent}")
    correct = 0
    try:
        with partial.open("x", encoding="utf-8") as handle:
            for index, item in enumerate(items, 1):
                row = run_item(item, args)
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
                handle.flush()
                correct += int(row["correct"])
                suffix = " truncated" if row["truncated"] else ""
                suffix += " error" if row["api_error"] is not None else ""
                print(f"[{index:3d}/100] {item['id']}: {'correct' if row['correct'] else 'incorrect'}{suffix}", flush=True)
    except IdentityFailure as error:
        if partial.exists():
            os.replace(partial, quarantine)
        print(f"CELL QUARANTINED: {error}; partial rows={quarantine}", file=sys.stderr)
        return 3
    os.replace(partial, args.output)
    print(f"{args.label}: {correct}/100; output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
