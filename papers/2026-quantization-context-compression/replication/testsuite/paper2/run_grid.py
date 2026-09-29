#!/usr/bin/env python3
"""Run one provenance-verified P2R1 deployment across four prompt conditions.

Raw request rows are append-only, private, and strictly write-once per
question/condition key. A crash resumes only missing conditions; rows may cite
different verified sessions, but no completed key is ever rerun or superseded.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from identity_guard import (  # noqa: E402
    IdentityError,
    receipt_sha256,
    sha256_file,
    utc_now,
    validate_receipt,
)
from r1_safety import (  # noqa: E402
    EvidenceLossError,
    RunSafetyError,
    attempt_ambiguity_error,
    append_checked_jsonl,
    append_session,
    atomic_summary,
    checked_jsonl_head_path,
    ensure_run_contract,
    file_identity,
    invalidate_attempt,
    read_checked_jsonl,
    require_attempt_eligible,
    require_attempt_id,
    require_r1_label,
)
import gen_questions as genq  # noqa: E402
from protocol_bindings import source_hashes  # noqa: E402


RESULTS = HERE.parent / "evals" / "results"
PROMPT = """Conversation excerpt:

---
{window}
---

Question: {q}

Give your final answer on the last line in exactly this format:
ANSWER: <short exact phrase from the excerpt>"""
MAX_RAW_API_RESPONSE_BYTES = 8 * 1024 * 1024


class RequestEvidenceError(RuntimeError):
    def __init__(self, error_type: str, message: str, evidence: dict):
        super().__init__(message)
        self.error_type = error_type
        self.error_message = message
        self.evidence = evidence


def transport_evidence(
    raw_body: bytes,
    *,
    status: int | None,
    reason: str | None,
    headers: list[list[str]],
) -> dict:
    if len(raw_body) > MAX_RAW_API_RESPONSE_BYTES:
        raise EvidenceLossError(
            "evaluation API response exceeds 8 MiB evidence cap"
        )
    return {
        "http_status": status,
        "http_reason": reason,
        "http_headers": headers,
        "raw_api_response_base64": base64.b64encode(raw_body).decode("ascii"),
        "raw_api_response_bytes": len(raw_body),
        "raw_api_response_sha256": hashlib.sha256(raw_body).hexdigest(),
    }


def visible_projection(raw_content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw_content, flags=re.S).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--identity-receipt", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--question-manifest", required=True, type=Path)
    parser.add_argument("--question-contract", required=True, type=Path)
    parser.add_argument("--question-journal", required=True, type=Path)
    parser.add_argument("--question-attempts", required=True, type=Path)
    parser.add_argument("--question-sessions", required=True, type=Path)
    parser.add_argument("--conditions", default="O,A,L2,E")
    parser.add_argument("--max-items", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip().strip('"\'`.,:;')


def extract_answer(response: str) -> str | None:
    matches = list(re.finditer(r"ANSWER\s*:\s*(.+)", response, flags=re.I))
    return matches[-1].group(1).splitlines()[0].strip() if matches else None


def chat(
    *,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout: int,
) -> dict:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "seed": seed,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "p2r1-grid/1"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            raw_body = exc.read(MAX_RAW_API_RESPONSE_BYTES + 1)
        except Exception as read_exc:
            raise EvidenceLossError(
                "failed while retaining HTTP error response bytes"
            ) from read_exc
        evidence = transport_evidence(
            raw_body,
            status=int(exc.code),
            reason=str(exc.reason) if exc.reason is not None else None,
            headers=[[str(k), str(v)] for k, v in exc.headers.items()],
        )
        raise RequestEvidenceError("HTTPError", str(exc), evidence) from exc
    except Exception as exc:
        evidence = transport_evidence(
            b"", status=None, reason=None, headers=[]
        )
        raise RequestEvidenceError(type(exc).__name__, str(exc), evidence) from exc
    try:
        with response:
            status = int(response.status)
            reason = str(response.reason) if response.reason is not None else None
            headers = [[str(k), str(v)] for k, v in response.headers.items()]
            raw_body = response.read(MAX_RAW_API_RESPONSE_BYTES + 1)
    except Exception as exc:
        raise EvidenceLossError(
            "failed while retaining evaluation API response bytes"
        ) from exc
    evidence = transport_evidence(
        raw_body, status=status, reason=reason, headers=headers
    )
    if not raw_body:
        raise RequestEvidenceError("EmptyResponse", "evaluation API response is empty", evidence)
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RequestEvidenceError(type(exc).__name__, str(exc), evidence) from exc
    if not isinstance(payload, dict):
        raise RequestEvidenceError("ResponseTypeError", "API payload is not an object", evidence)
    response_model = payload.get("model")
    try:
        raw_content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RequestEvidenceError(type(exc).__name__, str(exc), evidence) from exc
    if not isinstance(raw_content, str):
        raise RequestEvidenceError(
            "ResponseTypeError", "API message content is not text", evidence
        )
    content = visible_projection(raw_content)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        **evidence,
        "response_model": response_model,
        "usage": usage,
        "raw_message_content": raw_content,
        "raw_message_content_sha256": hashlib.sha256(
            raw_content.encode("utf-8")
        ).hexdigest(),
        "visible_response": content,
        "visible_response_sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
    }


def validate_api_evidence(value: object, *, success: bool) -> tuple[str, dict, object]:
    if not isinstance(value, dict):
        raise RunSafetyError("response lacks exact API transport evidence")
    raw_b64 = value.get("raw_api_response_base64")
    if not isinstance(raw_b64, str):
        raise RunSafetyError("API response body is not retained as base64")
    try:
        raw_body = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise RunSafetyError("API response body has invalid base64") from exc
    headers = value.get("http_headers")
    if (
        len(raw_body) > MAX_RAW_API_RESPONSE_BYTES
        or len(raw_body) != value.get("raw_api_response_bytes")
        or hashlib.sha256(raw_body).hexdigest()
        != value.get("raw_api_response_sha256")
        or not isinstance(headers, list)
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(part, str) for part in pair)
            for pair in headers
        )
        or (
            value.get("http_status") is not None
            and type(value.get("http_status")) is not int
        )
        or (
            value.get("http_reason") is not None
            and not isinstance(value.get("http_reason"), str)
        )
    ):
        raise RunSafetyError("API transport evidence hash/schema changed")
    if not success:
        return "", {}, None
    if value.get("http_status") != 200 or not raw_body:
        raise RunSafetyError("successful response lacks an HTTP 200 body")
    try:
        payload = json.loads(raw_body)
        raw_content = payload["choices"][0]["message"]["content"]
    except (
        json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError
    ) as exc:
        raise RunSafetyError("successful API body cannot be replayed") from exc
    if not isinstance(payload, dict) or not isinstance(raw_content, str):
        raise RunSafetyError("successful API response has invalid types")
    visible = visible_projection(raw_content)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    if (
        value.get("response_model") != payload.get("model")
        or value.get("usage") != usage
        or value.get("raw_message_content") != raw_content
        or value.get("raw_message_content_sha256")
        != hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
        or value.get("visible_response") != visible
        or value.get("visible_response_sha256")
        != hashlib.sha256(visible.encode("utf-8")).hexdigest()
    ):
        raise RunSafetyError("successful API parser projection changed")
    return visible, usage, payload.get("model")


def item_complete(rows: dict[tuple[str, str], dict], qid: str, conds: list[str]) -> bool:
    return all((qid, cond) in rows for cond in conds)


def score_response(response: str, gold: str) -> tuple[str | None, int, int]:
    answer = extract_answer(response)
    answer_norm, gold_norm = norm(answer or ""), norm(gold)
    exact = int(bool(answer_norm) and answer_norm == gold_norm)
    lenient = int(bool(
        exact
        or (answer_norm and gold_norm and gold_norm in answer_norm)
        or (answer_norm and len(answer_norm) >= 3 and answer_norm in gold_norm)
    ))
    return answer, exact, lenient


def validate_prior_rows(
    *,
    outdir: Path,
    output: Path,
    run_contract_sha256: str,
    deployment: dict,
    questions: dict[str, dict],
    conds: list[str],
    recover_torn_tail: bool = False,
) -> dict[tuple[str, str], dict]:
    sessions_path = outdir / "run_sessions.jsonl"
    sessions: dict[str, dict] = {}
    if sessions_path.exists():
        for session in read_checked_jsonl(
            sessions_path, recover_torn_tail=recover_torn_tail
        ):
            sid = session.get("session_id")
            if not sid or sid in sessions:
                raise RunSafetyError("missing or duplicate registered session ID")
            if session.get("schema") != "paper2-r1-session-v1":
                raise RunSafetyError(f"unrecognized session schema for {sid}")
            if session.get("run_contract_sha256") != run_contract_sha256:
                raise RunSafetyError(f"session {sid} belongs to another run contract")
            if not isinstance(session.get("identity_receipt"), str) or not session[
                "identity_receipt"
            ]:
                raise RunSafetyError(f"session {sid} lacks its identity-receipt path")
            for key, value in deployment.items():
                if session.get(key) != value:
                    raise RunSafetyError(f"session {sid} deployment mismatch: {key}")
            sessions[sid] = session

    latest: dict[tuple[str, str], dict] = {}
    if not output.exists():
        return latest
    for row in read_checked_jsonl(
        output, recover_torn_tail=recover_torn_tail
    ):
        if row.get("schema") != "paper2-r1-response-v1":
            raise RunSafetyError("prior output contains an unrecognized row schema")
        qid, cond = row.get("qid"), row.get("condition")
        if qid not in questions or cond not in conds:
            raise RunSafetyError(f"prior output contains an unregistered key: {qid}/{cond}")
        question = questions[qid]
        if row.get("window_id") != question["window_id"]:
            raise RunSafetyError(f"prior row has wrong window for {qid}")
        if row.get("source_cluster_id") != question["source_cluster_id"]:
            raise RunSafetyError(f"prior row has wrong source cluster for {qid}")
        if row.get("question_generator_identity_sha256") != question[
            "generator_identity_sha256"
        ]:
            raise RunSafetyError(f"prior row has wrong question provenance for {qid}")
        for key, expected in (
            ("stratum", question["stratum"]),
            ("has_code", question["has_code"]),
            ("gold", question["answer"]),
        ):
            if row.get(key) != expected:
                raise RunSafetyError(f"prior row has wrong {key}: {qid}/{cond}")
        response = row.get("response")
        if not isinstance(response, str):
            raise RunSafetyError(f"prior row lacks a full response: {qid}/{cond}")
        intent_id = row.get("request_intent_id")
        outcome = row.get("request_outcome")
        if not isinstance(intent_id, str) or not re.fullmatch(r"[0-9a-f]{32}", intent_id):
            raise RunSafetyError(f"prior row lacks a durable request intent: {qid}/{cond}")
        if outcome == "success":
            if row.get("request_error_type") is not None or row.get("request_error") is not None:
                raise RunSafetyError(f"successful row carries error telemetry: {qid}/{cond}")
            api_response, api_usage, api_model = validate_api_evidence(
                row.get("api_evidence"), success=True
            )
            if (
                response != api_response
                or row.get("usage") != api_usage
                or api_model != deployment["server_model_id"]
            ):
                raise RunSafetyError(f"successful API evidence changed: {qid}/{cond}")
        elif outcome == "request-error-identity-preserved":
            validate_api_evidence(row.get("api_evidence"), success=False)
            if (
                response != ""
                or row.get("usage") != {}
                or not isinstance(row.get("request_error_type"), str)
                or not isinstance(row.get("request_error"), str)
            ):
                raise RunSafetyError(f"request-error row is malformed: {qid}/{cond}")
        else:
            raise RunSafetyError(f"prior row has unknown request outcome: {qid}/{cond}")
        answer, exact, lenient = score_response(response, question["answer"])
        if row.get("answer_extracted") != answer:
            raise RunSafetyError(f"prior row parser result changed: {qid}/{cond}")
        for binary_key in ("has_marker", "exact", "lenient", "has_code"):
            if type(row.get(binary_key)) is not int or row[binary_key] not in (0, 1):
                raise RunSafetyError(
                    f"prior row has invalid binary {binary_key}: {qid}/{cond}"
                )
        if row.get("has_marker") != int(answer is not None):
            raise RunSafetyError(f"prior row marker result changed: {qid}/{cond}")
        if row.get("exact") != exact or row.get("lenient") != lenient:
            raise RunSafetyError(f"prior row score changed: {qid}/{cond}")
        if not isinstance(row.get("usage"), dict):
            raise RunSafetyError(f"prior row has malformed usage telemetry: {qid}/{cond}")
        for timing_key in (
            "wall_s", "identity_guard_before_s", "identity_guard_after_s"
        ):
            value = row.get(timing_key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise RunSafetyError(
                    f"prior row has invalid {timing_key}: {qid}/{cond}"
                )
        if row.get("run_contract_sha256") != run_contract_sha256:
            raise RunSafetyError(f"prior row belongs to another run contract: {qid}/{cond}")
        session = sessions.get(row.get("session_id"))
        if session is None:
            raise RunSafetyError(f"prior row cites an unregistered session: {qid}/{cond}")
        if row.get("identity_receipt") != session.get("identity_receipt"):
            raise RunSafetyError(f"prior row receipt path mismatch: {qid}/{cond}")
        if row.get("identity_receipt_sha256") != session["identity_receipt_sha256"]:
            raise RunSafetyError(f"prior row receipt mismatch: {qid}/{cond}")
        for key, value in deployment.items():
            row_key = "server_model_id" if key == "server_model_id" else key
            if row.get(row_key) != value:
                raise RunSafetyError(f"prior row deployment mismatch ({key}): {qid}/{cond}")
        for timestamp_key in ("request_started_at", "request_finished_at"):
            if not isinstance(row.get(timestamp_key), str) or not row[timestamp_key]:
                raise RunSafetyError(
                    f"prior row lacks {timestamp_key}: {qid}/{cond}"
                )
        key = (qid, cond)
        if key in latest:
            raise RunSafetyError(f"duplicate append-only response key: {qid}/{cond}")
        latest[key] = row
    return latest


def validate_request_ledger(
    path: Path,
    *,
    responses: dict[tuple[str, str], dict],
    questions: dict[str, dict],
    windows: dict[str, dict],
    conds: list[str],
    run_contract_sha256: str,
    deployment: dict,
    recover_torn_tail: bool = False,
) -> list[dict]:
    """Require one durable intent and one terminal for every evaluation POST."""
    events = read_checked_jsonl(path, recover_torn_tail=recover_torn_tail)
    intents: dict[str, dict] = {}
    intent_by_key: dict[tuple[str, str], str] = {}
    terminals: dict[str, dict] = {}
    for event in events:
        schema = event.get("schema")
        if schema == "paper2-r1-eval-intent-v1":
            intent_id = event.get("intent_id")
            qid, cond = event.get("qid"), event.get("condition")
            key = (qid, cond)
            if (
                not isinstance(intent_id, str)
                or not intent_id
                or intent_id in intents
                or key in intent_by_key
                or qid not in questions
                or cond not in conds
            ):
                raise RunSafetyError(f"duplicate or invalid evaluation intent: {key}")
            question = questions[qid]
            window = windows[question["window_id"]]
            text = window["text_O" if cond == "O" else f"text_{cond}"]
            expected_prompt = PROMPT.format(window=text[:14000], q=question["question"])
            if (
                event.get("run_contract_sha256") != run_contract_sha256
                or event.get("session_id") is None
                or event.get("server_model_id") != deployment["server_model_id"]
                or event.get("tier") != deployment["tier"]
                or event.get("prompt_sha256")
                != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
                or event.get("sampling")
                != {"temperature": 0.2, "top_p": 0.95, "seed": 42,
                    "max_tokens": 2048, "timeout_s": 600}
            ):
                raise RunSafetyError(f"evaluation intent provenance changed: {key}")
            intents[intent_id] = event
            intent_by_key[key] = intent_id
        elif schema == "paper2-r1-eval-terminal-v1":
            intent_id = event.get("intent_id")
            if intent_id not in intents or intent_id in terminals:
                raise RunSafetyError("orphan or duplicate evaluation terminal")
            intent = intents[intent_id]
            key = (intent["qid"], intent["condition"])
            status = event.get("status")
            if status == "identity-loss":
                raise RunSafetyError(
                    "endpoint identity was lost after a durable evaluation intent; "
                    "start a new named attempt"
                )
            if status not in {"success", "request-error-identity-preserved"}:
                raise RunSafetyError(f"unknown evaluation terminal status: {intent_id}")
            row = responses.get(key)
            if (
                row is None
                or row.get("request_intent_id") != intent_id
                or row.get("request_outcome") != status
                or row.get("session_id") != intent["session_id"]
                or row.get("identity_receipt_sha256")
                != intent.get("identity_receipt_sha256")
                or event.get("response_event_sha256") != row.get("_jsonl_sha256")
            ):
                raise RunSafetyError(f"evaluation terminal/response mismatch: {key}")
            terminals[intent_id] = event
        else:
            raise RunSafetyError("unrecognized evaluation request-ledger schema")
    if set(intents) != set(terminals):
        raise RunSafetyError(
            "evaluation request intent has no durable terminal; response completion "
            "is ambiguous, so this named attempt must be quarantined and restarted"
        )
    if set(intent_by_key) != set(responses):
        raise RunSafetyError("response/intent key sets differ in evaluation evidence")
    return events


def validate_session_usage(
    *,
    outdir: Path,
    ledger_events: list[dict],
    responses: dict[tuple[str, str], dict],
    attempt_root: Path,
    attempt_id: str,
    recover_torn_tail: bool = False,
) -> None:
    """Reject a launch session that never reached a durable request intent.

    A SIGKILL after session registration but before the intent append leaves no
    evidence that the session was ever safe to use.  The whole isolated attempt
    becomes forensic-only rather than silently registering a replacement
    session under the same result label.
    """
    sessions = read_checked_jsonl(
        outdir / "run_sessions.jsonl", recover_torn_tail=recover_torn_tail
    )
    registered = {row.get("session_id") for row in sessions}
    if None in registered or len(registered) != len(sessions):
        raise RunSafetyError("evaluation session registry is malformed")
    intent_sessions = {
        event.get("session_id")
        for event in ledger_events
        if event.get("schema") == "paper2-r1-eval-intent-v1"
    }
    response_sessions = {row.get("session_id") for row in responses.values()}
    unknown = sorted((intent_sessions | response_sessions) - registered)
    unused = sorted(registered - intent_sessions)
    if unknown or unused:
        marker = invalidate_attempt(
            attempt_root,
            attempt_id=attempt_id,
            reason="evaluation-session-without-complete-provenance",
            details={
                "result_label": outdir.name,
                "registered_but_without_intent": unused,
                "unregistered_session_references": unknown,
            },
        )
        raise RunSafetyError(
            "evaluation session registry/intent ledger mismatch; this attempt "
            f"is permanently ineligible ({marker})"
        )


def append_after_request(
    path: Path, value: dict, *, attempt_root: Path, attempt_id: str, reason: str
) -> dict:
    """Append post-POST evidence or permanently invalidate before propagating."""
    try:
        return append_checked_jsonl(path, value)
    except (RunSafetyError, OSError) as exc:
        raise attempt_ambiguity_error(
            attempt_root, attempt_id=attempt_id, reason=reason, error=exc,
        ) from exc


def main() -> int:
    args = parse_args()
    require_r1_label(args.label)
    require_attempt_id(args.attempt_id)
    attempt_question_root = (
        HERE / "r1" / "private" / "attempts" / args.attempt_id / "questions"
    ).resolve()
    for path in (
        args.questions, args.question_manifest, args.question_contract,
        args.question_journal, args.question_attempts, args.question_sessions,
    ):
        try:
            path.resolve().relative_to(attempt_question_root)
        except ValueError as exc:
            raise RunSafetyError(
                f"question evidence is outside attempt {args.attempt_id}: {path}"
            ) from exc
    if (
        args.temperature != 0.2
        or args.seed != 42
        or args.max_tokens != 2048
        or args.timeout != 600
    ):
        raise RunSafetyError(
            "P2R1 freezes temperature=0.2, seed=42, max_tokens=2048, timeout=600"
        )
    conds = args.conditions.split(",")
    if conds != ["O", "A", "L2", "E"]:
        raise RunSafetyError("P2R1 freezes condition set O,A,L2,E")

    receipt_path = args.identity_receipt.resolve()
    receipt = validate_receipt(receipt_path, full_artifact_hash=False)
    receipt_hash = receipt_sha256(receipt_path)
    base_url = receipt["base_url"]
    model_id = receipt["expected_model_id"]

    _, label_tier, label_corpus = args.label.split("-")
    artifact_prefix = receipt["artifact"]["sha256"][:12]
    official_targets = {"TSI": 117, "WC": 120}
    if args.max_items != official_targets[label_corpus]:
        raise RunSafetyError(
            f"{label_corpus} P2R1 target is {official_targets[label_corpus]}, "
            f"not {args.max_items}"
        )
    if label_tier in {"Q4KXL", "IQ2S"}:
        required_model_id = f"P2R1-{label_tier}-{artifact_prefix}"
        required_engine = "llama.cpp"
    else:
        required_model_id = "qwen38-exl3-2.0"
        required_engine = "ExLlamaV3/tabbyAPI"
    if model_id != required_model_id or receipt["engine"] != required_engine:
        raise IdentityError(
            f"label {args.label} requires {required_engine}/{required_model_id}; "
            f"receipt proves {receipt['engine']}/{model_id}"
        )

    input_manifest = json.loads(args.input_manifest.read_text())
    question_manifest = json.loads(args.question_manifest.read_text())
    expected_input_schema = (
        "paper2-r1-tsi-input-v1" if label_corpus == "TSI"
        else "paper2-r1-wildchat-input-v1"
    )
    if input_manifest.get("schema") != expected_input_schema:
        raise RunSafetyError(f"wrong input-manifest schema for {label_corpus}")
    if input_manifest.get("corpus") not in {None, label_corpus}:
        raise RunSafetyError("input-manifest corpus does not match result label")
    if input_manifest.get("output_sha256") != sha256_file(args.windows):
        raise RunSafetyError("window file does not match its frozen input manifest")
    if question_manifest.get("schema") != "paper2-r1-question-manifest-v1":
        raise RunSafetyError("question manifest schema is not P2R1")
    if question_manifest.get("attempt_id") != args.attempt_id:
        raise RunSafetyError("question manifest belongs to another P2R1 attempt")
    if question_manifest.get("corpus") != label_corpus:
        raise RunSafetyError("question manifest corpus does not match result label")
    if question_manifest.get("questions", {}).get("sha256") != sha256_file(args.questions):
        raise RunSafetyError("question file does not match its frozen manifest")

    if question_manifest.get("target_items") != official_targets[label_corpus]:
        raise RunSafetyError("question-manifest target does not match official plan")
    if question_manifest.get("written_items") != official_targets[label_corpus]:
        raise RunSafetyError("question-manifest written count is incomplete")
    expected_quotas = (
        {"S-nocode": 37, "S-code": 37, "L-nocode": 6, "L-code": 37}
        if label_corpus == "TSI"
        else {"S-nocode": 30, "S-code": 30, "L-nocode": 30, "L-code": 30}
    )
    if question_manifest.get("selection_quotas") != expected_quotas:
        raise RunSafetyError("question-manifest quotas do not match official plan")
    if question_manifest.get("windows", {}).get("sha256") != sha256_file(args.windows):
        raise RunSafetyError("question generation used a different window file")

    generation_contract = json.loads(args.question_contract.read_text())
    if generation_contract.get("schema") != "paper2-r1-question-generation-contract-v1":
        raise RunSafetyError("question-generation contract schema is not P2R1")
    if generation_contract.get("attempt_id") != args.attempt_id:
        raise RunSafetyError("question-generation contract belongs to another P2R1 attempt")
    if generation_contract.get("corpus") != label_corpus:
        raise RunSafetyError("question-generation contract corpus changed")
    if generation_contract.get("recovery_harness_sha256") != source_hashes(HERE.parents[1]):
        raise RunSafetyError("question-generation recovery harness changed")
    if question_manifest.get("generation_contract") != file_identity(args.question_contract):
        raise RunSafetyError(
            "question manifest does not bind the supplied generation contract"
        )
    if question_manifest.get("journal") != file_identity(args.question_journal):
        raise RunSafetyError("question journal does not match the frozen manifest")
    if question_manifest.get("journal_head") != file_identity(
        checked_jsonl_head_path(args.question_journal)
    ):
        raise RunSafetyError("question journal chain head changed")
    if question_manifest.get("attempt_log") != file_identity(args.question_attempts):
        raise RunSafetyError("question attempt ledger does not match the frozen manifest")
    if question_manifest.get("attempt_log_head") != file_identity(
        checked_jsonl_head_path(args.question_attempts)
    ):
        raise RunSafetyError("question attempt-ledger chain head changed")
    if question_manifest.get("sessions") != file_identity(args.question_sessions):
        raise RunSafetyError("question-generation sessions do not match the frozen manifest")
    if question_manifest.get("sessions_head") != file_identity(
        checked_jsonl_head_path(args.question_sessions)
    ):
        raise RunSafetyError("question session-registry chain head changed")

    window_rows = read_jsonl(args.windows)
    window_ids = [row["window_id"] for row in window_rows]
    if len(window_ids) != len(set(window_ids)):
        raise RunSafetyError("window IDs are not unique")
    expected_frames = (
        {"S-nocode": 40, "S-code": 40, "L-nocode": 6, "L-code": 40}
        if label_corpus == "TSI"
        else {"S-nocode": 40, "S-code": 40, "L-nocode": 40, "L-code": 40}
    )
    if input_manifest.get("windows") != len(window_rows):
        raise RunSafetyError("input-manifest window count does not match input")
    try:
        frame_counts = {
            key: sum(
                f"{row['stratum']}-{'code' if int(row['has_code']) else 'nocode'}" == key
                for row in window_rows
            )
            for key in expected_frames
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RunSafetyError("window stratum/code metadata is malformed") from exc
    if frame_counts != expected_frames or input_manifest.get("frame_counts") != expected_frames:
        raise RunSafetyError("input-manifest frame does not match the official corpus frame")
    ids_hash = hashlib.sha256("\n".join(window_ids).encode("utf-8")).hexdigest()
    if input_manifest.get("window_ids_sha256") != ids_hash:
        raise RunSafetyError("input-manifest window ID sequence changed")
    for condition in ("O", "A", "L2", "E"):
        digest = hashlib.sha256()
        for row in window_rows:
            identifier = row["window_id"].encode("utf-8")
            value = row[f"text_{condition}"].encode("utf-8")
            digest.update(len(identifier).to_bytes(8, "big"))
            digest.update(identifier)
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        if (
            input_manifest.get("condition_text_sha256", {}).get(condition)
            != digest.hexdigest()
        ):
            raise RunSafetyError(
                f"input-manifest {condition} condition hash changed"
            )
    q4_deployment = generation_contract.get("q4_deployment")
    if not isinstance(q4_deployment, dict):
        raise RunSafetyError("question-generation contract lacks Q4 deployment")
    question_sessions = genq.load_generation_sessions(
        args.question_sessions, sha256_file(args.question_contract), q4_deployment
    )
    if any(
        session.get("attempt_id") != args.attempt_id
        for session in question_sessions.values()
    ):
        raise RunSafetyError("question-generation session belongs to another attempt")
    ordered_generation_windows = genq.stratified_window_order(window_rows)
    question_journal_rows = read_checked_jsonl(args.question_journal)
    question_attempt_rows = read_checked_jsonl(args.question_attempts)
    if len(question_journal_rows) != sum(expected_frames.values()):
        raise RunSafetyError("question journal does not cover the exact full corpus frame")
    genq.validate_journal(
        question_journal_rows,
        ordered_generation_windows,
        corpus=label_corpus,
        sessions=question_sessions,
        base_seed=42,
    )
    genq.validate_generation_attempt_ledger(
        question_attempt_rows,
        ordered_generation_windows,
        corpus=label_corpus,
        sessions=question_sessions,
        base_seed=42,
        journal_rows=question_journal_rows,
    )
    windows = {row["window_id"]: row for row in window_rows}
    question_rows = read_jsonl(args.questions)
    derived_question_rows = genq.derive_question_rows(
        question_journal_rows, label_corpus, 42
    )
    if question_rows != derived_question_rows:
        raise RunSafetyError("frozen questions do not equal the journal-derived selection")
    canonical_question_bytes = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        for row in derived_question_rows
    )
    if args.questions.read_bytes() != canonical_question_bytes:
        raise RunSafetyError("question file is not the canonical journal projection")
    if len(question_rows) != args.max_items:
        raise RunSafetyError(
            f"question file must contain exactly {args.max_items} rows; "
            f"found {len(question_rows)}"
        )
    questions = sorted(question_rows, key=lambda row: row["qid"])
    if len(questions) != args.max_items:
        raise RunSafetyError(
            f"frozen plan requires {args.max_items} items; found {len(questions)}"
        )
    qids = [row["qid"] for row in questions]
    if len(qids) != len(set(qids)):
        raise RunSafetyError("question IDs are not unique")
    if any(row.get("corpus") != label_corpus for row in questions):
        raise RunSafetyError(
            f"label corpus {label_corpus} does not match question-set provenance"
        )
    for row in questions:
        if row.get("schema") != "paper2-r1-question-v1":
            raise RunSafetyError(f"invalid question schema: {row.get('qid')}")
        if row.get("generator_tier") != "Q4KXL":
            raise RunSafetyError(f"non-Q4 question generator: {row.get('qid')}")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("generator_identity_sha256", "")):
            raise RunSafetyError(f"invalid generator receipt hash: {row.get('qid')}")
        if not re.fullmatch(r"P2R1-Q4KXL-[0-9a-f]{12}", row.get("generator_model_id", "")):
            raise RunSafetyError(f"invalid Q4 generator model ID: {row.get('qid')}")
        if not row.get("question") or not row.get("answer") or not row.get("source_cluster_id"):
            raise RunSafetyError(f"incomplete question provenance: {row.get('qid')}")
    manifest_receipts = sorted(question_manifest.get("generator_receipt_sha256s", []))
    row_receipts = sorted({row["generator_identity_sha256"] for row in questions})
    if manifest_receipts != row_receipts:
        raise RunSafetyError("question generator receipt set does not match manifest")
    if question_manifest.get("selected_qids") != [row["qid"] for row in questions]:
        raise RunSafetyError("question IDs do not match frozen manifest selection")
    missing_windows = sorted({row["window_id"] for row in questions} - windows.keys())
    if missing_windows:
        raise RunSafetyError(f"questions reference missing windows: {missing_windows[:3]}")
    for item in questions:
        window = windows[item["window_id"]]
        for key in ("stratum", "has_code", "source_cluster_id"):
            if item.get(key) != window.get(key):
                raise RunSafetyError(
                    f"question/window {key} mismatch for {item['qid']}"
                )
        for cond in conds:
            key = "text_O" if cond == "O" else f"text_{cond}"
            if not window.get(key):
                raise RunSafetyError(f"missing {key} for {item['window_id']}")

    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    deployment = {
        "tier": label_tier,
        "server_model_id": model_id,
        "engine": receipt["engine"],
        "engine_revision": receipt["engine_revision"],
        "artifact_sha256": receipt["artifact"]["sha256"],
        "artifact_record_sha256": receipt["artifact_record_sha256"],
    }
    attempt_root = RESULTS / "paper2-r1" / args.attempt_id
    require_attempt_eligible(attempt_root)
    outdir = attempt_root / args.label
    request_ledger = outdir / "request_ledger.jsonl"
    contract = {
        "schema": "paper2-r1-run-contract-v1",
        "research_contract": "EMPIRICAL_RESEARCH_LOOP.md@1.1.0",
        "recovery_harness_sha256": source_hashes(HERE.parents[1]),
        "label": args.label,
        "attempt_id": args.attempt_id,
        "attempt_root": str(attempt_root.resolve()),
        "windows": file_identity(args.windows),
        "questions": file_identity(args.questions),
        "input_manifest": file_identity(args.input_manifest),
        "question_manifest": file_identity(args.question_manifest),
        "question_generation_contract": file_identity(args.question_contract),
        "question_generation_journal": file_identity(args.question_journal),
        "question_generation_attempts": file_identity(args.question_attempts),
        "question_generation_sessions": file_identity(args.question_sessions),
        "question_generator_receipts": sorted(
            {row["generator_identity_sha256"] for row in questions}
        ),
        "deployment": deployment,
        "conditions": conds,
        "n_items": args.max_items,
        "temperature": args.temperature,
        "top_p": 0.95,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "request_timeout_s": args.timeout,
        "context_character_cap": 14000,
        "prompt_sha256": prompt_sha,
        "scoring": "final ANSWER marker; exact primary; symmetric substring lenient sensitivity",
        "request_ledger": str(request_ledger.resolve()),
    }
    run_contract_path = ensure_run_contract(outdir, contract)
    run_contract_hash = sha256_file(run_contract_path)
    output = outdir / "p2grid.jsonl"
    failures = outdir / "failures.jsonl"
    # Any recovered tail in an official append-only stream makes the named
    # attempt scientifically ambiguous. Recovery is forensic only; startup
    # aborts instead of resampling a response whose append may have torn.
    questions_by_qid = {item["qid"]: item for item in questions}
    try:
        read_checked_jsonl(failures, recover_torn_tail=True)
        prior = validate_prior_rows(
            outdir=outdir,
            output=output,
            run_contract_sha256=run_contract_hash,
            deployment=deployment,
            questions=questions_by_qid,
            conds=conds,
            recover_torn_tail=True,
        )
        ledger_events = validate_request_ledger(
            request_ledger,
            responses=prior,
            questions=questions_by_qid,
            windows=windows,
            conds=conds,
            run_contract_sha256=run_contract_hash,
            deployment=deployment,
            recover_torn_tail=True,
        )
        validate_session_usage(
            outdir=outdir,
            ledger_events=ledger_events,
            responses=prior,
            attempt_root=attempt_root,
            attempt_id=args.attempt_id,
            recover_torn_tail=True,
        )
    except (RunSafetyError, OSError) as exc:
        raise attempt_ambiguity_error(
            attempt_root, attempt_id=args.attempt_id,
            reason="evaluation-stored-evidence-ambiguity", error=exc,
        ) from exc
    needs_work = any(
        not item_complete(prior, item["qid"], conds) for item in questions
    )
    session_id = None
    if needs_work:
        session_id = f"{receipt_hash[:16]}-{uuid.uuid4().hex}"
        append_session(outdir, {
            "schema": "paper2-r1-session-v1",
            "session_id": session_id,
            "run_contract_sha256": run_contract_hash,
            "identity_receipt": str(receipt_path),
            "identity_receipt_sha256": receipt_hash,
            "engine": receipt["engine"],
            "engine_revision": receipt["engine_revision"],
            "artifact_sha256": receipt["artifact"]["sha256"],
            "server_model_id": model_id,
            "tier": label_tier,
            "artifact_record_sha256": receipt["artifact_record_sha256"],
        })
    start = time.monotonic()
    for index, item in enumerate(questions):
        if item_complete(prior, item["qid"], conds):
            continue
        window = windows[item["window_id"]]
        order = conds[:]
        random.Random(args.seed * 100000 + index).shuffle(order)
        for cond in order:
            if (item["qid"], cond) in prior:
                continue
            text = window["text_O" if cond == "O" else f"text_{cond}"]
            prompt = PROMPT.format(window=text[:14000], q=item["question"])
            request_started = utc_now()
            guard_t0 = time.monotonic()
            validate_receipt(receipt_path, full_artifact_hash=False)
            guard_before = time.monotonic() - guard_t0
            intent_id = uuid.uuid4().hex
            append_checked_jsonl(request_ledger, {
                "schema": "paper2-r1-eval-intent-v1",
                "intent_id": intent_id,
                "created_at": request_started,
                "run_contract_sha256": run_contract_hash,
                "session_id": session_id,
                "identity_receipt_sha256": receipt_hash,
                "tier": label_tier,
                "server_model_id": model_id,
                "qid": item["qid"],
                "condition": cond,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "sampling": {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "seed": 42,
                    "max_tokens": 2048,
                    "timeout_s": 600,
                },
            })
            t0 = time.monotonic()
            request_error = None
            response, usage, response_model = "", {}, model_id
            api_evidence = transport_evidence(
                b"", status=None, reason=None, headers=[]
            )
            try:
                api_evidence = chat(
                    base_url=base_url,
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    timeout=args.timeout,
                )
                response = api_evidence["visible_response"]
                usage = api_evidence["usage"]
                response_model = api_evidence["response_model"]
            except Exception as exc:
                request_error = exc
                if isinstance(exc, RequestEvidenceError):
                    api_evidence = exc.evidence
            wall = time.monotonic() - t0
            guard_t1 = time.monotonic()
            identity_error = None
            try:
                validate_receipt(receipt_path, full_artifact_hash=False)
            except Exception as exc:
                identity_error = exc
            if isinstance(request_error, IdentityError):
                # Never relabel a completion response-model mismatch as an
                # ordinary incorrect answer from the intended deployment.
                identity_error = request_error
            if isinstance(request_error, EvidenceLossError):
                identity_error = request_error
            if request_error is not None and not isinstance(
                request_error,
                (RequestEvidenceError, IdentityError, EvidenceLossError),
            ):
                identity_error = EvidenceLossError(
                    "unexpected evaluation exception prevented exact bounded "
                    "request-evidence classification"
                )
            if request_error is None and response_model != model_id:
                identity_error = IdentityError(
                    f"completion identity mismatch: expected {model_id!r}, "
                    f"got {response_model!r}"
                )
            guard_after = time.monotonic() - guard_t1
            if identity_error is not None:
                append_after_request(request_ledger, {
                    "schema": "paper2-r1-eval-terminal-v1",
                    "intent_id": intent_id,
                    "finished_at": utc_now(),
                    "status": "identity-loss",
                    "error_type": type(identity_error).__name__,
                    "error": str(identity_error),
                    "observed_response_model": response_model,
                    "response": response,
                    "usage": usage,
                    "api_evidence": api_evidence,
                }, attempt_root=attempt_root, attempt_id=args.attempt_id,
                    reason="evaluation-identity-terminal-append-ambiguity")
                append_after_request(failures, {
                    "schema": "paper2-r1-request-failure-v1",
                    "ts": utc_now(),
                    "session_id": session_id,
                    "identity_receipt_sha256": receipt_hash,
                    "qid": item["qid"],
                    "condition": cond,
                    "error_type": type(identity_error).__name__,
                    "error": str(identity_error),
                }, attempt_root=attempt_root, attempt_id=args.attempt_id,
                    reason="evaluation-identity-failure-log-append-ambiguity")
                raise attempt_ambiguity_error(
                    attempt_root, attempt_id=args.attempt_id,
                    reason="evaluation-endpoint-identity-loss",
                    error=identity_error,
                ) from identity_error
            outcome = (
                "request-error-identity-preserved" if request_error is not None
                else "success"
            )
            if request_error is not None:
                response, usage, response_model = "", {}, model_id
            answer, exact, lenient = score_response(response, item["answer"])
            row = {
                "schema": "paper2-r1-response-v1",
                "request_intent_id": intent_id,
                "request_outcome": outcome,
                "request_error_type": (
                    (
                        request_error.error_type
                        if isinstance(request_error, RequestEvidenceError)
                        else type(request_error).__name__
                    ) if request_error is not None else None
                ),
                "request_error": (
                    request_error.error_message
                    if isinstance(request_error, RequestEvidenceError)
                    else str(request_error)
                ) if request_error is not None else None,
                "api_evidence": api_evidence,
                "session_id": session_id,
                "identity_receipt": str(receipt_path),
                "identity_receipt_sha256": receipt_hash,
                "run_contract_sha256": run_contract_hash,
                "engine": receipt["engine"],
                "engine_revision": receipt["engine_revision"],
                "artifact_sha256": receipt["artifact"]["sha256"],
                "artifact_record_sha256": receipt["artifact_record_sha256"],
                "tier": label_tier,
                "server_model_id": response_model,
                "qid": item["qid"],
                "window_id": item["window_id"],
                "source_cluster_id": item["source_cluster_id"],
                "question_generator_identity_sha256": item[
                    "generator_identity_sha256"
                ],
                "condition": cond,
                "stratum": window["stratum"],
                "has_code": window["has_code"],
                "request_started_at": request_started,
                "request_finished_at": utc_now(),
                "response": response,
                "answer_extracted": answer,
                "has_marker": int(answer is not None),
                "gold": item["answer"],
                "exact": int(exact),
                "lenient": int(lenient),
                "wall_s": wall,
                "identity_guard_before_s": guard_before,
                "identity_guard_after_s": guard_after,
                "usage": usage,
            }
            stored = append_after_request(
                output, row, attempt_root=attempt_root,
                attempt_id=args.attempt_id,
                reason="evaluation-response-append-ambiguity",
            )
            append_after_request(request_ledger, {
                "schema": "paper2-r1-eval-terminal-v1",
                "intent_id": intent_id,
                "finished_at": row["request_finished_at"],
                "status": outcome,
                "response_event_sha256": stored["_jsonl_sha256"],
            }, attempt_root=attempt_root, attempt_id=args.attempt_id,
                reason="evaluation-terminal-append-ambiguity")
            if request_error is not None:
                append_after_request(failures, {
                    "schema": "paper2-r1-request-failure-v1",
                    "ts": row["request_finished_at"],
                    "session_id": session_id,
                    "identity_receipt_sha256": receipt_hash,
                    "qid": item["qid"],
                    "condition": cond,
                    "error_type": (
                        request_error.error_type
                        if isinstance(request_error, RequestEvidenceError)
                        else type(request_error).__name__
                    ),
                    "error": (
                        request_error.error_message
                        if isinstance(request_error, RequestEvidenceError)
                        else str(request_error)
                    ),
                }, attempt_root=attempt_root, attempt_id=args.attempt_id,
                    reason="evaluation-failure-log-append-ambiguity")
            prior[(item["qid"], cond)] = stored
        if (index + 1) % 10 == 0:
            elapsed = (time.monotonic() - start) / 60
            print(
                f"[{index + 1}/{len(questions)}] items visited ({elapsed:.0f} min)",
                flush=True,
            )

    try:
        selected = validate_prior_rows(
            outdir=outdir,
            output=output,
            run_contract_sha256=run_contract_hash,
            deployment=deployment,
            questions=questions_by_qid,
            conds=conds,
        )
        ledger_events = validate_request_ledger(
            request_ledger,
            responses=selected,
            questions=questions_by_qid,
            windows=windows,
            conds=conds,
            run_contract_sha256=run_contract_hash,
            deployment=deployment,
        )
        validate_session_usage(
            outdir=outdir,
            ledger_events=ledger_events,
            responses=selected,
            attempt_root=attempt_root,
            attempt_id=args.attempt_id,
        )
        incomplete = [
            item["qid"] for item in questions
            if not item_complete(selected, item["qid"], conds)
        ]
        if incomplete:
            raise RunSafetyError(f"grid incomplete for {len(incomplete)} item(s)")
    except (RunSafetyError, OSError) as exc:
        raise attempt_ambiguity_error(
            attempt_root, attempt_id=args.attempt_id,
            reason="evaluation-final-evidence-ambiguity", error=exc,
        ) from exc
    summaries = []
    for cond in conds:
        rows = [selected[(item["qid"], cond)] for item in questions]
        n = len(rows)
        summaries.append({
            "condition": cond,
            "n": n,
            "accuracy": sum(row["exact"] for row in rows) / n,
            "accuracy_lenient": sum(row["lenient"] for row in rows) / n,
            "mean_wall_s": sum(row["wall_s"] for row in rows) / n,
            "complete": True,
        })
    summary = {
        "schema": "paper2-r1-summary-v1",
        "label": args.label,
        "updated_at": max(row["request_finished_at"] for row in selected.values()),
        "run_contract_sha256": hashlib.sha256(
            (outdir / "run_contract.json").read_bytes()
        ).hexdigest(),
        "n_items": len(questions),
        "append_only_heads": {
            "responses": file_identity(checked_jsonl_head_path(output)),
            "request_ledger": file_identity(checked_jsonl_head_path(request_ledger)),
            "sessions": file_identity(checked_jsonl_head_path(outdir / "run_sessions.jsonl")),
            "failures": (
                file_identity(checked_jsonl_head_path(failures))
                if checked_jsonl_head_path(failures).exists() else None
            ),
        },
        "conditions": summaries,
    }
    # The summary is a deterministic projection of durable response rows, not
    # primary evidence.  Replace it atomically so a power loss after the last
    # response cannot strand an otherwise complete run behind a torn JSON file.
    atomic_summary(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IdentityError, RunSafetyError) as exc:
        raise SystemExit(f"P2R1 ABORTED: {exc}") from exc
