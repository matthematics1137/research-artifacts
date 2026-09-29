#!/usr/bin/env python3
"""Generate fresh, Q4-verified P2R1 extractive-QA question sets.

Outputs and per-window journals are append-only and private. The generator
requires a live endpoint-identity receipt and refuses every legacy P2G question
file. No raw question or conversation text belongs in the public artifact.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
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
    checked_jsonl_head_path,
    ensure_immutable_json,
    file_identity,
    invalidate_attempt,
    read_checked_jsonl,
    require_attempt_eligible,
    require_attempt_id,
)
from protocol_bindings import source_hashes  # noqa: E402


PROMPT = """Read this conversation excerpt:

---
{window}
---

Write exactly 2 factual questions about specific details in the excerpt. Each answer must be a short string copied VERBATIM from the excerpt (a name, number, identifier, term, or short phrase; at most 8 words). Do not ask about anything not stated in the excerpt.

Use exactly this format:
Q1: <question>
A1: <verbatim answer>
Q2: <question>
A2: <verbatim answer>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=["TSI", "WC"])
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--attempt-log", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--sessions", required=True, type=Path)
    parser.add_argument("--identity-receipt", required=True, type=Path)
    parser.add_argument("--max-windows", type=int, required=True)
    parser.add_argument("--target-items", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


STRATA = ("S-nocode", "S-code", "L-nocode", "L-code")
QUOTAS = {
    "TSI": {"S-nocode": 37, "S-code": 37, "L-nocode": 6, "L-code": 37},
    "WC": {"S-nocode": 30, "S-code": 30, "L-nocode": 30, "L-code": 30},
}
FRAME_COUNTS = {
    "TSI": {"S-nocode": 40, "S-code": 40, "L-nocode": 6, "L-code": 40},
    "WC": {"S-nocode": 40, "S-code": 40, "L-nocode": 40, "L-code": 40},
}
RETRY_OFFSETS = (0, 1_000_000, 2_000_000)
MAX_RAW_API_RESPONSE_BYTES = 8 * 1024 * 1024
QUESTION_MIN_CHARS = 13
QUESTION_MAX_CHARS = 500
ANSWER_MIN_CHARS = 2
ANSWER_MAX_CHARS = 80
ANSWER_MAX_WORDS = 8


class GeneratorRequestError(RuntimeError):
    def __init__(self, error_type: str, message: str, evidence: dict):
        super().__init__(message)
        self.error_type = error_type
        self.error_message = message
        self.evidence = evidence


def generator_transport_evidence(
    raw_body: bytes,
    *,
    status: int | None,
    reason: str | None,
    headers: list[list[str]],
) -> dict:
    if len(raw_body) > MAX_RAW_API_RESPONSE_BYTES:
        raise EvidenceLossError(
            "question-generation API response exceeds 8 MiB evidence cap"
        )
    return {
        "http_status": status,
        "http_reason": reason,
        "http_headers": headers,
        "raw_api_response_base64": base64.b64encode(raw_body).decode("ascii"),
        "raw_api_response_bytes": len(raw_body),
        "raw_api_response_sha256": hashlib.sha256(raw_body).hexdigest(),
    }


def stratum_key(stratum: str, has_code: int) -> str:
    return f"{stratum}-{'code' if has_code else 'nocode'}"


def stratified_window_order(windows: list[dict]) -> list[dict]:
    buckets = {
        key: [
            row for row in windows
            if stratum_key(row["stratum"], int(row["has_code"])) == key
        ]
        for key in STRATA
    }
    ordered = []
    offset = 0
    while len(ordered) < len(windows):
        for key in STRATA:
            if offset < len(buckets[key]):
                ordered.append(buckets[key][offset])
        offset += 1
    return ordered


def derive_question_rows(
    journal_rows: list[dict], corpus: str, selection_seed: int
) -> list[dict]:
    """Derive the frozen set from fsynced per-window events.

    The journal—not questions.jsonl—is the source of truth. This makes both
    crash points safe: a crash before a journal append leaves nothing to
    recover, while a crash after it is repaired by deterministic derivation.
    """
    candidates: dict[str, list[dict]] = {key: [] for key in STRATA}
    seen_windows: set[str] = set()
    for event in journal_rows:
        window_id = event["window_id"]
        if window_id in seen_windows:
            raise RunSafetyError(f"duplicate question-generation journal event: {window_id}")
        seen_windows.add(window_id)
        if event.get("corpus") != corpus:
            raise RunSafetyError("mixed corpora in question-generation journal")
        accepted_rows = event.get("accepted_questions", [])
        if not accepted_rows:
            continue

        def rank(accepted: dict) -> str:
            material = (
                f"{selection_seed}|{corpus}|{window_id}|"
                f"{accepted['attempt_index']}|{accepted['pair_index']}"
            )
            return hashlib.sha256(material.encode()).hexdigest()

        accepted = min(accepted_rows, key=rank)
        row = {
            "schema": "paper2-r1-question-v1",
            "qid": f"{window_id}q0",
            "window_id": window_id,
            "corpus": event["corpus"],
            "question": accepted["question"],
            "answer": accepted["answer"],
            "generator_tier": "Q4KXL",
            "generator_model_id": event["generator_model_id"],
            "generator_identity_receipt": event["generator_identity_receipt"],
            "generator_identity_sha256": event["generator_identity_sha256"],
            "generator_seed": accepted["seed"],
            "generator_attempt_index": accepted["attempt_index"],
            "generator_pair_index": accepted["pair_index"],
            "selection_rank_sha256": rank(accepted),
            "stratum": event["stratum"],
            "has_code": event["has_code"],
            "source_cluster_id": event["source_cluster_id"],
        }
        candidates[stratum_key(event["stratum"], int(event["has_code"]))].append(row)

    quotas = QUOTAS[corpus]
    for key in STRATA:
        candidates[key].sort(key=lambda row: row["selection_rank_sha256"])
        if len(candidates[key]) < quotas[key]:
            raise RunSafetyError(
                f"{corpus}/{key} has {len(candidates[key])} eligible windows; "
                f"frozen quota is {quotas[key]} after {len(RETRY_OFFSETS)} attempts"
            )

    questions: list[dict] = []
    selected_windows: set[str] = set()
    selected_clusters: set[str] = set()
    per_stratum = {key: 0 for key in STRATA}
    # Prefer distinct source conversations in deterministic stratum round-robin.
    made_progress = True
    while made_progress:
        made_progress = False
        for key in STRATA:
            if per_stratum[key] >= quotas[key]:
                continue
            choice = next((
                row for row in candidates[key]
                if row["window_id"] not in selected_windows
                and row["source_cluster_id"] not in selected_clusters
            ), None)
            if choice is not None:
                questions.append(choice)
                selected_windows.add(choice["window_id"])
                selected_clusters.add(choice["source_cluster_id"])
                per_stratum[key] += 1
                made_progress = True
    # TSI has fewer source conversations than selected windows. Fill the
    # remaining predeclared quota without ever selecting two QA per window.
    for key in STRATA:
        for row in candidates[key]:
            if per_stratum[key] >= quotas[key]:
                break
            if row["window_id"] in selected_windows:
                continue
            questions.append(row)
            selected_windows.add(row["window_id"])
            per_stratum[key] += 1
    if per_stratum != quotas:
        raise RunSafetyError(f"selected strata {per_stratum} do not match quotas {quotas}")
    questions.sort(key=lambda row: row["qid"])
    qids = [row["qid"] for row in questions]
    if len(qids) != len(set(qids)):
        raise RunSafetyError("derived question IDs are not unique")
    if len({row["window_id"] for row in questions}) != len(questions):
        raise RunSafetyError("selection violated the one-question-per-window rule")
    return questions


def write_questions_atomic(path: Path, rows: list[dict]) -> None:
    """Install canonical questions once; later resumes only verify exact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise RunSafetyError(
                f"frozen question file differs from journal projection: {path}"
            )
        return
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RunSafetyError(
                    f"concurrent question freeze disagrees with journal: {path}"
                )
    finally:
        if temp.exists():
            temp.unlink()


def validate_journal(
    journal_rows: list[dict],
    windows: list[dict],
    *,
    corpus: str,
    sessions: dict[str, dict],
    base_seed: int,
) -> None:
    planned = {row["window_id"]: (index, row) for index, row in enumerate(windows)}
    seen: set[str] = set()
    for event in journal_rows:
        if event.get("schema") != "paper2-r1-question-window-v1":
            raise RunSafetyError("unrecognized question journal schema")
        window_id = event.get("window_id")
        if window_id not in planned or window_id in seen:
            raise RunSafetyError(f"substituted or duplicate journal window: {window_id}")
        seen.add(window_id)
        index, window = planned[window_id]
        expected = {
            "corpus": corpus,
            "window_index": index,
            "stratum": window["stratum"],
            "has_code": int(window["has_code"]),
            "source_cluster_id": window["source_cluster_id"],
        }
        for key, value in expected.items():
            if event.get(key) != value:
                raise RunSafetyError(f"journal provenance mismatch for {window_id}: {key}")
        session = sessions.get(event.get("generation_session_id"))
        if session is None:
            raise RunSafetyError(f"journal cites unregistered generation session: {window_id}")
        if event.get("generator_identity_sha256") != session["identity_receipt_sha256"]:
            raise RunSafetyError(f"journal receipt/session mismatch for {window_id}")
        if event.get("generator_identity_receipt") != session["identity_receipt"]:
            raise RunSafetyError(f"journal receipt path/session mismatch for {window_id}")
        if event.get("generator_model_id") != session["model_id"]:
            raise RunSafetyError(f"journal model/session mismatch for {window_id}")
        attempts = event.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise RunSafetyError(f"journal lacks attempt record for {window_id}")
        expected_seeds = [base_seed + index + offset for offset in RETRY_OFFSETS]
        actual_seeds = [attempt.get("seed") for attempt in attempts]
        if actual_seeds != expected_seeds[:len(actual_seeds)]:
            raise RunSafetyError(f"journal retry seeds changed for {window_id}")
        if len(actual_seeds) > len(RETRY_OFFSETS):
            raise RunSafetyError(f"too many generator attempts for {window_id}")
        recomputed_accepted = []
        for attempt_index, attempt in enumerate(attempts):
            if attempt.get("attempt_index") != attempt_index:
                raise RunSafetyError(f"non-sequential attempt index for {window_id}")
            if not isinstance(attempt.get("intent_id"), str) or not attempt["intent_id"]:
                raise RunSafetyError(f"attempt lacks durable request intent for {window_id}")
            outcome = attempt.get("outcome")
            if outcome == "request-error-identity-preserved":
                validate_generator_transport(attempt)
                if (
                    attempt.get("response_model") is not None
                    or not isinstance(attempt.get("error_type"), str)
                    or not isinstance(attempt.get("error"), str)
                    or attempt.get("pairs_parsed") != 0
                    or attempt.get("accepted") != 0
                    or attempt.get("rejected_or_unparsed") != 2
                    or attempt.get("usage") != {}
                ):
                    raise RunSafetyError(f"request-error evidence changed for {window_id}")
                continue
            if outcome != "success" or attempt.get("response_model") != session["model_id"]:
                raise RunSafetyError(f"attempt response model/outcome changed for {window_id}")
            raw_body = validate_generator_transport(attempt)
            if (
                not raw_body
                or len(raw_body) != attempt.get("raw_api_response_bytes")
                or hashlib.sha256(raw_body).hexdigest()
                != attempt.get("raw_api_response_sha256")
            ):
                raise RunSafetyError(f"raw API body hash/length mismatch for {window_id}")
            try:
                payload = json.loads(raw_body)
                raw_content = payload["choices"][0]["message"]["content"]
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                raise RunSafetyError(f"raw API body cannot be replayed for {window_id}") from exc
            if not isinstance(payload, dict) or not isinstance(raw_content, str):
                raise RunSafetyError(f"raw API response has invalid types for {window_id}")
            if payload.get("model") != attempt.get("response_model"):
                raise RunSafetyError(f"raw API model changed for {window_id}")
            if (
                attempt.get("http_status") != 200
            ):
                raise RunSafetyError(f"raw API transport metadata changed for {window_id}")
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            if attempt.get("usage") != usage:
                raise RunSafetyError(f"raw API usage changed for {window_id}")
            if (
                attempt.get("raw_message_content") != raw_content
                or hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
                != attempt.get("raw_message_content_sha256")
            ):
                raise RunSafetyError(f"raw message content changed for {window_id}")
            response = visible_projection(raw_content)
            if (
                attempt.get("visible_response") != response
                or hashlib.sha256(response.encode("utf-8")).hexdigest()
                != attempt.get("visible_response_sha256")
            ):
                raise RunSafetyError(f"visible parser projection changed for {window_id}")
            accepted, rejected, pairs_parsed = parse_and_verify(
                response, window["text_O"][:14000]
            )
            if attempt.get("accepted") != len(accepted):
                raise RunSafetyError(f"attempt accepted count changed for {window_id}")
            if attempt.get("rejected_or_unparsed") != rejected:
                raise RunSafetyError(f"attempt rejection count changed for {window_id}")
            if attempt.get("pairs_parsed") != pairs_parsed:
                raise RunSafetyError(f"attempt parser count changed for {window_id}")
            for item in accepted:
                recomputed_accepted.append({
                    "attempt_index": attempt_index,
                    "seed": actual_seeds[attempt_index],
                    **item,
                })
            if accepted and attempt_index != len(attempts) - 1:
                raise RunSafetyError(f"retry continued after success for {window_id}")
        if not recomputed_accepted and len(attempts) != len(RETRY_OFFSETS):
            raise RunSafetyError(f"retry schedule stopped before max attempts for {window_id}")
        if recomputed_accepted != event.get("accepted_questions"):
            raise RunSafetyError(f"stored accepted pairs changed for {window_id}")
        if event.get("accepted_by_verifier") != len(recomputed_accepted):
            raise RunSafetyError(f"event accepted total changed for {window_id}")
        if event.get("rejected_or_unparsed") != sum(
            attempt["rejected_or_unparsed"] for attempt in attempts
        ):
            raise RunSafetyError(f"event rejection total changed for {window_id}")


def load_generation_sessions(
    path: Path, contract_sha256: str, deployment: dict, *,
    recover_torn_tail: bool = False,
) -> dict[str, dict]:
    sessions: dict[str, dict] = {}
    for session in read_checked_jsonl(path, recover_torn_tail=recover_torn_tail):
        sid = session.get("session_id")
        if not sid or sid in sessions:
            raise RunSafetyError("missing or duplicate question-generation session ID")
        if session.get("schema") != "paper2-r1-question-session-v1":
            raise RunSafetyError(f"wrong generation-session schema: {sid}")
        if session.get("generation_contract_sha256") != contract_sha256:
            raise RunSafetyError(f"generation session belongs to another contract: {sid}")
        for key, value in deployment.items():
            if session.get(key) != value:
                raise RunSafetyError(f"generation session deployment mismatch ({key}): {sid}")
        sessions[sid] = session
    return sessions


def append_generation_session(path: Path, session: dict) -> None:
    append_checked_jsonl(path, session)


def validate_generation_attempt_ledger(
    rows: list[dict],
    windows: list[dict],
    *,
    corpus: str,
    sessions: dict[str, dict],
    base_seed: int,
    journal_rows: list[dict] | None = None,
) -> None:
    """Validate durable intent -> one terminal for every generator POST.

    An unmatched intent is deliberately unrecoverable in the same named
    attempt: a model response may have returned before the process died. A
    response is therefore never silently resampled.
    """
    planned = {row["window_id"]: (index, row) for index, row in enumerate(windows)}
    intents: dict[str, dict] = {}
    intent_by_key: dict[tuple[str, int], str] = {}
    terminals: dict[str, dict] = {}
    for event in rows:
        schema = event.get("schema")
        if schema == "paper2-r1-generation-intent-v1":
            intent_id = event.get("intent_id")
            window_id = event.get("window_id")
            attempt_index = event.get("attempt_index")
            if not isinstance(intent_id, str) or not intent_id or intent_id in intents:
                raise RunSafetyError("duplicate or missing generation intent ID")
            if window_id not in planned or type(attempt_index) is not int:
                raise RunSafetyError(f"generation intent has an unknown key: {window_id}")
            key = (window_id, attempt_index)
            if key in intent_by_key or attempt_index not in range(len(RETRY_OFFSETS)):
                raise RunSafetyError(f"duplicate or invalid generation intent key: {key}")
            index, window = planned[window_id]
            session = sessions.get(event.get("generation_session_id"))
            expected_seed = base_seed + index + RETRY_OFFSETS[attempt_index]
            expected_prompt = PROMPT.format(window=window["text_O"][:14000])
            if (
                session is None
                or event.get("corpus") != corpus
                or event.get("window_index") != index
                or event.get("seed") != expected_seed
                or event.get("prompt_sha256")
                != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
                or event.get("generator_identity_receipt") != session["identity_receipt"]
                or event.get("generator_identity_sha256")
                != session["identity_receipt_sha256"]
                or event.get("generator_model_id") != session["model_id"]
            ):
                raise RunSafetyError(f"generation intent provenance changed: {key}")
            intents[intent_id] = event
            intent_by_key[key] = intent_id
        elif schema == "paper2-r1-generation-terminal-v1":
            intent_id = event.get("intent_id")
            if intent_id not in intents or intent_id in terminals:
                raise RunSafetyError("orphan or duplicate generation terminal")
            attempt = event.get("attempt")
            intent = intents[intent_id]
            if not isinstance(attempt, dict):
                raise RunSafetyError(f"generation terminal lacks attempt evidence: {intent_id}")
            if (
                attempt.get("intent_id") != intent_id
                or attempt.get("attempt_index") != intent["attempt_index"]
                or attempt.get("seed") != intent["seed"]
                or event.get("status") != attempt.get("outcome")
            ):
                raise RunSafetyError(f"generation terminal/intent mismatch: {intent_id}")
            if event.get("status") == "identity-loss":
                raise RunSafetyError(
                    "generation endpoint identity was lost after durable intent; "
                    "start a new named attempt"
                )
            if event.get("status") not in {
                "success", "request-error-identity-preserved"
            }:
                raise RunSafetyError(f"unknown generation terminal status: {intent_id}")
            terminals[intent_id] = event
        else:
            raise RunSafetyError("unrecognized generation attempt-ledger schema")
    unmatched = sorted(set(intents) - set(terminals))
    if unmatched:
        raise RunSafetyError(
            "generation request intent has no durable terminal; response completion "
            "is ambiguous, so this named attempt must be quarantined and restarted"
        )

    if journal_rows is None:
        return
    journal_attempts: dict[tuple[str, int], dict] = {}
    for window_event in journal_rows:
        for attempt in window_event.get("attempts", []):
            key = (window_event.get("window_id"), attempt.get("attempt_index"))
            if key in journal_attempts:
                raise RunSafetyError(f"duplicate journal attempt key: {key}")
            journal_attempts[key] = attempt
    ledger_attempts = {
        key: terminals[intent_id]["attempt"]
        for key, intent_id in intent_by_key.items()
    }
    if journal_attempts != ledger_attempts:
        raise RunSafetyError(
            "generation ledger is not exactly represented by complete per-window "
            "journal events; do not resume this named attempt"
        )


def validate_generation_session_usage(
    *,
    sessions: dict[str, dict],
    attempt_rows: list[dict],
    journal_rows: list[dict],
    attempt_root: Path,
    attempt_id: str,
    corpus: str,
) -> None:
    """Make session-registration-before-intent crashes permanently visible."""
    registered = set(sessions)
    intent_sessions = {
        row.get("generation_session_id")
        for row in attempt_rows
        if row.get("schema") == "paper2-r1-generation-intent-v1"
    }
    journal_sessions = {row.get("generation_session_id") for row in journal_rows}
    unknown = sorted((intent_sessions | journal_sessions) - registered)
    unused = sorted(registered - intent_sessions)
    if unknown or unused:
        marker = invalidate_attempt(
            attempt_root,
            attempt_id=attempt_id,
            reason="question-generation-session-without-complete-provenance",
            details={
                "corpus": corpus,
                "registered_but_without_intent": unused,
                "unregistered_session_references": unknown,
            },
        )
        raise RunSafetyError(
            "question-generation session registry/intent ledger mismatch; this "
            f"attempt is permanently ineligible ({marker})"
        )


def append_generation_after_request(
    path: Path, value: dict, *, attempt_root: Path, attempt_id: str, reason: str
) -> dict:
    """Append post-POST generator evidence or mark the attempt before raising."""
    try:
        return append_checked_jsonl(path, value)
    except (RunSafetyError, OSError) as exc:
        raise attempt_ambiguity_error(
            attempt_root, attempt_id=attempt_id, reason=reason, error=exc,
        ) from exc


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip().strip('"\'`.,:;')


def parse_and_verify(response: str, original_text: str) -> tuple[list[dict], int, int]:
    pairs = re.findall(r"Q\d\s*:\s*(.+?)\s*\nA\d\s*:\s*(.+?)(?:\n|$)", response)
    original_norm = norm(original_text)
    accepted: list[dict] = []
    rejected = max(0, 2 - len(pairs[:2]))
    for pair_index, (question, answer) in enumerate(pairs[:2]):
        question, answer = question.strip(), answer.strip()
        if (
            ANSWER_MIN_CHARS <= len(answer) <= ANSWER_MAX_CHARS
            and len(answer.split()) <= ANSWER_MAX_WORDS
            and QUESTION_MIN_CHARS <= len(question) <= QUESTION_MAX_CHARS
            and norm(answer)
            and answer in original_text
            and norm(answer) in original_norm
        ):
            accepted.append({
                "pair_index": pair_index,
                "question": question,
                "answer": answer,
            })
        else:
            rejected += 1
    return accepted, rejected, min(2, len(pairs))


def visible_projection(raw_content: str) -> str:
    """Frozen parser-visible projection; exact API content remains private."""
    return re.sub(r"<think>.*?</think>", "", raw_content, flags=re.S).strip()


def validate_generator_transport(value: dict) -> bytes:
    raw_b64 = value.get("raw_api_response_base64")
    if not isinstance(raw_b64, str):
        raise RunSafetyError("generator transport evidence lacks base64 body")
    try:
        raw_body = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise RunSafetyError("generator transport body has invalid base64") from exc
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
        raise RunSafetyError("generator transport evidence hash/schema changed")
    return raw_body


def call_generator(
    *, base_url: str, model_id: str, prompt: str, seed: int, timeout: int
) -> dict:
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": seed,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "p2r1-genq/1"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            raw_body = exc.read(MAX_RAW_API_RESPONSE_BYTES + 1)
        except Exception as read_exc:
            raise EvidenceLossError(
                "failed while retaining generator HTTP error bytes"
            ) from read_exc
        evidence = generator_transport_evidence(
            raw_body,
            status=int(exc.code),
            reason=str(exc.reason) if exc.reason is not None else None,
            headers=[[str(k), str(v)] for k, v in exc.headers.items()],
        )
        raise GeneratorRequestError("HTTPError", str(exc), evidence) from exc
    except Exception as exc:
        evidence = generator_transport_evidence(
            b"", status=None, reason=None, headers=[]
        )
        raise GeneratorRequestError(type(exc).__name__, str(exc), evidence) from exc
    try:
        with response:
            status = int(response.status)
            reason = str(response.reason) if response.reason is not None else None
            headers = [[str(k), str(v)] for k, v in response.headers.items()]
            raw_body = response.read(MAX_RAW_API_RESPONSE_BYTES + 1)
    except Exception as exc:
        raise EvidenceLossError(
            "failed while retaining question-generation response bytes"
        ) from exc
    evidence = generator_transport_evidence(
        raw_body, status=status, reason=reason, headers=headers
    )
    if not raw_body:
        raise GeneratorRequestError(
            "EmptyResponse", "question-generation API response is empty", evidence
        )
    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GeneratorRequestError(type(exc).__name__, str(exc), evidence) from exc
    if not isinstance(payload, dict):
        raise GeneratorRequestError(
            "ResponseTypeError", "question-generation API payload is not an object",
            evidence,
        )
    try:
        raw_content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeneratorRequestError(type(exc).__name__, str(exc), evidence) from exc
    if not isinstance(raw_content, str):
        raise GeneratorRequestError(
            "ResponseTypeError", "question-generation message content is not text",
            evidence,
        )
    content = visible_projection(raw_content)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        **evidence,
        # Missing/nonmatching model identity is handled as identity loss by the
        # caller while retaining these exact response bytes.  Do not raise a
        # bare KeyError here and discard the transport evidence.
        "response_model": payload.get("model"),
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


def main() -> int:
    args = parse_args()
    require_attempt_id(args.attempt_id)
    if args.seed != 42 or args.timeout != 600:
        raise RunSafetyError("P2R1 question generation freezes seed=42 and timeout=600")
    r1_root = (HERE / "r1").resolve()
    attempt_private_root = (
        r1_root / "private" / "attempts" / args.attempt_id / "questions"
    ).resolve()
    attempt_root = (
        HERE.parent / "evals" / "results" / "paper2-r1" / args.attempt_id
    ).resolve()
    require_attempt_eligible(attempt_root)
    for path in (
        args.out, args.journal, args.attempt_log, args.manifest, args.contract,
        args.sessions,
    ):
        try:
            path.resolve().relative_to(attempt_private_root)
        except ValueError as exc:
            raise RunSafetyError(
                f"P2R1 generated evidence must stay under the isolated attempt "
                f"root {attempt_private_root}: {path}"
            ) from exc

    receipt_path = args.identity_receipt.resolve()
    receipt = validate_receipt(receipt_path, full_artifact_hash=False)
    required_q4_id = f"P2R1-Q4KXL-{receipt['artifact']['sha256'][:12]}"
    if receipt["engine"] != "llama.cpp" or receipt["expected_model_id"] != required_q4_id:
        raise IdentityError("fresh R1 questions must be generated by verified Q4_K_XL")
    receipt_hash = receipt_sha256(receipt_path)
    base_url = receipt["base_url"]
    model_id = receipt["expected_model_id"]

    windows = read_jsonl(args.windows)
    if len(windows) != args.max_windows:
        raise RunSafetyError(
            f"expected exactly {args.max_windows} frozen windows; found {len(windows)}"
        )
    for row in windows:
        try:
            has_code = row["has_code"]
            key = stratum_key(row["stratum"], int(has_code))
        except (KeyError, TypeError, ValueError) as exc:
            raise RunSafetyError(
                f"unknown window stratum/code value: {row.get('window_id')}"
            ) from exc
        if key not in STRATA or has_code not in {0, 1, False, True}:
            raise RunSafetyError(f"unknown window stratum/code value: {row.get('window_id')}")
    windows = stratified_window_order(windows)
    window_ids = [row["window_id"] for row in windows]
    if len(window_ids) != len(set(window_ids)):
        raise RunSafetyError("input window IDs are not unique")
    if any(not row.get("source_cluster_id") for row in windows):
        raise RunSafetyError("every window requires an opaque source_cluster_id")
    frame_counts = {
        key: sum(stratum_key(row["stratum"], int(row["has_code"])) == key for row in windows)
        for key in STRATA
    }
    if frame_counts != FRAME_COUNTS[args.corpus]:
        raise RunSafetyError(
            f"{args.corpus} frame changed: {frame_counts}; expected {FRAME_COUNTS[args.corpus]}"
        )
    if args.max_windows != sum(FRAME_COUNTS[args.corpus].values()):
        raise RunSafetyError("max-windows must cover the entire frozen corpus frame")
    if args.target_items != sum(QUOTAS[args.corpus].values()):
        raise RunSafetyError("target-items does not match the frozen stratum quotas")
    deployment = {
        "model_id": model_id,
        "engine": receipt["engine"],
        "engine_revision": receipt["engine_revision"],
        "artifact_sha256": receipt["artifact"]["sha256"],
        "artifact_record_sha256": receipt["artifact_record_sha256"],
    }
    generation_contract = {
        "schema": "paper2-r1-question-generation-contract-v1",
        "research_contract": "EMPIRICAL_RESEARCH_LOOP.md@1.1.0",
        "attempt_id": args.attempt_id,
        "recovery_harness_sha256": source_hashes(HERE.parents[1]),
        "corpus": args.corpus,
        "windows": file_identity(args.windows),
        "max_windows": args.max_windows,
        "target_items": args.target_items,
        "seed": args.seed,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 2048,
        "context_character_cap": 14000,
        "question_character_bounds": [QUESTION_MIN_CHARS, QUESTION_MAX_CHARS],
        "answer_character_bounds": [ANSWER_MIN_CHARS, ANSWER_MAX_CHARS],
        "answer_max_words": ANSWER_MAX_WORDS,
        "answer_verification": "exact and normalized substring of text_O[:14000]",
        "raw_response_retention": (
            "exact HTTP response body bytes (private base64+SHA256), exact message "
            "content, and separately hashed parser-visible projection"
        ),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "frame_counts": FRAME_COUNTS[args.corpus],
        "selection_quotas": QUOTAS[args.corpus],
        "window_order": "all windows, round-robin S-nocode,S-code,L-nocode,L-code",
        "selection": "one QA/window; SHA256(seed,corpus,window,attempt,pair) rank; prefer unique source clusters; no post-hoc redistribution",
        "retry_seed_offsets": list(RETRY_OFFSETS),
        "q4_deployment": deployment,
        "output": str(args.out.resolve()),
        "journal": str(args.journal.resolve()),
        "attempt_log": str(args.attempt_log.resolve()),
        "sessions": str(args.sessions.resolve()),
    }
    ensure_immutable_json(args.contract, generation_contract)
    contract_hash = sha256_file(args.contract)
    try:
        sessions = load_generation_sessions(
            args.sessions, contract_hash, deployment, recover_torn_tail=True
        )
        if any(
            session.get("attempt_id") != args.attempt_id
            for session in sessions.values()
        ):
            raise RunSafetyError("question-generation session belongs to another attempt")
        journal_rows = read_checked_jsonl(args.journal, recover_torn_tail=True)
        attempt_rows = read_checked_jsonl(args.attempt_log, recover_torn_tail=True)
        validate_journal(
            journal_rows, windows, corpus=args.corpus, sessions=sessions,
            base_seed=args.seed,
        )
        validate_generation_attempt_ledger(
            attempt_rows,
            windows,
            corpus=args.corpus,
            sessions=sessions,
            base_seed=args.seed,
            journal_rows=journal_rows,
        )
        validate_generation_session_usage(
            sessions=sessions,
            attempt_rows=attempt_rows,
            journal_rows=journal_rows,
            attempt_root=attempt_root,
            attempt_id=args.attempt_id,
            corpus=args.corpus,
        )
        if args.out.exists() and not journal_rows:
            raise RunSafetyError("question output exists without its append-only journal")
        output_rows = []
        if len({row["window_id"] for row in journal_rows}) == args.max_windows:
            output_rows = derive_question_rows(journal_rows, args.corpus, args.seed)
            write_questions_atomic(args.out, output_rows)
        if len(output_rows) > args.target_items:
            raise RunSafetyError("existing R1 question file exceeds frozen target")
        if any(row.get("generator_tier") != "Q4KXL" for row in output_rows):
            raise RunSafetyError("existing question set contains non-Q4 generator provenance")
    except (RunSafetyError, OSError) as exc:
        raise attempt_ambiguity_error(
            attempt_root, attempt_id=args.attempt_id,
            reason="question-generation-stored-evidence-ambiguity", error=exc,
        ) from exc
    done_windows = {row["window_id"] for row in journal_rows}

    generation_session_id = None
    if len(done_windows) < args.max_windows:
        generation_session_id = f"{receipt_hash[:16]}-{uuid.uuid4().hex}"
        generation_session = {
            "schema": "paper2-r1-question-session-v1",
            "session_id": generation_session_id,
            "attempt_id": args.attempt_id,
            "generation_contract_sha256": contract_hash,
            "identity_receipt": str(receipt_path),
            "identity_receipt_sha256": receipt_hash,
            **deployment,
        }
        append_generation_session(args.sessions, generation_session)
        sessions[generation_session_id] = generation_session

    args.journal.parent.mkdir(parents=True, exist_ok=True)
    args.journal.parent.chmod(0o700)
    accepted_total = sum(len(row.get("accepted_questions", [])) for row in journal_rows)
    for window_index, window in enumerate(windows):
        if window["window_id"] in done_windows:
            continue
        accepted_questions = []
        attempts = []
        for attempt_index, offset in enumerate(RETRY_OFFSETS):
            seed = args.seed + window_index + offset
            validate_receipt(receipt_path, full_artifact_hash=False)
            prompt = PROMPT.format(window=window["text_O"][:14000])
            intent_id = uuid.uuid4().hex
            intent = {
                "schema": "paper2-r1-generation-intent-v1",
                "intent_id": intent_id,
                "created_at": utc_now(),
                "corpus": args.corpus,
                "window_id": window["window_id"],
                "window_index": window_index,
                "attempt_index": attempt_index,
                "seed": seed,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "generation_session_id": generation_session_id,
                "generator_identity_receipt": str(receipt_path),
                "generator_identity_sha256": receipt_hash,
                "generator_model_id": model_id,
            }
            intent = append_checked_jsonl(args.attempt_log, intent)
            attempt_rows.append(intent)
            generator_response = None
            request_error = None
            try:
                generator_response = call_generator(
                    base_url=base_url,
                    model_id=model_id,
                    prompt=prompt,
                    seed=seed,
                    timeout=args.timeout,
                )
            except Exception as exc:  # exact error becomes a declared terminal attempt
                request_error = exc
            identity_error = None
            try:
                validate_receipt(receipt_path, full_artifact_hash=False)
            except Exception as exc:
                identity_error = exc
            if isinstance(request_error, IdentityError):
                # A completion that names another model is an identity failure,
                # even if the subsequent /v1/models probe still looks healthy.
                identity_error = request_error
            if isinstance(request_error, EvidenceLossError):
                identity_error = request_error
            if request_error is not None and not isinstance(
                request_error,
                (GeneratorRequestError, IdentityError, EvidenceLossError),
            ):
                identity_error = EvidenceLossError(
                    "unexpected generator exception prevented exact bounded "
                    "request-evidence classification"
                )
            if (
                generator_response is not None
                and generator_response.get("response_model") != model_id
            ):
                identity_error = IdentityError(
                    f"question response model mismatch: expected {model_id!r}, "
                    f"got {generator_response.get('response_model')!r}"
                )

            if identity_error is not None:
                attempt = {
                    "intent_id": intent_id,
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "outcome": "identity-loss",
                    **(
                        generator_response
                        or (
                            request_error.evidence
                            if isinstance(request_error, GeneratorRequestError)
                            else {"response_model": None}
                        )
                    ),
                    "error_type": type(identity_error).__name__,
                    "error": str(identity_error),
                }
                terminal = append_generation_after_request(
                    args.attempt_log, {
                        "schema": "paper2-r1-generation-terminal-v1",
                        "intent_id": intent_id,
                        "finished_at": utc_now(),
                        "status": "identity-loss",
                        "attempt": attempt,
                    }, attempt_root=attempt_root, attempt_id=args.attempt_id,
                    reason="question-generation-terminal-append-ambiguity",
                )
                attempt_rows.append(terminal)
                raise attempt_ambiguity_error(
                    attempt_root, attempt_id=args.attempt_id,
                    reason="question-generation-endpoint-identity-loss",
                    error=identity_error,
                ) from identity_error
            if request_error is not None:
                accepted, rejected, pairs_parsed = [], 2, 0
                attempt = {
                    "intent_id": intent_id,
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "outcome": "request-error-identity-preserved",
                    "response_model": None,
                    **(
                        request_error.evidence
                        if isinstance(request_error, GeneratorRequestError)
                        else generator_transport_evidence(
                            b"", status=None, reason=None, headers=[]
                        )
                    ),
                    "error_type": (
                        request_error.error_type
                        if isinstance(request_error, GeneratorRequestError)
                        else type(request_error).__name__
                    ),
                    "error": (
                        request_error.error_message
                        if isinstance(request_error, GeneratorRequestError)
                        else str(request_error)
                    ),
                    "pairs_parsed": 0,
                    "accepted": 0,
                    "rejected_or_unparsed": 2,
                    "usage": {},
                }
            else:
                accepted, rejected, pairs_parsed = parse_and_verify(
                    generator_response["visible_response"], window["text_O"][:14000]
                )
                attempt = {
                    "intent_id": intent_id,
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "outcome": "success",
                    **generator_response,
                    "pairs_parsed": pairs_parsed,
                    "accepted": len(accepted),
                    "rejected_or_unparsed": rejected,
                }
            terminal = append_generation_after_request(
                args.attempt_log, {
                    "schema": "paper2-r1-generation-terminal-v1",
                    "intent_id": intent_id,
                    "finished_at": utc_now(),
                    "status": attempt["outcome"],
                    "attempt": attempt,
                }, attempt_root=attempt_root, attempt_id=args.attempt_id,
                reason="question-generation-terminal-append-ambiguity",
            )
            attempt_rows.append(terminal)
            for item in accepted:
                accepted_questions.append({
                    "attempt_index": attempt_index,
                    "seed": seed,
                    **item,
                })
            attempts.append(attempt)
            if accepted_questions:
                break
        journal = {
            "schema": "paper2-r1-question-window-v1",
            "ts": utc_now(),
            "corpus": args.corpus,
            "window_id": window["window_id"],
            "window_index": window_index,
            "stratum": window["stratum"],
            "has_code": int(window["has_code"]),
            "source_cluster_id": window["source_cluster_id"],
            "generator_identity_receipt": str(receipt_path),
            "generator_identity_sha256": receipt_hash,
            "generator_model_id": model_id,
            "generation_session_id": generation_session_id,
            "attempts": attempts,
            "accepted_by_verifier": len(accepted_questions),
            "rejected_or_unparsed": sum(a["rejected_or_unparsed"] for a in attempts),
            "accepted_questions": accepted_questions,
        }
        journal = append_generation_after_request(
            args.journal, journal, attempt_root=attempt_root,
            attempt_id=args.attempt_id,
            reason="question-generation-window-journal-append-ambiguity",
        )
        journal_rows.append(journal)
        done_windows.add(window["window_id"])
        accepted_total += len(accepted_questions)
        print(
            f"{window['window_id']}: accepted {len(accepted_questions)}; total candidates {accepted_total}",
            flush=True,
        )

    try:
        journal_rows = read_checked_jsonl(args.journal)
        attempt_rows = read_checked_jsonl(args.attempt_log)
        completed_windows = {row["window_id"] for row in journal_rows}
        if len(completed_windows) != args.max_windows:
            raise RunSafetyError(
                f"question generation covered {len(completed_windows)}/{args.max_windows} windows"
            )
        validate_journal(
            journal_rows, windows, corpus=args.corpus, sessions=sessions,
            base_seed=args.seed,
        )
        validate_generation_attempt_ledger(
            attempt_rows,
            windows,
            corpus=args.corpus,
            sessions=sessions,
            base_seed=args.seed,
            journal_rows=journal_rows,
        )
        validate_generation_session_usage(
            sessions=sessions,
            attempt_rows=attempt_rows,
            journal_rows=journal_rows,
            attempt_root=attempt_root,
            attempt_id=args.attempt_id,
            corpus=args.corpus,
        )
        output_rows = derive_question_rows(journal_rows, args.corpus, args.seed)
        write_questions_atomic(args.out, output_rows)
        if len(output_rows) != args.target_items:
            raise RunSafetyError(
                f"question generation ended with {len(output_rows)}/{args.target_items} items"
            )
    except (RunSafetyError, OSError) as exc:
        raise attempt_ambiguity_error(
            attempt_root, attempt_id=args.attempt_id,
            reason="question-generation-final-evidence-ambiguity", error=exc,
        ) from exc
    manifest = {
        "schema": "paper2-r1-question-manifest-v1",
        "research_contract": "EMPIRICAL_RESEARCH_LOOP.md@1.1.0",
        "attempt_id": args.attempt_id,
        "completed_at": journal_rows[-1]["ts"],
        "corpus": args.corpus,
        "windows": file_identity(args.windows),
        "questions": file_identity(args.out),
        "journal": file_identity(args.journal),
        "journal_head": file_identity(checked_jsonl_head_path(args.journal)),
        "attempt_log": file_identity(args.attempt_log),
        "attempt_log_head": file_identity(checked_jsonl_head_path(args.attempt_log)),
        "sessions": file_identity(args.sessions),
        "sessions_head": file_identity(checked_jsonl_head_path(args.sessions)),
        "generation_contract": file_identity(args.contract),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "max_windows": args.max_windows,
        "target_items": args.target_items,
        "frame_counts": FRAME_COUNTS[args.corpus],
        "selection_quotas": QUOTAS[args.corpus],
        "selection_algorithm": "one QA/window; SHA256 ID-only rank; prefer distinct source clusters",
        "base_seed": args.seed,
        "generator_tier": "Q4KXL",
        "q4_deployment": deployment,
        "generator_receipt_sha256s": sorted(
            {row["generator_identity_sha256"] for row in output_rows}
        ),
        "source_windows_used": len({row["window_id"] for row in output_rows}),
        "selected_qids": [row["qid"] for row in output_rows],
        "selected_source_clusters": len({row["source_cluster_id"] for row in output_rows}),
        "item_strata_counts": {
            key: sum(
                stratum_key(row["stratum"], int(row["has_code"])) == key
                for row in output_rows
            )
            for key in STRATA
        },
        "accepted_by_verifier_before_target_truncation": sum(
            row["accepted_by_verifier"] for row in journal_rows
        ),
        "written_items": len(output_rows),
        "rejected_or_unparsed": sum(row["rejected_or_unparsed"] for row in journal_rows),
    }
    ensure_immutable_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IdentityError, RunSafetyError) as exc:
        raise SystemExit(f"P2R1 QUESTION GENERATION ABORTED: {exc}") from exc
