#!/usr/bin/env python3
"""Analysis-only self-tests for P2R1 identity and append-only safeguards."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
import socket
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import identity_guard as guard  # noqa: E402
import campaign_guard as campaign  # noqa: E402
import gen_questions as gen  # noqa: E402
import run_grid as grid  # noqa: E402
from r1_safety import (  # noqa: E402
    RunSafetyError,
    append_checked_jsonl,
    append_session,
    attempt_ambiguity_error,
    atomic_summary,
    checked_event,
    checked_jsonl_head_path,
    ensure_run_contract,
    read_checked_jsonl,
    require_attempt_eligible,
    torn_tail_recovery_receipts,
)


MODEL_ID = "P2R1-Q4KXL-selftest"
POST_MARKER: str | None = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = {"status": "ok"}
            status = 200
        elif self.path == "/v1/models":
            payload = {
                "object": "list",
                "data": [{"id": MODEL_ID, "owned_by": "llamacpp"}],
            }
            status = 200
        else:
            payload = {"error": "not found"}
            status = 404
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        if POST_MARKER:
            Path(POST_MARKER).write_text(self.path)
        self.send_response(500)
        self.end_headers()


def serve(port: int) -> None:
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def idle_process() -> None:
    time.sleep(30)


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def expect_failure(callable_obj, message: str) -> None:
    try:
        callable_obj()
    except (guard.IdentityError, RunSafetyError):
        return
    raise AssertionError(message)


def test_graphical_desktop_cool_gate() -> None:
    assert campaign.parse_compute_process_pids("") == []
    assert campaign.parse_compute_process_pids("123\n456\n") == [123, 456]
    expect_failure(
        lambda: campaign.parse_compute_process_pids("No running processes found"),
        "malformed compute-process output did not fail closed",
    )

    def state(
        *, utilization: float = 37.0, temperature: float = 50.0,
        memory_mib: float = 1024.0, load: float = 2.0,
        available_gib: float = 32.0, compute_pids: list[int] | None = None,
    ) -> dict:
        return {
            "gpus": [{
                "temperature_c": temperature,
                "utilization_percent": utilization,
                "memory_used_mib": memory_mib,
            }],
            "load_average": {"one_min": load},
            "memory": {"MemAvailable": available_gib * 1024**3},
            "compute_process_pids": [] if compute_pids is None else compute_pids,
        }

    def admitted(**changes: object) -> bool:
        return campaign.cool_sample_admissible(
            state(**changes), max_gpu_temp=50.0, max_load=2.0,
            min_available_gib=32.0, max_gpu_memory_mib=1024.0,
        )

    assert admitted(utilization=37.0)
    assert admitted(utilization=100.0)
    assert not admitted(utilization=-0.1)
    assert not admitted(utilization=100.1)
    assert not admitted(utilization=float("nan"))
    assert not admitted(temperature=50.01)
    assert not admitted(memory_mib=1024.01)
    assert not admitted(load=2.01)
    assert not admitted(available_gib=31.99)
    assert not admitted(compute_pids=[1234])
    empty_gpu = state()
    empty_gpu["gpus"] = []
    assert not campaign.cool_sample_admissible(
        empty_gpu, max_gpu_temp=50.0, max_load=2.0,
        min_available_gib=32.0, max_gpu_memory_mib=1024.0,
    )


def synthetic_question_event(
    *, corpus: str, window: dict, index: int, session_id: str, receipt_hash: str
) -> dict:
    seed = 42 + index
    response = "Q1: What exact token appears in this excerpt?\nA1: value\n"
    raw_payload = json.dumps({
        "model": MODEL_ID,
        "choices": [{"message": {"content": response}}],
        "usage": {},
    }, sort_keys=True).encode("utf-8")
    accepted, rejected, parsed = gen.parse_and_verify(response, window["text_O"][:14000])
    accepted_rows = [
        {"attempt_index": 0, "seed": seed, **item} for item in accepted
    ]
    return {
        "schema": "paper2-r1-question-window-v1",
        "ts": "2026-08-30T00:00:00Z",
        "corpus": corpus,
        "window_id": window["window_id"],
        "window_index": index,
        "stratum": window["stratum"],
        "has_code": window["has_code"],
        "source_cluster_id": window["source_cluster_id"],
        "generator_identity_receipt": "/private/receipt.json",
        "generator_identity_sha256": receipt_hash,
        "generator_model_id": MODEL_ID,
        "generation_session_id": session_id,
        "attempts": [{
            "intent_id": f"intent-{corpus}-{index}",
            "attempt_index": 0,
            "seed": seed,
            "outcome": "success",
            "response_model": MODEL_ID,
            "http_status": 200,
            "http_reason": "OK",
            "http_headers": [["Content-Type", "application/json"]],
            "raw_api_response_base64": base64.b64encode(raw_payload).decode("ascii"),
            "raw_api_response_bytes": len(raw_payload),
            "raw_api_response_sha256": hashlib.sha256(raw_payload).hexdigest(),
            "raw_message_content": response,
            "raw_message_content_sha256": hashlib.sha256(
                response.encode("utf-8")
            ).hexdigest(),
            "visible_response": response.strip(),
            "visible_response_sha256": hashlib.sha256(
                response.strip().encode("utf-8")
            ).hexdigest(),
            "pairs_parsed": parsed,
            "accepted": len(accepted),
            "rejected_or_unparsed": rejected,
            "usage": {},
        }],
        "accepted_by_verifier": len(accepted_rows),
        "rejected_or_unparsed": rejected,
        "accepted_questions": accepted_rows,
    }


def test_question_journal_recovery() -> None:
    session_id = "synthetic-session"
    receipt_hash = "a" * 64
    session = {
        "session_id": session_id,
        "identity_receipt": "/private/receipt.json",
        "identity_receipt_sha256": receipt_hash,
        "model_id": MODEL_ID,
    }
    for corpus, counts in gen.FRAME_COUNTS.items():
        windows = []
        index = 0
        for key in gen.STRATA:
            stratum, code_name = key.split("-")
            for _ in range(counts[key]):
                windows.append({
                    "window_id": f"{corpus.lower()}{index:04d}",
                    "stratum": stratum,
                    "has_code": int(code_name == "code"),
                    "source_cluster_id": f"cluster{index:04d}",
                    "text_O": "This excerpt contains value exactly.",
                })
                index += 1
        windows = gen.stratified_window_order(windows)
        events = [
            synthetic_question_event(
                corpus=corpus,
                window=window,
                index=i,
                session_id=session_id,
                receipt_hash=receipt_hash,
            )
            for i, window in enumerate(windows)
        ]
        # Crash before journal append: an empty journal validates but cannot
        # produce a frozen set. Crash after one fsynced event: it validates and
        # is safely resumable, but still cannot be promoted.
        gen.validate_journal([], windows, corpus=corpus, sessions={session_id: session}, base_seed=42)
        gen.validate_journal(events[:1], windows, corpus=corpus, sessions={session_id: session}, base_seed=42)
        expect_failure(
            lambda: gen.derive_question_rows(events[:1], corpus, 42),
            "partial question journal was promoted",
        )
        gen.validate_journal(events, windows, corpus=corpus, sessions={session_id: session}, base_seed=42)
        ledger = []
        for event in events:
            attempt = event["attempts"][0]
            intent_id = attempt["intent_id"]
            prompt = gen.PROMPT.format(window=(
                windows[event["window_index"]]["text_O"][:14000]
            ))
            ledger.extend([
                {
                    "schema": "paper2-r1-generation-intent-v1",
                    "intent_id": intent_id,
                    "corpus": corpus,
                    "window_id": event["window_id"],
                    "window_index": event["window_index"],
                    "attempt_index": 0,
                    "seed": attempt["seed"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "generation_session_id": session_id,
                    "generator_identity_receipt": session["identity_receipt"],
                    "generator_identity_sha256": receipt_hash,
                    "generator_model_id": MODEL_ID,
                },
                {
                    "schema": "paper2-r1-generation-terminal-v1",
                    "intent_id": intent_id,
                    "status": "success",
                    "attempt": attempt,
                },
            ])
        gen.validate_generation_attempt_ledger(
            ledger,
            windows,
            corpus=corpus,
            sessions={session_id: session},
            base_seed=42,
            journal_rows=events,
        )
        expect_failure(
            lambda: gen.validate_generation_attempt_ledger(
                ledger[:-1],
                windows,
                corpus=corpus,
                sessions={session_id: session},
                base_seed=42,
                journal_rows=events,
            ),
            "kill after generation intent/response before terminal was resumable",
        )
        selected = gen.derive_question_rows(events, corpus, 42)
        assert len(selected) == sum(gen.QUOTAS[corpus].values())
        assert len({row["window_id"] for row in selected}) == len(selected)
        with tempfile.TemporaryDirectory(prefix="p2r1-questions-") as temp:
            frozen = Path(temp) / "questions.jsonl"
            gen.write_questions_atomic(frozen, selected)
            before_bytes = frozen.read_bytes()
            before_stat = frozen.stat()
            gen.write_questions_atomic(frozen, selected)
            after_stat = frozen.stat()
            assert frozen.read_bytes() == before_bytes
            assert (after_stat.st_ino, after_stat.st_mtime_ns) == (
                before_stat.st_ino, before_stat.st_mtime_ns
            ), "completed question resume rewrote the frozen file"


def test_grid_write_validate_resume(root: Path) -> None:
    outdir = root / "synthetic-grid"
    outdir.mkdir()
    contract_hash = "c" * 64
    receipt_hash = "d" * 64
    session_id = "synthetic-grid-session-1"
    deployment = {
        "tier": "Q4KXL",
        "server_model_id": MODEL_ID,
        "engine": "llama.cpp",
        "engine_revision": "selftest",
        "artifact_sha256": "e" * 64,
        "artifact_record_sha256": "f" * 64,
    }
    append_session(outdir, {
        "schema": "paper2-r1-session-v1",
        "session_id": session_id,
        "run_contract_sha256": contract_hash,
        "identity_receipt": "/private/receipt-1.json",
        "identity_receipt_sha256": receipt_hash,
        **deployment,
    })
    question = {
        "qid": "q0",
        "window_id": "w0",
        "source_cluster_id": "c0",
        "generator_identity_sha256": "a" * 64,
        "stratum": "S",
        "has_code": 0,
        "question": "Which exact token appears?",
        "answer": "value",
    }
    def make_row(condition: str, sid: str, receipt_path: str) -> dict:
        intent_id = hashlib.sha256(f"{sid}|q0|{condition}".encode()).hexdigest()[:32]
        raw_body = json.dumps({
            "model": MODEL_ID,
            "choices": [{"message": {"content": "ANSWER: value"}}],
            "usage": {},
        }, sort_keys=True).encode()
        api_evidence = {
            **grid.transport_evidence(
                raw_body, status=200, reason="OK",
                headers=[["Content-Type", "application/json"]],
            ),
            "response_model": MODEL_ID,
            "usage": {},
            "raw_message_content": "ANSWER: value",
            "raw_message_content_sha256": hashlib.sha256(
                b"ANSWER: value"
            ).hexdigest(),
            "visible_response": "ANSWER: value",
            "visible_response_sha256": hashlib.sha256(
                b"ANSWER: value"
            ).hexdigest(),
        }
        return {
            "schema": "paper2-r1-response-v1",
            "request_intent_id": intent_id,
            "request_outcome": "success",
            "request_error_type": None,
            "request_error": None,
            "api_evidence": api_evidence,
            "session_id": sid,
            "identity_receipt": receipt_path,
            "identity_receipt_sha256": receipt_hash,
            "run_contract_sha256": contract_hash,
            **deployment,
            "qid": "q0",
            "window_id": "w0",
            "source_cluster_id": "c0",
            "question_generator_identity_sha256": "a" * 64,
            "condition": condition,
            "stratum": "S",
            "has_code": 0,
            "request_started_at": "2026-08-30T00:00:00Z",
            "request_finished_at": "2026-08-30T00:00:01Z",
            "response": "ANSWER: value",
            "answer_extracted": "value",
            "has_marker": 1,
            "gold": "value",
            "exact": 1,
            "lenient": 1,
            "wall_s": 1.0,
            "identity_guard_before_s": 0.01,
            "identity_guard_after_s": 0.01,
            "usage": {},
        }

    rows = [
        make_row(condition, session_id, "/private/receipt-1.json")
        for condition in ("O", "A")
    ]
    output = outdir / "p2grid.jsonl"
    ledger = outdir / "request_ledger.jsonl"
    windows = {"w0": {
        "window_id": "w0", "text_O": "value", "text_A": "value",
        "text_L2": "value", "text_E": "value",
    }}

    def append_eval_evidence(row: dict) -> dict:
        cond = row["condition"]
        text = windows["w0"]["text_O" if cond == "O" else f"text_{cond}"]
        prompt = grid.PROMPT.format(window=text, q=question["question"])
        append_checked_jsonl(ledger, {
            "schema": "paper2-r1-eval-intent-v1",
            "intent_id": row["request_intent_id"],
            "run_contract_sha256": contract_hash,
            "session_id": row["session_id"],
            "identity_receipt_sha256": receipt_hash,
            "tier": deployment["tier"],
            "server_model_id": MODEL_ID,
            "qid": "q0",
            "condition": cond,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "sampling": {
                "temperature": 0.2, "top_p": 0.95, "seed": 42,
                "max_tokens": 2048, "timeout_s": 600,
            },
        })
        stored = append_checked_jsonl(output, row)
        append_checked_jsonl(ledger, {
            "schema": "paper2-r1-eval-terminal-v1",
            "intent_id": row["request_intent_id"],
            "status": row["request_outcome"],
            "response_event_sha256": stored["_jsonl_sha256"],
        })
        return stored

    for row in rows:
        append_eval_evidence(row)
    kwargs = {
        "outdir": outdir,
        "output": output,
        "run_contract_sha256": contract_hash,
        "deployment": deployment,
        "questions": {"q0": question},
        "conds": ["O", "A", "L2", "E"],
    }
    first = grid.validate_prior_rows(**kwargs)
    grid.validate_request_ledger(
        ledger, responses=first, questions={"q0": question}, windows=windows,
        conds=kwargs["conds"], run_contract_sha256=contract_hash,
        deployment=deployment,
    )
    assert not grid.item_complete(first, "q0", kwargs["conds"])
    prefix = output.read_bytes()

    resumed_session = "synthetic-grid-session-2"
    append_session(outdir, {
        "schema": "paper2-r1-session-v1",
        "session_id": resumed_session,
        "run_contract_sha256": contract_hash,
        "identity_receipt": "/private/receipt-2.json",
        "identity_receipt_sha256": receipt_hash,
        **deployment,
    })
    missing = [
        make_row(condition, resumed_session, "/private/receipt-2.json")
        for condition in ("L2", "E")
    ]
    for row in missing:
        append_eval_evidence(row)
    assert output.read_bytes().startswith(prefix), "resume changed prior response bytes"
    rows.extend(missing)
    resumed = grid.validate_prior_rows(**kwargs)
    grid.validate_request_ledger(
        ledger, responses=resumed, questions={"q0": question}, windows=windows,
        conds=kwargs["conds"], run_contract_sha256=contract_hash,
        deployment=deployment,
    )
    assert grid.item_complete(resumed, "q0", kwargs["conds"])
    assert resumed[("q0", "O")]["session_id"] == session_id
    assert resumed[("q0", "L2")]["session_id"] == resumed_session

    unmatched = root / "unmatched-eval-ledger.jsonl"
    orphan = dict(rows[0])
    orphan["request_intent_id"] = "9" * 32
    text = windows["w0"]["text_O"]
    prompt = grid.PROMPT.format(window=text, q=question["question"])
    append_checked_jsonl(unmatched, {
        "schema": "paper2-r1-eval-intent-v1",
        "intent_id": orphan["request_intent_id"],
        "run_contract_sha256": contract_hash,
        "session_id": session_id,
        "identity_receipt_sha256": receipt_hash,
        "tier": deployment["tier"],
        "server_model_id": MODEL_ID,
        "qid": "q0",
        "condition": "O",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "sampling": {
            "temperature": 0.2, "top_p": 0.95, "seed": 42,
            "max_tokens": 2048, "timeout_s": 600,
        },
    })
    expect_failure(
        lambda: grid.validate_request_ledger(
            unmatched, responses={}, questions={"q0": question}, windows=windows,
            conds=kwargs["conds"], run_contract_sha256=contract_hash,
            deployment=deployment,
        ),
        "kill after evaluation intent/response before terminal was resumable",
    )

    append_checked_jsonl(output, rows[0])
    expect_failure(
        lambda: grid.validate_prior_rows(**kwargs),
        "duplicate response key was accepted",
    )

    # Restore a unique file, then prove scores are recomputed rather than
    # trusted from retained booleans.
    output.unlink()
    checked_jsonl_head_path(output).unlink()
    rows[0]["exact"] = 0
    for row in rows:
        append_checked_jsonl(output, row)
    expect_failure(
        lambda: grid.validate_prior_rows(**kwargs),
        "tampered prior score was accepted",
    )


def test_killed_mid_append_invalidation(root: Path) -> None:
    """Simulate the exact durable state left by SIGKILL during one append."""
    for relative in (
        "questions/question_journal.jsonl",
        "questions/question_attempts.jsonl",
        "questions/generation_sessions.jsonl",
        "grid/run_sessions.jsonl",
        "grid/request_ledger.jsonl",
        "grid/p2grid.jsonl",
        "grid/failures.jsonl",
    ):
        path = root / relative
        append_checked_jsonl(path, {"schema": "synthetic-complete-v1", "index": 0})
        verified_prefix = path.read_bytes()
        with path.open("ab") as handle:
            handle.write(b'{"schema":"synthetic-torn-v1","index":')
            handle.flush()
            os.fsync(handle.fileno())
        original = path.read_bytes()
        expect_failure(
            lambda p=path: read_checked_jsonl(p, recover_torn_tail=True),
            f"killed-mid-append tail was resumed for {relative}",
        )
        receipts = torn_tail_recovery_receipts(path)
        assert len(receipts) == 1, f"missing persistent recovery receipt for {relative}"
        receipt = json.loads(receipts[0].read_text())
        assert receipt["promotion_eligible"] is False
        assert path.read_bytes() == verified_prefix
        quarantine = Path(receipt["quarantine"])
        assert quarantine.read_bytes() == original
        expect_failure(
            lambda p=path: read_checked_jsonl(p),
            f"tail-recovered official stream later became eligible: {relative}",
        )

    # A complete but corrupted event is not a recoverable tail and may never
    # be dropped. It fails without creating a forensic recovery receipt.
    corrupt = root / "complete-checksum-corruption.jsonl"
    append_checked_jsonl(corrupt, {"schema": "synthetic-complete-v1", "index": 1})
    corrupt.write_bytes(corrupt.read_bytes().replace(b'"index": 1', b'"index": 2'))
    expect_failure(
        lambda: read_checked_jsonl(corrupt, recover_torn_tail=True),
        "checksum-corrupt complete event was silently recovered",
    )
    assert not torn_tail_recovery_receipts(corrupt)

    deleted_tail = root / "deleted-tail.jsonl"
    append_checked_jsonl(deleted_tail, {"schema": "synthetic-v1", "index": 0})
    append_checked_jsonl(deleted_tail, {"schema": "synthetic-v1", "index": 1})
    lines = deleted_tail.read_bytes().splitlines(keepends=True)
    deleted_tail.write_bytes(b"".join(lines[:-1]))
    expect_failure(
        lambda: read_checked_jsonl(deleted_tail),
        "selective final-event deletion escaped the chain/head receipt",
    )

    deleted_stream = root / "deleted-stream.jsonl"
    append_checked_jsonl(deleted_stream, {"schema": "synthetic-v1", "index": 0})
    deleted_stream.unlink()
    expect_failure(
        lambda: read_checked_jsonl(deleted_stream),
        "whole-stream deletion escaped the surviving head receipt",
    )

    # Exact power-loss state after the event fsync but before the atomic head
    # replacement: the file has one additional complete chained event while
    # the owner-only head still anchors the previous event.
    stale_head = root / "post-fsync-pre-head-crash.jsonl"
    append_checked_jsonl(stale_head, {"schema": "synthetic-v1", "index": 0})
    head = json.loads(checked_jsonl_head_path(stale_head).read_text())
    stranded = checked_event(
        {"schema": "synthetic-v1", "index": 1},
        previous_sha256=head["head_sha256"],
    )
    with stale_head.open("ab") as handle:
        handle.write((json.dumps(stranded, sort_keys=True) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    attempt_id = "p2r1-stale-head-selftest"
    attempt_root = root / "stale-head-attempt"
    try:
        read_checked_jsonl(stale_head)
    except RunSafetyError as exc:
        attempt_ambiguity_error(
            attempt_root,
            attempt_id=attempt_id,
            reason="synthetic-post-fsync-pre-head-ambiguity",
            error=exc,
        )
    else:
        raise AssertionError("post-fsync/pre-head crash state was accepted")
    marker = json.loads((attempt_root / "ATTEMPT_INVALIDATED.json").read_text())
    assert marker["promotion_eligible"] is False
    expect_failure(
        lambda: require_attempt_eligible(attempt_root),
        "stale-head attempt marker did not block restart/promotion",
    )

    oversized = b"x" * (grid.MAX_RAW_API_RESPONSE_BYTES + 1)
    cap_cases = []
    for status, reason, outcome in ((200, "OK", "success"), (500, "error", "error")):
        cap_cases.extend((
            (
                f"evaluation-{outcome}",
                lambda status=status, reason=reason: grid.transport_evidence(
                    oversized, status=status, reason=reason, headers=[]
                ),
            ),
            (
                f"generation-{outcome}",
                lambda status=status, reason=reason: gen.generator_transport_evidence(
                    oversized, status=status, reason=reason, headers=[]
                ),
            ),
        ))
    for label, callable_obj in cap_cases:
        expect_failure(
            callable_obj,
            f"oversized {label} body was converted into a scored retry/outcome",
        )

    summary = root / "grid" / "summary.json"
    atomic_summary(summary, {"schema": "summary-v1", "complete": False})
    atomic_summary(summary, {"schema": "summary-v1", "complete": True})
    assert json.loads(summary.read_text())["complete"] is True
    assert not list(summary.parent.glob("summary.json.tmp.*"))


def test_kill_after_session_before_intent(root: Path) -> None:
    """The exact fsynced state from the session/first-intent crash window."""
    eval_attempt_id = "p2r1-eval-orphan-selftest"
    eval_attempt = root / "eval-attempt"
    eval_outdir = eval_attempt / "P2R1-Q4KXL-TSI"
    append_session(eval_outdir, {
        "schema": "paper2-r1-session-v1",
        "session_id": "registered-without-intent",
    })
    expect_failure(
        lambda: grid.validate_session_usage(
            outdir=eval_outdir,
            ledger_events=[],
            responses={},
            attempt_root=eval_attempt,
            attempt_id=eval_attempt_id,
        ),
        "evaluation session/first-intent crash remained resumable",
    )
    marker = json.loads((eval_attempt / "ATTEMPT_INVALIDATED.json").read_text())
    assert marker["promotion_eligible"] is False
    assert marker["reason"] == "evaluation-session-without-complete-provenance"

    gen_attempt_id = "p2r1-generation-orphan-selftest"
    gen_attempt = root / "generation-attempt"
    expect_failure(
        lambda: gen.validate_generation_session_usage(
            sessions={"registered-without-intent": {}},
            attempt_rows=[],
            journal_rows=[],
            attempt_root=gen_attempt,
            attempt_id=gen_attempt_id,
            corpus="TSI",
        ),
        "question-generation session/first-intent crash remained resumable",
    )
    marker = json.loads((gen_attempt / "ATTEMPT_INVALIDATED.json").read_text())
    assert marker["promotion_eligible"] is False
    assert marker["reason"] == (
        "question-generation-session-without-complete-provenance"
    )


def test_post_request_oserror_marks_attempt(root: Path) -> None:
    """ENOSPC/EIO after a response may never exit without a permanent marker."""
    def raise_io(*_args, **_kwargs):
        raise OSError("synthetic ENOSPC after response")

    real_grid_append = grid.append_checked_jsonl
    grid.append_checked_jsonl = raise_io
    try:
        attempt_id = "p2r1-eval-io-selftest"
        attempt_root = root / "eval-io-attempt"
        expect_failure(
            lambda: grid.append_after_request(
                root / "unused-eval.jsonl", {"schema": "synthetic-v1"},
                attempt_root=attempt_root, attempt_id=attempt_id,
                reason="synthetic-eval-post-response-io",
            ),
            "evaluation post-response OSError escaped without invalidation",
        )
        assert json.loads(
            (attempt_root / "ATTEMPT_INVALIDATED.json").read_text()
        )["promotion_eligible"] is False
    finally:
        grid.append_checked_jsonl = real_grid_append

    real_gen_append = gen.append_checked_jsonl
    gen.append_checked_jsonl = raise_io
    try:
        attempt_id = "p2r1-gen-io-selftest"
        attempt_root = root / "gen-io-attempt"
        expect_failure(
            lambda: gen.append_generation_after_request(
                root / "unused-gen.jsonl", {"schema": "synthetic-v1"},
                attempt_root=attempt_root, attempt_id=attempt_id,
                reason="synthetic-gen-post-response-io",
            ),
            "generation post-response OSError escaped without invalidation",
        )
        assert json.loads(
            (attempt_root / "ATTEMPT_INVALIDATED.json").read_text()
        )["promotion_eligible"] is False
    finally:
        gen.append_checked_jsonl = real_gen_append


def main() -> int:
    test_graphical_desktop_cool_gate()
    test_question_journal_recovery()
    with tempfile.TemporaryDirectory(prefix="p2r1-selftest-") as temp:
        root = Path(temp)
        test_grid_write_validate_resume(root)
        test_killed_mid_append_invalidation(root)
        test_kill_after_session_before_intent(root)
        test_post_request_oserror_marks_attempt(root)
        artifact = root / "artifact.bin"
        artifact.write_bytes(b"frozen-test-artifact")
        artifact_record = root / "artifact.json"
        source_manifest = root / "source-manifest.json"
        source_manifest.write_text(json.dumps({
            "schema": "qwen38-model-artifacts-v1",
            "artifacts": [{
                "artifact_id": "synthetic",
                "upstream_repo": "synthetic/local",
                "revision": "0" * 40,
                "filename": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": guard.sha256_file(artifact),
                "license": "test-only",
            }],
        }, indent=2, sort_keys=True) + "\n")
        real_source_path = guard.MODEL_SOURCE_MANIFEST
        real_source_sha = guard.MODEL_SOURCE_MANIFEST_SHA256
        guard.MODEL_SOURCE_MANIFEST = source_manifest
        guard.MODEL_SOURCE_MANIFEST_SHA256 = guard.sha256_file(source_manifest)
        guard.create_artifact_record(
            artifact, artifact_record,
            source_manifest_path=source_manifest,
            artifact_id="synthetic",
        )
        official_alias = "P2R1-Q4KXL-0123456789ab"
        official_argv = [
            sys.executable, "-m", str(artifact), "-ngl", "33", "-c", "8192",
            "-ctk", "q8_0", "-ctv", "q8_0", "--flash-attn", "on",
            "--jinja", "--host", "127.0.0.1", "--port", "18090",
            "-t", "16", "--alias", official_alias, "-lv", "4",
        ]
        process_contract = {"executable": str(Path(sys.executable).resolve())}
        launch_contract = guard.official_llama_launch_contract(
            cmdline=official_argv, process=process_contract,
            artifact_path=str(artifact), expected_model_id=official_alias,
            base_url="http://127.0.0.1:18090",
        )
        assert launch_contract
        assert launch_contract["options"]["log_verbosity"] == 4

        def reject_llama_argv(argv: list[str], message: str) -> None:
            expect_failure(
                lambda: guard.official_llama_launch_contract(
                    cmdline=argv, process=process_contract,
                    artifact_path=str(artifact), expected_model_id=official_alias,
                    base_url="http://127.0.0.1:18090",
                ),
                message,
            )

        wrong_argv = list(official_argv)
        wrong_argv[4] = "32"
        reject_llama_argv(
            wrong_argv,
            "llama launch contract accepted wrong GPU-layer count",
        )
        reject_llama_argv(
            official_argv[:-2],
            "llama launch contract accepted omitted -lv 4",
        )
        reject_llama_argv(
            official_argv + ["-lv", "4"],
            "llama launch contract accepted duplicate -lv 4",
        )
        reject_llama_argv(
            official_argv[:-6] + ["-lv", "4"] + official_argv[-6:-2],
            "llama launch contract accepted reordered -lv 4",
        )
        wrong_verbosity = list(official_argv)
        wrong_verbosity[-1] = "3"
        reject_llama_argv(
            wrong_verbosity,
            "llama launch contract accepted wrong -lv value",
        )

        iq2_alias = "P2R1-IQ2S-0123456789ab"
        iq2_argv = list(official_argv)
        iq2_argv[4] = "99"
        iq2_argv[-3] = iq2_alias
        assert guard.official_llama_launch_contract(
            cmdline=iq2_argv, process=process_contract,
            artifact_path=str(artifact), expected_model_id=iq2_alias,
            base_url="http://127.0.0.1:18090",
        )

        assert campaign.exact_llama_offload_readback(
            "load_tensors: offloaded 33/66 layers to GPU\n", "33/66"
        ) == "33/66"
        for placement_log, message in (
            ("model loaded\n", "missing llama placement line was accepted"),
            (
                "load_tensors: offloaded 33/66 layers to GPU\n" * 2,
                "duplicate llama placement lines were accepted",
            ),
            (
                "load_tensors: offloaded 33/65 layers to GPU\n",
                "wrong llama placement line was accepted",
            ),
        ):
            expect_failure(
                lambda value=placement_log: campaign.exact_llama_offload_readback(
                    value, "33/66"
                ),
                message,
            )
        log = root / "server.log"
        log.write_text("self-test server ready\n")

        port = unused_port()
        post_marker = root / "unexpected-post"
        global POST_MARKER
        POST_MARKER = str(post_marker)
        process = multiprocessing.Process(target=serve, args=(port,))
        process.start()
        try:
            process_record = root / "process.json"
            guard.create_process_record(process.pid, process_record)
            deadline = time.monotonic() + 5
            while not guard.listening_pids(port):
                if time.monotonic() > deadline:
                    raise AssertionError("fake identity server did not bind")
                time.sleep(0.02)
            base_url = f"http://127.0.0.1:{port}"

            # The live path must never read artifact contents. If it calls the
            # full hasher, this replacement raises and the test fails.
            real_hasher = guard.sha256_path

            def forbidden_weight_read(_path: Path):
                raise AssertionError("live receipt read full artifact contents")

            guard.sha256_path = forbidden_weight_read
            receipt_path = root / "receipt.json"
            receipt = guard.create_receipt(
                base_url=base_url,
                expected_pid=process.pid,
                expected_model_id=MODEL_ID,
                expected_owned_by="llamacpp",
                engine="llama.cpp",
                engine_revision="selftest",
                artifact_record_path=artifact_record,
                process_record_path=process_record,
                server_log=log,
                receipt_path=receipt_path,
                cmd_fragments=[],
                forbidden_log_patterns=["couldn't bind"],
                expected_cwd=None,
                config_file=None,
                required_config_lines=[],
                required_log_patterns=[],
                wait_seconds=1,
            )
            assert receipt["expected_model_id"] == MODEL_ID
            guard.validate_receipt(receipt_path, full_artifact_hash=False)
            guard.sha256_path = real_hasher

            config = root / "dedicated.yml"
            config.write_text("model_name: expected-artifact\n")
            config_receipt = root / "config-receipt.json"
            guard.create_receipt(
                base_url=base_url,
                expected_pid=process.pid,
                expected_model_id=MODEL_ID,
                expected_owned_by="llamacpp",
                engine="llama.cpp",
                engine_revision="selftest",
                artifact_record_path=artifact_record,
                process_record_path=process_record,
                server_log=log,
                receipt_path=config_receipt,
                cmd_fragments=[],
                forbidden_log_patterns=[],
                expected_cwd=str(Path.cwd()),
                config_file=config,
                required_config_lines=["model_name: expected-artifact"],
                required_log_patterns=["self-test server ready"],
                wait_seconds=1,
            )
            config.write_text("model_name: wrong-artifact\n")
            expect_failure(
                lambda: guard.validate_receipt(config_receipt, full_artifact_hash=False),
                "same API model ID with changed config/artifact selector was accepted",
            )

            changed = json.loads(receipt_path.read_text())
            changed["identity"]["process_fingerprint"]["starttime_ticks"] += 1
            changed_path = root / "reused-pid.json"
            changed_path.write_text(json.dumps(changed))
            expect_failure(
                lambda: guard.validate_receipt(changed_path, full_artifact_hash=False),
                "PID start-time mismatch did not fail closed",
            )

            expect_failure(
                lambda: guard.create_receipt(
                    base_url=base_url,
                    expected_pid=process.pid,
                    expected_model_id="wrong-model",
                    expected_owned_by="llamacpp",
                    engine="llama.cpp",
                    engine_revision="selftest",
                    artifact_record_path=artifact_record,
                    process_record_path=process_record,
                    server_log=log,
                    receipt_path=root / "wrong-model.json",
                    cmd_fragments=[],
                    forbidden_log_patterns=[],
                    expected_cwd=None,
                    config_file=None,
                    required_config_lines=[],
                    required_log_patterns=[],
                    wait_seconds=0.1,
                ),
                "model mismatch did not fail closed",
            )
            unrelated = multiprocessing.Process(target=idle_process)
            unrelated.start()
            try:
                unrelated_record = root / "unrelated-process.json"
                guard.create_process_record(unrelated.pid, unrelated_record)
                expect_failure(
                    lambda: guard.create_receipt(
                        base_url=base_url,
                        expected_pid=unrelated.pid,
                        expected_model_id=MODEL_ID,
                        expected_owned_by="llamacpp",
                        engine="llama.cpp",
                        engine_revision="selftest",
                        artifact_record_path=artifact_record,
                        process_record_path=unrelated_record,
                        server_log=log,
                        receipt_path=root / "wrong-pid.json",
                        cmd_fragments=[],
                        forbidden_log_patterns=[],
                        expected_cwd=None,
                        config_file=None,
                        required_config_lines=[],
                        required_log_patterns=[],
                        wait_seconds=0.1,
                    ),
                    "foreign port owner did not fail closed",
                )
            finally:
                unrelated.terminate()
                unrelated.join(timeout=5)
            expect_failure(
                lambda: guard.assert_port_free("127.0.0.1", port),
                "occupied port passed preflight",
            )

            # Historical failure shape: a stale healthy listener remains while
            # the intended process exits. Receipt creation must abort before
            # any generation POST reaches the stale endpoint.
            dead = multiprocessing.Process(target=idle_process)
            dead.start()
            dead_record = root / "dead-process.json"
            guard.create_process_record(dead.pid, dead_record)
            dead.terminate()
            dead.join(timeout=5)
            expect_failure(
                lambda: guard.create_receipt(
                    base_url=base_url,
                    expected_pid=dead.pid,
                    expected_model_id=MODEL_ID,
                    expected_owned_by="llamacpp",
                    engine="llama.cpp",
                    engine_revision="selftest",
                    artifact_record_path=artifact_record,
                    process_record_path=dead_record,
                    server_log=log,
                    receipt_path=root / "dead-intended.json",
                    cmd_fragments=[],
                    forbidden_log_patterns=[],
                    expected_cwd=None,
                    config_file=None,
                    required_config_lines=[],
                    required_log_patterns=[],
                    wait_seconds=0.1,
                ),
                "dead intended process with stale healthy listener was accepted",
            )
            assert not post_marker.exists(), "identity guard sent a generation POST"

            outdir = root / "P2R1-Q4KXL-TSI"
            contract = {"schema": "test", "input_sha256": "a" * 64}
            path = ensure_run_contract(outdir, contract)
            before = path.read_bytes()
            ensure_run_contract(outdir, contract)
            expect_failure(
                lambda: ensure_run_contract(
                    outdir, {"schema": "test", "input_sha256": "b" * 64}
                ),
                "contract mismatch overwrote existing run",
            )
            assert path.read_bytes() == before
        finally:
            guard.MODEL_SOURCE_MANIFEST = real_source_path
            guard.MODEL_SOURCE_MANIFEST_SHA256 = real_source_sha
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        guard.assert_port_free("127.0.0.1", port)
    print(
        "P2R1 self-test: identity/PID/config, stale-listener, journal-crash, "
        "mid-item write-once resume, killed-append and orphan-session attempt "
        "invalidation, graphical-desktop admission, exact raw-generator replay, "
        "live no-weight-read, and non-overwrite checks passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
