#!/usr/bin/env python3
"""Rebuild Paper 2 claims exclusively from validated P2R1 evidence.

Analysis-only: reads the per-item grid JSONLs, the window/question files, the
GGUF headers of the two llama.cpp artifacts, and the EXL3 artifact directory,
then writes derived/claims.json and derived/claims.tex (macros consumed by
main.tex and by scripts/make_figures.py).

PRIVACY INVARIANT: the tsi arm's inputs are private. This script may read them
but must emit only aggregate counts, rates, sizes, and test statistics. No
window text, question text, answer text, response text, or per-item row may
appear in any output. test_claims.py enforces a scan for this.

The quarantined P2G campaign is never an input. A private promotion receipt
from scripts/validate_r1.py is a hard prerequisite.

Statistical functions are imported from paper4/scripts/verify_claims.py — the
audited implementations (exact McNemar, Newcombe method-10 paired interval,
Fisher exact, Wilson) — so papers 2/3/4 share one statistical code path. The
cross-tier analysis below is paired across conditions and deployment tiers.
Its interval and primary CR2/Satterthwaite t test cluster on the private
source-conversation grouping under an intercept-only iid working model.
The exhaustive sign-flip calculation is only a secondary sensitivity analysis
conditional on cluster-sign symmetry; it is not treated as assumption-free.
The grouping unit is the source-conversation ID, not the window or question ID.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parent
RESULTS = REPO_ROOT / "testsuite" / "evals" / "results"
P2 = REPO_ROOT / "testsuite" / "paper2"
OUT_JSON = PAPER_DIR / "derived" / "claims.json"
OUT_TEX = PAPER_DIR / "derived" / "claims.tex"

_spec = importlib.util.spec_from_file_location(
    "p4_verify", REPO_ROOT / "paper4" / "scripts" / "verify_claims.py")
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)

TIERS = {"Qf": "Q4KXL", "Ex": "EXL3", "Iq": "IQ2S"}
CONDS = {"O": "O", "A": "A", "L": "L2", "E": "E"}
CORPORA = {"Tsi": "TSI", "Wc": "WC"}
R1 = P2 / "r1"
sys.path.insert(0, str(P2))
from identity_guard import load_artifact_record  # noqa: E402
from protocol_bindings import analysis_source_hashes  # noqa: E402
from runtime_provenance import (  # noqa: E402
    CUDA_RUNTIME_CONTRACT,
    sanitize_exl3_python_runtime_content,
    sanitize_llama_cuda_runtime,
)

ANALYSIS_IMPLEMENTATION = REPO_ROOT / "paper4" / "scripts" / "verify_claims.py"
PROMOTION = R1 / "private" / "promotion.json"
PROMOTION_STATUS = "validated-for-claims-rebuild-not-publication"
PROMOTION_STUDY_STATUS = (
    "corrective exploratory rerun; not independent confirmation"
)
GENERATOR_DEPLOYMENT_FIELDS = (
    "model_id",
    "engine",
    "engine_revision",
    "artifact_sha256",
    "artifact_record_sha256",
)
CLAIMS_OUTPUT_MODE = 0o600


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _safe_parent(path: Path) -> Path:
    """Require an owner-controlled, non-symlink output directory."""
    parent = path.parent
    try:
        metadata = parent.lstat()
    except FileNotFoundError as exc:
        raise AssertionError(f"claims output directory is missing: {parent}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise AssertionError(f"claims output directory is unsafe: {parent}")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise AssertionError(
            f"claims output directory is not owner-controlled: {parent}"
        )
    return parent


def _read_safe_output(path: Path, *, allow_missing: bool = False) -> bytes | None:
    """Read one regular owner-controlled output without following symlinks."""
    _safe_parent(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise AssertionError(f"claims output is missing: {path}")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise AssertionError(f"claims output path is unsafe: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise AssertionError(f"claims output changed while opening: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read()
    finally:
        os.close(descriptor)
    if len(value) != metadata.st_size:
        raise AssertionError(f"claims output changed while reading: {path}")
    return value


def _snapshot_output(path: Path) -> dict | None:
    value = _read_safe_output(path, allow_missing=True)
    if value is None:
        return None
    metadata = path.lstat()
    return {
        "identity": (metadata.st_dev, metadata.st_ino),
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "mode": stat.S_IMODE(metadata.st_mode),
        "bytes": value,
        "sha256": _sha256_bytes(value),
    }


def _snapshot_unchanged(path: Path, snapshot: dict | None) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return snapshot is None
    if snapshot is None or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        return False
    current = _read_safe_output(path)
    return (
        (metadata.st_dev, metadata.st_ino) == snapshot["identity"]
        and metadata.st_size == snapshot["size"]
        and metadata.st_mtime_ns == snapshot["mtime_ns"]
        and _sha256_bytes(current) == snapshot["sha256"]
    )


def _stage_owner_only(parent: Path, name: str, content: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{name}.", dir=parent)
    staged = Path(raw_path)
    try:
        os.fchmod(descriptor, CLAIMS_OUTPUT_MODE)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        os.close(descriptor)
        staged.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return staged


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_claim_outputs(outputs: dict[Path, bytes]) -> None:
    """Replace the claims pair from fully staged owner-only files.

    Both files are staged and fsynced before either canonical pathname changes.
    A normal exception during the two replacements restores the exact prior pair;
    each individual pathname transition is atomic. Callers must still bind the
    JSON and TeX bytes together at every render boundary.
    """
    if len(outputs) != 2 or len(set(outputs)) != 2:
        raise AssertionError("claims output transaction requires two distinct paths")
    parents = {_safe_parent(path) for path in outputs}
    if len(parents) != 1:
        raise AssertionError("claims outputs must share one directory")
    parent = parents.pop()
    snapshots = {path: _snapshot_output(path) for path in outputs}
    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for path, content in outputs.items():
            if not isinstance(content, bytes):
                raise TypeError("claims output content must be bytes")
            staged[path] = _stage_owner_only(parent, path.name, content)
        if any(
            not _snapshot_unchanged(path, snapshots[path])
            for path in outputs
        ):
            raise AssertionError("claims output changed during transaction setup")
        for path in outputs:
            os.replace(staged[path], path)
            committed.append(path)
        _fsync_directory(parent)
    except BaseException as error:
        rollback_error: BaseException | None = None
        try:
            for path in reversed(committed):
                prior = snapshots[path]
                if prior is None:
                    metadata = path.lstat()
                    if (
                        stat.S_ISLNK(metadata.st_mode)
                        or not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_uid != os.geteuid()
                    ):
                        raise AssertionError(
                            f"unsafe claims output appeared during rollback: {path}"
                        )
                    path.unlink()
                else:
                    restored = _stage_owner_only(parent, path.name, prior["bytes"])
                    try:
                        os.chmod(restored, prior["mode"] & ~0o022)
                        os.replace(restored, path)
                    finally:
                        restored.unlink(missing_ok=True)
            _fsync_directory(parent)
        except BaseException as exc:
            rollback_error = exc
        if rollback_error is not None:
            raise RuntimeError(
                "claims output transaction failed and rollback was incomplete"
            ) from rollback_error
        raise error
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)

    for path, expected in outputs.items():
        observed = _read_safe_output(path)
        metadata = path.lstat()
        if observed != expected or stat.S_IMODE(metadata.st_mode) != CLAIMS_OUTPUT_MODE:
            raise AssertionError(f"claims output verification failed: {path}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generation_attempt_summary(rows: list[dict]) -> dict:
    """Privacy-safe aggregate of the validated generator attempt ledger."""
    terminals = [
        row for row in rows
        if row.get("schema") == "paper2-r1-generation-terminal-v1"
    ]
    outcomes = Counter(row.get("status") for row in terminals)
    allowed = {"success", "request-error-identity-preserved"}
    if set(outcomes) - allowed:
        raise AssertionError("promoted question ledger contains ineligible outcome")
    successful = [row["attempt"] for row in terminals if row["status"] == "success"]
    request_errors = [
        row["attempt"] for row in terminals
        if row["status"] == "request-error-identity-preserved"
    ]
    error_types = Counter(row.get("error_type") for row in request_errors)
    if None in error_types or any(not isinstance(key, str) for key in error_types):
        raise AssertionError("generator request-error aggregate lacks an error type")
    return {
        "total_http_attempts": len(terminals),
        "successful_http_attempts": len(successful),
        "identity_preserved_request_errors": len(request_errors),
        "request_error_type_counts": dict(sorted(error_types.items())),
        "pairs_parsed_after_successful_requests": sum(
            row["pairs_parsed"] for row in successful
        ),
        "candidate_pairs_accepted_by_verifier": sum(
            row["accepted"] for row in successful
        ),
        "candidate_pairs_rejected_or_unparsed_after_successful_requests": sum(
            row["rejected_or_unparsed"] for row in successful
        ),
    }


def build_corpus_claim(
    windows: list[dict], question_manifest: dict,
    question_attempts: list[dict], questions: list[dict],
) -> dict:
    """Build the public aggregate for one corpus from validated private rows."""
    generator_summary = generation_attempt_summary(question_attempts)
    successful_rejected = generator_summary[
        "candidate_pairs_rejected_or_unparsed_after_successful_requests"
    ]
    if (
        generator_summary["candidate_pairs_accepted_by_verifier"]
        != question_manifest["accepted_by_verifier_before_target_truncation"]
        or successful_rejected
        + 2 * generator_summary["identity_preserved_request_errors"]
        != question_manifest["rejected_or_unparsed"]
    ):
        raise AssertionError(
            "generator aggregate disagrees with the frozen manifest"
        )
    chars_o = sum(window["chars_O"] for window in windows)
    retention = {}
    for condition_tag, key in (
        ("A", "chars_A"), ("L", "chars_L2"), ("E", "chars_E")
    ):
        retention[condition_tag] = sum(
            window[key] for window in windows
        ) / chars_o
    strata = {}
    for window in windows:
        key = f"{window['stratum']}-{'code' if window['has_code'] else 'nocode'}"
        strata[key] = strata.get(key, 0) + 1
    return {
        "windows": len(windows),
        "strata_counts": strata,
        "char_retention_vs_O": retention,
        "questions_kept": question_manifest["written_items"],
        "question_generation": generator_summary,
        "candidate_pair_rejection_rate_after_successful_requests": (
            successful_rejected
            / max(
                1,
                successful_rejected
                + generator_summary["candidate_pairs_accepted_by_verifier"],
            )
        ),
        "questions_in_file": len(questions),
    }


def require_canonical_public_cuda_runtime(
    private_runtime: dict, public_runtime: dict,
) -> dict:
    """Require the exact path-free projection of the private CUDA receipt."""
    expected = sanitize_llama_cuda_runtime(private_runtime)
    if public_runtime != expected:
        raise AssertionError(
            "promoted CUDA runtime is not the canonical public projection"
        )
    return expected


def load_cell_rows(
    tier_tag: str, corpus_tag: str, result_dirs: dict[str, Path]
) -> dict:
    label = f"P2R1-{TIERS[tier_tag]}-{CORPORA[corpus_tag]}"
    rows = {}
    for r in read_jsonl(result_dirs[label] / "p2grid.jsonl"):
        key = (r["qid"], r["condition"])
        if key in rows:
            raise AssertionError(f"duplicate append-only P2R1 response key: {label}/{key}")
        rows[key] = r
    return rows


def survival_form(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip().strip('"\'`.,:;!?()[]{}')


def validate_promotion_header(value: dict, *, protocol_sha256: str) -> None:
    """Validate the semantics the public claims later repeat verbatim."""
    if (
        not isinstance(value, dict)
        or value.get("schema") != "paper2-r1-private-promotion-v1"
        or value.get("status") != PROMOTION_STATUS
        or value.get("study_status") != PROMOTION_STUDY_STATUS
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("legacy_p2g_accepted") is not False
        or value.get("official_runs") != 6
        or value.get("official_condition_cells") != 24
    ):
        raise AssertionError(
            "P2R1 promotion receipt has invalid status, chronology, protocol, or shape"
        )


def _generator_projection(value: object, *, source: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(GENERATOR_DEPLOYMENT_FIELDS):
        raise AssertionError(f"{source} lacks the exact five-field Q4 deployment")
    projected = {field: value[field] for field in GENERATOR_DEPLOYMENT_FIELDS}
    if any(not isinstance(item, str) or not item for item in projected.values()):
        raise AssertionError(f"{source} has an invalid Q4 deployment field")
    for field in ("artifact_sha256", "artifact_record_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", projected[field]):
            raise AssertionError(f"{source} has an invalid {field}")
    return projected


def _evaluation_projection(value: object, *, source: str) -> dict[str, str]:
    required = {
        "tier",
        "server_model_id", "engine", "engine_revision",
        "artifact_sha256", "artifact_record_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise AssertionError(
            f"{source} lacks the exact six-field Q4 evaluation deployment"
        )
    if value["tier"] != "Q4KXL":
        raise AssertionError(f"{source} does not identify the Q4KXL tier")
    return _generator_projection(
        {
            "model_id": value["server_model_id"],
            "engine": value["engine"],
            "engine_revision": value["engine_revision"],
            "artifact_sha256": value["artifact_sha256"],
            "artifact_record_sha256": value["artifact_record_sha256"],
        },
        source=source,
    )


def validate_deployment_bindings(
    promotion: dict,
    run_contracts: dict[str, dict],
    generation_contracts: dict[str, dict],
) -> None:
    """Bind official labels and both generator contracts to verified Q4 evidence."""
    expected_labels = {
        f"P2R1-{tier}-{corpus}"
        for tier in TIERS.values() for corpus in CORPORA.values()
    }
    if set(run_contracts) != expected_labels:
        raise AssertionError("run-contract set differs from the official six labels")
    promoted_runs = promotion.get("runs")
    if not isinstance(promoted_runs, list):
        raise AssertionError("promotion lacks its official run records")
    promoted_by_label = {
        row.get("label"): row for row in promoted_runs if isinstance(row, dict)
    }
    if set(promoted_by_label) != expected_labels or len(promoted_runs) != 6:
        raise AssertionError("promotion run set differs from the official six labels")

    q4_evaluation: dict[str, dict[str, str]] = {}
    q4_evidence: dict[str, dict] = {}
    for label in sorted(expected_labels):
        match = re.fullmatch(r"P2R1-(Q4KXL|EXL3|IQ2S)-(TSI|WC)", label)
        if match is None:
            raise AssertionError(f"malformed official run label: {label}")
        expected_tier, corpus = match.groups()
        contract = run_contracts[label]
        if (
            not isinstance(contract, dict)
            or contract.get("schema") != "paper2-r1-run-contract-v1"
            or contract.get("label") != label
        ):
            raise AssertionError(f"run contract does not bind its label: {label}")
        deployment = contract.get("deployment")
        if not isinstance(deployment, dict) or deployment.get("tier") != expected_tier:
            raise AssertionError(
                f"official label and deployment tier disagree: {label}"
            )
        evidence = promoted_by_label[label].get("artifact_evidence")
        if not isinstance(evidence, dict) or evidence.get("tier") != expected_tier:
            raise AssertionError(
                f"official label and promoted artifact tier disagree: {label}"
            )
        artifact = evidence.get("artifact")
        if (
            not isinstance(artifact, dict)
            or deployment.get("artifact_sha256") != artifact.get("sha256")
            or deployment.get("artifact_record_sha256")
            != evidence.get("artifact_record_sha256")
        ):
            raise AssertionError(
                f"run deployment and promoted artifact evidence disagree: {label}"
            )
        if expected_tier == "Q4KXL":
            q4_evaluation[corpus] = _evaluation_projection(
                deployment, source=f"{label} evaluation"
            )
            q4_evidence[corpus] = evidence

    if set(q4_evaluation) != set(CORPORA.values()):
        raise AssertionError("both official Q4KXL evaluation deployments are required")
    if q4_evaluation["TSI"] != q4_evaluation["WC"]:
        raise AssertionError("the two official Q4KXL evaluation deployments disagree")
    official_q4 = q4_evaluation["TSI"]
    if (
        official_q4["engine"] != "llama.cpp"
        or official_q4["model_id"]
        != f"P2R1-Q4KXL-{official_q4['artifact_sha256'][:12]}"
    ):
        raise AssertionError("official Q4KXL evaluation model/engine identity is invalid")

    post_artifacts = promotion.get("post_campaign_artifacts")
    post_q4 = (
        post_artifacts.get("Q4KXL") if isinstance(post_artifacts, dict) else None
    )
    if not isinstance(post_q4, dict):
        raise AssertionError("promotion lacks post-campaign Q4KXL evidence")
    if q4_evidence["TSI"] != q4_evidence["WC"]:
        raise AssertionError("the two promoted Q4KXL artifact records disagree")
    observed_q4 = q4_evidence["TSI"]
    if (
        post_q4.get("artifact") != observed_q4.get("artifact")
        or post_q4.get("artifact_record") != observed_q4.get("artifact_record")
        or post_q4.get("artifact_record_sha256")
        != observed_q4.get("artifact_record_sha256")
        or post_q4.get("full_hash_verified") is not True
    ):
        raise AssertionError(
            "official Q4KXL evaluation differs from post-campaign artifact evidence"
        )

    if set(generation_contracts) != set(CORPORA.values()):
        raise AssertionError("both corpus question-generation contracts are required")
    generator_projections: dict[str, dict[str, str]] = {}
    for corpus in sorted(CORPORA.values()):
        contract = generation_contracts[corpus]
        if (
            not isinstance(contract, dict)
            or contract.get("schema")
            != "paper2-r1-question-generation-contract-v1"
            or contract.get("corpus") != corpus
        ):
            raise AssertionError(
                f"{corpus} question-generation contract is malformed"
            )
        projection = _generator_projection(
            contract.get("q4_deployment"), source=f"{corpus} question generator"
        )
        if projection != q4_evaluation[corpus]:
            raise AssertionError(
                f"{corpus} question generator differs from official Q4KXL evidence"
            )
        generator_projections[corpus] = projection
    if generator_projections["TSI"] != generator_projections["WC"]:
        raise AssertionError("the two corpus Q4 question generators disagree")


def promotion_record() -> dict:
    if not PROMOTION.is_file():
        raise AssertionError(
            "P2R1 promotion receipt is missing; run scripts/validate_r1.py after the rerun"
        )
    if (
        PROMOTION.is_symlink()
        or PROMOTION.parent.is_symlink()
        or PROMOTION != R1 / "private" / "promotion.json"
    ):
        raise AssertionError("P2R1 claims require the canonical private promotion path")
    value = json.loads(PROMOTION.read_text())
    protocol_path = PAPER_DIR / "R1_PROTOCOL.json"
    validate_promotion_header(
        value, protocol_sha256=p4.sha256_file(protocol_path)
    )
    if value.get("analysis_harness_sha256") != analysis_source_hashes(REPO_ROOT):
        raise AssertionError(
            "post-run analysis code changed after promotion; run a new explicit "
            "analysis audit/promotion before rebuilding claims"
        )
    by_label = {row["label"]: row for row in value.get("runs", [])}
    expected = {
        f"P2R1-{tier}-{corpus}"
        for tier in TIERS.values() for corpus in CORPORA.values()
    }
    if set(by_label) != expected:
        raise AssertionError("P2R1 promotion receipt does not contain the official six runs")
    attempt_id = value.get("attempt_id")
    protocol = json.loads(protocol_path.read_text())
    result_root = Path(value.get("result_root", "")).resolve()
    question_root = Path(value.get("question_root", "")).resolve()
    attempt_receipt = Path(value.get("campaign", "")) / "attempt.json"
    attempt_host_binding = result_root / "HOST_BINDING.json"
    if (
        not isinstance(attempt_id, str)
        or attempt_id != protocol.get("official_attempt_id")
        or result_root
        != (RESULTS / "paper2-r1" / attempt_id).resolve()
        or question_root
        != (R1 / "private" / "attempts" / attempt_id / "questions").resolve()
        or (result_root / "ATTEMPT_INVALIDATED.json").exists()
        or not attempt_receipt.is_file()
        or value.get("attempt_namespace_sha256")
        != p4.sha256_file(attempt_receipt)
        or not attempt_host_binding.is_file()
        or value.get("attempt_host_binding_sha256")
        != p4.sha256_file(attempt_host_binding)
    ):
        raise AssertionError("P2R1 promotion has an invalid attempt namespace")
    for label, row in by_label.items():
        outdir = Path(row.get("result_directory", "")).resolve()
        if (
            row.get("attempt_id") != attempt_id
            or outdir != (result_root / label).resolve()
        ):
            raise AssertionError(f"promoted run belongs to another attempt: {label}")
        checks = {
            "run_contract_sha256": outdir / "run_contract.json",
            "responses_sha256": outdir / "p2grid.jsonl",
            "request_ledger_sha256": outdir / "request_ledger.jsonl",
            "sessions_sha256": outdir / "run_sessions.jsonl",
            "summary_sha256": outdir / "summary.json",
        }
        for key, path in checks.items():
            if p4.sha256_file(path) != row[key]:
                raise AssertionError(f"promoted evidence changed after validation: {label}/{key}")
        failures_path = outdir / "failures.jsonl"
        expected_failures = row.get("failures_sha256")
        if expected_failures is None:
            if failures_path.exists():
                raise AssertionError(
                    f"unpromoted failure evidence appeared after validation: {label}"
                )
        elif not failures_path.is_file() or p4.sha256_file(failures_path) != expected_failures:
            raise AssertionError(f"promoted failure evidence changed: {label}")
        head_paths = {
            "responses": outdir / "p2grid.jsonl.head.json",
            "request_ledger": outdir / "request_ledger.jsonl.head.json",
            "sessions": outdir / "run_sessions.jsonl.head.json",
            "failures": outdir / "failures.jsonl.head.json",
        }
        promoted_heads = row.get("append_only_head_sha256")
        if not isinstance(promoted_heads, dict):
            raise AssertionError(f"promoted run lacks append-only heads: {label}")
        for key, path in head_paths.items():
            expected_hash = promoted_heads.get(key)
            if expected_hash is None and key == "failures" and not path.exists():
                continue
            if not path.is_file() or p4.sha256_file(path) != expected_hash:
                raise AssertionError(f"promoted chain head changed: {label}/{key}")
    supporting = value.get("supporting_evidence_sha256")
    if not isinstance(supporting, dict) or not supporting:
        raise AssertionError("P2R1 promotion lacks supporting-evidence retention hashes")
    for raw_path, expected_hash in supporting.items():
        path = Path(raw_path)
        if not path.is_file() or p4.sha256_file(path) != expected_hash:
            raise AssertionError(f"promoted supporting evidence changed: {path}")
    return value


def t_critical_975(df: float) -> float:
    """Deterministic two-sided 95% Student-t critical value by bisection."""
    if not math.isfinite(df) or df < 1:
        raise AssertionError("invalid Student-t reference degrees of freedom")
    lo, hi = 0.0, 2.0
    while student_t_two_sided(hi, df) > 0.05:
        hi *= 2.0
        if hi > 1e6:
            raise AssertionError("could not bracket Student-t critical value")
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if student_t_two_sided(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fmt_p(p: float) -> str:
    """LaTeX-ready p-value: two significant figures, scientific under 1e-3."""
    if p >= 0.001:
        return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction for the incomplete beta."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d, h = 1.0, 1.0 - qab * x / qap, 1.0
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise AssertionError("incomplete-beta continued fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0 or b <= 0:
        raise AssertionError("invalid regularized-beta arguments")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided(t_value: float, df: float) -> float:
    """Two-sided Student-t p-value without a runtime SciPy dependency."""
    if df < 1 or not math.isfinite(t_value):
        raise AssertionError("invalid Student-t test arguments")
    x = df / (df + t_value * t_value)
    return min(1.0, max(0.0, regularized_beta(x, df / 2.0, 0.5)))


def cr2_satterthwaite_df(cluster_sizes: list[int]) -> float:
    """Bell--McCaffrey CR2 df for an intercept-only iid working model."""
    if len(cluster_sizes) < 3 or any(type(n) is not int or n < 1 for n in cluster_sizes):
        raise AssertionError("CR2 requires at least three nonempty clusters")
    n = sum(cluster_sizes)
    leverages = [size / n for size in cluster_sizes]
    trace = sum(h / n for h in leverages)
    squared = 0.0
    for g, hg in enumerate(leverages):
        for h, hh in enumerate(leverages):
            if g == h:
                element = hg / n
            else:
                element = -hg * hh / (
                    n * math.sqrt((1.0 - hg) * (1.0 - hh))
                )
            squared += element * element
    return trace * trace / squared


def cr2_intercept_stats(values: list[float], clusters: list[str]) -> dict:
    """Intercept-only OLS estimate with Bell--McCaffrey CR2 inference.

    The iid working model supplies the CR2 adjustment. Source conversations
    are the independent-cluster approximation; dependence within a source
    conversation remains unrestricted.
    """
    if len(values) != len(clusters) or not values:
        raise AssertionError("CR2 values and clusters must be nonempty and aligned")
    if any(not math.isfinite(float(value)) for value in values):
        raise AssertionError("CR2 values must be finite")
    if any(not isinstance(cluster, str) or not cluster for cluster in clusters):
        raise AssertionError("CR2 clusters must be nonempty strings")

    n = len(values)
    estimate = sum(values) / n
    grouped_values: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        grouped_values.setdefault(cluster, []).append(float(value))
    if len(grouped_values) < 3:
        raise AssertionError("CR2 requires at least three source clusters")

    cluster_sizes = [len(group) for group in grouped_values.values()]
    cluster_scores = {
        cluster: sum(value - estimate for value in group)
        for cluster, group in grouped_values.items()
    }
    variance = sum(
        cluster_scores[cluster] ** 2
        / (1.0 - len(grouped_values[cluster]) / n)
        for cluster in grouped_values
    ) / n**2
    if variance < 0 or not math.isfinite(variance):
        raise AssertionError("CR2 variance is invalid")
    se = math.sqrt(variance)
    satt_df = cr2_satterthwaite_df(cluster_sizes)
    if satt_df < 5 - 1e-12:
        raise AssertionError(
            "CR2/Satterthwaite reference df below the predeclared minimum of 5"
        )
    if se == 0.0 and estimate != 0.0:
        raise AssertionError(
            "degenerate CR2 contrast: nonzero estimate with zero standard error"
        )
    t_statistic = estimate / se if se else 0.0
    tcrit = t_critical_975(satt_df)
    raw_cluster_sums = [sum(group) for group in grouped_values.values()]
    histogram = Counter(
        (len(group), sum(group)) for group in grouped_values.values()
    )
    return {
        "estimate": estimate,
        "cluster_se_cr2": se,
        "variance_cr2": variance,
        "t_critical_975": tcrit,
        "ci_cluster_95": [estimate - tcrit * se, estimate + tcrit * se],
        "t_cluster_cr2": t_statistic,
        "p_cluster_cr2_satterthwaite": student_t_two_sided(
            t_statistic, satt_df
        ),
        "cluster_reference_df_satterthwaite": satt_df,
        "n_items": n,
        "n_source_clusters": len(grouped_values),
        "cluster_size_counts": {
            str(size): count
            for size, count in sorted(Counter(cluster_sizes).items())
        },
        # Aggregate-only reconstruction record: no IDs, item order, or text.
        "cluster_n_items_raw_z_sum_histogram": [
            {
                "cluster_n_items": size,
                "raw_z_sum": raw_sum,
                "multiplicity": multiplicity,
            }
            for (size, raw_sum), multiplicity in sorted(histogram.items())
        ],
        "raw_cluster_sums_private_order": raw_cluster_sums,
    }


def cluster_sign_symmetry_sensitivity(cluster_sums: list[int]) -> float:
    """Exhaustive sign-flip p-value, exact only under cluster-sign symmetry."""
    values = [abs(v) for v in cluster_sums if v]
    if not values:
        return 1.0
    observed = abs(sum(cluster_sums))
    distribution = {0: 1}
    for value in values:
        updated: dict[int, int] = {}
        for total, count in distribution.items():
            updated[total + value] = updated.get(total + value, 0) + count
            updated[total - value] = updated.get(total - value, 0) + count
        distribution = updated
    extreme = sum(count for total, count in distribution.items()
                  if abs(total) >= observed)
    return extreme / (2 ** len(values))


def holm_family(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjustment with deterministic tie ordering."""
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    running_adjusted = 0.0
    still_rejecting = True
    result: dict[str, dict] = {}
    for index, (key, pvalue) in enumerate(ordered):
        remaining = m - index
        running_adjusted = max(
            running_adjusted, min(1.0, remaining * pvalue)
        )
        threshold = alpha / remaining
        reject = still_rejecting and pvalue <= threshold
        if not reject:
            still_rejecting = False
        result[key] = {
            "p_raw": pvalue,
            "p_holm": running_adjusted,
            "holm_threshold": threshold,
            "reject_fwer_005": reject,
        }
    return result


def interaction_result_branch(family: dict[str, dict]) -> str:
    """The frozen study-level branch; secondary sensitivities cannot set it."""
    return (
        "interaction-detected"
        if any(row["reject_fwer_005"] for row in family.values())
        else "no-interaction-detected"
    )


def paired_cluster_interaction(
        data: dict, qids: list[str], low_tier: str, condition: str) -> dict:
    """Marginal paired difference-in-differences, clustered by conversation.

    For item i, z_i = (O_low - C_low) - (O_Q4ref - C_Q4ref), where each
    correctness outcome is binary. Its mean is the difference in marginal
    compression cost between deployments. This avoids the changing denominator
    caused by conditioning separately on each tier's original-correct items.
    """
    z_by_qid: dict[str, int] = {}
    cluster_by_qid: dict[str, str] = {}
    for qid in qids:
        reference_o = int(data["Qf"][(qid, "O")]["exact"])
        reference_c = int(data["Qf"][(qid, condition)]["exact"])
        low_o = int(data[low_tier][(qid, "O")]["exact"])
        low_c = int(data[low_tier][(qid, condition)]["exact"])
        z_by_qid[qid] = (low_o - low_c) - (reference_o - reference_c)
        cluster_by_qid[qid] = data["Qf"][(qid, "O")]["source_cluster_id"]

    n = len(qids)
    cluster_sums: dict[str, int] = {}
    for qid in qids:
        cluster = cluster_by_qid[qid]
        cluster_sums[cluster] = cluster_sums.get(cluster, 0) + z_by_qid[qid]
    stats = cr2_intercept_stats(
        [z_by_qid[qid] for qid in qids],
        [cluster_by_qid[qid] for qid in qids],
    )
    raw_cluster_sums = stats.pop("raw_cluster_sums_private_order")
    estimate = stats.pop("estimate")
    return {
        "estimand": "(O_low-C_low)-(O_Q4_reference-C_Q4_reference)",
        "delta_low_minus_q4_reference": estimate,
        **stats,
        "cr2_working_model": "intercept-only iid working covariance",
        "p_sign_symmetry_sensitivity": cluster_sign_symmetry_sensitivity(
            raw_cluster_sums),
        "sign_symmetry_sensitivity_null": (
            "cluster contrast signs are exchangeable/symmetric under a sharp "
            "zero-effect null; reported only as a secondary sensitivity analysis"
        ),
        "n_nonzero_source_clusters": sum(v != 0 for v in cluster_sums.values()),
        "item_value_counts": {str(v): list(z_by_qid.values()).count(v)
                              for v in sorted(set(z_by_qid.values()))},
    }


def build() -> dict:
    promotion = promotion_record()
    result_dirs = {
        row["label"]: Path(row["result_directory"]).resolve()
        for row in promotion["runs"]
    }
    run_contracts = {
        label: json.loads((directory / "run_contract.json").read_text())
        for label, directory in result_dirs.items()
    }
    windows: dict[str, Path] = {}
    questions: dict[str, Path] = {}
    question_manifests: dict[str, Path] = {}
    question_attempts: dict[str, Path] = {}
    question_generation_contracts: dict[str, Path] = {}
    for corpus_tag, corpus in CORPORA.items():
        contracts = [
            run_contracts[f"P2R1-{tier}-{corpus}"] for tier in TIERS.values()
        ]
        for identity_key, destination in (
            ("windows", windows), ("questions", questions),
            ("question_manifest", question_manifests),
            ("question_generation_attempts", question_attempts),
            ("question_generation_contract", question_generation_contracts),
        ):
            identities = [contract.get(identity_key) for contract in contracts]
            if any(identity != identities[0] for identity in identities[1:]):
                raise AssertionError(
                    f"deployed tiers used different {corpus} {identity_key} evidence"
                )
            identity = identities[0]
            path = Path(identity["path"])
            if p4.sha256_file(path) != identity["sha256"]:
                raise AssertionError(f"promoted {corpus} {identity_key} changed")
            destination[corpus_tag] = path
    validate_deployment_bindings(
        promotion,
        run_contracts,
        {
            corpus: json.loads(question_generation_contracts[tag].read_text())
            for tag, corpus in CORPORA.items()
        },
    )
    claims: dict = {
        "schema": "paper2-r1-claims-v3",
        "study_status": (
            "corrective exploratory rerun after hypotheses, corpus frames, and "
            "contaminated outputs were observed; not independent confirmation"
        ),
        "promotion_sha256": p4.sha256_file(PROMOTION),
        "attempt_id": promotion["attempt_id"],
        "promotion_completed_at": promotion["completed_at"],
        "analysis_harness_sha256": promotion["analysis_harness_sha256"],
        "analysis_implementation": "paper4/scripts/verify_claims.py",
        "analysis_implementation_sha256": p4.sha256_file(ANALYSIS_IMPLEMENTATION),
    }

    # ---- artifacts -------------------------------------------------------
    promoted_by_tier: dict[str, dict] = {}
    for run in promotion["runs"]:
        tier = run["label"].split("-")[1]
        evidence = run.get("artifact_evidence")
        if not isinstance(evidence, dict) or evidence.get("tier") != tier:
            raise AssertionError(f"promotion lacks artifact evidence for {run['label']}")
        if tier in promoted_by_tier and promoted_by_tier[tier] != evidence:
            raise AssertionError(f"promoted corpora disagree on {tier} artifact")
        promoted_by_tier[tier] = evidence
    if set(promoted_by_tier) != set(TIERS.values()):
        raise AssertionError("promotion does not bind all three deployed artifacts")

    verified_artifacts: dict[str, tuple[Path, dict, dict]] = {}
    for tier, evidence in promoted_by_tier.items():
        record_path = Path(evidence["artifact_record"])
        if p4.sha256_file(record_path) != evidence["artifact_record_sha256"]:
            raise AssertionError(f"promoted artifact record changed for {tier}")
        record = load_artifact_record(record_path, full_hash=True)
        if record.get("artifact") != evidence["artifact"]:
            raise AssertionError(f"promoted artifact identity changed for {tier}")
        verified_artifacts[tier] = (
            Path(record["artifact"]["path"]), evidence, record,
        )

    artifacts = {}
    for tag in ("Qf", "Iq"):
        tier = TIERS[tag]
        path, evidence, source_record = verified_artifacts[tier]
        header = p4.read_gguf_header(path)
        nbytes = path.stat().st_size
        params = header["stored_parameters"]
        artifacts[tag] = {
            "file": path.name,
            "file_bytes": nbytes,
            "stored_parameters_from_gguf_tensor_shapes": params,
            "gguf_tensor_count": header["tensor_count"],
            "effective_file_bpw": nbytes * 8 / params,
            "artifact_sha256": evidence["artifact"]["sha256"],
            "artifact_record_sha256": evidence["artifact_record_sha256"],
            "upstream_repo": source_record["source_provenance"]["upstream_repo"],
            "upstream_revision": source_record["source_provenance"]["revision"],
            "upstream_filename": source_record["source_provenance"]["source_entries"][0]["filename"],
            "upstream_license": source_record["source_provenance"]["license"],
            "source_manifest_sha256": source_record["source_provenance"]["manifest_sha256"],
        }
    EXL3_DIR, ex_evidence, ex_record = verified_artifacts["EXL3"]
    st_bytes = sum(f.stat().st_size for f in EXL3_DIR.glob("model-*.safetensors"))
    artifacts["Ex"] = {
        "file": EXL3_DIR.name,
        "weight_safetensors_bytes": st_bytes,
        "vendor_label": "EXL3 SC_2.00bpw_H3",
        "artifact_sha256": ex_evidence["artifact"]["sha256"],
        "artifact_record_sha256": ex_evidence["artifact_record_sha256"],
        "upstream_repo": ex_record["source_provenance"]["upstream_repo"],
        "upstream_revision": ex_record["source_provenance"]["revision"],
        "upstream_weight_shards": [
            {key: row[key] for key in ("filename", "bytes", "sha256")}
            for row in ex_record["source_provenance"]["source_entries"]
        ],
        "upstream_license": ex_record["source_provenance"]["license"],
        "source_manifest_sha256": ex_record["source_provenance"]["manifest_sha256"],
        "source_coverage": ex_record["source_provenance"]["coverage"],
        "note": ("effective bpw not derived: EXL3 shard headers are not parsed "
                 "here and mixed head/embedding precision makes a file-level "
                 "ratio misleading; the vendor bulk label is reported as such"),
    }
    claims["artifacts"] = artifacts

    # ---- corpora, retention, verifier ------------------------------------
    corpora = {}
    for ctag, wpath in windows.items():
        wins = read_jsonl(wpath)
        qmanifest = json.loads(question_manifests[ctag].read_text())
        try:
            corpora[ctag] = build_corpus_claim(
                wins,
                qmanifest,
                read_jsonl(question_attempts[ctag]),
                read_jsonl(questions[ctag]),
            )
        except AssertionError as exc:
            raise AssertionError(f"{ctag}: {exc}") from exc
    claims["corpora"] = corpora

    # ---- grid ------------------------------------------------------------
    grid: dict = {}
    interaction: dict = {}
    survival: dict = {}
    for ctag in CORPORA:
        data = {t: load_cell_rows(t, ctag, result_dirs) for t in TIERS}
        qids = None
        for t in TIERS:
            for cond in CONDS.values():
                have = {q for (q, cc) in data[t] if cc == cond}
                qids = have if qids is None else (qids & have)
        qids = sorted(qids)
        n = len(qids)
        qf_o_rows = {q: data["Qf"][(q, "O")] for q in qids}
        item_strata: dict[str, int] = {}
        for row in qf_o_rows.values():
            key = f"{row['stratum']}-{'code' if row['has_code'] else 'nocode'}"
            item_strata[key] = item_strata.get(key, 0) + 1
        grid[ctag] = {
            "n_items": n,
            "n_source_windows": len({r["window_id"] for r in qf_o_rows.values()}),
            "n_source_clusters": len({r["source_cluster_id"] for r in qf_o_rows.values()}),
            "item_strata_counts": item_strata,
            "cells": {},
        }
        for t in TIERS:
            O = {q: data[t][(q, "O")] for q in qids}
            kO = sum(r["exact"] for r in O.values())
            for dtag, cond in CONDS.items():
                rows = {q: data[t][(q, cond)] for q in qids}
                k = sum(r["exact"] for r in rows.values())
                klen = sum(r["lenient"] for r in rows.values())
                wall = sum(r["wall_s"] for r in rows.values()) / n
                request_outcomes = Counter(
                    row.get("request_outcome") for row in rows.values()
                )
                allowed_outcomes = {
                    "success", "request-error-identity-preserved"
                }
                if set(request_outcomes) - allowed_outcomes:
                    raise AssertionError(
                        f"{ctag}/{t}/{dtag} has an ineligible request outcome"
                    )
                error_rows = [
                    row for row in rows.values()
                    if row["request_outcome"]
                    == "request-error-identity-preserved"
                ]
                if any(row["exact"] or row["lenient"] for row in error_rows):
                    raise AssertionError(
                        f"{ctag}/{t}/{dtag} request-error row was not scored incorrect"
                    )
                error_types = Counter(row.get("request_error_type") for row in error_rows)
                if None in error_types or any(
                    not isinstance(key, str) for key in error_types
                ):
                    raise AssertionError(
                        f"{ctag}/{t}/{dtag} request-error row lacks an error type"
                    )
                lo, hi = p4.wilson_interval(k, n)
                cell = {
                    "k_exact": k, "k_lenient": klen, "n": n,
                    "acc": k / n, "acc_lenient": klen / n,
                    "wilson": [lo, hi],
                    "mean_wall_s": wall,
                    "acc_per_min": (k / n) / (wall / 60.0),
                    "request_outcomes": {
                        "total": n,
                        "success": request_outcomes.get("success", 0),
                        "identity_preserved_request_errors": len(error_rows),
                        "request_error_type_counts": dict(sorted(error_types.items())),
                    },
                }
                if cond != "O":
                    b = sum(1 for q in qids
                            if O[q]["exact"] and not rows[q]["exact"])
                    c = sum(1 for q in qids
                            if not O[q]["exact"] and rows[q]["exact"])
                    n11 = kO - b
                    n22 = n - (n11 + b + c)
                    diff, dlo, dhi, phi = \
                        p4.newcombe_paired_risk_difference_interval(n11, c, b, n22)
                    cell["paired_vs_O"] = {
                        "O_only": b, "cond_only": c,
                        "delta": diff, "ci": [dlo, dhi], "phi": phi,
                        "p_mcnemar": p4.mcnemar_exact_two_sided(b, c),
                    }
                grid[ctag]["cells"][f"{t}-{dtag}"] = cell

        # Cross-tier interaction. The audited primary reanalysis is the
        # marginal paired difference-in-differences, clustered by opaque source
        # conversation. The originally reported hurt-flip Fisher comparison is kept
        # only as a historical sensitivity analysis: it treats shared items as
        # independent across tiers and conditions on tier-specific O-correct
        # denominators, so it cannot carry the paper's main inference.
        inter = {}
        OQ = {q: data["Qf"][(q, "O")]["exact"] for q in qids}
        nOQ = sum(OQ.values())
        for dtag, cond in CONDS.items():
            if cond == "O":
                continue
            XQ = {q: data["Qf"][(q, cond)]["exact"] for q in qids}
            bQ = sum(1 for q in qids if OQ[q] and not XQ[q])
            for t in ("Ex", "Iq"):
                Ot = {q: data[t][(q, "O")]["exact"] for q in qids}
                Xt = {q: data[t][(q, cond)]["exact"] for q in qids}
                bt = sum(1 for q in qids if Ot[q] and not Xt[q])
                nOt = sum(Ot.values())
                entry = paired_cluster_interaction(data, qids, t, cond)
                entry["historical_hurt_flip_sensitivity"] = {
                    "hurt_low": [bt, nOt], "hurt_q4_reference": [bQ, nOQ],
                    "p_fisher": p4.fisher_exact_two_sided(bt, nOt, bQ, nOQ),
                }
                inter[f"{dtag}-{t}"] = entry
        interaction[ctag] = inter

        # Frozen post-failure answer-survival classifier for E and L2. It is
        # descriptive, was not independently preregistered, and preserves
        # numeric answers instead of dropping them during alpha-only cleanup.
        wins = {w["window_id"]: w for w in read_jsonl(windows[ctag])}
        surv_c: dict = {}
        for dtag, cond in (("E", "E"), ("L", "L2")):
            key = f"text_{cond}"
            surv = {}
            for q in qids:
                r = data["Qf"][(q, "O")]
                g = survival_form(r["gold"])
                wt = wins[r["window_id"]].get(key, "")
                surv[q] = bool(g) and g in survival_form(wt)
            ns = sum(surv.values())
            entry = {"gold_survives": ns, "n": n, "tiers": {}}
            for t in TIERS:
                ks = sum(data[t][(q, cond)]["exact"] for q in qids if surv[q])
                kd = sum(data[t][(q, cond)]["exact"] for q in qids if not surv[q])
                entry["tiers"][t] = {
                    "surv_correct": ks, "surv_n": ns,
                    "dest_correct": kd, "dest_n": n - ns,
                }
            surv_c[dtag] = entry
        survival[ctag] = surv_c

    claims["grid"] = grid
    claims["interaction"] = interaction
    interaction_hypotheses = {
        f"{corpus}:{key}": value["p_cluster_cr2_satterthwaite"]
        for corpus, comparisons in interaction.items()
        for key, value in comparisons.items()
    }
    if len(interaction_hypotheses) != 12:
        raise AssertionError("frozen interaction family must contain exactly 12 tests")
    holm = holm_family(interaction_hypotheses)
    for hypothesis, family_result in holm.items():
        corpus, key = hypothesis.split(":", 1)
        interaction[corpus][key]["familywise"] = family_result
    claims["survival"] = survival

    # ---- headline ranges -------------------------------------------------
    def deltas(dtags):
        return [grid[c]["cells"][f"{t}-{d}"]["paired_vs_O"]["delta"]
                for c in CORPORA for t in TIERS for d in dtags]

    naive, l2 = deltas(("A", "E")), deltas(("L",))
    inter_entries = [v for c in interaction.values() for v in c.values()]
    inter_ps = [v["p_cluster_cr2_satterthwaite"] for v in inter_entries]
    sign_sensitivity_ps = [
        v["p_sign_symmetry_sensitivity"] for v in inter_entries
    ]
    inter_deltas = [v["delta_low_minus_q4_reference"] for v in inter_entries]
    inter_ci_lows = [v["ci_cluster_95"][0] for v in inter_entries]
    inter_ci_highs = [v["ci_cluster_95"][1] for v in inter_entries]
    n_pointwise_ci_detected = sum(
        lo > 0 or hi < 0 for lo, hi in zip(inter_ci_lows, inter_ci_highs)
    )
    n_cluster_t_raw_detected = sum(p < 0.05 for p in inter_ps)
    n_signflip_raw_detected = sum(p < 0.05 for p in sign_sensitivity_ps)
    n_holm_detected = sum(
        value["reject_fwer_005"] for value in holm.values()
    )

    def acc_range(dtags):
        vals = [grid[c]["cells"][f"{t}-{d}"]["acc"]
                for c in CORPORA for t in TIERS for d in dtags]
        return min(vals), max(vals)

    def apm_range(dtags):
        vals = [grid[c]["cells"][f"{t}-{d}"]["acc_per_min"]
                for c in CORPORA for t in TIERS for d in dtags]
        return min(vals), max(vals)

    def wall_range(dtags):
        vals = [grid[c]["cells"][f"{t}-{d}"]["mean_wall_s"]
                for c in CORPORA for t in TIERS for d in dtags]
        return min(vals), max(vals)

    # Largest descriptive across-tier spread of the paired delta within one
    # (corpus, condition). This is not an equivalence or independence test.
    tier_spread = max(
        max(grid[c]["cells"][f"{t}-{d}"]["paired_vs_O"]["delta"] for t in TIERS)
        - min(grid[c]["cells"][f"{t}-{d}"]["paired_vs_O"]["delta"] for t in TIERS)
        for c in CORPORA for d in ("A", "L", "E"))

    surv_acc = {
        d: [
            tv["surv_correct"] / tv["surv_n"]
            for c in CORPORA for tv in survival[c][d]["tiers"].values()
            if tv["surv_n"] > 0
        ]
        for d in ("L", "E")
    }
    dest_trials = sum(tv["dest_n"] for c in CORPORA for d in ("L", "E")
                      for tv in survival[c][d]["tiers"].values())
    dest_item_condition_pairs = sum(
        survival[c][d]["n"] - survival[c][d]["gold_survives"]
        for c in CORPORA for d in ("L", "E"))
    dest_hits = sum(tv["dest_correct"] for c in CORPORA for d in ("L", "E")
                    for tv in survival[c][d]["tiers"].values())
    interval_endpoint_pp = max(
        abs(min(inter_ci_lows)), abs(max(inter_ci_highs))
    ) * 100
    result_branch = interaction_result_branch(holm)
    if result_branch == "interaction-detected":
        interaction_inference = (
            f"Holm's procedure rejects {n_holm_detected} of {len(inter_entries)} "
            "model-assisted CR2/Satterthwaite source-cluster hypotheses at "
            "familywise alpha 0.05; "
            "this corrective exploratory rerun therefore detects at least one "
            "deployed-tier by compression interaction in the tested family. "
            f"Separately, {n_pointwise_ci_detected} pointwise clustered 95% "
            "intervals exclude zero; those intervals are descriptive and do not "
            "set the family-level result branch. No equivalence margin "
            "or equivalence test was pre-registered, so no equivalence claim is "
            "supported."
        )
    else:
        result_branch = "no-interaction-detected"
        interaction_inference = (
            f"Holm's procedure rejects none of {len(inter_entries)} source-"
            "cluster-robust CR2/Satterthwaite hypotheses at familywise alpha "
            "0.05, so this "
            "corrective exploratory rerun detects no deployed-tier by compression "
            "interaction in the tested family. The pointwise clustered 95% "
            f"intervals are descriptive ({n_pointwise_ci_detected} exclude zero); "
            "the largest absolute pointwise "
            f"interval endpoint is {interval_endpoint_pp:.1f} percentage points; "
            "it is not a pre-specified resolution or equivalence bound. "
            f"The unadjusted CR2/Satterthwaite tests have {n_cluster_t_raw_detected} "
            f"of {len(inter_entries)} p-values below 0.05. The conditional "
            f"cluster-sign-symmetry sensitivity calculation has "
            f"{n_signflip_raw_detected} below 0.05. No equivalence margin "
            "or equivalence test was pre-registered, so this does not establish "
            "independence."
        )
    all_compressed_apm = [
        grid[c]["cells"][f"{t}-{d}"]["acc_per_min"]
        for c in CORPORA for t in TIERS for d in ("A", "L", "E")
    ]
    evaluation_error_types: Counter[str] = Counter()
    evaluation_request_errors = 0
    for corpus_grid in grid.values():
        for cell in corpus_grid["cells"].values():
            outcomes = cell["request_outcomes"]
            if outcomes["success"] + outcomes[
                "identity_preserved_request_errors"
            ] != outcomes["total"]:
                raise AssertionError("evaluation request-outcome aggregate is incomplete")
            evaluation_request_errors += outcomes[
                "identity_preserved_request_errors"
            ]
            evaluation_error_types.update(outcomes["request_error_type_counts"])
    generator_request_errors = sum(
        corpus["question_generation"]["identity_preserved_request_errors"]
        for corpus in corpora.values()
    )
    generator_error_types: Counter[str] = Counter()
    for corpus in corpora.values():
        generator_error_types.update(
            corpus["question_generation"]["request_error_type_counts"]
        )
    qualitative_assertions = {
        "interaction_primary_branch": result_branch,
        "interaction_holm_rejections_fwer_005": n_holm_detected,
        "interaction_pointwise_ci_excluding_zero": n_pointwise_ci_detected,
        "interaction_unadjusted_signflip_p_below_005": n_signflip_raw_detected,
        "interaction_unadjusted_cluster_t_p_below_005": n_cluster_t_raw_detected,
        "llmlingua_cost_below_every_naive_cost": max(l2) < min(naive),
        "original_accuracy_per_minute_above_every_compressed_condition": (
            min(apm_range(("O",))) > max(all_compressed_apm)
        ),
        "destroyed_answer_recovery_observed": dest_hits > 0,
        "evaluation_request_errors_observed": evaluation_request_errors > 0,
        "question_generation_request_errors_observed": generator_request_errors > 0,
    }
    claims["headline"] = {
        "naive_delta_pp_min": min(naive) * 100, "naive_delta_pp_max": max(naive) * 100,
        "l2_delta_pp_min": min(l2) * 100, "l2_delta_pp_max": max(l2) * 100,
        "interaction_p_min": min(inter_ps), "interaction_p_max": max(inter_ps),
        "interaction_delta_pp_range": [min(inter_deltas) * 100,
                                        max(inter_deltas) * 100],
        "interaction_ci_pp_envelope": [min(inter_ci_lows) * 100,
                                        max(inter_ci_highs) * 100],
        "interaction_max_abs_ci_endpoint_pp": interval_endpoint_pp,
        "interaction_holm_rejections_fwer_005": n_holm_detected,
        "interaction_pointwise_ci_excluding_zero": n_pointwise_ci_detected,
        "interaction_unadjusted_signflip_p_below_005": n_signflip_raw_detected,
        "interaction_unadjusted_cluster_t_p_below_005": n_cluster_t_raw_detected,
        "interaction_familywise_method": (
            "Holm step-down over 12 model-assisted CR2 source-conversation-"
            "cluster-robust two-sided t tests with Satterthwaite degrees of freedom"
        ),
        "interaction_signflip_role": (
            "secondary exhaustive sensitivity analysis, exact only conditional "
            "on cluster-sign symmetry"
        ),
        "interaction_result_branch": result_branch,
        "interaction_inference": interaction_inference,
        "qualitative_assertions": qualitative_assertions,
        "n_interaction_tests": len(inter_ps),
        "n_cells": sum(len(g["cells"]) for g in grid.values()),
        "n_scored_requests": sum(
            cell["n"] for g in grid.values() for cell in g["cells"].values()),
        "evaluation_request_errors": evaluation_request_errors,
        "evaluation_request_error_type_counts": dict(
            sorted(evaluation_error_types.items())
        ),
        "question_generation_request_errors": generator_request_errors,
        "question_generation_request_error_type_counts": dict(
            sorted(generator_error_types.items())
        ),
        "o_acc_range": acc_range(("O",)), "l_acc_range": acc_range(("L",)),
        "apm_o_range": apm_range(("O",)), "apm_l_range": apm_range(("L",)),
        "apm_naive_range": apm_range(("A", "E")),
        "wall_o_range": wall_range(("O",)),
        "wall_naive_range": wall_range(("A", "E")),
        "tier_delta_spread_pp_max": tier_spread * 100,
        "surv_acc_l_range": (
            [min(surv_acc["L"]), max(surv_acc["L"])] if surv_acc["L"] else None
        ),
        "surv_acc_e_range": (
            [min(surv_acc["E"]), max(surv_acc["E"])] if surv_acc["E"] else None
        ),
        "destroyed_trials": dest_trials,
        "destroyed_item_condition_pairs": dest_item_condition_pairs,
        "destroyed_recovered": dest_hits,
    }

    # ---- protocol constants (from the runner sources) --------------------
    claims["protocol"] = {
        "seed": 42, "temperature": 0.2, "top_p": 0.95, "max_tokens": 2048,
        "context_tokens": 8192, "window_char_cap": 14000,
        "runner": "testsuite/paper2/run_grid.py",
        "question_generator": "testsuite/paper2/gen_questions.py",
        "conditions_builder": "testsuite/paper2/make_conditions_r1.py",
        "protocol": "paper2/R1_PROTOCOL.json",
        "study_status": "corrective exploratory rerun; not independent confirmation",
    }
    host = promotion.get("host")
    required_host = {
        "cpu_model", "logical_cpus", "mem_total_bytes", "gpu_name",
        "gpu_memory_total_mib", "gpu_current_power_limit_w",
        "gpu_requested_power_limit_w", "gpu_default_power_limit_w",
        "nvidia_driver",
    }
    if not isinstance(host, dict) or set(host) != required_host:
        raise AssertionError("promotion receipt lacks a sanitized bound host inventory")
    claims["hardware"] = {
        **host,
        "ram_gib": host["mem_total_bytes"] / 2**30,
        "gpu_vram_gib": host["gpu_memory_total_mib"] / 1024,
        "cpu_threads_flag": 16,
        "source": "validated campaign host receipt bound by promotion SHA-256",
    }
    toolchain_path = Path(promotion["campaign"]) / "toolchain.json"
    toolchain = json.loads(toolchain_path.read_text())
    if p4.sha256_file(toolchain_path) != promotion.get("campaign_toolchain_sha256"):
        raise AssertionError("promoted toolchain receipt changed")
    installed = sanitize_exl3_python_runtime_content(
        toolchain.get("exl3_runtime", {}).get("installed_content", {})
    )
    build_identity = toolchain.get("llama_cpp", {}).get("build_identity", {})
    private_cuda_runtime = toolchain.get("llama_cpp", {}).get("cuda_runtime", {})
    cuda_runtime = promotion.get("llama_cuda_runtime", {})
    canonical_cuda_runtime = require_canonical_public_cuda_runtime(
        private_cuda_runtime, cuda_runtime,
    )
    if (
        installed.get("schema") != "paper2-r1-python-runtime-content-v1"
        or build_identity.get("schema") != "paper2-r1-llama-build-identity-v1"
    ):
        raise AssertionError("promoted engine build/runtime identity is incomplete")
    claims["software"] = {
        "llama_cpp_revision": toolchain["llama_cpp"]["revision"],
        "llama_server_sha256": toolchain["llama_cpp"]["server_sha256"],
        "llama_build": build_identity,
        "llama_cuda_runtime": canonical_cuda_runtime,
        "tabbyapi_revision": toolchain["tabbyapi"]["revision"],
        "python_runtime": installed,
        "nvidia_driver": host["nvidia_driver"],
        "privacy": "absolute executable/site-package paths omitted",
    }
    return claims


def emit_tex(claims: dict) -> str:
    lines = [
        "% Generated by paper2/scripts/build_claims.py. DO NOT EDIT.",
        "% Every macro is derived from the frozen grid JSONLs, the window",
        "% files' character counts, GGUF headers, or run-log tallies.",
    ]

    seen: set[str] = set()

    def cmd(name: str, value) -> None:
        if name in seen:
            raise AssertionError(f"duplicate generated LaTeX macro: \\Ptwo{name}")
        seen.add(name)
        lines.append(rf"\newcommand{{\Ptwo{name}}}{{{value}}}")

    a = claims["artifacts"]
    cmd("QfGiB", f"{a['Qf']['file_bytes'] / 2**30:.1f}")
    cmd("QfBpw", f"{a['Qf']['effective_file_bpw']:.2f}")
    cmd("IqGiB", f"{a['Iq']['file_bytes'] / 2**30:.1f}")
    cmd("IqBpw", f"{a['Iq']['effective_file_bpw']:.2f}")
    cmd("ExGiB", f"{a['Ex']['weight_safetensors_bytes'] / 2**30:.1f}")
    cmd("ParamsB", f"{a['Qf']['stored_parameters_from_gguf_tensor_shapes'] / 1e9:.1f}")

    for ctag, c in claims["corpora"].items():
        cmd(f"{ctag}Windows", c["windows"])
        cmd(f"{ctag}QKept", c["questions_kept"])
        cmd(
            f"{ctag}RejPct",
            f"{c['candidate_pair_rejection_rate_after_successful_requests'] * 100:.0f}",
        )
        for dtag, r in c["char_retention_vs_O"].items():
            cmd(f"Ret{dtag}{ctag}", f"{r * 100:.0f}")

    for ctag, g in claims["grid"].items():
        cmd(f"{ctag}Items", g["n_items"])
        cmd(f"{ctag}SourceWindows", g["n_source_windows"])
        cmd(f"{ctag}SourceClusters", g["n_source_clusters"])
        for stratum in ("S-nocode", "S-code", "L-nocode", "L-code"):
            suffix = stratum.replace("-", "").replace("nocode", "NoCode").replace("code", "Code")
            cmd(f"{ctag}{suffix}Items", g["item_strata_counts"].get(stratum, 0))
        for cell_key, cell in g["cells"].items():
            t, d = cell_key.split("-")
            base = f"{t}{ctag}{d}"
            cmd(f"Acc{base}", f"{cell['acc'] * 100:.0f}")
            cmd(f"AccLen{base}", f"{cell['acc_lenient'] * 100:.0f}")
            cmd(f"Lo{base}", f"{cell['wilson'][0] * 100:.0f}")
            cmd(f"Hi{base}", f"{cell['wilson'][1] * 100:.0f}")
            cmd(f"Wall{base}", f"{cell['mean_wall_s']:.1f}")
            cmd(f"Apm{base}", f"{cell['acc_per_min']:.1f}")
            if "paired_vs_O" in cell:
                pv = cell["paired_vs_O"]
                cmd(f"Del{base}", f"{pv['delta'] * 100:.0f}")
                cmd(f"Dlo{base}", f"{pv['ci'][0] * 100:.0f}")
                cmd(f"Dhi{base}", f"{pv['ci'][1] * 100:.0f}")
                cmd(f"P{base}", fmt_p(pv["p_mcnemar"]))

    for ctag, inter in claims["interaction"].items():
        for key, v in inter.items():
            d, t = key.split("-")
            cmd(f"Int{ctag}{d}{t}",
                f"{v['delta_low_minus_q4_reference'] * 100:.1f}")
            cmd(f"IntLo{ctag}{d}{t}",
                f"{v['ci_cluster_95'][0] * 100:.1f}")
            cmd(f"IntHi{ctag}{d}{t}",
                f"{v['ci_cluster_95'][1] * 100:.1f}")
            cmd(f"IntP{ctag}{d}{t}",
                f"{v['p_cluster_cr2_satterthwaite']:.3f}")
            cmd(f"IntPSignSym{ctag}{d}{t}",
                f"{v['p_sign_symmetry_sensitivity']:.3f}")
            cmd(f"IntPHolm{ctag}{d}{t}", f"{v['familywise']['p_holm']:.3f}")
            cmd(f"IntHolmReject{ctag}{d}{t}",
                int(v["familywise"]["reject_fwer_005"]))

    for ctag, sc in claims["survival"].items():
        for dtag, entry in sc.items():
            cmd(f"Surv{ctag}{dtag}N", entry["gold_survives"])
            cmd(f"Surv{ctag}{dtag}Tot", entry["n"])
            cmd(f"Surv{ctag}{dtag}Pct",
                f"{entry['gold_survives'] / entry['n'] * 100:.0f}")
            for t, tv in entry["tiers"].items():
                cmd(f"Surv{ctag}{dtag}{t}K", tv["surv_correct"])
                cmd(f"Dest{ctag}{dtag}{t}K", tv["dest_correct"])
            cmd(f"Dest{ctag}{dtag}N", entry["n"] - entry["gold_survives"])

    h = claims["headline"]
    cmd("NaiveDelMin", f"{h['naive_delta_pp_min']:.0f}")
    cmd("NaiveDelMax", f"{h['naive_delta_pp_max']:.0f}")
    cmd("LDelMin", f"{h['l2_delta_pp_min']:.0f}")
    cmd("LDelMax", f"{h['l2_delta_pp_max']:.0f}")
    cmd("InterPMin", f"{h['interaction_p_min']:.2f}")
    cmd("InterPMax", f"{h['interaction_p_max']:.2f}")
    cmd("InterDeltaMin", f"{h['interaction_delta_pp_range'][0]:.1f}")
    cmd("InterDeltaMax", f"{h['interaction_delta_pp_range'][1]:.1f}")
    cmd("InterCiLowMin", f"{h['interaction_ci_pp_envelope'][0]:.1f}")
    cmd("InterCiHighMax", f"{h['interaction_ci_pp_envelope'][1]:.1f}")
    cmd("InterResolution", f"{h['interaction_max_abs_ci_endpoint_pp']:.1f}")
    cmd("InterHolmDetected", h["interaction_holm_rejections_fwer_005"])
    cmd("InterPointwiseCiDetected", h["interaction_pointwise_ci_excluding_zero"])
    cmd("InterRawSignflipDetected",
        h["interaction_unadjusted_signflip_p_below_005"])
    cmd("InterRawClusterTDetected",
        h["interaction_unadjusted_cluster_t_p_below_005"])
    cmd("InterTests", h["n_interaction_tests"])
    cmd("NCells", h["n_cells"])
    cmd("NRequests", h["n_scored_requests"])
    cmd("EvalRequestErrors", h["evaluation_request_errors"])
    cmd("GenRequestErrors", h["question_generation_request_errors"])
    cmd("OAccMin", f"{h['o_acc_range'][0] * 100:.0f}")
    cmd("OAccMax", f"{h['o_acc_range'][1] * 100:.0f}")
    cmd("LAccMin", f"{h['l_acc_range'][0] * 100:.0f}")
    cmd("LAccMax", f"{h['l_acc_range'][1] * 100:.0f}")
    cmd("ApmOMin", f"{h['apm_o_range'][0]:.1f}")
    cmd("ApmOMax", f"{h['apm_o_range'][1]:.1f}")
    cmd("ApmLMin", f"{h['apm_l_range'][0]:.1f}")
    cmd("ApmLMax", f"{h['apm_l_range'][1]:.1f}")
    cmd("ApmNaiveMin", f"{h['apm_naive_range'][0]:.1f}")
    cmd("ApmNaiveMax", f"{h['apm_naive_range'][1]:.1f}")
    cmd("WallOMin", f"{h['wall_o_range'][0]:.1f}")
    cmd("WallOMax", f"{h['wall_o_range'][1]:.1f}")
    cmd("WallNaiveMin", f"{h['wall_naive_range'][0]:.1f}")
    cmd("WallNaiveMax", f"{h['wall_naive_range'][1]:.1f}")
    cmd("TierSpreadMax", f"{h['tier_delta_spread_pp_max']:.0f}")
    for tag, values in (
        ("SurvAccL", h["surv_acc_l_range"]),
        ("SurvAccE", h["surv_acc_e_range"]),
    ):
        cmd(f"{tag}Min", "N/A" if values is None else f"{values[0] * 100:.0f}")
        cmd(f"{tag}Max", "N/A" if values is None else f"{values[1] * 100:.0f}")
    cmd("DestTrials", h["destroyed_trials"])
    cmd("DestPairs", h["destroyed_item_condition_pairs"])
    cmd("DestRecovered", h["destroyed_recovered"])
    return "\n".join(lines) + "\n"


def synthetic_self_test() -> None:
    """Exercise the owner-only pair transaction without campaign evidence."""
    with tempfile.TemporaryDirectory(prefix="paper2-claims-selftest-") as raw:
        directory = Path(raw)
        first = directory / "claims.json"
        second = directory / "claims.tex"
        expected = {first: b'{"schema":"synthetic"}\n', second: b"% synthetic\n"}
        atomic_write_claim_outputs(expected)
        for path, content in expected.items():
            if (
                _read_safe_output(path) != content
                or stat.S_IMODE(path.lstat().st_mode) != CLAIMS_OUTPUT_MODE
            ):
                raise AssertionError("claims output self-test failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="rebuild and diff against the frozen derived outputs")
    mode.add_argument("--self-test", action="store_true",
                      help="run lightweight output-safety checks without evidence")
    args = ap.parse_args()

    if args.self_test:
        synthetic_self_test()
        print("Paper 2 claims builder output-safety self-test passed")
        return 0

    claims = build()
    js = json.dumps(claims, indent=2, sort_keys=True) + "\n"
    tex = emit_tex(claims)

    if args.check:
        ok = True
        for path, fresh in ((OUT_JSON, js), (OUT_TEX, tex)):
            if _read_safe_output(path, allow_missing=True) != fresh.encode("utf-8"):
                print(f"STALE: {path} does not match a fresh rebuild")
                ok = False
        if ok:
            print("claims check OK: derived outputs match a fresh rebuild")
        return 0 if ok else 1

    atomic_write_claim_outputs({
        OUT_JSON: js.encode("utf-8"),
        OUT_TEX: tex.encode("utf-8"),
    })
    print(f"wrote {OUT_JSON} and {OUT_TEX}")
    h = claims["headline"]
    print(f"cells={h['n_cells']} paired-cluster interaction p in "
          f"[{h['interaction_p_min']:.2f},{h['interaction_p_max']:.2f}] "
          f"interaction {h['interaction_delta_pp_range'][0]:.1f} to "
          f"{h['interaction_delta_pp_range'][1]:.1f}pp "
          f"naiveΔ {h['naive_delta_pp_min']:.0f}-{h['naive_delta_pp_max']:.0f}pp "
          f"L2Δ {h['l2_delta_pp_min']:.0f}-{h['l2_delta_pp_max']:.0f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
