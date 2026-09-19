#!/usr/bin/env python3
"""Fail-closed post-run promotion validator for Paper 2 R1.

The validator performs no inference and publishes nothing. It accepts only the
six official P2R1 labels, recomputes every retained score, verifies registered
sessions and offline identity/shutdown/warmup evidence, and writes one private
promotion record. Legacy P2G paths are never inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
P2 = ROOT / "testsuite" / "paper2"
R1 = P2 / "r1"
RESULTS = ROOT / "testsuite" / "evals" / "results"
sys.path.insert(0, str(P2))

from identity_guard import (  # noqa: E402
    artifact_fast_state,
    load_artifact_record,
    official_llama_launch_contract,
    sha256_file,
)
from r1_safety import (  # noqa: E402
    RunSafetyError,
    attempt_ambiguity_error,
    append_checked_jsonl,
    checked_jsonl_head_path,
    ensure_immutable_json,
    file_identity,
    read_checked_jsonl,
    require_attempt_eligible,
    require_attempt_id,
    torn_tail_recovery_receipts,
)
from protocol_bindings import (  # noqa: E402
    analysis_source_hashes,
    require_analysis_committed_clean,
    source_hashes,
)
from run_grid import (  # noqa: E402
    item_complete,
    read_jsonl,
    validate_api_evidence,
    validate_prior_rows,
    validate_request_ledger,
)
import gen_questions as genq  # noqa: E402
import warmup_r1 as warm  # noqa: E402
import campaign_guard as campaign_guard  # noqa: E402
import runtime_provenance as runtime_provenance  # noqa: E402


OFFICIAL = {
    "P2R1-Q4KXL-TSI": 117,
    "P2R1-Q4KXL-WC": 120,
    "P2R1-EXL3-TSI": 117,
    "P2R1-EXL3-WC": 120,
    "P2R1-IQ2S-TSI": 117,
    "P2R1-IQ2S-WC": 120,
}
OFFICIAL_TIER_BY_LABEL = {
    label: label.split("-")[1] for label in OFFICIAL
}
Q4_GENERATION_DEPLOYMENT_FIELDS = (
    "model_id",
    "engine",
    "engine_revision",
    "artifact_sha256",
    "artifact_record_sha256",
)
Q4_EVALUATION_DEPLOYMENT_FIELDS = (
    "tier",
    "server_model_id",
    "engine",
    "engine_revision",
    "artifact_sha256",
    "artifact_record_sha256",
)
CONDS = ["O", "A", "L2", "E"]
SUPPORTING_FILES: set[Path] = set()
ACTIVE_ATTEMPT_CONTEXT: tuple[Path, str] | None = None
RUNTIME_IDENTITY_CACHE: dict[tuple[str, str], tuple[dict, dict, dict, dict]] = {}


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunSafetyError(f"cannot read valid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RunSafetyError(f"expected a JSON object: {path}")
    return value


def git_head(repo: Path) -> str:
    """Read one repository revision while preserving fail-closed error type."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunSafetyError(f"cannot verify engine repository revision: {repo}") from exc


def require_official_deployment(label: str, contract: dict) -> dict:
    """Bind an official result label to its independently derived tier."""
    expected_tier = OFFICIAL_TIER_BY_LABEL.get(label)
    if expected_tier is None:
        raise RunSafetyError(f"unrecognized official result label: {label}")
    deployment = contract.get("deployment")
    if (
        not isinstance(deployment, dict)
        or not isinstance(deployment.get("tier"), str)
        or deployment.get("tier") != expected_tier
    ):
        raise RunSafetyError(
            f"run contract deployment tier does not match result label: {label}"
        )
    return deployment


def require_exact_string_fields(
    value: object, fields: tuple[str, ...], *, description: str,
) -> dict:
    """Return one exact string-only protocol projection or fail closed."""
    if not isinstance(value, dict) or set(value) != set(fields):
        raise RunSafetyError(f"{description} has missing or extra fields")
    if any(not isinstance(value.get(key), str) or not value[key] for key in fields):
        raise RunSafetyError(f"{description} has a missing or non-string field")
    return value


def require_q4_question_deployment_binding(
    generation_by_corpus: object,
    evaluation_by_corpus: object,
    post_q4_evidence: object,
) -> None:
    """Bind both question generators to official Q4 evaluation and post-state."""
    corpora = {"TSI", "WC"}
    if (
        not isinstance(generation_by_corpus, dict)
        or set(generation_by_corpus) != corpora
        or not isinstance(evaluation_by_corpus, dict)
        or set(evaluation_by_corpus) != corpora
    ):
        raise RunSafetyError(
            "Q4 question/evaluation deployment evidence must cover TSI and WC exactly"
        )

    generations: dict[str, dict] = {}
    evaluations: dict[str, dict] = {}
    for corpus in sorted(corpora):
        generations[corpus] = require_exact_string_fields(
            generation_by_corpus[corpus],
            Q4_GENERATION_DEPLOYMENT_FIELDS,
            description=f"{corpus} Q4 question-generation deployment",
        )
        evaluations[corpus] = require_exact_string_fields(
            evaluation_by_corpus[corpus],
            Q4_EVALUATION_DEPLOYMENT_FIELDS,
            description=f"{corpus} Q4 evaluation deployment",
        )
        if evaluations[corpus]["tier"] != "Q4KXL":
            raise RunSafetyError(f"{corpus} official Q4 evaluation has the wrong tier")
        expected_generation = {
            "model_id": evaluations[corpus]["server_model_id"],
            "engine": evaluations[corpus]["engine"],
            "engine_revision": evaluations[corpus]["engine_revision"],
            "artifact_sha256": evaluations[corpus]["artifact_sha256"],
            "artifact_record_sha256": evaluations[corpus]["artifact_record_sha256"],
        }
        if generations[corpus] != expected_generation:
            raise RunSafetyError(
                f"{corpus} Q4 question generator differs from official Q4 evaluation"
            )

    if generations["TSI"] != generations["WC"]:
        raise RunSafetyError("TSI and WC used different Q4 question generators")
    if evaluations["TSI"] != evaluations["WC"]:
        raise RunSafetyError("TSI and WC used different official Q4 deployments")

    if not isinstance(post_q4_evidence, dict):
        raise RunSafetyError("Q4 post-campaign artifact evidence is malformed")
    post_artifact = post_q4_evidence.get("artifact")
    if not isinstance(post_artifact, dict):
        raise RunSafetyError("Q4 post-campaign artifact identity is malformed")
    generator = generations["TSI"]
    if (
        post_artifact.get("sha256") != generator["artifact_sha256"]
        or post_q4_evidence.get("artifact_record_sha256")
        != generator["artifact_record_sha256"]
    ):
        raise RunSafetyError(
            "Q4 question generator differs from post-campaign artifact evidence"
        )


def retain_file(path: Path) -> Path:
    """Bind one required private evidence file into the promotion receipt."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise RunSafetyError(f"required supporting evidence is missing: {resolved}")
    SUPPORTING_FILES.add(resolved)
    return resolved


def retain_checked_stream(path: Path, *, optional: bool = False) -> None:
    """Bind both an append-only stream and its atomic chain-head receipt."""
    resolved = path.resolve()
    head = checked_jsonl_head_path(resolved)
    if not resolved.exists():
        if optional and not head.exists():
            return
        raise RunSafetyError(f"required append-only evidence is missing: {resolved}")
    # The caller also performs stream-specific semantic validation. This read
    # independently proves the checksum chain and exact head/count.
    read_checked_jsonl(resolved)
    retain_file(resolved)
    retain_file(head)


def retain_source_manifest(
    input_manifest_path: Path, *, corpus: str, require_canonical_path: bool = True,
) -> Path:
    """Verify and retain the private source manifest behind one input build.

    The public input manifest records only the source-manifest digest. The
    private source manifest contains the exact upstream mapping needed to
    audit the corpus build, so promotion must retain those bytes—not merely a
    copied digest—through claims and release.
    """
    directory_name = {"TSI": "tsi", "WC": "wildchat"}.get(corpus)
    source_schema = {
        "TSI": "paper2-r1-tsi-source-v1",
        "WC": "paper2-r1-wildchat-source-v1",
    }.get(corpus)
    if directory_name is None or source_schema is None:
        raise RunSafetyError(f"unsupported P2R1 corpus for source retention: {corpus}")
    input_path = input_manifest_path.resolve()
    expected_input = (
        R1 / "private_inputs" / directory_name / "input_manifest.json"
    ).resolve()
    if require_canonical_path and input_path != expected_input:
        raise RunSafetyError(
            f"{corpus} input manifest is outside its canonical private corpus"
        )
    input_value = load_json(input_path)
    source_path = input_path.with_name("source_manifest.json")
    if not source_path.is_file():
        raise RunSafetyError(f"{corpus} private source manifest is missing")
    source_value = load_json(source_path)
    if (
        source_value.get("schema") != source_schema
        or source_value.get("corpus") != corpus
        or input_value.get("source_manifest_sha256") != sha256_file(source_path)
    ):
        raise RunSafetyError(f"{corpus} private source manifest changed")
    return retain_file(source_path)


def finite_number(value: object, *, positive: bool = False) -> bool:
    """Reject bools and non-finite JSON numbers before quantitative use."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (float(value) > 0 if positive else float(value) >= 0)
    )


def require_identity(path: Path) -> tuple[dict, str]:
    try:
        path.resolve().relative_to((R1 / "receipts" / "campaigns").resolve())
    except ValueError as exc:
        raise RunSafetyError(f"identity receipt is outside private campaign state: {path}") from exc
    receipt = load_json(path)
    SUPPORTING_FILES.add(path.resolve())
    if (
        receipt.get("schema") != "paper2-endpoint-identity-v1"
        or receipt.get("research_contract") != "EMPIRICAL_RESEARCH_LOOP.md@1.1.0"
    ):
        raise RunSafetyError(f"wrong identity-receipt schema: {path}")
    artifact = receipt.get("artifact", {})
    required_artifact_keys = {
        "kind", "path", "sha256", "file_count", "total_bytes", "max_mtime_ns"
    }
    if not isinstance(artifact, dict) or not required_artifact_keys <= set(artifact):
        raise RunSafetyError(f"identity receipt has malformed artifact state: {path}")
    current = artifact_fast_state(Path(artifact["path"]))
    frozen = (artifact["file_count"], artifact["total_bytes"], artifact["max_mtime_ns"])
    if current != frozen:
        raise RunSafetyError(f"artifact fast state changed after run: {path}")
    artifact_record = Path(receipt["artifact_record"]).resolve()
    try:
        artifact_record.relative_to((R1 / "receipts" / "artifacts").resolve())
    except ValueError as exc:
        raise RunSafetyError(f"artifact record is outside frozen private state: {path}") from exc
    SUPPORTING_FILES.add(artifact_record.resolve())
    if sha256_file(artifact_record) != receipt["artifact_record_sha256"]:
        raise RunSafetyError(f"artifact-record hash changed: {path}")
    artifact_record_value = load_json(artifact_record)
    if (
        artifact_record_value.get("schema") != "paper2-artifact-identity-v1"
        or artifact_record_value.get("artifact") != artifact
        or not isinstance(artifact_record_value.get("source_provenance"), dict)
    ):
        raise RunSafetyError(f"artifact record does not bind receipt artifact: {path}")

    campaign = path.resolve().parent.parent
    process_record = Path(receipt["launch_process_record"]).resolve()
    try:
        process_record.relative_to((campaign / "process").resolve())
    except ValueError as exc:
        raise RunSafetyError(f"process record is outside its campaign: {path}") from exc
    SUPPORTING_FILES.add(process_record.resolve())
    if sha256_file(process_record) != receipt["launch_process_record_sha256"]:
        raise RunSafetyError(f"process-record hash changed: {path}")
    process_record_value = load_json(process_record)
    identity = receipt.get("identity")
    if not isinstance(identity, dict):
        raise RunSafetyError(f"identity receipt lacks endpoint readback: {path}")
    if (
        process_record_value.get("schema") != "paper2-launched-process-v1"
        or process_record_value.get("process") != identity.get("process_fingerprint")
        or process_record_value.get("cmdline") != identity.get("launch_cmdline")
        or process_record_value.get("cwd") != identity.get("cwd")
        or process_record_value.get("process", {}).get("pid") != receipt.get("launched_pid")
    ):
        raise RunSafetyError(f"process record does not bind endpoint identity: {path}")
    if receipt.get("engine") == "llama.cpp":
        expected_launch = official_llama_launch_contract(
            cmdline=process_record_value["cmdline"],
            process=process_record_value["process"],
            artifact_path=artifact["path"],
            expected_model_id=receipt["expected_model_id"],
            base_url=receipt["base_url"],
        )
        if expected_launch is None or receipt.get("llama_launch_contract") != expected_launch:
            raise RunSafetyError(f"llama.cpp launch argv is not the frozen contract: {path}")
    log = Path(receipt["identity"]["server_log"]).resolve()
    try:
        log.relative_to((campaign / "logs").resolve())
    except ValueError as exc:
        raise RunSafetyError(f"server log is outside its campaign: {path}") from exc
    SUPPORTING_FILES.add(log.resolve())
    nbytes = int(receipt["identity"]["server_log_prefix_bytes"])
    with log.open("rb") as handle:
        prefix = handle.read(nbytes)
    if len(prefix) != nbytes or hashlib.sha256(prefix).hexdigest() != receipt["identity"]["server_log_prefix_sha256"]:
        raise RunSafetyError(f"server identity-log prefix changed: {path}")
    if receipt.get("config_file"):
        config = Path(receipt["config_file"]).resolve()
        try:
            config.relative_to((R1 / "private_inputs" / "runtime").resolve())
        except ValueError as exc:
            raise RunSafetyError(f"server config is outside private runtime state: {path}") from exc
        SUPPORTING_FILES.add(config)
        if sha256_file(config) != receipt["config_sha256"]:
            raise RunSafetyError(f"dedicated server config changed: {path}")
    return receipt, sha256_file(path)


def terminal_snapshot(campaign: Path, stage: str) -> tuple[Path, dict]:
    """Accept one normal post-state or one crash/interrupt abort-state receipt."""
    choices = [
        (campaign / "host" / f"{stage}-post.json", f"{stage}-post"),
        (campaign / "host" / f"{stage}-abort.json", f"{stage}-abort"),
    ]
    present = [(path, label) for path, label in choices if path.is_file()]
    if len(present) != 1:
        raise RunSafetyError(
            f"stage requires exactly one post/abort host-state receipt: {stage}"
        )
    path, label = present[0]
    value = load_json(path)
    if value.get("schema") != "paper2-r1-host-state-v1" or value.get("stage") != label:
        raise RunSafetyError(f"terminal host snapshot is malformed: {path}")
    return path, value


def validate_campaign_context(
    campaign: Path, stage: str, *, attempt_id: str,
    attempt_root: Path, question_root: Path,
) -> None:
    try:
        campaign.resolve().relative_to((R1 / "receipts" / "campaigns").resolve())
    except ValueError as exc:
        raise RunSafetyError(f"referenced campaign is outside private state: {campaign}") from exc
    toolchain_path = campaign / "toolchain.json"
    attempt_path = campaign / "attempt.json"
    cool_path = campaign / "host" / f"{stage}-pre.json"
    if (
        not toolchain_path.is_file()
        or not attempt_path.is_file()
        or not cool_path.is_file()
    ):
        raise RunSafetyError(f"campaign lacks toolchain/cool evidence for {stage}: {campaign}")
    attempt = load_json(attempt_path)
    host_binding_path = Path(attempt.get("private_host_binding", "")).resolve()
    expected_host_binding_path = (attempt_root / "HOST_BINDING.json").resolve()
    if (
        attempt.get("schema") != "paper2-r1-attempt-namespace-v1"
        or attempt.get("attempt_id") != attempt_id
        or attempt.get("result_root") != str(attempt_root.resolve())
        or attempt.get("question_root") != str(question_root.resolve())
        or host_binding_path != expected_host_binding_path
        or not host_binding_path.is_file()
        or attempt.get("private_host_binding_sha256") != sha256_file(host_binding_path)
    ):
        raise RunSafetyError(f"session campaign belongs to another attempt: {campaign}")
    host_binding = load_json(host_binding_path)
    if (
        host_binding.get("schema") != "paper2-r1-private-attempt-host-binding-v1"
        or host_binding.get("attempt_id") != attempt_id
    ):
        raise RunSafetyError(f"attempt host binding is malformed: {campaign}")
    toolchain = load_json(toolchain_path)
    exl_runtime = toolchain.get("exl3_runtime", {})
    exl_layout = exl_runtime.get("private_runtime_layout", {})
    if (
        toolchain.get("schema") != "paper2-r1-toolchain-v1"
        or toolchain.get("protocol_sha256") != sha256_file(PAPER_DIR / "R1_PROTOCOL.json")
        or toolchain.get("recovery_harness_sha256") != source_hashes(ROOT)
        or toolchain.get("llama_cpp", {}).get("revision") != campaign_guard.LLAMA_REVISION
        or toolchain.get("tabbyapi", {}).get("revision") != campaign_guard.TABBY_REVISION
        or toolchain.get("llama_cpp", {}).get("repository_clean") is not True
        or toolchain.get("cuda_compiler")
        != toolchain.get("llama_cpp", {}).get("build_identity", {}).get(
            "tool_versions", {}
        ).get("nvcc")
        or toolchain.get("llama_cpp", {}).get("cuda_runtime", {}).get("schema")
        != "paper2-r1-llama-cuda-runtime-private-v1"
        or toolchain.get("tabbyapi", {}).get("repository_clean") is not True
        or toolchain.get("exl3_runtime", {}).get("torch") != "2.10.0+cu128"
        or toolchain.get("exl3_runtime", {}).get("exllamav3") != "1.4.3+cu128.torch2.10.0"
        or toolchain.get("exl3_runtime", {}).get("tabbyAPI") != "0.0.1"
        or toolchain.get("exl3_runtime", {}).get("uvicorn") != "0.52.4"
        or toolchain.get("exl3_runtime", {}).get("fastapi") != "0.141.1"
        or exl_runtime.get("launch_loader_contract")
        != "all inherited LD_* variables removed; no LD_LIBRARY_PATH"
        or exl_layout.get("schema")
        != "paper2-r1-exl3-private-runtime-layout-v1"
        or exl_layout.get("loader_environment_contract")
        != "all inherited LD_* variables removed; no LD_LIBRARY_PATH"
        or exl_runtime.get("installed_content", {}).get("observed_ld_variables") != {}
    ):
        raise RunSafetyError(f"campaign toolchain differs from frozen protocol: {campaign}")
    host_path = Path(toolchain.get("host_receipt", "")).resolve()
    try:
        host_path.relative_to((campaign / "host").resolve())
    except ValueError as exc:
        raise RunSafetyError(f"host inventory is outside its campaign: {campaign}") from exc
    if (
        not host_path.is_file()
        or toolchain.get("host_receipt_sha256") != sha256_file(host_path)
        or Path(toolchain.get("attempt_host_binding", "")).resolve()
        != host_binding_path
        or toolchain.get("attempt_host_binding_sha256")
        != sha256_file(host_binding_path)
    ):
        raise RunSafetyError(f"campaign host inventory changed: {campaign}")
    host = load_json(host_path)
    if (
        host.get("schema") != "paper2-r1-host-inventory-v1"
        or host.get("attempt_id") != attempt_id
        or {key: host.get(key) for key in campaign_guard.EXPECTED_HOST}
        != campaign_guard.EXPECTED_HOST
        or host.get("private_host_binding")
        != host_binding.get("private_host_binding")
    ):
        raise RunSafetyError(f"campaign ran on an unregistered host: {campaign}")
    llama_server = Path(toolchain["llama_cpp"]["server"])
    exl_python = Path(toolchain["exl3_runtime"]["python"])
    if (
        exl_python.absolute()
        != (ROOT / "engines" / "exl3-env" / "bin" / "python").absolute()
        or sha256_file(llama_server) != toolchain["llama_cpp"]["server_sha256"]
        or sha256_file(exl_python.resolve()) != toolchain["exl3_runtime"]["python_sha256"]
        or "commit 035e227" not in toolchain["llama_cpp"].get("server_version", "")
        or toolchain.get("methodology_sha256")
        != sha256_file(ROOT / "methodology" / "EMPIRICAL_RESEARCH_LOOP.md")
    ):
        raise RunSafetyError(f"campaign executable/methodology evidence changed: {campaign}")
    cache_key = (str(llama_server.resolve()), str(exl_python.absolute()))
    if cache_key not in RUNTIME_IDENTITY_CACHE:
        RUNTIME_IDENTITY_CACHE[cache_key] = (
            runtime_provenance.llama_build_identity(llama_server),
            runtime_provenance.llama_cuda_runtime_identity(llama_server),
            runtime_provenance.exl3_private_runtime_layout(exl_python),
            runtime_provenance.python_runtime_identity(exl_python),
        )
    llama_build, cuda_runtime, private_layout, python_runtime = RUNTIME_IDENTITY_CACHE[cache_key]
    if (
        toolchain.get("llama_cpp", {}).get("build_identity") != llama_build
        or toolchain.get("llama_cpp", {}).get("cuda_runtime") != cuda_runtime
        or toolchain.get("exl3_runtime", {}).get("private_runtime_layout")
        != private_layout
        or toolchain.get("exl3_runtime", {}).get("installed_content")
        != python_runtime
    ):
        raise RunSafetyError(f"engine build/runtime contents changed: {campaign}")
    for repo, expected in (
        (ROOT / "engines" / "llama.cpp", campaign_guard.LLAMA_REVISION),
        (ROOT / "engines" / "tabbyAPI", campaign_guard.TABBY_REVISION),
    ):
        observed = git_head(repo)
        if observed != expected:
            raise RunSafetyError(f"engine repository changed after campaign: {repo}")
    cool = load_json(cool_path)
    thresholds = {
        "max_gpu_temperature_c": 50.0,
        "gpu_utilization_policy": "record-only-device-total",
        "max_gpu_memory_used_mib": 1536.0,
        "max_one_min_load": 8.0,
        "min_mem_available_gib": campaign_guard.MIN_MEM_AVAILABLE_GIB,
        "foreign_compute_processes": 0,
        "consecutive_samples": 3,
        "sample_interval_seconds": 10,
    }
    samples = cool.get("samples")
    if (
        cool.get("schema") != "paper2-r1-cool-gate-v2"
        or cool.get("stage") != stage
        or cool.get("thresholds") != thresholds
        or not isinstance(samples, list)
        or len(samples) != 3
    ):
        raise RunSafetyError(f"cool-gate contract changed: {cool_path}")
    for sample in samples:
        if (
            sample.get("stage") != stage
            or sample.get("compute_process_pids") != []
            or not finite_number(sample.get("load_average", {}).get("one_min"))
            or float(sample["load_average"]["one_min"]) > 8.0
            or not finite_number(sample.get("memory", {}).get("MemAvailable"))
            or float(sample["memory"]["MemAvailable"])
            < campaign_guard.MIN_MEM_AVAILABLE_GIB * 1024**3
            or not sample.get("gpus")
        ):
            raise RunSafetyError(f"cool-gate sample failed: {cool_path}")
        for gpu in sample["gpus"]:
            if (
                not finite_number(gpu.get("temperature_c"))
                or not finite_number(gpu.get("utilization_percent"))
                or not finite_number(gpu.get("memory_used_mib"))
                or float(gpu["temperature_c"]) > 50.0
                or not 0.0 <= float(gpu["utilization_percent"]) <= 100.0
                or float(gpu["memory_used_mib"]) > 1536.0
            ):
                raise RunSafetyError(f"GPU cool-gate sample failed: {cool_path}")
    SUPPORTING_FILES.update(
        {
            toolchain_path.resolve(), attempt_path.resolve(), cool_path.resolve(),
            host_path.resolve(), host_binding_path.resolve(),
        }
    )


def validate_stage_evidence(
    identity_path: Path, receipt: dict, digest: str, *, stage: str,
    require_warmup: bool, attempt_id: str, attempt_root: Path,
    question_root: Path,
) -> None:
    campaign = identity_path.resolve().parent.parent
    validate_campaign_context(
        campaign, stage, attempt_id=attempt_id, attempt_root=attempt_root,
        question_root=question_root,
    )
    if identity_path.name != f"{stage}.json" or identity_path.parent.name != "identity":
        raise RunSafetyError(f"identity receipt has wrong stage location: {identity_path}")
    shutdown_path = campaign / "shutdown" / f"{stage}.json"
    placement_path = campaign / "placement" / f"{stage}.json"
    if not shutdown_path.is_file() or not placement_path.is_file():
        raise RunSafetyError(
            f"stage lacks canonical shutdown/placement evidence: {stage}"
        )
    terminal_path, _terminal = terminal_snapshot(campaign, stage)
    SUPPORTING_FILES.update(
        {shutdown_path.resolve(), placement_path.resolve(), terminal_path.resolve()}
    )
    shutdown = load_json(shutdown_path)
    expected_shutdown = {
        "schema": "paper2-endpoint-shutdown-v1",
        "identity_receipt": str(identity_path.resolve()),
        "identity_receipt_sha256": digest,
        "launched_pid": receipt["launched_pid"],
        "host": receipt["identity"]["host"],
        "port": receipt["identity"]["port"],
        "verified_pid_absent": True,
        "verified_port_free": True,
    }
    for key, value in expected_shutdown.items():
        if shutdown.get(key) != value:
            raise RunSafetyError(f"shutdown evidence mismatch ({key}): {shutdown_path}")
    placement = load_json(placement_path)
    if (
        placement.get("schema") != "paper2-r1-placement-v1"
        or placement.get("identity_receipt") != str(identity_path.resolve())
        or placement.get("identity_receipt_sha256") != digest
        or placement.get("engine") != receipt["engine"]
        or placement.get("server_log") != receipt["identity"]["server_log"]
        or not placement.get("launched_tree_gpu_processes")
    ):
        raise RunSafetyError(f"placement evidence is incomplete: {placement_path}")
    log = Path(placement["server_log"])
    prefix_n = placement.get("server_log_prefix_bytes")
    prefix_sha = placement.get("server_log_prefix_sha256")
    if not isinstance(prefix_n, int) or prefix_n <= 0:
        raise RunSafetyError(f"placement log length is invalid: {placement_path}")
    with log.open("rb") as handle:
        prefix = handle.read(prefix_n)
    if len(prefix) != prefix_n or hashlib.sha256(prefix).hexdigest() != prefix_sha:
        raise RunSafetyError(f"placement log prefix changed: {placement_path}")
    if receipt["engine"] == "llama.cpp":
        expected_layers = "65/65" if "IQ2S" in receipt["expected_model_id"] else "33/66"
        loaded_runtime = placement.get("llama_cuda_process_runtime", {})
        toolchain_path = (campaign / "toolchain.json").resolve()
        if (
            placement.get("offloaded_layers") != expected_layers
            or Path(placement.get("toolchain_receipt", "")).resolve()
            != toolchain_path
            or placement.get("toolchain_receipt_sha256") != sha256_file(toolchain_path)
            or loaded_runtime.get("schema")
            != "paper2-r1-llama-cuda-process-maps-private-v1"
            or loaded_runtime.get("pid") != receipt["launched_pid"]
            or loaded_runtime.get("libraries")
            != load_json(toolchain_path)["llama_cpp"]["cuda_runtime"]["libraries"]
        ):
            raise RunSafetyError(f"wrong llama.cpp placement: {placement_path}")
    else:
        matched = placement["launched_tree_gpu_processes"]
        toolchain_path = (campaign / "toolchain.json").resolve()
        toolchain = load_json(toolchain_path)
        exl_runtime = toolchain.get("exl3_runtime", {})
        layout = exl_runtime.get("private_runtime_layout", {})
        mapped_runtime = placement.get("exl3_cuda_process_runtime", {})
        try:
            runtime_provenance.validate_exl3_process_cuda_mapping_identity(
                mapped_runtime,
                expected_root_pid=receipt["launched_pid"],
                expected_site_packages_root=Path(layout["site_packages_root"]),
                expected_extension=exl_runtime["installed_content"]["precompiled_extension"],
            )
        except (KeyError, TypeError) as exc:
            raise RunSafetyError(
                f"EXL3 runtime mapping evidence is malformed: {placement_path}"
            ) from exc
        if (
            placement.get("backend") != "ExLlamaV3"
            or placement.get("loaded_artifact") != receipt["artifact"]["path"]
            or Path(placement.get("toolchain_receipt", "")).resolve()
            != toolchain_path
            or placement.get("toolchain_receipt_sha256")
            != sha256_file(toolchain_path)
            or any(not finite_number(row.get("used_memory_mib"), positive=True) for row in matched)
            # Mirrors campaign_guard.EXL3_MIN_RESIDENT_ALLOCATION_MIB (7,000 MiB;
            # the frozen deployment allocates 7,666-7,706 MiB, measured 2026-09-13).
            or max(float(row["used_memory_mib"]) for row in matched) < 7000
        ):
            raise RunSafetyError(f"EXL3 full-GPU placement is not proven: {placement_path}")
    if require_warmup:
        warmup_path = campaign / "warmup" / f"{stage}.json"
        if not warmup_path.is_file():
            raise RunSafetyError(f"stage lacks canonical warmup evidence: {stage}")
        SUPPORTING_FILES.add(warmup_path.resolve())
        value = load_json(warmup_path)
        expected_prompt = warm.PROMPT.format(
            window=warm.WARMUP_WINDOW, q=warm.WARMUP_QUESTION
        )
        if (
            value.get("schema") != "paper2-r1-unmeasured-warmup-v1"
            or value.get("identity_receipt") != str(identity_path.resolve())
            or value.get("identity_receipt_sha256") != digest
            or value.get("model_id") != receipt["expected_model_id"]
            or value.get("prompt_sha256") != hashlib.sha256(expected_prompt.encode()).hexdigest()
            or value.get("sampling") != {"temperature": 0.0, "seed": 42, "max_tokens": 2048}
            or value.get("excluded_from_analysis") is not True
            or value.get("extractive_token_verified") is not True
            or not isinstance(value.get("usage"), dict)
            or not finite_number(value.get("wall_s_unmeasured"), positive=True)
        ):
            raise RunSafetyError(f"warmup evidence is incomplete: {warmup_path}")
        response, usage, response_model = validate_api_evidence(
            value.get("api_evidence"), success=True
        )
        if (
            response_model != receipt["expected_model_id"]
            or value.get("model_id") != response_model
            or value.get("usage") != usage
            or value.get("response_sha256")
            != hashlib.sha256(response.encode()).hexdigest()
        ):
            raise RunSafetyError(f"warmup raw API evidence changed: {warmup_path}")


def validate_session_receipts(
    sessions: list[dict], *, tier: str, attempt_id: str,
    attempt_root: Path, question_root: Path,
) -> tuple[list[str], dict]:
    hashes: list[str] = []
    artifact_evidence: dict | None = None
    for session in sessions:
        path = Path(session["identity_receipt"])
        receipt, digest = require_identity(path)
        if digest != session["identity_receipt_sha256"]:
            raise RunSafetyError(f"session identity hash changed: {path}")
        if receipt["expected_model_id"] != session["server_model_id"]:
            raise RunSafetyError(f"session/API identity mismatch: {path}")
        if receipt["engine"] != session["engine"] or receipt["engine_revision"] != session["engine_revision"]:
            raise RunSafetyError(f"session engine mismatch: {path}")
        if receipt["artifact"]["sha256"] != session["artifact_sha256"]:
            raise RunSafetyError(f"session artifact mismatch: {path}")
        if receipt["artifact_record_sha256"] != session["artifact_record_sha256"]:
            raise RunSafetyError(f"session artifact-record mismatch: {path}")
        observed_artifact = {
            "tier": tier,
            "artifact": receipt["artifact"],
            "artifact_record": receipt["artifact_record"],
            "artifact_record_sha256": receipt["artifact_record_sha256"],
        }
        if artifact_evidence is None:
            artifact_evidence = observed_artifact
        elif artifact_evidence != observed_artifact:
            raise RunSafetyError(f"sessions for {tier} used different artifacts")
        stage = {
            "Q4KXL": "q4-evaluation",
            "EXL3": "exl3-evaluation",
            "IQ2S": "iq2s-evaluation",
        }[tier]
        validate_stage_evidence(
            path, receipt, digest, stage=stage, require_warmup=True,
            attempt_id=attempt_id, attempt_root=attempt_root,
            question_root=question_root,
        )
        hashes.append(digest)
    if artifact_evidence is None:
        raise RunSafetyError(f"official tier has no registered session: {tier}")
    return sorted(set(hashes)), artifact_evidence


def validate_post_campaign_artifacts(campaign: Path) -> tuple[Path, dict]:
    path = campaign / "artifacts" / "post-campaign.json"
    value = load_json(path)
    if (
        value.get("schema")
        != "paper2-r1-post-campaign-artifact-verification-v1"
        or value.get("timing_relation")
        != "after all model stages, timed requests, and shutdowns"
        or set(value.get("artifacts", {})) != {"Q4KXL", "EXL3", "IQ2S"}
    ):
        raise RunSafetyError("post-campaign artifact verification is malformed")
    for tier, evidence in value["artifacts"].items():
        record_path = Path(evidence.get("artifact_record", "")).resolve()
        try:
            record_path.relative_to((R1 / "receipts" / "artifacts").resolve())
        except ValueError as exc:
            raise RunSafetyError(f"{tier} artifact record is outside private state") from exc
        if (
            evidence.get("full_hash_verified") is not True
            or evidence.get("artifact_record_sha256") != sha256_file(record_path)
        ):
            raise RunSafetyError(f"{tier} post-campaign artifact evidence changed")
        record = load_artifact_record(record_path, full_hash=True)
        if record.get("artifact") != evidence.get("artifact"):
            raise RunSafetyError(f"{tier} post-campaign artifact identity changed")
        SUPPORTING_FILES.add(record_path)
    SUPPORTING_FILES.add(path.resolve())
    return path, value


def validate_post_campaign_runtime(campaign: Path, toolchain: dict) -> tuple[Path, dict]:
    """Require unchanged CUDA 12.6 resolution before, during, and after P2R1."""
    path = campaign / "runtime" / "post-campaign.json"
    value = load_json(path)
    toolchain_path = (campaign / "toolchain.json").resolve()
    before = toolchain.get("llama_cpp", {}).get("cuda_runtime")
    after = value.get("post_campaign_cuda_runtime")
    if (
        value.get("schema")
        != "paper2-r1-post-campaign-runtime-verification-v1"
        or value.get("timing_relation")
        != "after all model stages, timed requests, and shutdowns"
        or Path(value.get("toolchain_receipt", "")).resolve() != toolchain_path
        or value.get("toolchain_receipt_sha256") != sha256_file(toolchain_path)
        or value.get("unchanged_from_pre_campaign") is not True
        or not isinstance(before, dict)
        or value.get("pre_campaign_cuda_runtime") != before
        or after != before
        or value.get("public_cuda_runtime")
        != runtime_provenance.sanitize_llama_cuda_runtime(before)
    ):
        raise RunSafetyError("post-campaign CUDA runtime verification is malformed")
    server = Path(toolchain.get("llama_cpp", {}).get("server", ""))
    if runtime_provenance.llama_cuda_runtime_identity(server) != after:
        raise RunSafetyError("CUDA runtime changed after post-campaign verification")
    SUPPORTING_FILES.add(path.resolve())
    return path, value


def validate_question_evidence(
    contract: dict, *, attempt_id: str, question_root: Path,
) -> dict:
    for key in (
        "questions", "question_manifest", "question_generation_contract",
        "question_generation_journal", "question_generation_attempts",
        "question_generation_sessions",
    ):
        try:
            Path(contract[key]["path"]).resolve().relative_to(question_root.resolve())
        except (KeyError, TypeError, ValueError) as exc:
            raise RunSafetyError(
                f"{key} is outside the campaign attempt question namespace"
            ) from exc
    generation_contract_path = Path(contract["question_generation_contract"]["path"])
    generation_contract = load_json(generation_contract_path)
    try:
        generation_contract_path.resolve().relative_to(question_root.resolve())
    except ValueError as exc:
        raise RunSafetyError("question contract is outside the campaign attempt") from exc
    if (
        contract.get("attempt_id") != attempt_id
        or generation_contract.get("attempt_id") != attempt_id
    ):
        raise RunSafetyError("question-generation evidence belongs to another attempt")
    question_manifest = load_json(Path(contract["question_manifest"]["path"]))
    if question_manifest.get("attempt_id") != attempt_id:
        raise RunSafetyError("question manifest belongs to another attempt")

    # Retain every frozen input/static projection plus all three raw
    # append-only generation streams and their chain heads. These paths remain
    # private, but their hashes must survive through claims and canonical build.
    for key in (
        "windows", "questions", "input_manifest", "question_manifest",
        "question_generation_contract",
    ):
        retain_file(Path(contract[key]["path"]))
    for key in (
        "question_generation_journal", "question_generation_attempts",
        "question_generation_sessions",
    ):
        retain_checked_stream(Path(contract[key]["path"]))
    deployment = require_exact_string_fields(
        generation_contract.get("q4_deployment"),
        Q4_GENERATION_DEPLOYMENT_FIELDS,
        description="Q4 question-generation deployment",
    )
    session_path = Path(contract["question_generation_sessions"]["path"])
    sessions = genq.load_generation_sessions(
        session_path, sha256_file(generation_contract_path), deployment
    )
    if any(session.get("attempt_id") != attempt_id for session in sessions.values()):
        raise RunSafetyError("question-generation session belongs to another attempt")
    windows = genq.stratified_window_order(
        read_jsonl(Path(contract["windows"]["path"]))
    )
    journal = read_checked_jsonl(
        Path(contract["question_generation_journal"]["path"])
    )
    attempts = read_checked_jsonl(
        Path(contract["question_generation_attempts"]["path"])
    )
    for key, source in (
        ("journal_head", Path(contract["question_generation_journal"]["path"])),
        ("attempt_log_head", Path(contract["question_generation_attempts"]["path"])),
        ("sessions_head", Path(contract["question_generation_sessions"]["path"])),
    ):
        if question_manifest.get(key) != file_identity(checked_jsonl_head_path(source)):
            raise RunSafetyError(f"question-generation {key} changed")
    corpus = generation_contract.get("corpus")
    if corpus not in {"TSI", "WC"}:
        raise RunSafetyError("question-generation contract has an invalid corpus")
    if len(journal) != sum(genq.FRAME_COUNTS[corpus].values()):
        raise RunSafetyError(f"{corpus} question journal does not cover its full frame")
    genq.validate_journal(
        journal, windows, corpus=corpus, sessions=sessions, base_seed=42
    )
    genq.validate_generation_attempt_ledger(
        attempts,
        windows,
        corpus=corpus,
        sessions=sessions,
        base_seed=42,
        journal_rows=journal,
    )
    if set(sessions) != {row.get("generation_session_id") for row in journal}:
        raise RunSafetyError(f"{corpus} has an orphan question-generation session")
    questions = read_jsonl(Path(contract["questions"]["path"]))
    if questions != genq.derive_question_rows(journal, corpus, 42):
        raise RunSafetyError(f"{corpus} questions changed from their journal projection")
    for session in sessions.values():
        receipt, digest = require_identity(Path(session["identity_receipt"]))
        if digest != session["identity_receipt_sha256"]:
            raise RunSafetyError(f"{corpus} generator receipt hash changed")
        if (
            receipt["expected_model_id"] != session["model_id"]
            or receipt["engine"] != "llama.cpp"
            or receipt["engine_revision"] != deployment["engine_revision"]
            or receipt["artifact"]["sha256"] != deployment["artifact_sha256"]
            or receipt["artifact_record_sha256"] != deployment["artifact_record_sha256"]
        ):
            raise RunSafetyError(f"{corpus} generator model identity changed")
        validate_stage_evidence(
            Path(session["identity_receipt"]), receipt, digest,
            stage="q4-question-generation", require_warmup=False,
            attempt_id=attempt_id, attempt_root=(
                RESULTS / "paper2-r1" / attempt_id
            ).resolve(), question_root=question_root,
        )
    return dict(deployment)


def validate_one(
    label: str, target: int, *, attempt_id: str, attempt_root: Path,
    question_root: Path,
) -> tuple[dict, dict, dict]:
    outdir = attempt_root / label
    contract_path = outdir / "run_contract.json"
    output_path = outdir / "p2grid.jsonl"
    summary_path = outdir / "summary.json"
    sessions_path = outdir / "run_sessions.jsonl"
    request_ledger_path = outdir / "request_ledger.jsonl"
    for path in (
        contract_path, output_path, summary_path, sessions_path, request_ledger_path,
    ):
        if not path.exists():
            raise RunSafetyError(f"incomplete official run {label}: missing {path.name}")
    # Reading every append-only stream validates per-event checksums and also
    # rejects the persistent receipt left by any forensic torn-tail recovery.
    failures_path = outdir / "failures.jsonl"
    failures = read_checked_jsonl(failures_path)
    for path in (contract_path, summary_path):
        retain_file(path)
    for path in (output_path, request_ledger_path, sessions_path):
        retain_checked_stream(path)
    retain_checked_stream(failures_path, optional=True)
    contract = load_json(contract_path)
    if contract.get("schema") != "paper2-r1-run-contract-v1" or contract.get("label") != label:
        raise RunSafetyError(f"wrong run contract for {label}")
    deployment = require_official_deployment(label, contract)
    if (
        contract.get("attempt_id") != attempt_id
        or contract.get("attempt_root") != str(attempt_root.resolve())
    ):
        raise RunSafetyError(f"run contract belongs to another P2R1 attempt: {label}")
    if contract.get("request_ledger") != str(request_ledger_path.resolve()):
        raise RunSafetyError(f"run contract has a noncanonical request ledger for {label}")
    if contract.get("n_items") != target or contract.get("conditions") != CONDS:
        raise RunSafetyError(f"official item/condition contract changed for {label}")
    if contract.get("recovery_harness_sha256") != source_hashes(ROOT):
        raise RunSafetyError(f"recovery harness changed after run: {label}")
    frozen = {
        "temperature": 0.2, "top_p": 0.95, "seed": 42,
        "max_tokens": 2048, "request_timeout_s": 600,
    }
    for key, value in frozen.items():
        if contract.get(key) != value:
            raise RunSafetyError(f"official sampling parameter changed ({key}) for {label}")
    for key in (
        "windows", "questions", "input_manifest", "question_manifest",
        "question_generation_contract", "question_generation_journal",
        "question_generation_attempts", "question_generation_sessions",
    ):
        identity = contract.get(key)
        if not isinstance(identity, dict):
            raise RunSafetyError(f"run contract lacks {key}: {label}")
        if file_identity(Path(identity["path"])) != identity:
            raise RunSafetyError(f"frozen input changed ({key}): {label}")
    label_corpus = label.rsplit("-", 1)[-1]
    retain_source_manifest(
        Path(contract["input_manifest"]["path"]), corpus=label_corpus,
    )
    q_contract = load_json(Path(contract["question_generation_contract"]["path"]))
    if q_contract.get("recovery_harness_sha256") != source_hashes(ROOT):
        raise RunSafetyError(f"question-generation harness changed for {label}")
    questions = read_jsonl(Path(contract["questions"]["path"]))
    if len(questions) != target:
        raise RunSafetyError(f"question count changed for {label}")
    question_map = {row["qid"]: row for row in questions}
    if len(question_map) != target:
        raise RunSafetyError(f"duplicate question IDs for {label}")
    window_map = {
        row["window_id"]: row for row in read_jsonl(Path(contract["windows"]["path"]))
    }
    question_generation_deployment = validate_question_evidence(
        contract, attempt_id=attempt_id, question_root=question_root,
    )
    selected = validate_prior_rows(
        outdir=outdir,
        output=output_path,
        run_contract_sha256=sha256_file(contract_path),
        deployment=deployment,
        questions=question_map,
        conds=CONDS,
    )
    validate_request_ledger(
        request_ledger_path,
        responses=selected,
        questions=question_map,
        windows=window_map,
        conds=CONDS,
        run_contract_sha256=sha256_file(contract_path),
        deployment=deployment,
    )
    if len(selected) != target * len(CONDS):
        raise RunSafetyError(f"{label} does not contain all {target * len(CONDS)} cells")
    if any(not item_complete(selected, qid, CONDS) for qid in question_map):
        raise RunSafetyError(f"{label} has an incomplete item")
    if any(not selected[(qid, cond)].get("source_cluster_id") for qid in question_map for cond in CONDS):
        raise RunSafetyError(f"{label} lacks opaque source-cluster IDs")
    sessions = read_checked_jsonl(sessions_path)
    session_ids = {row.get("session_id") for row in sessions}
    used_session_ids = {row["session_id"] for row in selected.values()}
    if session_ids != used_session_ids:
        raise RunSafetyError(
            f"{label} has an orphan or unregistered evaluation session"
        )
    for failure in failures:
        if (
            failure.get("schema") != "paper2-r1-request-failure-v1"
            or failure.get("session_id") not in session_ids
            or failure.get("qid") not in question_map
            or failure.get("condition") not in CONDS
            or not isinstance(failure.get("error_type"), str)
            or not isinstance(failure.get("error"), str)
        ):
            raise RunSafetyError(f"malformed request-failure evidence for {label}")
    receipt_hashes, artifact_evidence = validate_session_receipts(
        sessions, tier=deployment["tier"], attempt_id=attempt_id,
        attempt_root=attempt_root, question_root=question_root,
    )
    summary = load_json(summary_path)
    if summary.get("schema") != "paper2-r1-summary-v1" or summary.get("label") != label:
        raise RunSafetyError(f"wrong summary schema/label for {label}")
    expected_conditions = []
    for cond in CONDS:
        rows = [selected[(qid, cond)] for qid in sorted(question_map)]
        expected_conditions.append({
            "condition": cond,
            "n": target,
            "accuracy": sum(row["exact"] for row in rows) / target,
            "accuracy_lenient": sum(row["lenient"] for row in rows) / target,
            "mean_wall_s": sum(row["wall_s"] for row in rows) / target,
            "complete": True,
        })
    if summary.get("conditions") != expected_conditions:
        raise RunSafetyError(f"derived summary changed for {label}")
    expected_heads = {
        "responses": file_identity(checked_jsonl_head_path(output_path)),
        "request_ledger": file_identity(checked_jsonl_head_path(request_ledger_path)),
        "sessions": file_identity(checked_jsonl_head_path(sessions_path)),
        "failures": (
            file_identity(checked_jsonl_head_path(failures_path))
            if checked_jsonl_head_path(failures_path).exists() else None
        ),
    }
    if summary.get("append_only_heads") != expected_heads:
        raise RunSafetyError(f"append-only chain heads changed for {label}")
    run = {
        "label": label,
        "attempt_id": attempt_id,
        "result_directory": str(outdir.resolve()),
        "target_items": target,
        "cells": target * len(CONDS),
        "source_clusters": len({row["source_cluster_id"] for row in question_map.values()}),
        "run_contract_sha256": sha256_file(contract_path),
        "responses_sha256": sha256_file(output_path),
        "request_ledger_sha256": sha256_file(request_ledger_path),
        "sessions_sha256": sha256_file(sessions_path),
        "summary_sha256": sha256_file(summary_path),
        "failures_sha256": sha256_file(failures_path) if failures_path.exists() else None,
        "append_only_head_sha256": {
            "responses": sha256_file(checked_jsonl_head_path(output_path)),
            "request_ledger": sha256_file(checked_jsonl_head_path(request_ledger_path)),
            "sessions": sha256_file(checked_jsonl_head_path(sessions_path)),
            "failures": (
                sha256_file(checked_jsonl_head_path(failures_path))
                if checked_jsonl_head_path(failures_path).exists() else None
            ),
        },
        "identity_receipt_sha256s": receipt_hashes,
        "artifact_evidence": artifact_evidence,
    }
    return run, dict(deployment), question_generation_deployment


def synthetic_self_test() -> None:
    def expect_run_safety(callable_, failure: str) -> None:
        try:
            callable_()
        except RunSafetyError:
            return
        raise AssertionError(failure)

    def clone(value: object) -> object:
        return json.loads(json.dumps(value))

    if campaign_guard.MIN_MEM_AVAILABLE_GIB != 32.0:
        raise AssertionError("P2R1 memory reserve is not 32 GiB")
    if campaign_guard.EXL3_MIN_RESIDENT_ALLOCATION_MIB != 7000.0:
        raise AssertionError("EXL3 resident-allocation bound differs from the validator's 7,000 MiB")
    cool_args = campaign_guard.parser().parse_args([
        "cool", "--stage", "synthetic", "--output", "/tmp/synthetic.json",
        "--gpu-utilization-policy", "record-only-device-total",
    ])
    if cool_args.min_available_gib != 32.0:
        raise AssertionError("cool-gate CLI default differs from the 32 GiB reserve")
    if cool_args.gpu_utilization_policy != campaign_guard.GPU_UTILIZATION_POLICY:
        raise AssertionError("cool-gate utilization policy is not choice-locked")
    protocol = load_json(PAPER_DIR / "R1_PROTOCOL.json")
    expected_campaign_order = [
        "cool-gate-Q4KXL-question-generation",
        "Q4KXL-question-generation-launch",
        "Q4KXL-generate-TSI-then-WC",
        "Q4KXL-question-generation-post-snapshot-and-shutdown",
        "cool-gate-Q4KXL-evaluation",
        "Q4KXL-evaluation-relaunch-and-warmup",
        "Q4KXL-TSI", "Q4KXL-WC",
        "Q4KXL-evaluation-post-snapshot-and-shutdown",
        "cool-gate-EXL3-evaluation", "EXL3-launch-and-warmup",
        "EXL3-TSI", "EXL3-WC", "EXL3-post-snapshot-and-shutdown",
        "cool-gate-IQ2S-evaluation", "IQ2S-launch-and-warmup",
        "IQ2S-TSI", "IQ2S-WC", "IQ2S-post-snapshot-and-shutdown",
        "post-campaign-full-artifact-verification",
        "post-campaign-llama-CUDA-runtime-verification",
        "promotion-validation",
    ]
    if (
        "MemAvailable >=32 GiB"
        not in protocol.get("timing", {}).get("cool_gate_thresholds", "")
        or "utilization recorded but not capped"
        not in protocol.get("timing", {}).get("cool_gate_thresholds", "")
        or protocol.get("llama_cuda_runtime", {}).get("libraries")
        != runtime_provenance.CUDA_RUNTIME_CONTRACT
        or protocol.get("campaign_order") != expected_campaign_order
        or "no LD_LIBRARY_PATH"
        not in protocol.get("exl3_cuda_runtime", {}).get("launch_environment", "")
    ):
        raise AssertionError("prospective protocol differs from memory/CUDA contract")
    launcher = (P2 / "p2_grid_r1.sh").read_text()
    exl_launcher = launcher.split("launch_exl3() {", 1)[1].split("\nstop_current() {", 1)[0]
    if (
        "--min-available-gib 32" not in launcher
        or "--min-available-gib 8" in launcher
        or "--gpu-utilization-policy record-only-device-total" not in launcher
        or "--max-gpu-memory-mib 1536" not in launcher
        or "--max-load 8" not in launcher
        or 'LD_LIBRARY_PATH="$LLAMA_CUDA_LIBRARY_DIR"' not in launcher
        or "post-runtime" not in launcher
        or "compgen -A variable LD_" not in exl_launcher
        or "export LD_LIBRARY_PATH" in exl_launcher
    ):
        raise AssertionError("P2R1 launcher differs from memory/CUDA contract")
    matrix = {
        label: {condition: target for condition in CONDS}
        for label, target in OFFICIAL.items()
    }
    if set(matrix) != set(OFFICIAL) or sum(len(value) for value in matrix.values()) != 24:
        raise AssertionError("official 6-run/24-cell matrix is malformed")
    broken = dict(matrix)
    broken.pop("P2R1-IQ2S-WC")
    if set(broken) == set(OFFICIAL):
        raise AssertionError("synthetic missing-run fixture was accepted")

    official_tiers = {"Q4KXL", "EXL3", "IQ2S"}
    for label, expected_tier in OFFICIAL_TIER_BY_LABEL.items():
        contract = {"deployment": {"tier": expected_tier}}
        if require_official_deployment(label, contract) is not contract["deployment"]:
            raise AssertionError(f"correct official deployment was not retained: {label}")
        for wrong_tier in sorted(official_tiers - {expected_tier}):
            expect_run_safety(
                lambda label=label, wrong_tier=wrong_tier: require_official_deployment(
                    label, {"deployment": {"tier": wrong_tier}}
                ),
                f"cross-tier contract was accepted: {label}/{wrong_tier}",
            )
    for malformed_contract in (
        {},
        {"deployment": None},
        {"deployment": []},
        {"deployment": "Q4KXL"},
        {"deployment": {}},
        {"deployment": {"tier": None}},
        {"deployment": {"tier": {}}},
    ):
        expect_run_safety(
            lambda malformed_contract=malformed_contract: require_official_deployment(
                "P2R1-Q4KXL-TSI", malformed_contract
            ),
            "missing/non-dict official deployment or tier was accepted",
        )

    generator = {
        "model_id": "P2R1-Q4KXL-aaaaaaaaaaaa",
        "engine": "llama.cpp",
        "engine_revision": "revision-q4",
        "artifact_sha256": "a" * 64,
        "artifact_record_sha256": "b" * 64,
    }
    evaluation = {
        "tier": "Q4KXL",
        "server_model_id": generator["model_id"],
        "engine": generator["engine"],
        "engine_revision": generator["engine_revision"],
        "artifact_sha256": generator["artifact_sha256"],
        "artifact_record_sha256": generator["artifact_record_sha256"],
    }
    generation_by_corpus = {
        "TSI": dict(generator), "WC": dict(generator),
    }
    evaluation_by_corpus = {
        "TSI": dict(evaluation), "WC": dict(evaluation),
    }
    post_q4 = {
        "artifact": {"sha256": generator["artifact_sha256"]},
        "artifact_record_sha256": generator["artifact_record_sha256"],
    }
    require_q4_question_deployment_binding(
        generation_by_corpus, evaluation_by_corpus, post_q4,
    )

    for field in Q4_GENERATION_DEPLOYMENT_FIELDS:
        mutated = clone(generation_by_corpus)
        mutated["TSI"][field] += "-mutated"
        expect_run_safety(
            lambda mutated=mutated: require_q4_question_deployment_binding(
                mutated, evaluation_by_corpus, post_q4,
            ),
            f"mutated generator field was accepted: {field}",
        )
        missing = clone(generation_by_corpus)
        del missing["TSI"][field]
        expect_run_safety(
            lambda missing=missing: require_q4_question_deployment_binding(
                missing, evaluation_by_corpus, post_q4,
            ),
            f"missing generator field was accepted: {field}",
        )
    extra = clone(generation_by_corpus)
    extra["TSI"]["unregistered"] = "value"
    expect_run_safety(
        lambda: require_q4_question_deployment_binding(
            extra, evaluation_by_corpus, post_q4,
        ),
        "extra generator deployment field was accepted",
    )
    for malformed_generators in (
        None,
        {},
        {"TSI": dict(generator)},
        {"TSI": None, "WC": dict(generator)},
        {"TSI": dict(generator), "WC": dict(generator), "EXTRA": dict(generator)},
    ):
        expect_run_safety(
            lambda malformed_generators=malformed_generators: require_q4_question_deployment_binding(
                malformed_generators, evaluation_by_corpus, post_q4,
            ),
            "malformed generator deployment map was accepted",
        )

    # Prove the explicit cross-corpus equality check independently of each
    # generator-to-evaluation comparison.
    split_generators = clone(generation_by_corpus)
    split_evaluations = clone(evaluation_by_corpus)
    split_generators["TSI"]["model_id"] += "-other"
    split_evaluations["TSI"]["server_model_id"] += "-other"
    expect_run_safety(
        lambda: require_q4_question_deployment_binding(
            split_generators, split_evaluations, post_q4,
        ),
        "different TSI/WC Q4 deployments were accepted",
    )

    for malformed_evaluations in (
        None,
        {"TSI": dict(evaluation)},
        {"TSI": None, "WC": dict(evaluation)},
    ):
        expect_run_safety(
            lambda malformed_evaluations=malformed_evaluations: require_q4_question_deployment_binding(
                generation_by_corpus, malformed_evaluations, post_q4,
            ),
            "malformed Q4 evaluation deployment map was accepted",
        )
    missing_eval_tier = clone(evaluation_by_corpus)
    del missing_eval_tier["TSI"]["tier"]
    expect_run_safety(
        lambda: require_q4_question_deployment_binding(
            generation_by_corpus, missing_eval_tier, post_q4,
        ),
        "Q4 evaluation deployment without tier was accepted",
    )
    nondict_eval_tier = clone(evaluation_by_corpus)
    nondict_eval_tier["TSI"]["tier"] = {}
    expect_run_safety(
        lambda: require_q4_question_deployment_binding(
            generation_by_corpus, nondict_eval_tier, post_q4,
        ),
        "Q4 evaluation deployment with non-string tier was accepted",
    )

    for post_field in ("artifact_sha256", "artifact_record_sha256"):
        mutated_post = clone(post_q4)
        if post_field == "artifact_sha256":
            mutated_post["artifact"]["sha256"] = "c" * 64
        else:
            mutated_post["artifact_record_sha256"] = "d" * 64
        expect_run_safety(
            lambda mutated_post=mutated_post: require_q4_question_deployment_binding(
                generation_by_corpus, evaluation_by_corpus, mutated_post,
            ),
            f"mutated Q4 post evidence was accepted: {post_field}",
        )
    for malformed_post in (None, {}, {"artifact": None}):
        expect_run_safety(
            lambda malformed_post=malformed_post: require_q4_question_deployment_binding(
                generation_by_corpus, evaluation_by_corpus, malformed_post,
            ),
            "malformed Q4 post-artifact evidence was accepted",
        )

    with tempfile.TemporaryDirectory(prefix="p2r1-terminal-") as temp:
        campaign = Path(temp)
        malformed_json = campaign / "malformed.json"
        malformed_json.write_text("{")
        expect_run_safety(
            lambda: load_json(malformed_json),
            "malformed JSON escaped the fail-closed RunSafetyError path",
        )
        expect_run_safety(
            lambda: git_head(campaign / "missing-repository"),
            "git subprocess failure escaped the fail-closed RunSafetyError path",
        )
        source_manifest = campaign / "source_manifest.json"
        input_manifest = campaign / "input_manifest.json"
        source_manifest.write_text(json.dumps({
            "schema": "paper2-r1-wildchat-source-v1",
            "corpus": "WC",
        }))
        input_manifest.write_text(json.dumps({
            "source_manifest_sha256": sha256_file(source_manifest),
        }))
        retain_source_manifest(
            input_manifest, corpus="WC", require_canonical_path=False,
        )
        source_manifest.write_text(json.dumps({
            "schema": "paper2-r1-wildchat-source-v1",
            "corpus": "WC",
            "mutated": True,
        }))
        try:
            retain_source_manifest(
                input_manifest, corpus="WC", require_canonical_path=False,
            )
        except RunSafetyError:
            pass
        else:
            raise AssertionError("mutated private source manifest was accepted")
        (campaign / "host").mkdir()
        abort = campaign / "host" / "q4-evaluation-abort.json"
        abort.write_text(json.dumps({
            "schema": "paper2-r1-host-state-v1",
            "stage": "q4-evaluation-abort",
        }))
        selected, _ = terminal_snapshot(campaign, "q4-evaluation")
        if selected != abort:
            raise AssertionError("crash-resume abort snapshot was not accepted")
        post = campaign / "host" / "q4-evaluation-post.json"
        post.write_text(json.dumps({
            "schema": "paper2-r1-host-state-v1",
            "stage": "q4-evaluation-post",
        }))
        try:
            terminal_snapshot(campaign, "q4-evaluation")
        except RunSafetyError:
            pass
        else:
            raise AssertionError("ambiguous post+abort snapshots were accepted")

        stream = campaign / "p2grid.jsonl"
        append_checked_jsonl(stream, {"schema": "synthetic-response-v1"})
        with stream.open("ab") as handle:
            handle.write(b'{"schema":"torn-response-v1"')
            handle.flush()
        try:
            read_checked_jsonl(stream, recover_torn_tail=True)
        except RunSafetyError:
            pass
        else:
            raise AssertionError("validator accepted a killed-mid-append response stream")
        if len(torn_tail_recovery_receipts(stream)) != 1:
            raise AssertionError("validator recovery did not leave an invalidation receipt")
        try:
            read_checked_jsonl(stream)
        except RunSafetyError:
            pass
        else:
            raise AssertionError("validator later accepted a tail-recovered stream")

        # Same-spec machines are still distinct empirical hosts. An attempt's
        # private GPU/machine hashes are immutable across resumed campaigns.
        attempt_id = "p2r1-host-binding-selftest"
        host_a = campaign / "host-a.json"
        host_b = campaign / "host-b.json"
        binding_path = campaign / "HOST_BINDING.json"
        base = {
            "schema": "paper2-r1-host-inventory-v1",
            "attempt_id": attempt_id,
        }
        host_a.write_text(json.dumps({
            **base,
            "private_host_binding": {
                "scheme": "sha256 domain-separated by P2R1 attempt ID",
                "gpu_uuid_sha256": "a" * 64,
                "machine_id_sha256": "b" * 64,
            },
        }))
        host_b.write_text(json.dumps({
            **base,
            "private_host_binding": {
                "scheme": "sha256 domain-separated by P2R1 attempt ID",
                "gpu_uuid_sha256": "c" * 64,
                "machine_id_sha256": "d" * 64,
            },
        }))
        campaign_guard.bind_attempt_host(
            attempt_id=attempt_id, host_receipt=host_a, output=binding_path,
        )
        try:
            campaign_guard.bind_attempt_host(
                attempt_id=attempt_id, host_receipt=host_b, output=binding_path,
            )
        except RunSafetyError:
            pass
        else:
            raise AssertionError("same P2R1 attempt crossed physical hosts")

        private_cuda = {
            "schema": "paper2-r1-llama-cuda-runtime-private-v1",
            "toolkit_release": "12.6",
            "libraries": {
                soname: {
                    "soname": soname,
                    "loader_path": f"/private/cuda/{expected['filename']}",
                    "resolved_path": f"/private/cuda/{expected['filename']}",
                    **expected,
                }
                for soname, expected in runtime_provenance.CUDA_RUNTIME_CONTRACT.items()
            },
        }
        toolchain_path = campaign / "toolchain.json"
        toolchain = {
            "llama_cpp": {
                "server": "/synthetic/llama-server",
                "cuda_runtime": private_cuda,
            },
        }
        toolchain_path.write_text(json.dumps(toolchain))
        runtime_dir = campaign / "runtime"
        runtime_dir.mkdir()
        post_runtime_path = runtime_dir / "post-campaign.json"
        public_cuda = runtime_provenance.sanitize_llama_cuda_runtime(private_cuda)
        post_runtime = {
            "schema": "paper2-r1-post-campaign-runtime-verification-v1",
            "timing_relation": "after all model stages, timed requests, and shutdowns",
            "toolchain_receipt": str(toolchain_path.resolve()),
            "toolchain_receipt_sha256": sha256_file(toolchain_path),
            "pre_campaign_cuda_runtime": private_cuda,
            "post_campaign_cuda_runtime": private_cuda,
            "unchanged_from_pre_campaign": True,
            "public_cuda_runtime": public_cuda,
        }
        post_runtime_path.write_text(json.dumps(post_runtime))
        real_runtime_identity = runtime_provenance.llama_cuda_runtime_identity
        runtime_provenance.llama_cuda_runtime_identity = lambda _server: private_cuda
        try:
            validate_post_campaign_runtime(campaign, toolchain)
            post_runtime["unchanged_from_pre_campaign"] = False
            post_runtime_path.write_text(json.dumps(post_runtime))
            try:
                validate_post_campaign_runtime(campaign, toolchain)
            except RunSafetyError:
                pass
            else:
                raise AssertionError("mutated post-campaign CUDA receipt was accepted")
        finally:
            runtime_provenance.llama_cuda_runtime_identity = real_runtime_identity


def main() -> int:
    global ACTIVE_ATTEMPT_CONTEXT
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--campaign", type=Path)
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
        print("P2R1 post-run validator synthetic matrix test passed")
        return 0
    try:
        args.campaign.resolve().relative_to((R1 / "receipts" / "campaigns").resolve())
    except ValueError as exc:
        raise RunSafetyError("campaign path is outside private P2R1 state") from exc
    toolchain = load_json(args.campaign / "toolchain.json")
    if toolchain.get("schema") != "paper2-r1-toolchain-v1":
        raise RunSafetyError("campaign lacks a valid toolchain receipt")
    protocol = PAPER_DIR / "R1_PROTOCOL.json"
    protocol_value = load_json(protocol)
    if toolchain.get("protocol_sha256") != sha256_file(protocol):
        raise RunSafetyError("campaign used a different prospective protocol")
    if toolchain.get("recovery_harness_sha256") != source_hashes(ROOT):
        raise RunSafetyError("campaign recovery harness changed after execution")
    attempt_receipt_path = args.campaign / "attempt.json"
    attempt_receipt = load_json(attempt_receipt_path)
    attempt_id = attempt_receipt.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise RunSafetyError("campaign attempt receipt lacks an attempt ID")
    require_attempt_id(attempt_id)
    if attempt_id != protocol_value.get("official_attempt_id"):
        raise RunSafetyError("campaign attempt ID differs from the frozen protocol")
    attempt_root = (RESULTS / "paper2-r1" / attempt_id).resolve()
    question_root = (
        R1 / "private" / "attempts" / attempt_id / "questions"
    ).resolve()
    if (
        attempt_receipt.get("schema") != "paper2-r1-attempt-namespace-v1"
        or attempt_receipt.get("result_root") != str(attempt_root)
        or attempt_receipt.get("question_root") != str(question_root)
    ):
        raise RunSafetyError("campaign attempt namespace receipt is malformed")
    attempt_host_binding = (attempt_root / "HOST_BINDING.json").resolve()
    if (
        Path(attempt_receipt.get("private_host_binding", "")).resolve()
        != attempt_host_binding
        or not attempt_host_binding.is_file()
        or attempt_receipt.get("private_host_binding_sha256")
        != sha256_file(attempt_host_binding)
    ):
        raise RunSafetyError("campaign lacks the immutable attempt host binding")
    require_attempt_eligible(attempt_root)
    SUPPORTING_FILES.add(attempt_receipt_path.resolve())
    require_analysis_committed_clean(ROOT)
    # From this point onward, a failure concerns the frozen empirical evidence
    # chain rather than pre-run CLI/configuration. Persist the attempt marker
    # before the top-level handler reports it.
    ACTIVE_ATTEMPT_CONTEXT = (attempt_root, attempt_id)
    for stage in (
        "q4-question-generation", "q4-evaluation", "exl3-evaluation", "iq2s-evaluation"
    ):
        identity_path = args.campaign / "identity" / f"{stage}.json"
        receipt, digest = require_identity(identity_path)
        validate_stage_evidence(
            identity_path, receipt, digest, stage=stage,
            require_warmup=stage != "q4-question-generation",
            attempt_id=attempt_id, attempt_root=attempt_root,
            question_root=question_root,
        )
    post_artifact_path, post_artifacts = validate_post_campaign_artifacts(
        args.campaign
    )
    post_runtime_path, post_runtime = validate_post_campaign_runtime(
        args.campaign, toolchain,
    )
    validated_runs = [
        validate_one(
            label, target, attempt_id=attempt_id, attempt_root=attempt_root,
            question_root=question_root,
        )
        for label, target in OFFICIAL.items()
    ]
    runs = [run for run, _deployment, _generator in validated_runs]
    if sum(4 for _run in runs) != 24:
        raise RunSafetyError("official campaign does not contain exactly 24 cells")
    generation_by_corpus = {}
    evaluation_by_corpus = {}
    for run, deployment, generator in validated_runs:
        if OFFICIAL_TIER_BY_LABEL[run["label"]] != "Q4KXL":
            continue
        corpus = run["label"].rsplit("-", 1)[-1]
        generation_by_corpus[corpus] = generator
        evaluation_by_corpus[corpus] = deployment
    post_by_tier = post_artifacts.get("artifacts")
    if not isinstance(post_by_tier, dict):
        raise RunSafetyError("post-campaign artifact evidence lacks tier mapping")
    require_q4_question_deployment_binding(
        generation_by_corpus,
        evaluation_by_corpus,
        post_by_tier.get("Q4KXL"),
    )
    for run in runs:
        tier = run["artifact_evidence"]["tier"]
        post = post_artifacts["artifacts"][tier]
        if (
            run["artifact_evidence"]["artifact"] != post["artifact"]
            or run["artifact_evidence"]["artifact_record"]
            != post["artifact_record"]
            or run["artifact_evidence"]["artifact_record_sha256"]
            != post["artifact_record_sha256"]
        ):
            raise RunSafetyError(
                f"{run['label']} artifact differs from post-campaign full verification"
            )
    completed = max(
        load_json(attempt_root / label / "summary.json")["updated_at"]
        for label in OFFICIAL
    )
    host_receipt_path = Path(toolchain["host_receipt"])
    host_receipt = load_json(host_receipt_path)
    public_host = {
        key: host_receipt[key] for key in campaign_guard.EXPECTED_HOST
    }
    promotion = {
        "schema": "paper2-r1-private-promotion-v1",
        "status": "validated-for-claims-rebuild-not-publication",
        "study_status": "corrective exploratory rerun; not independent confirmation",
        "completed_at": completed,
        "protocol_sha256": sha256_file(protocol),
        "campaign": str(args.campaign.resolve()),
        "attempt_id": attempt_id,
        "attempt_namespace_sha256": sha256_file(attempt_receipt_path),
        "attempt_host_binding_sha256": sha256_file(attempt_host_binding),
        "result_root": str(attempt_root),
        "question_root": str(question_root),
        "campaign_toolchain_sha256": sha256_file(args.campaign / "toolchain.json"),
        "campaign_host_receipt_sha256": sha256_file(host_receipt_path),
        "host": public_host,
        "post_campaign_artifacts_sha256": sha256_file(post_artifact_path),
        "post_campaign_artifacts": post_artifacts["artifacts"],
        "post_campaign_runtime_sha256": sha256_file(post_runtime_path),
        "llama_cuda_runtime": post_runtime["public_cuda_runtime"],
        "analysis_harness_sha256": analysis_source_hashes(ROOT),
        "supporting_evidence_sha256": {
            str(path): sha256_file(path) for path in sorted(SUPPORTING_FILES)
        },
        "runs": runs,
        "official_runs": 6,
        "official_condition_cells": 24,
        "legacy_p2g_accepted": False,
    }
    # The authoritative receipt has one canonical, private location.  Do not
    # accept a caller-controlled output path: downstream claims bind this exact
    # file and a redirected lookalike must never be mistaken for promotion.
    ensure_immutable_json(R1 / "private" / "promotion.json", promotion)
    print(json.dumps(promotion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        if ACTIVE_ATTEMPT_CONTEXT is not None:
            attempt_root, attempt_id = ACTIVE_ATTEMPT_CONTEXT
            marked = attempt_ambiguity_error(
                attempt_root, attempt_id=attempt_id,
                reason="postrun-promotion-evidence-ambiguity", error=exc,
            )
            raise SystemExit(f"P2R1 PROMOTION ABORTED: {marked}") from exc
        raise SystemExit(f"P2R1 PROMOTION ABORTED: {exc}") from exc
