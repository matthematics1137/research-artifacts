#!/usr/bin/env python3
"""Static and mock regressions for the P4R1 validation contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix() -> None:
    guard = load("p4_guard_matrix", "server_guard.py")
    prepare = load("p4_prepare_matrix", "prepare_model_manifest.py")
    assert {row[0] for row in prepare.SPECS} == set(guard.P4R1_MATRIX)
    assert all(values[3] == 4096 for values in guard.P4R1_MATRIX.values())
    script = (HERE / "run_matrix.sh").read_text(encoding="utf-8")
    for alias, (_filename, _sha, ngl, context) in guard.P4R1_MATRIX.items():
        assert f'"${{run[@]}}" {alias} ' in script
        assert f" {ngl} {context} " in script
    assert "MODEL_ROOT" not in script
    assert script.count("preflight P4R1-") == 6
    assert "verify-manifest-content" in script
    assert '--toolchain-receipt "$results/toolchain-receipt.json"' in script
    assert "--post-model-verification" in script
    assert "umask 077" in script
    assert "chmod 0600" in script and "chmod 0700" in script
    runner = (HERE / "run_one.sh").read_text(encoding="utf-8")
    assert "umask 077" in runner
    assert "--jinja -lv 4" in runner
    assert "verify-placement-log" in runner
    assert runner.index("verify-placement-log") < runner.index(
        'python3 -B "$script_dir/run_gsm8k.py"'
    )
    assert "record-shutdown" in runner and "--shutdown-identity" in runner
    assert "chmod 0600" in runner and "chmod 0700" in runner


def test_stat_only_live_identity() -> None:
    guard = load("p4_guard_stat", "server_guard.py")
    with tempfile.TemporaryDirectory(prefix="p4r1-stat-only-") as raw:
        root = Path(raw)
        entries = []
        paths = {}
        for index, (alias, (filename, digest, _ngl, _context)) in enumerate(
            guard.P4R1_MATRIX.items()
        ):
            model_path = root / f"{index}-{filename}"
            # Sparse placeholders make a production-shape manifest without
            # reading or allocating the real multi-GiB artifacts.
            with model_path.open("wb") as handle:
                handle.truncate(guard.P4R1_BYTES[alias])
            stat = model_path.stat()
            paths[alias] = model_path
            entries.append(
                {
                    "alias": alias,
                    "family": "synthetic-stat-regression",
                    "filename": model_path.name,
                    "path": str(model_path.resolve()),
                    "bytes": stat.st_size,
                    "sha256": digest,
                    "stat": {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "mtime_ns": stat.st_mtime_ns,
                    },
                }
            )
        # Production requires canonical filenames. Rename after unique creation
        # and refresh each stat identity.
        for entry, (alias, (filename, _digest, _ngl, _context)) in zip(
            entries, guard.P4R1_MATRIX.items(), strict=True
        ):
            source = Path(entry["path"])
            target = root / filename
            source.rename(target)
            stat = target.stat()
            paths[alias] = target
            entry["filename"] = filename
            entry["path"] = str(target.resolve())
            entry["stat"] = {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
            }
        payload = (
            json.dumps(
                {
                    "schema": "paper4-validation-model-manifest-v1",
                    "content_verification": "full SHA-256 completed before the campaign; live cells use stat-only checks",
                    "artifact_count": 6,
                    "total_bytes": sum(guard.P4R1_BYTES.values()),
                    "artifacts": entries,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        manifest = root / "manifest.json"
        manifest.write_bytes(payload)
        output = root / "receipt.json"
        alias = "P4R1-QWEN3MOE-IQ1M"
        model = paths[alias]
        model.chmod(0)
        try:
            guard.verify_model_entry(
                SimpleNamespace(
                    manifest=manifest,
                    manifest_sha256=hashlib.sha256(payload).hexdigest(),
                    model=model,
                    alias=alias,
                    output=output,
                )
            )
            assert output.is_file()
        finally:
            model.chmod(0o600)
        manifest_value = json.loads(payload)
        cuda_root = "/home/private/person/cuda-12.6"
        cuda_runtime = {
            "schema": "paper4-p4r1-cuda-runtime-resolution-v1",
            "configured_cudatoolkit_root": cuda_root,
            "configured_lib64": f"{cuda_root}/lib64",
            "resolved_lib64": f"{cuda_root}/targets/x86_64-linux/lib",
            "ld_library_path_first": f"{cuda_root}/lib64",
            "loader_probe": {
                "tool_path": "/usr/bin/ldd", "tool_sha256": "1" * 64,
                "returncode": 0, "stdout_sha256": "2" * 64,
                "stderr_sha256": "3" * 64,
            },
            "libraries": [
                {
                    "soname": soname,
                    "loader_path": f"{cuda_root}/lib64/{soname}",
                    "resolved_path": f"{cuda_root}/targets/x86_64-linux/lib/{expected['resolved_filename']}",
                    **expected,
                }
                for soname, expected in guard.EXPECTED_CUDA_RUNTIME_LIBRARIES.items()
            ],
            "verification": "ldd resolved the exact hashed CUDA 12.6 runtime libraries beneath the CMake-configured toolkit root",
        }
        toolchain_sha256 = "4" * 64
        toolchain = {"observed_cuda_runtime": cuda_runtime}
        post_receipt = {
            "schema": "paper4-p4r1-post-campaign-model-verification-v1",
            "model_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "toolchain_receipt_sha256": toolchain_sha256,
            "artifact_count": 6,
            "total_bytes": sum(guard.P4R1_BYTES.values()),
            "artifacts": [
                {
                    "alias": entry["alias"], "filename": entry["filename"],
                    "bytes": entry["bytes"], "sha256": entry["sha256"],
                    "pre_campaign_stat": entry["stat"],
                    "post_campaign_stat": entry["stat"],
                }
                for entry in entries
            ],
            "post_campaign_cuda_runtime": json.loads(json.dumps(cuda_runtime)),
            "verification": "synthetic post-campaign binding fixture",
        }
        guard.validate_post_model_verification(
            post_receipt, hashlib.sha256(payload).hexdigest(), manifest_value,
            toolchain_sha256, toolchain,
        )
        tampered = json.loads(json.dumps(post_receipt))
        tampered["artifacts"][0]["post_campaign_stat"]["mtime_ns"] += 1
        try:
            guard.validate_post_model_verification(
                tampered, hashlib.sha256(payload).hexdigest(), manifest_value,
                toolchain_sha256, toolchain,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("post-campaign verifier accepted a changed artifact stat")
        tampered = json.loads(json.dumps(post_receipt))
        tampered["post_campaign_cuda_runtime"]["libraries"][0]["sha256"] = "0" * 64
        try:
            guard.validate_post_model_verification(
                tampered, hashlib.sha256(payload).hexdigest(), manifest_value,
                toolchain_sha256, toolchain,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("post-campaign verifier accepted a changed CUDA runtime")


def test_timing_boundary_and_model_response() -> None:
    harness = load("p4_harness_timing", "run_gsm8k.py")
    args = SimpleNamespace(
        model_id="P4R1-MOCK",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_tokens=2048,
        reasoning_effort="medium",
        base_url="http://127.0.0.1:1",
        http_wall_s=0.0,
        guard_wall_s=0.0,
    )

    class Response:
        def __enter__(self):
            time.sleep(0.01)
            return self

        def __exit__(self, *_):
            return False

        def read(self, *_):
            return json.dumps({"model": "P4R1-MOCK", "choices": []}).encode()

    def identity(namespace):
        started = time.monotonic()
        time.sleep(0.03)
        namespace.guard_wall_s += time.monotonic() - started

    with mock.patch.object(harness, "prove_identity", identity), mock.patch.object(
        harness.urllib.request, "urlopen", return_value=Response()
    ):
        response, retry_error = harness.request_completion(args, "prompt", 4_200_001)
    assert response["model"] == "P4R1-MOCK" and retry_error is None
    assert 0.008 <= args.http_wall_s < 0.025
    assert args.guard_wall_s >= 0.055


def test_malformed_200_and_retry_telemetry() -> None:
    harness = load("p4_harness_malformed", "run_gsm8k.py")
    args = SimpleNamespace(
        model_id="P4R1-MOCK", label="P4R1-MOCK", temperature=1.0,
        top_p=0.95, top_k=20, max_tokens=2048, reasoning_effort="medium",
        base_url="http://127.0.0.1:1", http_wall_s=0.0, guard_wall_s=0.0,
        sampler_seed_base=4_200_000, model_manifest_sha256="a" * 64,
        startup_identity_sha256="b" * 64, dataset_sha256="c" * 64,
        harness_sha256="d" * 64, server_guard_sha256="e" * 64,
        sampler_contract_sha256="f" * 64,
    )

    class MalformedResponse:
        status = 200
        reason = "OK"
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def geturl(self):
            return "http://127.0.0.1:1/v1/chat/completions"

        def read(self, *_):
            return b'{"model":'

    with mock.patch.object(harness, "prove_identity", return_value=None), mock.patch.object(
        harness.urllib.request, "urlopen", return_value=MalformedResponse()
    ):
        try:
            harness.request_completion(args, "prompt", 4_200_001)
        except json.JSONDecodeError as error:
            telemetry = harness.exception_telemetry(error)
        else:
            raise AssertionError("malformed HTTP 200 JSON was accepted")
    assert args.http_wall_s >= 0
    assert telemetry["http_status"] == 200
    assert telemetry["body_bytes"] == len(b'{"model":')
    assert telemetry["body_sha256"] == hashlib.sha256(b'{"model":').hexdigest()

    # If a retry succeeds at HTTP level but later response interpretation
    # fails, the original retry telemetry must remain attached to the row.
    retry = {**telemetry, "body_base64": "eyJtb2RlbCI6", "body_bytes": 9}
    retry["body_sha256"] = hashlib.sha256(b'{"model":').hexdigest()
    with mock.patch.object(
        harness, "request_completion", return_value=({"model": "P4R1-MOCK", "choices": []}, retry)
    ), mock.patch.object(harness, "prove_identity", return_value=None):
        row = harness.run_item(
            {"id": "gsm8k_1", "question": "1+1?", "answer_number": 2}, args
        )
    assert row["api_error"] is not None
    assert row["chat_template_retry_error"] == retry
    assert row["chat_template_kwargs_dropped"] is True


def test_live_argv_contract() -> None:
    guard = load("p4_guard_argv", "server_guard.py")
    args = SimpleNamespace(
        pid=123,
        start_time="9",
        launcher=Path("/tmp/llama-server"),
        model=Path("/tmp/model.gguf"),
        alias="P4R1-MOCK",
        host="127.0.0.1",
        port=8090,
        context=4096,
        ngl="26",
    )
    argv = [
        "/tmp/llama-server", "-m", "/tmp/model.gguf", "-c", "4096",
        "-ngl", "26", "--alias", "P4R1-MOCK", "--host", "127.0.0.1",
        "--port", "8090", "-lv", "4",
    ]
    with mock.patch.object(guard, "same_process", return_value=True), mock.patch.object(
        guard, "process_cmdline", return_value=argv
    ), mock.patch.object(guard, "launcher_matches", return_value=True), mock.patch.object(
        guard, "process_socket_inodes", return_value={"1"}
    ), mock.patch.object(guard, "listener_inodes", return_value={"1"}):
        guard.verify_process(args)
        args.context = 8192
        try:
            guard.verify_process(args)
        except RuntimeError:
            pass
        else:
            raise AssertionError("live argv check accepted the wrong context")
        args.context = 4096
        args.ngl = "auto"
        try:
            guard.verify_process(args)
        except RuntimeError:
            pass
        else:
            raise AssertionError("auto-fit argv check accepted an explicit -ngl override")


def test_cuda_runtime_resolution_gate() -> None:
    guard = load("p4_guard_cuda_runtime", "server_guard.py")
    with tempfile.TemporaryDirectory(prefix="p4r1-cuda-runtime-") as raw:
        root = Path(raw) / "cuda-12.6"
        lib64 = root / "lib64"
        lib64.mkdir(parents=True)
        expected = {}
        lines = []
        for index, soname in enumerate(guard.EXPECTED_CUDA_RUNTIME_LIBRARIES, 1):
            resolved_name = f"{soname}.6.{index}"
            resolved = lib64 / resolved_name
            payload = f"synthetic-{soname}".encode()
            resolved.write_bytes(payload)
            loader_path = lib64 / soname
            loader_path.symlink_to(resolved.name)
            expected[soname] = {
                "resolved_filename": resolved_name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            lines.append(f"\t{soname} => {loader_path} (0x00000001)")
        cache = {
            "CUDAToolkit_ROOT": str(root),
            "CUDA_CUDART": str(lib64 / "libcudart.so.12"),
        }
        completed = SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr="")
        environment = {"LD_LIBRARY_PATH": str(lib64)}
        with mock.patch.object(
            guard, "EXPECTED_CUDA_RUNTIME_LIBRARIES", expected
        ), mock.patch.object(guard.subprocess, "run", return_value=completed), mock.patch.dict(
            guard.os.environ, environment, clear=False
        ):
            receipt = guard.cuda_runtime_resolution(Path("/tmp/llama-server"), cache)
            assert len(receipt["libraries"]) == 3
            assert all(Path(item["resolved_path"]).is_relative_to(root) for item in receipt["libraries"])
            outside = Path(raw) / "outside.so"
            outside.write_bytes(b"outside")
            bad_lines = lines.copy()
            bad_lines[0] = f"\tlibcudart.so.12 => {outside} (0x00000001)"
            completed.stdout = "\n".join(bad_lines)
            try:
                guard.cuda_runtime_resolution(Path("/tmp/llama-server"), cache)
            except RuntimeError:
                pass
            else:
                raise AssertionError("CUDA runtime gate accepted a library outside the toolkit")


def test_placement_and_privacy() -> None:
    guard = load("p4_guard_placement", "server_guard.py")
    promoter = load("p4_promoter_privacy", "promote_validation.py")
    assert guard.PLACEMENT_LINE.search("common_fit_params: n_gpu_layers already set by user to 26")
    assert guard.PLACEMENT_LINE.search("load_model: loading model '/private/model.gguf'")
    assert guard.OFFLOADED_LAYERS.search("load_tensors: offloaded 26/33 layers to GPU")
    trace_line = "0.11.232.901 I load_tensors: offloaded 26/33 layers to GPU"
    with tempfile.TemporaryDirectory(prefix="p4r1-placement-probe-") as raw:
        root = Path(raw)
        stdout = root / "stdout.log"
        stderr = root / "stderr.log"
        stdout.write_text("", encoding="utf-8")
        stderr.write_text(trace_line + "\n", encoding="utf-8")
        lines, gpu, total = guard.parse_placement_logs(stdout, stderr, "26")
        assert lines == [trace_line] and (gpu, total) == (26, 33)
        guard.verify_placement_log(
            SimpleNamespace(
                alias="P4R1-MIXTRAL-IQ2XXS", ngl="26",
                stdout=stdout, stderr=stderr,
            )
        )
        try:
            guard.parse_placement_logs(stdout, stderr, "29")
        except RuntimeError as error:
            assert "differs from requested 29" in str(error)
        else:
            raise AssertionError("placement probe accepted the wrong explicit layer count")
        stderr.write_text("load_model: model loaded\n", encoding="utf-8")
        try:
            guard.parse_placement_logs(stdout, stderr, "26")
        except RuntimeError as error:
            assert "unambiguous offloaded-layer readback" in str(error)
        else:
            raise AssertionError("placement probe accepted a log without the exact readback")
    with tempfile.TemporaryDirectory(prefix="p4r1-privacy-") as raw:
        root = Path(raw)
        (root / "safe.json").write_text('{"filename":"model.gguf"}\n', encoding="utf-8")
        promoter.public_scan(root)
        (root / "leak.json").write_text('{"path":"/home/private/person/model.gguf"}\n', encoding="utf-8")
        try:
            promoter.public_scan(root)
        except promoter.PromotionError:
            pass
        else:
            raise AssertionError("privacy scanner accepted an absolute home path")
    for name, payload in (
        ("email", b'{"contact":"person@example.com"}\n'),
        ("ip", b'{"endpoint":"192.168.1.10"}\n'),
        ("gpu", b'{"device":"GPU-12345678-1234-1234-1234-123456789abc"}\n'),
    ):
        with tempfile.TemporaryDirectory(prefix=f"p4r1-{name}-") as raw:
            root = Path(raw)
            (root / "candidate.json").write_bytes(payload)
            try:
                promoter.public_scan(root)
            except promoter.PromotionError:
                pass
            else:
                raise AssertionError(f"privacy scanner accepted a {name} leak")
    with tempfile.TemporaryDirectory(prefix="p4r1-name-") as raw:
        root = Path(raw)
        (root / "person@example.com.json").write_text("{}\n", encoding="utf-8")
        try:
            promoter.public_scan(root)
        except promoter.PromotionError:
            pass
        else:
            raise AssertionError("privacy scanner accepted an email in an output filename")


def test_host_receipt_exact_storage_model() -> None:
    guard = load("p4_guard_host", "server_guard.py")
    receipt = {
        "schema": "paper4-p4r1-host-receipt-v1",
        "cpu": {
            "model": "Intel(R) Core(TM) i9-14900HX", "logical_cpus": 32,
            "physical_cores": 24, "sockets": 1,
        },
        "memory": {"mem_total_kib": 65445316},
        "operating_system": {
            "pretty_name": "Ubuntu 24.04.3 LTS",
            "kernel": "Linux 7.0.0-29-generic x86_64",
        },
        "gpu": {
            "name": "NVIDIA GeForce RTX 4080 Laptop GPU",
            "uuid": "GPU-11111111-2222-3333-4444-555555555555",
            "memory_total_mib": 12282,
            "power_limits": {
                "current_w": 80.0, "requested_w": 80.0, "default_w": 80.0,
            },
            "driver_version": "580.167.08",
        },
        "storage": {
            "verified_paths": [
                {
                    "role": role, "path": f"/private/{role}",
                    "source": "/dev/nvme0n1p2", "filesystem": "ext4",
                }
                for role in ("workspace", "mixtral_models", "qwen_models")
            ],
            "source": "/dev/nvme0n1p2", "filesystem": "ext4",
            "device": "/dev/nvme0n1",
            "model": "WD PC SN740 SDDPNQE-2T00-1102",
            "transport": "nvme", "rotational": 0,
        },
        "verification": "synthetic exact host fixture",
    }
    guard.validate_host_receipt(receipt)
    try:
        guard.validate_host_receipt(
            {**receipt, "storage": {**receipt["storage"], "model": "WD"}}
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("host receipt accepted a whitespace-truncated storage model")


def test_analysis_freeze_checks_clean_blob() -> None:
    promoter = load("p4_promoter_analysis_freeze", "promote_validation.py")
    relative = (
        "publication/papers/moe-quantization-granularity/public/replication/"
        "check_p4r1_evidence.py"
    )
    path = promoter.WORKSPACE / relative
    item = {
        "path": relative, "bytes": path.stat().st_size,
        "sha256": promoter.sha256_file(path), "git_blob": "a" * 40,
    }

    def clean_git(*arguments: str) -> str:
        if arguments[0] == "ls-files":
            return relative
        if arguments[0] == "status":
            return ""
        if arguments[0] == "rev-parse":
            return "a" * 40
        raise AssertionError(arguments)

    with mock.patch.object(promoter, "git_text", side_effect=clean_git):
        promoter.verify_current_analysis_freeze({relative: item})
    with mock.patch.object(
        promoter, "git_text",
        side_effect=lambda *arguments: (
            relative if arguments[0] == "ls-files" else
            " M " + relative if arguments[0] == "status" else "a" * 40
        ),
    ):
        try:
            promoter.verify_current_analysis_freeze({relative: item})
        except promoter.PromotionError:
            pass
        else:
            raise AssertionError("analysis freeze accepted a dirty result-interpreting source")


def test_private_row_rescore_and_schema() -> None:
    promoter = load("p4_promoter_row", "promote_validation.py")
    alias = "P4R1-MIXTRAL-IQ1M"
    raw_response = {
        "model": alias,
        "choices": [
            {
                "message": {"content": "<think>one two</think>\nAnswer: 5"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 12},
    }
    row = {
        "id": "gsm8k_1",
        "label": alias,
        "correct": True,
        "wall_s": 1.25,
        "identity_guard_wall_s": 0.5,
        "completion_tokens": 12,
        "thinking_tokens_estimate": 2,
        "truncated": False,
        "raw_response": raw_response,
        "scoring_content": "Answer: 5",
        "parsed_answer_number": 5.0,
        "expected_answer_number": 5.0,
        "api_error": None,
        "chat_template_retry_error": None,
        "chat_template_kwargs_dropped": False,
        "model_id": alias,
        "model_manifest_sha256": "a" * 64,
        "startup_identity_sha256": "b" * 64,
        "sampler_seed": 4_200_001,
        "dataset_sha256": promoter.DATASET_SHA256,
        "harness_sha256": "c" * 64,
        "server_guard_sha256": "d" * 64,
        "sampler_contract_sha256": "e" * 64,
    }
    sources = {
        "harness_sha256": "c" * 64,
        "server_guard_sha256": "d" * 64,
        "run_one_sha256": "f" * 64,
        "run_matrix_sha256": "0" * 64,
    }
    promoter.validate_private_row(
        row, alias, 5.0, "a" * 64, "e" * 64, "b" * 64, sources
    )
    for field, bad in (
        ("correct", False),
        ("thinking_tokens_estimate", 999),
        ("expected_answer_number", 6.0),
        ("wall_s", float("nan")),
    ):
        tampered = {**row, field: bad}
        try:
            promoter.validate_private_row(
                tampered, alias, 5.0, "a" * 64, "e" * 64, "b" * 64, sources
            )
        except promoter.PromotionError:
            pass
        else:
            raise AssertionError(f"private row validator accepted tampered {field}")
    missing = dict(row)
    missing.pop("scoring_content")
    try:
        promoter.validate_private_row(
            missing, alias, 5.0, "a" * 64, "e" * 64, "b" * 64, sources
        )
    except promoter.PromotionError:
        pass
    else:
        raise AssertionError("private row validator accepted a missing key")

    empty_sha = hashlib.sha256(b"").hexdigest()
    telemetry = {
        "schema": "paper4-p4r1-api-error-v1",
        "exception_type": "RuntimeError",
        "message": "synthetic model error",
        "http_status": None,
        "http_reason": None,
        "url": None,
        "headers": [],
        "body_bytes": 0,
        "body_sha256": empty_sha,
        "body_base64": "",
    }
    error_row = {
        **row,
        "correct": False,
        "completion_tokens": None,
        "thinking_tokens_estimate": 0,
        "raw_response": None,
        "scoring_content": "",
        "parsed_answer_number": None,
        "api_error": telemetry,
    }
    promoter.validate_private_row(
        error_row, alias, 5.0, "a" * 64, "e" * 64, "b" * 64, sources
    )
    tampered_telemetry = {**telemetry, "body_sha256": "0" * 64}
    try:
        promoter.validate_private_row(
            {**error_row, "api_error": tampered_telemetry}, alias, 5.0,
            "a" * 64, "e" * 64, "b" * 64, sources,
        )
    except promoter.PromotionError:
        pass
    else:
        raise AssertionError("private row validator accepted tampered exact API-error bytes")


def test_shutdown_and_private_permissions() -> None:
    guard = load("p4_guard_shutdown", "server_guard.py")
    prepare = load("p4_prepare_permissions", "prepare_model_manifest.py")
    promoter = load("p4_promoter_permissions", "promote_validation.py")
    with tempfile.TemporaryDirectory(prefix="p4r1-shutdown-") as raw:
        root = Path(raw)
        root.chmod(0o700)
        private = root / "private"
        private.mkdir(mode=0o700)
        private_file = private / "receipt.json"
        prepare.write_exclusive(private_file, b"{}\n")
        assert private_file.stat().st_mode & 0o777 == 0o600
        promoter.private_directory(private)
        promoter.private_file(private_file)

        process = subprocess.Popen(["sleep", "0.1"])
        start_time = guard.process_start_time(process.pid)
        process.wait(timeout=5)
        startup = root / "startup.json"
        completion = root / "completion.json"
        launcher = root / "llama-server"
        launcher.write_text("mock launcher\n", encoding="utf-8")
        launcher_hash = hashlib.sha256(launcher.read_bytes()).hexdigest()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        endpoint = {
            "host": "127.0.0.1", "port": port,
            "base_url": f"http://127.0.0.1:{port}",
            "model_id": "P4R1-SHUTDOWN-TEST",
        }
        common = {
            "schema": "paper4-validation-server-identity-v1",
            "alias": "P4R1-SHUTDOWN-TEST",
            "pid": process.pid,
            "process_start_time_ticks": start_time,
            "launcher": {"path": str(launcher.resolve()), "sha256": launcher_hash},
            "model_identity_sha256": "a" * 64,
            "model_manifest_sha256": "b" * 64,
            "endpoint": endpoint,
            "request": {"context_tokens": 4096, "gpu_layers": "auto"},
            "observed_gpu_process": {
                "pid": process.pid, "used_gpu_memory_mib": 1,
                "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
            },
            "verification": "synthetic strict lifecycle fixture",
        }
        startup.write_text(json.dumps({**common, "phase": "startup"}), encoding="utf-8")
        completion.write_text(json.dumps({**common, "phase": "completion"}), encoding="utf-8")
        output = root / "shutdown.json"
        guard.record_shutdown(
            SimpleNamespace(
                startup_identity=startup,
                completion_identity=completion,
                alias="P4R1-SHUTDOWN-TEST",
                pid=process.pid,
                start_time=start_time,
                host="127.0.0.1",
                port=port,
                output=output,
            )
        )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        assert receipt["same_process_alive"] is False and receipt["port_free"] is True


def main() -> int:
    test_matrix()
    test_stat_only_live_identity()
    test_timing_boundary_and_model_response()
    test_malformed_200_and_retry_telemetry()
    test_live_argv_contract()
    test_cuda_runtime_resolution_gate()
    test_placement_and_privacy()
    test_host_receipt_exact_storage_model()
    test_analysis_freeze_checks_clean_blob()
    test_private_row_rescore_and_schema()
    test_shutdown_and_private_permissions()
    print("P4R1 PROTOCOL REGRESSIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
