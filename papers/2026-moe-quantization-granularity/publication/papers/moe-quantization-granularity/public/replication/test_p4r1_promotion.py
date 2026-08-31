#!/usr/bin/env python3
"""End-to-end synthetic test of P4R1 promotion, public checking, and claims build.

This test performs no inference. It constructs six hash-bound mock cells over
the real pinned local GSM8K subset, then exercises the exact post-run path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
DATASET_SOURCE = WORKSPACE / "testsuite/evals/datasets/gsm8k_100.jsonl"
BUILDER_PATH = WORKSPACE / "paper4/scripts/build_p4r1_claims.py"


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    path.parent.chmod(0o700)


def private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    path.parent.chmod(0o700)


def main() -> int:
    promoter = load("p4_promotion_e2e", HERE / "promote_validation.py")
    checker = load("p4_checker_e2e", HERE / "check_p4r1_evidence.py")
    prepare = load("p4_prepare_e2e", HERE / "prepare_model_manifest.py")
    builder = load("p4_builder_e2e", BUILDER_PATH)
    manuscript_gate = load(
        "p4_manuscript_gate_e2e", WORKSPACE / "paper4/scripts/check_p4r1_manuscript.py"
    )
    promotion_tool_inventory = []
    for role, path in promoter.PROMOTION_TOOL_PATHS.items():
        promotion_tool_inventory.append(
            {
                "role": role,
                "path": path.relative_to(promoter.WORKSPACE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "git_blob": "f" * 40,
            }
        )
    promoter.promotion_tooling_receipt = lambda: {
        "schema": "paper4-p4r1-promotion-tooling-v1",
        "git_commit": "f" * 40,
        "git_state": "all promotion/checker/builder sources tracked and clean",
        "inventory": promotion_tool_inventory,
    }
    # The synthetic fixture uses uncommitted test bytes by construction. The
    # production promoter separately enforces the frozen full analysis tree's
    # tracked/clean/blob identity before accepting real evidence.
    promoter.verify_current_analysis_freeze = lambda _inventory: None
    bytes_by_alias = {alias: size for alias, _family, _filename, size, _sha in prepare.SPECS}
    dataset_rows = [
        json.loads(line)
        for line in DATASET_SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sha256_file(DATASET_SOURCE) == promoter.DATASET_SHA256

    with tempfile.TemporaryDirectory(prefix="p4r1-promotion-e2e-") as raw:
        root = Path(raw)
        root.chmod(0o700)
        campaign = root / "campaign"
        campaign.mkdir(mode=0o700)
        dataset = root / "gsm8k_100.jsonl"
        shutil.copyfile(DATASET_SOURCE, dataset)
        dataset.chmod(0o600)

        artifacts = []
        for alias, (filename, artifact_sha, _ngl, _context) in promoter.MATRIX.items():
            artifacts.append(
                {
                    "alias": alias,
                    "family": "synthetic",
                    "filename": filename,
                    "path": f"/home/private/person/models/{filename}",
                    "bytes": bytes_by_alias[alias],
                    "sha256": artifact_sha,
                    "stat": {"device": 1, "inode": len(artifacts) + 100, "mtime_ns": 1},
                }
            )
        manifest = root / "model-manifest.json"
        private_json(
            manifest,
            {
                "schema": "paper4-validation-model-manifest-v1",
                "content_verification": "full SHA-256 completed before the campaign; live cells use stat-only checks",
                "artifact_count": 6,
                "total_bytes": sum(item["bytes"] for item in artifacts),
                "artifacts": artifacts,
            },
        )
        manifest_sha = sha256_file(manifest)
        sources = {
            key: sha256_file(path)
            for key, path in promoter.SOURCE_PATHS.items()
        }
        sampler_sha = "e" * 64
        campaign_cells = []

        plan_inventory = []
        for relative in sorted(promoter.PLAN_SOURCE_PATHS):
            path = promoter.WORKSPACE / relative
            plan_inventory.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "git_blob": "a" * 40,
                }
            )
        analysis_inventory = []
        for relative in sorted(promoter.PLAN_ANALYSIS_PATHS):
            path = promoter.WORKSPACE / relative
            analysis_inventory.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "git_blob": "f" * 40,
                }
            )
        plan = {
            "schema": "paper4-p4r1-prospective-plan-v1",
            "status": "prospectively frozen corrective exploratory validation; not independent",
            "frozen_at_utc": "2026-08-30T00:00:00+00:00",
            "git_commit": "a" * 40,
            "git_state": "all inference/scoring and result-interpreting sources tracked and clean in index and worktree",
            "protocol": "publication/papers/moe-quantization-granularity/public/P4R1_PROTOCOL.md",
            "dataset_sha256": promoter.DATASET_SHA256,
            "artifact_sha256": [value[1] for value in promoter.MATRIX.values()],
            "inference_source_inventory": plan_inventory,
            "analysis_source_inventory": analysis_inventory,
        }
        plan_path = campaign / "protocol-plan.json"
        private_json(plan_path, plan)
        approval = {
            "schema": "paper4-p4r1-plan-approval-v1",
            "operation": "run-p4r1-validation",
            "actor": "Matthew Schwartz",
            "plan_sha256": sha256_file(plan_path),
            "plan_git_commit": plan["git_commit"],
            "approved_at_utc": "2026-08-30T00:01:00+00:00",
            "method": "interactive-exact-challenge",
        }
        approval_path = campaign / "protocol-plan-approval.json"
        private_json(approval_path, approval)

        launcher = root / "llama-server"
        launcher.write_text("synthetic 035e227 llama-server\n", encoding="utf-8")
        launcher.chmod(0o700)
        launcher_sha = sha256_file(launcher)
        toolchain = {
            "schema": "paper4-p4r1-toolchain-receipt-v1",
            "executable": {
                "path": str(launcher.resolve()), "filename": launcher.name,
                "bytes": launcher.stat().st_size, "sha256": launcher_sha,
            },
            "version_command": {
                "argv": [launcher.name, "--version"], "returncode": 0,
                "stdout": "version 035e227\n", "stderr": "",
                "combined_sha256": "b" * 64,
            },
            "observed_build_checkout": {
                "path": str(root / "llama.cpp"),
                "git_commit": "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3",
                "git_status": "clean including untracked files",
            },
            "observed_cmake_cache": {
                "path": str(root / "llama.cpp/build/CMakeCache.txt"),
                "sha256": "c" * 64,
                "settings": {
                    "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "89",
                    "GGML_BLAS": "OFF", "GGML_CUDA": "ON", "GGML_NATIVE": "ON",
                    "GGML_OPENMP": "ON", "LLAMA_CURL": "OFF",
                },
                "configured_tool_paths": {}, "resolved_tool_paths": {},
                "tool_sha256": {"cmake": "d" * 64, "cxx": "e" * 64, "nvcc": "f" * 64},
                "tool_versions": {
                    "cmake": "cmake version 3.28.3", "cxx": "GNU 13.3.0",
                    "nvcc": "Cuda compilation tools, release 12.6, V12.6.85",
                },
            },
            "observed_driver": {
                "driver_version": "580.167.08", "gpu_name": "Synthetic GPU",
                "driver_reported_cuda_compatibility": "13.0",
            },
            "observed_cuda_runtime": {
                "schema": "paper4-p4r1-cuda-runtime-resolution-v1",
                "configured_cudatoolkit_root": "/home/private/person/cuda-12.6",
                "configured_lib64": "/home/private/person/cuda-12.6/lib64",
                "resolved_lib64": "/home/private/person/cuda-12.6/targets/x86_64-linux/lib",
                "ld_library_path_first": "/home/private/person/cuda-12.6/lib64",
                "loader_probe": {
                    "tool_path": "/usr/bin/ldd", "tool_sha256": "1" * 64,
                    "returncode": 0, "stdout_sha256": "2" * 64,
                    "stderr_sha256": "3" * 64,
                },
                "libraries": [
                    {
                        "soname": soname,
                        "loader_path": f"/home/private/person/cuda-12.6/lib64/{soname}",
                        "resolved_path": f"/home/private/person/cuda-12.6/targets/x86_64-linux/lib/{values[0]}",
                        "resolved_filename": values[0],
                        "bytes": values[1],
                        "sha256": values[2],
                    }
                    for soname, values in promoter.CUDA_RUNTIME_LIBRARIES.items()
                ],
                "verification": "ldd resolved the exact hashed CUDA 12.6 runtime libraries beneath the CMake-configured toolkit root",
            },
            "historical_build_association_reference": {
                "llama_cpp_commit": "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3",
                "campaign_build_id": "b1-035e227",
                "meaning": "synthetic regression fixture",
            },
            "verification": "synthetic exact toolchain fixture",
        }
        toolchain_path = campaign / "toolchain-receipt.json"
        private_json(toolchain_path, toolchain)

        host = {
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
                        "role": role, "path": f"/home/private/person/{suffix}",
                        "source": "/dev/nvme0n1p2", "filesystem": "ext4",
                    }
                    for role, suffix in (
                        ("workspace", "workspace"),
                        ("mixtral_models", "models/mixtral"),
                        ("qwen_models", "models/qwen"),
                    )
                ],
                "source": "/dev/nvme0n1p2", "filesystem": "ext4",
                "device": "/dev/nvme0n1",
                "model": "WD PC SN740 SDDPNQE-2T00-1102",
                "transport": "nvme", "rotational": 0,
            },
            "verification": "synthetic exact host fixture",
        }
        host_path = campaign / "host-receipt.json"
        private_json(host_path, host)
        host_sha = sha256_file(host_path)
        post_model_verification = {
            "schema": "paper4-p4r1-post-campaign-model-verification-v1",
            "model_manifest_sha256": manifest_sha,
            "toolchain_receipt_sha256": sha256_file(toolchain_path),
            "artifact_count": 6,
            "total_bytes": sum(item["bytes"] for item in artifacts),
            "artifacts": [
                {
                    "alias": item["alias"], "filename": item["filename"],
                    "bytes": item["bytes"], "sha256": item["sha256"],
                    "pre_campaign_stat": item["stat"],
                    "post_campaign_stat": item["stat"],
                }
                for item in artifacts
            ],
            "post_campaign_cuda_runtime": json.loads(
                json.dumps(toolchain["observed_cuda_runtime"])
            ),
            "verification": "synthetic post-campaign content fixture",
        }
        # ldd output embeds ASLR-dependent addresses; the post-run probe may
        # legitimately differ only in that diagnostic stdout hash.
        post_model_verification["post_campaign_cuda_runtime"]["loader_probe"][
            "stdout_sha256"
        ] = "4" * 64
        post_path = campaign / "post-model-verification.json"
        private_json(post_path, post_model_verification)
        post_sha = sha256_file(post_path)

        for cell_index, (alias, (_filename, _artifact_sha, ngl, context)) in enumerate(
            promoter.MATRIX.items()
        ):
            log_dir = campaign / f"{alias}-server"
            log_dir.mkdir(mode=0o700)
            port = 18090 + cell_index
            model_entry = next(item for item in artifacts if item["alias"] == alias)
            model_identity = {
                "schema": "paper4-validation-live-model-identity-v1",
                "alias": alias,
                "model_manifest_sha256": manifest_sha,
                "model": model_entry,
                "verification": "synthetic stat-bound model fixture",
            }
            private_json(log_dir / "model-identity.json", model_identity)
            model_identity_sha = sha256_file(log_dir / "model-identity.json")
            startup = {
                "schema": "paper4-validation-server-identity-v1",
                "phase": "startup",
                "alias": alias,
                "pid": 12345,
                "process_start_time_ticks": "99",
                "launcher": {"path": str(launcher.resolve()), "sha256": launcher_sha},
                "model_identity_sha256": model_identity_sha,
                "model_manifest_sha256": manifest_sha,
                "endpoint": {
                    "host": "127.0.0.1", "port": port,
                    "base_url": f"http://127.0.0.1:{port}", "model_id": alias,
                },
                "request": {"context_tokens": context, "gpu_layers": ngl},
                "observed_gpu_process": {
                    "pid": 12345,
                    "used_gpu_memory_mib": 4096,
                    "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
                },
                "verification": "synthetic strict startup fixture",
            }
            completion = {
                **startup,
                "phase": "completion",
                "observed_gpu_process": {
                    "pid": 12345,
                    "used_gpu_memory_mib": 4100,
                    "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
                },
            }
            private_json(log_dir / "startup-identity.json", startup)
            private_json(log_dir / "completion-identity.json", completion)
            private_json(
                log_dir / "shutdown-identity.json",
                {
                    "schema": "paper4-validation-server-shutdown-v1",
                    "phase": "shutdown",
                    "alias": alias,
                    "pid": 12345,
                    "process_start_time_ticks": "99",
                    "host": "127.0.0.1",
                    "port": port,
                    "same_process_alive": False,
                    "port_free": True,
                    "startup_identity_sha256": sha256_file(log_dir / "startup-identity.json"),
                    "completion_identity_sha256": sha256_file(log_dir / "completion-identity.json"),
                    "verification": "synthetic strict shutdown fixture",
                },
            )
            observed_gpu = int(ngl) if ngl != "auto" else 48
            model_path = next(item["path"] for item in artifacts if item["alias"] == alias)
            placement_lines = [
                f"load_model: loading model '{model_path}'",
                f"load_tensors: offloaded {observed_gpu}/64 layers to GPU",
            ]
            private_text(log_dir / "stdout.log", "\n".join(placement_lines) + "\n")
            private_text(log_dir / "stderr.log", "server completed normally\n")

            startup_sha = sha256_file(log_dir / "startup-identity.json")
            result = campaign / f"{alias}.jsonl"
            result_rows = []
            for item in dataset_rows:
                expected = float(item["answer_number"])
                content = f"Answer: {expected}"
                parsed = promoter.parse_gsm8k_answer(content)
                result_rows.append(
                    {
                        "id": item["id"],
                        "label": alias,
                        "correct": True,
                        "wall_s": 1.0,
                        "identity_guard_wall_s": 0.1,
                        "completion_tokens": 4,
                        "thinking_tokens_estimate": 0,
                        "truncated": False,
                        "raw_response": {
                            "model": alias,
                            "choices": [
                                {
                                    "message": {"content": content},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"completion_tokens": 4},
                        },
                        "scoring_content": content,
                        "parsed_answer_number": parsed,
                        "expected_answer_number": expected,
                        "api_error": None,
                        "chat_template_retry_error": None,
                        "chat_template_kwargs_dropped": False,
                        "model_id": alias,
                        "model_manifest_sha256": manifest_sha,
                        "startup_identity_sha256": startup_sha,
                        "sampler_seed": promoter.SEED_BASE + int(item["id"].split("_", 1)[1]),
                        "dataset_sha256": promoter.DATASET_SHA256,
                        "harness_sha256": sources["harness_sha256"],
                        "server_guard_sha256": sources["server_guard_sha256"],
                        "sampler_contract_sha256": sampler_sha,
                    }
                )
            private_text(
                result,
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in result_rows
                ),
            )
            file_paths = {
                "result": result,
                "stdout": log_dir / "stdout.log",
                "stderr": log_dir / "stderr.log",
                "model_identity": log_dir / "model-identity.json",
                "startup_identity": log_dir / "startup-identity.json",
                "completion_identity": log_dir / "completion-identity.json",
                "shutdown_identity": log_dir / "shutdown-identity.json",
            }
            files = {
                key: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for key, path in file_paths.items()
            }
            cell_receipt = log_dir / "cell-receipt.json"
            private_json(
                cell_receipt,
                {
                    "schema": "paper4-validation-cell-receipt-v1",
                    "alias": alias,
                    "row_count": 100,
                    "model_manifest_sha256": manifest_sha,
                    "dataset_sha256": promoter.DATASET_SHA256,
                    "sampler_contract_sha256": sampler_sha,
                    "scores_recomputed_from_raw_response_and_pinned_gold": True,
                    "api_error_telemetry_preserved_and_validated": True,
                    "host_receipt_sha256": host_sha,
                    "shutdown_verified": True,
                    "engine_executable": {"path": str(launcher.resolve()), "sha256": launcher_sha},
                    "requested": {"context_tokens": context, "gpu_layers": ngl},
                    "observed_placement_log_lines": placement_lines,
                    "observed_offloaded_layers": {
                        "gpu": observed_gpu,
                        "total": 64,
                        "readback": f"{observed_gpu}/64",
                    },
                    "observed_gpu_process": {
                        "startup": startup["observed_gpu_process"],
                        "completion": completion["observed_gpu_process"],
                    },
                    "protocol_sources": sources,
                    "files": files,
                },
            )
            campaign_cells.append(
                {
                    "alias": alias,
                    "path": f"{alias}-server/cell-receipt.json",
                    "sha256": sha256_file(cell_receipt),
                }
            )

        private_json(
            campaign / "campaign-receipt.json",
            {
                "schema": "paper4-validation-campaign-receipt-v1",
                "model_manifest_sha256": manifest_sha,
                "cell_count": 6,
                "row_count": 600,
                "dataset_sha256": promoter.DATASET_SHA256,
                "sampler_contract_sha256": sampler_sha,
                "protocol_sources": sources,
                "prospective_plan": {
                    "path": plan_path.name,
                    "sha256": sha256_file(plan_path),
                    "git_commit": plan["git_commit"],
                },
                "prospective_plan_approval": {
                    "path": approval_path.name,
                    "sha256": sha256_file(approval_path),
                    "actor": approval["actor"],
                    "plan_sha256": sha256_file(plan_path),
                },
                "host_receipt": {
                    "path": host_path.name,
                    "sha256": host_sha,
                },
                "post_model_verification": {
                    "path": post_path.name,
                    "sha256": post_sha,
                },
                "toolchain_receipt": {
                    "path": toolchain_path.name,
                    "sha256": sha256_file(toolchain_path),
                    "executable_sha256": launcher_sha,
                    "llama_cpp_commit": "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3",
                },
                "cells": sorted(campaign_cells, key=lambda item: item["alias"]),
            },
        )
        destination = root / "promoted"
        result = promoter.validate_and_promote(campaign, manifest, dataset, destination)
        assert destination.stat().st_mode & 0o777 == 0o700
        assert (destination / "private").stat().st_mode & 0o777 == 0o700
        assert all(
            path.stat().st_mode & 0o777 == 0o600
            for path in (destination / "private").rglob("*")
            if path.is_file()
        )
        checked = checker.check(destination / "public-candidate")
        assert checked["status"] == "pass" and checked["row_count"] == 600
        claims, tex, report = builder.build(
            destination, result["promotion_receipt_sha256"]
        )
        assert claims["p4r1_validation"]["status"] == "promoted-private-evidence"
        assert r"\newcommand{\PfourQwenHealthyContextTokens}{4096}" in tex
        assert r"\newcommand{\PfourQwenPlacementAuditable}{yes}" in tex
        assert r"\newcommand{\PfourMixtralIqOneSignedDirection}{the two tiers tied}" in tex
        assert "P4R1 is exploratory, not independent" in report

        # A signed reversal must not pass a stale Q4-preservation/low-bit-loss
        # narrative merely because its absolute effect size is large.
        reversal = json.loads(json.dumps(claims))
        reversal_signature = reversal["outcome_signature"]
        reversal_signature["cell_correct"]["P4R1-MIXTRAL-IQ1M"] = 100
        reversal_signature["cell_correct"]["P4R1-MIXTRAL-Q4KM"] = 90
        reversal_signature["q4_minus_low_delta_pp"]["mixtral_iq1_vs_q4"] = -10.0
        reversal_signature["signed_result_branch"]["mixtral_iq1_vs_q4"] = "low_tier_higher"
        signed_macros = " ".join(
            values[2] for values in manuscript_gate.SIGNED_OUTCOMES.values()
        )
        stale_reversal_text = (
            signed_macros
            + ". Mixtral IQ1 shows low-bit collapse relative to a Q4 anchor that preserves quality."
        )
        try:
            manuscript_gate.validate_signed_outcome_narrative(
                reversal, stale_reversal_text, signed_macros, "P4R1 corrective not independent"
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("signed reversal bypassed the stale-narrative guard")

        # Synthetic all-100 outcomes must never flow straight into a preview.
        # The exact claims/outcome signature requires a separate post-result
        # author review of the exact paper, explainer, and brief bytes.
        synthetic_claims_path = root / "synthetic-claims.json"
        private_json(synthetic_claims_path, claims)
        try:
            manuscript_gate.check(
                synthetic_claims_path, sha256_file(synthetic_claims_path),
                root / "missing-narrative-review.json", "0" * 64,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("all-100 synthetic outcomes bypassed narrative review")

        # Even if an attacker updates both receipt hashes after editing the
        # plan, promotion must independently cross-bind plan source bytes to
        # the source hashes recorded by every cell.
        original_plan = plan_path.read_bytes()
        original_approval = approval_path.read_bytes()
        original_campaign = (campaign / "campaign-receipt.json").read_bytes()
        tampered_plan = json.loads(original_plan)
        for item in tampered_plan["inference_source_inventory"]:
            if item["path"].endswith("run_gsm8k.py"):
                item["sha256"] = "0" * 64
        private_json(plan_path, tampered_plan)
        tampered_approval = json.loads(original_approval)
        tampered_approval["plan_sha256"] = sha256_file(plan_path)
        private_json(approval_path, tampered_approval)
        tampered_campaign = json.loads(original_campaign)
        tampered_campaign["prospective_plan"]["sha256"] = sha256_file(plan_path)
        tampered_campaign["prospective_plan_approval"].update(
            {
                "sha256": sha256_file(approval_path),
                "plan_sha256": sha256_file(plan_path),
            }
        )
        private_json(campaign / "campaign-receipt.json", tampered_campaign)
        try:
            promoter.validate_and_promote(
                campaign, manifest, dataset, root / "tampered-plan-promotion"
            )
        except promoter.PromotionError:
            pass
        else:
            raise AssertionError("promotion accepted a plan/cell source-hash divergence")
        plan_path.write_bytes(original_plan)
        plan_path.chmod(0o600)
        approval_path.write_bytes(original_approval)
        approval_path.chmod(0o600)
        (campaign / "campaign-receipt.json").write_bytes(original_campaign)
        (campaign / "campaign-receipt.json").chmod(0o600)

        public_rows = destination / "public-candidate/primary_gsm8k.jsonl"
        public_rows.chmod(0o600)
        lines = public_rows.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first.pop("api_error_recorded")
        lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        public_rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            checker.check(destination / "public-candidate")
        except RuntimeError:
            pass
        else:
            raise AssertionError("public checker accepted a row with a missing required key")

    print("P4R1 SYNTHETIC PROMOTION REGRESSION PASSED")
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
