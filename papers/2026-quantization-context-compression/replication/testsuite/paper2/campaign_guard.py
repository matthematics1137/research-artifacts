#!/usr/bin/env python3
"""Static, thermal, and placement receipts for the fail-closed P2R1 campaign.

This module never sends an inference request.  It writes private, exclusive
receipts that the launcher and post-run validator can require.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from identity_guard import (  # noqa: E402
    descends_from,
    load_artifact_record,
    sha256_file,
    utc_now,
)
from protocol_bindings import require_committed_clean, source_hashes  # noqa: E402
from r1_safety import RunSafetyError, require_attempt_id, write_exclusive  # noqa: E402
from runtime_provenance import (  # noqa: E402
    exl3_private_runtime_layout,
    exl3_process_cuda_mapping_identity,
    llama_build_identity,
    llama_cuda_loader_environment,
    llama_cuda_runtime_identity,
    llama_process_cuda_runtime_identity,
    python_runtime_identity,
    sanitized_no_ld_environment,
    sanitize_llama_cuda_runtime,
    validate_exl3_process_cuda_mapping_identity,
)


LLAMA_REVISION = "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3"
TABBY_REVISION = "4a4f9f44820303593844f092d424bb7506008733"
MIN_MEM_AVAILABLE_GIB = 32.0
GPU_UTILIZATION_POLICY = "record-only-device-total"
# Smallest per-process GPU allocation that proves the EXL3 tier is resident on
# the GPU.  Measured 2026-09-13 for the frozen 8192-token Q8-cache deployment:
# 7,666 MiB at load, 7,706 MiB after a 2,048-token completion.  The original
# 8,000 MiB figure was an unmeasured estimate that no attempt had reached.
EXL3_MIN_RESIDENT_ALLOCATION_MIB = 7000.0
EXPECTED_HOST = {
    "cpu_model": "Intel(R) Core(TM) i9-14900HX",
    "logical_cpus": 32,
    "mem_total_bytes": 67016003584,
    "gpu_name": "NVIDIA GeForce RTX 4080 Laptop GPU",
    "gpu_memory_total_mib": 12282.0,
    "gpu_current_power_limit_w": 80.0,
    "gpu_requested_power_limit_w": 80.0,
    "gpu_default_power_limit_w": 80.0,
    "nvidia_driver": "580.167.08",
}


def command_output(
    argv: list[str], *, required: bool = True,
    env: dict[str, str] | None = None,
) -> str | None:
    try:
        return subprocess.run(
            argv, check=required, capture_output=True, text=True, timeout=30,
            env=env,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if required:
            raise RunSafetyError(f"required command failed: {argv!r}")
        return None


def command_combined(
    argv: list[str], *, env: dict[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            argv, check=True, capture_output=True, text=True, timeout=30, env=env,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RunSafetyError(f"required command failed: {argv!r}") from exc
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def git_revision(path: Path) -> str:
    return command_output(["git", "-C", str(path), "rev-parse", "HEAD"]) or ""


def require_clean_repo(path: Path) -> None:
    dirty = command_output(["git", "-C", str(path), "status", "--porcelain"]) or ""
    if dirty:
        raise RunSafetyError(f"engine repository is not clean: {path}")


def host_inventory(attempt_id: str) -> dict:
    require_attempt_id(attempt_id)
    cpu_models = {
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    }
    if len(cpu_models) != 1:
        raise RunSafetyError(f"host CPU model is ambiguous: {sorted(cpu_models)}")
    memory = memory_snapshot()
    query = command_output([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,power.default_limit,driver_version",
        "--format=csv,noheader,nounits",
    ], required=True) or ""
    lines = [line for line in query.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RunSafetyError(f"P2R1 requires exactly one GPU; observed {len(lines)}")
    fields = [part.strip() for part in lines[0].split(",")]
    if len(fields) != 5:
        raise RunSafetyError(f"malformed host GPU inventory: {lines[0]!r}")
    try:
        gpu_memory = float(fields[2])
        default_power_limit = float(fields[3])
    except ValueError as exc:
        raise RunSafetyError(f"non-numeric host GPU inventory: {lines[0]!r}") from exc
    power_report = command_combined(["nvidia-smi", "-q", "-d", "POWER"])
    power_values: dict[str, float] = {}
    for key, label in (
        ("gpu_current_power_limit_w", "Current Power Limit"),
        ("gpu_requested_power_limit_w", "Requested Power Limit"),
        ("gpu_default_power_limit_w", "Default Power Limit"),
    ):
        match = re.search(rf"^\s*{re.escape(label)}\s*:\s*([0-9.]+)\s+W\s*$",
                          power_report, flags=re.M)
        if not match:
            raise RunSafetyError(f"host power report lacks {label!r}")
        power_values[key] = float(match.group(1))
    if power_values["gpu_default_power_limit_w"] != default_power_limit:
        raise RunSafetyError("nvidia-smi default power-limit readbacks disagree")
    observed = {
        "cpu_model": next(iter(cpu_models)),
        "logical_cpus": os.cpu_count(),
        "mem_total_bytes": memory.get("MemTotal"),
        "gpu_name": fields[0],
        "gpu_memory_total_mib": gpu_memory,
        **power_values,
        "nvidia_driver": fields[4],
    }
    if observed != EXPECTED_HOST:
        raise RunSafetyError(
            f"campaign host differs from the frozen P2R1 machine: {observed}"
        )
    try:
        machine_id = Path("/etc/machine-id").read_text().strip()
    except OSError as exc:
        raise RunSafetyError("cannot read stable machine identifier") from exc
    if not machine_id:
        raise RunSafetyError("stable machine identifier is empty")
    private_binding = {
        "scheme": "sha256 domain-separated by P2R1 attempt ID",
        "gpu_uuid_sha256": sha256_text(
            f"paper2-r1|{attempt_id}|gpu|{fields[1]}"
        ),
        "machine_id_sha256": sha256_text(
            f"paper2-r1|{attempt_id}|machine|{machine_id}"
        ),
    }
    return {
        "schema": "paper2-r1-host-inventory-v1",
        "attempt_id": attempt_id,
        "captured_at": utc_now(),
        **observed,
        "private_host_binding": private_binding,
    }


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bind_attempt_host(
    *, attempt_id: str, host_receipt: Path, output: Path,
) -> None:
    require_attempt_id(attempt_id)
    host = json.loads(host_receipt.read_text())
    binding = host.get("private_host_binding")
    if (
        host.get("schema") != "paper2-r1-host-inventory-v1"
        or host.get("attempt_id") != attempt_id
        or not isinstance(binding, dict)
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(binding.get(key, "")))
               for key in ("gpu_uuid_sha256", "machine_id_sha256"))
    ):
        raise RunSafetyError("host receipt lacks a valid private attempt binding")
    value = {
        "schema": "paper2-r1-private-attempt-host-binding-v1",
        "attempt_id": attempt_id,
        "private_host_binding": binding,
    }
    if output.exists():
        prior = json.loads(output.read_text())
        if prior != value:
            raise RunSafetyError(
                "P2R1 attempt crossed machines; preserve it and require a committed "
                "protocol amendment before any replacement attempt"
            )
    else:
        write_exclusive(output, value)


def make_toolchain_receipt(
    output: Path, host_receipt: Path, attempt_host_binding: Path,
) -> None:
    require_committed_clean(ROOT)
    host = json.loads(host_receipt.read_text())
    if (
        host.get("schema") != "paper2-r1-host-inventory-v1"
        or {key: host.get(key) for key in EXPECTED_HOST} != EXPECTED_HOST
        or not isinstance(host.get("private_host_binding"), dict)
    ):
        raise RunSafetyError("toolchain host receipt is malformed or unexpected")
    bound_host = json.loads(attempt_host_binding.read_text())
    if (
        bound_host.get("schema") != "paper2-r1-private-attempt-host-binding-v1"
        or bound_host.get("attempt_id") != host.get("attempt_id")
        or bound_host.get("private_host_binding") != host.get("private_host_binding")
    ):
        raise RunSafetyError("toolchain host differs from immutable attempt host")
    llama_repo = ROOT / "engines" / "llama.cpp"
    tabby_repo = ROOT / "engines" / "tabbyAPI"
    llama_server = llama_repo / "build" / "bin" / "llama-server"
    exl_python = ROOT / "engines" / "exl3-env" / "bin" / "python"
    observed_llama = git_revision(llama_repo)
    observed_tabby = git_revision(tabby_repo)
    if observed_llama != LLAMA_REVISION or observed_tabby != TABBY_REVISION:
        raise RunSafetyError(
            "engine revision changed: "
            f"llama.cpp={observed_llama}, tabbyAPI={observed_tabby}"
        )
    require_clean_repo(llama_repo)
    require_clean_repo(tabby_repo)
    for path in (llama_server, exl_python):
        if not path.exists():
            raise RunSafetyError(f"required executable is missing: {path}")
    package_probe = (
        "import importlib.metadata as m,json,torch; "
        "print(json.dumps({'torch':torch.__version__,"
        "'exllamav3':m.version('exllamav3'),'tabbyAPI':m.version('tabbyAPI'),"
        "'uvicorn':m.version('uvicorn'),'fastapi':m.version('fastapi')}))"
    )
    exl_loader_env = sanitized_no_ld_environment()
    packages = json.loads(command_output(
        [str(exl_python), "-c", package_probe], env=exl_loader_env,
    ) or "{}")
    expected_packages = {
        "torch": "2.10.0+cu128",
        "exllamav3": "1.4.3+cu128.torch2.10.0",
        "tabbyAPI": "0.0.1",
        "uvicorn": "0.52.4",
        "fastapi": "0.141.1",
    }
    if packages != expected_packages:
        raise RunSafetyError(f"EXL3 runtime changed: {packages}")
    exl_layout = exl3_private_runtime_layout(exl_python)
    exl_installed_content = python_runtime_identity(exl_python)
    protocol = ROOT / "paper2" / "R1_PROTOCOL.json"
    methodology = ROOT / "methodology" / "EMPIRICAL_RESEARCH_LOOP.md"
    _cuda_library_dir, llama_loader_env = llama_cuda_loader_environment(llama_server)
    server_version = command_combined(
        [str(llama_server), "--version"], env=llama_loader_env,
    )
    if "commit 035e227" not in server_version:
        raise RunSafetyError(f"unexpected llama-server version output: {server_version}")
    llama_build = llama_build_identity(llama_server)
    receipt = {
        "schema": "paper2-r1-toolchain-v1",
        "created_at": utc_now(),
        "llama_cpp": {
            "revision": observed_llama,
            "server": str(llama_server.resolve()),
            "server_sha256": sha256_file(llama_server),
            "server_version": server_version,
            "repository_clean": True,
            "build_identity": llama_build,
            "cuda_runtime": llama_cuda_runtime_identity(
                llama_server, loader_env=llama_loader_env,
            ),
        },
        "tabbyapi": {
            "revision": observed_tabby,
            "root": str(tabby_repo.resolve()),
            "repository_clean": True,
        },
        "exl3_runtime": {
            # Preserve the virtual-environment entry point.  Resolving this
            # symlink would record /usr/bin/python and lose venv selection.
            "python": str(exl_python.absolute()),
            "python_sha256": sha256_file(exl_python.resolve()),
            **packages,
            "launch_loader_contract": (
                "all inherited LD_* variables removed; no LD_LIBRARY_PATH"
            ),
            "private_runtime_layout": exl_layout,
            "installed_content": exl_installed_content,
        },
        "cuda_compiler": llama_build["tool_versions"]["nvcc"],
        "nvidia_driver": command_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            required=True,
        ),
        "protocol": str(protocol.resolve()),
        "protocol_sha256": sha256_file(protocol),
        "methodology": str(methodology.resolve()),
        "methodology_sha256": sha256_file(methodology),
        "recovery_harness_sha256": source_hashes(ROOT),
        "host_receipt": str(host_receipt.resolve()),
        "host_receipt_sha256": sha256_file(host_receipt),
        "attempt_host_binding": str(attempt_host_binding.resolve()),
        "attempt_host_binding_sha256": sha256_file(attempt_host_binding),
    }
    write_exclusive(output, receipt)


def memory_snapshot() -> dict:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        if key in {"MemTotal", "MemFree", "MemAvailable", "Cached", "Buffers", "SwapFree"}:
            values[key] = int(rest.split()[0]) * 1024
    return values


def gpu_snapshot() -> list[dict]:
    query = (
        "index,name,uuid,temperature.gpu,power.draw,memory.used,memory.total,"
        "utilization.gpu,clocks.current.graphics"
    )
    text = command_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        required=True,
    ) or ""
    keys = [
        "index", "name", "uuid", "temperature_c", "power_w", "memory_used_mib",
        "memory_total_mib", "utilization_percent", "graphics_clock_mhz",
    ]
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(keys):
            raise RunSafetyError(f"malformed nvidia-smi GPU row: {line!r}")
        row = dict(zip(keys, parts))
        try:
            row["index"] = int(row["index"])
            for key in (
                "temperature_c", "power_w", "memory_used_mib",
                "memory_total_mib", "utilization_percent", "graphics_clock_mhz",
            ):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"non-finite {key}")
            if not 0.0 <= row["utilization_percent"] <= 100.0:
                raise ValueError("GPU utilization is outside [0, 100]")
        except ValueError as exc:
            raise RunSafetyError(f"non-numeric nvidia-smi GPU row: {line!r}") from exc
        rows.append(row)
    return rows


def cpu_temperatures() -> list[dict]:
    rows = []
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            value = int(path.read_text().strip()) / 1000.0
            kind = (path.parent / "type").read_text().strip()
        except (FileNotFoundError, ValueError, PermissionError):
            continue
        rows.append({"zone": path.parent.name, "type": kind, "temperature_c": value})
    return rows


def snapshot(stage: str) -> dict:
    one, five, fifteen = os.getloadavg()
    return {
        "schema": "paper2-r1-host-state-v1",
        "captured_at": utc_now(),
        "stage": stage,
        "load_average": {"one_min": one, "five_min": five, "fifteen_min": fifteen},
        "memory": memory_snapshot(),
        "gpus": gpu_snapshot(),
        "cpu_thermal_zones": cpu_temperatures(),
    }


def cool_sample_admissible(
    state: dict, *, max_gpu_temp: float, max_load: float,
    min_available_gib: float, max_gpu_memory_mib: float,
) -> bool:
    gpus = state.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return False
    try:
        gpu_temps = [float(row["temperature_c"]) for row in gpus]
        gpu_utils = [float(row["utilization_percent"]) for row in gpus]
        gpu_memory = [float(row["memory_used_mib"]) for row in gpus]
        one_min_load = float(state["load_average"]["one_min"])
        available = float(state["memory"]["MemAvailable"]) / (1024 ** 3)
    except (KeyError, TypeError, ValueError):
        return False
    numeric = gpu_temps + gpu_utils + gpu_memory + [one_min_load, available]
    if not all(math.isfinite(value) for value in numeric):
        return False
    return (
        max(gpu_temps) <= max_gpu_temp
        and all(0.0 <= value <= 100.0 for value in gpu_utils)
        and max(gpu_memory) <= max_gpu_memory_mib
        and one_min_load <= max_load
        and available >= min_available_gib
        and state.get("compute_process_pids") == []
    )


def parse_compute_process_pids(text: str) -> list[int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(not re.fullmatch(r"[0-9]+", line) for line in lines):
        raise RunSafetyError("nvidia-smi compute-process output is malformed")
    return [int(line) for line in lines]


def cool_and_record(
    *, stage: str, output: Path, max_gpu_temp: float, max_load: float,
    min_available_gib: float, max_gpu_memory_mib: float,
    gpu_utilization_policy: str, consecutive_samples: int, timeout_s: int,
) -> None:
    if gpu_utilization_policy != GPU_UTILIZATION_POLICY:
        raise RunSafetyError(
            f"unknown GPU-utilization policy: {gpu_utilization_policy}"
        )
    deadline = time.monotonic() + timeout_s
    passing: list[dict] = []
    while True:
        state = snapshot(stage)
        compute_processes = command_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            required=True,
        )
        state["compute_process_pids"] = parse_compute_process_pids(
            compute_processes or ""
        )
        cool = cool_sample_admissible(
            state, max_gpu_temp=max_gpu_temp, max_load=max_load,
            min_available_gib=min_available_gib,
            max_gpu_memory_mib=max_gpu_memory_mib,
        )
        if cool:
            passing.append(state)
            if len(passing) >= consecutive_samples:
                write_exclusive(output, {
                    "schema": "paper2-r1-cool-gate-v2",
                    "stage": stage,
                    "passed_at": utc_now(),
                    "thresholds": {
                        "max_gpu_temperature_c": max_gpu_temp,
                        "gpu_utilization_policy": gpu_utilization_policy,
                        "max_gpu_memory_used_mib": max_gpu_memory_mib,
                        "max_one_min_load": max_load,
                        "min_mem_available_gib": min_available_gib,
                        "foreign_compute_processes": 0,
                        "consecutive_samples": consecutive_samples,
                        "sample_interval_seconds": 10,
                    },
                    "samples": passing,
                })
                return
        else:
            passing = []
        if time.monotonic() >= deadline:
            raise RunSafetyError(
                f"host did not reach frozen cool-state gate for {stage}: {state}"
            )
        time.sleep(10)


def exact_llama_offload_readback(text: str, expected_layers: str | None) -> str:
    """Return one exact llama.cpp placement readback or fail closed."""
    if not isinstance(expected_layers, str) or not re.fullmatch(
        r"\d+/\d+", expected_layers
    ):
        raise RunSafetyError("llama.cpp placement expectation is malformed")
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text)
    if len(matches) != 1:
        raise RunSafetyError(
            f"expected one llama.cpp placement readback; got {matches}"
        )
    observed = f"{matches[0][0]}/{matches[0][1]}"
    if observed != expected_layers:
        raise RunSafetyError(
            f"GPU placement changed: expected {expected_layers}, got {observed}"
        )
    return observed


def placement_receipt(
    *, engine: str, log: Path, identity_receipt: Path, output: Path,
    expected_layers: str | None, toolchain_receipt: Path,
) -> None:
    text = log.read_text(errors="replace")
    identity = json.loads(identity_receipt.read_text())
    toolchain = json.loads(toolchain_receipt.read_text())
    if toolchain.get("schema") != "paper2-r1-toolchain-v1":
        raise RunSafetyError("placement lacks a valid toolchain receipt")
    if engine == "llama.cpp":
        observed = exact_llama_offload_readback(text, expected_layers)
        expected_runtime = toolchain.get("llama_cpp", {}).get("cuda_runtime")
        if (
            toolchain.get("schema") != "paper2-r1-toolchain-v1"
            or not isinstance(expected_runtime, dict)
        ):
            raise RunSafetyError("placement lacks frozen CUDA runtime evidence")
        loaded_runtime = llama_process_cuda_runtime_identity(
            int(identity["launched_pid"]), expected_runtime,
        )
        detail = {
            "offloaded_layers": observed,
            "toolchain_receipt": str(toolchain_receipt.resolve()),
            "toolchain_receipt_sha256": sha256_file(toolchain_receipt),
            "llama_cuda_process_runtime": loaded_runtime,
        }
    else:
        artifact_path = identity["artifact"]["path"]
        normalized = " ".join(text.split())
        if f"Loading model: {artifact_path}" not in normalized:
            raise RunSafetyError("tabbyAPI log does not bind the loaded artifact path")
        if "Loading with a manual GPU split (or a one GPU setup)" not in normalized:
            raise RunSafetyError("tabbyAPI did not report its GPU load path")
        exl_runtime = toolchain.get("exl3_runtime", {})
        layout = exl_runtime.get("private_runtime_layout", {})
        installed = exl_runtime.get("installed_content", {})
        if (
            layout.get("schema") != "paper2-r1-exl3-private-runtime-layout-v1"
            or exl_runtime.get("launch_loader_contract")
            != "all inherited LD_* variables removed; no LD_LIBRARY_PATH"
            or installed.get("observed_ld_variables") != {}
        ):
            raise RunSafetyError("placement lacks the frozen LD-free EXL3 runtime")
        loaded_runtime = exl3_process_cuda_mapping_identity(
            int(identity["launched_pid"]), Path(layout.get("site_packages_root", "")),
        )
        expected_extension = installed.get("precompiled_extension", {})
        validate_exl3_process_cuda_mapping_identity(
            loaded_runtime,
            expected_root_pid=int(identity["launched_pid"]),
            expected_site_packages_root=Path(layout.get("site_packages_root", "")),
            expected_extension=expected_extension,
        )
        mapped_extensions = [
            value for value in loaded_runtime["mapped_libraries"]
            if value["filename"].startswith("exllamav3_ext")
        ]
        if (
            len(mapped_extensions) != 1
            or any(
                mapped_extensions[0].get(key) != expected_extension.get(key)
                for key in ("filename", "bytes", "sha256")
            )
        ):
            raise RunSafetyError(
                "live EXL3 extension differs from frozen installed content"
            )
        detail = {
            "backend": "ExLlamaV3",
            "loaded_artifact": artifact_path,
            "placement_interpretation": "GPU-only ExLlamaV3 load; no CPU weight fallback",
            "toolchain_receipt": str(toolchain_receipt.resolve()),
            "toolchain_receipt_sha256": sha256_file(toolchain_receipt),
            "exl3_cuda_process_runtime": loaded_runtime,
        }
    processes = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        required=True,
    )
    process_rows = []
    for line in (processes or "").splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        try:
            memory_mib = float(fields[1])
            if not math.isfinite(memory_mib) or memory_mib < 0:
                raise ValueError("invalid GPU allocation")
            process_rows.append({"pid": int(fields[0]), "used_memory_mib": memory_mib})
        except ValueError:
            continue
    launched_pid = int(identity["launched_pid"])
    matched_processes = [
        row for row in process_rows if descends_from(row["pid"], launched_pid)
    ]
    if not matched_processes:
        raise RunSafetyError(
            "nvidia-smi did not bind GPU allocation to the launched process tree"
        )
    # The frozen 8192-token Q8-cache EXL3 deployment allocates 7,666 MiB at load
    # and 7,706 MiB after a 2,048-token completion on this driver (measured
    # 2026-09-13 under the private tabby.yml; the earlier 8,000 MiB figure was
    # an unmeasured estimate that no attempt had ever reached).  The bound
    # still rejects a non-resident or wrong-size load; the exact allocation is
    # recorded below either way.
    if engine != "llama.cpp" and max(
        row["used_memory_mib"] for row in matched_processes
    ) < EXL3_MIN_RESIDENT_ALLOCATION_MIB:
        raise RunSafetyError("EXL3 GPU allocation is too small for the frozen full-GPU tier")
    log_bytes = log.read_bytes()
    write_exclusive(output, {
        "schema": "paper2-r1-placement-v1",
        "captured_at": utc_now(),
        "engine": engine,
        "identity_receipt": str(identity_receipt.resolve()),
        "identity_receipt_sha256": sha256_file(identity_receipt),
        "server_log": str(log.resolve()),
        "server_log_prefix_bytes": len(log_bytes),
        "server_log_prefix_sha256": __import__("hashlib").sha256(log_bytes).hexdigest(),
        "gpu_compute_processes": process_rows,
        "launched_tree_gpu_processes": matched_processes,
        **detail,
    })


def verify_artifacts_after_campaign(
    *, q4_record: Path, exl3_record: Path, iq2_record: Path, output: Path
) -> None:
    verified = {}
    for tier, path in (
        ("Q4KXL", q4_record), ("EXL3", exl3_record), ("IQ2S", iq2_record)
    ):
        record = load_artifact_record(path, full_hash=True)
        verified[tier] = {
            "artifact_record": str(path.resolve()),
            "artifact_record_sha256": sha256_file(path),
            "artifact": record["artifact"],
            "full_hash_verified": True,
        }
    write_exclusive(output, {
        "schema": "paper2-r1-post-campaign-artifact-verification-v1",
        "verified_at": utc_now(),
        "timing_relation": "after all model stages, timed requests, and shutdowns",
        "artifacts": verified,
    })


def verify_runtime_after_campaign(*, toolchain_receipt: Path, output: Path) -> None:
    """Re-resolve and hash the pinned llama.cpp CUDA runtime after timing."""
    toolchain = json.loads(toolchain_receipt.read_text())
    if toolchain.get("schema") != "paper2-r1-toolchain-v1":
        raise RunSafetyError("post-campaign runtime check lacks a valid toolchain receipt")
    server = Path(toolchain.get("llama_cpp", {}).get("server", ""))
    before = toolchain.get("llama_cpp", {}).get("cuda_runtime")
    if not isinstance(before, dict):
        raise RunSafetyError("pre-campaign CUDA runtime identity is missing")
    after = llama_cuda_runtime_identity(server)
    if after != before:
        raise RunSafetyError("llama.cpp CUDA runtime changed during the campaign")
    write_exclusive(output, {
        "schema": "paper2-r1-post-campaign-runtime-verification-v1",
        "verified_at": utc_now(),
        "timing_relation": "after all model stages, timed requests, and shutdowns",
        "toolchain_receipt": str(toolchain_receipt.resolve()),
        "toolchain_receipt_sha256": sha256_file(toolchain_receipt),
        "pre_campaign_cuda_runtime": before,
        "post_campaign_cuda_runtime": after,
        "unchanged_from_pre_campaign": True,
        "public_cuda_runtime": sanitize_llama_cuda_runtime(after),
    })


def record_attempt_namespace(
    *, attempt_id: str, result_root: Path, question_root: Path,
    host_binding: Path, output: Path,
) -> None:
    require_attempt_id(attempt_id)
    protocol = json.loads((ROOT / "paper2" / "R1_PROTOCOL.json").read_text())
    if attempt_id != protocol.get("official_attempt_id"):
        raise RunSafetyError(
            "attempt ID differs from the committed protocol; a retry requires "
            "an explicit protocol amendment before any new inference"
        )
    expected_result = (
        ROOT / "testsuite" / "evals" / "results" / "paper2-r1" / attempt_id
    ).resolve()
    expected_questions = (
        HERE / "r1" / "private" / "attempts" / attempt_id / "questions"
    ).resolve()
    if result_root.resolve() != expected_result or question_root.resolve() != expected_questions:
        raise RunSafetyError("attempt namespace paths are not canonical")
    expected_host_binding = expected_result / "HOST_BINDING.json"
    if host_binding.resolve() != expected_host_binding or not host_binding.is_file():
        raise RunSafetyError("attempt host binding is missing or noncanonical")
    bound_host = json.loads(host_binding.read_text())
    if (
        bound_host.get("schema") != "paper2-r1-private-attempt-host-binding-v1"
        or bound_host.get("attempt_id") != attempt_id
    ):
        raise RunSafetyError("attempt host binding is malformed")
    write_exclusive(output, {
        "schema": "paper2-r1-attempt-namespace-v1",
        "attempt_id": attempt_id,
        "result_root": str(expected_result),
        "question_root": str(expected_questions),
        "private_host_binding": str(expected_host_binding),
        "private_host_binding_sha256": sha256_file(expected_host_binding),
    })


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="command", required=True)
    toolchain = sub.add_parser("toolchain")
    toolchain.add_argument("--output", type=Path, required=True)
    toolchain.add_argument("--host-receipt", type=Path, required=True)
    toolchain.add_argument("--attempt-host-binding", type=Path, required=True)
    host = sub.add_parser("host")
    host.add_argument("--output", type=Path, required=True)
    host.add_argument("--attempt-id", required=True)
    bind_host = sub.add_parser("bind-host")
    bind_host.add_argument("--attempt-id", required=True)
    bind_host.add_argument("--host-receipt", type=Path, required=True)
    bind_host.add_argument("--output", type=Path, required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--stage", required=True)
    snap.add_argument("--output", type=Path, required=True)
    cool = sub.add_parser("cool")
    cool.add_argument("--stage", required=True)
    cool.add_argument("--output", type=Path, required=True)
    cool.add_argument("--max-gpu-temp", type=float, default=50.0)
    cool.add_argument("--max-load", type=float, default=2.0)
    cool.add_argument(
        "--min-available-gib", type=float, default=MIN_MEM_AVAILABLE_GIB,
    )
    cool.add_argument(
        "--gpu-utilization-policy", choices=(GPU_UTILIZATION_POLICY,),
        required=True,
    )
    cool.add_argument("--max-gpu-memory-mib", type=float, default=1024.0)
    cool.add_argument("--consecutive-samples", type=int, default=3)
    cool.add_argument("--timeout", type=int, default=1800)
    placement = sub.add_parser("placement")
    placement.add_argument("--engine", choices=["llama.cpp", "ExLlamaV3/tabbyAPI"], required=True)
    placement.add_argument("--log", type=Path, required=True)
    placement.add_argument("--identity-receipt", type=Path, required=True)
    placement.add_argument("--output", type=Path, required=True)
    placement.add_argument("--expected-layers")
    placement.add_argument("--toolchain-receipt", type=Path, required=True)
    post_artifacts = sub.add_parser("post-artifacts")
    post_artifacts.add_argument("--q4-record", type=Path, required=True)
    post_artifacts.add_argument("--exl3-record", type=Path, required=True)
    post_artifacts.add_argument("--iq2-record", type=Path, required=True)
    post_artifacts.add_argument("--output", type=Path, required=True)
    post_runtime = sub.add_parser("post-runtime")
    post_runtime.add_argument("--toolchain-receipt", type=Path, required=True)
    post_runtime.add_argument("--output", type=Path, required=True)
    attempt = sub.add_parser("attempt")
    attempt.add_argument("--attempt-id", required=True)
    attempt.add_argument("--result-root", type=Path, required=True)
    attempt.add_argument("--question-root", type=Path, required=True)
    attempt.add_argument("--host-binding", type=Path, required=True)
    attempt.add_argument("--output", type=Path, required=True)
    return top


def main() -> int:
    args = parser().parse_args()
    if args.command == "toolchain":
        make_toolchain_receipt(
            args.output, args.host_receipt, args.attempt_host_binding,
        )
    elif args.command == "host":
        write_exclusive(args.output, host_inventory(args.attempt_id))
    elif args.command == "bind-host":
        bind_attempt_host(
            attempt_id=args.attempt_id, host_receipt=args.host_receipt,
            output=args.output,
        )
    elif args.command == "snapshot":
        write_exclusive(args.output, snapshot(args.stage))
    elif args.command == "cool":
        cool_and_record(
            stage=args.stage, output=args.output, max_gpu_temp=args.max_gpu_temp,
            max_load=args.max_load, min_available_gib=args.min_available_gib,
            max_gpu_memory_mib=args.max_gpu_memory_mib,
            gpu_utilization_policy=args.gpu_utilization_policy,
            consecutive_samples=args.consecutive_samples,
            timeout_s=args.timeout,
        )
    elif args.command == "placement":
        placement_receipt(
            engine=args.engine, log=args.log, identity_receipt=args.identity_receipt,
            output=args.output, expected_layers=args.expected_layers,
            toolchain_receipt=args.toolchain_receipt,
        )
    elif args.command == "post-artifacts":
        verify_artifacts_after_campaign(
            q4_record=args.q4_record,
            exl3_record=args.exl3_record,
            iq2_record=args.iq2_record,
            output=args.output,
        )
    elif args.command == "post-runtime":
        verify_runtime_after_campaign(
            toolchain_receipt=args.toolchain_receipt, output=args.output,
        )
    elif args.command == "attempt":
        record_attempt_namespace(
            attempt_id=args.attempt_id,
            result_root=args.result_root,
            question_root=args.question_root,
            host_binding=args.host_binding,
            output=args.output,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 CAMPAIGN GUARD ABORTED: {exc}") from exc
