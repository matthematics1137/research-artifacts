#!/usr/bin/env python3
"""Content-level build/runtime provenance for the P2R1 engines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from identity_guard import sha256_file
from r1_safety import RunSafetyError


CUDA_RUNTIME_CONTRACT = {
    "libcudart.so.12": {
        "filename": "libcudart.so.12.6.77",
        "bytes": 716128,
        "sha256": "095296727cfb5f51e5b3ae69a3cb88cbd7b6592f314ace0fdfc4e1d758c2aa85",
    },
    "libcublas.so.12": {
        "filename": "libcublas.so.12.6.4.1",
        "bytes": 108244960,
        "sha256": "d9343511e1d2ed51e4d65fa94a25cb8f73b0ebc5ae5487da794e18e6950c98dc",
    },
    "libcublasLt.so.12": {
        "filename": "libcublasLt.so.12.6.4.1",
        "bytes": 491106832,
        "sha256": "a4ddaed1410acc604815f0e84cdd6c4a70c61cfd2b95dc96360f0799169f0374",
    },
}

CUDA_LIBRARY_FAMILY = re.compile(
    r"^lib(?:cudart|cublas|cublasLt)\.so(?:\..+)?$"
)

# This is deliberately described as a classifier, not a universal inventory of
# every library that could ever participate in CUDA.  It covers the CUDA,
# PyTorch-CUDA, and ExLlama shared-library families used by the frozen P2R1
# environment.  Live evidence reports the classifier scope verbatim.
EXL_CUDA_USERSPACE_FAMILY = re.compile(
    r"^(?:"
    r"lib(?:cuda(?:rt)?|cublas(?:Lt)?|cufft(?:w)?|curand|cusolver(?:Mg)?|"
    r"cusparse(?:Lt)?|cupti|cudnn[^.]*|nccl|nvJitLink|nvrtc[^.]*|"
    r"nvToolsExt|nvblas|nvperf[^.]*|nvshmem[^.]*|torch_cuda[^.]*|"
    r"c10_cuda|caffe2_nvrtc|exllamav3_ext.*)|"
    r"exllamav3_ext.*"
    r")\.so(?:\..+)?$",
    flags=re.IGNORECASE,
)
NVIDIA_DRIVER_FAMILY = re.compile(
    r"^lib(?:cuda|nvidia(?:-[A-Za-z0-9_-]+)?)\.so(?:\..+)?$",
    flags=re.IGNORECASE,
)
EXL_REQUIRED_LIVE_FAMILIES = {
    "cudart": re.compile(r"^libcudart\.so(?:\..+)?$", re.IGNORECASE),
    "torch_cuda": re.compile(r"^libtorch_cuda(?:_[^.]+)?\.so(?:\..+)?$", re.IGNORECASE),
    "exllamav3_ext": re.compile(r"^exllamav3_ext.*\.so(?:\..+)?$", re.IGNORECASE),
}
# Anonymous shared memory is reported by /proc/<pid>/maps as a deleted
# pseudo-file.  The NVIDIA driver maps several ``/dev/zero (deleted)`` segments
# in every CUDA process; they carry no file identity and cannot be resolved on
# disk.  Any other deleted mapping remains a fault (2026-09-13 amendment).
ANONYMOUS_SHARED_MAPPING = re.compile(
    r"^/(?:dev/zero|memfd:.*|SYSV[0-9A-Fa-f]{8}) \(deleted\)$"
)
# Triton (a pinned torch dependency whose version is bound by the distribution
# inventory) compiles one small CUDA driver-API helper into the launching
# user's Triton cache on first use; the Triton kernels this model runs through
# load via that helper.  It is the only runtime-compiled shared object the
# live gate admits outside the pinned site-packages tree, and only beneath the
# Triton cache root, recorded with path, byte count, and SHA-256.
TRITON_JIT_HELPER_FAMILY = re.compile(
    r"^cuda_utils\.cpython-3\d+-x86_64-linux-gnu\.so$"
)
TRITON_RUNTIME_HELPER_EXCEPTION = (
    "only Triton's runtime-compiled cuda_utils helper, resolved beneath the "
    "launching user's Triton cache root, may be outside the pinned "
    "site-packages tree; it is recorded with path, byte count, and SHA-256"
)


def triton_jit_cache_root() -> Path:
    """Triton's default runtime cache root for the launching user."""
    return (Path.home() / ".triton" / "cache").resolve()


def sanitized_no_ld_environment(
    *, environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Copy an environment after removing every inherited ``LD_*`` key."""
    clean = dict(os.environ if environ is None else environ)
    for key in tuple(clean):
        if key.startswith("LD_"):
            clean.pop(key)
    if any(key.startswith("LD_") for key in clean):
        raise RunSafetyError("could not construct an LD-free process environment")
    return clean


def require_no_ld_environment(environment: dict[str, str]) -> None:
    """Reject a purported EXL3 probe/launch environment with loader controls."""
    observed = sorted(key for key in environment if key.startswith("LD_"))
    if observed:
        raise RunSafetyError(f"EXL3 environment retains loader controls: {observed}")


def command(
    argv: list[str], *, timeout: int = 30, env: dict[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            argv, check=True, capture_output=True, text=True, timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RunSafetyError(f"runtime provenance command failed: {argv!r}") from exc
    return "\n".join(x.strip() for x in (result.stdout, result.stderr) if x.strip())


def cmake_cache_values(llama_server: Path) -> tuple[Path, dict[str, str]]:
    checkout = llama_server.resolve().parents[2]
    cache = checkout / "build" / "CMakeCache.txt"
    if not cache.is_file() or cache.is_symlink():
        raise RunSafetyError("llama.cpp build lacks a regular CMakeCache.txt")
    values: dict[str, str] = {}
    for line in cache.read_text(errors="strict").splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key, _kind = key_type.split(":", 1)
        values[key] = value
    return cache, values


def llama_build_identity(llama_server: Path) -> dict:
    cache, values = cmake_cache_values(llama_server)
    expected = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "89",
        "GGML_BLAS": "OFF",
        "GGML_CUDA": "ON",
        "GGML_NATIVE": "ON",
        "GGML_OPENMP": "ON",
        "LLAMA_CURL": "OFF",
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise RunSafetyError("llama.cpp CMake feature/architecture contract changed")
    try:
        tools = {
            "cmake": Path(values.get("CMAKE_COMMAND", "")).resolve(strict=True),
            "cxx": Path(values.get("CMAKE_CXX_COMPILER", "")).resolve(strict=True),
            "nvcc": Path(values.get("CMAKE_CUDA_COMPILER", "")).resolve(strict=True),
        }
    except OSError as exc:
        raise RunSafetyError("llama.cpp configured compiler/tool is missing") from exc
    versions = {name: command([str(path), "--version"])
                for name, path in tools.items()}
    if (
        "cmake version 3.28.3" not in versions["cmake"].casefold()
        or "13.3.0" not in versions["cxx"]
        or "12.6.85" not in versions["nvcc"]
    ):
        raise RunSafetyError("llama.cpp compiler/CUDA toolchain changed")
    return {
        "schema": "paper2-r1-llama-build-identity-v1",
        "cmake_cache_sha256": sha256_file(cache),
        "settings": expected,
        "tool_sha256": {name: sha256_file(path) for name, path in tools.items()},
        "tool_versions": versions,
    }


def llama_cuda_library_dir(llama_server: Path) -> Path:
    """Return the private CUDA 12.6 runtime directory frozen at build time."""
    _cache, values = cmake_cache_values(llama_server)
    try:
        toolkit_root = Path(values["CUDAToolkit_ROOT"]).resolve(strict=True)
        compiler = Path(values["CMAKE_CUDA_COMPILER"]).resolve(strict=True)
        library_dir = (toolkit_root / "lib64").resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise RunSafetyError("llama.cpp CUDA 12.6 toolkit path is missing") from exc
    if (
        toolkit_root.name != "cuda-12.6"
        or compiler != (toolkit_root / "bin" / "nvcc").resolve(strict=True)
        or not library_dir.is_dir()
    ):
        raise RunSafetyError("llama.cpp is not bound to the frozen CUDA 12.6 toolkit")
    return library_dir


def sanitized_cuda_loader_environment(
    library_dir: Path, *, environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with exactly one pinned ``LD_*`` variable."""
    try:
        frozen_dir = library_dir.resolve(strict=True)
    except OSError as exc:
        raise RunSafetyError("frozen CUDA runtime directory is missing") from exc
    if not frozen_dir.is_dir():
        raise RunSafetyError("frozen CUDA runtime path is not a directory")
    clean = dict(os.environ if environ is None else environ)
    for key in tuple(clean):
        if key.startswith("LD_"):
            clean.pop(key)
    clean["LD_LIBRARY_PATH"] = str(frozen_dir)
    observed = {key: value for key, value in clean.items() if key.startswith("LD_")}
    if observed != {"LD_LIBRARY_PATH": str(frozen_dir)}:
        raise RunSafetyError("could not construct the pinned CUDA loader environment")
    return clean


def llama_cuda_loader_environment(llama_server: Path) -> tuple[Path, dict[str, str]]:
    """Derive the one loader environment used by all llama.cpp probes/launches."""
    library_dir = llama_cuda_library_dir(llama_server)
    return library_dir, sanitized_cuda_loader_environment(library_dir)


def require_pinned_cuda_loader_environment(
    llama_server: Path, loader_env: dict[str, str],
) -> Path:
    """Reject a caller-supplied environment with any extra loader control."""
    library_dir = llama_cuda_library_dir(llama_server)
    expected = {"LD_LIBRARY_PATH": str(library_dir)}
    observed = {
        key: value for key, value in loader_env.items() if key.startswith("LD_")
    }
    if observed != expected:
        raise RunSafetyError("llama.cpp probe did not use the pinned CUDA loader environment")
    return library_dir


def parse_cuda_ldd(text: str) -> dict[str, Path]:
    """Parse only the three CUDA user-space libraries in one ldd readback."""
    if "not found" in text:
        raise RunSafetyError("llama.cpp dynamic-loader readback contains a missing library")
    found: dict[str, Path] = {}
    wanted = set(CUDA_RUNTIME_CONTRACT)
    pattern = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match or match.group(1) not in wanted:
            continue
        soname, raw_path = match.groups()
        if soname in found or not Path(raw_path).is_absolute():
            raise RunSafetyError(f"ambiguous CUDA dynamic-loader readback for {soname}")
        found[soname] = Path(raw_path)
    if set(found) != wanted:
        raise RunSafetyError(
            f"llama.cpp dynamic-loader readback lacks frozen CUDA libraries: "
            f"{sorted(wanted - set(found))}"
        )
    return found


def _pinned_library_identity(soname: str, loader_path: Path, library_dir: Path) -> dict:
    expected = CUDA_RUNTIME_CONTRACT[soname]
    try:
        resolved = loader_path.resolve(strict=True)
        expected_path = (library_dir / expected["filename"]).resolve(strict=True)
    except OSError as exc:
        raise RunSafetyError(f"resolved CUDA library is missing: {soname}") from exc
    if (
        resolved != expected_path
        or resolved.parent != library_dir
        or not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size != expected["bytes"]
        or sha256_file(resolved) != expected["sha256"]
    ):
        raise RunSafetyError(f"CUDA 12.6 runtime contract changed: {soname}")
    return {
        "soname": soname,
        "loader_path": str(loader_path),
        "resolved_path": str(resolved),
        **expected,
    }


def llama_cuda_runtime_identity(
    llama_server: Path, *, loader_env: dict[str, str] | None = None,
) -> dict:
    """Resolve and hash the CUDA runtime under the exact launch environment."""
    server = llama_server.resolve(strict=True)
    if loader_env is None:
        library_dir, loader_env = llama_cuda_loader_environment(server)
    else:
        loader_env = dict(loader_env)
        library_dir = require_pinned_cuda_loader_environment(server, loader_env)
    resolutions = parse_cuda_ldd(command(["ldd", str(server)], env=loader_env))
    libraries = {
        soname: _pinned_library_identity(soname, resolutions[soname], library_dir)
        for soname in sorted(CUDA_RUNTIME_CONTRACT)
    }
    return {
        "schema": "paper2-r1-llama-cuda-runtime-private-v1",
        "toolkit_release": "12.6",
        "loader_contract": "LD_LIBRARY_PATH contains only the frozen CUDA library directory",
        "inherited_ld_variables_removed": True,
        "private_library_directory": str(library_dir),
        "llama_server_sha256": sha256_file(server),
        "libraries": libraries,
    }


def cuda_family_paths_from_proc_maps(lines: list[str]) -> set[Path]:
    """Return every mapped cudart/cublas/cublasLt-family file, fail closed."""
    mapped: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        mapped_name = fields[5]
        deleted_suffix = " (deleted)"
        is_deleted = mapped_name.endswith(deleted_suffix)
        if is_deleted:
            mapped_name = mapped_name[:-len(deleted_suffix)]
        raw_path = Path(mapped_name)
        if not CUDA_LIBRARY_FAMILY.fullmatch(raw_path.name):
            continue
        if is_deleted:
            raise RunSafetyError(
                f"llama.cpp has a deleted mapped CUDA-family runtime file: {raw_path}"
            )
        if not raw_path.is_absolute():
            raise RunSafetyError(
                f"live llama.cpp CUDA-family mapping is nonabsolute: {raw_path}"
            )
        try:
            resolved = raw_path.resolve(strict=True)
        except OSError as exc:
            raise RunSafetyError(
                f"live llama.cpp CUDA-family mapping is unreadable: {raw_path}"
            ) from exc
        if raw_path != resolved:
            raise RunSafetyError(
                f"live llama.cpp CUDA-family mapping is noncanonical: {raw_path}"
            )
        mapped.add(resolved)
    return mapped


def require_exact_cuda_family_mappings(
    lines: list[str], expected_paths: set[Path],
) -> set[Path]:
    """Require the complete CUDA-family mapping set, with no aliases or extras."""
    mapped = cuda_family_paths_from_proc_maps(lines)
    if mapped != expected_paths:
        missing = sorted(str(path) for path in expected_paths - mapped)
        extra = sorted(str(path) for path in mapped - expected_paths)
        raise RunSafetyError(
            "live llama.cpp CUDA-family mapping set changed: "
            f"missing={missing}, extra={extra}"
        )
    return mapped


def llama_process_cuda_runtime_identity(pid: int, expected: dict) -> dict:
    """Prove a live llama.cpp process mapped the preflight-resolved libraries."""
    if pid <= 0:
        raise RunSafetyError("invalid llama.cpp PID for CUDA map verification")
    try:
        lines = Path(f"/proc/{pid}/maps").read_text(errors="strict").splitlines()
    except OSError as exc:
        raise RunSafetyError("cannot read live llama.cpp CUDA mappings") from exc
    libraries = expected.get("libraries")
    if (
        expected.get("schema") != "paper2-r1-llama-cuda-runtime-private-v1"
        or not isinstance(libraries, dict)
        or set(libraries) != set(CUDA_RUNTIME_CONTRACT)
    ):
        raise RunSafetyError("pre-campaign CUDA runtime identity is malformed")
    try:
        expected_paths = {
            Path(identity["resolved_path"]).resolve(strict=True)
            for identity in libraries.values()
        }
    except (KeyError, OSError, TypeError) as exc:
        raise RunSafetyError("pre-campaign CUDA runtime paths are malformed") from exc
    if len(expected_paths) != len(CUDA_RUNTIME_CONTRACT):
        raise RunSafetyError("pre-campaign CUDA runtime paths are not one-to-one")
    require_exact_cuda_family_mappings(lines, expected_paths)
    for soname, identity in libraries.items():
        resolved = Path(identity.get("resolved_path", ""))
        _pinned_library_identity(
            soname, Path(identity.get("loader_path", "")),
            Path(expected["private_library_directory"]),
        )
    return {
        "schema": "paper2-r1-llama-cuda-process-maps-private-v1",
        "pid": pid,
        "libraries": libraries,
    }


def exec_llama_with_pinned_cuda(
    llama_server: Path, expected_library_dir: Path, argv: list[str],
) -> None:
    """Replace this process with llama-server under the frozen loader env."""
    try:
        server = llama_server.resolve(strict=True)
        expected_dir = expected_library_dir.resolve(strict=True)
    except OSError as exc:
        raise RunSafetyError("pinned llama.cpp executable/runtime is missing") from exc
    library_dir, loader_env = llama_cuda_loader_environment(server)
    if library_dir != expected_dir:
        raise RunSafetyError("launch CUDA directory differs from the toolchain receipt")
    if not server.is_file() or not os.access(server, os.X_OK):
        raise RunSafetyError("pinned llama-server is not executable")
    os.execve(str(server), [str(server), *argv], loader_env)


def sanitize_llama_cuda_runtime(identity: dict) -> dict:
    """Project a private loader receipt into a path-free public record."""
    libraries = identity.get("libraries")
    if (
        identity.get("schema") != "paper2-r1-llama-cuda-runtime-private-v1"
        or identity.get("toolkit_release") != "12.6"
        or not isinstance(libraries, dict)
        or set(libraries) != set(CUDA_RUNTIME_CONTRACT)
    ):
        raise RunSafetyError("cannot sanitize malformed CUDA runtime identity")
    public_libraries = {}
    for soname in sorted(CUDA_RUNTIME_CONTRACT):
        value = libraries[soname]
        expected = CUDA_RUNTIME_CONTRACT[soname]
        if any(value.get(key) != expected[key] for key in ("filename", "bytes", "sha256")):
            raise RunSafetyError(f"cannot sanitize changed CUDA runtime: {soname}")
        public_libraries[soname] = {
            "soname": soname,
            "filename": expected["filename"],
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }
    value = {
        "schema": "paper2-r1-llama-cuda-runtime-public-v1",
        "toolkit_release": "12.6",
        "dynamic_resolution_verified": True,
        "libraries": public_libraries,
        "privacy": "absolute toolkit and library paths omitted",
    }
    if re.search(r"(?:^|[\s\"'])/(?:home|root|Users)/", json.dumps(value)):
        raise RunSafetyError("sanitized CUDA runtime identity leaked a private path")
    return value


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _system_driver_path(
    path: Path, *, roots: set[Path] | None = None,
) -> bool:
    if roots is None:
        roots = {Path("/usr/lib").resolve(), Path("/lib").resolve()}
    return any(_path_within(path, root) for root in roots)


def _cuda_mapping_class(
    path: Path, site_packages: Path, *, system_driver_roots: set[Path] | None = None,
    triton_cache_root: Path | None = None,
) -> str | None:
    """Classify the frozen environment's CUDA-related shared-library families."""
    name = path.name
    if NVIDIA_DRIVER_FAMILY.fullmatch(name):
        if not _system_driver_path(path, roots=system_driver_roots):
            raise RunSafetyError(
                f"NVIDIA driver-family mapping is outside system library roots: {path}"
            )
        return "system_nvidia_driver"
    if ".so" not in name:
        return None
    if TRITON_JIT_HELPER_FAMILY.fullmatch(name):
        root = triton_jit_cache_root() if triton_cache_root is None else triton_cache_root
        if not _path_within(path, root):
            raise RunSafetyError(
                f"Triton runtime helper mapping is outside the Triton cache root: {path}"
            )
        return "triton_jit_cache"
    lowered_parts = {part.casefold() for part in path.parts}
    under_cuda_toolkit = any(part.startswith("cuda-") for part in lowered_parts)
    under_nvidia_package = "nvidia" in lowered_parts
    if (
        EXL_CUDA_USERSPACE_FAMILY.fullmatch(name)
        or under_cuda_toolkit
        or under_nvidia_package
        or "cuda" in name.casefold()
    ):
        return "cuda_userspace"
    # Binding extensions under the frozen cuda package may have ABI suffixes
    # and generic module names (for example, driver.cpython-312-*.so).
    if _path_within(path, site_packages) and "cuda" in lowered_parts:
        return "cuda_userspace"
    return None


def _mapped_file_record(
    path: Path, *, mapping_class: str, pids: set[int], site_packages: Path,
    triton_cache_root: Path | None = None,
) -> dict:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RunSafetyError(f"live EXL3 mapping is unreadable: {path}") from exc
    if not resolved.is_file():
        raise RunSafetyError(f"live EXL3 mapping is not a regular file: {path}")
    if mapping_class == "cuda_userspace" and not _path_within(resolved, site_packages):
        raise RunSafetyError(
            f"EXL3 mapped CUDA user-space library outside pinned site-packages: {resolved}"
        )
    if mapping_class == "triton_jit_cache" and not _path_within(
        resolved,
        triton_jit_cache_root() if triton_cache_root is None else triton_cache_root,
    ):
        raise RunSafetyError(
            f"EXL3 mapped Triton runtime helper outside the Triton cache root: {resolved}"
        )
    if mapping_class == "system_nvidia_driver" and not _system_driver_path(resolved):
        raise RunSafetyError(f"EXL3 mapped NVIDIA driver outside system roots: {resolved}")
    return {
        "mapping_class": mapping_class,
        "mapped_path": str(path),
        "resolved_path": str(resolved),
        "filename": resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "mapped_by_pids": sorted(pids),
    }


def _process_parent(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(errors="strict")
        fields = text[text.rfind(")") + 2:].split()
        return int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def _descendant_pids(root_pid: int) -> set[int]:
    if root_pid <= 0 or not Path(f"/proc/{root_pid}").is_dir():
        raise RunSafetyError("EXL3 root PID is not live")
    parents = {
        int(entry.name): _process_parent(int(entry.name))
        for entry in Path("/proc").iterdir()
        if entry.name.isdigit()
    }
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in descendants and parent in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _live_exl_process_snapshots(root_pid: int) -> dict[int, dict]:
    snapshots: dict[int, dict] = {}
    for pid in sorted(_descendant_pids(root_pid)):
        try:
            maps = Path(f"/proc/{pid}/maps").read_text(errors="strict").splitlines()
            environ_bytes = Path(f"/proc/{pid}/environ").read_bytes()
            executable = Path(f"/proc/{pid}/exe").resolve(strict=True)
        except OSError as exc:
            raise RunSafetyError(
                f"cannot capture live EXL3 process evidence for PID {pid}"
            ) from exc
        ld_names = sorted({
            field.split(b"=", 1)[0].decode("ascii", errors="strict")
            for field in environ_bytes.split(b"\0")
            if field.startswith(b"LD_") and b"=" in field
        })
        snapshots[pid] = {
            "maps": maps,
            "executable": str(executable),
            "ld_variable_names": ld_names,
        }
    return snapshots


def exl3_process_cuda_mapping_identity(
    root_pid: int, site_packages_root: Path, *, snapshots: dict[int, dict] | None = None,
    triton_cache_root: Path | None = None,
) -> dict:
    """Record and contain the classified CUDA mappings of a live EXL3 tree.

    This proves containment for the explicitly reported classifier families. It
    does not claim that ``/proc/*/maps`` can identify every possible GPU-runtime
    mechanism or establish a universal CUDA dependency inventory.
    """
    if root_pid <= 0:
        raise RunSafetyError("invalid EXL3 root PID for mapping verification")
    try:
        site_packages = site_packages_root.resolve(strict=True)
    except OSError as exc:
        raise RunSafetyError("pinned EXL3 site-packages root is missing") from exc
    if not site_packages.is_dir():
        raise RunSafetyError("pinned EXL3 site-packages root is not a directory")
    observed = _live_exl_process_snapshots(root_pid) if snapshots is None else snapshots
    if root_pid not in observed or not observed:
        raise RunSafetyError("EXL3 mapping evidence omits the launched process")

    path_pids: dict[tuple[Path, str], set[int]] = {}
    process_records = []
    for pid in sorted(observed):
        snapshot = observed[pid]
        ld_names = snapshot.get("ld_variable_names")
        maps = snapshot.get("maps")
        executable = snapshot.get("executable")
        if (
            not isinstance(pid, int)
            or not isinstance(ld_names, list)
            or not isinstance(maps, list)
            or not isinstance(executable, str)
        ):
            raise RunSafetyError("EXL3 process snapshot is malformed")
        if ld_names:
            raise RunSafetyError(
                f"live EXL3 process retained LD_* loader controls: pid={pid}, keys={ld_names}"
            )
        process_records.append({
            "pid": pid,
            "executable": executable,
            "observed_ld_variable_names": [],
        })
        for line in maps:
            if " (deleted)" in line and ".so" in line:
                raise RunSafetyError("live EXL3 process has a deleted shared-library mapping")
            fields = line.split(maxsplit=5)
            if len(fields) < 6 or not fields[5].startswith("/"):
                continue
            mapped_name = fields[5]
            if mapped_name.endswith(" (deleted)"):
                # Anonymous shared memory (the driver's /dev/zero segments)
                # has no on-disk identity; every other deleted mapping is a
                # fault, whether or not it is a shared library.
                if ANONYMOUS_SHARED_MAPPING.fullmatch(mapped_name):
                    continue
                raise RunSafetyError(
                    f"live EXL3 process has a deleted file mapping: {mapped_name}"
                )
            raw_path = Path(mapped_name)
            try:
                resolved = raw_path.resolve(strict=True)
            except OSError as exc:
                raise RunSafetyError(f"live EXL3 mapping is unreadable: {raw_path}") from exc
            mapping_class = _cuda_mapping_class(
                resolved, site_packages, triton_cache_root=triton_cache_root,
            )
            if mapping_class is not None:
                path_pids.setdefault((raw_path, mapping_class), set()).add(pid)

    libraries = [
        _mapped_file_record(
            path, mapping_class=mapping_class, pids=pids,
            site_packages=site_packages, triton_cache_root=triton_cache_root,
        )
        for (path, mapping_class), pids in sorted(
            path_pids.items(), key=lambda item: (item[0][1], str(item[0][0]))
        )
    ]
    userspace = [item for item in libraries if item["mapping_class"] == "cuda_userspace"]
    if not userspace:
        raise RunSafetyError("live EXL3 process has no classified CUDA user-space mappings")
    missing = [
        family for family, pattern in EXL_REQUIRED_LIVE_FAMILIES.items()
        if not any(pattern.fullmatch(item["filename"]) for item in userspace)
    ]
    if missing:
        raise RunSafetyError(
            f"live EXL3 process lacks required mapped runtime families: {missing}"
        )
    return {
        "schema": "paper2-r1-exl3-cuda-process-maps-private-v1",
        "root_pid": root_pid,
        "site_packages_root": str(site_packages),
        "launch_loader_contract": "all inherited LD_* variables removed; no LD_LIBRARY_PATH",
        "processes": process_records,
        "mapped_libraries": libraries,
        "classified_cuda_userspace_mappings_contained": True,
        "system_driver_exception": (
            "only system-root libcuda.so and libnvidia-*.so driver-family mappings "
            "may be outside the pinned site-packages tree"
        ),
        "triton_runtime_helper_exception": TRITON_RUNTIME_HELPER_EXCEPTION,
        "evidence_scope": (
            "containment of the recorded CUDA/PyTorch-CUDA/ExLlama shared-library "
            "classifier families in this live process-tree snapshot; not a universal "
            "inventory of every possible GPU-runtime mechanism"
        ),
    }


def validate_exl3_process_cuda_mapping_identity(
    identity: dict, *, expected_root_pid: int, expected_site_packages_root: Path,
    expected_extension: dict, triton_cache_root: Path | None = None,
) -> None:
    """Validate a retained private EXL3 mapping receipt after process shutdown."""
    try:
        site_packages = expected_site_packages_root.resolve(strict=True)
    except OSError as exc:
        raise RunSafetyError("retained EXL3 site-packages tree is missing") from exc
    processes = identity.get("processes")
    libraries = identity.get("mapped_libraries")
    if (
        identity.get("schema") != "paper2-r1-exl3-cuda-process-maps-private-v1"
        or identity.get("root_pid") != expected_root_pid
        or identity.get("site_packages_root") != str(site_packages)
        or identity.get("launch_loader_contract")
        != "all inherited LD_* variables removed; no LD_LIBRARY_PATH"
        or identity.get("classified_cuda_userspace_mappings_contained") is not True
        or not isinstance(processes, list)
        or not processes
        or not isinstance(libraries, list)
        or not libraries
    ):
        raise RunSafetyError("retained EXL3 mapping receipt is malformed")
    process_pids = set()
    for process in processes:
        pid = process.get("pid")
        if (
            not isinstance(pid, int)
            or pid <= 0
            or pid in process_pids
            or process.get("observed_ld_variable_names") != []
            or not isinstance(process.get("executable"), str)
            or not process["executable"].startswith("/")
        ):
            raise RunSafetyError("retained EXL3 process evidence is malformed")
        process_pids.add(pid)
    if expected_root_pid not in process_pids:
        raise RunSafetyError("retained EXL3 process evidence omits its root")

    userspace = []
    for record in libraries:
        try:
            mapped = Path(record["mapped_path"])
            resolved = Path(record["resolved_path"]).resolve(strict=True)
            mapped_pids = record["mapped_by_pids"]
        except (KeyError, OSError, TypeError) as exc:
            raise RunSafetyError("retained EXL3 library evidence is malformed") from exc
        mapping_class = record.get("mapping_class")
        if (
            not mapped.is_absolute()
            or record.get("resolved_path") != str(resolved)
            or record.get("filename") != resolved.name
            or record.get("bytes") != resolved.stat().st_size
            or record.get("sha256") != sha256_file(resolved)
            or not isinstance(mapped_pids, list)
            or not mapped_pids
            or mapped_pids != sorted(set(mapped_pids))
            or not set(mapped_pids) <= process_pids
            or mapping_class != _cuda_mapping_class(
                resolved, site_packages, triton_cache_root=triton_cache_root,
            )
        ):
            raise RunSafetyError("retained EXL3 library evidence changed")
        if mapping_class == "cuda_userspace":
            if not _path_within(resolved, site_packages):
                raise RunSafetyError("retained EXL3 CUDA user-space mapping escaped")
            userspace.append(record)
        elif mapping_class == "triton_jit_cache":
            if not _path_within(
                resolved,
                triton_jit_cache_root() if triton_cache_root is None else triton_cache_root,
            ):
                raise RunSafetyError("retained EXL3 Triton runtime helper escaped its cache")
        elif mapping_class != "system_nvidia_driver":
            raise RunSafetyError("retained EXL3 library has an unknown class")
    missing = [
        family for family, pattern in EXL_REQUIRED_LIVE_FAMILIES.items()
        if not any(pattern.fullmatch(item["filename"]) for item in userspace)
    ]
    extensions = [
        item for item in userspace if item["filename"].startswith("exllamav3_ext")
    ]
    if (
        missing
        or len(extensions) != 1
        or any(
            extensions[0].get(key) != expected_extension.get(key)
            for key in ("filename", "bytes", "sha256")
        )
    ):
        raise RunSafetyError("retained EXL3 mapped runtime differs from its fingerprint")


PYTHON_FINGERPRINT_PROBE = r'''
import hashlib, importlib.metadata as m, importlib.util, json, pathlib, sysconfig, torch
required = {"torch", "exllamav3", "tabbyapi", "uvicorn", "fastapi"}
dists = {d.metadata["Name"].casefold(): d for d in m.distributions() if d.metadata["Name"]}
targets = sorted(name for name in dists if name in required or name.startswith("nvidia-"))
if not required <= set(targets):
    raise SystemExit("missing required inference distribution")
out = {}; extension_files = []
for name in targets:
    d = dists[name]
    h = hashlib.sha256(); count = total = 0
    for rel in sorted(d.files or [], key=lambda p: str(p)):
        path = pathlib.Path(d.locate_file(rel))
        if not path.is_file():
            continue
        fh = hashlib.sha256(); size = 0
        with path.open("rb") as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block: break
                fh.update(block); size += len(block)
        rel_b = str(rel).encode()
        h.update(len(rel_b).to_bytes(8, "big")); h.update(rel_b)
        h.update(size.to_bytes(8, "big")); h.update(fh.digest())
        if path.name.startswith("exllamav3_ext") and ".so" in path.name:
            extension_files.append({"path": str(path.resolve()), "filename": path.name,
                                    "bytes": size, "sha256": fh.hexdigest()})
        count += 1; total += size
    out[name] = {"version": d.version, "file_count": count,
                 "total_bytes": total, "content_tree_sha256": h.hexdigest()}
inventory = sorted((d.metadata["Name"], d.version) for d in m.distributions() if d.metadata["Name"])
purelib = pathlib.Path(sysconfig.get_path("purelib")).resolve()
platlib = pathlib.Path(sysconfig.get_path("platlib")).resolve()
spec = importlib.util.find_spec("exllamav3_ext")
if purelib != platlib or not spec or not spec.origin or len(extension_files) != 1:
    raise SystemExit("EXL3 site-packages/precompiled-extension identity is ambiguous")
origin = pathlib.Path(spec.origin).resolve()
if origin != pathlib.Path(extension_files[0]["path"]):
    raise SystemExit("EXL3 precompiled extension spec differs from installed content")
print(json.dumps({"distributions": out,
 "all_distribution_inventory_sha256": hashlib.sha256(json.dumps(inventory, separators=(",",":"), ensure_ascii=True).encode()).hexdigest(),
 "site_packages_root": str(purelib),
 "precompiled_extension": extension_files[0],
 "torch_runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version()}}, sort_keys=True))
'''

EXL_PRIVATE_LAYOUT_PROBE = r'''
import importlib.util, json, pathlib, sysconfig
purelib = pathlib.Path(sysconfig.get_path("purelib")).resolve()
platlib = pathlib.Path(sysconfig.get_path("platlib")).resolve()
spec = importlib.util.find_spec("exllamav3_ext")
if purelib != platlib or not spec or not spec.origin:
    raise SystemExit("EXL3 private runtime layout is ambiguous")
print(json.dumps({"site_packages_root": str(purelib),
                  "precompiled_extension_path": str(pathlib.Path(spec.origin).resolve())},
                 sort_keys=True))
'''


def exl3_private_runtime_layout(python: Path) -> dict:
    """Return private paths needed only for live EXL3 containment checks."""
    environment = sanitized_no_ld_environment()
    require_no_ld_environment(environment)
    try:
        result = subprocess.run(
            [str(python), "-c", EXL_PRIVATE_LAYOUT_PROBE], check=True,
            capture_output=True, text=True, timeout=30, env=environment,
        )
        value = json.loads(result.stdout)
        environment_root = python.absolute().parent.parent.resolve(strict=True)
        site_packages = Path(value["site_packages_root"]).resolve(strict=True)
        extension = Path(value["precompiled_extension_path"]).resolve(strict=True)
        site_packages.relative_to(environment_root)
        extension.relative_to(site_packages)
    except (
        OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
        json.JSONDecodeError, KeyError, TypeError, ValueError,
    ) as exc:
        raise RunSafetyError("failed to bind the private EXL3 runtime layout") from exc
    if (
        not site_packages.is_dir()
        or not extension.is_file()
        or extension.is_symlink()
        or not extension.name.startswith("exllamav3_ext")
        or ".so" not in extension.name
    ):
        raise RunSafetyError("EXL3 private runtime layout is malformed")
    return {
        "schema": "paper2-r1-exl3-private-runtime-layout-v1",
        "site_packages_root": str(site_packages),
        "precompiled_extension_path": str(extension),
        "loader_environment_contract": (
            "all inherited LD_* variables removed; no LD_LIBRARY_PATH"
        ),
    }


def sanitize_exl3_python_runtime_content(value: dict) -> dict:
    """Remove private environment paths from installed-content evidence."""
    clean = json.loads(json.dumps(value))
    if not isinstance(clean.get("precompiled_extension"), dict):
        raise RunSafetyError("EXL3 installed-content evidence lacks its extension")
    clean.pop("site_packages_root", None)
    clean["precompiled_extension"].pop("path", None)
    def absolute_strings(item: object) -> list[str]:
        if isinstance(item, str):
            return [item] if item.startswith("/") else []
        if isinstance(item, dict):
            return [path for child in item.values() for path in absolute_strings(child)]
        if isinstance(item, list):
            return [path for child in item for path in absolute_strings(child)]
        return []

    if absolute_strings(clean):
        raise RunSafetyError("EXL3 installed-content projection leaked a private path")
    return clean


def python_runtime_identity(python: Path) -> dict:
    environment = sanitized_no_ld_environment()
    require_no_ld_environment(environment)
    try:
        result = subprocess.run(
            [str(python), "-c", PYTHON_FINGERPRINT_PROBE],
            check=True, capture_output=True, text=True, timeout=1800,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RunSafetyError("failed to fingerprint installed Python inference runtime") from exc
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RunSafetyError("Python inference runtime fingerprint is not JSON") from exc
    if not isinstance(value.get("distributions"), dict) or not value["distributions"]:
        raise RunSafetyError("Python inference runtime fingerprint is malformed")
    try:
        # Do not resolve the interpreter symlink itself: this is a virtual
        # environment whose executable legitimately resolves to /usr/bin.
        environment_root = python.absolute().parent.parent.resolve(strict=True)
        site_packages = Path(value["site_packages_root"]).resolve(strict=True)
        extension = Path(value["precompiled_extension"]["path"]).resolve(strict=True)
        site_packages.relative_to(environment_root)
        extension.relative_to(site_packages)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RunSafetyError("EXL3 Python runtime escaped its pinned environment") from exc
    extension_value = value["precompiled_extension"]
    if (
        not site_packages.is_dir()
        or not extension.is_file()
        or extension.is_symlink()
        or not extension.name.startswith("exllamav3_ext")
        or ".so" not in extension.name
        or extension_value.get("filename") != extension.name
        or extension_value.get("bytes") != extension.stat().st_size
        or not re.fullmatch(r"[0-9a-f]{64}", str(extension_value.get("sha256", "")))
    ):
        raise RunSafetyError("EXL3 precompiled extension identity is malformed")
    # Absolute paths are private campaign evidence.  The installed-content
    # receipt is deliberately path-free because it is later projected into
    # public claims.
    value = sanitize_exl3_python_runtime_content(value)
    return {
        "schema": "paper2-r1-python-runtime-content-v1",
        "loader_environment_contract": "all inherited LD_* variables removed; no LD_LIBRARY_PATH",
        "observed_ld_variables": {},
        **value,
    }


def synthetic_self_test() -> None:
    if "content_tree_sha256" not in PYTHON_FINGERPRINT_PROBE:
        raise AssertionError("runtime probe does not content-hash distribution files")
    inventory = {"b": "2", "a": "1"}
    left = hashlib.sha256(json.dumps(sorted(inventory.items())).encode()).hexdigest()
    right = hashlib.sha256(json.dumps(sorted(reversed(list(inventory.items())))).encode()).hexdigest()
    if left != right:
        raise AssertionError("runtime inventory projection is order-dependent")
    sample = "\n".join(
        f"\t{soname} => /private/cuda/{value['filename']} (0x0123456789abcdef)"
        for soname, value in CUDA_RUNTIME_CONTRACT.items()
    )
    parsed = parse_cuda_ldd(sample)
    if set(parsed) != set(CUDA_RUNTIME_CONTRACT):
        raise AssertionError("CUDA ldd parser lost a required library")
    for broken in (
        sample.replace("libcudart.so.12 =>", "libcudart.so.12 => not found #"),
        "\n".join(sample.splitlines()[1:]),
        sample + "\n" + sample.splitlines()[0],
    ):
        try:
            parse_cuda_ldd(broken)
        except RunSafetyError:
            pass
        else:
            raise AssertionError("CUDA ldd parser accepted missing/ambiguous evidence")
    private = {
        "schema": "paper2-r1-llama-cuda-runtime-private-v1",
        "toolkit_release": "12.6",
        "libraries": {
            soname: {
                "soname": soname,
                "loader_path": f"/private/cuda/{value['filename']}",
                "resolved_path": f"/private/cuda/{value['filename']}",
                **value,
            }
            for soname, value in CUDA_RUNTIME_CONTRACT.items()
        },
    }
    public = sanitize_llama_cuda_runtime(private)
    encoded = json.dumps(public, sort_keys=True)
    if "/private/" in encoded or "loader_path" in encoded or "resolved_path" in encoded:
        raise AssertionError("public CUDA receipt leaked a private path")

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        clean = sanitized_cuda_loader_environment(
            tmp,
            environ={
                "PATH": "/bin",
                "P2R1_SENTINEL": "preserved",
                "LD_LIBRARY_PATH": "/wrong",
                "LD_PRELOAD": "/hostile/preload.so",
                "LD_AUDIT": "/hostile/audit.so",
                "LD_P2R1_UNKNOWN": "must also be removed",
            },
        )
        if clean.get("P2R1_SENTINEL") != "preserved" or {
            key: value for key, value in clean.items() if key.startswith("LD_")
        } != {"LD_LIBRARY_PATH": str(tmp.resolve())}:
            raise AssertionError("clean CUDA loader environment retained inherited LD_* state")

        exact_paths: set[Path] = set()
        map_lines = []
        for index, value in enumerate(CUDA_RUNTIME_CONTRACT.values()):
            path = tmp / value["filename"]
            path.touch()
            exact_paths.add(path.resolve())
            map_lines.append(
                f"7f{index:010x}-7f{index + 1:010x} r-xp 00000000 00:00 1 {path}"
            )
        require_exact_cuda_family_mappings(map_lines, exact_paths)
        require_exact_cuda_family_mappings(
            map_lines + [
                "7fffffff9000-7fffffff9fff rw-s 00000000 00:01 3 "
                "/dev/zero (deleted)",
                "7fffffffa000-7fffffffafff rw-s 00000000 00:01 4 "
                "/memfd:/.nvidia_drv.synthetic (deleted)",
            ],
            exact_paths,
        )
        extra = tmp / "libcublas.so.99.0"
        extra.touch()
        for broken_maps in (
            map_lines[:-1],
            map_lines
            + [f"7ffffffff000-7fffffffffff r-xp 00000000 00:00 2 {extra}"],
        ):
            try:
                require_exact_cuda_family_mappings(broken_maps, exact_paths)
            except RunSafetyError:
                pass
            else:
                raise AssertionError(
                    "live CUDA map verifier accepted a missing/extra family mapping"
                )
        for index, value in enumerate(CUDA_RUNTIME_CONTRACT.values()):
            deleted_absolute = list(map_lines)
            deleted_absolute[index] += " (deleted)"
            deleted_nonabsolute = map_lines + [
                "7fffffffb000-7fffffffbfff r-xp 00000000 00:00 5 "
                f"{value['filename']} (deleted)"
            ]
            for broken_maps in (deleted_absolute, deleted_nonabsolute):
                try:
                    require_exact_cuda_family_mappings(broken_maps, exact_paths)
                except RunSafetyError:
                    pass
                else:
                    raise AssertionError(
                        "live CUDA map verifier accepted a deleted CUDA-family mapping"
                    )

        site_packages = tmp / "exl3-env" / "lib" / "python3.12" / "site-packages"
        outside = tmp / "outside"
        system_driver_root = tmp / "system-driver"
        site_packages.mkdir(parents=True)
        outside.mkdir()
        system_driver_root.mkdir()
        multi_token_driver = system_driver_root / "libnvidia-egl-gbm.so.1"
        multi_token_driver.write_bytes(b"synthetic system driver")
        if _cuda_mapping_class(
            multi_token_driver.resolve(), site_packages.resolve(),
            system_driver_roots={system_driver_root.resolve()},
        ) != "system_nvidia_driver":
            raise AssertionError("multi-token system NVIDIA driver was not classified")
        outside_driver = outside / "libnvidia-egl-gbm.so.1"
        outside_driver.write_bytes(b"synthetic non-system driver")
        try:
            _cuda_mapping_class(
                outside_driver.resolve(), site_packages.resolve(),
                system_driver_roots={system_driver_root.resolve()},
            )
        except RunSafetyError:
            pass
        else:
            raise AssertionError("out-of-root NVIDIA driver-family mapping was accepted")
        exl_files = {
            "libcudart.so.12": b"wheel cudart",
            "libtorch_cuda.so": b"wheel torch cuda",
            "exllamav3_ext.cpython-312-x86_64-linux-gnu.so": b"wheel exl extension",
        }
        exl_map_lines = []
        for index, (name, content) in enumerate(exl_files.items(), start=10):
            path = site_packages / name
            path.write_bytes(content)
            exl_map_lines.append(
                f"7f{index:010x}-7f{index + 1:010x} r-xp 00000000 00:00 1 {path}"
            )
        extension_path = site_packages / "exllamav3_ext.cpython-312-x86_64-linux-gnu.so"
        expected_extension = {
            "filename": extension_path.name,
            "bytes": extension_path.stat().st_size,
            "sha256": sha256_file(extension_path),
        }
        snapshots = {
            1234: {
                "maps": exl_map_lines,
                "executable": "/private/exl3-env/bin/python",
                "ld_variable_names": [],
            },
        }
        exl_identity = exl3_process_cuda_mapping_identity(
            1234, site_packages, snapshots=snapshots,
        )
        validate_exl3_process_cuda_mapping_identity(
            exl_identity, expected_root_pid=1234,
            expected_site_packages_root=site_packages,
            expected_extension=expected_extension,
        )
        if any(
            not _path_within(Path(item["resolved_path"]), site_packages.resolve())
            for item in exl_identity["mapped_libraries"]
            if item["mapping_class"] == "cuda_userspace"
        ):
            raise AssertionError("EXL3 mapping receipt accepted an out-of-tree wheel library")

        hostile_ld = {1234: {**snapshots[1234], "ld_variable_names": ["LD_PRELOAD"]}}
        try:
            exl3_process_cuda_mapping_identity(
                1234, site_packages, snapshots=hostile_ld,
            )
        except RunSafetyError:
            pass
        else:
            raise AssertionError("EXL3 mapping receipt accepted a live LD_* control")

        outside_cuda = outside / "libcublas.so.12"
        outside_cuda.write_bytes(b"host CUDA")
        outside_snapshots = {
            1234: {
                **snapshots[1234],
                "maps": exl_map_lines + [
                    "7ffffffff000-7fffffffffff r-xp 00000000 00:00 2 "
                    + str(outside_cuda)
                ],
            },
        }
        try:
            exl3_process_cuda_mapping_identity(
                1234, site_packages, snapshots=outside_snapshots,
            )
        except RunSafetyError:
            pass
        else:
            raise AssertionError("EXL3 mapping receipt accepted out-of-tree CUDA user space")

        # Anonymous shared memory (the NVIDIA driver's /dev/zero segments, memfd
        # and SysV segments) is skipped; any other deleted mapping is a fault.
        anonymous_snapshots = {
            1234: {
                **snapshots[1234],
                "maps": exl_map_lines + [
                    "7fffffff9000-7fffffff9fff rw-s 00000000 00:01 3 /dev/zero (deleted)",
                    "7fffffffa000-7fffffffafff rw-s 00000000 00:01 4 "
                    "/memfd:cuda.synthetic (deleted)",
                    "7fffffffb000-7fffffffbfff rw-s 00000000 00:01 5 /SYSV0000abcd (deleted)",
                ],
            },
        }
        anonymous_identity = exl3_process_cuda_mapping_identity(
            1234, site_packages, snapshots=anonymous_snapshots,
        )
        if len(anonymous_identity["mapped_libraries"]) != len(exl_files):
            raise AssertionError("anonymous shared memory changed the EXL3 mapping receipt")
        for deleted_line in (
            f"7ffffffff000-7fffffffffff r--p 00000000 00:00 6 {tmp / 'model.safetensors'} (deleted)",
            "7ffffffff000-7fffffffffff r-xp 00000000 00:00 7 "
            f"{site_packages / 'libcudart.so.12'} (deleted)",
        ):
            deleted_snapshots = {
                1234: {**snapshots[1234], "maps": exl_map_lines + [deleted_line]},
            }
            try:
                exl3_process_cuda_mapping_identity(
                    1234, site_packages, snapshots=deleted_snapshots,
                )
            except RunSafetyError:
                pass
            else:
                raise AssertionError("EXL3 mapping receipt accepted a deleted file mapping")

        # Triton's runtime helper is admitted only beneath the Triton cache root
        # and is recorded with its identity.
        triton_root = (tmp / "triton-cache").resolve()
        triton_helper = (
            triton_root / "SYNTHETICKEY" / "cuda_utils.cpython-312-x86_64-linux-gnu.so"
        )
        triton_helper.parent.mkdir(parents=True)
        triton_helper.write_bytes(b"triton runtime helper")
        triton_snapshots = {
            1234: {
                **snapshots[1234],
                "maps": exl_map_lines + [
                    f"7ffffffff000-7fffffffffff r-xp 00000000 00:00 8 {triton_helper}"
                ],
            },
        }
        triton_identity = exl3_process_cuda_mapping_identity(
            1234, site_packages, snapshots=triton_snapshots, triton_cache_root=triton_root,
        )
        helper_records = [
            item for item in triton_identity["mapped_libraries"]
            if item["mapping_class"] == "triton_jit_cache"
        ]
        if (
            len(helper_records) != 1
            or helper_records[0]["sha256"] != sha256_file(triton_helper)
            or helper_records[0]["bytes"] != triton_helper.stat().st_size
            or triton_identity.get("triton_runtime_helper_exception")
            != TRITON_RUNTIME_HELPER_EXCEPTION
        ):
            raise AssertionError("Triton runtime helper was not recorded with its identity")
        validate_exl3_process_cuda_mapping_identity(
            triton_identity, expected_root_pid=1234,
            expected_site_packages_root=site_packages,
            expected_extension=expected_extension, triton_cache_root=triton_root,
        )
        stray_helper = outside / "cuda_utils.cpython-312-x86_64-linux-gnu.so"
        stray_helper.write_bytes(b"stray helper")
        stray_snapshots = {
            1234: {
                **snapshots[1234],
                "maps": exl_map_lines + [
                    f"7ffffffff000-7fffffffffff r-xp 00000000 00:00 9 {stray_helper}"
                ],
            },
        }
        try:
            exl3_process_cuda_mapping_identity(
                1234, site_packages, snapshots=stray_snapshots, triton_cache_root=triton_root,
            )
        except RunSafetyError:
            pass
        else:
            raise AssertionError("EXL3 mapping receipt accepted a Triton helper outside its cache")

        private_python = {
            "site_packages_root": str(site_packages),
            "precompiled_extension": {"path": str(extension_path), **expected_extension},
            "distributions": {"torch": {"version": "synthetic"}},
        }
        public_python = sanitize_exl3_python_runtime_content(private_python)
        if "site_packages_root" in public_python or "path" in public_python["precompiled_extension"]:
            raise AssertionError("path-free EXL3 installed-content projection retained a path")


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] == "self-test":
        synthetic_self_test()
        print("P2R1 runtime-provenance static self-test passed")
        return 0
    if sys.argv[1] != "exec-llama" or "--" not in sys.argv[2:]:
        raise SystemExit(
            "usage: runtime_provenance.py [self-test | exec-llama "
            "LLAMA_SERVER CUDA_LIBRARY_DIR -- LLAMA_ARGS...]"
        )
    separator = sys.argv.index("--", 2)
    fixed = sys.argv[2:separator]
    if len(fixed) != 2:
        raise SystemExit(
            "exec-llama requires LLAMA_SERVER and CUDA_LIBRARY_DIR before --"
        )
    exec_llama_with_pinned_cuda(Path(fixed[0]), Path(fixed[1]), sys.argv[separator + 1:])
    raise AssertionError("os.execve unexpectedly returned")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 RUNTIME PROVENANCE ABORTED: {exc}") from exc
