#!/usr/bin/env python3
"""Verify a prospective source component; optionally run isolated CPU-only tests.

Not an inference launcher, release controller, author approval or security
sandbox. Self-tests block Python-level network/process operations as an
accidental-launch guard; they execute inspected, trusted source only.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import site
import subprocess
import sys
import tempfile

import replication_check as A

SCHEMA = "paper-replication-source-preparation-v1"
ENV_SCHEMA = "paper-replication-environment-preparation-v1"
TESTS = {
    "paper5": (("testsuite/paper5", "test_request_pacing.py"),
               ("testsuite/paper5_followups", "test*.py")),
    "paper6": (("testsuite/paper6", "test*.py"),),
}
ENTRYPOINTS = {
    "paper5": (("paper5/scripts/primary_execution.py", ["--help"]),
               ("paper5/scripts/align_oracle_dates.py", ["--help"]),
               ("testsuite/paper5_followups/followups.py", ["plan", "cross-model"])),
    "paper6": (("testsuite/paper6/run_cells.py", ["--help"]),
               ("prepare_gsm8k.py", ["--help"])),
}
TEST_DRIVER = r'''
import json, os, pathlib, runpy, sys, unittest
root = pathlib.Path(sys.argv[1]).resolve()
suite_path, pattern, dependency_json = sys.argv[2:]
for dependency in json.loads(dependency_json):
    path = pathlib.Path(dependency)
    if not path.is_absolute() or path.name not in ('site-packages', 'dist-packages'):
        raise RuntimeError('invalid explicit dependency directory')
    if str(path) not in sys.path:
        sys.path.append(str(path))
study_home = pathlib.Path.home()
protected = (study_home / 'models', study_home / '.cache/huggingface', pathlib.Path('/proc'), pathlib.Path('/dev'))
def guard(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork',
                 'socket.connect', 'socket.bind', 'socket.getaddrinfo'):
        raise RuntimeError('replication self-test attempted a process/network operation: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        value = pathlib.Path(os.fsdecode(args[0])).absolute()
        # NumPy may read entropy from the null/random devices, but never an
        # accelerator device. No real model/cache/process-state files are needed.
        if str(value) in ('/dev/null', '/dev/urandom', '/dev/random'):
            return
        if any(value.is_relative_to(p) for p in protected):
            raise RuntimeError('replication self-test attempted protected data/device access')
sys.addaudithook(guard)
if suite_path.endswith('.py'):
    script = root / suite_path
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *json.loads(pattern)]
    try:
        runpy.run_path(str(script), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    print('SOURCE_ENTRYPOINT_OK')
    raise SystemExit(0)
sys.path.insert(0, str(root / suite_path))
suite = unittest.defaultTestLoader.discover(str(root / suite_path), pattern=pattern)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun == 0 or not result.wasSuccessful() or result.skipped:
    raise SystemExit(1)
print('SOURCE_SELFTEST_OK', result.testsRun)
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dependency_paths():
    """Explicit installed package directories, never ambient project/PYTHONPATH.

Isolated Python omits user-site packages. Add the installed package directory
itself without executing its .pth files or sitecustomize hooks.
"""
    user = site.getusersitepackages()
    paths = [*site.getsitepackages(), *([user] if isinstance(user, str) else user)]
    return sorted({str(Path(p).resolve()) for p in paths if Path(p).is_dir() and
                   Path(p).resolve().name in ("site-packages", "dist-packages")})


def tree(directory):
    directory = Path(directory)
    A.require(directory.is_dir() and not any(p.is_symlink() for p in (directory, *directory.parents)),
              "invalid source component directory")
    result = {}
    for path in sorted(directory.rglob("*")):
        A.require(not path.is_symlink(), "symlink in source component")
        if path.is_dir():
            continue
        A.require(path.is_file() and path.stat().st_size <= 4 * 1024 * 1024, "invalid source-component file")
        result[path.relative_to(directory).as_posix()] = sha(path.read_bytes())
    return result


def validate_environment(record, paper):
    """Check the typed projection, never import packages or inspect the host."""
    A.require(set(record) == {"schema", "paper", "status", "author_approval", "metadata_collection_date_utc",
                             "original_observation", "study_hardware", "environments", "build", "serving", "not_established"},
              "unexpected environment fields")
    A.require(record["schema"] == ENV_SCHEMA and record["paper"] == paper and paper in TESTS and
              record["status"] == "preparation-not-a-release" and record["author_approval"] is False,
              "wrong environment identity/authority")
    A.require(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", record["metadata_collection_date_utc"]), "invalid metadata date")
    historical, current = "p5-pre-primary-snapshot", "current-metadata-not-historical-execution"

    def evidence(row, *, native=False):
        extra = {"mapped_at_snapshot", "current_contents_rehashed"} if native else set()
        A.require(set(row) == {"source", "bytes", "sha256", "basis"} | extra, "unexpected metadata identity fields")
        A.relative(row["source"])
        A.digest(row["sha256"])
        A.require(type(row["bytes"]) is int and row["bytes"] > 0 and row["basis"] in (historical, current),
                  "invalid runtime metadata evidence")
        if native:
            A.require(row["basis"] == historical and row["current_contents_rehashed"] is False and
                      type(row["mapped_at_snapshot"]) is bool, "native inventory is not a fresh verification")

    observation = record["original_observation"]
    A.require(set(observation) == {"snapshot_sha256", "started_at", "finished_at", "scope"}, "invalid dated observation")
    A.digest(observation["snapshot_sha256"])
    hardware = record["study_hardware"]
    A.require(set(hardware) == {"basis", "cpu_model", "host_mem_kib", "kernel", "gpu"} and
              set(hardware["gpu"]) == {"name", "memory_mib", "driver"}, "unexpected hardware identifiers")
    A.require(type(hardware["host_mem_kib"]) is int and hardware["host_mem_kib"] > 0 and
              re.fullmatch(r"[0-9]+", str(hardware["gpu"]["memory_mib"])), "invalid hardware memory size")
    environments = record["environments"]
    expected = {"exl3-base", "analysis"} | ({"exl3-overlay", "lingua"} if paper == "paper5" else set())
    A.require(len(environments) == len(expected) and {env["id"] for env in environments} == expected,
              "missing/duplicate runtime environment")
    for env in environments:
        expected_basis = historical if env["id"] in ("exl3-base", "exl3-overlay") else current
        A.require(env["basis"] == expected_basis and env["packages"], "environment overstates its observation")
        extra = {"python_from", "wheel"} if env["id"] == "exl3-overlay" else {"python"} if env["id"] == "analysis" else {"python", "torch"}
        A.require(set(env) == {"id", "basis", "scope", "packages"} | extra, "unexpected installed environment fields")
        names = set()
        for package in env["packages"]:
            A.require(set(package) == {"name", "version", "metadata"} and
                      re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", package["name"]) and
                      re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", package["version"]), "invalid distribution identity")
            name = re.sub(r"[-_.]+", "-", package["name"]).lower()
            A.require(name not in names, "duplicate distribution in environment")
            names.add(name)
            evidence(package["metadata"])
            A.require(package["metadata"]["basis"] == env["basis"], "package metadata observation differs")
        if env["id"] == "exl3-overlay":
            A.require(env["python_from"] == "exl3-base" and names == {"exllamav3"}, "invalid EXL3 overlay")
            evidence(env["wheel"]["metadata"])
        elif env["id"] != "analysis":
            evidence(env["python"]["metadata"])
            evidence(env["torch"]["metadata"])
    build = record["build"]
    A.require(re.fullmatch("[0-9a-f]{40}", build["revision"]), "invalid engine revision")
    A.require(build["component"] == ("llama.cpp" if paper == "paper5" else "llama.cpp-p6fa") and
              build["basis"] == (historical if paper == "paper5" else current), "wrong engine build observation")
    cmake = build["cmake"]
    evidence(cmake["metadata"])
    A.require(cmake["metadata"]["basis"] == build["basis"] and
              cmake["options"]["GGML_CUDA_FA_ALL_QUANTS"] == ("OFF" if paper == "paper5" else "ON"),
              "wrong build metadata/mixed-KV flag")
    A.require(all(re.fullmatch(r"[A-Za-z0-9_.;-]+", value) for value in cmake["options"].values()), "unsafe CMake projection")
    A.require(set(cmake["compilers"]) == {"cxx", "cuda"}, "incomplete compiler metadata")
    for compiler in cmake["compilers"].values():
        evidence(compiler["metadata"])
        A.require(compiler["metadata"]["basis"] == current, "compiler inspection is not a historical capture")
    for row in build["native_artifacts"]:
        evidence(row, native=True)
    if paper == "paper5":
        A.require(record["serving"]["basis"] == historical, "wrong P5 serving observation")
        patched = record["serving"]["patched_sources"]
        A.require(len(patched) == 2 and {row["path"] for row in patched} == {
            "exllamav3/generator/generator.py", "exllamav3/architecture/qwen3_5_mtp.py"}, "incomplete EXL3 patch identities")
        for row in patched:
            evidence(row["before"])
            evidence(row["after"])
            A.require(row["before"]["basis"] == current and row["after"]["basis"] == historical,
                      "patch input/output observation boundary differs")
        for row in record["serving"]["native_artifacts"]:
            evidence(row, native=True)
    else:
        A.require(not build["native_artifacts"] and record["serving"]["basis"] == "prospective-P6-not-executed",
                  "prospective P6 metadata cannot attest execution")
    return record


def verify(directory):
    directory = Path(directory)
    inventory = tree(directory)
    manifest = A.read_json(directory / "SOURCE_MANIFEST.json")
    A.require(manifest["schema"] == SCHEMA and manifest["paper"] in TESTS and
              manifest["author_approval"] is False and manifest["status"] == "preparation-not-a-release",
              "wrong source component identity")
    files = manifest["files"]
    for name, digest in files.items():
        A.relative(name)
        A.digest(digest)
    A.require(inventory == {**files, "SOURCE_MANIFEST.json": sha((directory / "SOURCE_MANIFEST.json").read_bytes())},
              "source-component file inventory/hash mismatch")
    A.require(files.get("replication_source.py") == sha(Path(__file__).read_bytes()) and
              files.get("replication_check.py") == sha(Path(A.__file__).read_bytes()), "checker source differs from component")
    assets = A.validate(A.read_json(directory / "ASSETS.json"))
    A.require(assets["paper"] == manifest["paper"], "assets belong to another paper")
    validate_environment(A.read_json(directory / "ENVIRONMENT.json"), manifest["paper"])
    source_paths = set()
    for record in manifest["sources"]:
        output = record["output"]
        A.relative(record["source"])
        A.relative(output)
        A.digest(record["source_sha256"])
        A.require(output in files and output not in source_paths and
                  record["export_sha256"] == files[output], "invalid source projection record")
        source_paths.add(output)
        A.require(record["binding"] in ("p5-pre-primary-observation", "p5-followup-preparation",
                                        "p6-input-generator-freeze", "current-source-not-an-execution-attestation"),
                  "unknown source evidence boundary")
        A.require(type(record["changes"]) is list, "missing source transformation declaration")
        A.require(record["changes"] or record["source_sha256"] == record["export_sha256"],
                  "undeclared source transformation")
    A.require({name for name in files if name.startswith("replication/")} == source_paths,
              "unmapped experimental source")
    documents = manifest["documents"]
    A.require({row["output"] for row in documents} == {"FULL_RERUN.md", "RUNTIME.md", "THIRD_PARTY_NOTICES.md"} and
              len(documents) == 3, "incomplete document projection map")
    for row in documents:
        A.relative(row["source"])
        A.digest(row["source_sha256"])
        A.require(files[row["output"]] == row["export_sha256"] and isinstance(row["changes"], list),
                  "document projection hash/declaration differs")
        A.require(row["changes"] or row["source_sha256"] == row["export_sha256"], "undeclared document change")
    for name in files:
        if name.endswith(".py"):
            ast.parse((directory / name).read_bytes(), filename=name)
    return manifest, inventory


def selftests(directory):
    directory = Path(directory).resolve()
    manifest, before = verify(directory)
    results, entrypoints = [], []
    with tempfile.TemporaryDirectory(prefix="replication-source-tests-") as tmp:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD")}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", MPLCONFIGDIR=tmp)
        for suite, pattern in TESTS[manifest["paper"]]:
            run = subprocess.run([sys.executable, "-I", "-B", "-c", TEST_DRIVER,
                                  str(directory / "replication"), suite, pattern, json.dumps(dependency_paths())],
                                 cwd=tmp, env=env, capture_output=True, text=True, timeout=90)
            if run.returncode:
                raise ValueError("isolated source tests failed for " + suite + ":\n" + run.stderr[-6000:])
            marker = [line for line in run.stdout.splitlines() if line.startswith("SOURCE_SELFTEST_OK ")]
            A.require(len(marker) == 1, "missing isolated-test result")
            results.append({"suite": suite, "tests": int(marker[0].split()[-1]), "passed": True})
        for script, args in ENTRYPOINTS[manifest["paper"]]:
            run = subprocess.run([sys.executable, "-I", "-B", "-c", TEST_DRIVER,
                                  str(directory / "replication"), script, json.dumps(args), json.dumps(dependency_paths())],
                                 cwd=tmp, env=env, capture_output=True, text=True, timeout=30)
            A.require(run.returncode == 0 and run.stdout.splitlines()[-1:] == ["SOURCE_ENTRYPOINT_OK"],
                      "isolated entrypoint check failed for " + script + ":\n" + run.stderr[-3000:])
            entrypoints.append({"script": script, "arguments": args, "passed": True})
    A.require(tree(directory) == before, "self-tests changed source component")
    return {"paper": manifest["paper"], "tests": results, "entrypoints": entrypoints, "input_unchanged": True,
            "network_and_process_test_guard": True, "author_approval": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = selftests(args.directory)
        else:
            manifest, _ = verify(args.directory)
            result = {"paper": manifest["paper"], "source_files": len(manifest["sources"]),
                      "status": manifest["status"], "author_approval": False}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, subprocess.SubprocessError) as exc:
        parser.exit(2, "Source verification refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
