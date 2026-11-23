#!/usr/bin/env python3
"""Read-only evidence collection for P5, with explicit create-only recording.

No serving calls, model imports, launches, signals, scheduler or resource claims.
`plan` lists the selected files/bytes without hashing large files. `capture`
hashes them and records a pre-primary snapshot only while a verified owned
Tabby process is live and the primary manifest/journal are still absent.
`check` rehashes the recorded files and verifies the preserved journal prefix.
This is a dated observation, NOT retrospective proof of the earlier pilot or
summary execution, or proof of any future request's execution environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from evidence import ROOT, FROZEN_HARNESS
from draft_support import read_json, sha
from followup_results import TABBY, LLAMA, WEIGHTS
from migrate_followup_preparation import once, safe_path

GPU_NAME = "NVIDIA GeForce RTX 4080 Laptop GPU"

# Recorded, field-limited amendments to the 2026-09-20 pre-primary baseline (see
# research/P5_AMENDMENT_2026-09-25.md). Only the kernel release, the kernel-reported
# MemTotal (within 1%) and non-GPU OS system libraries can be amended; the GPU, driver,
# CPU, host, CUDA/NVIDIA libraries, engines, Python environment, models and harness never.
AMENDMENTS_REL = "paper5/execution_machine_amendments.json"
BASELINE_SNAPSHOT_REL = "testsuite/paper5/work/execution/pre-primary-20260920/SNAPSHOT.json"
AMENDABLE_MACHINE_FIELDS = ("kernel", "host_mem_kib")
AMENDABLE_FILE_DIRS = ("/usr/lib/x86_64-linux-gnu/", "/lib/x86_64-linux-gnu/")
NON_AMENDABLE_LIB = re.compile(r"nvidia|cuda|libcu|nvrtc|nvjitlink|cublas|cudnn|nccl|libtorch|python", re.I)
TRITON_CACHE_LIBRARY = re.compile(r"^/path/to/operator/\.triton/cache/[A-Z0-9]{52}/"
                                  r"(__triton_launcher|cuda_utils)\.cpython-312-x86_64-linux-gnu\.so$")


def apply_amendments(baseline, root=ROOT):
    """Deep copy of `baseline` with the recorded amendments applied. Idempotent."""
    out = json.loads(json.dumps(baseline))
    path = Path(root) / AMENDMENTS_REL
    if not path.exists():
        return out
    record = read_json(path)
    require(record.get("schema") == "p5-machine-amendments-v1", "unknown machine-amendment schema")
    snap = Path(root) / BASELINE_SNAPSHOT_REL
    if snap.exists():
        require(record.get("baseline_sha256") == sha(snap), "machine amendments bind a different baseline")
    machine = out.get("machine")
    items = record.get("amendments", [])
    for index, item in enumerate(items):
        field = item["field"]
        require(field in AMENDABLE_MACHINE_FIELDS, "machine field is not amendable: " + str(field))
        # Successive amendments of one field form a chain (29 -> 31 -> 38); a record that is
        # already at this link or a later one is left alone, so re-application is a no-op.
        later = [x["to"] for x in items[index:] if x["field"] == field]
        if machine is None or machine.get(field) in later:
            continue
        require(machine.get(field) == item["from"], "machine amendment does not start from the baseline value")
        if field == "host_mem_kib":
            require(abs(item["to"] - item["from"]) <= 0.01 * item["from"], "host_mem_kib amendment exceeds 1%")
        machine[field] = item["to"]
    files = out.get("files")
    for item in record.get("file_amendments", []):
        name = item["path"]
        require(name.startswith(AMENDABLE_FILE_DIRS) and not NON_AMENDABLE_LIB.search(name),
                "only non-GPU OS system libraries can be amended: " + name)
        if files is None or name not in files:
            continue
        # Successive updates of one library form a chain (snapshot -> v1 -> v2 ...). The live
        # file must match the latest link; outcomes recorded under an earlier link stay valid
        # through "accepted_sha256". Re-applying the same chain is a no-op.
        accepted = files[name].get("accepted_sha256") or [files[name]["sha256"]]
        if item["to_sha256"] in accepted:
            continue
        require(files[name]["sha256"] == item["from_sha256"], "file amendment does not start from the baseline value")
        files[name] = {**files[name], "sha256": item["to_sha256"], "bytes": item["to_bytes"],
                       "accepted_sha256": accepted + [item["to_sha256"]]}
    # Triton JIT cache libraries are compiled at run time by the pinned Triton. All of
    # ~/.triton was deleted by the 2026-10-25 disk incident (research/P5_AMENDMENT_2026-10-29.md).
    # A regenerated library with new bytes is accepted alongside its baseline hash; one not
    # yet regenerated is "regenerable": it may be absent, and when present must match.
    for item in record.get("cache_amendments", []):
        name = item["path"]
        require(TRITON_CACHE_LIBRARY.match(name), "only Triton JIT cache libraries can be cache-amended: " + name)
        if files is None or name not in files:
            continue
        entry = files[name]
        accepted = entry.get("accepted_sha256") or [entry["sha256"]]
        to = item.get("to_sha256")
        if to is None:
            require(item["from_sha256"] in accepted, "cache amendment does not start from the baseline value")
            files[name] = {**entry, "regenerable": True, "accepted_sha256": accepted}
            continue
        if entry["sha256"] == to and to in accepted:
            continue
        require(entry["sha256"] == item["from_sha256"], "cache amendment does not start from the baseline value")
        files[name] = {**entry, "sha256": to, "bytes": item["to_bytes"], "regenerable": True,
                       "accepted_sha256": accepted + [to]}
    return out


def machine_history(baseline, root=ROOT):
    """Machine records along the recorded amendment chain, newest first. Evidence saved under
    an earlier link (e.g. generation under kernel 7.0.0-31) stays valid; live checks use only
    the newest record. Built by walking the chain back from the amended record."""
    state = apply_amendments(baseline, root)["machine"]
    path = Path(root) / AMENDMENTS_REL
    items = read_json(path).get("amendments", []) if path.exists() else []
    states = [state]
    for item in reversed(items):
        if state.get(item["field"]) == item["to"]:
            state = {**state, item["field"]: item["from"]}
            states.append(state)
    return states


def present(name, expected):
    """False only for a regenerable Triton cache library that is currently absent."""
    return not (expected.get("regenerable") and not os.path.exists(name))


def matches(name, expected):
    digest = sha(name)
    if expected.get("regenerable"):
        return digest in expected["accepted_sha256"]
    return digest == expected["sha256"]
MODEL_SMALL = ("config.json", "generation_config.json", "quantization_config.json",
               "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "vocab.json", "merges.txt")
SOURCE_SUFFIXES = {".py", ".cu", ".cpp", ".h", ".cuh", ".jinja", ".toml"}
SAFE_ENV = {"PYTHONPATH", "EXL3_CKPT_PP", "CUDA_VISIBLE_DEVICES", "P5_REQUEST_IDLE_SECONDS"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def identity(path):
    s = Path(path).stat()
    require(stat.S_ISREG(s.st_mode), "not a regular evidence file: " + str(path))
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def file_record(path):
    before = identity(path)
    result = {**before, "sha256": sha(path)}
    require(identity(path) == before, "file changed while hashing: " + str(path))
    return result


def process_stat(text):
    end = text.rfind(")")
    require(end >= 0, "invalid process stat")
    fields = text[end + 2:].split()
    require(len(fields) >= 20, "truncated process stat")
    return {"pid": int(text.split("(", 1)[0]), "ppid": int(fields[1]), "start_ticks": int(fields[19])}


def mapped_libraries(text):
    paths = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if ".so" not in Path(path).name and "x" not in fields[1]:
            continue
        require(not path.endswith(" (deleted)"), "a mapped runtime file was deleted")
        s = Path(path).stat()
        major, minor = (int(v, 16) for v in fields[3].split(":"))
        require(stat.S_ISREG(s.st_mode) and s.st_ino == int(fields[4]) and
                os.major(s.st_dev) == major and os.minor(s.st_dev) == minor,
                "mapped runtime path no longer identifies its loaded inode: " + path)
        paths[str(Path(path).resolve())] = True
    require(paths, "no mapped runtime files found")
    return set(paths)


def listening_inodes(text, port):
    result = set()
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 10 and fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
            # IPv4 loopback only: P5 must not silently expose an unauthenticated server.
            require(fields[1].split(":")[0] == "0100007F", "study port is not bound only to IPv4 loopback")
            result.add(fields[9])
    return result


def owned_server(pid, root=ROOT):
    proc = Path("/proc") / str(pid)
    before = process_stat((proc / "stat").read_text())
    require(before["pid"] == pid, "wrong process identity")
    argv = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    expected = [str(root / "engines/exl3-env/bin/python"), "main.py", "--config", str(root / "testsuite/paper5/tabby-p5.yml")]
    require(argv == expected, "PID is not the exact P5 Tabby command")
    require((proc / "cwd").resolve() == root / "engines/tabbyAPI", "wrong server working directory")
    executable = (proc / "exe").resolve()
    require(executable == (root / "engines/exl3-env/bin/python").resolve(), "wrong running Python executable")
    listeners = listening_inodes((proc / "net/tcp").read_text(), 18125)
    require(len(listeners) == 1, "study listener is missing or ambiguous")
    fd_targets = set()
    for fd in (proc / "fd").iterdir():
        try:
            fd_targets.add(os.readlink(fd))
        except FileNotFoundError:  # another unrelated descriptor may close during inspection
            continue
    require({"socket:[" + inode + "]" for inode in listeners} <= fd_targets,
            "the supplied PID does not own the study listener")
    log = (proc / "fd/1").resolve()
    require(log.is_file() and log.is_relative_to(root / "testsuite/paper5/work/logs") and
            re.fullmatch(r"tabby-p5-\d{8}-\d{6}\.log", log.name), "server stdout is not its P5 log")
    environment = {}
    # Never persist unrelated environment variables (which may contain credentials).
    for entry in (proc / "environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        if key.decode(errors="replace") in SAFE_ENV:
            environment[key.decode()] = value.decode()
    require(environment.get("PYTHONPATH") == str(root / "engines/exl3-1.4.7-lab") and
            environment.get("EXL3_CKPT_PP") == "1024", "wrong serving-code override or checkpoint setting")
    libraries = mapped_libraries((proc / "maps").read_text())
    required_extension = str(root / "engines/exl3-1.4.7-lab/exllamav3_ext.cpython-312-x86_64-linux-gnu.so")
    require(required_extension in libraries, "the pinned native EXL3 extension is not mapped by this server")
    require(process_stat((proc / "stat").read_text()) == before, "process identity changed while inspecting")
    return {**before, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "argv": argv, "cwd": str((proc / "cwd").resolve()), "executable": str(executable),
            "environment_allowlist": environment, "listener_port": 18125, "log": str(log),
            "mapped_files": sorted(libraries)}


def startup_prefix(path, pid):
    data = bytearray()
    with Path(path).open("rb") as stream:
        for line in stream:
            data.extend(line)
            require(len(data) <= 1024 * 1024, "startup log prefix exceeds 1 MiB")
            if b"Application startup complete." in line:
                break
        else:
            raise ValueError("server startup completion is not in its log")
    text = bytes(data).decode()
    for expected in ("exllamav3 version: 1.4.7", "Using main model MTP component for drafting",
                     "Using configured max_seq_len: 32768 tokens.", "Model successfully loaded.",
                     f"Started server process [{pid}]"):
        require(expected in text, "startup log is missing: " + expected)
    return bytes(data)


def hardware():
    lines = command(["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version",
                     "--format=csv,noheader,nounits"]).splitlines()
    require(len(lines) == 1, "expected a single study GPU")
    fields = [x.strip() for x in lines[0].split(",")]
    require(len(fields) == 5 and fields[0] == GPU_NAME, "wrong study GPU")
    gpu = dict(zip(("name", "uuid", "pci_bus_id", "memory_mib", "driver"), fields))
    gpu["memory_mib"] = int(gpu["memory_mib"])
    return {"host": os.uname().nodename, "kernel": os.uname().release, "gpu": gpu,
            "cpu_model": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                              if line.startswith("model name")),
            "host_mem_kib": int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                                      if line.startswith("MemTotal:")))}


def gpu_processes():
    rows = command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits"]).splitlines()
    return [{"pid": int(p), "gpu_uuid": u.strip(), "memory_mib": int(m)}
            for p, u, m in (line.split(",") for line in rows if line.strip())]


def inventory(root=ROOT, model_base=None):
    model_base = Path(model_base or Path.home() / "models")
    pins = {str(model_base / "qwen38-exl3-2.0" / name): WEIGHTS[name]
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")}
    pins[str(model_base / "gpt-oss-20b/gpt-oss-20b-MXFP4.gguf")] = WEIGHTS["gpt-oss-20b-MXFP4.gguf"]
    paths = set(pins)
    paths.update(str(model_base / "qwen38-exl3-2.0" / name) for name in MODEL_SMALL)
    paths.update(str(root / "testsuite/paper5" / name) for name in FROZEN_HARNESS)
    paths.update(str(root / name) for name in ("testsuite/paper5/work/cap.json", "testsuite/paper5/work/journals/pilot.jsonl",
                                              "research/design-overflow-policies.md"))
    revisions = {}
    for engine, revision in (("tabbyAPI", TABBY), ("llama.cpp", LLAMA)):
        base = root / "engines" / engine
        require(command(["git", "-C", str(base), "rev-parse", "HEAD"]) == revision, "wrong pinned engine commit: " + engine)
        # Keep the pinned engine repositories clean; the EXL3 lab override is
        # separately and fully hashed below instead of claiming it is upstream.
        status = command(["git", "-C", str(base), "status", "--porcelain", "--untracked-files=normal"])
        require(not status, "pinned engine checkout is dirty: " + engine)
        revisions[engine] = revision
    tabby = root / "engines/tabbyAPI"
    # Include source files even if ignored by git, which could shadow an import.
    for base in (tabby, root / "engines/exl3-1.4.7-lab"):
        selected = [p for p in base.rglob("*") if p.is_file() and ".git" not in p.parts and
                    (p.suffix in SOURCE_SUFFIXES or ".so" in p.name or p.name in ("METADATA", "RECORD", "WHEEL"))]
        require(selected, "engine source tree is empty")
        paths.update(str(p.resolve()) for p in selected)
    binary = root / "engines/llama.cpp/build/bin"
    paths.add(str((binary / "llama-server").resolve()))
    paths.update(str(p.resolve()) for p in binary.glob("*.so*") if p.is_file())
    paths.add(str(root / "engines/llama.cpp/build/CMakeCache.txt"))
    environment = root / "engines/exl3-env"
    paths.add(str(environment / "pyvenv.cfg"))
    paths.update(str(p) for p in (environment / "lib/python3.12/site-packages").glob("*.dist-info/METADATA"))
    paths.add(str(environment / "lib/python3.12/site-packages/torch/version.py"))
    return {"files": sorted(paths), "model_pins": pins, "engine_revisions": revisions}


def prefix_record(path, limit=None):
    path = Path(path)
    before = path.stat()
    length = before.st_size if limit is None else limit
    require(type(length) is int and 0 <= length <= before.st_size, "recorded prefix exceeds journal")
    h = hashlib.sha256()
    remaining, last = length, b""
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            require(bool(block), "journal shrank during prefix read")
            h.update(block)
            remaining -= len(block)
            last = block[-1:]
    after = path.stat()
    require(before.st_ino == after.st_ino and before.st_dev == after.st_dev and after.st_size >= length,
            "journal was replaced or truncated during prefix read")
    require(length == 0 or last == b"\n", "journal prefix ends in an unfinished append; retry inspection")
    return {"path": str(path), "bytes": length, "sha256": h.hexdigest()}


def require_pre_primary(root):
    require(not (root / "testsuite/paper5/PROMPTS_MANIFEST.json").exists(), "primary prompt freeze already exists")
    require(not (root / "testsuite/paper5/work/journals/grid.jsonl").exists(), "primary generation already exists")


def capture(pid, root=ROOT):
    root = Path(root).resolve()
    require_pre_primary(root)
    destination = safe_path(root, "testsuite/paper5/work/execution/pre-primary-20260920")
    require(not destination.exists(), "snapshot destination already exists; use check, never overwrite it")
    started = utc()
    collector_sources = [Path(__file__).resolve(), ROOT / "publication/draft_support.py"]
    collector_sources += [ROOT / "paper5/scripts" / name for name in
                          ("evidence.py", "followup_results.py", "diagnostics.py", "migrate_followup_preparation.py")]
    collector_bytes = {str(p.relative_to(ROOT)): p.read_bytes() for p in collector_sources}
    before = inventory(root)
    server = owned_server(pid, root)
    machine = hardware()
    processes = gpu_processes()
    require(any(p["pid"] == pid and p["gpu_uuid"] == machine["gpu"]["uuid"] for p in processes),
            "owned study server is not present on the recorded GPU")
    require({p["pid"] for p in processes} == {pid}, "another GPU compute process is active")
    log_bytes = startup_prefix(server["log"], pid)
    require(str(Path.home() / "models/qwen38-exl3-2.0") in log_bytes.decode(), "log model path differs")
    paths = sorted(set(before["files"]) | set(server["mapped_files"]) | {server["executable"]})
    records = {}
    for path in paths:
        if Path(path).stat().st_size >= 1024**3:
            print("Hashing " + str(path), flush=True)
        records[path] = file_record(path)
    for path, expected in before["model_pins"].items():
        require(records[path]["sha256"] == expected, "model file differs from its predeclared pin: " + path)
    summaries = prefix_record(root / "testsuite/paper5/work/journals/summaries.jsonl")
    require(inventory(root) == before and hardware() == machine and owned_server(pid, root) == server,
            "inventory, machine or serving identity changed during capture")
    require(all(identity(p) == {k: v for k, v in record.items() if k != "sha256"} for p, record in records.items()),
            "a previously hashed artifact changed during capture")
    require_pre_primary(root)
    require(all(p.read_bytes() == collector_bytes[str(p.relative_to(ROOT))] for p in collector_sources),
            "collector source changed during capture")
    report = {"schema": "p5-pre-primary-execution-snapshot-v1", "started_at": started, "finished_at": utc(),
              "scope": "file and live-server observation before primary freeze/generation, after summary preparation began",
              "not_established": "earlier pilot/summary file state, future request ownership, continuous GPU exclusivity or completed measurements",
              "primary_manifest_present": False, "primary_generation_present": False,
              "inventory": before, "files": records, "machine": machine, "server": server,
              "gpu_processes_at_start": processes, "summaries_prefix": summaries,
              "startup_prefix_sha256": hashlib.sha256(log_bytes).hexdigest(), "startup_prefix_bytes": len(log_bytes),
              "collector_sources": {name: hashlib.sha256(data).hexdigest() for name, data in collector_bytes.items()}}
    for name, data in collector_bytes.items():
        once(destination / "sources" / name, data)
    once(destination / "tabby-startup-prefix.log", log_bytes)
    once(destination / "SNAPSHOT.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"record": str(destination / "SNAPSHOT.json"), "files": len(records),
            "hashed_bytes": sum(r["bytes"] for r in records.values()), "model_pins_match": True,
            "scope": report["scope"], "not_established": report["not_established"]}


def check(path, root=ROOT):
    record = read_json(path)
    require(record["schema"] == "p5-pre-primary-execution-snapshot-v1" and
            record["primary_manifest_present"] is False and record["primary_generation_present"] is False,
            "wrong snapshot schema/stage")
    require(inventory(root) == record["inventory"], "runtime inventory changed since snapshot")
    amended = apply_amendments({"machine": record["machine"], "files": record["files"]}, root)["files"]
    for name, expected in amended.items():
        if present(name, expected):
            require(matches(name, expected), "snapshotted artifact changed: " + name)
    prefix = record["summaries_prefix"]
    require(prefix_record(prefix["path"], prefix["bytes"]) == prefix, "preserved summary-journal prefix changed")
    archive = Path(path).parent
    require(all(sha(safe_path(archive / "sources", name)) == digest for name, digest in record["collector_sources"].items()) and
            sha(archive / "tabby-startup-prefix.log") == record["startup_prefix_sha256"], "snapshot archive changed")
    require(prefix_record(record["server"]["log"], record["startup_prefix_bytes"])["sha256"] == record["startup_prefix_sha256"],
            "original server startup-log prefix changed")
    return {"record": str(path), "files": len(record["files"]), "read_only": True,
            "historical_prefix_preserved": True, "inference_started": False,
            "scope": "current file hashes and preserved prefixes, not historical or future execution attestation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("plan")
    cap = sub.add_parser("capture")
    cap.add_argument("--server-pid", type=int, required=True)
    verify = sub.add_parser("check")
    verify.add_argument("record", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "capture":
            result = capture(args.server_pid)
        elif args.operation == "check":
            result = check(args.record)
        else:
            selected = inventory()
            result = {"files": len(selected["files"]), "bytes_before_live_runtime_files": sum(Path(p).stat().st_size for p in selected["files"]),
                      "model_pins": selected["model_pins"], "engine_revisions": selected["engine_revisions"],
                      "hashing_started": False, "inference_started": False}
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"P5 execution evidence stopped: {exc}\n")


if __name__ == "__main__":
    main()
