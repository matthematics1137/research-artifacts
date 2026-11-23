#!/usr/bin/env python3
"""Paper 6: run cells (model x weight tier x KV type) on llama-server, the FA_ALL_QUANTS
build of the pinned commit (2026-09-19 amendment §2-6).

  run_cells.py --cells qwen38-27b:Q2_K_XL:f16              # one cell
  run_cells.py --cells 'qwen3-8b:*:*'                       # fnmatch wildcards, comma lists
  run_cells.py --cells all [--tasks longctx,gsm8k]          # the 24 cells, canonical order
  run_cells.py --cells all --dry-run                        # plan only: no server, no GPU
  run_cells.py --selftest                                   # CPU-only checks, no server

Per cell: one llama-server (our own process group) with
  -m <gguf> -c 20480 --fit on -fa on -ctk <K> -ctv <V> -np 1 --jinja --no-webui
  --host localhost --port 18130                     (k8v4 = -ctk q8_0 -ctv q4_0)
then the 200 long-context items (thinking off, max 128 tokens) and GSM8K-100 (thinking on:
reasoning_effort medium for Qwen3.8, the template default for Qwen3-8B; max 8,192), all at
temperature 0 with cache_prompt false, in a fixed order. Journal per cell:
work/journals/<model>__<tier>__<kv>.jsonl, append-only and resumable (a job with a terminal
row is never re-sent unless --retry-failed and that row failed). Every server start appends
a `server_start` identity row: GGUF path + SHA-256, argv, engine commit and build flags, the
layer count --fit chose, the reported model/KV/RS/compute buffers (raw log lines kept), the
process's nvidia-smi memory, and /props.

Operational rules are Paper 5's and adjustable: touch testsuite/paper6/PAUSE to pause (the
server is stopped while paused); GPU >= --temp-stop C pauses until < --temp-resume C.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6common as P  # noqa: E402

sys.path.insert(0, P.PAPER5)
import p5run  # noqa: E402  (server process-group handling, Runner base, http_json, log, gpu_temp)
from p5run import log  # noqa: E402

HARNESS_FILES = [os.path.join(HERE, f) for f in ("p6common.py", "make_longctx.py", "run_cells.py", "analyze.py", "p6evidence.py")] + \
    [os.path.join(P.PAPER5, "p5run.py"), os.path.join(P.PAPER5, "p5common.py"), os.path.join(P.EVALS, "run_eval.py"),
     os.path.join(P.REPO, "research", "design-kv-allocation.md")]
TASKS = ("longctx", "gsm8k")
DEFAULT_TIMEOUT = {"longctx": 1800, "gsm8k": 5400}      # seconds per request (operational)
SHA_CACHE = os.path.join(P.WORK, "gguf_sha_cache.json")
TOKEN_COUNTS = os.path.join(P.WORK, "token_counts")


def harness_hashes() -> dict:
    return {os.path.relpath(f, P.REPO): P.sha256_file(f) for f in HARNESS_FILES}


def placement_signature(load: dict) -> dict:
    """Placement, not transient process memory: check every restart, including the first pass."""
    return {k: load.get(k) for k in ("offloaded_layers", "total_layers", "fully_resident", "kv_cache", "recurrent")} | {
        "buffers": {k: (load.get("buffers_mib") or {}).get(k) for k in ("model", "kv", "rs")},
        "overflowing": (load.get("fit") or {}).get("overflowing"),
    }


def assert_study_ports_idle():
    """Never overlap any P5 generator/judge, including its follow-ups or paused phase units.

    This does not replace the operator's live candidate-machine/headroom check.
    """
    for port in (18125, 18126, 18127, 18130):
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex(("localhost", port)) == 0:
                raise SystemExit(f"study port {port} is occupied; P6 must run after P5 GPU work")
    for unit in ("p5-phase1.service", "p5-phase2.service"):
        check = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10)
        if check.stdout.strip() in ("active", "activating", "reloading"):
            raise SystemExit(f"{unit} is still active; P6 must run after P5 GPU work")


# ---------------------------------------------------------------- cells
def parse_cells(spec: str) -> list[tuple[str, str, str]]:
    grid = P.all_cells()
    if spec.strip() == "all":
        return grid
    out = []
    for part in [s.strip() for s in spec.split(",") if s.strip()]:
        bits = part.split(":")
        if len(bits) != 3:
            raise SystemExit(f"bad cell {part!r}: want model:tier:kv")
        hit = [c for c in grid if all(fnmatch.fnmatchcase(v, pat) for v, pat in zip(c, bits))]
        if not hit:
            raise SystemExit(f"cell {part!r} matches nothing; models {list(P.MODELS)}, tiers {P.TIERS}, kv {P.KV_ORDER}")
        out += [c for c in hit if c not in out]
    return [c for c in grid if c in out]                  # canonical order whatever the spec order


def server_argv(model: str, tier: str, kv: str) -> list[str]:
    ctk, ctv = P.KV_TYPES[kv]
    return [P.SERVER_EXE, "-m", P.gguf_path(model, tier), "-c", str(P.N_CTX), "--fit", "on", "-fa", "on",
            "-ctk", ctk, "-ctv", ctv, "-np", "1", "--jinja", "--no-webui",
            "--host", "localhost", "--port", str(P.PORT)]


# ---------------------------------------------------------------- requests
def request_body(model: str, task: str, content: str) -> dict:
    body = {"messages": [{"role": "user", "content": content}], "stream": False,
            "temperature": 0.0, "cache_prompt": False}
    if task == "longctx":
        body["max_tokens"] = P.LONGCTX_MAX_TOKENS
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif task == "gsm8k":
        body["max_tokens"] = P.GSM8K_MAX_TOKENS
        if model == P.HYBRID:
            body["chat_template_kwargs"] = {"reasoning_effort": "medium"}
        # Qwen3-8B: no kwargs -> the template's default (thinking on)
    else:
        raise ValueError(task)
    return body


def redact(body: dict, item_id: str) -> dict:
    """The journal keeps long-context prompts by hash (the frozen item file holds the text)."""
    b = json.loads(json.dumps(body))
    c = b["messages"][0]["content"]
    b["messages"][0]["content"] = {"item_id": item_id, "sha256": P.sha256_text(c), "chars": len(c)}
    return b


def build_jobs(model: str, tasks, longctx: list[dict], gsm8k: list[dict], timeouts: dict,
               limit: int = 0) -> list[dict]:
    jobs = []
    if "longctx" in tasks:
        for it in (longctx[:limit] if limit else longctx):
            body = request_body(model, "longctx", it["content"])
            jobs.append({"job_id": P.job_id("longctx", it["id"]), "task": "longctx", "item": it, "body": body,
                         "stored_request": redact(body, it["id"]), "timeout": timeouts["longctx"],
                         "meta": {"family": it["family"], "content_sha256": it["sha256"],
                                  "haystack_tokens_qwen38": it["haystack_tokens"],
                                  "prompt_tokens_expected": it["prompt_tokens_qwen38_nothink"]
                                  if model == P.HYBRID else None}})
    if "gsm8k" in tasks:
        for it in (gsm8k[:limit] if limit else gsm8k):
            body = request_body(model, "gsm8k", P.gsm8k_prompt(it))
            jobs.append({"job_id": P.job_id("gsm8k", it["id"]), "task": "gsm8k", "item": it, "body": body,
                         "stored_request": body, "timeout": timeouts["gsm8k"], "meta": {}})
    return jobs


# ---------------------------------------------------------------- server log
RX = {
    "build": re.compile(r"build (\d+) \(([0-9a-f]+)\)"),
    "offloaded": re.compile(r"offloaded (\d+)/(\d+) layers to GPU"),
    "model_buf": re.compile(r"(\S+) model buffer size\s*=\s*([\d.]+) MiB"),
    "kv_buf": re.compile(r"(\S+) KV buffer size\s*=\s*([\d.]+) MiB"),
    "rs_buf": re.compile(r"(\S+) RS buffer size\s*=\s*([\d.]+) MiB"),
    "compute_buf": re.compile(r"(\S+) compute buffer size\s*=\s*([\d.]+) MiB"),
    "output_buf": re.compile(r"(\S+)\s+output buffer size\s*=\s*([\d.]+) MiB"),
    "kv_summary": re.compile(r"llama_kv_cache\w*: size =\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers.*?"
                             r"K \((\w+)\):\s*([\d.]+) MiB, V \((\w+)\):\s*([\d.]+) MiB"),
    "kv_self": re.compile(r"KV self size\s*=\s*([\d.]+) MiB(?:, K \((\w+)\):\s*([\d.]+) MiB, V \((\w+)\):\s*([\d.]+) MiB)?"),
    "rs_summary": re.compile(r"llama_memory_recurrent: size =\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers.*?"
                             r"R \((\w+)\):\s*([\d.]+) MiB, S \((\w+)\):\s*([\d.]+) MiB"),
    "n_ctx": re.compile(r"llama_context: n_ctx\s*=\s*(\d+)"),
    "flash_attn": re.compile(r"llama_context: flash_attn\s*=\s*(\S+)"),
    "fit_line": re.compile(r"common_params_fit_impl:|common_fit_params:|common_memory_breakdown_print:|fitting params"),
    "fit_overflow": re.compile(r"-\s*(\S+):\s*(\d+) layers \(\s*(\d+) overflowing\)"),
}
DEVICE = re.compile(r"^(CUDA|ROCm|Vulkan|Metal|SYCL)\d+$")


def _is_device(name: str) -> bool:
    return bool(DEVICE.match(name))


def parse_server_log(text: str) -> dict:
    """Load-time facts from a llama-server log. Buffer lines are read from the LAST
    'offloaded X/Y layers' line on (a --fit dry run or an earlier load cannot double count);
    every matching line is also kept verbatim under 'raw'."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if RX["offloaded"].search(ln)]
    start = starts[-1] if starts else 0
    out = {"build": None, "offloaded_layers": None, "total_layers": None, "n_ctx": None, "flash_attn": None,
           "buffers_mib": {"model": {}, "kv": {}, "rs": {}, "compute": {}, "output": {}},
           "kv_cache": None, "recurrent": None,
           "fit": {"status": None, "changed": None, "overflowing": None, "lines": []}, "raw": []}
    for i, ln in enumerate(lines):
        if RX["fit_line"].search(ln):
            out["fit"]["lines"].append(ln.strip())
            if "successfully fit params" in ln:
                out["fit"]["status"] = "success"
            elif "failed to fit params" in ln:
                out["fit"]["status"] = "failed"
            elif "encountered an error while trying to fit" in ln:
                out["fit"]["status"] = "error"
            if "no changes needed" in ln:
                out["fit"]["changed"] = False if out["fit"]["changed"] is None else out["fit"]["changed"]
            if "need to reduce device memory" in ln or "reduced from" in ln:
                out["fit"]["changed"] = True
            m = RX["fit_overflow"].search(ln)
            if m:
                out["fit"]["overflowing"] = max(out["fit"]["overflowing"] or 0, int(m.group(3)))
        m = RX["build"].search(ln)
        if m and out["build"] is None:
            out["build"] = {"number": int(m.group(1)), "commit": m.group(2)}
            out["raw"].append(ln.strip())
        if i < start:
            continue
        hit = False
        m = RX["offloaded"].search(ln)
        if m:
            out["offloaded_layers"], out["total_layers"] = int(m.group(1)), int(m.group(2))
            hit = True
        for key, rx in (("model", "model_buf"), ("kv", "kv_buf"), ("rs", "rs_buf"),
                        ("compute", "compute_buf"), ("output", "output_buf")):
            m = RX[rx].search(ln)
            if m:
                d = out["buffers_mib"][key]
                d[m.group(1)] = round(d.get(m.group(1), 0.0) + float(m.group(2)), 2)
                hit = True
        m = RX["kv_summary"].search(ln)
        if m:
            out["kv_cache"] = {"total_mib": float(m.group(1)), "cells": int(m.group(2)), "layers": int(m.group(3)),
                               "type_k": m.group(4), "k_mib": float(m.group(5)),
                               "type_v": m.group(6), "v_mib": float(m.group(7))}
            hit = True
        elif RX["kv_self"].search(ln):
            m = RX["kv_self"].search(ln)
            out["kv_cache"] = {"total_mib": float(m.group(1)), "cells": None, "layers": None,
                               "type_k": m.group(2), "k_mib": float(m.group(3)) if m.group(3) else None,
                               "type_v": m.group(4), "v_mib": float(m.group(5)) if m.group(5) else None,
                               "legacy_line": True}
            hit = True
        m = RX["rs_summary"].search(ln)
        if m:
            out["recurrent"] = {"total_mib": float(m.group(1)), "cells": int(m.group(2)), "layers": int(m.group(3)),
                                "type_r": m.group(4), "r_mib": float(m.group(5)),
                                "type_s": m.group(6), "s_mib": float(m.group(7))}
            hit = True
        for key in ("n_ctx", "flash_attn"):
            m = RX[key].search(ln)
            if m:
                out[key] = int(m.group(1)) if key == "n_ctx" else m.group(1)
                hit = True
        if hit:
            out["raw"].append(ln.strip())
    out["fit"]["lines"] = out["fit"]["lines"][:80]
    dev = {k: round(sum(v for n, v in d.items() if _is_device(n)), 2) for k, d in out["buffers_mib"].items()}
    host = {k: round(sum(v for n, v in d.items() if not _is_device(n)), 2) for k, d in out["buffers_mib"].items()}
    dev["total_model_kv_rs_compute"] = round(dev["model"] + dev["kv"] + dev["rs"] + dev["compute"], 2)
    out["device_mib"], out["host_mib"] = dev, host
    out["all_layers_on_gpu"] = (out["offloaded_layers"] is not None and out["offloaded_layers"] == out["total_layers"])
    # Fully resident: every layer offloaded, no KV or recurrent state on the host, and no
    # layer split by --fit (the token embedding always stays on the host in llama.cpp).
    out["fully_resident"] = bool(out["all_layers_on_gpu"] and host["kv"] == 0 and host["rs"] == 0
                                 and not out["fit"]["overflowing"])
    return out


# ---------------------------------------------------------------- nvidia-smi
def parse_nvidia(apps_text: str, gpu_text: str, pid: int | None) -> dict:
    def num(s):
        s = s.strip()
        return int(float(s)) if re.fullmatch(r"[\d.]+", s) else None

    procs = []
    for ln in (apps_text or "").splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]), "name": ",".join(parts[1:-1]), "mib": num(parts[-1])})
    g = [p.strip() for p in (gpu_text or "").strip().splitlines()[0].split(",")] if (gpu_text or "").strip() else []
    mine = [p for p in procs if pid is not None and p["pid"] == pid]
    return {"process_mib": mine[0]["mib"] if mine else None, "processes": procs,
            "other_processes_mib": sum(p["mib"] or 0 for p in procs if p["pid"] != pid),
            "gpu_used_mib": num(g[0]) if len(g) > 0 else None, "gpu_total_mib": num(g[1]) if len(g) > 1 else None,
            "gpu_free_mib": num(g[2]) if len(g) > 2 else None, "temp_c": num(g[3]) if len(g) > 3 else None}


def nvidia_query(pid: int | None) -> dict:
    def run(args):
        try:
            return subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=20).stdout
        except Exception as e:  # recorded, never fatal
            return f"error: {e!r}"
    apps = run(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    gpu = run(["--query-gpu=memory.used,memory.total,memory.free,temperature.gpu", "--format=csv,noheader,nounits"])
    d = parse_nvidia(apps, gpu, pid)
    d["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return d


# ---------------------------------------------------------------- identity
def _git(args, cwd):
    try:
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"error: {e!r}"


def engine_identity() -> dict:
    cm = {}
    cache = os.path.join(P.ENGINE_DIR, "build", "CMakeCache.txt")
    if os.path.exists(cache):
        with open(cache) as f:
            for ln in f:
                m = re.match(r"^(GGML_CUDA_FA_ALL_QUANTS|GGML_CUDA|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|"
                             r"BUILD_SHARED_LIBS|CMAKE_CUDA_COMPILER):\w+=(.*)$", ln.strip())
                if m:
                    cm[m.group(1)] = m.group(2)
    libs = {}
    bindir = os.path.dirname(P.SERVER_EXE)
    for name in ("llama-server", "libllama.so", "libllama-common.so", "libggml-cuda.so", "libggml-base.so", "libggml.so", "libggml-cpu.so"):
        p = os.path.join(bindir, name)
        if os.path.exists(p):
            libs[name] = P.sha256_file(os.path.realpath(p))
    commit = _git(["rev-parse", "HEAD"], P.ENGINE_DIR)
    dirty = _git(["status", "--porcelain", "--untracked-files=no"], P.ENGINE_DIR)
    return {"dir": os.path.relpath(P.ENGINE_DIR, P.REPO), "commit": commit, "commit_is_pinned": commit == P.ENGINE_COMMIT,
            "tracked_changes": len([x for x in dirty.splitlines() if x.strip()]), "cmake": cm, "sha256": libs}


def lab_identity() -> dict:
    return {"repo_commit": _git(["rev-parse", "HEAD"], P.REPO),
            "paper6_uncommitted": [x for x in _git(["status", "--porcelain", "testsuite/paper6"], P.REPO).splitlines()
                                   if x.strip() and "/work/" not in x][:40]}


def machine_identity() -> dict:
    """Capture study hardware, not a claim of present compute availability."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,memory.total,driver_version",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
    rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise SystemExit("P6 requires a single recorded study GPU")
    name, gpu_uuid, pci, memory, driver = [s.strip() for s in rows[0]]
    if name != "NVIDIA GeForce RTX 4080 Laptop GPU":
        raise SystemExit(f"P6's registered hardware is the RTX 4080 Laptop GPU, not {name}")
    return {"gpu_name": name, "gpu_uuid": gpu_uuid, "gpu_pci_bus": pci,
            "gpu_total_mib": int(memory), "driver_version": driver}


def gguf_identity(path: str, do_hash: bool = True, cache_path: str = SHA_CACHE) -> dict:
    """Receipt (downloads.jsonl or the amendment-named pin) plus the file's own SHA-256.
    The computed hash is cached by (path, size, mtime) so a file is read once, not per cell."""
    st = os.stat(path)
    rec = P.receipt_for(path)
    d = {"path": path, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
         "receipt": rec, "sha256": None,
         "verified": False}
    if do_hash:
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
        c = cache.get(path)
        if (c and c["bytes"] == st.st_size and c["mtime_ns"] == st.st_mtime_ns
                and c.get("ctime_ns") == st.st_ctime_ns):
            d["sha256"], d["sha256_from_cache"] = c["sha256"], True
        else:
            t0 = time.time()
            d["sha256"] = P.sha256_file(path)
            after = os.stat(path)
            if (st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise SystemExit(f"GGUF changed while hashing: {path}")
            cache[path] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
                           "sha256": d["sha256"]}
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=1)
            log(f"hashed {os.path.basename(path)} in {time.time() - t0:.0f}s")
        d["verified"] = bool(rec and rec["sha256"] == d["sha256"])
    return d


def items_identity() -> dict:
    with open(P.LONGCTX_MANIFEST) as f:
        man = json.load(f)
    got = P.sha256_file(P.LONGCTX_ITEMS)
    if got != man["output"]["sha256"]:
        raise SystemExit(f"{P.LONGCTX_ITEMS} does not match longctx_manifest.json ({got[:16]} vs "
                         f"{man['output']['sha256'][:16]}): the frozen items changed")
    return {"longctx_sha256": got, "longctx_manifest_sha256": P.sha256_file(P.LONGCTX_MANIFEST),
            "longctx_seed": man["seed"], "gsm8k_sha256": P.GSM8K_SHA256}


# ---------------------------------------------------------------- server
class P6Server(p5run._Server):
    """llama-server from engines/llama.cpp-p6fa, started as our own process group
    (stop() is Paper 5's: SIGINT, SIGTERM, SIGKILL to our pgid only)."""
    name = "llama-p6"
    port = P.PORT

    def __init__(self, cell: tuple[str, str, str], on_start=None, log_dir: str = P.LOGS):
        super().__init__()
        self.cell, self.on_start, self.log_dir = cell, on_start, log_dir
        self.starts, self.fresh, self.load = 0, False, None
        self.instance_id = None

    def cmd_env_cwd(self):
        env = dict(os.environ)
        # This pinned build links against the local CUDA 12.6 runtime. Scope
        # the path to the owned child; do not change the caller or other jobs.
        env["LD_LIBRARY_PATH"] = os.path.join(os.path.expanduser("~"), "cuda-12.6", "lib64")
        # Pinned revision 035e227 routes the library's info messages (offloaded layers, buffer
        # sizes, which parse_server_log needs) to log level 4; the default is 3. Logging only;
        # the argv is unchanged. Found by Paper 5's judge on 2026-10-29, before any P6 cell.
        env["LLAMA_ARG_LOG_VERBOSITY"] = "4"
        return ["nice", "-n", "5"] + server_argv(*self.cell), env, P.REPO

    def start(self, timeout_s: float = 900):
        if self.proc and self.proc.poll() is None:
            return
        assert_study_ports_idle()
        cmd, env, cwd = self.cmd_env_cwd()
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{P.cell_name(*self.cell)}-{time.strftime('%Y%m%d-%H%M%S')}.log")
        lf = open(self.log_path, "ab")
        pre = nvidia_query(None)
        self.proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.starts += 1
        log(f"{self.name}: started pid {self.proc.pid} ({P.cell_name(*self.cell)}), log {self.log_path}")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise SystemExit(f"{self.name} exited during startup (code {self.proc.returncode}); see {self.log_path}")
            if self.healthy():
                break
            time.sleep(2)
        else:
            self.stop()
            raise SystemExit(f"{self.name} not healthy after {timeout_s}s")
        log(f"{self.name}: healthy after {time.time() - t0:.0f}s")
        with open(self.log_path, "rb") as f:
            load_log = f.read()
        import hashlib
        self.load = parse_server_log(load_log.decode("utf-8", errors="replace"))
        self.instance_id = uuid.uuid4().hex
        info = {"cmd": cmd, "log_path": os.path.relpath(self.log_path, P.REPO), "start_seq": self.starts,
                "instance_id": self.instance_id, "pid": self.proc.pid,
                "load_log_bytes": len(load_log), "load_log_sha256": hashlib.sha256(load_log).hexdigest(),
                "load_seconds": round(time.time() - t0, 1), "load": self.load,
                "gpu_before_start": pre, "gpu_loaded": nvidia_query(self.proc.pid), "props": self.props()}
        self.fresh = True
        log(f"{self.name}: layers {self.load['offloaded_layers']}/{self.load['total_layers']}, "
            f"device MiB {self.load['device_mib']}, fully resident {self.load['fully_resident']}, "
            f"nvidia-smi {info['gpu_loaded']['process_mib']} MiB")
        if self.on_start:
            self.on_start(info)

    def props(self) -> dict:
        try:
            pr = p5run.http_json(self.base + "/props")
        except Exception as e:
            return {"error": repr(e)}
        tmpl = pr.get("chat_template") or ""
        dgs = pr.get("default_generation_settings") or {}
        return {"build_info": pr.get("build_info"), "model_path": pr.get("model_path"),
                "total_slots": pr.get("total_slots"), "n_ctx": dgs.get("n_ctx"),
                "chat_template_sha256": P.sha256_text(tmpl), "chat_template_chars": len(tmpl)}

    def identity(self) -> dict:
        return {"pid": self.proc.pid if self.proc else None, "start_seq": self.starts,
                "instance_id": self.instance_id}


# ---------------------------------------------------------------- runner
class P6Runner(p5run.Runner):
    """Paper 5's append-only journal (resume, fsync) with a chat-completions job that keeps
    the full response (content, reasoning_content, usage, timings, finish_reason) and its
    score. `post`, `temp_fn` and `sleep` are injectable for the CPU self-test."""

    def __init__(self, server, journal: str, retry_failed: bool = False, post=None, temp_fn=None,
                 sleep=time.sleep, pause_file: str = P.PAUSE_FILE, temp_stop: int = p5run.TEMP_STOP_C,
                 temp_resume: int = p5run.TEMP_RESUME_C, poll_s: float = 30, retries: int = 2,
                 idle_seconds: float = 10):
        super().__init__(server, journal)
        self.hashes = harness_hashes()
        if retry_failed:
            self.done = {k: v for k, v in self.done.items() if v.get("ok")}
        self.post = post or p5run.http_json
        self.temp_fn = temp_fn or p5run.gpu_temp
        self.sleep, self.pause_file, self.poll_s, self.retries = sleep, pause_file, poll_s, retries
        self.temp_stop, self.temp_resume = temp_stop, temp_resume
        if not isinstance(idle_seconds, (int, float)) or not 0 <= idle_seconds <= 3600:
            raise ValueError("idle_seconds must be between 0 and 3600")
        self.idle_seconds = idle_seconds

    def event(self, kind: str, **kw):
        self._append({"kind": kind, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw})

    def gate(self):
        """Pause file and thermal pause (with hysteresis): stop our server, wait, restart."""
        paused, hot = False, False
        while True:
            t = self.temp_fn()
            if os.path.exists(self.pause_file):
                why = "pause file"
            elif t >= self.temp_stop or (hot and t >= self.temp_resume):
                why, hot = f"GPU {t} C", True
            else:
                break
            if not paused:
                log(f"pausing ({why}); stopping {self.server.name}")
                self.server.stop()
                self.event("pause", reason=why)
                paused = True
            self.sleep(self.poll_s)
        if paused:
            log("resuming")
            self.event("resume")
        if not (self.server.proc and self.server.proc.poll() is None):
            self.server.start()

    def run(self, job: dict) -> dict:
        if job["job_id"] in self.done:
            return self.done[job["job_id"]]
        base = {"job_id": job["job_id"], "task": job["task"], "item_id": job["item"]["id"]}
        last_err = None
        for attempt in range(self.retries + 1):
            self.gate()
            t0 = time.time()
            try:
                resp = self.post(self.server.base + "/v1/chat/completions", job["body"], timeout=job["timeout"])
                choice = resp["choices"][0]
                msg = choice.get("message") or {}
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read()[:2000].decode("utf-8", "replace")
                except Exception:
                    detail = ""
                last_err = f"HTTP {e.code}: {detail}"
                self._append({**base, "terminal": False, "attempt": attempt, "error": last_err,
                              "seconds": round(time.time() - t0, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                log(f"{job['job_id']}: {last_err[:200]}")
                if 400 <= e.code < 500:
                    break                                  # a rejected request will be rejected again
                self.server.stop()
                self.sleep(self.idle_seconds)
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, KeyError, IndexError,
                    ValueError) as e:
                last_err = repr(e)
                self._append({**base, "terminal": False, "attempt": attempt, "error": last_err,
                              "seconds": round(time.time() - t0, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                log(f"{job['job_id']}: error {last_err[:200]}; restarting server")
                self.server.stop()
                self.sleep(self.idle_seconds)
                continue
            dt = time.time() - t0
            usage = resp.get("usage") or {}
            row = {**base, "terminal": True, "ok": True, "attempt": attempt,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(dt, 3),
                   "request": job["stored_request"], "request_sha256": P.sha256_text(json.dumps(job["body"], sort_keys=True)),
                   "content": msg.get("content"), "reasoning_content": msg.get("reasoning_content"),
                   "finish_reason": choice.get("finish_reason"), "usage": usage, "timings": resp.get("timings"),
                   "system_fingerprint": resp.get("system_fingerprint"), "response_model": resp.get("model"),
                   "meta": dict(job["meta"]), "server": self.server.identity(), "harness": self.hashes}
            exp = job["meta"].get("prompt_tokens_expected")
            if exp is not None and usage.get("prompt_tokens") is not None:
                row["meta"]["prompt_tokens_delta"] = usage["prompt_tokens"] - exp   # recorded, not a gate
            row["score"] = P.score_row(job["task"], row, job["item"])
            self._append(row)
            self.done[job["job_id"]] = row
            if getattr(self.server, "fresh", False):
                self.server.fresh = False
                self.event("gpu_after_first_request", start_seq=self.server.identity()["start_seq"],
                           gpu=nvidia_query(self.server.proc.pid if self.server.proc else None))
            self.sleep(self.idle_seconds)
            return row
        row = {**base, "terminal": True, "ok": False, "error": last_err, "attempt": attempt,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "request": job["stored_request"],
               "meta": dict(job["meta"]), "harness": self.hashes}
        self._append(row)
        self.done[job["job_id"]] = row
        self.sleep(self.idle_seconds)
        return row


# ---------------------------------------------------------------- per-model token counts
def tokenize_pass(server, model: str, longctx: list[dict], post=None, counts_dir=None) -> str:
    """Count every haystack and full user message with the served model's own tokenizer
    (llama-server /tokenize, CPU). Gives the Qwen3-8B count the amendment asks to report,
    and a llama.cpp-vs-tokenizers parity check for Qwen3.8. Written once per model."""
    post = post or p5run.http_json
    counts_dir = TOKEN_COUNTS if counts_dir is None else counts_dir
    path = os.path.join(counts_dir, f"{model}.json")
    if os.path.exists(path):
        import p6evidence as E
        E.validate_token_counts(E.read_json(path), model, longctx)
        return path
    counts = {}
    for it in longctx:
        a, b = it["haystack_span"]
        hay = post(server.base + "/tokenize", {"content": it["content"][a:b], "add_special": False})
        full = post(server.base + "/tokenize", {"content": it["content"], "add_special": False})
        counts[it["id"]] = {"haystack": len(hay["tokens"]), "content": len(full["tokens"])}
    os.makedirs(counts_dir, exist_ok=True)
    payload = {"schema": "p6-token-counts-v1", "model": model, "cell": P.cell_name(*server.cell),
               "items_sha256": P.sha256_file(P.LONGCTX_ITEMS), "server": server.identity(),
               "harness": harness_hashes(), "props": server.props(), "items": counts}
    # Create-only, never clobber an existing model's inventory. Written in place (no temporary file to
    # remove: agents on this machine never delete files); a torn write fails validation on the next read.
    with open(path, "x") as f:
        json.dump(payload, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    log(f"token counts for {model} -> {path}")
    return path


# ---------------------------------------------------------------- cells
def run_cell(cell, tasks, a) -> dict:
    model, tier, kv = cell
    name = P.cell_name(*cell)
    journal = P.journal_path(*cell, journals=a.journal_dir)
    longctx = P.load_longctx() if "longctx" in tasks else []
    gsm8k = P.load_gsm8k() if "gsm8k" in tasks else []
    jobs = build_jobs(model, tasks, longctx, gsm8k, {"longctx": a.timeout_longctx, "gsm8k": a.timeout_gsm8k}, a.limit)
    gguf = P.gguf_path(model, tier)
    if a.dry_run:
        done = P.read_journal(journal)["outcomes"]
        todo = [j for j in jobs if j["job_id"] not in done or (a.retry_failed and not done[j["job_id"]].get("ok"))]
        print(f"\n== {name}: {len(jobs)} jobs, {len(todo)} to run -> {journal}")
        print("   argv:", " ".join(["nice", "-n", "5"] + server_argv(*cell)))
        print(f"   gguf exists: {os.path.exists(gguf)}; receipt: {P.receipt_for(gguf)}")
        seen = set()
        for j in jobs:
            if j["task"] not in seen:
                seen.add(j["task"])
                print(f"   first {j['task']} request:", json.dumps(j["stored_request"])[:600])
        return {"cell": name, "dry_run": True, "todo": len(todo)}

    if not os.path.exists(P.SERVER_EXE):
        raise SystemExit(f"{P.SERVER_EXE} not built yet")
    if not os.path.exists(gguf):
        raise SystemExit(f"{gguf} missing")
    eng = engine_identity()
    if not eng["commit_is_pinned"] or eng["cmake"].get("GGML_CUDA_FA_ALL_QUANTS") != "ON":
        raise SystemExit(f"engine is not the pinned FA_ALL_QUANTS build: {eng['commit']} {eng['cmake']}")
    if eng["tracked_changes"]:
        raise SystemExit("P6 requires an unmodified pinned engine source tree")
    gid = gguf_identity(gguf, do_hash=not a.no_hash)
    if gid["receipt"] is None:
        raise SystemExit(f"no pinned SHA-256 for {gguf} (downloads.jsonl receipt missing)")
    if gid["sha256"] is not None and not gid["verified"]:
        raise SystemExit(f"SHA-256 mismatch for {gguf}: {gid['sha256']} vs receipt {gid['receipt']['sha256']}")
    if (gid["receipt"]["repo"], gid["receipt"]["revision"]) != (P.MODELS[model]["repo"], P.MODELS[model]["revision"]):
        raise SystemExit("GGUF receipt is not from the registered model repository/revision")
    pin = P.PINNED_SHA256.get(os.path.basename(gguf))
    if pin and gid["receipt"]["sha256"] != pin[0]:
        raise SystemExit("GGUF receipt conflicts with the amendment's pre-existing artifact pin")
    if not a.no_hash:
        import p6evidence as E
        E.audit_journal(journal, cell, P.load_longctx(), P.load_gsm8k(), complete=False)
    static = {"cell": {"model": model, "tier": tier, "kv": kv, "ctk": P.KV_TYPES[kv][0], "ctv": P.KV_TYPES[kv][1],
                       "arch": P.MODELS[model]["arch"]},
              "gguf": gid, "argv": server_argv(*cell), "engine": eng, "lab": lab_identity(), "machine": machine_identity(),
              "inputs": items_identity(), "harness": harness_hashes(),
              "decoding": {"temperature": 0.0, "cache_prompt": False, "longctx_max_tokens": P.LONGCTX_MAX_TOKENS,
                           "gsm8k_max_tokens": P.GSM8K_MAX_TOKENS,
                           "idle_seconds": a.idle_seconds,
                           "timeouts_s": {"longctx": a.timeout_longctx, "gsm8k": a.timeout_gsm8k}}}
    first = P.read_journal(journal)["identity"]
    first_placement = placement_signature(first[0]["load"]) if first else None

    runner = None

    def on_start(info):
        nonlocal first_placement
        current = placement_signature(info["load"])
        changed = first_placement is not None and current != first_placement
        if first_placement is None:
            first_placement = current
        runner.event("server_start", **static, **info, placement_changed=changed)
        if changed:
            raise SystemExit(f"{name}: placement changed on restart; inspect before resuming")
        if not a.no_hash:
            E.validate_identity({**static, **info, "placement_changed": False}, cell)

    server = P6Server(cell, on_start=on_start)
    runner = P6Runner(server, journal, retry_failed=a.retry_failed, temp_stop=a.temp_stop,
                      temp_resume=a.temp_resume, idle_seconds=a.idle_seconds)
    todo = [j for j in jobs if j["job_id"] not in runner.done]
    log(f"== {name}: {len(jobs)} jobs, {len(todo)} to run -> {journal}")
    if not todo:
        return {"cell": name, "ran": 0}
    n_ok = n_correct = 0
    t0 = time.time()
    try:
        runner.gate()                                       # starts the server
        if "longctx" in tasks and not a.limit:
            tokenize_pass(server, model, longctx, counts_dir=os.path.join(os.path.dirname(a.journal_dir), "token_counts"))
        for n, job in enumerate(todo):
            row = runner.run(job)
            if row.get("ok"):
                n_ok += 1
                n_correct += bool(row["score"]["correct"])
            if (n + 1) % 10 == 0 or n + 1 == len(todo):
                log(f"{name}: {n + 1}/{len(todo)} this pass, ok {n_ok}, correct {n_correct}, "
                    f"{(time.time() - t0) / (n + 1):.1f} s/job; last {job['job_id']} "
                    f"{row.get('seconds', '-')}s {str((row.get('score') or {}).get('visible_answer', ''))[:60]!r}")
    finally:
        server.stop()
        runner.event("pass_end", ran=n_ok, correct=n_correct, seconds=round(time.time() - t0, 1))
    return {"cell": name, "ran": len(todo), "ok": n_ok, "correct": n_correct}


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def check(name, got, want=True):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got={got!r} want={want!r}"), flush=True)
        if not ok:
            fails.append(name)

    print("selftest: cells and argv")
    check("24 cells", len(parse_cells("all")), 24)
    check("wildcard", parse_cells("qwen3-8b:*:k8v4"),
          [("qwen3-8b", t, "k8v4") for t in P.TIERS])
    check("canonical order kept", parse_cells("qwen38-27b:Q4_K_XL:f16,qwen38-27b:Q2_K_XL:f16"),
          [("qwen38-27b", "Q2_K_XL", "f16"), ("qwen38-27b", "Q4_K_XL", "f16")])
    av = server_argv("qwen38-27b", "Q2_K_XL", "k8v4")
    check("k8v4 argv", av[1:], ["-m", os.path.expanduser("~/models/qwen38/Qwen3.8-27B-UD-Q2_K_XL.gguf"),
                                "-c", "20480", "--fit", "on", "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0",
                                "-np", "1", "--jinja", "--no-webui", "--host", "localhost", "--port", "18130"])
    check("server exe is the p6fa build", av[0].endswith("engines/llama.cpp-p6fa/build/bin/llama-server"), True)
    check("8B gguf path", P.gguf_path("qwen3-8b", "IQ3_XXS"),
          os.path.expanduser("~/models/qwen3-8b/Qwen3-8B-UD-IQ3_XXS.gguf"))

    print("selftest: request bodies")
    b = request_body("qwen3-8b", "longctx", "x")
    check("longctx body", b, {"messages": [{"role": "user", "content": "x"}], "stream": False, "temperature": 0.0,
                              "cache_prompt": False, "max_tokens": 128,
                              "chat_template_kwargs": {"enable_thinking": False}})
    b = request_body("qwen38-27b", "gsm8k", "q")
    check("gsm8k hybrid body", (b["max_tokens"], b["chat_template_kwargs"], b["cache_prompt"], b["temperature"]),
          (8192, {"reasoning_effort": "medium"}, False, 0.0))
    check("gsm8k 8B body has no kwargs", "chat_template_kwargs" in request_body("qwen3-8b", "gsm8k", "q"), False)
    gs = P.load_gsm8k()
    check("gsm8k-100 verified, 100 items", len(gs), 100)
    check("gsm8k prompt is run_eval's", P.gsm8k_prompt(gs[0]),
          gs[0]["question"] + "\n\nEnd your response with: Answer: <number>")
    fake_item = {"id": "single-00", "content": "c" * 50, "family": "single", "sha256": "s", "haystack_tokens": 1,
                 "prompt_tokens_qwen38_nothink": 9, "answers": ["1234567"], "distractors": []}
    jobs = build_jobs("qwen38-27b", TASKS, [fake_item], gs[:2], DEFAULT_TIMEOUT)
    check("job order: long-context first, then GSM8K",
          [j["job_id"] for j in jobs], ["longctx|single-00", "gsm8k|" + gs[0]["id"], "gsm8k|" + gs[1]["id"]])
    check("long-context prompt stored by hash", jobs[0]["stored_request"]["messages"][0]["content"],
          {"item_id": "single-00", "sha256": P.sha256_text("c" * 50), "chars": 50})

    print("selftest: scoring")
    check("visible answer plain", P.visible_answer({"content": " 1234567 "}), "1234567")
    check("visible answer strips inline think", P.visible_answer({"content": "<think>9999999</think> 1234567"}), "1234567")
    check("visible answer unterminated think is empty", P.visible_answer({"content": "<think>1234567"}), "")
    it = {"answers": ["ABCDE", "FGHIJ"]}
    check("containment all required", P.score_longctx("The variables are abcde and FGHIJ.", it), True)
    check("containment misses one", P.score_longctx("ABCDE only", it), False)
    row = {"content": "The number is 7654321.", "reasoning_content": None}
    sc = P.score_row("longctx", row, {"answers": ["7654321"], "distractors": ["1111111"]})
    check("score_row longctx", (sc["correct"], sc["answers_found"], sc["distractors_found"]), (True, ["7654321"], []))
    g = {"answer_number": 18.0}
    check("gsm8k via Paper 1 scorer (reasoning ignored)",
          P.score_row("gsm8k", {"content": "So.\nAnswer: 18", "reasoning_content": "Answer: 3"}, g)["correct"], True)
    check("gsm8k wrong", P.score_row("gsm8k", {"content": "Answer: 17", "reasoning_content": None}, g)["correct"], False)
    check("gsm8k empty content (unclosed thinking)",
          P.score_row("gsm8k", {"content": "", "reasoning_content": "... 18"}, g)["correct"], False)

    print("selftest: server-log parser")
    full = "\n".join([
        "0.00.122.606 I cmn  common_param: common_params_print_info: build 1 (035e227) with GNU 13.3.0 for Linux x86_64",
        "0.00.579.983 I common_params_fit_impl: projected to use 8576 MiB of device memory vs. 10402 MiB of free device memory",
        "0.00.579.987 I common_params_fit_impl: will leave 1825 >= 1024 MiB of free device memory, no changes needed",
        "0.00.579.987 I common_fit_params: successfully fit params to free device memory",
        "0.05.213.501 I load_tensors: offloading output layer to GPU",
        "0.05.213.508 I load_tensors: offloaded 65/65 layers to GPU",
        "0.05.213.511 I load_tensors:   CPU_Mapped model buffer size =   397.85 MiB",
        "0.05.213.512 I load_tensors:        CUDA0 model buffer size =  7575.80 MiB",
        "0.06.086.930 I llama_context: n_ctx                 = 20480",
        "0.06.086.931 I llama_context: flash_attn            = enabled",
        "0.06.090.603 I llama_context:  CUDA_Host  output buffer size =     3.79 MiB",
        "0.06.090.866 I llama_kv_cache:      CUDA0 KV buffer size =  1280.00 MiB",
        "0.06.091.740 I llama_kv_cache: size = 1280.00 MiB ( 20480 cells,  16 layers,  1/1 seqs), "
        "K (f16):  640.00 MiB, V (f16):  640.00 MiB",
        "0.06.094.148 I llama_memory_recurrent:      CUDA0 RS buffer size =   149.62 MiB",
        "0.06.094.156 I llama_memory_recurrent: size =  149.62 MiB (     1 cells,  64 layers,  1 seqs  0 rs_seq), "
        "R (f32):    5.62 MiB, S (f32):  144.00 MiB",
        "0.06.116.923 I sched_reserve:      CUDA0 compute buffer size =   130.28 MiB",
        "0.06.116.926 I sched_reserve:  CUDA_Host compute buffer size =    28.28 MiB",
    ])
    L = parse_server_log(full)
    check("layers 65/65", (L["offloaded_layers"], L["total_layers"]), (65, 65))
    check("fully resident", L["fully_resident"], True)
    check("device buffers", L["device_mib"], {"model": 7575.8, "kv": 1280.0, "rs": 149.62, "compute": 130.28,
                                              "output": 0, "total_model_kv_rs_compute": 9135.7})
    check("host buffers", (L["host_mib"]["model"], L["host_mib"]["compute"], L["host_mib"]["output"]),
          (397.85, 28.28, 3.79))
    check("kv summary", (L["kv_cache"]["cells"], L["kv_cache"]["type_k"], L["kv_cache"]["v_mib"]), (20480, "f16", 640.0))
    check("recurrent state f32", (L["recurrent"]["type_r"], L["recurrent"]["type_s"]), ("f32", "f32"))
    check("fit status", (L["fit"]["status"], L["fit"]["changed"]), ("success", False))
    check("n_ctx / flash / build", (L["n_ctx"], L["flash_attn"], L["build"]["commit"]), (20480, "enabled", "035e227"))
    check("raw lines kept", len(L["raw"]) >= 12, True)
    partial = "\n".join([
        "I common_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce device memory by 2100 MiB",
        "I common_params_fit_impl:   - CUDA0: 40 layers ( 1 overflowing),   9500 MiB used,   1100 MiB free",
        "I load_tensors: offloaded 99/99 layers to GPU",                         # an earlier load: must be ignored
        "I load_tensors:        CUDA0 model buffer size =  99999.00 MiB",
        "I load_tensors: offloaded 33/66 layers to GPU",
        "I load_tensors:   CPU_Mapped model buffer size =  7898.27 MiB",
        "I load_tensors:        CUDA0 model buffer size =  8502.24 MiB",
        "I llama_kv_cache:        CPU KV buffer size =   136.00 MiB",
        "I llama_kv_cache:      CUDA0 KV buffer size =   136.00 MiB",
        "I llama_memory_recurrent:        CPU RS buffer size =   311.72 MiB",
        "I llama_memory_recurrent:      CUDA0 RS buffer size =   286.78 MiB",
    ])
    L = parse_server_log(partial)
    check("partial: last load block only", (L["offloaded_layers"], L["total_layers"], L["device_mib"]["model"]),
          (33, 66, 8502.24))
    check("partial: not resident; host KV/RS seen", (L["fully_resident"], L["host_mib"]["kv"], L["host_mib"]["rs"]),
          (False, 136.0, 311.72))
    check("partial: fit changed, overflowing parsed", (L["fit"]["changed"], L["fit"]["overflowing"]), (True, 1))
    L = parse_server_log("llama_kv_cache_unified: KV self size  =  576.00 MiB, K (q8_0):  288.00 MiB, V (q8_0):  288.00 MiB")
    check("legacy 'KV self size' line", (L["kv_cache"]["total_mib"], L["kv_cache"]["type_k"], L["kv_cache"]["legacy_line"]),
          (576.0, "q8_0", True))
    L = parse_server_log("I load_tensors: offloaded 65/65 layers to GPU\nI load_tensors: CUDA0 model buffer size = 10 MiB\n"
                         "I common_params_fit_impl:   - CUDA0: 65 layers ( 2 overflowing),   9500 MiB used,   1100 MiB free")
    check("split layers are not fully resident", L["fully_resident"], False)

    print("selftest: nvidia-smi parser")
    nv = parse_nvidia("4242, /x/llama-server, 10234\n77, Xorg, 312\n99, foo, [N/A]\n", "10700, 12282, 1400, 61\n", 4242)
    check("per-process MiB", (nv["process_mib"], nv["other_processes_mib"], nv["gpu_total_mib"], nv["temp_c"]),
          (10234, 312, 12282, 61))
    check("no processes", parse_nvidia("", "", 1)["process_mib"], None)

    print("selftest: receipts")
    with tempfile.TemporaryDirectory(dir=P.WORK) as td:
        dl = os.path.join(td, "downloads.jsonl")
        with open(dl, "w") as f:
            f.write(json.dumps({"repo": "r", "revision": "v", "path": "Qwen3-8B-UD-Q2_K_XL.gguf", "bytes": 5,
                                "sha256": "ab" * 32, "dest": "/nowhere/Qwen3-8B-UD-Q2_K_XL.gguf"}) + "\n")
        check("receipt from downloads.jsonl", (P.receipt_for("/m/Qwen3-8B-UD-Q2_K_XL.gguf", dl) or {}).get("sha256"),
              "ab" * 32)
        check("pinned 27B Q4 fallback", (P.receipt_for("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf", dl) or {}).get("sha256", "")[:8],
              "3f227079")
        check("unknown file has no receipt", P.receipt_for("/m/other.gguf", dl), None)

    print("selftest: runner (fake HTTP, fake server; nothing is started)")
    with tempfile.TemporaryDirectory(dir=P.WORK) as td:
        class FakeProc:
            def __init__(self):
                self.alive, self.pid = True, 4242

            def poll(self):
                return None if self.alive else 0

        class FakeServer:
            name, base, fresh, cell = "fake", "http://fake", False, ("qwen3-8b", "Q2_K_XL", "f16")

            def __init__(self):
                self.proc, self.starts, self.stops = None, 0, 0

            def start(self):
                self.proc, self.starts, self.fresh = FakeProc(), self.starts + 1, False

            def stop(self):
                if self.proc:
                    self.proc.alive = False
                self.stops += 1

            def identity(self):
                return {"pid": 4242, "start_seq": self.starts}

        calls = []
        script = {"longctx|a": ["ok"], "longctx|b": ["conn", "conn", "ok"], "longctx|c": ["400"]}

        def post(url, body, timeout=None):
            jid = body["_jid"]
            calls.append(jid)
            step = script[jid].pop(0) if script[jid] else "ok"
            if step == "conn":
                raise urllib.error.URLError("refused")
            if step == "400":
                raise urllib.error.HTTPError(url, 400, "bad", {}, None)
            return {"choices": [{"message": {"content": "7654321", "reasoning_content": None}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 3}, "timings": {"predicted_n": 3}}

        def mkjob(x):
            body = request_body("qwen3-8b", "longctx", "hay " + x)
            body["_jid"] = "longctx|" + x
            item = {"id": x, "answers": ["7654321"], "distractors": []}
            return {"job_id": "longctx|" + x, "task": "longctx", "item": item, "body": body,
                    "stored_request": redact(body, x), "timeout": 5, "meta": {"prompt_tokens_expected": 10}}

        jr = os.path.join(td, "j.jsonl")
        srv = FakeServer()
        r = P6Runner(srv, jr, post=post, temp_fn=lambda: 40, sleep=lambda s: None,
                     pause_file=os.path.join(td, "PAUSE"), poll_s=0)
        rows = [r.run(mkjob(x)) for x in "abc"]
        check("ok / retried-ok / 4xx-failed", [(x["ok"], x.get("attempt")) for x in rows],
              [(True, 0), (True, 2), (False, 0)])
        check("server restarted after connection errors", (srv.starts, srv.stops), (3, 2))
        check("prompt-token delta recorded", rows[0]["meta"]["prompt_tokens_delta"], 1)
        check("score stored", rows[0]["score"]["correct"], True)
        check("timings stored", rows[0]["timings"], {"predicted_n": 3})
        n_calls = len(calls)
        r2 = P6Runner(srv, jr, post=post, temp_fn=lambda: 40, sleep=lambda s: None,
                      pause_file=os.path.join(td, "PAUSE"), poll_s=0)
        [r2.run(mkjob(x)) for x in "abc"]
        check("resume: nothing re-sent", len(calls), n_calls)
        r3 = P6Runner(srv, jr, retry_failed=True, post=post, temp_fn=lambda: 40, sleep=lambda s: None,
                      pause_file=os.path.join(td, "PAUSE"), poll_s=0)
        row = r3.run(mkjob("c"))
        check("--retry-failed re-sends only the failed job", (len(calls), row["ok"]), (n_calls + 1, True))
        J = P.read_journal(jr)
        check("journal outcome: ok row wins over the earlier failure", J["outcomes"]["longctx|c"]["ok"], True)
        check("non-terminal error rows kept", sum(1 for ln in open(jr) if '"terminal": false' in ln), 3)
        # pause file: present on the first check, lifted (moved aside) while "sleeping"
        pf = os.path.join(td, "PAUSE")
        open(pf, "w").close()
        temps = iter([40, 40, 40])

        def sleeper(s):
            if os.path.exists(pf):
                os.rename(pf, pf + ".lifted")
        r4 = P6Runner(srv, jr, post=post, temp_fn=lambda: next(temps, 40), sleep=sleeper, pause_file=pf, poll_s=0)
        stops = srv.stops
        r4.gate()
        ev = [json.loads(ln) for ln in open(jr) if '"kind"' in ln]
        check("pause file stops the server and journals pause/resume",
              (srv.stops > stops, [e["kind"] for e in ev][-2:]), (True, ["pause", "resume"]))
        # thermal hysteresis: 90 -> pause, 80 -> still paused (>= resume 75), 70 -> resume
        temps2 = iter([90, 80, 70])
        slept = []
        r5 = P6Runner(srv, jr, post=post, temp_fn=lambda: next(temps2, 70), sleep=slept.append, pause_file=pf,
                      poll_s=0, temp_stop=87, temp_resume=75)
        r5.gate()
        check("thermal pause waits for the resume threshold", len(slept), 2)

    if fails:
        print(f"\nSELFTEST FAILED: {fails}")
        return 1
    print("\nSELFTEST PASSED (all checks)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="", help="'all', or comma list of model:tier:kv (fnmatch wildcards)")
    ap.add_argument("--tasks", default="longctx,gsm8k")
    ap.add_argument("--limit", type=int, default=0, help="first N items per task (smoke tests; use --journal-dir)")
    ap.add_argument("--journal-dir", default=P.JOURNALS)
    ap.add_argument("--retry-failed", action="store_true", help="re-send jobs whose terminal row failed")
    ap.add_argument("--no-hash", action="store_true", help="skip hashing the GGUF (identity records verified=false)")
    ap.add_argument("--timeout-longctx", type=float, default=DEFAULT_TIMEOUT["longctx"])
    ap.add_argument("--timeout-gsm8k", type=float, default=DEFAULT_TIMEOUT["gsm8k"])
    ap.add_argument("--temp-stop", type=int, default=p5run.TEMP_STOP_C)
    ap.add_argument("--temp-resume", type=int, default=p5run.TEMP_RESUME_C)
    ap.add_argument("--idle-seconds", type=float, default=10, help="idle after each new attempt; cached jobs do not wait")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; start nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not 0 <= a.idle_seconds <= 3600:
        ap.error("--idle-seconds must be between 0 and 3600")
    if a.selftest:
        sys.exit(selftest())
    if not a.cells:
        ap.error("--cells is required (or --selftest)")
    tasks = [t for t in a.tasks.split(",") if t]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        ap.error(f"unknown tasks {bad}")
    if a.limit and os.path.abspath(a.journal_dir) == os.path.abspath(P.JOURNALS):
        ap.error("--limit writes partial journals: point --journal-dir away from work/journals")
    if a.no_hash and os.path.abspath(a.journal_dir) == os.path.abspath(P.JOURNALS):
        ap.error("--no-hash is smoke-only: point --journal-dir away from work/journals")
    cells = parse_cells(a.cells)
    log(f"{len(cells)} cells: {[P.cell_name(*c) for c in cells]}; tasks {tasks}")
    results = []
    for c in cells:
        results.append(run_cell(c, tasks, a))
    log("ALL DONE " + json.dumps(results))


if __name__ == "__main__":
    main()
