"""Paper 5 serving and request runner.

- `TabbyServer` / `LlamaServer`: start one server as our own child process group,
  wait for health, stop it (only our own PIDs; never pkill by pattern).
- `Runner.run(job)`: one raw completion with the budget-parity gate
  (server prompt_tokens must equal the harness count and be <= the budget),
  the pause file, and the thermal pause. Append-only JSONL journals; a job
  with a terminal row is never re-sent.

Operational thresholds here are adjustable (lab rule after P2R1); they are not
part of any frozen protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAUSE_FILE = os.path.join(HERE, "PAUSE")
LOGS = os.path.join(C.WORK, "logs")
TEMP_STOP_C = 87
TEMP_RESUME_C = 75

HARNESS_FILES = ["p5common.py", "p5run.py", "summarize.py", "build_prompts.py", "run_grid.py",
                 "judge.py", "tabby-p5.yml"]


def harness_hashes() -> dict:
    out = {}
    for f in HARNESS_FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[f] = C.sha256_file(p)[:16]
    return out


def log(msg: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def gpu_temp() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return 0


def http_json(url: str, body: dict | None = None, timeout: float = 30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class _Server:
    name = "server"
    port = 0

    def __init__(self):
        self.proc = None
        self.log_path = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def cmd_env_cwd(self):
        raise NotImplementedError

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self, timeout_s: float = 600):
        if self.proc and self.proc.poll() is None:
            return
        if self.healthy():
            raise SystemExit(f"port {self.port} already serves something we did not start")
        cmd, env, cwd = self.cmd_env_cwd()
        os.makedirs(LOGS, exist_ok=True)
        self.log_path = os.path.join(LOGS, f"{self.name}-{time.strftime('%Y%m%d-%H%M%S')}.log")
        lf = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        log(f"{self.name}: started pid {self.proc.pid}, log {self.log_path}")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise SystemExit(f"{self.name} exited during startup (code {self.proc.returncode}); see {self.log_path}")
            if self.healthy():
                log(f"{self.name}: healthy after {time.time()-t0:.0f}s")
                return
            time.sleep(2)
        self.stop()
        raise SystemExit(f"{self.name} not healthy after {timeout_s}s")

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            self.proc = None
            return
        pgid = os.getpgid(self.proc.pid)
        for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        log(f"{self.name}: stopped")
        self.proc = None

    def identity(self) -> dict:
        return {"server": self.name, "pid": self.proc.pid if self.proc else None, "port": self.port}

    def count_tokens_server(self, text: str) -> int:
        """The server's own tokenizer count, for the budget-parity gate."""
        raise NotImplementedError


class TabbyServer(_Server):
    name = "tabby-p5"
    port = 18125

    def cmd_env_cwd(self):
        repo = C.REPO
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        env["PYTHONPATH"] = os.path.join(repo, "engines", "exl3-1.4.7-lab")
        env["EXL3_CKPT_PP"] = "1024"
        py = os.path.join(repo, "engines", "exl3-env", "bin", "python")
        cmd = ["nice", "-n", "5", py, "main.py", "--config", os.path.join(HERE, "tabby-p5.yml")]
        return cmd, env, os.path.join(repo, "engines", "tabbyAPI")

    def count_tokens_server(self, text: str) -> int:
        r = http_json(self.base + "/v1/token/encode",
                      {"text": text, "add_bos_token": False, "encode_special_tokens": True}, timeout=120)
        return int(r["length"])

    def identity(self) -> dict:
        d = super().identity()
        try:
            m = http_json(self.base + "/v1/model")
            d["model_id"] = m.get("id")
            params = m.get("parameters") or {}
            d["max_seq_len"] = params.get("max_seq_len")
            d["draft"] = (params.get("draft") or {}).get("id") if isinstance(params.get("draft"), dict) else None
        except Exception as e:
            d["model_error"] = repr(e)
        return d


class LlamaServer(_Server):
    """The pinned llama.cpp server (engines/llama.cpp/build), used unmodified."""

    def count_tokens_server(self, text: str) -> int:
        r = http_json(self.base + "/tokenize",
                      {"content": text, "add_special": False, "parse_special": True}, timeout=120)
        return len(r["tokens"])

    def __init__(self, name: str, port: int, model: str, extra: list[str]):
        super().__init__()
        self.name, self.port, self.model, self.extra = name, port, model, extra

    def cmd_env_cwd(self):
        exe = os.path.join(C.REPO, "engines", "llama.cpp", "build", "bin", "llama-server")
        cmd = ["nice", "-n", "5", exe, "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port),
               "--no-webui"] + self.extra
        env = dict(os.environ)
        return cmd, env, C.REPO


class Runner:
    def __init__(self, server: _Server, journal: str, *, idle_seconds: float | None = None):
        # Operator pacing only: never change prompts, sampling, or cached rows.
        # Unit-scoped environment keeps other studies and servers unaffected.
        if idle_seconds is None:
            idle_seconds = os.environ.get("P5_REQUEST_IDLE_SECONDS", "0")
        try:
            self.request_idle_seconds = float(idle_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("P5_REQUEST_IDLE_SECONDS must be finite and nonnegative") from exc
        if not math.isfinite(self.request_idle_seconds) or self.request_idle_seconds < 0:
            raise ValueError("P5_REQUEST_IDLE_SECONDS must be finite and nonnegative")
        self._next_request_at = None
        self.server = server
        self.journal = journal
        os.makedirs(os.path.dirname(journal), exist_ok=True)
        self.done = {}
        if os.path.exists(journal):
            with open(journal) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("terminal"):
                        self.done[r["job_id"]] = r
        self.hashes = harness_hashes()

    def _wait_for_request_slot(self) -> float:
        """Leave an idle gap after each live attempt; cache hits never wait.

        This is a lower average duty cycle, not a GPU-utilization/VRAM cap.
        Delays precede the HTTP timer and are recorded separately in the row.
        The ordinary pause/thermal gate still runs before the next request.
        """
        if self._next_request_at is None:
            return 0.0
        start = time.monotonic()
        remaining = self._next_request_at - start
        if remaining <= 0:
            return 0.0
        time.sleep(remaining)
        return max(0.0, time.monotonic() - start)

    def _append(self, row: dict):
        with open(self.journal, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def gate(self):
        """Pause file and thermal pause: stop our server, wait, restart."""
        paused = False
        while os.path.exists(PAUSE_FILE) or gpu_temp() >= TEMP_STOP_C:
            if not paused:
                why = "pause file" if os.path.exists(PAUSE_FILE) else f"GPU {gpu_temp()} C"
                log(f"pausing ({why}); stopping {self.server.name}")
                self.server.stop()
                paused = True
            time.sleep(30)
            if not os.path.exists(PAUSE_FILE) and gpu_temp() >= TEMP_RESUME_C:
                while gpu_temp() >= TEMP_RESUME_C:
                    time.sleep(30)
        if paused:
            log("resuming")
        if not (self.server.proc and self.server.proc.poll() is None):
            self.server.start()

    def run(self, job: dict, endpoint: str = "/v1/completions", retries: int = 2) -> dict:
        """job: job_id, prompt (final string), n_prompt (harness count), budget (or None),
        params (sampling fields), meta (free-form)."""
        if job["job_id"] in self.done:
            return self.done[job["job_id"]]
        if job.get("budget") is not None and job["n_prompt"] is not None and job["n_prompt"] > job["budget"]:
            raise SystemExit(f"budget gate: {job['job_id']} has {job['n_prompt']} > {job['budget']} tokens")
        # TabbyAPI returns an empty usage block for raw completions, so the pre-registered
        # parity gate asks the server's own tokenizer instead (/v1/token/encode).
        if job.get("verify_tokens"):
            self.gate()          # the server must be up before it can tokenize
        server_n = None
        if job.get("verify_tokens") and "prompt" in job:
            try:
                server_n = self.server.count_tokens_server(job["prompt"])
            except Exception as e:
                log(f"{job['job_id']}: token count failed ({e!r}); restarting server")
                self.server.stop()
                self.gate()
                server_n = self.server.count_tokens_server(job["prompt"])
            if server_n != job["n_prompt"]:
                raise SystemExit(f"parity gate: {job['job_id']} server tokenized {server_n}, "
                                 f"harness {job['n_prompt']}")
            if job.get("budget") is not None and server_n > job["budget"]:
                raise SystemExit(f"budget gate: {job['job_id']} server count {server_n} > {job['budget']}")
        if "messages" in job:
            body = {"messages": job["messages"], "stream": False}
        else:
            body = {"prompt": job["prompt"], "stream": False}
        body.update(job["params"])
        last_err = None
        for attempt in range(retries + 1):
            idle_wait = self._wait_for_request_slot()
            self.gate()
            t0 = time.time()
            try:
                resp = http_json(self.server.base + endpoint, body, timeout=job.get("timeout", 900))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = repr(e)
                self._append({"job_id": job["job_id"], "terminal": False, "attempt": attempt,
                              "error": last_err, "seconds": round(time.time() - t0, 2),
                              "request_idle_seconds": self.request_idle_seconds,
                              "idle_wait_seconds": round(idle_wait, 3),
                              "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                log(f"{job['job_id']}: error {last_err}; restarting server")
                self.server.stop()
                continue
            finally:
                if self.request_idle_seconds:
                    self._next_request_at = time.monotonic() + self.request_idle_seconds
            dt = time.time() - t0
            choice = resp["choices"][0]
            usage = resp.get("usage") or {}
            sp = usage.get("prompt_tokens")
            if job["n_prompt"] is not None and sp is not None and sp != job["n_prompt"]:
                raise SystemExit(f"parity gate: {job['job_id']} server counted {sp}, harness {job['n_prompt']}")
            text_out = choice.get("text") if "text" in choice else (choice.get("message") or {}).get("content")
            row = {"job_id": job["job_id"], "terminal": True, "ok": True, "attempt": attempt,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(dt, 3),
                   "prompt_sha256": C.sha256_text(job.get("prompt") or json.dumps(job.get("messages"))),
                   "n_prompt": job["n_prompt"],
                   "budget": job.get("budget"), "usage": usage, "finish_reason": choice.get("finish_reason"),
                   "request_idle_seconds": self.request_idle_seconds,
                   "idle_wait_seconds": round(idle_wait, 3),
                   "server_prompt_tokens": server_n,
                   "completion_tokens_harness": (C.count_tokens(text_out) if text_out else 0),
                   "text": text_out,
                   "reasoning": (choice.get("message") or {}).get("reasoning_content"),
                   "params": job["params"], "meta": job.get("meta", {}),
                   "server": self.server.identity(), "harness": self.hashes}
            self._append(row)
            self.done[job["job_id"]] = row
            return row
        row = {"job_id": job["job_id"], "terminal": True, "ok": False, "error": last_err,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_prompt": job["n_prompt"],
               "request_idle_seconds": self.request_idle_seconds,
               "budget": job.get("budget"), "meta": job.get("meta", {}), "harness": self.hashes}
        self._append(row)
        self.done[job["job_id"]] = row
        return row


def split_thinking(text: str) -> tuple[str, str, bool]:
    """(thinking, visible answer, closed). The prompt ends with '<think>\\n', so the
    completion starts inside the thinking block. An unterminated block leaves an
    empty answer (amendment item 4)."""
    if text is None:
        return "", "", False
    if "</think>" in text:
        th, ans = text.split("</think>", 1)
        return th, ans.strip(), True
    return text, "", False
