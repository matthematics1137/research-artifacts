#!/usr/bin/env python3
"""Regression: a stale healthy server cannot impersonate a dead launch."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import importlib.util


HERE = Path(__file__).resolve().parent


def load_guard():
    spec = importlib.util.spec_from_file_location("p4_stale_guard", HERE / "server_guard.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_for_file(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def stop_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    guard = load_guard()
    with tempfile.TemporaryDirectory(prefix="paper4-server-guard-") as raw:
        root = Path(raw)
        port = free_port()
        pid_file = root / "stale.pid"
        stale = root / "stale_server.py"
        stale.write_text(
            """#!/usr/bin/env python3
import json, os, signal, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, alias, pid_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health': payload = {'status': 'ok'}
        elif self.path == '/v1/models': payload = {'object': 'list', 'data': [{'id': alias}]}
        else:
            self.send_response(404); self.end_headers(); return
        raw = json.dumps(payload).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def log_message(self, *_): pass
server = HTTPServer(('127.0.0.1', port), Handler)
open(pid_path, 'w').write(str(os.getpid()))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
server.serve_forever()
""",
            encoding="utf-8",
        )
        launcher = root / "fake-llama-server"
        launcher.write_text(
            """#!/usr/bin/env python3
import os, subprocess, sys, time
def value(flag): return sys.argv[sys.argv.index(flag) + 1]
child = subprocess.Popen(
    [sys.executable, os.environ['MOCK_STALE_SERVER'], value('--port'), value('--alias'), os.environ['MOCK_PID_FILE']],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    start_new_session=True,
)
time.sleep(0.30)
raise SystemExit(1)
""",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        alias = "P4R1-QWEN3MOE-IQ1M"
        artifacts = []
        model = None
        for item_alias, (filename, digest, _ngl, _context) in guard.P4R1_MATRIX.items():
            model_path = root / filename
            with model_path.open("wb") as handle:
                handle.truncate(guard.P4R1_BYTES[item_alias])
            stat = model_path.stat()
            artifacts.append(
                {
                    "alias": item_alias,
                    "family": "synthetic-stale-server-regression",
                    "filename": filename,
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
            if item_alias == alias:
                model = model_path
        assert model is not None
        manifest = root / "model-manifest.json"
        manifest_payload = (
            json.dumps(
                {
                    "schema": "paper4-validation-model-manifest-v1",
                    "content_verification": "full SHA-256 completed before the campaign; live cells use stat-only checks",
                    "artifact_count": 6,
                    "total_bytes": sum(guard.P4R1_BYTES.values()),
                    "artifacts": artifacts,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        manifest.write_bytes(manifest_payload)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        host_receipt = root / "host-receipt.json"
        host_payload = b'{"schema":"synthetic-stale-server-host"}\n'
        host_receipt.write_bytes(host_payload)
        host_receipt_sha256 = hashlib.sha256(host_payload).hexdigest()
        # The live verifier must succeed without opening weight contents. The
        # fake launcher also never opens the model; only the stat identity is used.
        model.chmod(0)
        dataset = root / "gsm8k.jsonl"
        with dataset.open("w", encoding="utf-8") as handle:
            for index in range(100):
                handle.write(json.dumps({"id": str(index), "question": "1+1?", "answer_number": 2}) + "\n")
        output = root / "results" / "out.jsonl"
        environment = {
            **os.environ,
            "LLAMA_SERVER": str(launcher),
            "GSM8K_DATASET": str(dataset),
            "PORT": str(port),
            "MOCK_STALE_SERVER": str(stale),
            "MOCK_PID_FILE": str(pid_file),
            "MODEL_IDENTITY_MANIFEST": str(manifest),
            "MODEL_IDENTITY_MANIFEST_SHA256": manifest_sha256,
            "P4R1_HOST_RECEIPT": str(host_receipt),
            "P4R1_HOST_RECEIPT_SHA256": host_receipt_sha256,
            "SERVER_START_ATTEMPTS": "20",
            "SERVER_STOP_ATTEMPTS": "2",
            "SERVER_POLL_SECONDS": "0.1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [
                "bash", str(HERE / "run_one.sh"), alias,
                str(model), "auto", "4096", str(output),
            ],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        wait_for_file(pid_file)
        stale_pid = int(pid_file.read_text(encoding="utf-8"))
        try:
            if result.returncode == 0:
                raise AssertionError("runner accepted a stale healthy server")
            if output.exists():
                raise AssertionError("runner began evaluation against a stale server")
            model_receipt = output.parent / f"{alias}-server" / "model-identity.json"
            if not model_receipt.is_file():
                raise AssertionError("stat-only model verification did not finish before fake launch")
            if model_receipt.stat().st_mode & 0o777 != 0o600:
                raise AssertionError("private model-identity receipt is not mode 0600")
            if model_receipt.parent.stat().st_mode & 0o777 != 0o700:
                raise AssertionError("private server log directory is not mode 0700")
            combined = result.stdout + result.stderr
            if "server identity was not verified" not in combined:
                raise AssertionError(f"runner did not report identity failure:\n{combined}")
            os.kill(stale_pid, 0)
        finally:
            stop_pid(stale_pid)
    print("STALE SERVER REGRESSION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
