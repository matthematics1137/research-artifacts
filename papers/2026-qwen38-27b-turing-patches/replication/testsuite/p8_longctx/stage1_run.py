#!/usr/bin/env python3
"""P8 long context, stage 1: each KV cache format's measured reach on the desktop 2080 Ti.

Registration: research/prereg/2026-11-01-p8-longctx-stage1.md (commit c75ee2e) and its Amendment 1. Runs ON the
desktop as one queue step, inside that step's capped turing-run unit, so the guard counts the server as the lab's.

For the harness check and then each ladder step, in the registered order (8,8; 4,4; 8,4; fp16; each format stops at
its first failure):
  1. write the step's TabbyAPI config (stage1_config.py) and record the GPU's baseline memory and processes;
  2. start TabbyAPI as a child process with the command tools/turing/lanes.sh builds for TabbyAPI lanes, plus the
     w13 environment of queue step x-q-w13s2-mtp (mode 2 for MTP);
  3. sample nvidia-smi at 1 Hz from server start to exit (memory.used, temperature, power, SM clock) and the
     server's resident memory;
  4. send the step's filled prompt (L - 512 tokens) with thinking off, temperature 0, up to 64 decoded tokens;
  5. stop the server (SIGTERM, SIGKILL after 60 s), wait for memory to return to baseline, journal the step.

Failure classes (Amendment 1, item 6): a CUDA out-of-memory error in load, prefill or decode; a completed step whose
peak is above 10,752 MiB; a hang past 25 minutes from server start. Any other error is a harness error: the stage
stops with no verdict for that step (exit 3). Steps with a verdict are never rerun; a step a guard stop interrupted is
run again and listed as interrupted. The 1.5 GPU-hour budget is checked before each step (Amendment 1, item 7).
Stdlib only. Nothing is deleted.

  python stage1_run.py --prompts DIR --out DIR [--port 8097]
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_config as C  # noqa: E402
import stage1_spec as S  # noqa: E402

LAB = os.environ.get("TURING_LAB", "/opt/p8")
TABBY = f"{LAB}/src/tabbyAPI"
PY = f"{LAB}/venv/bin/python"
# tools/turing/lanes.sh TABBY_BOOT, unchanged: upstream main.py with loguru's file sink moved to TURING_TABBY_LOGDIR and
# without retention or compression (both make loguru discard files).
TABBY_BOOT = ('import os, sys, runpy; from loguru import logger; d = os.environ.get("TURING_TABBY_LOGDIR"); '
              'a = logger.add; f = lambda s: (d + s[4:]) if (d and isinstance(s, str) and s.startswith("logs/")) else s; '
              'logger.add = lambda *p, **k: a(*((f(p[0]),) + p[1:] if p else ()), **{x: (f(y) if x == "sink" else y) '
              'for x, y in k.items() if x not in ("retention", "compression")}); sys.argv = ["main.py"] + sys.argv[1:]; '
              'runpy.run_path("main.py", run_name="__main__")')
W13_ENV = {"PYTHONPATH": f"{LAB}/overlay/exl3-w13", "EXL3_MGEMM_N_THRESHOLD": "0", "EXL3_FD75D_LIVE": "0",
           "EXL3_INT8_SQ_MAX_M": "2", "EXL3_GEMV_SK": "2"}
# Left unset so exllamav3's default allocator settings apply, as in the memory anchors (Amendment 1, item 1).
ALLOC_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "EXL3_EXPANDABLE_SEGMENTS")
OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda_error_out_of_memory", "cudaerrormemoryallocation")
HOT_C, HOT_SAMPLES = 86.0, 3      # stop our server if the GPU reads >= 86 C three samples running (guard: 87 C)
SETTLE_MIB, SETTLE_S = 200, 90
STOP_GRACE_S = 60


class HarnessError(Exception):
    pass


def now():
    return time.time()


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def smi(query, what="gpu"):
    out = subprocess.run(["nvidia-smi", f"--query-{what}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def mem_used():
    return int(float(smi("memory.used")[0]))


def compute_apps():
    return smi("pid,process_name,used_memory", "compute-apps")


def read_rss_mib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def oom_in(text):
    low = (text or "").lower()
    return next((m for m in OOM_MARKERS if m in low), None)


def http(method, url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


class Sampler(threading.Thread):
    """nvidia-smi -l 1 (memory.used, temperature, power, SM clock) plus the server's VmRSS, one CSV row a second."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.pid = path, None
        self.max_mem = self.max_temp = self.max_power = None
        self.max_rss = None
        self.hot_run, self.hot = 0, False
        self.last_t, self.n = None, 0
        self.proc = None

    def run(self):
        q = "timestamp,memory.used,temperature.gpu,power.draw,clocks.sm"
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-l", "1"],
                                     stdout=subprocess.PIPE, text=True)
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_unix", "smi_time", "mem_used_mib", "temp_c", "power_w", "sm_mhz", "server_rss_mib"])
            for line in self.proc.stdout:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 5:
                    continue
                try:
                    mem, temp, power = int(float(parts[1])), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                rss = read_rss_mib(self.pid) if self.pid else None
                t = now()
                w.writerow([f"{t:.3f}", parts[0], mem, temp, power, parts[4], rss if rss is not None else ""])
                f.flush()
                self.max_mem = mem if self.max_mem is None else max(self.max_mem, mem)
                self.max_temp = temp if self.max_temp is None else max(self.max_temp, temp)
                self.max_power = power if self.max_power is None else max(self.max_power, power)
                if rss is not None:
                    self.max_rss = rss if self.max_rss is None else max(self.max_rss, rss)
                self.hot_run = self.hot_run + 1 if temp >= HOT_C else 0
                self.hot = self.hot or self.hot_run >= HOT_SAMPLES
                self.last_t, self.n = t, self.n + 1

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.join(timeout=10)


class Journal:
    def __init__(self, path):
        self.path = path
        self.rows = []
        if os.path.exists(path):
            with open(path) as f:
                self.rows = [json.loads(line) for line in f if line.strip()]

    def add(self, row):
        row = {"time": stamp(), **row}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("server_card", "text", "gpu_apps")}), flush=True)
        return row

    def results(self):
        return [r for r in self.rows if r.get("kind") == "result"]

    def verdict(self, step):
        return next((r for r in self.results() if r["step"] == step), None)


def server_env(log_dir):
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH" and k not in ALLOC_VARS}
    env.update({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "XDG_CACHE_HOME": f"{LAB}/cache",
                "TRITON_HOME": f"{LAB}/cache/triton", "TRITON_CACHE_DIR": f"{LAB}/cache/triton/cache",
                "HF_HOME": f"{LAB}/cache/huggingface", "PIP_CACHE_DIR": f"{LAB}/cache/pip",
                "TURING_TABBY_LOGDIR": log_dir, **W13_ENV})
    return env


def stop_server(proc):
    if proc.poll() is not None:
        return "exited"
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "exited"
    try:
        proc.wait(timeout=STOP_GRACE_S)
        return "terminated"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)
        return "killed"


def settle(baseline):
    t0 = now()
    while now() - t0 < SETTLE_S:
        m = mem_used()
        if m <= baseline + SETTLE_MIB:
            return m, round(now() - t0, 1)
        time.sleep(2)
    return mem_used(), round(now() - t0, 1)


def log_tail_from(path, offset):
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def run_step(args, journal, fmt, length, prompt, judged):
    step = S.step_id(fmt, length) if judged else "check-" + S.step_id(fmt, length)
    tag = time.strftime("%Y%m%dT%H%M%S")
    cfg_path = os.path.join(args.out, "configs", f"{step}-{tag}.yml")
    text = C.render(fmt, length, args.port, f"{LAB}/models")
    missing = C.check(text, fmt, length, args.port)
    if missing:
        raise HarnessError(f"{step}: config lines missing: {missing}")
    with open(cfg_path, "w") as f:
        f.write(text)
    log_path = os.path.join(args.out, "logs", f"{step}.log")
    with open(log_path, "a") as f:
        f.write(f"== {stamp()} stage1_run.py {step} config {cfg_path}\n")
    offset = os.path.getsize(log_path)
    baseline, apps = mem_used(), compute_apps()
    csv_path = os.path.join(args.out, "samples", f"{step}-{tag}.csv")
    decode = S.DECODE_TOKENS if judged else S.CHECK_DECODE_TOKENS
    t0 = now()
    journal.add({"kind": "start", "step": step, "format": fmt, "length": length, "judged": judged, "t_start": t0,
                 "config": cfg_path, "samples": csv_path, "baseline_mib": baseline, "gpu_apps": apps,
                 "prompt_tokens_expected": prompt["prompt_tokens_qwen38_nothink"], "prompt_sha256": prompt["sha256"]})
    sampler = Sampler(csv_path)
    sampler.start()
    logf = open(log_path, "a")
    proc = subprocess.Popen([PY, "-c", TABBY_BOOT, "--config", cfg_path], cwd=TABBY, env=server_env(args.tabby_logs),
                            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    sampler.pid = proc.pid
    deadline = t0 + S.HANG_S
    url = f"http://localhost:{args.port}"
    res = {"kind": "result", "step": step, "format": fmt, "length": length, "judged": judged,
           "baseline_mib": baseline, "prompt_tokens_expected": prompt["prompt_tokens_qwen38_nothink"]}
    outcome, reason, card, resp, load_s, req_wall = None, None, None, None, None, None
    try:
        # 1. load: until /health answers 200, the server exits, the hang deadline or a thermal stop
        while outcome is None:
            if proc.poll() is not None:
                tail = log_tail_from(log_path, offset)
                m = oom_in(tail)
                outcome, reason = ("oom", f"server exited during load ({m})") if m else \
                    ("harness", f"server exited during load, rc {proc.returncode}, no out-of-memory text")
                break
            try:
                code, _ = http("GET", url + "/health", timeout=3)
            except (urllib.error.URLError, OSError):
                code = None
            except Exception:  # an HTTP error status (503 while loading)
                code = None
            if code == 200:
                load_s = round(now() - t0, 1)
                break
            if now() > deadline:
                outcome, reason = "hang", "no healthy server within 25 minutes"
            elif sampler.hot:
                outcome, reason = "thermal", f"GPU at or above {HOT_C} C for {HOT_SAMPLES} samples during load"
            else:
                time.sleep(2)
        # 2. the loaded model must be what the config says
        if outcome is None:
            _, body = http("GET", url + "/v1/model", timeout=10)
            card = json.loads(body)
            params = card.get("parameters") or {}
            want_mode = S.CACHE_MODE[fmt]
            got_size, got_mode = params.get("cache_size"), str(params.get("cache_mode", ""))
            if got_size is not None and int(got_size) != length:
                outcome, reason = "harness", f"server cache_size {got_size}, config {length}"
            elif got_mode and got_mode.replace(" ", "").upper() != want_mode.upper():
                outcome, reason = "harness", f"server cache_mode {got_mode!r}, config {want_mode!r}"
        # 3. the filled prompt, in a worker thread so hang, thermal and server-exit checks keep running
        if outcome is None:
            body = {"model": "qwen38-exl3-2.0", "messages": [{"role": "user", "content": prompt["content"]}],
                    "max_tokens": decode, "temperature": 0, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            box = {}

            def call():
                try:
                    box["resp"] = http("POST", url + "/v1/chat/completions", body, timeout=S.HANG_S)
                except urllib.error.HTTPError as e:
                    box["http_error"] = (e.code, e.read().decode("utf-8", "replace"))
                except Exception as e:  # connection reset when the server dies, socket timeout
                    box["error"] = repr(e)

            t_req = now()
            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            while worker.is_alive():
                if now() > deadline:
                    outcome, reason = "hang", "request not answered within 25 minutes of server start"
                    break
                if sampler.hot:
                    outcome, reason = "thermal", f"GPU at or above {HOT_C} C for {HOT_SAMPLES} samples"
                    break
                if proc.poll() is not None:
                    worker.join(timeout=5)
                    break
                worker.join(timeout=1)
            req_wall = round(now() - t_req, 2)
            if outcome is None:
                tail = log_tail_from(log_path, offset)
                if "resp" in box:
                    resp = json.loads(box["resp"][1])
                    outcome = "completed"
                else:
                    detail = box.get("http_error") or box.get("error") or f"server exited, rc {proc.returncode}"
                    m = oom_in(json.dumps(detail)) or oom_in(tail)
                    outcome, reason = ("oom", f"out of memory during the request ({m})") if m else \
                        ("harness", f"request failed without an out-of-memory cause: {str(detail)[:400]}")
    except Exception as e:  # a runner bug or an unexpected server answer: never a verdict on the format
        if outcome is None or outcome == "completed":
            outcome, reason = "harness", f"runner error: {e!r}"[:400]
    finally:
        how = stop_server(proc)
        logf.close()
        after, settle_s = settle(baseline)
        sampler.stop()
    wall = round(now() - t0, 1)
    res.update({"outcome": outcome, "reason": reason, "load_s": load_s, "request_wall_s": req_wall,
                "wall_s": wall, "t_start": t0, "stop": how, "mem_after_mib": after, "settle_s": settle_s,
                "peak_mib": sampler.max_mem, "peak_rss_mib": sampler.max_rss, "max_temp_c": sampler.max_temp,
                "max_power_w": sampler.max_power, "samples": sampler.n, "samples_csv": csv_path, "config": cfg_path,
                "server_card": card})
    if resp is not None:
        u = resp.get("usage") or {}
        ch = (resp.get("choices") or [{}])[0]
        res.update({"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "prompt_time_s": u.get("prompt_time"), "prompt_tok_s": u.get("prompt_tokens_per_sec"),
                    "completion_time_s": u.get("completion_time"), "completion_tok_s": u.get("completion_tokens_per_sec"),
                    "finish_reason": ch.get("finish_reason"),
                    "text": ((ch.get("message") or {}).get("content") or "")[:300]})
    if outcome == "harness":
        res["verdict"] = "harness_error"
    elif not judged:
        res["verdict"] = "check_ok" if outcome == "completed" else "check_failed"
    elif outcome == "completed":
        peak = sampler.max_mem
        res["verdict"] = "pass" if peak is not None and peak <= S.PASS_LIMIT_MIB else "fail"
        if res["verdict"] == "fail":
            res["reason"] = f"completed with peak {peak} MiB above {S.PASS_LIMIT_MIB}"
    elif outcome in ("oom", "hang"):
        res["verdict"] = "fail"
    else:  # thermal: a safety stop, not a verdict on the format
        res["verdict"] = "thermal_stop"
    return journal.add(res)


def elapsed_s(journal):
    """GPU time used so far: each finished step's wall time, plus each interrupted step's sampled span."""
    total = sum(r.get("wall_s") or 0 for r in journal.results())
    total += sum(r.get("span_s") or 0 for r in journal.rows if r.get("kind") == "interrupted")
    return total


def mark_interrupted(journal):
    """A start row with no result after it was cut off (a guard stop): journal it once with its sampled span."""
    done = {r["t_start"] for r in journal.rows if r.get("kind") in ("result", "interrupted") and "t_start" in r}
    for r in [r for r in journal.rows if r.get("kind") == "start" and r["t_start"] not in done]:
        span = 0.0
        try:
            with open(r["samples"]) as f:
                last = list(csv.reader(f))[-1]
            span = max(0.0, float(last[0]) - r["t_start"])
        except (OSError, IndexError, ValueError):
            pass
        journal.add({"kind": "interrupted", "step": r["step"], "t_start": r["t_start"], "span_s": round(span, 1)})


def estimate_s(journal, length):
    """Amendment 1, item 7: last load time + the last measured prefill scaled by (L / L_last)^1.5 + 2 minutes."""
    done = [r for r in journal.results() if r.get("outcome") == "completed" and r.get("prompt_time_s")]
    if done:
        r = done[-1]
        load = r.get("load_s") or 60
        prefill = r["prompt_time_s"] * (length / max(r["prompt_tokens"] or 1, 1)) ** 1.5
    else:  # the registration's disclosed smoke figure: 624 tokens/s at a 30K prompt
        load, prefill = 60, (30000 / 624) * (length / 30000) ** 1.5
    return load + prefill + 120


def load_prompt(prompts_dir, role, length):
    with open(os.path.join(prompts_dir, f"{role}-{length}.json"), encoding="utf-8") as f:
        p = json.load(f)
    if p["length"] != length or p["role"] != role:
        raise HarnessError(f"prompt file {role}-{length}.json holds {p['role']} {p['length']}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8097)
    args = ap.parse_args()
    for d in ("configs", "logs", "samples"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    args.tabby_logs = f"{LAB}/logs/tabby-p8lc-stage1"
    os.makedirs(args.tabby_logs, exist_ok=True)
    journal = Journal(os.path.join(args.out, "journal.jsonl"))
    if any(r.get("verdict") == "harness_error" for r in journal.results()):
        print("stage1_run: the journal holds a harness error; fix it and record an amendment first", flush=True)
        return 3
    if os.path.exists(os.path.join(TABBY, "config.yml")):
        print(f"stage1_run: {TABBY}/config.yml exists and would be merged into every step's config", flush=True)
        return 3
    limit = smi("power.limit")[0]
    if abs(float(limit) - 260.0) > 0.5:
        print(f"stage1_run: power limit reads {limit} W, the registration runs at 260 W; stopping", flush=True)
        return 4
    present = [v for v in ALLOC_VARS if v in os.environ]
    mark_interrupted(journal)
    journal.add({"kind": "session", "pid": os.getpid(), "power_limit_w": limit, "alloc_vars_dropped": present,
                 "env": {k: v for k, v in W13_ENV.items()}, "elapsed_before_s": round(elapsed_s(journal), 1)})
    try:
        chk = journal.verdict("check-" + S.step_id(*S.CHECK_STEP))
        if chk is None:
            chk = run_step(args, journal, *S.CHECK_STEP, load_prompt(args.prompts, "check", S.CHECK_STEP[1]), False)
        if chk["verdict"] == "harness_error":
            return 3
        if chk["verdict"] != "check_ok":
            journal.add({"kind": "stage_stop", "why": f"harness check {chk['verdict']}: {chk.get('reason')}"})
            return 3
        for fmt, ladder in S.LADDER:
            for length in ladder:
                step = S.step_id(fmt, length)
                prior = journal.verdict(step)
                if prior is not None:
                    if prior["verdict"] == "fail":
                        break
                    if prior["verdict"] == "pass":
                        continue
                    if prior["verdict"] == "thermal_stop":
                        return 0
                est = estimate_s(journal, length)
                used = elapsed_s(journal)
                if used + est > S.BUDGET_S:
                    journal.add({"kind": "budget_stop", "next_step": step, "used_s": round(used, 1),
                                 "estimate_s": round(est, 1), "budget_s": S.BUDGET_S})
                    return 0
                r = run_step(args, journal, fmt, length, load_prompt(args.prompts, "ladder", length), True)
                if r["verdict"] == "harness_error":
                    return 3
                if r["verdict"] == "thermal_stop":
                    journal.add({"kind": "stage_stop", "why": r.get("reason")})
                    return 0
                if r["verdict"] == "fail":
                    break
        journal.add({"kind": "stage_done", "used_s": round(elapsed_s(journal), 1)})
        return 0
    except HarnessError as e:
        journal.add({"kind": "result", "step": "harness", "verdict": "harness_error", "reason": str(e),
                     "wall_s": 0})
        return 3


if __name__ == "__main__":
    sys.exit(main())
