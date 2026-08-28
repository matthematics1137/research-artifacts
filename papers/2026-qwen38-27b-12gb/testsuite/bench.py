#!/usr/bin/env python3
"""Phase 1 harness: load-fit search, raw speed, and server-based real-world speed
for Qwen3.8-27B quants on the 12GB 4080 Laptop. Stdlib only.

Safety: refuses to launch anything that doesn't fit the VRAM/RAM guards.
Every result is appended to results/phase1.jsonl immediately.

Usage:
  ./bench.py fit        -m <gguf> [--ctx 8192] [--mtp gpu|cpu|off]
  ./bench.py bench      -m <gguf> --ngl N [--ctx 8192]
  ./bench.py servertest -m <gguf> --ngl N [--ctx 8192] [--mtp gpu|cpu|off]
  ./bench.py auto       -m <gguf> [...]   # fit -> bench -> servertest(mtp off,on)
  ./bench.py report                        # render results table
"""
import argparse, json, os, re, signal, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
BIN = os.environ.get("LLAMA_BIN", f"{HOME}/local_llms/engines/llama.cpp/build/bin")
MTP_DRAFT = os.environ.get(
    "MTP_DRAFT",
    os.path.join(HOME, "models", "qwen38", "MTP", "mtp-Qwen3.8-27B-Q4_0.gguf"),
)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phase1.jsonl")
PORT = 8090
N_LAYERS = 65               # 64 transformer blocks + output layer, llama.cpp convention
VRAM_SAFETY_MIB = 500
RAM_SAFETY_GIB = 8
OVERHEAD_MIB = 1500         # cuda ctx + compute buffers + small KV at 8K q8

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def vram_free_mib():
    r = sh(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    return int(r.stdout.strip().splitlines()[0])

def vram_used_mib():
    r = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(r.stdout.strip().splitlines()[0])

def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 2**20
    return 0

class VramPeak(threading.Thread):
    """Polls nvidia-smi while a run is in flight; .peak holds max MiB seen."""
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.stop = threading.Event()
    def run(self):
        while not self.stop.is_set():
            try: self.peak = max(self.peak, vram_used_mib())
            except Exception: pass
            self.stop.wait(0.5)

def guard(gguf, ngl, mtp):
    """Return (ok, reason). Conservative pre-launch fit check."""
    size_mib = os.path.getsize(gguf) / 2**20
    per_layer = size_mib / N_LAYERS
    # Measured 2026-08-25: MTP draft allocates ~1.2GiB GPU-side even with -ngld 0
    # (ctx/compute buffers), ~2.8GiB with -ngld 99 (weights + buffers).
    gpu_need = min(ngl, N_LAYERS) * per_layer + OVERHEAD_MIB + (2800 if mtp == "gpu" else 1200 if mtp == "cpu" else 0)
    cpu_need_gib = (size_mib - min(ngl, N_LAYERS) * per_layer) / 1024 + (1.3 if mtp == "cpu" else 0)
    free = vram_free_mib()
    if gpu_need > free - VRAM_SAFETY_MIB:
        return False, f"VRAM guard: need ~{gpu_need:.0f}MiB, free {free}MiB (safety {VRAM_SAFETY_MIB})"
    if cpu_need_gib + RAM_SAFETY_GIB > mem_available_gib():
        return False, f"RAM guard: need ~{cpu_need_gib:.1f}GiB + {RAM_SAFETY_GIB} safety, avail {mem_available_gib():.1f}GiB"
    return True, f"ok (est gpu {gpu_need:.0f}MiB, cpu {cpu_need_gib:.1f}GiB, vram free {free}MiB)"

def record(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("RECORDED:", json.dumps(row, indent=None)[:400])

def env():
    e = os.environ.copy()
    e["LD_LIBRARY_PATH"] = f"{BIN}:{BIN}/../lib:" + e.get("LD_LIBRARY_PATH", "")
    return e

def kv_flags(kv):
    k, v = kv.split("/") if "/" in kv else (kv, kv)
    return ["-ctk", k, "-ctv", v]

# ---------------------------------------------------------------- fit search
def try_load(gguf, ngl, ctx, kv, mtp, timeout=420):
    ok, why = guard(gguf, ngl, mtp)
    if not ok:
        print(f"  ngl={ngl}: SKIP ({why})"); return None
    cmd = [f"{BIN}/llama-cli", "-m", gguf, "-ngl", str(ngl), "-c", str(ctx),
           *kv_flags(kv), "--flash-attn", "on", "-n", "8", "-p", "The capital of France is",
           "-st", "--no-display-prompt", "--simple-io"]
    if mtp != "off":
        cmd += ["-md", MTP_DRAFT, "--spec-type", "draft-mtp",
                "-ngld", "99" if mtp == "gpu" else "0"]
    mon = VramPeak(); mon.start(); t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env())
    except subprocess.TimeoutExpired:
        mon.stop.set(); print(f"  ngl={ngl}: TIMEOUT"); return None
    mon.stop.set()
    if r.returncode != 0:
        tail = (r.stderr or "")[-300:].replace("\n", " | ")
        print(f"  ngl={ngl}: FAIL rc={r.returncode} :: {tail}")
        return None
    print(f"  ngl={ngl}: OK load+gen in {time.time()-t0:.0f}s, vram peak {mon.peak}MiB")
    return {"vram_peak_mib": mon.peak, "load_gen_s": round(time.time() - t0, 1)}

def fit(a):
    size_mib = os.path.getsize(a.model) / 2**20
    free = vram_free_mib()
    budget = free - VRAM_SAFETY_MIB - OVERHEAD_MIB - (1310 if a.mtp == "gpu" else 0)
    guess = min(99, int(budget / (size_mib / N_LAYERS)))
    if guess >= N_LAYERS: guess = 99
    print(f"[fit] {os.path.basename(a.model)} ({size_mib/1024:.1f}GiB) ctx={a.ctx} kv={a.kv} mtp={a.mtp} -> first guess ngl={guess}")
    best, meta = None, None
    n = guess
    while n >= 0:
        m = try_load(a.model, n, a.ctx, a.kv, a.mtp)
        if m: best, meta = n, m; break
        n -= 2 if n > 4 else 1
        if guess - n > 12:
            print("[fit] giving up: 6+ failed attempts"); break
    if best is not None and best not in (99,):
        while best < N_LAYERS:          # try to climb back up
            m = try_load(a.model, best + 1, a.ctx, a.kv, a.mtp)
            if not m: break
            best, meta = best + 1, m
    row = {"stage": "fit", "model": os.path.basename(a.model), "ctx": a.ctx, "kv": a.kv,
           "mtp": a.mtp, "ngl_fit": best, **(meta or {}), "ok": best is not None}
    record(row)
    return best

# ---------------------------------------------------------------- llama-bench
def bench(a):
    ok, why = guard(a.model, a.ngl, "off")
    if not ok: print("SKIP:", why); return
    cmd = [f"{BIN}/llama-bench", "-m", a.model, "-ngl", str(a.ngl), "-fa", "1",
           "-ctk", a.kv.split("/")[0], "-ctv", a.kv.split("/")[-1],
           "-p", "512", "-n", "128", "-r", str(a.reps), "-o", "json"]
    mon = VramPeak(); mon.start()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env())
    mon.stop.set()
    if r.returncode != 0:
        record({"stage": "bench", "model": os.path.basename(a.model), "ngl": a.ngl,
                "ok": False, "err": (r.stderr or "")[-300:]})
        return
    out = json.loads(r.stdout[r.stdout.find("["):])
    for t in out:
        record({"stage": "bench", "model": os.path.basename(a.model), "ngl": a.ngl,
                "kv": a.kv, "test": t.get("test_kind") or ("pp" if t.get("n_prompt") else "tg"),
                "n_prompt": t.get("n_prompt"), "n_gen": t.get("n_gen"),
                "tok_s": round(t.get("avg_ts", 0), 2), "tok_s_stddev": round(t.get("stddev_ts", 0), 2),
                "vram_peak_mib": mon.peak, "ok": True})

# ---------------------------------------------------------------- server test
PROMPTS = {
    "prose": "Write a detailed 400-word explanation of why the sky is blue, aimed at a curious 12-year-old.",
    "code": "Write a complete Python module implementing an LRU cache with TTL support, unit tests included. Code only.",
}

def post(url, payload, timeout=600):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)

def servertest(a):
    ok, why = guard(a.model, a.ngl, a.mtp)
    if not ok: print("SKIP:", why); return
    cmd = [f"{BIN}/llama-server", "-m", a.model, "-ngl", str(a.ngl), "-c", str(a.ctx),
           *kv_flags(a.kv), "--flash-attn", "on", "--port", str(PORT), "--jinja",
           "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0"]
    if a.mtp != "off":
        cmd += ["-md", MTP_DRAFT, "--spec-type", "draft-mtp", "-ngld", "99" if a.mtp == "gpu" else "0"]
    mon = VramPeak(); mon.start()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env())
    try:
        for _ in range(600):                     # wait for /health, up to 10 min (cold page cache)
            if proc.poll() is not None:
                raise RuntimeError("server died: " + proc.stderr.read()[-400:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception: time.sleep(1)
        else: raise RuntimeError("server never became healthy")
        for name, text in PROMPTS.items():
            t0 = time.time()
            resp = post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                        {"messages": [{"role": "user", "content": text}],
                         "max_tokens": a.maxtok, "timings_per_token": False})
            wall = time.time() - t0
            tim = resp.get("timings", {})
            record({"stage": "servertest", "model": os.path.basename(a.model), "ngl": a.ngl,
                    "ctx": a.ctx, "kv": a.kv, "mtp": a.mtp, "prompt": name,
                    "gen_tok": resp.get("usage", {}).get("completion_tokens"),
                    "tg_tok_s": round(tim.get("predicted_per_second", 0), 2),
                    "pp_tok_s": round(tim.get("prompt_per_second", 0), 2),
                    "draft_n": tim.get("draft_n"), "draft_accepted": tim.get("draft_n_accepted"),
                    "wall_s": round(wall, 1), "vram_peak_mib": mon.peak, "ok": True})
    except Exception as e:
        record({"stage": "servertest", "model": os.path.basename(a.model), "ngl": a.ngl,
                "mtp": a.mtp, "ok": False, "err": str(e)[:400]})
    finally:
        mon.stop.set()
        proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired: proc.kill()
        time.sleep(2)

# ---------------------------------------------------------------- auto + report
def auto(a):
    ngl = fit(a)
    if ngl is None: print("auto: nothing fits, stopping"); return
    a.ngl = ngl
    bench(a)
    for mtp in ["off", a.mtp if a.mtp != "off" else "gpu"]:
        a2 = argparse.Namespace(**vars(a)); a2.mtp = mtp
        servertest(a2)

def report(_):
    rows = [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []
    for stage in ["fit", "bench", "servertest"]:
        sel = [r for r in rows if r.get("stage") == stage]
        if not sel: continue
        print(f"\n== {stage} ==")
        keys = [k for k in ["model", "ngl", "ngl_fit", "ctx", "kv", "mtp", "test", "prompt",
                            "tok_s", "tg_tok_s", "pp_tok_s", "draft_n", "draft_accepted",
                            "vram_peak_mib", "ok"] if any(k in r for r in sel)]
        print(" | ".join(keys))
        for r in sel:
            print(" | ".join(str(r.get(k, "")) for k in keys))

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["fit", "bench", "servertest", "auto"]:
        s = sub.add_parser(name)
        s.add_argument("-m", "--model", required=True)
        s.add_argument("--ctx", type=int, default=8192)
        s.add_argument("--kv", default="q8_0")           # or "q8_0/q4_0" for K/V split
        s.add_argument("--mtp", default="off", choices=["off", "gpu", "cpu"])
        s.add_argument("--ngl", type=int, default=99)
        s.add_argument("--reps", type=int, default=3)
        s.add_argument("--maxtok", type=int, default=512)
    sub.add_parser("report")
    a = p.parse_args()
    {"fit": fit, "bench": bench, "servertest": servertest, "auto": auto, "report": report}[a.cmd](a)

if __name__ == "__main__":
    main()
