#!/usr/bin/env python3
"""Ledger item 21: tokens-per-joule on a power-capped laptop GPU, sustained for N minutes.
Samples nvidia-smi every --sample-s (power, SM/mem clocks, temp, util, throttle reasons) in a thread, runs
back-to-back single-user generations (cycling math/code/prose/rewrite prompts), integrates energy per request and
for the whole run, reports gross and idle-corrected tokens/J, 5-minute windows (drift), throttle-reason
histogram, and aborts on dangerous temperatures (GPU >= 90 C or CPU package >= 98 C, three samples in a row).
usage: sustained_tpj.py --port 18120 --label L --minutes 30 [--max-tokens 2048] [--sample-s 2] [--effort medium]"""
import argparse, json, os, statistics, subprocess, threading, time, urllib.request

PROMPTS = {
    "math": "Solve step by step, then end with 'Answer: <number>'.\n\nA rectangular garden has a perimeter of 56 meters. Its length is 4 meters more than twice its width. A path 1 meter wide runs along the inside of the whole perimeter. What is the area of the garden not covered by the path?",
    "code": "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals. Include type hints, a docstring, and five pytest tests covering edge cases.",
    "prose": "Write a 600-word essay on why laptops with 12 GB of GPU memory are an interesting target for running large language models locally, covering quantization, speculative decoding, and power limits.",
    "rewrite": "Rewrite the following paragraph in a formal academic register, keeping every fact: 'So basically our laptop has a 4080 with 12 gigs and it's capped at 80 watts. We tried a bunch of 2-bit quants and honestly the trellis one was way better than we expected, it kept up with the 4-bit one on code and only lost a couple points on math. Speculative decoding with the built-in MTP head gave us like 1.6x on code.'",
}
QUERY = "timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active"


def cpu_pkg_temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "Package id 0" in line:
                return float(line.split("+")[1].split("°")[0])
    except Exception:
        pass
    try:
        return max(int(open(p).read()) for p in __import__("glob").glob("/sys/class/thermal/thermal_zone*/temp")) / 1000
    except Exception:
        return None


class Sampler(threading.Thread):
    def __init__(self, interval, path):
        super().__init__(daemon=True); self.interval = interval; self.path = path; self.rows = []; self.stop = False; self.abort = None; self.hot = 0

    def run(self):
        with open(self.path, "a") as f:
            n = 0
            while not self.stop:
                t = time.time()
                try:
                    out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
                    p = [x.strip() for x in out.split(",")]
                    row = {"t": t, "power_w": float(p[1]), "sm_mhz": int(p[2]), "mem_mhz": int(p[3]), "gpu_c": int(p[4]), "util": int(p[5]), "mem_mib": int(p[6]), "throttle": p[7]}
                except Exception as e:
                    row = {"t": t, "err": repr(e)}
                if n % 5 == 0: row["cpu_c"] = cpu_pkg_temp()
                n += 1; self.rows.append(row); f.write(json.dumps(row) + "\n"); f.flush()
                g = row.get("gpu_c") or 0; c = row.get("cpu_c") or 0
                self.hot = self.hot + 1 if (g >= 90 or c >= 98) else 0
                if self.hot >= 3: self.abort = f"THERMAL gpu={g} cpu={c}"
                time.sleep(max(0.2, self.interval - (time.time() - t)))

    def energy(self, t0, t1):
        rows = [r for r in self.rows if "power_w" in r and t0 <= r["t"] <= t1]
        if len(rows) < 2: return 0.0, 0.0
        e = sum((rows[i + 1]["t"] - rows[i]["t"]) * (rows[i]["power_w"] + rows[i + 1]["power_w"]) / 2 for i in range(len(rows) - 1))
        return e, e / max(1e-6, rows[-1]["t"] - rows[0]["t"])


def chat(port, prompt, max_tokens, effort, timeout):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "stream": False, "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    u = d.get("usage") or {}; tm = d.get("timings") or {}
    ct = u.get("completion_tokens") or tm.get("predicted_n")
    if ct is None:
        msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
        txt = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        ereq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/token/encode", data=json.dumps({"text": txt}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(ereq, timeout=60) as er: ct = json.loads(er.read()).get("length")
    return ct or 0, tm.get("prompt_ms"), tm.get("predicted_per_second")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18120); ap.add_argument("--label", required=True)
    ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--sample-s", type=float, default=2.0); ap.add_argument("--effort", default="medium")
    ap.add_argument("--idle-s", type=float, default=30); ap.add_argument("--prompts", default="math,code,prose,rewrite")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpj_rows.jsonl"))
    ap.add_argument("--concurrency", type=int, default=1, help="N client threads issuing requests back-to-back (batching test)")
    args = ap.parse_args()
    gpu_log = os.path.join(os.path.dirname(args.out), f"tpj_gpu_{args.label}.jsonl")
    s = Sampler(args.sample_s, gpu_log); s.start()
    time.sleep(args.idle_s)
    idle_e, idle_w = s.energy(time.time() - args.idle_s, time.time())
    print(json.dumps({"label": args.label, "phase": "idle", "idle_w": round(idle_w, 1), "mem_mib": s.rows[-1].get("mem_mib")}), flush=True)
    names = args.prompts.split(","); reqs = []; t_start = time.time(); aborted = None
    lock = threading.Lock()

    def worker(sid):
        i = sid
        while time.time() - t_start < args.minutes * 60 and not s.abort:
            name = names[i % len(names)]; i += args.concurrency; t0 = time.time()
            try:
                ct, pms, tps = chat(args.port, PROMPTS[name], args.max_tokens, args.effort, timeout=1200)
            except Exception as e:
                print(json.dumps({"label": args.label, "stream": sid, "req": i, "error": repr(e)}), flush=True); time.sleep(5); continue
            t1 = time.time(); e, pw = s.energy(t0, t1)
            row = {"label": args.label, "stream": sid, "req": i, "prompt": name, "tokens": ct, "wall_s": round(t1 - t0, 2), "tok_s": round(ct / max(1e-6, t1 - t0), 2), "server_tps": tps,
                   "energy_j": round(e, 1), "mean_w": round(pw, 1), "tok_per_j": round(ct / max(1e-6, e), 3), "elapsed_min": round((t1 - t_start) / 60, 1)}
            with lock:
                reqs.append(row); print(json.dumps(row), flush=True)
                with open(args.out, "a") as f: f.write(json.dumps(row) + "\n")

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(args.concurrency)]
    for t in threads: t.start()
    for t in threads: t.join()
    aborted = s.abort; t_end = time.time(); s.stop = True; time.sleep(args.sample_s + 0.5)
    run_rows = [r for r in s.rows if "power_w" in r and t_start <= r["t"] <= t_end]
    tot_e, mean_w = s.energy(t_start, t_end); tot_tok = sum(r["tokens"] for r in reqs); dur = t_end - t_start
    windows = []
    for w0 in range(0, int(dur), 300):
        rr = [r for r in reqs if w0 <= (r["elapsed_min"] * 60) < w0 + 300]
        wr = [r for r in run_rows if t_start + w0 <= r["t"] < t_start + w0 + 300]
        if rr and wr:
            we = sum((wr[k + 1]["t"] - wr[k]["t"]) * (wr[k]["power_w"] + wr[k + 1]["power_w"]) / 2 for k in range(len(wr) - 1))
            windows.append({"min": w0 // 60, "tok": sum(r["tokens"] for r in rr), "tok_s": round(sum(r["tokens"] for r in rr) / sum(r["wall_s"] for r in rr), 1),
                            "mean_w": round(sum(r["power_w"] for r in wr) / len(wr), 1), "sm_mhz": round(sum(r["sm_mhz"] for r in wr) / len(wr)), "gpu_c_max": max(r["gpu_c"] for r in wr),
                            "tok_per_j": round(sum(r["tokens"] for r in rr) / max(1e-6, we), 3)})
    thr = {}
    for r in run_rows: thr[r["throttle"]] = thr.get(r["throttle"], 0) + 1
    per_stream = round(statistics.mean(r["tok_s"] for r in reqs), 2) if reqs else None
    summ = {"label": args.label, "phase": "summary", "concurrency": args.concurrency, "minutes": round(dur / 60, 1), "requests": len(reqs), "tokens": tot_tok, "tok_s": round(tot_tok / dur, 2), "per_stream_tok_s": per_stream,
            "energy_kj": round(tot_e / 1000, 2), "mean_w": round(mean_w, 1), "idle_w": round(idle_w, 1), "tok_per_j_gross": round(tot_tok / max(1e-6, tot_e), 3),
            "tok_per_j_net": round(tot_tok / max(1e-6, tot_e - idle_w * dur), 3), "gpu_c_max": max(r["gpu_c"] for r in run_rows), "sm_mhz_min": min(r["sm_mhz"] for r in run_rows),
            "sm_mhz_mean": round(sum(r["sm_mhz"] for r in run_rows) / len(run_rows)), "cpu_c_max": max((r.get("cpu_c") or 0) for r in run_rows),
            "throttle_hist": thr, "windows": windows, "aborted": aborted}
    print(json.dumps(summ), flush=True)
    with open(args.out, "a") as f: f.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
