#!/usr/bin/env python3
"""Paper 8 confirmation speed run on the desktop 2080 Ti (research/prereg/2026-11-04-p8-confirmation-speed.md).

One session, every condition of plan.json a fresh process, three rounds in rotated order, as F2 ran
(tools/turing/ab_speed.sh). exllamav3-family conditions run tools/census.py unchanged (its sampler, prompts, warm-up
and filler). Server conditions (NInferno, llama.cpp v0.9.0, ik_llama.cpp, buun-llama-cpp) start the engine's server,
send one short warm-up request, then the same three prompts with the same 28,000-token filler (built once with
census.filler and exllamav3's tokenizer) and the same sampler, and time each answer the way census.py does: from the
first streamed token to the last, over the tokens generated (256, with the stop tokens ignored).

  python3 run_confirm.py --plan plan.json --out DIR          (the queue step)
  python3 run_confirm.py --plan plan.json --out DIR --dry-run (print the commands, touch nothing)

Rows go to DIR/rows.jsonl (census rows and server rows share the tag, task, fill, new_tokens and decode_wall_s
fields), events to DIR/journal.jsonl, server logs to DIR/logs/. Stdlib only. Nothing is deleted.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

L = os.environ.get("TURING_LAB", "/opt/p8")
PROMPTS = {  # tools/turing/smoke.py, verbatim (census.py imports them from there)
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "math": "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed? Show the steps.",
    "prose": "Describe, in two paragraphs, how a lighthouse keeper's day might have looked in 1890.",
}


def chat(text):  # smoke.chat: Qwen's template with thinking off
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def subst(x, table):
    if isinstance(x, str):
        for k, v in table.items():
            x = x.replace("{" + k + "}", str(v))
        return x
    if isinstance(x, list):
        return [subst(v, table) for v in x]
    if isinstance(x, dict):
        return {k: subst(v, table) for k, v in x.items()}
    return x


def load_plan(path, lab):
    with open(path) as f:
        plan = json.load(f)
    table = {"L": lab}  # {port} stays until a server starts (two ports alternate, so a slow release never blocks)
    plan["models"] = subst(plan["models"], table)
    table.update(plan["models"])
    for key in ("engines", "configs", "quality_check", "conditional"):
        plan[key] = subst(plan[key], table)
    ids = [c["id"] for c in plan["conditions"]]
    assert len(ids) == len(set(ids)), "duplicate condition ids"
    for c in plan["conditions"]:
        assert c["engine"] in plan["engines"] and c["config"] in plan["configs"], c["id"]
    return plan


def gpu_state():
    q = "memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().split(", ")
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return dict(zip(["mem_used_mib", "mem_total_mib", "temp_c", "power_w", "power_limit_w", "sm_mhz", "throttle"], out))


class Journal:
    def __init__(self, path):
        self.path = path

    def __call__(self, kind, **kw):
        row = {"time": now(), "kind": kind, **kw}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


# ---------------------------------------------------------------- the server client (CPU-tested in test_confirm.py)

def request_body(api, text, max_new, sampler, model_id="model"):
    """The request for one measured answer: the census prompt, `max_new` tokens with the stop tokens ignored."""
    if api == "completion":  # llama-server and forks: the raw chat-formatted prompt, as census.py tokenizes it
        return {"prompt": chat(text), "n_predict": max_new, "ignore_eos": True, "cache_prompt": True, "stream": True,
                "temperature": sampler["temperature"], "min_p": sampler["min_p"], "top_k": sampler["top_k"],
                "top_p": sampler["top_p"]}
    if api == "chat":  # NInferno: OpenAI chat with its documented extensions (top_k, min_p, ignore_eos, thinking off)
        return {"model": model_id, "messages": [{"role": "user", "content": text}], "max_tokens": max_new,
                "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}, "temperature": sampler["temperature"],
                "min_p": sampler["min_p"], "top_k": sampler["top_k"], "top_p": sampler["top_p"]}
    raise ValueError(api)


def stream_answer(base, api, body, timeout=900):
    """POST one streamed request; time the first and last token-bearing events. Returns a row fragment."""
    path = "/completion" if api == "completion" else "/v1/chat/completions"
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t_send = time.perf_counter()
    t_first = t_last = None
    chunks, text, final = 0, [], {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            piece = ev.get("content", "") if api == "completion" else "".join(
                (c.get("delta") or {}).get("content") or "" for c in ev.get("choices") or [])
            if piece:
                t = time.perf_counter()
                t_first = t if t_first is None else t_first
                t_last = t
                chunks += 1
                text.append(piece)
            for key in ("timings", "usage", "tokens_predicted", "stop_type", "stopped_limit", "stopped_eos"):
                if key in ev:
                    final[key] = ev[key]
            for c in ev.get("choices") or []:
                if c.get("finish_reason"):
                    final["finish_reason"] = c["finish_reason"]
    timings = final.get("timings") or {}
    usage = final.get("usage") or {}
    new = timings.get("predicted_n") or usage.get("completion_tokens") or final.get("tokens_predicted")
    if t_first is None or not new:
        raise RuntimeError(f"no tokens streamed ({final})")
    return {"new_tokens": int(new), "decode_wall_s": round(t_last - t_first, 4),
            "first_token_s": round(t_first - t_send, 4), "chunks": chunks, "text_chars": len("".join(text)),
            "server_timings": timings, "usage": usage,
            "finish": final.get("finish_reason") or final.get("stop_type"),
            "prompt_tokens": timings.get("prompt_n", 0) + timings.get("cache_n", 0) or usage.get("prompt_tokens")}


def wait_ready(base, proc, deadline_s):
    """Poll /health (llama-server, NInferno), then /v1/models, until 200 or the process exits."""
    t_end = time.time() + deadline_s
    while time.time() < t_end:
        if proc.poll() is not None:
            return f"exited {proc.returncode}"
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(base + path, timeout=5) as r:
                    if r.status == 200:
                        return None
            except (urllib.error.URLError, OSError, ValueError):
                pass
        time.sleep(2)
    return "not ready in time"


def model_id(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
        data = json.load(r).get("data") or [{}]
    return data[0].get("id", "model")


# ---------------------------------------------------------------- the run

class Run:
    def __init__(self, plan, out, dry):
        self.plan, self.out, self.dry = plan, out, dry
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        self.rows = os.path.join(out, "rows.jsonl")
        self.log = Journal(os.path.join(out, "journal.jsonl"))
        self.ports = [plan["port"], plan["port"] + 1]
        self.starts = 0
        self.base = f"http://localhost:{plan['port']}"
        self.filler = ""
        self.dropped = {}      # condition id -> reason (never re-added)
        self.durations = {}    # condition id -> seconds of its last run
        self.t0 = time.time()

    # -- preflight
    def preflight(self):
        avail = {}
        for name, e in self.plan["engines"].items():
            if e["kind"] == "census":
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
                if e["overlay"]:
                    env["PYTHONPATH"] = os.path.join(L, "overlay", e["overlay"])
                got = subprocess.run([os.path.join(L, "venv/bin/python"), "-c",
                                      "import os, exllamav3; from exllamav3.version import __version__ as v; "
                                      "print(v, os.path.dirname(exllamav3.__file__))"],
                                     env=env, capture_output=True, text=True, timeout=300).stdout.strip()
                ok = got.split(" ")[0] == e["expect"] and (not e["overlay"] or e["overlay"] in got)
                avail[name] = ok
                self.log("preflight", engine=name, imports=got, expected=e["expect"], ok=ok)
            else:
                ok = os.access(e["bin"], os.X_OK)
                avail[name] = ok
                self.log("preflight", engine=name, bin=e["bin"], ok=ok)
        for m, p in self.plan["models"].items():
            self.log("model", name=m, path=p, exists=os.path.exists(p),
                     bytes=os.path.getsize(p) if os.path.isfile(p) else None)
        for c in self.plan["conditions"]:
            if not avail[c["engine"]]:
                self.dropped[c["id"]] = "engine not available (build failed or overlay import mismatch)"
                self.log("drop", condition=c["id"], reason=self.dropped[c["id"]])

    def make_filler(self):
        """census.filler with exllamav3's tokenizer, on the CPU: the same text census.py builds for --fill 28000."""
        path = os.path.join(self.out, f"filler-{self.plan['fill_long']}.txt")
        code = ("import sys; sys.path.insert(0, sys.argv[1]); from census import filler; "
                "from exllamav3 import Config, Tokenizer; t = Tokenizer.from_config(Config.from_directory(sys.argv[2])); "
                "open(sys.argv[3], 'w').write(filler(t, int(sys.argv[4])))")
        subprocess.run([os.path.join(L, "venv/bin/python"), "-c", code, os.path.join(L, "tools"),
                        self.plan["models"]["exl3"], path, str(self.plan["fill_long"])],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True, timeout=600)
        with open(path) as f:
            self.filler = f.read()
        self.log("filler", path=path, chars=len(self.filler))

    # -- one condition
    def census_cmd(self, c, tag):
        e = self.plan["engines"][c["engine"]]
        drafted = self.plan["configs"][c["config"]]["drafted"]
        cmd = [os.path.join(L, "venv/bin/python"), os.path.join(L, "tools/census.py"),
               "-m", self.plan["models"]["exl3"], "-cs", "32768", "-cq", e["cache"], "-rcs", "2",
               "-chunk_size", "1024", "--tag", tag, "--out", self.rows]
        if c["fill"]:
            cmd += ["--fill", str(c["fill"])]
        if drafted:
            cmd += ["-mtp", "-ndt", "6", "-dds"]
        env = dict(os.environ, XDG_CACHE_HOME=os.path.join(L, "cache"))
        env.update(e["env_mtp"] if drafted else e["env"])
        if e["overlay"]:
            env["PYTHONPATH"] = os.path.join(L, "overlay", e["overlay"])
        return cmd, env

    def server_cmd(self, c, extra=()):
        e = self.plan["engines"][c["engine"]]
        port = self.ports[self.starts % 2]
        self.starts += 1
        self.base = f"http://localhost:{port}"
        cmd = subst([e["bin"]] + self.plan["configs"][c["config"]]["args"] + e["common"] + list(extra),
                     {"port": port})
        env = dict(os.environ, XDG_CACHE_HOME=os.path.join(L, "cache"))
        env.update(e["env"])
        if "HOME" in e["env"]:
            os.makedirs(e["env"]["HOME"], exist_ok=True)
        return cmd, env

    def start_server(self, cmd, env, label):
        logf = open(os.path.join(self.out, "logs", f"{label}.log"), "a")
        logf.write(f"\n== {now()} {' '.join(cmd)}\n")
        logf.flush()
        return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True), logf

    def stop_server(self, proc, logf):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
        logf.close()
        self.wait_memory()

    def wait_memory(self, budget_s=90):
        """After a server stops, wait until the card's memory is back near the session's baseline."""
        t_end = time.time() + budget_s
        while time.time() < t_end:
            g = gpu_state()
            if g and int(g.get("mem_used_mib", 0)) <= self.baseline_mib + 256:
                return
            time.sleep(2)
        self.log("memory-not-back", gpu=gpu_state())

    def cool(self):
        g = gpu_state()
        if g and int(g.get("temp_c", 0)) >= 84:
            t_end = time.time() + 600
            while time.time() < t_end and int(gpu_state().get("temp_c", 0)) > 78:
                time.sleep(10)
            self.log("cooled", gpu=gpu_state())

    def run_server_condition(self, c, tag):
        cfg = self.plan["configs"][c["config"]]
        e = self.plan["engines"][c["engine"]]
        cmd, env = self.server_cmd(c)
        if self.dry:
            print(tag, " ".join(cmd))
            return True
        proc, logf = self.start_server(cmd, env, c["id"])
        try:
            err = wait_ready(self.base, proc, 600)
            if err:
                raise RuntimeError(f"server {err}")
            mid = model_id(self.base) if e["api"] == "chat" else "model"
            warm = request_body(e["api"], PROMPTS["code"], 32, self.plan["sampler"], mid)
            stream_answer(self.base, e["api"], warm)
            pre = self.filler if c["fill"] else ""
            for task in self.plan["tasks"]:
                body = request_body(e["api"], pre + PROMPTS[task], self.plan["max_new"], self.plan["sampler"], mid)
                row = stream_answer(self.base, e["api"], body, timeout=self.plan["condition_timeout_s"])
                row.update(tag=tag, task=task, fill=c["fill"], mtp=cfg["drafted"], engine=e["label"],
                           condition=c["id"], time=now(), gpu=gpu_state(), sampler=self.plan["sampler"])
                with open(self.rows, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("tag", "task", "new_tokens", "decode_wall_s")}), flush=True)
            return True
        finally:
            self.stop_server(proc, logf)

    def run_condition(self, c, rnd):
        tag = f"confirm-{c['id']}-r{rnd}"
        self.cool()
        t = time.time()
        self.log("start", condition=c["id"], round=rnd, gpu=gpu_state())
        try:
            if self.plan["engines"][c["engine"]]["kind"] == "census":
                cmd, env = self.census_cmd(c, tag)
                if self.dry:
                    print(tag, " ".join(cmd))
                    return
                with open(os.path.join(self.out, "logs", f"{c['id']}.log"), "a") as logf:
                    logf.write(f"\n== {now()} {' '.join(cmd)}\n")
                    logf.flush()
                    subprocess.run(cmd, env=env, cwd=os.path.join(L, "tools"), stdout=logf, stderr=subprocess.STDOUT,
                                   timeout=self.plan["condition_timeout_s"], check=True)
            else:
                self.run_server_condition(c, tag)
            self.durations[c["id"]] = time.time() - t
            self.log("end", condition=c["id"], round=rnd, seconds=round(self.durations[c["id"]], 1))
        except Exception as exc:  # noqa: BLE001 - one condition's failure drops that condition, never the session
            self.dropped[c["id"]] = f"round {rnd}: {type(exc).__name__}: {exc}"[:500]
            self.log("drop", condition=c["id"], round=rnd, reason=self.dropped[c["id"]],
                     seconds=round(time.time() - t, 1))
            self.wait_memory()

    def prepare(self):
        """Start each NInferno model once (its GGUF is converted on the first start, into the GGUF's folder)."""
        seen = set()
        for c in self.plan["conditions"]:
            if c["engine"] != "ninferno" or c["id"] in self.dropped or c["config"] in seen:
                continue
            seen.add(c["config"])
            cmd, env = self.server_cmd(c)
            if self.dry:
                print("prepare", " ".join(cmd))
                continue
            t = time.time()
            proc, logf = self.start_server(cmd, env, "prepare-" + c["config"])
            try:
                err = wait_ready(self.base, proc, 1800)
            finally:
                self.stop_server(proc, logf)
            self.log("prepare", config=c["config"], ok=err is None, error=err, seconds=round(time.time() - t, 1))
            if err:
                for x in self.plan["conditions"]:
                    if x["config"] == c["config"]:
                        self.dropped[x["id"]] = f"prepare: server {err}"

    # -- the budget rule: before rounds 2 and 3, cut the lowest priorities until the projection fits the cap
    def cut_to_cap(self, rnd, kept, extra_s):
        cap = self.plan["cap_hours"] * 3600
        rounds_left = self.plan["rounds"] - rnd + 1

        def projection(ids):
            return (time.time() - self.t0) + rounds_left * sum(self.durations.get(i, 120) for i in ids) + extra_s

        order = sorted((c for c in self.plan["conditions"] if c["id"] in kept),
                       key=lambda c: (-c["priority"], 0 if c["fill"] else 1))
        for c in order:
            if projection(kept) <= cap or c["priority"] == 1:
                break
            kept.remove(c["id"])
            self.dropped[c["id"]] = f"cut before round {rnd} by the budget rule"
            self.log("cut", condition=c["id"], round=rnd, projected_h=round(projection(kept) / 3600, 2))
        return kept

    def quality_check(self):
        q = self.plan["quality_check"]
        if all(c["id"] in self.dropped for c in self.plan["conditions"] if c["engine"] == q["engine"]):
            self.log("quality-skip", reason="its engine's conditions were all dropped or cut")
            return
        cmd, env = self.server_cmd({"engine": q["engine"], "config": q["config"]}, q["extra_args"])
        if self.dry:
            print("quality", " ".join(cmd))
            return
        proc, logf = self.start_server(cmd, env, "quality-" + q["label"])
        try:
            err = wait_ready(self.base, proc, 600)
            if err:
                self.log("quality-fail", reason=err)
                return
            r = subprocess.run([os.path.join(L, "venv/bin/python"), os.path.join(L, "tools/quality/run_eval_turing.py"),
                                "--suite", q["suite"], "--label", q["label"], "--limit", str(q["limit"]),
                                "--base-url", self.base, "--temperature", "0", "--reasoning-effort", "medium",
                                "--max-tokens", "4096", "--concurrency", "1",
                                "--results-dir", os.path.join(self.out, "quality"),
                                "--summary-out", os.path.join(self.out, "quality", "summary.jsonl")],
                               timeout=3600, capture_output=True, text=True)
            self.log("quality", returncode=r.returncode, tail=r.stdout[-1500:])
        finally:
            self.stop_server(proc, logf)

    def conditional(self):
        p = self.plan["conditional"]["power240"]
        ok = subprocess.run(["sudo", "-n", "-l", "/usr/bin/nvidia-smi", "-pl", str(p["cap_w"])],
                            capture_output=True).returncode == 0
        self.log("conditional", arm="power240", runs=ok, rule=p["runs_if"])
        if ok and not self.dry:
            subprocess.run(["sudo", "-n", "/usr/bin/nvidia-smi", "-pl", str(p["cap_w"])], check=False)
            try:
                for cid in p["conditions"]:
                    c = next(x for x in self.plan["conditions"] if x["id"] == cid)
                    self.run_condition(dict(c, id=cid + "-240w"), 1)
            finally:
                subprocess.run(["sudo", "-n", "/usr/bin/nvidia-smi", "-pl", "260"], check=False)
        d = self.plan["conditional"]["nif-dflash2"]
        hub = os.path.join(L, "cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2")
        self.log("conditional", arm="nif-dflash2", runs=os.path.isdir(hub), rule=d["runs_if"])
        if os.path.isdir(hub):
            self.plan["configs"]["nif-xxs-df2"] = {"drafted": True, "args": d["config_args"]}
            for fill in d["fills"]:
                c = {"id": f"nif-xxs-df2-{'28k' if fill else '0'}", "engine": "ninferno", "config": "nif-xxs-df2",
                     "fill": fill, "priority": 5}
                self.run_condition(c, 1)

    def main(self):
        g = gpu_state()
        self.baseline_mib = int(g.get("mem_used_mib", 400)) if g else 400
        self.log("session", plan=self.plan["registration"], gpu=g, load=os.getloadavg(), baseline_mib=self.baseline_mib)
        self.preflight()
        self.make_filler()
        self.prepare()
        kept = [c["id"] for c in self.plan["conditions"] if c["id"] not in self.dropped]
        extra = 900 + 300  # the quality check and the conditional arms, reserved in the projection
        for rnd in range(1, self.plan["rounds"] + 1):
            if rnd > 1:
                kept = self.cut_to_cap(rnd, [k for k in kept if k not in self.dropped], extra)
            ordered = [c for c in self.plan["conditions"] if c["id"] in kept]
            n = len(ordered)
            ordered = [ordered[(i + rnd - 1) % n] for i in range(n)]  # ab_speed.sh's rotation
            self.log("round", round=rnd, conditions=[c["id"] for c in ordered])
            for c in ordered:
                if c["id"] not in self.dropped:
                    self.run_condition(c, rnd)
        self.quality_check()
        self.conditional()
        self.log("done", hours=round((time.time() - self.t0) / 3600, 2), dropped=self.dropped)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan = load_plan(args.plan, L)
    run = Run(plan, args.out, args.dry_run)
    if args.dry_run:
        run.baseline_mib = 0
        for c in plan["conditions"]:
            if plan["engines"][c["engine"]]["kind"] == "census":
                cmd, env = run.census_cmd(c, f"confirm-{c['id']}-r1")
                extra = {k: v for k, v in env.items() if k.startswith(("EXL3_", "PYTHONPATH", "CUDA_HOME", "NVCC"))}
                print(c["id"], extra, " ".join(cmd))
            else:
                cmd, env = run.server_cmd(c)
                print(c["id"], " ".join(cmd))
        return
    run.main()


if __name__ == "__main__":
    sys.exit(main())
