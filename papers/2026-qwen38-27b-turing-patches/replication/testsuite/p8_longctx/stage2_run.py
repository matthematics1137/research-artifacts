#!/usr/bin/env python3
"""P8 long context, stages 2-3: run a plan of arms on the desktop 2080 Ti, item by item (research/prereg/
2026-11-01-p8-longctx-stages2-3.md and its Amendment 1).

Runs ON the desktop as one queue step, inside that step's capped turing-run unit. A plan (JSON, written on the laptop)
lists arms in the registered order. Each arm gets one fresh TabbyAPI server (stage 1's launcher: stage1_run.py), then
its items one at a time in manifest order: thinking off, temperature 0, no prefix reuse, and the arm's output-token
limit (768 for graded and second-calibration items, 128 otherwise: Amendments 2-3). Every item is journaled as it
completes, so a stopped arm resumes exactly; scoring happens on the laptop (stage2_analyze.py).

  - Exclusions: a request over 25 minutes (the server is stopped and a fresh one started for the next item), or a
    server error with no out-of-memory cause (a fresh server if it died). Both are counted per arm.
  - An out-of-memory error stops that arm with no retry; the plan continues with the next arm.
  - Budget: each plan names its group (stage2: 22 GPU hours with both calibrations and the rescue, Amendment 3;
    stage3: 8 with the probe; fd75d-arm: 3.5; replication: 15).
    Use is the summed wall time of every arm session in the group's journal. Before each item the arm's mean item
    time so far (first item: stage 1's prefill time at that length plus 30 s) must fit, or the arm and the plan stop.
    A cliff arm starts only if its whole estimate fits (Amendment 1, item 9). The replication gives every arm a
    registered per-item estimate (if_budget per_item_s) and cuts from the end of its priority order: once an arm's
    whole estimate does not fit, it and every later arm are skipped (plan field cut_from_end).
  - The replication's R2 reruns stage 2's fp16 arm at 32K only if a 16-item check in a fresh server does not
    reproduce stage 2's answer texts exactly (arm field skip_if_reproduced).
  - Thermal: at 86 C or more on three samples running, our server is stopped, the run waits until the GPU is under
    70 C and a fresh server resumes the arm (the guard's own stop is 87 C).
  - Stage 3's reach probe runs stage 1's step function unchanged (stage1_run.run_step), under stage 1's rules.
Stdlib only. Nothing is deleted.

  python stage2_run.py PLAN.json --items DIR --out DIR [--port 8098]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_config as C  # noqa: E402
import stage1_run as R1  # noqa: E402
import stage1_spec as S  # noqa: E402

MAX_TOKENS = 128          # every request unless its arm sets max_tokens (graded arms: 768, Amendment 3)
REQUEST_S = 25 * 60
COOL_C = 70.0
BUDGET_S = {"stage2": 22 * 3600, "stage3": 8 * 3600,
            "fd75d-arm": int(3.5 * 3600),   # research/prereg/2026-11-03-p8-fd75d-accuracy-arm.md: at most 3.5 h
            "replication": 15 * 3600}       # research/prereg/2026-11-04-p8-value-bits-replication.md: 15 GPU hours
# Formats beyond stage 1's map, for stages 2-3 only (stage 1's module is unchanged): the replication's R3 runs 8-bit
# keys with 6-bit values
EXTRA_FORMATS = {"8,6": "8,6"}
for _fmt, _mode in EXTRA_FORMATS.items():
    S.CACHE_MODE.setdefault(_fmt, _mode)
# Stage 1 reach (MiB headroom >= 512) and measured prefill seconds per prompt token at the lengths it ran. 8,6 was not
# measured: its cache is smaller than 8,8's at every length, so 8,8's measured reach bounds it (the replication's
# registration)
REACH = {"8,8": 72 * S.K, "4,4": 136 * S.K, "8,4": 96 * S.K, "fp16": 40 * S.K, "8,6": 72 * S.K}
PREFILL_S = {32 * S.K: 51.3, 64 * S.K: 113.5, 96 * S.K: 187.1, 128 * S.K: 271.3, 136 * S.K: 293.8}


class Journal(R1.Journal):
    def items_done(self, arm):
        return {r["uid"] for r in self.rows if r.get("kind") == "item" and r["arm"] == arm}

    def arm_closed(self, arm):
        return next((r for r in self.rows if r.get("kind") in ("arm_end", "arm_skipped") and r["arm"] == arm), None)


def group_used_s(journal, exclude=None):
    """GPU time used by every arm session (and probe step) in this journal, interrupted sessions included; `exclude`
    leaves out the session still running, which the caller counts itself."""
    used = 0.0
    starts = [r for r in journal.rows if r.get("kind") == "session_start" and r["session"] != exclude]
    ends = {r["session"]: r for r in journal.rows if r.get("kind") == "session_end"}
    for s in starts:
        if s["session"] in ends:
            used += ends[s["session"]]["wall_s"]
        else:  # cut off by a guard stop: count up to its last journaled row
            later = [r.get("t_end", r.get("t_req", 0)) for r in journal.rows
                     if r.get("session") == s["session"] and r.get("kind") == "item"]
            used += max([s["t_start"]] + later) - s["t_start"]
    used += sum(r.get("wall_s") or 0 for r in journal.rows if r.get("kind") == "result")   # probe steps (stage 1)
    return used


def first_item_estimate(length, prompt_tokens):
    known = min(PREFILL_S, key=lambda x: abs(x - length))
    return PREFILL_S[known] * prompt_tokens / max(known - S.PROMPT_MARGIN, 1) + 30


def load_items(path, select=None):
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if select and select.get("first_per_family"):
        n, seen, out = select["first_per_family"], {}, []
        for it in items:
            if seen.get(it["family"], 0) < n:
                out.append(it)
                seen[it["family"]] = seen.get(it["family"], 0) + 1
        items = out
    return items


def arm_env(spec):
    """An arm's own environment for its server (the fd75d accuracy arm sets EXL3_FD75D=0): EXL3_* names only, string
    values, merged over the launcher's environment. Anything else is a harness error."""
    env = spec.get("env") or {}
    bad = {k: v for k, v in env.items() if not k.startswith("EXL3_") or not isinstance(v, str)}
    if bad:
        raise R1.HarnessError(f"{spec['arm']}: arm env accepts EXL3_* names with string values only: {bad}")
    return dict(env)


def reproduced(args, journal, rule):
    """The replication's R2: a stage 2 arm stands in for a rerun only if a fresh server reproduces it. True when the
    check arm finished all `n` items and each answer text equals the reference arm's text for that item in the
    reference journal (a path relative to --out). Returns (ok, details for the journal)."""
    path = os.path.normpath(os.path.join(args.out, rule["reference_journal"]))
    ref = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == "item" and r.get("arm") == rule["reference_arm"] and r.get("status") == "ok":
                    ref[r["uid"]] = r.get("text")
    got = {r["uid"]: r.get("text") for r in journal.rows
           if r.get("kind") == "item" and r["arm"] == rule["check_arm"] and r.get("status") == "ok"}
    closed = journal.arm_closed(rule["check_arm"])
    same = sum(1 for u, t in got.items() if u in ref and ref[u] == t)
    ok = (closed is not None and closed.get("status") == "done" and len(got) == rule["n"] and same == rule["n"])
    return ok, {"reference": path, "reference_arm": rule["reference_arm"], "check_arm": rule["check_arm"],
                "n_check": len(got), "identical": same}


def server_env_for(args, spec):
    env = R1.server_env(args.tabby_logs)
    env.update(arm_env(spec))
    return env


class Server:
    """One TabbyAPI process for one arm session (stage 1's launcher and stop)."""

    def __init__(self, args, journal, arm, fmt, length, session, extra_env=None):
        self.args, self.journal, self.arm, self.fmt, self.length, self.session = args, journal, arm, fmt, length, session
        self.extra_env = extra_env or {}
        self.proc = self.logf = None
        self.url = f"http://localhost:{args.port}"

    def start(self, sampler):
        tag = time.strftime("%Y%m%dT%H%M%S")
        cfg = os.path.join(self.args.out, "configs", f"{self.arm}-{tag}.yml")
        text = C.render(self.fmt, self.length, self.args.port, f"{R1.LAB}/models")
        missing = C.check(text, self.fmt, self.length, self.args.port)
        if missing:
            raise R1.HarnessError(f"{self.arm}: config lines missing: {missing}")
        with open(cfg, "w") as f:
            f.write(text)
        self.log_path = os.path.join(self.args.out, "logs", f"{self.arm}.log")
        self.logf = open(self.log_path, "a")
        self.logf.write(f"== {R1.stamp()} stage2_run.py {self.arm} session {self.session} config {cfg}\n")
        self.logf.flush()
        self.offset = os.path.getsize(self.log_path)
        t0 = R1.now()
        self.proc = subprocess.Popen([R1.PY, "-c", R1.TABBY_BOOT, "--config", cfg], cwd=R1.TABBY,
                                     env=server_env_for(self.args, {"arm": self.arm, "env": self.extra_env}),
                                     stdout=self.logf,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        sampler.pid = self.proc.pid
        while True:
            if self.proc.poll() is not None:
                tail = R1.log_tail_from(self.log_path, self.offset)
                m = R1.oom_in(tail)
                return ("oom" if m else "harness"), f"server exited during load ({m or 'rc %s' % self.proc.returncode})"
            try:
                code, _ = R1.http("GET", self.url + "/health", timeout=3)
            except Exception:
                code = None
            if code == 200:
                break
            if R1.now() - t0 > S.HANG_S:
                return "harness", "no healthy server within 25 minutes"
            time.sleep(2)
        load_s = round(R1.now() - t0, 1)
        _, body = R1.http("GET", self.url + "/v1/model", timeout=10)
        params = (json.loads(body).get("parameters") or {})
        if params.get("cache_size") is not None and int(params["cache_size"]) != self.length:
            return "harness", f"server cache_size {params['cache_size']}, config {self.length}"
        mode = str(params.get("cache_mode", ""))
        if mode and mode.replace(" ", "").upper() != S.CACHE_MODE[self.fmt].upper():
            return "harness", f"server cache_mode {mode!r}, config {S.CACHE_MODE[self.fmt]!r}"
        self.journal.add({"kind": "server_ready", "arm": self.arm, "session": self.session, "load_s": load_s,
                          "arm_env": self.extra_env,
                          "config": cfg, "cache_mode": mode, "cache_size": params.get("cache_size")})
        return "ok", None

    def stop(self):
        how = R1.stop_server(self.proc) if self.proc else "none"
        if self.logf:
            self.logf.close()
        return how

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


def request_body(item, max_tokens):
    return {"model": "qwen38-exl3-2.0", "messages": [{"role": "user", "content": item["content"]}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}


def ask(server, item, sampler, max_tokens=MAX_TOKENS):
    """One item: (status, row fields). Status: ok, timeout, server_error, oom, thermal."""
    body = request_body(item, max_tokens)
    box = {}

    def call():
        try:
            box["resp"] = R1.http("POST", server.url + "/v1/chat/completions", body, timeout=REQUEST_S + 60)
        except urllib.error.HTTPError as e:
            box["http_error"] = (e.code, e.read().decode("utf-8", "replace"))
        except Exception as e:
            box["error"] = repr(e)

    offset = os.path.getsize(server.log_path)
    t0 = R1.now()
    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    status = None
    while worker.is_alive():
        if R1.now() - t0 > REQUEST_S:
            status = "timeout"
            break
        if sampler.hot:
            status = "thermal"
            break
        if not server.alive():
            worker.join(timeout=5)
            break
        worker.join(timeout=1)
    out = {"t_req": t0, "t_end": R1.now(), "wall_s": round(R1.now() - t0, 2)}
    if status:
        return status, out
    if "resp" in box:
        resp = json.loads(box["resp"][1])
        u = resp.get("usage") or {}
        ch = (resp.get("choices") or [{}])[0]
        out.update({"text": (ch.get("message") or {}).get("content") or "", "finish_reason": ch.get("finish_reason"),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "prompt_time_s": u.get("prompt_time"), "prompt_tok_s": u.get("prompt_tokens_per_sec"),
                    "completion_time_s": u.get("completion_time"),
                    "completion_tok_s": u.get("completion_tokens_per_sec")})
        return "ok", out
    detail = box.get("http_error") or box.get("error") or f"server exited, rc {server.proc.returncode}"
    tail = R1.log_tail_from(server.log_path, offset)
    m = R1.oom_in(json.dumps(detail)) or R1.oom_in(tail)
    out["error"] = str(detail)[:600]
    return ("oom" if m else "server_error"), out


def wait_cool(journal, arm):
    t0 = R1.now()
    while True:
        t = float(R1.smi("temperature.gpu")[0])
        if t < COOL_C:
            journal.add({"kind": "thermal_resume", "arm": arm, "waited_s": round(R1.now() - t0, 1), "temp_c": t})
            return
        time.sleep(30)


def run_arm(args, journal, spec, group, session_counter):
    arm, fmt, length = spec["arm"], spec["format"], spec["length"]
    if length > REACH.get(fmt, 0) and length not in args.probe_passed.get(fmt, set()):
        raise R1.HarnessError(f"{arm}: {fmt} at {length} passed neither stage 1 nor the probe")
    items = load_items(os.path.join(args.items, spec["items"]), spec.get("select"))
    done = journal.items_done(arm)
    todo = [it for it in items if it["uid"] not in done]
    if spec.get("skip_if_reproduced"):
        ok, info = reproduced(args, journal, spec["skip_if_reproduced"])
        journal.add({"kind": "reproduction_check", "arm": arm, "reproduced": ok, **info})
        if ok:
            journal.add({"kind": "arm_skipped", "arm": arm, "why": "reproduced", **info})
            return "reproduced"
    if spec.get("if_budget"):
        b = spec["if_budget"]
        if "per_item_s" in b:      # a registered estimate (the replication: every arm, cut from the end)
            per = float(b["per_item_s"])
        else:
            ref = [r for r in journal.rows if r.get("kind") == "item" and r["arm"] == b["per_item_from"]
                   and r.get("status") == "ok"]
            per = (sum(r["wall_s"] for r in ref) / len(ref)) if ref else first_item_estimate(length,
                                                                                               items[0]["prompt_tokens"])
        est = per * len(todo) + 120
        if group_used_s(journal) + est > BUDGET_S[group]:
            journal.add({"kind": "arm_skipped", "arm": arm, "why": "budget", "estimate_s": round(est, 1),
                         "used_s": round(group_used_s(journal), 1), "budget_s": BUDGET_S[group]})
            return "skipped"
    max_tokens = spec.get("max_tokens", MAX_TOKENS)
    extra_env = arm_env(spec)
    journal.add({"kind": "arm_plan", "arm": arm, "format": fmt, "length": length, "n_items": len(items),
                 "max_tokens": max_tokens, "arm_env": extra_env,
                 "n_todo": len(todo), "items_file": spec["items"]})
    status = "done"
    while todo:
        session_counter[0] += 1
        session = f"{arm}#{session_counter[0]}-{int(R1.now())}"
        csv_path = os.path.join(args.out, "samples", f"{session.replace('#', '-')}.csv")
        baseline, apps = R1.mem_used(), R1.compute_apps()
        t_start = R1.now()
        journal.add({"kind": "session_start", "arm": arm, "session": session, "t_start": t_start,
                     "baseline_mib": baseline, "gpu_apps": apps, "samples": csv_path})
        sampler = R1.Sampler(csv_path)
        sampler.start()
        server = Server(args, journal, arm, fmt, length, session, extra_env)
        restart, outcome = False, None
        try:
            ok, why = server.start(sampler)
            if ok != "ok":
                journal.add({"kind": "arm_end", "arm": arm, "status": "oom_stop" if ok == "oom" else "harness_error",
                             "reason": why})
                status = "oom_stop" if ok == "oom" else "harness_error"
                todo = []
            per_item = []
            while todo and not restart:
                it = todo[0]
                est = (sum(per_item) / len(per_item)) if per_item else first_item_estimate(length, it["prompt_tokens"])
                if group_used_s(journal, exclude=session) + (R1.now() - t_start) + est > BUDGET_S[group]:
                    journal.add({"kind": "arm_end", "arm": arm, "status": "budget_stop", "reason":
                                 f"next item estimated {est:.0f} s", "items_left": len(todo)})
                    status, todo = "budget_stop", []
                    break
                st, row = ask(server, it, sampler, max_tokens)
                if st == "thermal":
                    journal.add({"kind": "thermal_pause", "arm": arm, "uid": it["uid"], "session": session})
                    restart = True
                    outcome = "thermal"
                    break
                journal.add({"kind": "item", "arm": arm, "session": session, "uid": it["uid"],
                             "family": it["family"], "level": it["level"], "status": st,
                             "prompt_tokens_built": it["prompt_tokens"], **row})
                todo = todo[1:]
                if st == "ok":
                    per_item.append(row["wall_s"])
                elif st == "oom":
                    journal.add({"kind": "arm_end", "arm": arm, "status": "oom_stop", "reason": row.get("error"),
                                 "items_left": len(todo)})
                    status, todo = "oom_stop", []
                elif st == "timeout" or not server.alive():
                    restart = True      # a hung or dead server: a fresh one for the next item
        finally:
            how = server.stop()
            after, settle_s = R1.settle(baseline)
            sampler.stop()
            journal.add({"kind": "session_end", "arm": arm, "session": session, "wall_s": round(R1.now() - t_start, 1),
                         "stop": how, "mem_after_mib": after, "settle_s": settle_s, "peak_mib": sampler.max_mem,
                         "peak_rss_mib": sampler.max_rss, "max_temp_c": sampler.max_temp,
                         "max_power_w": sampler.max_power, "samples": sampler.n})
        if outcome == "thermal":
            wait_cool(journal, arm)
    if status == "done":
        journal.add({"kind": "arm_end", "arm": arm, "status": "done"})
    return status


def run_probe(args, journal, probe):
    """Stage 3's reach probe: stage 1's step function, unchanged, at each length until the first failure."""
    passed = set()
    for length in probe["lengths"]:
        step = S.step_id(probe["format"], length)
        prior = next((r for r in journal.results() if r.get("step") == step), None)
        if prior is None:
            with open(os.path.join(args.items, probe["prompts"], f"ladder-{length}.json"), encoding="utf-8") as f:
                prompt = json.load(f)
            prior = R1.run_step(args, journal, probe["format"], length, prompt, True)
        if prior["verdict"] == "pass":
            passed.add(length)
        elif prior["verdict"] in ("fail", "harness_error", "thermal_stop"):
            break
    args.probe_passed.setdefault(probe["format"], set()).update(passed)
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8098)
    args = ap.parse_args()
    with open(args.plan) as f:
        plan = json.load(f)
    for d in ("configs", "logs", "samples"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    args.tabby_logs = f"{R1.LAB}/logs/tabby-p8lc-{plan['group']}"
    os.makedirs(args.tabby_logs, exist_ok=True)
    args.probe_passed = {}
    journal = Journal(os.path.join(args.out, "journal.jsonl"))
    if os.path.exists(os.path.join(R1.TABBY, "config.yml")):
        print("stage2_run: a stray tabbyAPI config.yml would be merged into every config", flush=True)
        return 3
    limit = R1.smi("power.limit")[0]
    if abs(float(limit) - 260.0) > 0.5:
        print(f"stage2_run: power limit reads {limit} W, not 260 W; stopping", flush=True)
        return 4
    journal.add({"kind": "plan_start", "plan": plan["name"], "group": plan["group"], "pid": os.getpid(),
                 "power_limit_w": limit, "used_before_s": round(group_used_s(journal), 1),
                 "alloc_vars_present": [v for v in R1.ALLOC_VARS if v in os.environ]})
    sessions = [sum(1 for r in journal.rows if r.get("kind") == "session_start")]
    try:
        if plan.get("probe"):
            passed = run_probe(args, journal, plan["probe"])
            journal.add({"kind": "probe_done", "format": plan["probe"]["format"], "passed": sorted(passed)})
        cut = False   # cut_from_end: once an arm is skipped by the budget, every later arm is skipped too
        for spec in plan["arms"]:
            if spec["length"] == "longest_probe":
                ok = sorted(args.probe_passed.get(spec["format"], ()))
                if not ok:
                    journal.add({"kind": "arm_skipped", "arm": spec["arm"], "why": "no probe length passed"})
                    continue
                spec = dict(spec, length=ok[-1], arm=f"{spec['arm']}-{ok[-1] // S.K}k",
                            items=spec["items_by_length"][str(ok[-1])])
            prior = journal.arm_closed(spec["arm"])
            if prior is not None:
                if prior.get("status") == "harness_error":
                    print("stage2_run: the journal holds a harness error; fix it and amend first", flush=True)
                    return 3
                if prior.get("status") == "budget_stop":
                    break
                if plan.get("cut_from_end") and prior.get("kind") == "arm_skipped" and \
                        str(prior.get("why", "")).startswith("budget"):
                    cut = True
                continue
            if cut:
                journal.add({"kind": "arm_skipped", "arm": spec["arm"], "why": "budget: an earlier arm was cut"})
                continue
            st = run_arm(args, journal, spec, plan["group"], sessions)
            if st == "harness_error":
                journal.add({"kind": "plan_error", "plan": plan["name"], "reason": f"harness error in {spec['arm']}"})
                return 3
            if st == "budget_stop":
                break
            if st == "skipped" and plan.get("cut_from_end"):
                cut = True
        journal.add({"kind": "plan_end", "plan": plan["name"], "used_s": round(group_used_s(journal), 1)})
        return 0
    except R1.HarnessError as e:
        journal.add({"kind": "plan_error", "plan": plan["name"], "reason": str(e)})
        return 3


if __name__ == "__main__":
    sys.exit(main())
