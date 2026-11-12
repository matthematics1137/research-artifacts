#!/usr/bin/env python3
"""Paper 8 confirmation run, follow-up step x-p8-confirm-fit (Amendment 1 of
research/prereg/2026-11-04-p8-confirmation-speed.md).

One session, three rotated rounds, of the conditions in plan-fit.json: the seven competitor lanes the registered
session dropped before decoding, and the w13 anchors on 1.6.5, so every comparison stays inside one session. It reuses
run_confirm.py and plan.json unchanged (they are checked against their registered SHA-256 and imported, never
edited) and adds two things:
- **ports:** every server start takes a port from plan-fit's range that can be bound now, waiting up to 120 s when
  none can (the registered driver alternated two ports, so a server could find its port still held);
- **the fit ladder:** before round 1, each server configuration is started at R0 (as registered), then R1 (-ub 128),
  R2 (the smallest -c that holds the longest filled 28K prompt plus 256 + 1024 tokens) and R3 (no -ngl 99, the
  engine's own fit, for engines that have one), cumulatively and in that order. The first rung that loads and completes
  the warm-up is used in every round; a configuration that passes no rung is dropped as "did not fit".

  python3 run_confirm_fit.py --plan plan.json --fit plan-fit.json --out DIR [--dry-run]

Rows, journal and logs go to DIR as in run_confirm.py; DIR/rungs.json records each configuration's rung, its
arguments and, at R3, the layers the engine left on the CPU. Stdlib only. Nothing is deleted.
"""
import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_confirm as RC  # noqa: E402

L = RC.L
NEW_TOKENS, MARGIN = 256, 1024


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frozen_ok(fit, here=HERE):
    """The registered files are byte-identical to the registration (a guard retry of the registered session loads them)."""
    return {name: sha256(os.path.join(here, name)) == digest for name, digest in fit["frozen_sha256"].items()}


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("localhost", port))   # no SO_REUSEADDR: a port in TIME_WAIT or still held counts as busy
        return True
    except OSError:
        return False
    finally:
        s.close()


def choose_port(lo, hi, start, wait_s, poll_s=2.0):
    """The first bindable port of [lo, hi], scanning from `start`; waits up to wait_s when none is free."""
    n = hi - lo + 1
    t_end = time.time() + wait_s
    while True:
        for i in range(n):
            p = lo + (start - lo + i) % n
            if port_free(p):
                return p
        if time.time() >= t_end:
            return None
        time.sleep(poll_s)


def context_for(prompt_tokens):
    """R2's context: the smallest multiple of 1024 holding the longest prompt, the 256 new tokens and a 1K margin."""
    return int(math.ceil((prompt_tokens + NEW_TOKENS + MARGIN) / 1024) * 1024)


def set_arg(args, flag, value):
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def drop_arg(args, flag):
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    return out


def rung_args(args, rung, engine, ctx, auto_fit_engines):
    """The server arguments at a rung (cumulative), or None when the rung does not apply to this engine."""
    if rung == "R0":
        return list(args)
    out = set_arg(args, "-ub", "128")
    if rung == "R1":
        return out
    if ctx is None:
        return None
    out = set_arg(out, "-c", str(ctx))
    if rung == "R2":
        return out
    if rung == "R3":
        return drop_arg(out, "-ngl") if engine in auto_fit_engines else None
    raise ValueError(rung)


def offloaded(log_text):
    """llama.cpp's "offloaded X/Y layers to GPU" from a server log: (on GPU, total), the last one printed."""
    m = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log_text)
    return (int(m[-1][0]), int(m[-1][1])) if m else None


class FitRun(RC.Run):
    def __init__(self, plan, fit, out, dry):
        super().__init__(plan, out, dry)
        self.fit = fit
        self.auto_fit = fit["ladder"]["R3"]["engines_with_auto_fit"]
        self.rungs = {}            # config -> {"rung", "engine", "gpu_layers", "partial_offload"}
        self.next_port = fit["port_range"][0]
        self.ctx = None
        self.last_offload = None
        self.full_conditions = plan["conditions"]
        ids = {c["id"]: c["priority"] for c in fit["conditions"]}   # the fit session's conditions and priorities
        self.plan = dict(plan, rounds=fit["rounds"], cap_hours=fit["cap_hours"],
                         conditions=[dict(c, priority=ids[c["id"]]) for c in plan["conditions"] if c["id"] in ids])

    # -- ports: a bindable port for every start (the registered driver's two alternating ports could still be held)
    def server_cmd(self, c, extra=()):
        e = self.plan["engines"][c["engine"]]
        lo, hi = self.fit["port_range"]
        port = self.next_port if self.dry else choose_port(lo, hi, self.next_port, self.fit["port_wait_s"])
        if port is None:
            raise RuntimeError("no bindable port in the range after the wait")
        self.next_port = lo + (port - lo + 1) % (hi - lo + 1)
        self.base = f"http://localhost:{port}"
        args = RC.subst(self.plan["configs"][c["config"]]["args"] + e["common"] + list(extra), {"port": port})
        r = self.rungs.get(c["config"])
        if r:
            args = rung_args(args, r["rung"], r["engine"], self.ctx, self.auto_fit)
        env = dict(os.environ, XDG_CACHE_HOME=os.path.join(L, "cache"))
        env.update(e["env"])
        if "HOME" in e["env"]:
            os.makedirs(e["env"]["HOME"], exist_ok=True)
        return [e["bin"]] + args, env

    # -- R2's context, counted by the GGUF tokenizer (vocabulary only, CPU) on the longest filled prompt
    def count_context(self):
        bindir = os.path.join(L, "src/llama.cpp-v0.9.0/build-sm75-cu134")
        tok = os.path.join(bindir, "bin/llama-tokenize")
        if not os.access(tok, os.X_OK):
            subprocess.run(["nice", "-n", "19", "cmake", "--build", bindir, "-j", "4", "--target", "llama-tokenize"],
                           check=True, timeout=1800, capture_output=True)
        longest = 0
        for task in self.plan["tasks"]:
            path = os.path.join(self.out, f"prompt-{task}.txt")
            with open(path, "w") as f:
                f.write(RC.chat(self.filler + RC.PROMPTS[task]))
            r = subprocess.run([tok, "-m", self.plan["models"]["iq2xxs"], "-f", path, "--ids", "--show-count",
                                "--log-disable"], capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                        LD_LIBRARY_PATH="/usr/local/cuda-13.4/lib64"))
            m = re.search(r"Total number of tokens: (\d+)", r.stdout)
            if not m:
                raise RuntimeError(f"llama-tokenize gave no count: {r.stderr[-300:]}")
            longest = max(longest, int(m.group(1)))
        self.ctx = context_for(longest)
        self.log("context", longest_prompt_tokens=longest, context=self.ctx)

    # -- the ladder: the first rung that loads and completes the warm-up, per server configuration
    def try_rung(self, c, rung):
        cfg, engine = c["config"], c["engine"]
        probe = RC.subst(self.plan["configs"][cfg]["args"] + self.plan["engines"][engine]["common"], {"port": 0})
        if rung_args(probe, rung, engine, self.ctx, self.auto_fit) is None:
            return None, "not applicable"
        self.rungs[cfg] = {"rung": rung, "engine": engine}
        cmd, env = self.server_cmd(c)
        e = self.plan["engines"][engine]
        if self.dry:
            print(cfg, rung, " ".join(cmd))
            return True, "dry"
        label = f"fit-{cfg}-{rung}"
        proc, logf = self.start_server(cmd, env, label)
        try:
            err = RC.wait_ready(self.base, proc, 600)
            if err:
                return False, f"server {err}"
            mid = RC.model_id(self.base) if e["api"] == "chat" else "model"
            RC.stream_answer(self.base, e["api"], RC.request_body(e["api"], RC.PROMPTS["code"], 32,
                                                                  self.plan["sampler"], mid))
            if any(x["config"] == cfg and x["fill"] for x in self.plan["conditions"]):
                RC.stream_answer(self.base, e["api"], RC.request_body(
                    e["api"], self.filler + RC.PROMPTS["code"], 32, self.plan["sampler"], mid),
                    timeout=self.plan["condition_timeout_s"])
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - a rung that fails is the next rung's cue
            return False, f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self.stop_server(proc, logf)
            with open(os.path.join(self.out, "logs", f"{label}.log")) as f:
                self.last_offload = offloaded(f.read())

    def fit_ladder(self):
        configs = {}
        for c in self.plan["conditions"]:
            if self.plan["engines"][c["engine"]]["kind"] == "server" and c["id"] not in self.dropped:
                configs.setdefault(c["config"], c)
        record = {}
        for cfg, c in configs.items():
            chosen = None
            for rung in self.fit["ladder"]["order"]:
                if rung in ("R2", "R3") and self.ctx is None and not self.dry:
                    try:
                        self.count_context()
                    except Exception as exc:  # noqa: BLE001 - without a count, R2 and R3 cannot be built
                        self.log("context-fail", reason=str(exc)[:300])
                        break
                self.last_offload = None
                ok, why = self.try_rung(c, rung)
                self.log("fit", config=cfg, rung=rung, ok=ok, why=why, offloaded=self.last_offload)
                if ok:
                    chosen = rung
                    break
            if chosen is None:
                self.rungs.pop(cfg, None)
                for x in self.plan["conditions"]:
                    if x["config"] == cfg:
                        self.dropped[x["id"]] = "did not fit on 11 GB at full offload (every rung failed)"
                        self.log("drop", condition=x["id"], reason=self.dropped[x["id"]])
                record[cfg] = {"rung": None}
            else:
                self.rungs[cfg] = {"rung": chosen, "engine": c["engine"]}
                record[cfg] = {"rung": chosen, "gpu_layers": self.last_offload, "partial_offload": chosen == "R3",
                               "context": self.ctx if chosen in ("R2", "R3") else None}
        if not self.dry:
            with open(os.path.join(self.out, "rungs.json"), "w") as f:
                json.dump(record, f, indent=1)

    def registered_quality_done(self):
        root = os.path.join(L, "runs/confirm")
        for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            j = os.path.join(root, d, "journal.jsonl")
            if not os.path.exists(j):
                continue
            for line in open(j):
                r = json.loads(line)
                if r.get("kind") == "quality" and r.get("returncode") == 0:
                    return True
        return False

    def conditional(self):
        """The registered conditional arms, under the registered rules (they look up the registered conditions)."""
        mine = self.plan["conditions"]
        self.plan["conditions"] = self.full_conditions
        try:
            super().conditional()
        finally:
            self.plan["conditions"] = mine

    def main(self):
        g = RC.gpu_state()
        self.baseline_mib = int(g.get("mem_used_mib", 400)) if g else 400
        self.log("session", plan="Amendment 1: x-p8-confirm-fit", frozen=frozen_ok(self.fit), gpu=g,
                 load=os.getloadavg(), baseline_mib=self.baseline_mib,
                 conditions=[c["id"] for c in self.plan["conditions"]])
        self.preflight()
        self.make_filler()
        self.fit_ladder()
        kept = [c["id"] for c in self.plan["conditions"] if c["id"] not in self.dropped]
        quality = not self.registered_quality_done()
        extra = (900 if quality else 0) + 300
        for rnd in range(1, self.plan["rounds"] + 1):
            if rnd > 1:
                kept = self.cut_to_cap(rnd, [k for k in kept if k not in self.dropped], extra)
            ordered = [c for c in self.plan["conditions"] if c["id"] in kept]
            n = len(ordered)
            ordered = [ordered[(i + rnd - 1) % n] for i in range(n)]
            self.log("round", round=rnd, conditions=[c["id"] for c in ordered])
            for c in ordered:
                if c["id"] not in self.dropped:
                    self.run_condition(c, rnd)
        if quality and "buun-mtp" in self.rungs:
            self.quality_check()
        else:
            self.log("quality-skip", reason="completed by the registered session" if not quality
                     else "buun did not fit or was cut")
        self.conditional()
        self.log("done", hours=round((time.time() - self.t0) / 3600, 2), dropped=self.dropped,
                 rungs={k: v["rung"] for k, v in self.rungs.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with open(args.fit) as f:
        fit = json.load(f)
    bad = [k for k, ok in frozen_ok(fit, os.path.dirname(os.path.abspath(args.plan))).items() if not ok]
    if bad:
        raise SystemExit(f"refusing: {', '.join(bad)} differ from the registration")
    plan = RC.load_plan(args.plan, L)
    run = FitRun(plan, fit, args.out, args.dry_run)
    if args.dry_run:
        run.baseline_mib = 0
        run.ctx = context_for(28050)
        seen = set()
        for c in run.plan["conditions"]:
            if run.plan["engines"][c["engine"]]["kind"] == "server" and c["config"] not in seen:
                seen.add(c["config"])
                for rung in fit["ladder"]["order"]:
                    run.try_rung(c, rung)
        return
    run.main()


if __name__ == "__main__":
    sys.exit(main())
