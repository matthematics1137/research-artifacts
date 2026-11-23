#!/usr/bin/env python3
"""P8 release refresh. Stdlib only; --dry-run is pure CPU and launches nothing.

Reuses the November 4 confirmation request/timing/census code and fit utilities,
without modifying those registrations. A single exclusive output directory and
deadline cover R1, R2 undrafted, fit probes and conditional drafted comparisons.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "confirm"))
import run_confirm as RC
import run_confirm_fit as FIT


class BudgetExceeded(BaseException):
    """Escapes legacy per-condition exception handling; finalizers still run."""


class CampaignStopped(BaseException):
    """A guard/user termination must never become a local phase timeout."""


def telemetry():
    fields = ["memory.used", "memory.total", "temperature.gpu", "power.draw", "power.limit", "clocks.sm",
              "clocks_throttle_reasons.active", "clocks.mem", "driver_version", "name"]
    values = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                      "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    if "\n" in values:
        raise RuntimeError("expected one visible GPU")
    return dict(zip(["mem_used_mib", "mem_total_mib", "temp_c", "power_w", "power_limit_w", "sm_mhz",
                     "throttle", "mem_mhz", "driver", "name"], values.split(", ")))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key: " + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def check_freeze(here, expected):
    here = Path(here)
    if digest(here / "freeze.json") != expected:
        raise ValueError("freeze.json digest differs from the queued digest")
    frozen = read_json(here / "freeze.json")
    for name, sha in frozen["files"].items():
        path = (here / name).resolve()
        if not path.is_relative_to(here.parent):
            raise ValueError("freeze path outside tools tree: " + name)
        if digest(path) != sha:
            raise ValueError("frozen file changed: " + name)
    return frozen


def load(here=HERE, lab=RC.L):
    p = read_json(Path(here) / "plan.json")
    if (p["schema"] != "p8-refresh-v1" or p["cap_seconds"] != 9000 or
            p["rounds"] != 3 or p["fill_long"] != 28000 or p["max_new"] != 256):
        raise ValueError("unexpected campaign dimensions or budget")
    if p["sampler"] != {"temperature": 0.8, "min_p": 0.08, "top_k": 0, "top_p": 1.0}:
        raise ValueError("sampler differs from P8")
    p["models"] = RC.subst(p["models"], {"L": lab})
    replacements = dict(L=lab, **p["models"])
    for key in ("engines", "configs"):
        p[key] = RC.subst(p[key], replacements)
    phases = [read_json(Path(here) / f) for f in p["phase_files"]]
    p["conditions"] = [c for phase in phases for c in phase.get("conditions", [])]
    ids = [c["id"] for c in p["conditions"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate condition")
    for c in p["conditions"]:
        if c["engine"] not in p["engines"] or c["config"] not in p["configs"] or c["fill"] not in (0, 28000):
            raise ValueError("invalid condition: " + c["id"])
    return p, phases


def verify_files(records, lab):
    checked = []
    for r in records:
        path = Path(RC.subst(r["path"], {"L": lab}))
        if not path.is_file() or ("bytes" in r and path.stat().st_size != r["bytes"]):
            raise ValueError("missing/wrong-sized artifact: " + r["key"])
        if digest(path) != r["sha256"]:
            raise ValueError("artifact digest mismatch: " + r["key"])
        checked.append(dict(key=r["key"], path=str(path), sha256=r["sha256"]))
    return checked


def verify_overlay(spec, lab):
    root = Path(RC.subst(spec["root"], {"L": lab}))
    actual = {str(f.relative_to(root)): digest(f) for f in sorted(root.rglob("*"))
              if f.is_file() and "__pycache__" not in f.parts and f.suffix in spec["suffixes"]}
    if not actual or actual != spec["files"]:
        raise ValueError("w13 overlay differs from CPU preparation snapshot")
    return len(actual)


def verify_builds(plan, lab):
    from build_cpu import commands
    recipes = commands(plan, lab)
    receipt = read_json(Path(lab) / "builds/p8-refresh-76b5/builds.json")
    for key in ("ninferno", "llama"):
        expected, got = plan["engines"][key], receipt["engines"][key]
        if got["commit"] != expected["commit"] or got["arch"] != "75" or got["cuda_release"] != "13.4":
            raise ValueError("build provenance mismatch: " + key)
        if got["configure"] != recipes[key]["configure"] or got["compile"] != recipes[key]["compile"]:
            raise ValueError("build recipe mismatch: " + key)
        if digest(Path(recipes[key]["build"]) / "CMakeCache.txt") != got["cmake_cache_sha256"]:
            raise ValueError("build cache changed: " + key)
        source = Path(expected["source"])
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
        if head != expected["commit"] or dirty:
            raise ValueError("source changed after CPU build: " + key)
        for binary in got["binaries"]:
            path = Path(RC.subst(binary["path"], {"L": lab}))
            if digest(path) != binary["sha256"]:
                raise ValueError("binary changed after CPU build: " + str(path))
        if expected["bin"] not in [RC.subst(b["path"], {"L": lab}) for b in got["binaries"]]:
            raise ValueError("served binary absent from build receipt: " + key)
    return receipt


def preflight(plan, lab, here=HERE):
    # Deliberately does not import torch, call nvidia-smi, start a server or convert a model.
    artifacts = read_json(Path(here) / "artifacts.json")
    return dict(artifacts=verify_files(artifacts["files"], lab),
                w13_files=verify_overlay(read_json(Path(here) / "w13-files.json"), lab),
                builds=verify_builds(plan, lab))


def output_path(lab, out):
    path = Path(out).resolve()
    if not path.is_relative_to((Path(lab) / "runs").resolve()):
        raise ValueError("run output must stay under the lab's runs directory")
    if path.exists():
        raise ValueError("attempt already exists; automatic GPU retry is forbidden")
    return path


def rotations(conditions, rounds=3):
    for rnd in range(1, rounds + 1):
        n = len(conditions)
        yield rnd, conditions[(rnd - 1) % n:] + conditions[:(rnd - 1) % n] if n else []


def residency(text, min_models=2):
    import re
    matches = [(int(a), int(b)) for a, b in re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)]
    # Both the target and separate drafter need evidence. One absent log is unknown.
    return {"layers": matches, "verified_full": len(matches) >= min_models and all(a == b and b > 0 for a, b in matches)}


class Refresh(RC.Run):
    def __init__(self, plan, out, lab, started):
        super().__init__(plan, str(out), False)
        self.lab = lab
        self.started = started
        self.end = started + plan["cap_seconds"] - plan["cleanup_reserve_seconds"]
        self.phase_end = self.end
        self.next_port = plan["port_range"][0]
        self.rungs = {}
        self.ctx = None
        self.phase = ""
        self.fit_records = {}

    def census_cmd(self, c, tag):
        cmd, env = super().census_cmd(c, tag)
        e = self.plan["engines"][c["engine"]]
        env = {k: v for k, v in env.items() if not k.startswith("EXL3_")}
        env.update(e["env_mtp"] if self.plan["configs"][c["config"]]["drafted"] else e["env"])
        env["HF_HUB_OFFLINE"] = "1"
        return cmd, env

    def arm_timer(self, seconds=None):
        left = min(self.end, self.phase_end) - time.monotonic()
        if seconds is not None:
            left = min(left, seconds)
        if left <= 0:
            raise BudgetExceeded("combined/phase deadline")
        signal.setitimer(signal.ITIMER_REAL, max(0.001, left))

    def wait_memory(self, budget_s=30):
        until = min(time.monotonic() + budget_s, self.started + self.plan["cap_seconds"] - 5)
        while time.monotonic() < until:
            g = RC.gpu_state()
            if not g:
                raise RuntimeError("GPU telemetry unavailable")
            if int(g["mem_used_mib"]) <= self.baseline_mib + 256:
                return
            time.sleep(2)
        raise RuntimeError("memory did not return to baseline; refusing another start")

    def stop_server(self, proc, logf):
        # Short bounded cleanup fits the reserved tail. No existing process is targeted.
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        logf.close()
        self.wait_memory()

    def server_cmd(self, c, extra=()):
        e = self.plan["engines"][c["engine"]]
        lo, hi = self.plan["port_range"]
        port = FIT.choose_port(lo, hi, self.next_port, min(30, max(0, self.phase_end - time.monotonic())))
        if port is None:
            raise RuntimeError("no free port")
        self.next_port = lo + (port - lo + 1) % (hi - lo + 1)
        self.base = f"http://localhost:{port}"
        args = RC.subst(self.plan["configs"][c["config"]]["args"] + e["common"] + list(extra), {"port": port})
        if c["engine"] == "ninferno" and c["config"] == "nif-df2":
            i = args.index("--max-context")
            args[i:i + 2] = ["--max-context", "29696"] if c["fill"] else []
        if c["config"] in self.rungs:
            args = FIT.rung_args(args, self.rungs[c["config"]], "llama", self.ctx, ["llama"])
        env = {k: v for k, v in os.environ.items() if not k.startswith(("NINFERNO_", "LLAMA_"))}
        env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   XDG_CACHE_HOME=os.path.join(self.lab, "cache"),
                   XDG_CONFIG_HOME=os.path.join(self.lab, "cache/config"),
                   XDG_DATA_HOME=os.path.join(self.lab, "cache/data"))
        env.update(e["env"])
        return [e["bin"]] + args, env

    def count_context(self):
        tok = self.plan["engines"]["llama"]["tokenizer_bin"]
        longest = 0
        for task in self.plan["tasks"]:
            path = Path(self.out) / f"prompt-{task}.txt"
            path.write_text(RC.chat(self.filler + RC.PROMPTS[task]))
            r = subprocess.run([tok, "-m", self.plan["models"]["iq2xxs"], "-f", str(path),
                                "--ids", "--show-count", "--log-disable"], capture_output=True, text=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="/usr/local/cuda-13.4/lib64"),
                               check=True, timeout=90)
            import re
            count = re.search(r"Total number of tokens: (\d+)", r.stdout)
            if not count:
                raise RuntimeError("tokenizer count absent; no guessed R2 context")
            longest = max(longest, int(count[1]))
        self.ctx = FIT.context_for(longest)
        if self.ctx > 32768:
            raise RuntimeError("counted context exceeds registered maximum")
        self.log("context", longest_prompt_tokens=longest, context=self.ctx)

    def run_server_condition(self, c, tag):
        try:
            return super().run_server_condition(c, tag)
        finally:
            if c["engine"] == "llama":
                path = Path(self.out) / "logs" / (c["id"] + ".log")
                res = residency(path.read_text(errors="replace") if path.exists() else "",
                                2 if c.get("requires_fit") else 1)
                record = self.fit_records.setdefault(c["config"], {})
                record["full_offload"] = record.get("full_offload", True) and res["verified_full"]
                self.log("residency", condition=c["id"], **res)

    def fit(self, phase):
        for cfg in phase["configs"]:
            c = next(c for c in self.plan["conditions"] if c["config"] == cfg and c["fill"])
            record = {"rung": None, "status": "not-tested", "attempts": []}
            self.fit_records[cfg] = record
            for rung in phase["rungs"]:
                # A timeout or bad telemetry is not evidence of an out-of-memory boundary.
                self.arm_timer(phase["rung_timeout_seconds"])
                if rung in ("R2", "R3") and self.ctx is None:
                    self.count_context()
                self.rungs[cfg] = rung
                cmd, env = self.server_cmd(c)
                label = f"fit-{cfg}-{rung}"
                proc, logf = self.start_server(cmd, env, label)
                record["status"] = "in-progress"
                self.log("fit-start", config=cfg, rung=rung)
                ok, why = False, "not completed"
                try:
                    err = RC.wait_ready(self.base, proc, 90)
                    if err:
                        raise RuntimeError("server " + err)
                    for pre in ("", self.filler):
                        RC.stream_answer(self.base, "completion", RC.request_body(
                            "completion", pre + RC.PROMPTS["code"], 32, self.plan["sampler"]), timeout=150)
                    ok, why = True, "loaded and both warm-ups completed"
                except (BudgetExceeded, CampaignStopped) as exc:
                    record["status"] = "budget-or-guard-incomplete"
                    record["attempts"].append(dict(rung=rung, ok=None, why=str(exc)))
                    self.log("fit-interrupted", config=cfg, rung=rung, why=str(exc))
                    raise
                except Exception as exc:
                    why = f"{type(exc).__name__}: {exc}"[:500]
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    self.stop_server(proc, logf)
                text = (Path(self.out) / "logs" / (label + ".log")).read_text(errors="replace")
                res = residency(text)
                attempt = dict(rung=rung, ok=ok, why=why, **res)
                record["attempts"].append(attempt)
                self.log("fit", config=cfg, **attempt)
                if ok:
                    record.update(rung=rung, status="measured-fit", full_offload=res["verified_full"],
                                  partial_or_unknown=rung == "R3" and not res["verified_full"], context=self.ctx)
                    break
            if record["rung"] is None:
                self.rungs.pop(cfg, None)
                record["status"] = "failed-all-attempted-rungs"
                for c in self.plan["conditions"]:
                    if c["config"] == cfg:
                        self.dropped[c["id"]] = "no successful measured fit; see fit attempts (not necessarily OOM)"
            self.arm_timer()

    def phase_run(self, phase):
        self.phase = phase["name"]
        self.phase_end = min(self.end, time.monotonic() + phase["max_seconds"])
        self.log("phase", phase=self.phase, max_seconds=phase["max_seconds"])
        try:
            if phase["kind"] == "fit":
                self.fit(phase)
                return
            if self.phase == "R2-drafted" and not any(r.get("rung") for r in self.fit_records.values()):
                self.log("phase-skip", phase=self.phase, reason="no completed drafted fit")
                return
            for rnd, ordered in rotations(phase["conditions"], self.plan["rounds"]):
                self.log("round", phase=self.phase, round=rnd, conditions=[c["id"] for c in ordered])
                for c in ordered:
                    if c["id"] in self.dropped:
                        continue
                    if c.get("requires_fit") and self.fit_records.get(c["config"], {}).get("rung") is None:
                        self.dropped[c["id"]] = "drafted condition not timed: no completed fit"
                        continue
                    self.arm_timer(min(self.plan["condition_timeout_s"], phase["max_seconds"]))
                    before = len(Path(self.rows).read_text().splitlines()) if Path(self.rows).exists() else 0
                    self.run_condition(c, rnd)  # unchanged client/census and tag convention
                    self.log("condition-observation", condition=c["id"], phase=self.phase,
                             rows_before=before, fit=self.fit_records.get(c["config"]))
        except BudgetExceeded:
            self.log("phase-budget", phase=self.phase)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            (Path(self.out) / "rungs.json").write_text(json.dumps(self.fit_records, indent=2) + "\n")


def render(plan, phases):
    # Print a structured plan without constructing Run (which creates output paths).
    for phase in phases:
        print(json.dumps(dict(phase=phase["name"], max_seconds=phase["max_seconds"],
                              conditions=phase.get("conditions", []), fit=phase.get("configs", []))))
    print(json.dumps(dict(engines=plan["engines"], configs=plan["configs"],
                          combined_cap_seconds=plan["cap_seconds"], automatic_retry="refused")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze-sha", required=True)
    ap.add_argument("--lab", default=RC.L)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    check_freeze(HERE, args.freeze_sha)
    RC.L = FIT.L = args.lab
    plan, phases = load(HERE, args.lab)
    if args.dry_run:
        render(plan, phases)
        return 0
    if args.run and not args.out:
        ap.error("--run requires --out")
    out = output_path(args.lab, args.out) if args.run else None
    checked = preflight(plan, args.lab)
    if args.verify_only:
        print(json.dumps(checked, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)  # queue retries cannot create another GPU attempt
    (out / "preflight.json").write_text(json.dumps(checked, indent=2) + "\n")
    (out / "attempt.json").write_text(json.dumps(dict(start=RC.now(), cap_seconds=9000,
                                                       freeze_sha=args.freeze_sha)) + "\n")
    started = time.monotonic()
    run = Refresh(plan, out, args.lab, started)
    RC.gpu_state = telemetry  # only --run reaches this point
    status = "finished"
    exit_code = 0
    def stop(signum, frame):
        if signum == signal.SIGTERM:
            raise CampaignStopped("terminated by guard/operator")
        raise BudgetExceeded("deadline")
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        g = RC.gpu_state()
        if (not g or int(g["mem_total_mib"]) != 11264 or float(g["power_limit_w"]) != 260 or
                "RTX 2080 Ti" not in g["name"] or g["driver"] != "580.178.04"):
            raise RuntimeError("expected stock 11264 MiB card at 260 W with telemetry")
        run.baseline_mib = int(g["mem_used_mib"])
        run.log("session", gpu=g, preflight="passed", freeze_sha=args.freeze_sha)
        run.arm_timer()
        run.make_filler()
        for phase in phases:
            if time.monotonic() >= run.end:
                status = "budget-exhausted"
                break
            run.phase_run(phase)
    except (BudgetExceeded, CampaignStopped) as exc:
        status = str(exc)
    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        exit_code = 2
        run.log("fatal", error=status)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        run.log("done", status=status, elapsed_seconds=time.monotonic() - started, dropped=run.dropped)
        (out / "rungs.json").write_text(json.dumps(run.fit_records, indent=2) + "\n")
        from analyze_refresh import analyze
        result = analyze(plan, phases, out)
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        # Typed speed points for scoreboard integration; correctness/energy were not measured.
        if Path(run.rows).exists():
            with (out / "landscape-points.jsonl").open("x") as points:
                for line in Path(run.rows).read_text().splitlines():
                    row = json.loads(line)
                    points.write(json.dumps(dict(schema="p8-refresh-landscape-speed-v1", row=row,
                                                 artifact_manifest_sha256=digest(HERE / "artifacts.json"),
                                                 freeze_sha256=args.freeze_sha, correctness=None,
                                                 joules=None)) + "\n")
    return exit_code  # a cut is terminal; adoption depends on summary completeness


if __name__ == "__main__":
    sys.exit(main())
