#!/usr/bin/env python3
"""Guarded P5 primary generation/judge stages; dry-run unless --execute is given.

Check live candidate-machine headroom before a real launch. Never run beside
the active preparation units or another study endpoint. This adapter reuses
the frozen request runner in a separate process; it does not edit its sources,
change requests, replace saved outcomes, or publish anything.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from evidence import ROOT, rows
from diagnostics import require, verdicts
from draft_support import read_json, sha
from followup_results import source_modules
from migrate_followup_preparation import once, safe_path
import execution_snapshot as E
import primary_audit as P
import primary_requests as Q
from primary_receipts import BASELINE_REL, BASELINE_SHA, SOURCE_FILES, Bindings, baseline
from primary_server import ObservedServer, expected_machine, judge_server, min_free_vram_mib, preflight

STAGES = ("generate", "judge-gptoss", "judge-self")


def journal_path(stage, root=ROOT):
    return root / "testsuite/paper5/work/journals" / ("grid.jsonl" if stage == "generate" else f"judge_{stage[6:]}__grid.jsonl")


def records(path):
    return list(rows(path)) if Path(path).exists() else []


def generation_jobs(inputs, a):
    return {p["job_id"]: a.grid.to_job(p, inputs["cap"]) for p in inputs["prompts"]}


def requests(stage, inputs, a, root=ROOT):
    jobs = generation_jobs(inputs, a)
    gens = None
    if stage != "generate":
        binder = Bindings("generate", inputs["bindings"], root=root)
        state = Q.journal_state(records(journal_path("generate", root)), jobs, stage="generate",
                                harness={name: inputs["manifest"]["harness"][name] for name in P.RUNNER_FILES},
                                count_tokens=a.C.count_tokens, legacy_ids=inputs["smoke_ids"], check_binding=binder)
        gens = state["done"]
        require(not state["pending"] and gens.keys() == jobs.keys(), "complete primary generations are required before grading")
        binder.finished(journal_path("generate", root), terminal_outcomes=len(gens))
        binder.stable()
        inputs["bindings"][str(journal_path("generate", root))] = sha(journal_path("generate", root))
        judge = stage[6:]
        jobs = {}
        # Preserve actual generation terminal order, including the smoke prefix.
        for gen in gens.values():
            item = inputs["items"][gen["meta"]["question_id"]]
            for retry in ((False, True) if judge == "gptoss" else (False,)):
                job = Q.judge_job(a, item, gen, judge, retry)
                jobs[job["job_id"]] = job
    return jobs, gens


def runner_type(a, session_reference, jobs, state):
    class GuardedRunner(a.runner.Runner):
        def __init__(self, server, path, *, idle_seconds):
            super().__init__(server, str(path), idle_seconds=idle_seconds)
            # The frozen base runner does not know about postcheck resolutions; drop exactly the
            # resolved jobs (excluded, to be requested again) from its cache before comparing.
            for jid in state.get("excluded", []):
                self.done.pop(jid, None)
            require(self.done == state["done"], "journal changed before runner initialization")
            self.expected_size = Path(path).stat().st_size if Path(path).exists() else 0
            self.active_job = None
            self.offset = 0
            self.last_server = None
            self.endpoint = "/v1/completions"
            self.recorded_http_attempts = 0

        def gate(self):
            super().gate()
            self.server.verify_live()
            self.last_server = self.server.reference()

        def _append(self, row):
            path = Path(self.journal)
            actual = path.stat().st_size if path.exists() else 0
            require(actual == self.expected_size, "journal was concurrently appended or replaced")
            require(self.active_job is not None, "no active request intent")
            if "attempt" in row:
                row["attempt"] += self.offset
                self.recorded_http_attempts += 1
            row["execution"] = {"session": session_reference, "server": self.last_server,
                                "intent_sha256": Q.intent(self.active_job, self.endpoint), "postcheck_ok": None}
            failed_check = None
            if row.get("ok") is True:
                try:
                    self.server.verify_live()
                    row["execution"]["postcheck_ok"] = True
                except BaseException as exc:
                    # Preserve the received answer even if post-request evidence
                    # is invalid. It must be investigated, never silently reused.
                    row["execution"]["postcheck_ok"] = False
                    failed_check = exc
            super()._append(row)
            self.expected_size = path.stat().st_size
            if failed_check is not None:
                raise failed_check

        def run(self, job, endpoint="/v1/completions", retries=2):
            jid = job["job_id"]
            require(jid in jobs and Q.canonical(job) == Q.canonical(jobs[jid]), "request differs from verified plan")
            if jid in self.done:
                return self.done[jid]  # validated before construction; no server start or wait
            require(retries == 2, "primary retry allowance changed")
            prior = state["pending"].get(jid, [])
            self.active_job, self.offset, self.endpoint, self.last_server = job, len(prior), endpoint, None
            try:
                if len(prior) == 3:
                    row = {"job_id": jid, "terminal": True, "ok": False, "error": prior[-1]["error"],
                           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_prompt": job["n_prompt"],
                           "budget": job["budget"], "meta": job["meta"], "harness": self.hashes,
                           "request_idle_seconds": self.request_idle_seconds, "recovered_exhausted_failure": True}
                    self._append(row)
                    self.done[jid] = row
                    return row
                return super().run(job, endpoint, retries=2-len(prior))
            finally:
                self.active_job = None
    return GuardedRunner


@contextmanager
def writer_lock(root):
    # An ordinary advisory file lock protects this journal workflow; it is
    # not a compute reservation, scheduler, daemon or cross-machine lease.
    path = safe_path(root, "testsuite/paper5/work/execution/primary/.writer.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another guarded primary writer is active") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute(stage, *, live=False, idle_seconds=10., root=ROOT):
    require(stage in STAGES and math.isfinite(idle_seconds) and idle_seconds >= 0, "invalid stage or pacing")
    root = Path(root).resolve()
    saved = baseline(root)
    with source_modules():
        from legacy import algorithms
        with algorithms(SimpleNamespace(root=root, base=root / "testsuite/paper5/work", leg="locomo")) as a:
            inputs = Q.frozen_inputs(root, a)
            jobs, generations = requests(stage, inputs, a, root)
            path = journal_path(stage, root)
            raw_records = records(path)
            binder = Bindings(stage, inputs["bindings"], root=root, baseline_record=saved)
            state = Q.journal_state(raw_records, jobs, stage=stage,
                                    harness={name: inputs["manifest"]["harness"][name] for name in P.RUNNER_FILES},
                                    count_tokens=a.C.count_tokens, legacy_ids=inputs["smoke_ids"] if stage == "generate" else (),
                                    check_binding=binder)
            # Reject a saved gptoss retry that cannot follow its saved base.
            if stage == "judge-gptoss":
                for jid in set(state["done"]) | set(state["pending"]):
                    if jid.endswith("|retry"):
                        first = state["done"].get(jid[:-6])
                        require(first is not None and first["ok"] and not (first.get("text") or "").strip(), "cached retry is not warranted")
            binder.stable()
            plan = {"stage": stage, "base_requests": len(generations) if generations is not None else len(jobs),
                    "cached_terminal": len(state["done"]), "interrupted_requests": len(state["pending"]),
                    "legacy_smoke_outcomes": len(state["legacy_terminal_ids"]), "idle_seconds": idle_seconds,
                    "inference_started": False, "generation_arm_order": inputs["arm_order"]}
            if not live:
                return plan
            require(Path(sys.prefix).resolve() == root / "engines/exl3-env", "execute from the pinned exl3-env Python")
            require(not os.environ.get("LD_PRELOAD"), "an unrecorded LD_PRELOAD override is present")
            with writer_lock(root):
                preflight(saved, root)
                E.check(root / BASELINE_REL, root)  # full pre-stage artifact/prefix verification
                require(records(path) == raw_records, "journal changed during launch preflight")
                require(all(sha(p) == v for p, v in inputs["bindings"].items()), "frozen input changed before launch")
                session = safe_path(root, "testsuite/paper5/work/execution/primary/" + stage + "-" + uuid.uuid4().hex)
                source_paths = [root / name for name in SOURCE_FILES]
                sources = {str(p.relative_to(root)): p.read_bytes() for p in source_paths}
                raw_server = judge_server(a) if stage == "judge-gptoss" else a.runner.TabbyServer()
                command, _, cwd = raw_server.cmd_env_cwd()
                prefix = E.prefix_record(path) if path.exists() else {"path": str(path), "bytes": 0, "sha256": E.hashlib.sha256(b"").hexdigest()}
                record = {"schema": "p5-primary-session-v1", "stage": stage, "created_at": E.utc(),
                          "baseline_sha256": BASELINE_SHA, "inputs": inputs["bindings"], "journal_before": prefix,
                          "idle_seconds": idle_seconds, "legacy_smoke_outcomes": state["legacy_terminal_ids"],
                          "min_free_vram_mib": min_free_vram_mib(), "machine_expected": expected_machine(saved, root),
                          "generation_arm_order": inputs["arm_order"], "server_argv": command[3:], "server_cwd": cwd,
                          "source_sha256": {p: E.hashlib.sha256(data).hexdigest() for p, data in sources.items()}}
                for name, data in sources.items():
                    once(session / "sources" / name, data)
                once(session / "SESSION.json", Q.canonical(record) + "\n")
                ref = {"path": str((session / "SESSION.json").relative_to(root)), "sha256": sha(session / "SESSION.json")}
                # Per-request stat guards cover frozen inputs and launcher code
                # as well as the baseline artifacts. Full hashes bookend the run.
                extra = {p: {**E.identity(p), "sha256": v} for p, v in inputs["bindings"].items()}
                extra.update({str(p): {**E.identity(p), "sha256": sha(p)} for p in source_paths})
                guarded_baseline = {**saved, "files": {**saved["files"], **extra}}
                observed = ObservedServer(raw_server, session, ref["sha256"], guarded_baseline, root=root, inherited_placement=binder.placement)
                runner = runner_type(a, ref, jobs, state)(observed, path, idle_seconds=idle_seconds)
                completed = False
                try:
                    for i, (jid, job) in enumerate(jobs.items()):
                        if jid.endswith("|retry"):
                            first = runner.done[jid[:-6]]
                            if not first["ok"] or (first.get("text") or "").strip():
                                continue
                        endpoint = "/v1/chat/completions" if stage == "judge-gptoss" else "/v1/completions"
                        runner.run(job, endpoint)
                        if i % 25 == 0:
                            print(f"{stage}: {len(runner.done)} terminal outcomes retained", flush=True)
                    completed = True
                finally:
                    observed.stop()
                require(completed, "stage did not finish")
                E.check(root / BASELINE_REL, root)  # full post-stage artifact/prefix verification
                require(all(sha(p) == v for p, v in inputs["bindings"].items()), "frozen inputs changed during execution")
                require(all(p.read_bytes() == sources[str(p.relative_to(root))] for p in source_paths), "launcher changed during execution")
                require(all(sha(p) == r["sha256"] for p, r in observed.runtime_files.items()), "supplemental runtime changed during stage")
                require(E.prefix_record(path, prefix["bytes"])["sha256"] == prefix["sha256"], "prior journal prefix changed")
                final_binder = Bindings(stage, inputs["bindings"], root=root, baseline_record=saved)
                final = Q.journal_state(records(path), jobs, stage=stage, harness=runner.hashes, count_tokens=a.C.count_tokens,
                                        legacy_ids=inputs["smoke_ids"] if stage == "generate" else (), check_binding=final_binder)
                require(not final["pending"], "unfinished request remains")
                if stage == "generate":
                    require(final["done"].keys() == jobs.keys(), "generation grid incomplete")
                else:
                    verdicts(generations, final["done"], stage[6:])
                    require({g + "|judge=" + stage[6:] for g in generations} <= final["done"].keys(), "judge pass incomplete")
                final_binder.stable()
                finished = {"schema": "p5-primary-stage-finished-v1", "stage": stage, "session": ref,
                            "finished_at": E.utc(), "complete": True, "journal_sha256": sha(path),
                            "terminal_outcomes": len(final["done"]), "baseline_before_after_verified": BASELINE_SHA,
                            "recorded_http_attempts": runner.recorded_http_attempts,
                            "inference_started": runner.recorded_http_attempts > 0, "source_hashes_unchanged": True}
                once(session / "FINISHED.json", Q.canonical(finished) + "\n")
                return finished


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=10.)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.stage, live=args.execute, idle_seconds=args.idle_seconds), indent=2))
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"P5 primary stage stopped: {exc}\n")


if __name__ == "__main__":
    main()
