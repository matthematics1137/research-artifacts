#!/usr/bin/env python3
"""Explicit execution of one isolated P5 follow-up stage; dry-run by default.

Before --execute, the operator must check candidate-machine inventory/live
headroom. This runner additionally refuses active primary/P6 jobs, occupied
ports, a different GPU, unbound inputs, changed model weights and stale cached
requests. It creates no scheduler, allocation service or publication action.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time

from followups import verified_primary
from jobs import SUMMARIZER, generation_job, judgment_job, summary_job
from legacy import algorithms
from workspace import (LEGS, Workspace, canonical, digest_text, read_json, sha,
                       terminal_rows, verify_inputs)

LLAMA_COMMIT = "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3"
TABBY_COMMIT = "4a4f9f44820303593844f092d424bb7506008733"
EXL3_SHARDS = {
    "model-00001-of-00002.safetensors": "cf8966ebc88599d904d4b1ac65d567a15b00b7f9c97ef54c8d77eb534ff8eb93",
    "model-00002-of-00002.safetensors": "2a72afda6bf041732018ee34aca772a4a268ef90905da0c710397e68a9e1a746",
}
GGUF_HASHES = {
    "Qwen3-8B-UD-Q4_K_XL.gguf": "34a514d08f7449cb4a694a707aaa2eedccb7bb68290121bf5e5a569b2abe71c3",
    "gpt-oss-20b-MXFP4.gguf": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
}


def output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def min_free_vram_mib():
    """Operational launch floor (default 11000 MiB), adjustable like the primary launcher's:
    with the desktop session on this GPU about 10,750 MiB are free (research/P5_AMENDMENT_2026-10-30.md)."""
    floor = int(os.environ.get("P5_MIN_FREE_VRAM_MIB", "11000"))
    if not 9900 <= floor <= 12282:
        raise ValueError("P5_MIN_FREE_VRAM_MIB must be between 9900 and 12282")
    return floor


def preflight(work):
    """Read-only safety checks; no global resource claims or service changes."""
    for unit in ("p5-phase1.service", "p5-phase2.service"):
        status = output(["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"])
        if status in ("active", "activating", "deactivating", "reloading"):
            raise ValueError(f"primary unit is still active: {unit}")
    for port in (18125, 18126, 18127, 18130):
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"study GPU endpoint is already occupied: {port}")
    fields = output(["nvidia-smi", "--query-gpu=name,memory.free,temperature.gpu,driver_version",
                     "--format=csv,noheader,nounits"]).splitlines()
    if len(fields) != 1:
        raise ValueError("expected the original single-GPU study machine")
    name, free, temp, driver = [s.strip() for s in fields[0].split(",")]
    floor = min_free_vram_mib()
    if name != "NVIDIA GeForce RTX 4080 Laptop GPU" or int(free) < floor or int(temp) >= 75:
        raise ValueError(f"original GPU must have at least {floor} MiB free and be below 75 C before launch")
    memory = {line.split(":", 1)[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    if memory["MemAvailable"] < 24 * 1024 * 1024:
        raise ValueError("less than 24 GiB host RAM available")
    if work.path("PAUSE").exists() or (work.root / "testsuite/paper5/PAUSE").exists() or (work.root / "testsuite/paper6/PAUSE").exists():
        raise ValueError("study pause marker is present")
    return {"host": platform.node(), "gpu": name, "driver": driver}


def artifacts(a, stage, info):
    model = "gptoss" if stage == "judge-gptoss" else (SUMMARIZER if stage == "summaries" else info["model"])
    paths = {}
    if model == "gptoss":
        file = Path(a.judge.GPTOSS)
        paths[str(file)] = GGUF_HASHES[file.name]
    elif model == "qwen3-8b":
        file = Path(a.grid.QWEN3_8B)
        paths[str(file)] = GGUF_HASHES[file.name]
    else:
        directory = Path(a.C.model_dir("qwen38-exl3"))
        paths.update({str(directory / name): digest for name, digest in EXL3_SHARDS.items()})
    verify_inputs(paths)  # Full hashes, not filename/size/mtime trust.
    engine = "llama.cpp" if model != "qwen38-exl3" else "tabbyAPI"
    revision = output(["git", "-C", str(Path(a.C.REPO) / "engines" / engine), "rev-parse", "HEAD"])
    if revision != (LLAMA_COMMIT if engine == "llama.cpp" else TABBY_COMMIT):
        raise ValueError("engine is not the study's pinned commit")
    if engine == "llama.cpp":
        binary_dir = Path(a.C.REPO) / "engines/llama.cpp/build/bin"
        paths[str(binary_dir / "llama-server")] = sha(binary_dir / "llama-server")
        paths.update({str(p.resolve()): sha(p) for p in binary_dir.glob("*.so*") if p.is_file()})
    else:
        # Bind the actual local EXL3 copy, including tracked/untracked patches;
        # do not substitute the newer daily-driver engine during this study.
        code = Path(a.C.REPO) / "engines/exl3-1.4.7-lab"
        selected = sorted(p for p in code.rglob("*") if p.is_file() and p.suffix in (".py", ".cu", ".cpp", ".h", ".cuh"))
        if not selected:
            raise ValueError("pinned EXL3 source tree is missing")
        paths.update({str(p): sha(p) for p in selected})
    return {"served_model": model, "engine": engine, "revision": revision, "files": paths}


def server_for(a, stage, info):
    if stage == "summaries":
        return a.grid.make_server(SUMMARIZER)
    return a.judge.gptoss_server() if stage == "judge-gptoss" else a.grid.make_server(info["model"])


def primary_summaries(work):
    """The completed primary summary journal (read-only), as job id -> text."""
    path = work.root / "testsuite/paper5/work/journals/summaries.jsonl"
    before = sha(path)
    rows = terminal_rows(path)
    if sha(path) != before:
        raise ValueError("primary summary journal changed while reading")
    return {jid: row["text"] for jid, row in rows.items() if row["ok"] and (row.get("text") or "").strip()}


def runner_type(a, work, served_model, fingerprint):
    class FollowupRunner(a.runner.Runner):
        def gate(self):
            # Preserve the original shared pause/thermal behavior. A private
            # marker is also recognized without editing the primary harness.
            while work.path("PAUSE").exists() or (work.root / "testsuite/paper6/PAUSE").exists():
                self.server.stop()
                time.sleep(10)
            super().gate()

        def _append(self, row):
            row["execution_fingerprint_sha256"] = fingerprint
            if row.get("terminal") and row.get("ok"):
                if served_model in ("qwen38-exl3", "qwen3-8b"):
                    # The primary Runner always counts with its 27B tokenizer.
                    # Recount cross-model completions with the actual tokenizer.
                    row["completion_tokens_harness"] = a.C.count_tokens(row.get("text") or "", a.C.model_dir(served_model))
                    row["completion_tokenizer_model"] = served_model
                else:
                    # gpt-oss may have a hidden reasoning channel; report the
                    # server's usage, not a 27B-tokenizer count of final text.
                    row["completion_tokens_harness"] = None
                    row["completion_tokenizer_model"] = None
            super()._append(row)

        def run(self, job, endpoint="/v1/completions", retries=2):
            cached = self.done.get(job["job_id"])
            if cached is not None:
                if cached.get("execution_fingerprint_sha256") != fingerprint or cached.get("meta") != job.get("meta"):
                    raise ValueError("cached request belongs to different frozen inputs/execution")
                if cached.get("ok") and (cached.get("params") != job["params"] or cached.get("prompt_sha256") != job["meta"]["prompt_sha256"]):
                    raise ValueError("cached request body differs from the proposed request")
            return super().run(job, endpoint, retries)
    return FollowupRunner


def execute(work, stage, *, live=False, idle_seconds=10., limit=0):
    info, items = work.load()
    if not math.isfinite(idle_seconds) or idle_seconds < 0 or limit < 0:
        raise ValueError("idle seconds and limit must be nonnegative")
    if stage == "summaries":
        if work.leg == "locomo" and work.expected_arms() != ("T", "S"):
            raise ValueError("summaries are needed only for a selected LoCoMo S leg")
        prompts = None
    else:
        prompts = work.frozen_prompts()
    if stage == "judge-self" and work.leg != "cross-model":
        raise ValueError("LoCoMo's declared scorers are gptoss and token F1")
    if not live:
        return {"leg": work.leg, "stage": stage, "inference_started": False,
                "prompt_count": None if prompts is None else len(prompts), "idle_seconds": idle_seconds,
                "remaining_gate": "check live machine inventory/headroom, then use --execute"}
    # Both legs are downstream of the completed primary evidence. This check
    # never runs primary inference, overwrites claims, or chooses from a prefix.
    verified_primary(work.root)
    machine = preflight(work)
    paths = {"generate": "grid.jsonl", "judge-gptoss": "judge_gptoss__grid.jsonl",
             "judge-self": "judge_self__grid.jsonl", "summaries": "summaries.jsonl"}
    journal = work.path("journals/" + paths[stage])
    if journal.exists():
        terminal_rows(journal)  # reject duplicate/corrupt terminal rows before Runner deduplication
    with algorithms(work) as a:
        identity = artifacts(a, stage, info)
        signature = {"schema": "paper5-followup-execution-v1", "stage": stage, "leg": work.leg,
                     "machine": machine, "artifact_identity": identity,
                     "workspace_sha256": sha(work.path("WORKSPACE.json"))}
        if prompts is not None:
            signature["prompt_manifest_sha256"] = sha(work.path("PROMPTS_MANIFEST.json"))
        receipt = work.write_json_once(f"execution/{stage}.json", signature)
        fingerprint = sha(receipt)
        server = server_for(a, stage, info)
        Runner = runner_type(a, work, identity["served_model"], fingerprint)
        runner = Runner(server, str(journal), idle_seconds=idle_seconds)
        count = 0
        complete = True
        try:
            if stage == "generate":
                groups = [[p for p in prompts if p["arm"] == arm] for arm in work.expected_arms()]
                ordered = a.grid.interleave(groups)
                complete = not limit or limit >= len(ordered)
                for p in ordered[:limit or None]:
                    runner.run(generation_job(a, work, p, info))
                    count += 1
                    if count % 25 == 0:
                        a.runner.log(f"{work.leg} generation {count}/{len(ordered)}")
            elif stage.startswith("judge-"):
                generations = terminal_rows(work.path("journals/grid.jsonl"))
                if set(generations) != {p["job_id"] for p in prompts}:
                    raise ValueError("generation must be complete before judging")
                generation_signature = sha(work.path("execution/generate.json"))
                generation_done = read_json(work.path("execution/generate-finished.json"))
                if generation_done.get("complete") is not True or generation_done.get("journal_sha256") != sha(work.path("journals/grid.jsonl")):
                    raise ValueError("generation has not completed a verified stage")
                for p in prompts:
                    gen = generations[p["job_id"]]
                    expected = generation_job(a, work, p, info)
                    if gen.get("execution_fingerprint_sha256") != generation_signature or gen.get("meta") != expected["meta"]:
                        raise ValueError("generation belongs to another input/execution freeze")
                    if gen["ok"] and (gen.get("params") != expected["params"] or gen.get("prompt_sha256") != p["prompt_sha256"] or gen.get("server_prompt_tokens") != p["n_prompt"]):
                        raise ValueError("generation request/parity differs from its frozen prompt")
                judge = stage.removeprefix("judge-")
                by_item = {it["question_id"]: it for it in items}
                ordered = [generations[p["job_id"]] for p in prompts]
                complete = not limit or limit >= len(ordered)
                for gen in ordered[:limit or None]:
                    job = judgment_job(a, work, judge, by_item[gen["meta"]["question_id"]], gen, info)
                    endpoint = "/v1/chat/completions" if judge == "gptoss" else "/v1/completions"
                    row = runner.run(job, endpoint)
                    if judge == "gptoss" and row["ok"] and not (row.get("text") or "").strip():
                        retry = dict(job, job_id=job["job_id"] + "|retry", params=dict(job["params"], max_tokens=8192))
                        runner.run(retry, endpoint)
                    count += 1
            else:
                # Cross-model S reuses every primary summary; only sessions the primary never
                # summarized are requested. Selection mirrors build_S with the leg's tokenizer.
                primary = primary_summaries(work) if work.leg == "cross-model" else {}
                complete = not limit or limit >= len(items)
                for item in items[:limit or None]:
                    chosen, summaries = [], {}
                    for index in reversed(a.C.chronological(item)):
                        job = summary_job(a, work, item, index)
                        text = primary.get(job["job_id"])
                        if text is None:
                            row = runner.run(job)
                            if not row["ok"] or not (row.get("text") or "").strip():
                                raise ValueError("a required summary failed; inspect before proceeding")
                            text = row["text"]
                        summaries[index] = text
                        if a.BP.count(a.SM.s_prompt(item, chosen + [index], summaries, model=a.BP.MODEL)) > 4096:
                            break
                        chosen.append(index)
                    count += 1
        finally:
            server.stop()  # only the process group this object started
        verify_inputs(identity["files"])
        work.load()
        if prompts is not None:
            work.frozen_prompts()
        result = {"leg": work.leg, "stage": stage, "processed": count, "complete": complete,
                  "execution_fingerprint_sha256": fingerprint, "journal_sha256": sha(journal),
                  "idle_seconds": idle_seconds, "inference_started": True}
        if complete:
            work.write_json_once(f"execution/{stage}-finished.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", choices=tuple(LEGS))
    parser.add_argument("stage", choices=("generate", "judge-gptoss", "judge-self", "summaries"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=10.)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(Workspace(args.leg), args.stage, live=args.execute,
                                 idle_seconds=args.idle_seconds, limit=args.limit), indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"P5 follow-up stage stopped: {exc}\n")


if __name__ == "__main__":
    main()
