"""Read-only primary execution receipt checks, shared by resume and paper audit."""
from pathlib import Path

from evidence import ROOT
from diagnostics import require
from draft_support import read_json, sha
import execution_snapshot as E
from migrate_followup_preparation import safe_path
from primary_requests import canonical
from primary_server import AMENDMENTS_REL, llama_placement, validate_version

BASELINE_REL = "testsuite/paper5/work/execution/pre-primary-20260920/SNAPSHOT.json"
BASELINE_SHA = "dfd03f0f627ebda68a3af239e4d243241bb1605bc7815a7865604ca04fc35ab1"
SOURCE_FILES = ("publication/draft_support.py", "testsuite/paper5_followups/legacy.py") + tuple(
    "paper5/scripts/" + name for name in ("primary_execution.py", "primary_requests.py", "primary_server.py", "primary_receipts.py",
    "primary_audit.py", "execution_snapshot.py", "evidence.py", "diagnostics.py", "followup_results.py",
    "migrate_followup_preparation.py", "align_oracle_dates.py")) + ("paper5/execution_machine_amendments.json", "paper5/execution_postcheck_resolutions.json")
LATE_POLICY_SOURCES = {"paper5/execution_machine_amendments.json", "paper5/execution_postcheck_resolutions.json"}


def expected_launch(stage, root):
    if stage == "judge-gptoss":
        return ([str(root / "engines/llama.cpp/build/bin/llama-server"), "-m",
                 str(Path.home() / "models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf"), "--host", "127.0.0.1", "--port", "18126", "--no-webui",
                 "-c", "16384", "-ngl", "99", "--n-cpu-moe", "6", "--jinja", "-np", "1", "--temp", "0"], str(root))
    require(stage in ("generate", "judge-self"), "unknown primary stage")
    return ([str(root / "engines/exl3-env/bin/python"), "main.py", "--config", str(root / "testsuite/paper5/tabby-p5.yml")],
            str(root / "engines/tabbyAPI"))


def baseline(root=ROOT):
    path = safe_path(root, BASELINE_REL)
    require(sha(path) == BASELINE_SHA, "the preserved pre-primary baseline changed")
    return read_json(path)


class Bindings:
    def __init__(self, stage, inputs, *, root=ROOT, baseline_record=None):
        self.stage, self.inputs, self.root = stage, inputs, Path(root)
        self.baseline = E.apply_amendments(baseline_record if baseline_record is not None else baseline(root), root)
        self.files = {str(self.root / BASELINE_REL): BASELINE_SHA} if baseline_record is None else {}
        self.sessions, self.instances = {}, {}
        self.placement = None

    def document(self, reference, kind):
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "invalid execution reference")
        path = safe_path(self.root, reference["path"])
        require(path.is_relative_to(self.root / "testsuite/paper5/work/execution/primary") and path.name.endswith(".json"), "receipt is outside primary execution workspace")
        digest = sha(path)
        require(digest == reference["sha256"], "execution reference hash differs")
        self.files[str(path)] = digest
        key = str(path)
        cache = self.sessions if kind == "session" else self.instances
        if key in cache:
            return cache[key]
        record = read_json(path)
        if kind == "session":
            require(record["schema"] == "p5-primary-session-v1" and record["stage"] == self.stage, "wrong primary session")
            require(record["baseline_sha256"] == BASELINE_SHA and record["inputs"] == self.inputs, "session input/baseline binding differs")
            archived = set(record["source_sha256"])
            # The two policy files were added to the archive list during the campaign
            # (research/P5_AMENDMENT_2026-09-25.md); sessions older than a file need not hold it.
            require(archived <= set(SOURCE_FILES) and set(SOURCE_FILES) - LATE_POLICY_SOURCES <= archived,
                    "incomplete launcher-source archive")
            require((record["server_argv"], record["server_cwd"]) == expected_launch(self.stage, self.root), "session changed the declared launch command")
            for name, expected in record["source_sha256"].items():
                source = safe_path(path.parent / "sources", name)
                require(sha(source) == expected, "archived primary launcher source changed")
                self.files[str(source)] = expected
        else:
            require(record["schema"] == "p5-primary-server-instance-v1", "wrong server-instance receipt")
            # Saved instances may predate a later recorded amendment (generation ran under kernel
            # 7.0.0-31; judging under -38): any machine on the recorded chain is accepted here.
            require(record["machine"] in E.machine_history(self.baseline, self.root), "server instance changed machine/GPU/driver")
            server = record["server"]
            require(all(type(server.get(k)) is int and server[k] > 0 for k in ("pid", "ppid", "start_ticks")) and
                    isinstance(server.get("boot_id"), str) and bool(server["boot_id"]), "incomplete server process identity")
            executable = str(Path(expected_launch(self.stage, self.root)[0][0]).resolve())
            require(server.get("executable") == executable and executable in server["mapped_files"],
                    "serving executable is not fingerprinted in mapped runtime")
            require(record["gpu_processes"] and len(record["gpu_processes"]) == 1 and
                    record["gpu_processes"][0]["pid"] == server["pid"] and
                    record["gpu_processes"][0]["gpu_uuid"] == self.baseline["machine"]["gpu"]["uuid"], "instance GPU ownership mismatch")
            prefix = record["startup_prefix"]
            require(E.hashlib.sha256(prefix.encode()).hexdigest() == record["startup_prefix_sha256"], "instance startup prefix changed")
            if self.stage == "judge-gptoss":
                require(server["listener_port"] == 18126, "wrong recorded judge port")
                require(server["environment_allowlist"].get("LD_LIBRARY_PATH") == str(Path.home() / "cuda-12.6/lib64"), "judge used a different CUDA library path")
                require(server["environment_allowlist"].get("LLAMA_ARG_LOG_VERBOSITY") == "4", "judge placement log level is not recorded")
                validate_version(record["binary_version_readback"])
                expected = {"server": "judge-gptoss20b", **llama_placement(prefix)}
            else:
                extension = str(self.root / "engines/exl3-1.4.7-lab/exllamav3_ext.cpython-312-x86_64-linux-gnu.so")
                require(extension in server["mapped_files"] and
                        server["environment_allowlist"].get("PYTHONPATH") == str(self.root / "engines/exl3-1.4.7-lab") and
                        server["environment_allowlist"].get("EXL3_CKPT_PP") == "1024", "generator native extension/serving override is missing")
                require(server["listener_port"] == 18125 and f"Started server process [{server['pid']}]" in prefix and
                        "exllamav3 version: 1.4.7" in prefix and "Using main model MTP component for drafting" in prefix and
                        "Using configured max_seq_len: 32768 tokens." in prefix,
                        "missing generator PID/version/MTP evidence")
                expected = {"server": "tabby-p5", "mtp": True, "max_seq_len": 32768,
                            "config_sha256": self.inputs[str(self.root / "testsuite/paper5/tabby-p5.yml")]}
            require(record["placement"] == expected, "placement receipt differs from startup evidence")
            require(self.placement is None or self.placement == expected, "placement changed across saved instances")
            self.placement = expected
            require(set(server["mapped_files"]) == record["runtime_files"].keys(), "incomplete mapped runtime fingerprint")
            for p, observed in record["runtime_files"].items():
                if p in self.baseline["files"]:
                    accepted = self.baseline["files"][p].get("accepted_sha256") or [self.baseline["files"][p]["sha256"]]
                    require(observed["sha256"] in accepted, "instance runtime differs from baseline")
            original = E.prefix_record(server["log"], len(prefix.encode()))
            require(original["sha256"] == record["startup_prefix_sha256"], "original instance startup log changed")
        cache[key] = record
        return record

    def __call__(self, row):
        binding = row["execution"]
        require(set(binding) == {"session", "server", "intent_sha256", "postcheck_ok"}, "incomplete row execution binding")
        require(binding["postcheck_ok"] is (True if row.get("ok") is True else None), "request failed its execution postcheck")
        session = self.document(binding["session"], "session")
        reference = binding["server"]
        if reference is None:
            require(row.get("recovered_exhausted_failure") is True and row.get("terminal") is True and row.get("ok") is False,
                    "unobserved live request cannot lack a server instance")
            return
        instance = self.document(reference, "server")
        require(instance["session_sha256"] == binding["session"]["sha256"], "server belongs to a different invocation")
        require(instance["server"]["argv"] == session["server_argv"] and instance["server"]["cwd"] == session["server_cwd"],
                "observed command differs from session launch")
        if row.get("ok"):
            require((row.get("server") or {}).get("pid") == instance["server"]["pid"], "response PID differs from observed server")

    def stable(self):
        require(all(sha(p) == digest for p, digest in self.files.items()), "execution receipt changed during verification")

    def finished(self, journal, *, terminal_outcomes):
        digest = sha(journal)
        found = []
        for path in sorted((self.root / "testsuite/paper5/work/execution/primary").glob(self.stage + "-*/FINISHED.json")):
            safe_path(self.root, str(path.relative_to(self.root)))
            record = read_json(path)
            if record.get("journal_sha256") != digest:
                continue
            require(record["schema"] == "p5-primary-stage-finished-v1" and record["stage"] == self.stage and
                    record["complete"] is True and record["baseline_before_after_verified"] == BASELINE_SHA and
                    record["source_hashes_unchanged"] is True, "invalid stage-completion receipt")
            require(type(record.get("terminal_outcomes")) is int and record["terminal_outcomes"] == terminal_outcomes,
                    "stage-completion outcome count differs from journal")
            attempts = record.get("recorded_http_attempts")
            require(type(attempts) is int and attempts >= 0 and record.get("inference_started") is (attempts > 0),
                    "stage-completion request count/readback is invalid")
            self.document(record["session"], "session")
            self.files[str(path)] = sha(path)
            found.append(str(path))
        require(found, "completed stage still lacks its after-hash verification receipt; resume the same stage to finish checks")
        return found


def audit_complete(prompts, generation_records, judge_records, items, cap, manifest, algorithms,
                   primary_inputs, *, root=ROOT):
    """Execution bindings and completion receipts for all three primary stages.

    Request bodies were independently reconstructed by primary_audit. Here
    their complete intents (including timeouts/parity) are tied to archived
    sessions and observed server instances, without launching a server.
    """
    import primary_requests as Q
    import primary_audit as P
    from diagnostics import verdicts
    jobs = {p["job_id"]: algorithms.grid.to_job(p, cap) for p in prompts}
    by_file = {name: [] for name in Q.ARM_FILES}
    for p in prompts:
        by_file[f"{p['arm']}_{'none' if p['budget'] is None else p['budget']}.jsonl"].append(p)
    smoke = algorithms.grid.interleave([by_file[name] for name in Q.SMOKE_FILES])[:24]
    harness = {name: manifest["harness"][name] for name in P.RUNNER_FILES}
    # Terminal token counts have already been independently verified. Avoid
    # loading a tokenizer here; validate_row's completion-count callback must
    # still resolve every nonempty text to that verified count.
    counts = {}
    for row in generation_records + [r for batch in judge_records.values() for r in batch]:
        if row.get("ok") and row.get("text"):
            previous = counts.setdefault(row["text"], row["completion_tokens_harness"])
            require(previous == row["completion_tokens_harness"], "inconsistent prior tokenizer counts for identical text")
    counter = lambda text: counts[text]
    first = Bindings("generate", primary_inputs, root=root)
    generated = Q.journal_state(generation_records, jobs, stage="generate", harness=harness, count_tokens=counter,
                                legacy_ids=[p["job_id"] for p in smoke], check_binding=first)
    require(not generated["pending"] and generated["done"].keys() == jobs.keys(), "primary execution inventory incomplete")
    first.finished(root / "testsuite/paper5/work/journals/grid.jsonl", terminal_outcomes=len(generated["done"]))
    inputs = dict(primary_inputs)
    gen_path = root / "testsuite/paper5/work/journals/grid.jsonl"
    inputs[str(gen_path)] = sha(gen_path)
    bound = {"generate": first}
    by_id = {it["question_id"]: it for it in items}
    for judge, records in judge_records.items():
        stage = "judge-" + judge
        targets = {}
        for gen in generated["done"].values():
            it = by_id[gen["meta"]["question_id"]]
            for retry in ((False, True) if judge == "gptoss" else (False,)):
                job = Q.judge_job(algorithms, it, gen, judge, retry)
                targets[job["job_id"]] = job
        check = Bindings(stage, inputs, root=root)
        state = Q.journal_state(records, targets, stage=stage, harness=harness, count_tokens=counter, check_binding=check)
        require(not state["pending"], "unfinished judged request")
        verdicts(generated["done"], state["done"], judge)
        check.finished(root / f"testsuite/paper5/work/journals/judge_{judge}__grid.jsonl", terminal_outcomes=len(state["done"]))
        bound[stage] = check
    for check in bound.values():
        check.stable()
    return {"schema": "p5-primary-execution-audit-v1", "files": {p: v for check in bound.values() for p, v in check.files.items()},
            "bound_sessions": {stage: len(check.sessions) for stage, check in bound.items()},
            "bound_server_instances": {stage: len(check.instances) for stage, check in bound.items()},
            "legacy_smoke_outcomes": len(generated["legacy_terminal_ids"]),
            "legacy_boundary": "initial smoke prefix has original request/server readbacks, not the later wrapper's per-request observations"}
