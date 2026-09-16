"""Read-only completion checks before P5's aggregate analysis can become paper text.

Incomplete inference must not look like a completed all-incorrect experiment.
Actual terminal failures remain recorded outcomes under the fixed protocol.
No imports from the serving harness and no model or network calls.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "publication"))
from draft_support import integer, read_json, sha, _unique

FROZEN_HARNESS = {"p5common.py", "build_prompts.py", "summarize.py", "lingua_probs.py",
                  "run_grid.py", "p5run.py", "judge.py", "analyze.py", "tabby-p5.yml"}


def rows(path):
    def reject(value):
        raise ValueError(f"non-finite journal number: {value}")
    def finite(value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else reject(value)
    with Path(path).open() as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank journal row: {path}:{lineno}")
            row = json.loads(line, object_pairs_hook=_unique, parse_constant=reject, parse_float=finite)
            if not isinstance(row, dict):
                raise ValueError(f"invalid journal row: {path}:{lineno}")
            yield row


RESOLUTIONS_REL = "paper5/execution_postcheck_resolutions.json"


def excluded_rows(root=ROOT):
    """(journal file name, job id, row SHA-256) of every row excluded by a recorded
    exclude-and-rerun resolution. The runner applies the same rule when it resumes
    (primary_requests.journal_state); analysis must not count an excluded row."""
    path = Path(root) / RESOLUTIONS_REL
    if not path.exists():
        return set()
    record = read_json(path)
    if record.get("schema") != "p5-postcheck-resolutions-v1":
        raise ValueError("unknown postcheck-resolution schema")
    out = set()
    for item in record["resolutions"]:
        if item["decision"] != "exclude-and-rerun":
            raise ValueError("unknown postcheck resolution decision")
        out.add((Path(item["journal"]).name, item["job_id"], item["row_sha256"]))
    return out


def row_sha256(row):
    # Same canonical form as primary_requests.canonical, which keys the resolutions.
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def without_excluded(records, journal_name, root=ROOT):
    """The journal's records minus rows excluded by a recorded resolution (outcome analyses).
    Receipt audits keep the raw records: primary_requests.journal_state applies the same rule."""
    excluded = excluded_rows(root)
    out = []
    for row in records:
        if excluded and (journal_name, row.get("job_id"), row_sha256(row)) in excluded:
            if not (row.get("terminal") is True and row.get("ok") is True and
                    (row.get("execution") or {}).get("postcheck_ok") is False):
                raise ValueError("a resolution may only exclude a failed-postcheck outcome")
            continue
        out.append(row)
    return out


def terminals(path, root=ROOT):
    out = {}
    excluded = excluded_rows(root)
    for row in rows(path):
        if type(row.get("terminal")) is not bool:
            raise ValueError(f"explicit terminal status required in {path}")
        if row["terminal"]:
            jid = row["job_id"]
            if excluded and (Path(path).name, jid, row_sha256(row)) in excluded:
                if not (row.get("ok") is True and (row.get("execution") or {}).get("postcheck_ok") is False):
                    raise ValueError("a resolution may only exclude a failed-postcheck outcome")
                if jid in out:
                    raise ValueError("an excluded row cannot follow its job's terminal outcome")
                continue
            if jid in out:
                raise ValueError(f"duplicate terminal job in {path.name}: {jid}")
            if type(row.get("ok")) is not bool:
                raise ValueError(f"missing outcome status: {jid}")
            out[jid] = row
    return out


def coverage(prompts, generations, judges):
    """Check complete explicit attempts, not success rate. Suitable for tiny fixtures."""
    if set(judges) != {"gptoss", "self"}:
        raise ValueError("both declared judges are required")
    expected = {}
    cells = defaultdict(set)
    for row in prompts:
        jid = row["job_id"]
        if jid in expected:
            raise ValueError(f"duplicate prompt job: {jid}")
        expected[jid] = row
        group = f"{row['arm']}@{row['budget']}"
        if row["question_id"] in cells[group]:
            raise ValueError("duplicate item within a cell")
        cells[group].add(row["question_id"])
        integer(row["n_prompt"], "prompt tokens", 1)
        if row["budget"] is not None and row["n_prompt"] > row["budget"]:
            raise ValueError("prompt exceeds assigned budget")
    if set(generations) != set(expected):
        raise ValueError("grid has missing or unexpected terminal jobs")
    successful = set()
    for jid, row in generations.items():
        if type(row.get("ok")) is not bool:
            raise ValueError("missing terminal outcome status")
        if row["ok"]:
            successful.add(jid)
            if row.get("prompt_sha256") != expected[jid]["prompt_sha256"]:
                raise ValueError("generation used a different prompt")
            # The pinned Tabby raw-completion API leaves usage empty; the
            # runner records a separate server-tokenizer readback for parity.
            if row.get("server_prompt_tokens") != expected[jid]["n_prompt"]:
                raise ValueError(f"served token count mismatch: {jid}")
    for label, records in judges.items():
        seen = set()
        for row in records.values():
            jid = row["meta"]["gen_job_id"]
            if jid not in expected:
                raise ValueError(f"{label}: judge row for absent generation")
            seen.add(jid)
        if not successful <= seen:
            raise ValueError(f"{label}: judging is incomplete")
    return cells


def verify(root=ROOT):
    harness = Path(root) / "testsuite/paper5"
    work = harness / "work"
    manifest_path = harness / "PROMPTS_MANIFEST.json"
    manifest = read_json(manifest_path)  # absent until prompt preparation is finished
    if manifest["schema"] != "paper5-prompts-v1":
        raise ValueError("wrong prompt manifest schema")
    paths = [manifest_path, work / "cap.json"]
    prompts = []
    frozen_names = set(manifest["files"])
    if frozen_names != {p.name for p in (work / "prompts").glob("*.jsonl")}:
        raise ValueError("prompt inventory differs from frozen manifest")
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("prompt manifest path is not a basename")
        path = work / "prompts" / name
        if sha(path) != record["sha256"]:
            raise ValueError(f"prompt hash mismatch: {name}")
        batch = list(rows(path))
        if len(batch) != record["prompts"]:
            raise ValueError("prompt count differs from manifest")
        for row in batch:
            if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError("prompt content disagrees with its recorded hash")
        prompts.extend(batch)
        paths.append(path)
    if set(manifest["harness"]) != FROZEN_HARNESS:
        raise ValueError("incomplete frozen harness inventory")
    for name, digest in manifest["harness"].items():
        if Path(name).name != name or sha(harness / name)[:16] != digest:
            raise ValueError(f"frozen harness changed: {name}")
        paths.append(harness / name)  # preserve full source hashes in downstream bindings
    generation_path = work / "journals/grid.jsonl"
    judge_paths = {label: work / "journals" / f"judge_{label}__grid.jsonl" for label in ("gptoss", "self")}
    generations = terminals(generation_path)
    judges = {label: terminals(path) for label, path in judge_paths.items()}
    cells = coverage(prompts, generations, judges)
    required = {f"{a}@{b}" for a in ("T", "S", "C", "M", "Cw") for b in (1024, 4096)}
    required |= {"Mbm25@4096", "E@None", "O@4096"}
    if set(cells) != required:
        raise ValueError("primary prompt grid does not match the declared arms")
    sample = cells["T@4096"]
    if len(sample) != 500:
        raise ValueError("expected 500 primary items")
    for key, ids in cells.items():
        if key == "O@4096":
            if not ids or not ids <= sample:
                raise ValueError("invalid oracle-fit subset")
        elif ids != sample:
            raise ValueError("unpaired or incomplete cell: " + key)
    paths += [generation_path, *judge_paths.values()]
    return {"cells": {key: len(value) for key, value in cells.items()},
            "inputs": {str(path.relative_to(root)): sha(path) for path in paths},
            "completion_cap": read_json(work / "cap.json")["completion_cap"]}
