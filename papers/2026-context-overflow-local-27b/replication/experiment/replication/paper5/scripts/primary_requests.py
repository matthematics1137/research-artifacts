"""Primary P5 request construction and strict resume inventory; no serving calls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from evidence import FROZEN_HARNESS, ROOT, rows
from draft_support import integer, read_json, sha
import primary_audit as P
from diagnostics import require, visible

ARM_FILES = tuple(f"{arm}_{budget}.jsonl" for budget in (1024, 4096) for arm in ("T", "S", "C", "M", "Cw")) + (
    "Mbm25_4096.jsonl", "O_4096.jsonl", "E_none.jsonl")
SMOKE_FILES = tuple(arm + "_4096.jsonl" for arm in ("T", "S", "C", "M"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def intent(job, endpoint="/v1/completions"):
    return P.digest(canonical({"job": job, "endpoint": endpoint}))


def spec(job):
    return {key: job[key] for key in ("job_id", "n_prompt", "budget", "params", "meta")} | {
        "prompt_sha256": P.digest(job.get("prompt") or json.dumps(job.get("messages")))}


def frozen_inputs(root, a):
    root = Path(root)
    harness, work = root / "testsuite/paper5", root / "testsuite/paper5/work"
    manifest_path = harness / "PROMPTS_MANIFEST.json"
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "paper5-prompts-v1" and set(manifest["files"]) == set(ARM_FILES), "wrong frozen primary inventory")
    require({p.name for p in (work / "prompts").glob("*.jsonl")} == set(ARM_FILES), "unexpected primary prompt file")
    require(set(manifest["harness"]) == FROZEN_HARNESS, "incomplete primary source freeze")
    bindings = {str(manifest_path): sha(manifest_path)}
    for name, expected in manifest["harness"].items():
        path = harness / name
        bindings[str(path)] = sha(path)
        require(bindings[str(path)][:16] == expected, "primary source changed: " + name)
    cap_path = work / "cap.json"
    cap_record = read_json(cap_path)
    cap = cap_record["completion_cap"]
    require(type(cap) is int and cap in (8192, 16384), "invalid pilot-selected cap")
    lengths = cap_record["pilot_lengths"]
    require(len(lengths) == 20 and all(type(v) is int and 0 <= v <= 16384 for v in lengths), "invalid pilot length inventory")
    require(max(lengths) == cap_record["pilot_max_completion"] and cap == (16384 if max(lengths) >= 6144 else 8192),
            "selected cap differs from pilot decision rule")
    pilot_path = work / "journals/pilot.jsonl"
    pilot = P.journal_attempts(list(rows(pilot_path)))
    require(len(pilot) == 20 and all(r["ok"] and jid.startswith("pilot|") for jid, r in pilot.items()), "pilot outcomes are incomplete")
    require(sorted(r["completion_tokens_harness"] for r in pilot.values()) == lengths, "pilot lengths do not match saved outcomes")
    bindings.update({str(cap_path): sha(cap_path), str(pilot_path): sha(pilot_path)})
    items = a.C.load_json_verified(a.C.LME_S)
    bindings[str(a.C.LME_S)] = a.C.DATA_SHA256[a.C.LME_S]
    by_id = {it["question_id"]: it for it in items}
    require(len(items) == len(by_id) == 500, "primary source cohort differs")
    groups = {}
    for name in ARM_FILES:
        path = work / "prompts" / name
        require(sha(path) == manifest["files"][name]["sha256"], "frozen prompt file changed")
        batch = list(rows(path))
        require(len(batch) == manifest["files"][name]["prompts"], "prompt count changed")
        qids = [p["question_id"] for p in batch]
        require(len(qids) == len(set(qids)) and set(qids) <= by_id.keys(), "foreign or duplicate prompt item")
        require(bool(qids) if name == "O_4096.jsonl" else set(qids) == by_id.keys(), "incomplete prompt cell")
        for p in batch:
            it = by_id[p["question_id"]]
            expected_name = f"{p['arm']}_{'none' if p['budget'] is None else p['budget']}.jsonl"
            require(name == expected_name and p["question_type"] == it["question_type"], "mislabelled primary prompt")
            integer(p["n_prompt"], "prompt token count", 1)
            require(a.C.count_tokens(p["prompt"]) == p["n_prompt"] and P.digest(p["prompt"]) == p["prompt_sha256"], "prompt count/hash differs")
            require(p["budget"] is None or p["n_prompt"] <= p["budget"], "prompt budget exceeded")
            pre, post = a.C.WRAP_THINK
            require(p["prompt"].startswith(pre) and p["prompt"].endswith(post), "wrong thinking wrapper")
            require(p["prompt"][len(pre):-len(post)].endswith(a.C.empty_prompt(it["question_date"], it["question"])),
                    "wrong source question/current date")
            job = a.grid.to_job(p, cap)
            require(P.same(spec(job), P.generation_spec(p, cap)), "generation request constructor differs from declared request")
        groups[name] = batch
        bindings[str(path)] = sha(path)
    alignment = P.oracle_alignment(root, a.C.MODEL_DIR)
    bindings.update(alignment["files"])
    prompts = a.grid.interleave([groups[name] for name in ARM_FILES])
    smoke = a.grid.interleave([groups[name] for name in SMOKE_FILES])[:24]
    require(all(sha(p) == expected for p, expected in bindings.items()), "primary inputs changed while preparing requests")
    return {"manifest": manifest, "bindings": bindings, "cap": cap, "items": by_id,
            "prompts": prompts, "smoke_ids": [p["job_id"] for p in smoke], "arm_order": list(ARM_FILES)}


def judge_job(a, item, gen, judge, retry=False):
    answer, _ = visible(gen.get("text") if gen["ok"] else None)
    text = a.judge.judge_prompt(item, answer)
    expected = P.judgment_spec(item, gen, judge, a, retry)
    job = {key: expected[key] for key in ("job_id", "n_prompt", "budget", "params", "meta")}
    job["timeout"] = 600
    if judge == "gptoss":
        job["messages"] = [{"role": "user", "content": text}]
    else:
        job["prompt"] = a.C.wrap(text, thinking=False)
    require(P.same(spec(job), expected), "judge job differs from declared request")
    return job


RESOLUTIONS_REL = "paper5/execution_postcheck_resolutions.json"


def postcheck_resolutions(root=ROOT):
    """Human-investigated outcomes whose post-request check failed (append-only journal rows
    are never edited). Only the decision "exclude-and-rerun" exists: the named row is kept in
    the journal as evidence but excluded, so its job is requested again. Keyed by job id and
    the SHA-256 of the row's canonical JSON."""
    path = Path(root) / RESOLUTIONS_REL
    if not path.exists():
        return {}
    record = read_json(path)
    require(record.get("schema") == "p5-postcheck-resolutions-v1", "unknown postcheck-resolution schema")
    out = {}
    for item in record["resolutions"]:
        require(item["decision"] == "exclude-and-rerun", "unknown postcheck resolution decision")
        out[(item["job_id"], item["row_sha256"])] = item
    return out


def journal_state(records, jobs, *, stage, harness, count_tokens, legacy_ids=(), check_binding=None, root=ROOT):
    """Validate every completed and interrupted attempt before any cache reuse.

    The only unwrapped observations allowed are the initial registered smoke
    prefix. New rows bind their complete job intent, including timeout and
    token-parity settings, to separately checked execution receipts.
    """
    grouped = defaultdict(list)
    done, pending = {}, {}
    legacy_terminal_ids = []
    wrapped_seen = False
    endpoint = "/v1/chat/completions" if stage == "judge-gptoss" else "/v1/completions"
    resolutions = postcheck_resolutions(root)
    excluded = []
    for row in records:
        jid = row.get("job_id")
        key = (jid, hashlib.sha256(canonical(row).encode()).hexdigest())
        if key in resolutions:
            # Only an answered, wrapped, terminal row whose post-request check failed can be excluded.
            require("execution" in row and row.get("terminal") is True and row.get("ok") is True and
                    row["execution"].get("postcheck_ok") is False, "a resolution may only exclude a failed-postcheck outcome")
            require(jid not in done, "an excluded row cannot follow its job's terminal outcome")
            excluded.append(jid)
            continue
        require(jid in jobs, "journal contains an unexpected job")
        require(jid not in done, "journal row follows terminal outcome")
        require(type(row.get("terminal")) is bool, "explicit terminal flag required")
        wrapped = "execution" in row
        if wrapped:
            wrapped_seen = True
            require(row["execution"].get("intent_sha256") == intent(jobs[jid], endpoint), "cached request intent differs")
            require(check_binding is not None, "execution receipt verifier is required")
            check_binding(row)
        else:
            require(not wrapped_seen and stage == "generate" and jid in legacy_ids, "unbound outcome outside legacy smoke prefix")
        group = grouped[jid]
        require(not group or ("execution" in group[0]) == wrapped, "one job crosses legacy/wrapped attempt histories")
        if row["terminal"]:
            P.journal_attempts(group + [row])
            P.validate_row(row, spec(jobs[jid]), harness, stage=stage, count_tokens=count_tokens)
            done[jid] = row
            if not wrapped:
                legacy_terminal_ids.append(jid)
        else:
            # Validate a partial error sequence using a hypothetical terminal;
            # this record is never written or counted as an observed outcome.
            errors = group + [row]
            require(len(errors) <= 3, "too many recorded attempts")
            hypothetical = {"job_id": jid, "terminal": True, "ok": len(errors) < 3,
                            "attempt": len(errors), "error": row.get("error")}
            P.journal_attempts(errors + [hypothetical])
        group.append(row)
    require(legacy_terminal_ids == list(legacy_ids)[:len(legacy_terminal_ids)], "legacy smoke job ordering changed")
    for jid, group in grouped.items():
        if jid not in done:
            require("execution" in group[0], "unfinished unbound smoke request needs explicit investigation; not auto-retried")
            pending[jid] = group
    return {"done": done, "pending": pending, "legacy_terminal_ids": legacy_terminal_ids, "excluded": excluded}
