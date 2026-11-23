"""Read-only primary P5 request audit, outside the running, frozen harness.

This reconstructs the request information the legacy journals actually record.
It does not infer GPU placement, model-file hashes, MTP state or endpoint
ownership from a model name. Those require separate execution evidence.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import diagnostics as D
from draft_support import integer, number, read_json, sha

RUNNER_FILES = {"p5common.py", "p5run.py", "summarize.py", "build_prompts.py", "run_grid.py", "judge.py", "tabby-p5.yml"}
RAW = {"add_bos_token": False, "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]}
MODEL_ID = "qwen38-exl3-2.0"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def alignment_rows(old, new, receipt):
    """Independent readback: only one current-date line may differ per item."""
    before = {p["question_id"]: p for p in old}
    after = {p["question_id"]: p for p in new}
    D.require(len(old) == len(before) == receipt["original_fit_items"] and
              len(new) == len(after) == receipt["aligned_fit_items"], "oracle alignment inventory differs")
    D.require(before.keys() == after.keys() and receipt["fit_subset_changed"] is False,
              "oracle fit cohort changed during timestamp alignment")
    for qid, first in before.items():
        second = after[qid]
        D.require(first.keys() == second.keys(), "oracle alignment changed record fields")
        changed = {key for key in first if not same(first[key], second[key])}
        D.require(changed <= {"prompt", "prompt_sha256", "n_prompt"}, "oracle alignment changed evidence metadata")
        lines_old, lines_new = first["prompt"].splitlines(keepends=True), second["prompt"].splitlines(keepends=True)
        D.require(len(lines_old) == len(lines_new), "oracle alignment changed prompt structure")
        changed_lines = [(left, right) for left, right in zip(lines_old, lines_new) if left != right]
        D.require(len(changed_lines) == 1 and all(line.startswith("Current Date: ") for line in changed_lines[0]),
                  "oracle alignment changed more than the current-date line")
        for row in (first, second):
            D.require(digest(row["prompt"]) == row["prompt_sha256"], "oracle alignment prompt hash differs")
    D.require(bool(new) and receipt["aligned_token_range"] == [min(p["n_prompt"] for p in new), max(p["n_prompt"] for p in new)],
              "oracle alignment token range differs")


def oracle_alignment(root, model_dir):
    """Bind the preserved prospective correction; never rebuild real prompts."""
    from align_oracle_dates import DATA_PINS
    from evidence import rows
    root = Path(root)
    work = root / "testsuite/paper5/work"
    archive = work / "oracle-date-alignment-20260920"
    path = archive / "ALIGNMENT.json"
    receipt = read_json(path)
    D.require(receipt["schema"] == "p5-oracle-date-alignment-v1" and receipt["inference_started"] is False,
              "missing prospective oracle alignment record")
    D.require(receipt["source_items"] == receipt["source_timestamp_mismatches"] == 500,
              "oracle alignment source coverage differs")
    sources = {"paper5/scripts/align_oracle_dates.py", "testsuite/paper5/p5common.py", "testsuite/paper5/build_prompts.py",
               "testsuite/paper5_followups/legacy.py", "research/design-overflow-policies.md"}
    D.require(receipt["source_sha256"].keys() == sources, "oracle alignment source archive inventory differs")
    files = {str(path): sha(path)}
    for name, expected in receipt["source_sha256"].items():
        saved = archive / "sources" / name
        D.require(sha(saved) == expected, "archived oracle correction source changed")
        files[str(saved)] = expected
    inputs = {str(root / "testsuite/paper5_data/longmemeval" / name): expected for name, expected in DATA_PINS.items()}
    tokenizer = str(Path(model_dir) / "tokenizer.json")
    inputs[tokenizer] = sha(tokenizer)
    D.require(receipt["inputs"] == inputs, "oracle alignment data/tokenizer pins differ")
    files.update(inputs)
    old_path, new_path = archive / "original-O_4096.jsonl", work / "prompts/O_4096.jsonl"
    files[str(old_path)] = receipt["original_prompt_sha256"]
    files[str(new_path)] = receipt["aligned_prompt_sha256"]
    D.require(all(sha(p) == expected for p, expected in files.items()), "oracle alignment artifact changed")
    alignment_rows(list(rows(old_path)), list(rows(new_path)), receipt)
    return {"schema": "p5-oracle-alignment-readback-v1", "fit_items": receipt["aligned_fit_items"],
            "scope": "preserved original, date-only prompt difference, unchanged fit cohort and pinned source artifacts",
            "files": files}


def generation_spec(prompt, cap):
    qid, arm, budget = prompt["question_id"], prompt["arm"], prompt["budget"]
    return {"job_id": f"{qid}|{arm}|{budget}", "prompt_sha256": digest(prompt["prompt"]),
            "n_prompt": prompt["n_prompt"], "budget": budget,
            "params": {"max_tokens": cap, "temperature": .2, "top_p": .95, **RAW},
            "meta": {"question_id": qid, "arm": arm, "budget_label": budget,
                     "seed_recorded_not_applied": int(digest(f"{qid}|{arm}|{budget}")[:8], 16),
                     "prompt_file_sha256": prompt["prompt_sha256"]}}


def judgment_spec(item, generation, judge, algorithms, retry=False):
    # Reuse the hash-bound upstream template, but derive its input again from
    # the retained generation, not from stored judgment metadata or a label.
    answer, _ = D.visible(generation.get("text") if generation["ok"] else None)
    content = algorithms.judge.judge_prompt(item, answer)
    base = generation["job_id"] + "|judge=" + judge
    spec = {"job_id": base + ("|retry" if retry else ""), "budget": None,
            "meta": {"question_id": item["question_id"], "gen_job_id": generation["job_id"], "judge": judge}}
    if judge == "gptoss":
        # This exact default JSON serialization is the legacy runner's
        # messages hash (not its wire-body serialization).
        spec.update(prompt_sha256=digest(json.dumps([{"role": "user", "content": content}])), n_prompt=None,
                    params={"max_tokens": 8192 if retry else 2048, "temperature": 0.,
                            "chat_template_kwargs": {"reasoning_effort": "low"}})
    elif judge == "self" and not retry:
        prompt = algorithms.C.wrap(content, thinking=False)
        spec.update(prompt_sha256=digest(prompt), n_prompt=algorithms.C.count_tokens(prompt),
                    params={"max_tokens": 16, "temperature": 0., "top_k": 1, **RAW})
    else:
        raise ValueError("unsupported judge or retry")
    return spec


def journal_attempts(records):
    """Validate ordered retries, rather than letting the last terminal row win."""
    pending, terminals = {}, {}
    for row in records:
        jid = row.get("job_id")
        D.require(isinstance(jid, str) and jid and jid not in terminals, "missing job ID or record after terminal outcome")
        D.require(type(row.get("terminal")) is bool, "explicit terminal flag required")
        errors = pending.setdefault(jid, [])
        if not row["terminal"]:
            integer(row.get("attempt"), "retry attempt", 0, 2)
            D.require(row["attempt"] == len(errors), "retry sequence changed or an attempt is missing")
            D.require(isinstance(row.get("error"), str) and row["error"], "failed attempt needs its recorded error")
            number(row.get("seconds"), "failed request duration", 0)
            errors.append(row["error"])
            continue
        D.require(type(row.get("ok")) is bool, "explicit success/failure required")
        if row["ok"]:
            integer(row.get("attempt"), "successful attempt", 0, 2)
            D.require(row["attempt"] == len(errors), "successful attempt does not match retry history")
        else:
            D.require(len(errors) == 3 and row.get("error") == errors[-1], "terminal failure lacks three exhausted attempts")
        terminals[jid] = row
    D.require(pending.keys() == terminals.keys(), "journal contains an unfinished request")
    return terminals


def validate_row(row, expected, harness, *, stage, count_tokens):
    D.require(row.get("terminal") is True and type(row.get("ok")) is bool, "not a terminal request outcome")
    for key in ("job_id", "n_prompt", "budget", "meta"):
        D.require(same(row.get(key), expected[key]), "saved request differs from intended " + key)
    D.require(same(row.get("harness"), harness), "request came from a different harness fingerprint")
    number(row.get("request_idle_seconds"), "recorded idle setting", 0)
    stamp = row.get("ts")
    D.require(isinstance(stamp, str), "request timestamp is missing")
    D.require(datetime.fromisoformat(stamp).utcoffset() is not None, "request timestamp lacks timezone")
    if not row["ok"]:
        D.require(isinstance(row.get("error"), str) and bool(row["error"]), "terminal failure needs a recorded error")
        return
    for key in ("params", "prompt_sha256"):
        D.require(same(row.get(key), expected[key]), "saved request differs from intended " + key)
    number(row.get("seconds"), "successful request duration", 0)
    number(row.get("idle_wait_seconds"), "recorded idle wait", 0)
    text = row.get("text")
    D.require(text is None or isinstance(text, str), "invalid completion text")
    D.require(row.get("reasoning") is None or isinstance(row["reasoning"], str), "invalid reasoning field")
    D.require(row.get("finish_reason") in ("stop", "length"), "unknown completion termination")
    usage = row.get("usage")
    D.require(isinstance(usage, dict), "missing response usage object")
    if usage.get("prompt_tokens") is not None:
        integer(usage["prompt_tokens"], "server prompt usage", 0)
        if expected["n_prompt"] is not None:
            D.require(usage["prompt_tokens"] == expected["n_prompt"], "server usage/token parity differs")
    if usage.get("completion_tokens") is not None:
        integer(usage["completion_tokens"], "server completion usage", 0, expected["params"]["max_tokens"])
    server = row.get("server") or {}
    integer(server.get("pid"), "reported server PID", 1)
    if stage == "judge-gptoss":
        D.require(server.get("server") == "judge-gptoss20b" and server.get("port") == 18126,
                  "wrong reported primary-judge endpoint")
        D.require(row.get("server_prompt_tokens") is None, "unexpected gptoss token-parity claim")
        # The legacy runner counts final gptoss text with the 27B tokenizer.
        # This value can be reproduced but is NOT gptoss's completion length.
    else:
        D.require(server.get("server") == "tabby-p5" and server.get("port") == 18125
                  and server.get("model_id") == MODEL_ID and server.get("max_seq_len") == 32768
                  and not server.get("model_error"), "wrong or incomplete reported generator endpoint")
        if stage == "generate":
            integer(row.get("server_prompt_tokens"), "server tokenizer readback", 1)
            D.require(row["server_prompt_tokens"] == expected["n_prompt"], "generator token parity differs")
        else:
            # The old self-judge CLI does not request a tokenizer readback.
            D.require(row.get("server_prompt_tokens") in (None, expected["n_prompt"]), "self-judge token parity differs")
    integer(row.get("completion_tokens_harness"), "harness completion count", 0)
    D.require(row["completion_tokens_harness"] == (count_tokens(text) if text else 0), "completion count differs from recorded tokenizer")


def audit(prompts, generation_records, judge_records, items, cap, manifest, algorithms):
    """Pure audit with explicit data/algorithm arguments; no serving calls."""
    D.require(type(cap) is int and cap in (8192, 16384), "invalid pilot-selected completion cap")
    by_id = {item["question_id"]: item for item in items}
    D.require(len(by_id) == len(items), "duplicate source question ID")
    generations = journal_attempts(generation_records)
    judges = {name: journal_attempts(records) for name, records in judge_records.items()}
    expected = {p["job_id"]: p for p in prompts}
    D.require(len(expected) == len(prompts) and expected.keys() == generations.keys(), "incomplete or duplicate prompt/generation inventory")
    D.require(set(judges) == {"gptoss", "self"}, "both primary judge passes are required")
    harness = {name: manifest["harness"][name] for name in RUNNER_FILES}
    for p in prompts:
        item = by_id.get(p["question_id"])
        D.require(item is not None and p.get("question_type") == item["question_type"], "prompt question/type differs from frozen data")
        D.require(p["arm"] in ("T", "S", "C", "M", "Cw", "Mbm25", "O", "E"), "unknown primary arm")
        D.require((p["arm"] == "E" and p["budget"] is None) or
                  (p["arm"] != "E" and type(p["budget"]) is int and p["budget"] in (1024, 4096)), "wrong arm/budget")
        D.require(isinstance(p["prompt"], str) and digest(p["prompt"]) == p["prompt_sha256"], "prompt bytes/hash differ")
        integer(p["n_prompt"], "harness prompt count", 1)
        D.require(algorithms.C.count_tokens(p["prompt"]) == p["n_prompt"], "stored prompt count is not reproducible")
        D.require(p["budget"] is None or p["n_prompt"] <= p["budget"], "prompt exceeds its assigned budget")
        prefix, suffix = algorithms.C.WRAP_THINK
        D.require(p["prompt"].startswith(prefix) and p["prompt"].endswith(suffix), "wrong primary thinking wrapper")
        body = p["prompt"][len(prefix):-len(suffix)]
        D.require(body.endswith(algorithms.C.empty_prompt(item["question_date"], item["question"])),
                  "prompt does not end with the intended question and date")
        spec = generation_spec(p, cap)
        D.require(spec["job_id"] == p["job_id"], "primary job ID differs from arm/item/budget")
        validate_row(generations[p["job_id"]], spec, harness, stage="generate", count_tokens=algorithms.C.count_tokens)
    for name, records in judges.items():
        # Also checks required bases/retries and forbids an unrelated retry
        # replacing a valid verdict. Failed generations may still be judged.
        D.verdicts(generations, records, name)
        for row in records.values():
            gid = (row.get("meta") or {}).get("gen_job_id")
            D.require(gid in expected, "judge refers to a foreign generation")
            item = by_id[expected[gid]["question_id"]]
            spec = judgment_spec(item, generations[gid], name, algorithms, row["job_id"].endswith("|retry"))
            validate_row(row, spec, harness, stage="judge-" + name, count_tokens=algorithms.C.count_tokens)
    pacing = Counter(str(row["request_idle_seconds"]) for row in generations.values())
    return {"generations": generations, "judges": judges,
            "report": {"schema": "p5-primary-request-audit-v1", "generation_outcomes": len(generations),
                       "judge_outcomes": {k: len(v) for k, v in judges.items()}, "generation_idle_settings": dict(pacing),
                       "verified": "recorded requests, source question/type, tokenizer counts, reported endpoints, retries and judge input reconstruction",
                       "not_established_by_raw_rows": "GPU identity/placement, actual loaded model-file hashes, MTP state, or exclusive endpoint ownership",
                       "gptoss_legacy_completion_count": "27B-tokenizer count of final text; not gptoss completion usage"}}


def with_tokenizer(prompts, generation_records, judge_records, items, cap, manifest, algorithms, root):
    """Use one bounded CPU worker batch; never import model/runtime packages here."""
    from tokenizer_counts import TokenCounts
    generations = journal_attempts(generation_records)
    judges = {name: journal_attempts(records) for name, records in judge_records.items()}
    by_id = {item["question_id"]: item for item in items}
    texts = [p["prompt"] for p in prompts]
    for records in (generations, *judges.values()):
        texts.extend(row["text"] for row in records.values() if row["ok"] and row.get("text"))
    for row in judges["self"].values():
        gid = row["meta"]["gen_job_id"]
        gen = generations[gid]
        item = by_id[gen["meta"]["question_id"]]
        answer, _ = D.visible(gen.get("text") if gen["ok"] else None)
        texts.append(algorithms.C.wrap(algorithms.judge.judge_prompt(item, answer), thinking=False))
    counter = TokenCounts(Path(algorithms.C.MODEL_DIR) / "tokenizer.json", Path(root) / "engines/exl3-env/bin/python")
    proof = counter.prepare(texts)
    # Do not patch the imported experiment module. Give the verifier an
    # explicit read-only facade using the separately verified token counts.
    bridge = SimpleNamespace(C=SimpleNamespace(count_tokens=counter, wrap=algorithms.C.wrap,
                             WRAP_THINK=algorithms.C.WRAP_THINK, empty_prompt=algorithms.C.empty_prompt), judge=algorithms.judge)
    result = audit(prompts, generation_records, judge_records, items, cap, manifest, bridge)
    result["tokenizer"] = proof
    result["count_tokens"] = counter  # in-memory reuse by the execution audit; never serialized into evidence
    return result
