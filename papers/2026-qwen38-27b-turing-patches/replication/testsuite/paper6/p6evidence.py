"""Read-only P6 evidence gate. No server, tokenizer, download, or GPU is started.

Request reconstruction and long-context rescoring do not call the runner. GSM8K
keeps the registered Paper 1 scorer. SciPy cross-checks paired intervals, TOST,
and exact McNemar; integer arithmetic checks the seeded permutation calculation.
Replaying the original analyzer additionally checks every secondary table.

This is a local evidence gate, not a publication approval or proof of provenance
against a malicious experimenter. It checks recorded measurements, not new ones.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p6common as P

SOURCE_FILES = [f"testsuite/paper6/{name}" for name in (
    "p6common.py", "make_longctx.py", "run_cells.py", "analyze.py", "p6evidence.py")] + [
    "testsuite/paper5/p5run.py", "testsuite/paper5/p5common.py", "testsuite/evals/run_eval.py",
    "research/design-kv-allocation.md"]
BINARIES = ("llama-server", "libllama.so", "libllama-common.so", "libggml-cuda.so", "libggml-base.so", "libggml.so", "libggml-cpu.so")
EVENTS = {"pause", "resume", "gpu_after_first_request", "pass_end"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, low=0):
    require(type(value) is int and value >= low, f"{label}: invalid integer")
    return value


def number(value, label, low=None):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: invalid number")
    require(low is None or value >= low, f"{label}: negative number")
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA-256")
    return value


def loads(text):
    def pairs(rows):
        out = {}
        for k, v in rows:
            require(k not in out, f"duplicate JSON key: {k}")
            out[k] = v
        return out
    def bad(value):
        raise ValueError(f"non-finite JSON: {value}")
    def finite(value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else bad(value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad, parse_float=finite)


def read_json(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            require(bool(line.strip()), f"{path}:{line_no}: empty journal row")
            row = loads(line)
            require(isinstance(row, dict), f"{path}:{line_no}: expected object")
            result.append(row)
    return result


def harness_hashes():
    return {name: P.sha256_file(Path(P.REPO) / name) for name in SOURCE_FILES}


def input_identity():
    manifest = read_json(P.LONGCTX_MANIFEST)
    got = P.sha256_file(P.LONGCTX_ITEMS)
    require(got == manifest["output"]["sha256"], "frozen long-context file changed")
    require(P.sha256_file(P.GSM8K_ITEMS) == P.GSM8K_SHA256, "frozen GSM8K file changed")
    return {"longctx_sha256": got, "longctx_manifest_sha256": P.sha256_file(P.LONGCTX_MANIFEST),
            "longctx_seed": manifest["seed"], "gsm8k_sha256": P.GSM8K_SHA256}


def items_by_id(items):
    out = {}
    for item in items:
        key = item.get("id")
        require(isinstance(key, str) and key and "|" not in key and key not in out, "duplicate/invalid item ID")
        out[key] = item
    return out


def expected_request(model, task, item):
    content = item["content"] if task == "longctx" else P.gsm8k_prompt(item)
    body = {"messages": [{"role": "user", "content": content}], "stream": False,
            "temperature": 0.0, "cache_prompt": False,
            "max_tokens": 128 if task == "longctx" else 8192}
    if task == "longctx":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif model == P.HYBRID:
        body["chat_template_kwargs"] = {"reasoning_effort": "medium"}
    request_sha = P.sha256_text(json.dumps(body, sort_keys=True))
    if task == "longctx":
        body["messages"][0]["content"] = {"item_id": item["id"], "sha256": P.sha256_text(content), "chars": len(content)}
    return body, request_sha


def expected_score(task, row, item):
    content = row["content"] or ""
    if task == "gsm8k":
        visible = P.run_eval().visible_content(row)
        return {"correct": bool(P.run_eval().score_gsm8k(visible, item)), "visible_answer": visible[-400:]}
    if "</think>" in content:
        visible = content.split("</think>", 1)[1].strip()
    else:
        visible = "" if "<think>" in content else content.strip()
    hits = [x for x in item["answers"] if x.lower() in visible.lower()]
    return {"correct": len(hits) == len(item["answers"]), "visible_answer": visible[:2000],
            "answers_found": hits,
            "distractors_found": [x for x in item.get("distractors", []) if x.lower() in visible.lower()]}


def validate_load(load, cell):
    model, _, kv = cell
    require(load["n_ctx"] == 20480 and load["flash_attn"] == "enabled", "wrong context/flash-attention readback")
    kvc = load["kv_cache"]
    require((kvc["type_k"], kvc["type_v"]) == P.KV_TYPES[kv], "wrong actual KV types")
    require(kvc["cells"] == 20480, "unexpected actual KV context allocation")
    require(kvc["layers"] == (16 if model == P.HYBRID else 36), "unexpected KV layer count")
    off = integer(load["offloaded_layers"], "offloaded layers")
    total = integer(load["total_layers"], "total layers", 1)
    require(off <= total, "invalid layer placement")
    build = load["build"] or {}
    # This revision prints the build line at trace verbosity, not normal
    # startup verbosity. If present it must agree; /props below is mandatory.
    if build:
        require(isinstance(build.get("commit"), str) and len(build["commit"]) >= 7
                and P.ENGINE_COMMIT.startswith(build["commit"]), "wrong logged engine build")
    require(load["fit"]["status"] == "success", "engine --fit did not succeed")
    for kind, buffers in load["buffers_mib"].items():
        for amount in buffers.values():
            number(amount, f"{kind} buffer MiB", 0)
    require(load["device_mib"]["model"] > 0 and load["device_mib"]["kv"] > 0, "GPU buffer readback missing")
    resident = (off == total and load["host_mib"]["kv"] == 0 and load["host_mib"]["rs"] == 0
                and not load["fit"]["overflowing"])
    require(load["fully_resident"] is resident, "incorrect fully-resident classification")
    if model == P.HYBRID:
        rs = load["recurrent"] or {}
        require(rs.get("type_r") == rs.get("type_s") == "f32", "hybrid recurrent state is not F32")


def validate_identity(row, cell, *, hashes=None, inputs=None):
    # Log parsing is shared with capture; protocol/readback checks below are separate.
    import run_cells as R
    model, tier, kv = cell
    require(row["cell"] == {"model": model, "tier": tier, "kv": kv,
            "ctk": P.KV_TYPES[kv][0], "ctv": P.KV_TYPES[kv][1], "arch": P.MODELS[model]["arch"]}, "foreign cell identity")
    gid, eng = row["gguf"], row["engine"]
    machine = row["machine"]
    require(machine["gpu_name"] == "NVIDIA GeForce RTX 4080 Laptop GPU", "wrong study hardware")
    require(isinstance(machine["gpu_uuid"], str) and machine["gpu_uuid"].startswith("GPU-"), "missing GPU UUID")
    require(12000 <= integer(machine["gpu_total_mib"], "GPU memory") <= 12350, "wrong GPU memory capacity")
    require(isinstance(machine["driver_version"], str) and machine["driver_version"], "missing GPU driver version")
    path = P.gguf_path(model, tier)
    rec = P.receipt_for(path)
    require(rec is not None, "missing artifact receipt")
    require((rec["repo"], rec["revision"]) == (P.MODELS[model]["repo"], P.MODELS[model]["revision"]), "wrong receipt revision")
    pin = P.PINNED_SHA256.get(Path(path).name)
    require(pin is None or rec["sha256"] == pin[0], "receipt conflicts with pre-existing artifact pin")
    require(gid["path"] == path and gid["receipt"] == rec and gid["verified"] is True
            and digest(gid["sha256"]) == rec["sha256"], "unverified/wrong GGUF")
    integer(gid["bytes"], "GGUF bytes", 1)
    if rec["bytes"] is not None:
        require(gid["bytes"] == rec["bytes"], "wrong GGUF size")
    require(eng["commit"] == P.ENGINE_COMMIT and eng["commit_is_pinned"] is True
            and eng["tracked_changes"] == 0, "wrong/modified engine")
    flags = {"GGML_CUDA_FA_ALL_QUANTS": "ON", "GGML_CUDA": "ON", "BUILD_SHARED_LIBS": "ON",
             "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "89"}
    require(all(eng["cmake"].get(k) == v for k, v in flags.items()), "wrong CUDA build flags")
    require(set(eng["sha256"]) == set(BINARIES), "missing engine binary/library hashes")
    for value in eng["sha256"].values():
        digest(value)
    argv = [P.SERVER_EXE, "-m", path, "-c", "20480", "--fit", "on", "-fa", "on", "-ctk", P.KV_TYPES[kv][0],
            "-ctv", P.KV_TYPES[kv][1], "-np", "1", "--jinja", "--no-webui", "--host", "localhost", "--port", "18130"]
    require(row["argv"] == argv and row["cmd"] == ["nice", "-n", "5"] + argv, "unexpected server command")
    require(row["harness"] == (hashes if hashes is not None else harness_hashes()), "stale/mixed harness")
    require(row["inputs"] == (inputs if inputs is not None else input_identity()), "stale task inputs")
    require(row["placement_changed"] is False, "placement changed across restarts")
    require(re.fullmatch(r"[0-9a-f]{32}", row["instance_id"]) is not None, "missing server instance ID")
    integer(row["pid"], "server PID", 1)
    integer(row["start_seq"], "server start sequence", 1)
    decode = row["decoding"]
    require(type(decode["temperature"]) in (int, float) and decode["temperature"] == 0
            and decode["cache_prompt"] is False and decode["longctx_max_tokens"] == 128
            and decode["gsm8k_max_tokens"] == 8192, "unexpected decode policy")
    for timeout in decode["timeouts_s"].values():
        require(number(timeout, "timeout", 0) > 0, "nonpositive timeout")
    props = row["props"]
    running_build = re.fullmatch(r"b(\d+)-([0-9a-f]{7,40})", props.get("build_info") or "")
    require(running_build is not None and P.ENGINE_COMMIT.startswith(running_build.group(2)),
            "wrong or missing running engine build in props")
    if row["load"].get("build"):
        require(row["load"]["build"]["number"] == int(running_build.group(1)), "logged/served engine build numbers differ")
    require(props["total_slots"] == 1 and props["n_ctx"] == 20480 and props["model_path"] == path,
            "server props disagree with launch")
    digest(props["chat_template_sha256"])
    integer(props["chat_template_chars"], "template chars", 1)
    number(row["gpu_loaded"]["process_mib"], "measured GPU process MiB", 1)
    log_path = (Path(P.REPO) / row["log_path"]).resolve()
    require(log_path.is_relative_to(Path(P.LOGS).resolve()), "server log outside P6 logs")
    with log_path.open("rb") as stream:
        raw = stream.read(integer(row["load_log_bytes"], "load log bytes", 1))
    require(len(raw) == row["load_log_bytes"] and hashlib.sha256(raw).hexdigest() == row["load_log_sha256"],
            "server load log changed or truncated")
    require(R.parse_server_log(raw.decode("utf-8", errors="replace")) == row["load"], "load readback differs from raw log")
    validate_load(row["load"], cell)


def audit_journal(path, cell, longctx, gsm8k, *, complete=True, hashes=None, inputs=None):
    """One successful outcome per item; retries may precede it but never replace it."""
    import run_cells as R
    items = {"longctx": items_by_id(longctx), "gsm8k": items_by_id(gsm8k)}
    wanted = {f"{task}|{key}" for task, rows in items.items() for key in rows}
    hashes = harness_hashes() if hashes is None else hashes
    inputs = input_identity() if inputs is None else inputs
    outcomes, starts = {}, []
    current = None
    first_placement = None
    if not Path(path).exists():
        require(not complete, f"missing journal: {path}")
        return {"outcomes": outcomes, "identity": starts}
    for line_no, row in enumerate(read_jsonl(path), 1):
        label = f"{Path(path).name}:{line_no}"
        kind = row.get("kind")
        if kind:
            require("job_id" not in row, f"{label}: event masquerading as result")
            if kind == "server_start":
                validate_identity(row, cell, hashes=hashes, inputs=inputs)
                require(all(x["instance_id"] != row["instance_id"] for x in starts), "duplicate server instance")
                signature = R.placement_signature(row["load"])
                if first_placement is None:
                    first_placement = signature
                require(first_placement == signature, "unreported placement change across restarts")
                current = row
                starts.append(row)
            else:
                require(kind in EVENTS, f"{label}: unknown event {kind}")
                if kind in ("pause", "pass_end"):
                    current = None
            continue
        jid = row.get("job_id")
        require(jid in wanted, f"{label}: foreign job ID")
        task, key = jid.split("|", 1)
        require(row.get("task") == task and row.get("item_id") == key, f"{label}: inconsistent job ID")
        require(jid not in outcomes, f"{label}: repeated attempt after successful outcome")
        require(current is not None, f"{label}: request without an active server identity")
        require(type(row.get("terminal")) is bool, f"{label}: terminal must be boolean")
        integer(row.get("attempt"), "attempt")
        if not row["terminal"]:
            require(isinstance(row.get("error"), str) and row["error"], "nonterminal row is not an error")
            continue
        require(type(row.get("ok")) is bool, "ok must be boolean")
        request, request_sha = expected_request(cell[0], task, items[task][key])
        # Canonical JSON prevents Python's True == 1 from accepting changed request values.
        require(json.dumps(row["request"], sort_keys=True) == json.dumps(request, sort_keys=True), "request differs from frozen policy")
        require(row["harness"] == hashes, "response uses stale/mixed harness")
        if not row["ok"]:
            require(isinstance(row.get("error"), str) and row["error"], "failed row has no error")
            continue
        require(row["request_sha256"] == request_sha, "request hash mismatch")
        require(row["server"] == {k: current[k] for k in ("pid", "start_seq", "instance_id")}, "response is bound to another server")
        for k in ("content", "reasoning_content"):
            require(row[k] is None or isinstance(row[k], str), f"malformed response {k}")
        require(row["finish_reason"] in ("stop", "length"), "unexpected finish reason")
        number(row["seconds"], "request seconds", 0)
        usage = row["usage"]
        prompt = integer(usage["prompt_tokens"], "prompt tokens", 1)
        completion = integer(usage["completion_tokens"], "completion tokens")
        require(completion <= request["max_tokens"], "completion exceeds declared cap")
        require(prompt + completion <= 20480, "request exceeded fixed context capacity")
        if "total_tokens" in usage:
            require(type(usage["total_tokens"]) is int and usage["total_tokens"] == prompt + completion, "inconsistent token usage")
        meta = {}
        if task == "longctx":
            it = items[task][key]
            expected = it["prompt_tokens_qwen38_nothink"] if cell[0] == P.HYBRID else None
            meta = {"family": it["family"], "content_sha256": it["sha256"],
                    "haystack_tokens_qwen38": it["haystack_tokens"], "prompt_tokens_expected": expected}
            if expected is not None:
                meta["prompt_tokens_delta"] = prompt - expected
        require(row["meta"] == meta, "item metadata/prompt-token delta mismatch")
        score = expected_score(task, row, items[task][key])
        require(type(row["score"].get("correct")) is bool and row["score"] == score, "stored score differs from independent rescore")
        outcomes[jid] = row
    require(not complete or set(outcomes) == wanted, f"{path}: {len(wanted - outcomes.keys())} successful outcomes missing")
    return {"outcomes": outcomes, "identity": starts}


def validate_token_counts(data, model, items):
    require(data["schema"] == "p6-token-counts-v1" and data["model"] == model, "wrong token-count identity")
    require(data["items_sha256"] == P.sha256_file(P.LONGCTX_ITEMS), "token counts use different items")
    require(data["harness"] == harness_hashes(), "stale token-count harness")
    require(set(data["items"]) == set(items_by_id(items)), "incomplete/foreign token-count items")
    for row in data["items"].values():
        require(set(row) == {"haystack", "content"}, "unknown token-count fields")
        integer(row["haystack"], "haystack count", 1)
        require(integer(row["content"], "content count", 1) >= row["haystack"], "content shorter than haystack")


def evidence_snapshot(journal_dir):
    """Small inference/analysis inputs only; never re-read model weights or start a service."""
    root = Path(journal_dir).parent
    paths = list(Path(journal_dir).glob("*.jsonl")) + list((root / "token_counts").glob("*.json"))
    result = {str(p.relative_to(root)): P.sha256_file(p) for p in sorted(paths)}
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        for row in read_jsonl(path):
            if row.get("kind") != "server_start" or "load_log_bytes" not in row:
                continue  # historical synthetic fixtures; full evidence audit still requires these fields
            log_path = (Path(P.REPO) / row["log_path"]).resolve()
            require(log_path.is_relative_to(Path(P.LOGS).resolve()), "server log outside P6 logs")
            with log_path.open("rb") as stream:
                raw = stream.read(integer(row["load_log_bytes"], "load log bytes", 1))
            result[f"load_log_prefix/{path.stem}/{row['instance_id']}"] = hashlib.sha256(raw).hexdigest()
    return result


def compare(actual, expected, where="analysis"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{where}: fields differ")
        for key, value in expected.items():
            compare(actual[key], value, f"{where}/{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{where}: rows differ")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{where}/{i}")
    elif type(expected) in (float, int):
        number(actual, where)
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), f"{where}: numeric mismatch")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{where}: value differs")


def independent_statistics(a, audits):
    """Separate formulas/library; permutation uses integer sums, not analyzer's float BLAS."""
    import numpy as np
    import scipy
    from scipy import stats
    for task in ("longctx", "gsm8k"):
        # Align with frozen item order, including journals resumed in a different order.
        source = read_jsonl(P.LONGCTX_ITEMS if task == "longctx" else P.GSM8K_ITEMS)
        ids = [f"{task}|{it['id']}" for it in source]
        def y(model, tier, kv):
            return np.array([int(audits[P.cell_name(model, tier, kv)]["outcomes"][key]["score"]["correct"])
                             for key in ids], dtype=np.int64)
        def check_ci(record, v):
            n = len(v)
            mean = float(np.sum(v)) / n
            # Sum of squares, independently of statistics.stdev in the analyzer.
            sd = math.sqrt(max(0., (float(np.sum(v * v)) - n * mean * mean) / (n - 1)))
            se = sd / math.sqrt(n)
            q = float(stats.t.ppf(.975, n - 1))
            for k, expected in {"n": n, "mean_pts": mean * 100, "sd_pts": sd * 100, "se_pts": se * 100,
                                "lo_pts": (mean - q * se) * 100, "hi_pts": (mean + q * se) * 100}.items():
                compare(record[k], expected, "independent paired " + k)
            return mean, se
        def check_permutation(record, v):
            rng = np.random.default_rng(20260919)
            ge = 0
            for _ in range(20):
                bits = rng.integers(0, 2, size=(5000, len(v)), dtype=np.int8)
                sums = np.sum((bits.astype(np.int64) * 2 - 1) * v, axis=1)
                ge += int(np.count_nonzero(np.abs(sums) >= abs(int(np.sum(v)))))
            compare(record["n_ge"], ge, "independent sign-flip count")
            compare(record["p"], (ge + 1) / 100001, "independent sign-flip p")
        interactions = {}
        est = a["estimands"][task]
        for model in P.MODELS:
            ds = {tier: y(model, tier, "f16") - y(model, tier, "q4_0") for tier in P.TIERS}
            for tier, values in ds.items():
                check_ci(est["D"][model][tier], values)
            iv = ds["Q2_K_XL"] - ds["Q4_K_XL"]
            interactions[model] = iv
            mean, se = check_ci(est["I"][model], iv)
            check_permutation(est["I"][model]["permutation"], iv)
            if se:
                lo_p = float(stats.t.sf((mean + .1) / se, len(iv) - 1))
                hi_p = float(stats.t.cdf((mean - .1) / se, len(iv) - 1))
            else:
                lo_p, hi_p = float(mean <= -.1), float(mean >= .1)
            tost = est["I"][model]["tost"]
            compare(tost["p_tost"], max(lo_p, hi_p), "independent TOST p")
            half = float(stats.t.ppf(.95, len(iv) - 1)) * se
            compare(tost["ci90_lo_pts"], (mean - half) * 100, "independent TOST lower")
            compare(tost["ci90_hi_pts"], (mean + half) * 100, "independent TOST upper")
        mv = interactions[P.HYBRID] - interactions[P.GQA]
        check_ci(est["M"], mv)
        check_permutation(est["M"]["permutation"], mv)
        for row in a["secondary"][task]["kv_vs_f16"]:
            ref, other = y(row["model"], row["tier"], "f16"), y(row["model"], row["tier"], row["kv"])
            lost = int(np.count_nonzero((ref == 1) & (other == 0)))
            gained = int(np.count_nonzero((ref == 0) & (other == 1)))
            compare(row["lost"], lost, "independent McNemar lost")
            compare(row["gained"], gained, "independent McNemar gained")
            p = float(stats.binomtest(lost, lost + gained, .5).pvalue) if lost + gained else 1.
            compare(row["mcnemar_p"], p, "independent exact McNemar")
    return {"scipy": scipy.__version__, "numpy": np.__version__}


def verify_analysis(a):
    """Require complete canonical evidence, rescore it, cross-check it, and replay all tables."""
    import analyze as A
    i = a["inputs"]
    require(Path(i["journal_dir"]).resolve() == Path(P.JOURNALS).resolve()
            and Path(i["items"]).resolve() == Path(P.LONGCTX_ITEMS).resolve(), "noncanonical analysis inputs")
    require((i["permutation_draws"], i["permutation_seed"], i["tost_delta_points"]) == (100000, 20260919, 10),
            "analysis used unregistered test settings")
    before = evidence_snapshot(P.JOURNALS)
    require(i["evidence_sha256"] == before, "analysis evidence has changed; recompute")
    inputs, hashes = input_identity(), harness_hashes()
    longctx, gsm8k = read_jsonl(P.LONGCTX_ITEMS), read_jsonl(P.GSM8K_ITEMS)
    require(len(items_by_id(longctx)) == 200 and len(items_by_id(gsm8k)) == 100, "incomplete item sets")
    require(Counter(it["family"] for it in longctx) == {x: 50 for x in ("single", "multikey", "multivalue", "vartrack")},
            "wrong long-context family counts")
    for it in longctx:
        require(P.sha256_text(it["content"]) == it["sha256"] and it["answers"], "invalid frozen item")
    expected = {P.cell_name(*cell) + ".jsonl" for cell in P.all_cells()}
    require({x.name for x in Path(P.JOURNALS).glob("*.jsonl")} == expected, "missing/foreign cell journals")
    audits = {P.cell_name(*cell): audit_journal(P.journal_path(*cell, journals=P.JOURNALS), cell, longctx, gsm8k,
              hashes=hashes, inputs=inputs) for cell in P.all_cells()}
    starts = [s for result in audits.values() for s in result["identity"]]
    require(len({s["instance_id"] for s in starts}) == len(starts), "server instance reused across cells")
    require(all(s["engine"] == starts[0]["engine"] for s in starts), "engine changed between cells")
    require(all(s["machine"] == starts[0]["machine"] for s in starts), "study GPU/driver changed between cells")
    for model in P.MODELS:
        data = read_json(Path(P.WORK) / "token_counts" / f"{model}.json")
        validate_token_counts(data, model, longctx)
        source = [s for s in audits[data["cell"]]["identity"] if s["instance_id"] == data["server"]["instance_id"]]
        require(len(source) == 1 and source[0]["cell"]["model"] == model and source[0]["props"] == data["props"],
                "tokenizer counts not bound to a recorded server")
        templates = {s["props"]["chat_template_sha256"] for s in starts if s["cell"]["model"] == model}
        require(len(templates) == 1, "chat template changed across weight/KV cells")
    replay = A.analyse(P.JOURNALS, P.LONGCTX_ITEMS, 100000, 20260919, 10.0)
    compare({k: v for k, v in a.items() if k != "ts"}, {k: v for k, v in replay.items() if k != "ts"})
    versions = independent_statistics(a, audits)
    require(before == evidence_snapshot(P.JOURNALS) and hashes == harness_hashes() and inputs == input_identity(),
            "evidence/code changed during verification")
    return {"schema": "p6-local-evidence-audit-v1", "cells": 24, "successful_outcomes": 7200,
            "independent_statistics": versions, "evidence_sha256": before,
            "human_approval": False}
