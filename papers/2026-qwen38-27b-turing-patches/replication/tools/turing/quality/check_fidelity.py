#!/usr/bin/env python3
"""Paper 8 quality harness: prove the Turing fork is comparable with Paper 7's E47 rows. Laptop, CPU only, no GPU,
no network (every server is faked in-process):
  CUDA_VISIBLE_DEVICES='' nice -n 15 python3 tools/turing/quality/check_fidelity.py [--scratch DIR] [--json OUT]

Checks (exit status 1 if any hard check fails):
  0. static: every function the fork shares with run_eval.py is source-identical; constants equal; the fork's and
     run_quality.sh's dataset sha256 tables equal the laptop's dataset files; run_quality.sh passes E47's flags.
  1. rescoring: every stored row of EFF-h38a_mtp and EFF-h38a_nodraft on gsm8k_500 and mmlu_500 is rescored from its
     stored text. The rows keep only the last 200 characters of the visible answer (raw_answer_excerpt), so:
       (a) the fork's scorer and run_eval.py's scorer agree on every excerpt (must be 100%);
       (b) where the excerpt is the whole visible answer (< 200 chars, or an error row), rescoring must reproduce
           the stored correctness exactly;
       (c) where the answer was longer, the agreement with the stored correctness is reported, with every
           disagreement listed (the scorer saw less text than at run time).
  2. requests and rows: for every item of both suites, with the CLI of E47 (hardening.sh) parsed by run_eval.py's own
     main() and the CLI that run_quality.sh actually passes parsed by the fork's main(), both programs send
     byte-identical requests (URL, body, headers, method, timeout) and build identical rows from identical server
     responses: TabbyAPI-style (usage null, /v1/token/encode), llama.cpp-style (usage, timings), inline <think>,
     chat_template_kwargs rejected with HTTP 400, one dropped connection, a hard failure, a failing encode route.
     Also with other flag sets (default temperature, reasoning effort off, other max tokens, an extra body file).
     The fork's request_body_bytes() equals the captured bytes, and its sidecar records the right token source.
  3. statistics: compare_quality.pair_stats() reproduces every stored hardening_analyze.py output
     (testsuite/results/efficiency/rows/h38*.json) and compare_quality.anatomy() every stored identity_anatomy.py
     output, from the laptop rows.
  4. resume: the fork's run_suite() against a fake server in a scratch dir: error rows retried, finished items skipped,
     the latest row per id wins, nothing is truncated, a torn last line is survived, the consecutive-error stop fires.
     The scratch dir is left in place (nothing here discards files).
  5. response shape: the same generated texts (E47 traces, rows/traces_h38c_nodraft.jsonl, plus one item truncated
     inside the reasoning and five one-word variants) are served in E47's shape (lab TabbyAPI: inline, usage null) to
     run_eval.py, and to the fork in upstream TabbyAPI b1c2253's two shapes: reasoning off (inline + usage) and its
     defaults (split by upstream's own TagStreamParser, loaded from the bare source mirror, fed in small chunks).
     compare_quality must report the same byte identity, matched token identity, mean tokens, anatomy and
     inline-path accuracy for both shapes, flag the split one with a CHECK line, and bucket the one-word variants'
     token deltas on the matched basis. Also run_quality.sh's e47_shape() overrides (in check 0).
  6. lane table: a partial rerun and a partial variant do not shrink the stock lanes' common item set.
"""
import argparse
import contextlib
import glob
import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN_EVAL = os.path.join(REPO, "testsuite", "evals", "run_eval.py")
DATASETS = os.path.join(REPO, "testsuite", "evals", "datasets")
LAPTOP_RESULTS = os.path.join(REPO, "testsuite", "evals", "results")
ROWS = os.path.join(REPO, "testsuite", "results", "efficiency", "rows")
TABBY_MIRROR = os.path.join(REPO, "engines", "src-mirror", "theroyallab__tabbyAPI.git")
UPSTREAM_TABBY = "b1c2253"     # the desktop's TabbyAPI (reasoning parser on by default since 2186cdb)
LABELS = ["EFF-h38a_mtp", "EFF-h38a_nodraft"]
SUITES = ["gsm8k_500", "mmlu_500"]
SHARED_FUNCS = ["base_suite", "build_prompt", "_to_float", "score_gsm8k", "score_mmlu", "score_needle",
                "extract_boxed", "_convert_fracs", "normalize_math", "_parse_number", "extract_math_answer",
                "score_math25", "chat_completion", "estimate_thinking_tokens", "visible_content"]
URL = "http://localhost:8090"
E47_ARGV = ["--temperature", "0"]          # hardening.sh evsuite: --suite S --label L --base-url U --temperature 0 --limit 0

FAILS = []
REPORT = {}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hard(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ fake server

class FakeResp:
    def __init__(self, obj):
        self.data = json.dumps(obj).encode("utf-8")

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(url, code):
    return urllib.error.HTTPError(url, code, f"fake {code}", {}, io.BytesIO(b"{}"))


class FakeServer:
    """Replaces urllib.request.urlopen; records every request; answers per scenario."""

    def __init__(self, scenario, answer, reasoning, fail_ids=()):
        self.scenario, self.answer, self.reasoning = scenario, answer, reasoning
        self.calls = []
        self.chat_n = 0
        self.fail_ids = set(fail_ids)

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append((url, req.data, tuple(sorted(req.header_items())), req.get_method(), timeout))
        if url.endswith("/v1/token/encode"):
            return self._encode(url, json.loads(req.data)["text"])
        if url.endswith("/v1/chat/completions"):
            self.chat_n += 1
            return FakeResp(self._chat(url, json.loads(req.data)))
        raise http_error(url, 404)

    def _encode(self, url, text):
        if self.scenario in ("llamacpp", "inline_think"):
            raise http_error(url, 404)
        if self.scenario == "encode_fail":
            raise urllib.error.URLError("encode route down")
        n = len(text.split()) + 3
        return FakeResp({"tokens": list(range(n)), "length": n})

    def _chat(self, url, body):
        sc = self.scenario
        if sc == "hard_fail":
            raise http_error(url, 503)
        if sc == "dead":
            raise urllib.error.URLError("connection refused")
        if sc == "ctk400" and "chat_template_kwargs" in body:
            raise http_error(url, 400)
        if sc == "conn_drop" and self.chat_n == 1:
            raise urllib.error.URLError("connection reset")
        if self.fail_ids and any(f"[{i}]" in body["messages"][0]["content"] for i in self.fail_ids):
            raise http_error(url, 503)
        msg = {"role": "assistant", "content": self.answer, "reasoning_content": self.reasoning}
        usage = None
        finish = "stop"
        if sc in ("llamacpp", "inline_think"):
            ntok = len((self.reasoning + self.answer).split())
            usage = {"prompt_tokens": 57, "completion_tokens": ntok, "total_tokens": ntok + 57}
        if sc == "inline_think":
            msg = {"role": "assistant", "content": "<think>" + self.reasoning + "</think>\n\n" + self.answer}
            finish = "length"
        resp = {"id": "x", "object": "chat.completion", "model": "fake", "choices": [
            {"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}
        if sc == "llamacpp":
            resp["timings"] = {"predicted_n": usage["completion_tokens"], "predicted_per_second": 24.5,
                               "prompt_per_second": 610.0, "draft_n": 10, "draft_n_accepted": 6}
        return resp


@contextlib.contextmanager
def patched(server):
    real_open, real_sleep = urllib.request.urlopen, time.sleep
    urllib.request.urlopen = server
    time.sleep = lambda s: None
    try:
        yield server
    finally:
        urllib.request.urlopen, time.sleep = real_open, real_sleep


# ------------------------------------------------------------------ args exactly as each program parses them

def parsed_args_run_eval(ref, argv):
    captured = []
    real_run_suite, real_argv = ref.run_suite, sys.argv
    ref.run_suite = lambda s, a: captured.append(a)
    sys.argv = ["run_eval.py"] + argv
    try:
        ref.main()
    finally:
        ref.run_suite, sys.argv = real_run_suite, real_argv
    return captured[0]


def parsed_args_fork(fork, argv):
    captured = []
    real_run_suite = fork.run_suite
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    fork.run_suite = lambda s, a: captured.append(a)
    try:
        fork.main(argv)
    finally:
        fork.run_suite = real_run_suite
        for s, h in old.items():
            signal.signal(s, h)
    return captured[0]


def run_quality_fork_argv(text, suite, scratch):
    """The run_eval_turing.py argv that run_quality.sh builds, with its shell variables filled in."""
    m = re.search(r'"\$PY" "\$HERE/run_eval_turing\.py"( --suite "\$suite" --label "\$label" --base-url .*?)&\n', text, re.S)
    if not m:
        return None
    cmd = m.group(1).replace("\\\n", " ")
    vals = {"suite": suite, "label": "TURING-check", "url": URL, "limit": "0", "RES": os.path.join(scratch, "res"),
            "DS": DATASETS, "summary_file": os.path.join(scratch, "summary.json")}
    for k, v in vals.items():
        cmd = cmd.replace("$" + k, v)
    return [t for t in shlex.split(cmd) if not re.match(r"^\d*[<>]", t)]   # drop fd redirections such as 9>&-


# ------------------------------------------------------------------ checks

def check_static(ref, fork, rq_text):
    print("0. static identity")
    same = [f for f in SHARED_FUNCS if inspect.getsource(getattr(ref, f)) == inspect.getsource(getattr(fork, f))]
    hard("shared functions source-identical", len(same) == len(SHARED_FUNCS),
         f"{len(same)}/{len(SHARED_FUNCS)}" + ("" if len(same) == len(SHARED_FUNCS) else
                                              f" differ: {sorted(set(SHARED_FUNCS) - set(same))}"))
    hard("REQUEST_TIMEOUT_S, NUM_RE, HARDENING_SUITES equal",
         ref.REQUEST_TIMEOUT_S == fork.REQUEST_TIMEOUT_S and ref.NUM_RE.pattern == fork.NUM_RE.pattern
         and ref.HARDENING_SUITES == fork.HARDENING_SUITES)
    kept = [s for s in ref.SUITES if not s.startswith("humaneval")]
    hard("fork suites = run_eval suites minus code execution", fork.SUITES == kept, f"{fork.SUITES}")
    hard("DATASET_FILES equal for every kept suite",
         all(ref.DATASET_FILES[s] == fork.DATASET_FILES[s] for s in fork.DATASET_FILES))
    hard("SCORERS keys = run_eval's minus code execution",
         sorted(fork.SCORERS) == sorted(k for k in ref.SCORERS if not k.startswith("humaneval")))
    actual = {name: sha256_file(os.path.join(DATASETS, name)) for name in fork.DATASET_SHA256}
    hard("fork DATASET_SHA256 = laptop dataset files", actual == fork.DATASET_SHA256)
    for s in SUITES:
        m = re.search(rf"^\s*{s}\)\s*want=([0-9a-f]{{64}})", rq_text, re.M)
        hard(f"run_quality.sh sha256 for {s} = laptop file", bool(m) and m.group(1) == actual[f"{s}.jsonl"])
    hard("run_quality.sh passes E47's flags",
         '--temperature 0 --reasoning-effort medium --max-tokens 4096 --concurrency 1 --limit "$limit"' in rq_text)
    hard("run_quality.sh applies e47_shape before the lane starts",
         rq_text.find('e47_shape "$lane"') != -1 and rq_text.find('e47_shape "$lane"') < rq_text.find('"$LANES" start'))
    # e47_shape(), run by bash: E47's response shape is added to the lane overrides unless the caller set that key
    m = re.search(r"^e47_shape\(\) \{.*?^\}\n", rq_text, re.S | re.M)
    cases = [  # lane, TURING_LANE_ENV, TURING_LCPP_EXTRA -> expected TURING_LANE_ENV, TURING_LCPP_EXTRA
        ("exl3-mtp", "", "", "TABBY_MODEL_REASONING=false TABBY_MODEL_TOOL_FORMAT=", ""),
        ("exl3-mtp", "TABBY_MODEL_RECURRENT_STATE_FP32=true", "",
         "TABBY_MODEL_REASONING=false TABBY_MODEL_TOOL_FORMAT= TABBY_MODEL_RECURRENT_STATE_FP32=true", ""),
        ("exl3-nodraft", "TABBY_MODEL_REASONING=true", "", "TABBY_MODEL_TOOL_FORMAT= TABBY_MODEL_REASONING=true", ""),
        ("lcpp-iq2s", "", "", "", "--reasoning-format none"),
        ("lcpp-iq2xxs-dflash2", "", "--temp 0", "", "--reasoning-format none --temp 0"),
        ("lcpp-iq2s-mtp", "", "--reasoning-format deepseek", "", "--reasoning-format deepseek"),
        ("lcpp-iq2s", "", "-c 8192 --reasoning-format=deepseek", "", "-c 8192 --reasoning-format=deepseek"),
    ]
    got_cases = []
    if m:
        for lane, env, extra, want_env, want_extra in cases:
            e = {"PATH": "/usr/bin:/bin"}
            if env:
                e["TURING_LANE_ENV"] = env
            if extra:
                e["TURING_LCPP_EXTRA"] = extra
            r = subprocess.run(["bash", "-c", m.group(0) + 'e47_shape "$1"; printf "%s|%s" "${TURING_LANE_ENV:-}" '
                                '"${TURING_LCPP_EXTRA:-}"', "e47", lane], env=e, capture_output=True, text=True, timeout=30)
            got_cases.append((lane, env, extra, r.stdout == f"{want_env}|{want_extra}", r.stdout))
    hard("run_quality.sh e47_shape(): E47's response shape added unless the caller set the key",
         bool(m) and all(c[3] for c in got_cases), "; ".join(f"{c[0]} {c[1]!r} {c[2]!r} -> {c[4]!r}" for c in got_cases if not c[3]))
    REPORT["static"] = {"shared_functions_identical": len(same), "dataset_sha256": actual,
                        "e47_shape_cases": len(got_cases)}


def check_rescoring(ref, fork):
    print("1. rescoring the stored E47 rows from their stored text")
    out = {}
    for s in SUITES:
        items = {it["id"]: it for it in read_jsonl(os.path.join(DATASETS, f"{s}.jsonl"))}
        base = fork.base_suite(s)
        for lab in LABELS:
            rows = read_jsonl(os.path.join(LAPTOP_RESULTS, lab, f"{s}.jsonl"))
            agree_ref = agree_stored = 0
            whole = whole_agree = 0
            partial_dis = []
            for r in rows:
                ex = r.get("raw_answer_excerpt") or ""
                vf = fork.SCORERS[base](ex, items[r["id"]])
                vr = ref.SCORERS[base](ex, items[r["id"]])
                vf = vf[0] if isinstance(vf, tuple) else vf
                vr = vr[0] if isinstance(vr, tuple) else vr
                agree_ref += bool(vf) == bool(vr)
                agree_stored += bool(vf) == bool(r["correct"])
                if len(ex) < 200 or r.get("error"):
                    whole += 1
                    whole_agree += bool(vf) == bool(r["correct"])
                elif bool(vf) != bool(r["correct"]):
                    partial_dis.append({"id": r["id"], "stored": r["correct"], "rescored": bool(vf),
                                        "truncated": r.get("truncated"), "excerpt_head": ex[:60]})
            n = len(rows)
            key = f"{lab}/{s}"
            out[key] = {"rows": n, "fork_vs_run_eval_scorer": agree_ref, "rescored_eq_stored": agree_stored,
                        "whole_answer_rows": whole, "whole_answer_exact": whole_agree,
                        "long_answer_rows": n - whole, "long_answer_disagreements": partial_dis}
            hard(f"{key}: fork scorer = run_eval scorer on every stored text", agree_ref == n, f"{agree_ref}/{n}")
            hard(f"{key}: whole-answer rows rescore exactly", whole_agree == whole, f"{whole_agree}/{whole}")
            print(f"        {key}: rescored correctness = stored on {agree_stored}/{n} rows; long answers "
                  f"(excerpt cut at 200 chars) {n - whole - len(partial_dis)}/{n - whole} agree"
                  + (f"; disagreements: {[d['id'] for d in partial_dis]}" if partial_dis else ""))
    REPORT["rescoring"] = out


def check_requests(ref, fork, rq_text, scratch):
    print("2. requests and rows, item by item")
    os.makedirs(scratch, exist_ok=True)
    xb = os.path.join(scratch, "extra_body.json")
    with open(xb, "w", encoding="utf-8") as f:
        json.dump({"logit_bias": {"151667": -2.0}, "repetition_penalty": 1.05}, f)
    scenarios = ["tabby", "llamacpp", "inline_think", "ctk400", "conn_drop", "hard_fail", "encode_fail"]
    nodraft = {s: {r["id"]: r for r in read_jsonl(os.path.join(LAPTOP_RESULTS, "EFF-h38a_nodraft", f"{s}.jsonl"))}
               for s in SUITES}
    out = {}
    for s in SUITES:
        items = read_jsonl(os.path.join(DATASETS, f"{s}.jsonl"))
        base_argv = ["--suite", s, "--label", "TURING-check", "--base-url", URL]
        fork_e47 = run_quality_fork_argv(rq_text, s, scratch)
        hard(f"{s}: run_quality.sh's eval command found", fork_e47 is not None)
        flagsets = [("E47 (hardening.sh) vs run_quality.sh", base_argv + E47_ARGV + ["--limit", "0"], fork_e47)]
        for name, extra in [("defaults (temperature 1.0)", []),
                            ("reasoning effort off", ["--reasoning-effort", ""]),
                            ("temperature 0.6, max tokens 2048", ["--temperature", "0.6", "--max-tokens", "2048"]),
                            ("extra body file", ["--temperature", "0", "--extra-body-file", xb])]:
            flagsets.append((name, base_argv + extra, base_argv + extra + ["--datasets-dir", DATASETS]))
        for fname, ref_argv, fork_argv in flagsets:
            if fork_argv is None:
                continue
            a_ref, a_fork = parsed_args_run_eval(ref, ref_argv), parsed_args_fork(fork, fork_argv)
            req_ok = row_ok = body_ok = side_ok = total = 0
            first_bad = None
            for sc in (scenarios if fname.startswith("E47") else ["tabby", "ctk400"]):
                for it in items:
                    stored = nodraft[s].get(it["id"], {})
                    answer = (stored.get("raw_answer_excerpt") or "Answer: 1") + f" [{it['id']}]"
                    reasoning = f"Reasoning about {it['id']}: " + " ".join(["step"] * (len(it["id"]) % 7 + 3))
                    fork._ENCODE_OK.clear()
                    with patched(FakeServer(sc, answer, reasoning)) as srv_r:
                        row_r = ref.run_item(s, it, a_ref)
                    with patched(FakeServer(sc, answer, reasoning)) as srv_f:
                        row_f, full = fork.run_item_full(s, it, a_fork)
                    total += 1
                    calls_r, calls_f = srv_r.calls, srv_f.calls
                    extra_calls = calls_f[len(calls_r):]
                    this_req = (calls_f[:len(calls_r)] == calls_r
                                and all(c[0].endswith("/v1/token/encode") for c in extra_calls))
                    req_ok += this_req
                    wr, wf = dict(row_r), dict(row_f)
                    wr.pop("wall_s", None); wf.pop("wall_s", None)
                    this_row = (json.dumps(wr, ensure_ascii=False) == json.dumps(wf, ensure_ascii=False)
                                and isinstance(row_f.get("wall_s"), float))
                    row_ok += this_row
                    chats = [c for c in calls_f if c[0].endswith("/v1/chat/completions")]
                    use_ctk = not row_f.get("chat_template_kwargs_dropped")
                    this_body = bool(chats) and chats[-1][1] == fork.request_body_bytes(fork.build_prompt(s, it), a_fork, use_ctk)
                    this_body = this_body and full.get("request_sha256") == hashlib.sha256(chats[-1][1]).hexdigest()
                    body_ok += this_body
                    want_src = {"tabby": "encode", "ctk400": "encode", "conn_drop": "encode", "llamacpp": "usage",
                                "inline_think": "usage", "encode_fail": None, "hard_fail": "n/a"}[sc]
                    this_side = (("error" in full) if sc == "hard_fail" else full.get("completion_tokens_source") == want_src)
                    side_ok += this_side
                    if not (this_req and this_row and this_body and this_side) and first_bad is None:
                        first_bad = {"scenario": sc, "id": it["id"], "requests": this_req, "row": this_row,
                                     "body": this_body, "sidecar": this_side, "row_ref": wr, "row_fork": wf}
            key = f"{s} / {fname}"
            out[key] = {"cases": total, "requests_identical": req_ok, "rows_identical": row_ok,
                        "request_body_bytes_identical": body_ok, "sidecar_ok": side_ok, "first_mismatch": first_bad}
            hard(f"{key}: byte-identical requests", req_ok == total, f"{req_ok}/{total}")
            hard(f"{key}: identical rows", row_ok == total, f"{row_ok}/{total}")
            hard(f"{key}: request_body_bytes() = sent bytes", body_ok == total, f"{body_ok}/{total}")
            hard(f"{key}: sidecar token source", side_ok == total, f"{side_ok}/{total}")
            if first_bad:
                print("        first mismatch:", json.dumps(first_bad)[:600])
    REPORT["requests"] = out


def check_statistics():
    print("3. statistics: stored hardening_analyze.py / identity_anatomy.py outputs reproduced")
    sys.path.insert(0, HERE)
    cq = load_module("compare_quality", os.path.join(HERE, "compare_quality.py"))
    store = cq.Store(os.path.join(REPO, "testsuite", "results", "turing", "evals"), LAPTOP_RESULTS)
    out = {}
    for path in sorted(glob.glob(os.path.join(ROWS, "h38*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        if not isinstance(d, dict) or "A" not in d or "B" not in d or "suite" not in d:
            continue
        A, B = store.rows(d["A"], d["suite"]), store.rows(d["B"], d["suite"])
        order = cq.dataset_order(DATASETS, d["suite"])
        name = os.path.basename(path)
        if "accA" in d:
            limit = d["n"] + d["errors"] if d["missing"] == 0 and d["n"] + d["errors"] < len(order) else 0
            st = cq.pair_stats(d["suite"], d["A"], d["B"], A, B, order, limit)
            diffs = {k: [d[k], st.get(k)] for k in d if k != "verdict" and st.get(k) != d[k]}
            out[name] = {"kind": "hardening_analyze", "limit": limit, "fields": len(d) - ("verdict" in d), "diffs": diffs}
            hard(f"{name} (hardening_analyze, limit {limit or 'none'})", not diffs, json.dumps(diffs)[:300] if diffs else "")
        elif "identical" in d:
            ids = [i for i in order if i in A and i in B and not A[i].get("error") and not B[i].get("error")
                   and A[i].get("content_sha256") and B[i].get("content_sha256")]
            an = cq.anatomy(A, B, ids)
            diffs = {k: [d[k], an.get(k)] for k in d if k not in ("suite", "A", "B") and an.get(k) != d[k]}
            out[name] = {"kind": "identity_anatomy", "diffs": diffs}
            hard(f"{name} (identity_anatomy)", not diffs, json.dumps(diffs)[:300] if diffs else "")
    REPORT["statistics"] = out
    n = sum(1 for v in out.values() if not v["diffs"])
    print(f"        reproduced {n}/{len(out)} stored analysis files")
    hard("at least the 12 hardening_analyze + 3 anatomy files were found", len(out) >= 15, str(len(out)))


def check_resume(fork, scratch):
    print("4. resume, latest-row-wins, consecutive-error stop (fake server, scratch dir)")
    res_dir = os.path.join(scratch, "resume-results")
    os.makedirs(res_dir, exist_ok=True)
    label = "TURING-resume-" + time.strftime("%Y%m%dT%H%M%S")
    argv = ["--suite", "gsm8k_500", "--label", label, "--base-url", URL, "--temperature", "0", "--limit", "6",
            "--results-dir", res_dir, "--datasets-dir", DATASETS,
            "--summary-out", os.path.join(scratch, label + "-summary.jsonl")]
    args = parsed_args_fork(fork, argv)
    items = fork.load_dataset("gsm8k_500", 6, DATASETS)
    rows_path = os.path.join(res_dir, label, "gsm8k_500.jsonl")
    full_path = os.path.join(res_dir, label, "gsm8k_500.full.jsonl")
    sink = io.StringIO()

    def lines(p):
        with open(p, encoding="utf-8") as f:
            return [l for l in f if l.strip()]

    bad = items[2]["id"]
    with patched(FakeServer("tabby", "Answer: 5", "think", fail_ids=[bad])) as srv:
        # the fake server fails an item whose prompt carries "[id]": tag the prompt through a wrapped build_prompt
        real_bp = fork.build_prompt
        fork.build_prompt = lambda s, it: real_bp(s, it) + f" [{it['id']}]"
        try:
            with contextlib.redirect_stdout(sink):
                s1 = fork.run_suite("gsm8k_500", args)
        finally:
            fork.build_prompt = real_bp
    hard("run 1: 6 rows, 1 error row, 6 chat requests", len(lines(rows_path)) == 6 and s1["errors"] == 1
         and srv.chat_n == 6, f"rows {len(lines(rows_path))}, errors {s1['errors']}, chats {srv.chat_n}")
    with patched(FakeServer("tabby", "Answer: 5", "think")) as srv:
        with contextlib.redirect_stdout(sink):
            s2 = fork.run_suite("gsm8k_500", args)
    latest = fork.load_latest_rows(rows_path)
    hard("run 2: only the error item re-run; latest rows error-free; file appended (7 lines)",
         srv.chat_n == 1 and len(lines(rows_path)) == 7 and s2["errors"] == 0 and s2["n"] == 6
         and s2["rows_this_session"] == 1 and all("error" not in r for r in latest.values()),
         f"chats {srv.chat_n}, lines {len(lines(rows_path))}, summary n={s2['n']} errors={s2['errors']}")
    with patched(FakeServer("tabby", "Answer: 5", "think")) as srv:
        with contextlib.redirect_stdout(sink):
            s3 = fork.run_suite("gsm8k_500", args)
    hard("run 3: nothing left to run, file unchanged", srv.chat_n == 0 and len(lines(rows_path)) == 7
         and s3["n"] == 6, f"chats {srv.chat_n}")

    def todo(limit):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                fork.main(argv[:argv.index("--limit")] + ["--limit", str(limit)] + argv[argv.index("--limit") + 2:] + ["--todo"])
            except SystemExit:
                pass
        return buf.getvalue().strip()
    hard("--todo counts the items without an error-free row (limit 6 -> 0, limit 8 -> 2)",
         todo(6) == "0" and todo(8) == "2", f"{todo(6)!r} / {todo(8)!r}")
    with open(rows_path, "a", encoding="utf-8") as f:
        f.write('{"id": "gsm8k_torn", "correct": tr')          # a hard stop mid-write
    args.limit = 8
    with patched(FakeServer("tabby", "Answer: 5", "think")) as srv:
        with contextlib.redirect_stdout(sink):
            s4 = fork.run_suite("gsm8k_500", args)
    latest = fork.load_latest_rows(rows_path)
    hard("torn last line survived: 2 new items appended on a fresh line, 8 items complete",
         srv.chat_n == 2 and s4["n"] == 8 and len(latest) == 8 and "gsm8k_torn" not in latest,
         f"chats {srv.chat_n}, n {s4['n']}")
    full = fork.load_latest_rows(full_path)
    hard("sidecar holds the full text and token source for every item",
         set(full) == set(latest) and all(v.get("completion_tokens_source") == "encode" and v.get("content")
                                          for v in full.values()))
    args2 = parsed_args_fork(fork, argv[:3] + [label + "-dead"] + argv[4:])
    args2.limit = 10
    stopped = False
    with patched(FakeServer("dead", "", "")) as srv:
        with contextlib.redirect_stdout(sink):
            try:
                fork.run_suite("gsm8k_500", args2)
            except fork.ServerGone:
                stopped = True
    dead_rows = lines(os.path.join(res_dir, label + "-dead", "gsm8k_500.jsonl"))
    hard("dead server: stops after 3 consecutive failed requests, 3 error rows kept", stopped and len(dead_rows) == 3,
         f"stopped={stopped}, rows {len(dead_rows)}")
    summ = read_jsonl(os.path.join(scratch, label + "-summary.jsonl"))
    hard("--summary-out appended one summary per run (5)", len(summ) == 5 and summ[-1].get("stopped"), str(len(summ)))
    REPORT["resume"] = {"scratch": scratch, "label": label}


def upstream_tag_parser():
    """Upstream TabbyAPI's TagStreamParser at the desktop's commit, from the laptop's bare source mirror (stdlib-only
    module: re, typing). Returns (TagStreamParser, REASONING, CONTENT) or None if the mirror cannot be read."""
    try:
        r = subprocess.run(["git", "-C", TABBY_MIRROR, "show", f"{UPSTREAM_TABBY}:endpoints/OAI/utils/stream_parser.py"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or "class TagStreamParser" not in r.stdout:
        return None
    ns = {"__name__": "upstream_stream_parser"}
    exec(compile(r.stdout, f"{UPSTREAM_TABBY}:endpoints/OAI/utils/stream_parser.py", "exec"), ns)
    return ns["TagStreamParser"], ns["REASONING"], ns["CONTENT"]


def upstream_split(text, parser, chunk):
    """Non-streaming upstream b1c2253 for generated `text` with its defaults on Qwen3.8 (reasoning tags <think> /
    </think>, generation prompt ending inside the think block, a tool format detected): _chat_stream_collector feeds
    the parser chunk by chunk and finishes it; _compose_response returns (reasoning_content, content), each None when
    empty, and content None when it is only whitespace."""
    cls, REASONING, CONTENT = parser
    p = cls(reasoning_start="<think>", reasoning_end="</think>", tool_start="<tool_call>", tool_end="</tool_call>",
            start_in_reasoning=True, tool_calls_in_reasoning=True)
    rc = content = ""
    pieces = [text[k:k + chunk] for k in range(0, len(text), chunk)] or [""]
    for n, piece in enumerate(pieces):
        events = p.feed(piece) if piece else []
        if n == len(pieces) - 1:
            events += p.finish()
        for ch, sub in events:
            if ch == REASONING:
                rc += sub
            elif ch == CONTENT:
                content += sub
    return (rc or None), ((content if content.strip() else None) or None)


class ShapeServer:
    """A TabbyAPI that generated texts[id] for each item (finish[id] as finish_reason), answering in one shape:
      'lab'    E47's server, lab TabbyAPI 4a4f9f4 (reasoning parser and tool parsing off): content = the text, usage null;
      'inline' upstream b1c2253 with reasoning: false: content = the text, usage = generated tokens;
      'split'  upstream b1c2253 defaults: the text split by upstream's TagStreamParser, usage = generated tokens.
    Generated tokens are the text's whitespace-separated words; /v1/token/encode counts them plus BOS."""

    def __init__(self, shape, texts, finish, by_prompt, parser=None):
        self.shape, self.texts, self.finish, self.by_prompt, self.parser = shape, texts, finish, by_prompt, parser
        self.n = 0

    def __call__(self, req, timeout=None):
        url = req.full_url
        if url.endswith("/v1/token/encode"):
            k = len(json.loads(req.data)["text"].split()) + 1
            return FakeResp({"tokens": list(range(k)), "length": k})
        if not url.endswith("/v1/chat/completions"):
            raise http_error(url, 404)
        self.n += 1
        iid = self.by_prompt[json.loads(req.data)["messages"][0]["content"]]
        text = self.texts[iid]
        if self.shape == "split":
            rc, content = upstream_split(text, self.parser, 1 + self.n % 7)
            msg = {"role": "assistant", "content": content, "reasoning_content": rc, "tool_calls": None}
        else:
            msg = {"role": "assistant", "content": text if text.strip() else None}
        ntok = len(text.split())
        usage = None if self.shape == "lab" else {"prompt_tokens": 60, "completion_tokens": ntok, "total_tokens": ntok + 60}
        return FakeResp({"id": "x", "object": "chat.completion", "model": "fake", "usage": usage,
                         "choices": [{"index": 0, "message": msg, "finish_reason": self.finish[iid]}]})


def check_shapes(ref, fork, scratch):
    print("5. response shape: E47's inline text vs upstream TabbyAPI's split, same generated tokens")
    sys.path.insert(0, HERE)
    cq = load_module("compare_quality", os.path.join(HERE, "compare_quality.py"))
    parser = upstream_tag_parser()
    if not hard(f"upstream TagStreamParser loaded from the mirror ({UPSTREAM_TABBY})", parser is not None):
        return
    s = "gsm8k_500"
    items = {it["id"]: it for it in read_jsonl(os.path.join(DATASETS, f"{s}.jsonl"))}
    traces = [t for t in read_jsonl(os.path.join(ROWS, "traces_h38c_nodraft.jsonl"))
              if t["suite"] == s and t["id"] in items and t["content"].count("</think>") == 1 and not t["reasoning"]]
    hard("E47 traces with one inline '</think>' found", len(traces) >= 20, str(len(traces)))
    order = [i for i in cq.dataset_order(DATASETS, s) if i in {t["id"] for t in traces}]
    lap = {t["id"]: t["content"] for t in traces}
    finish = {i: "stop" for i in order}
    # one item truncated inside the reasoning at the cap, whose reasoning text E47's fallback scorer gets right
    trunc = None
    for i in reversed(order):
        exp = items[i]["answer_number"]
        cand = lap[i].split("</think>")[0].rstrip("\n") + "\n\nSo the result is " + (
            str(int(exp)) if float(exp).is_integer() else str(exp))
        if fork.score_gsm8k(fork.visible_content({"content": cand}), items[i]):
            trunc, lap[i], finish[i] = i, cand, "length"
            break
    hard("an item truncated inside the reasoning that E47's scorer gets right", trunc is not None)
    # the Turing side generates the same tokens, except one more word in the reasoning of five items
    tur = dict(lap)
    plus1 = [i for i in order if i != trunc][:5]
    for i in plus1:
        tur[i] = lap[i].replace("\n</think>", " Verified.\n</think>", 1)
    a_ref = parsed_args_run_eval(ref, ["--suite", s, "--label", "EFF-h38a_nodraft", "--base-url", URL] + E47_ARGV)
    a_fork = parsed_args_fork(fork, run_quality_fork_argv(open(os.path.join(HERE, "run_quality.sh"), encoding="utf-8").read(), s, scratch))
    by_prompt = {fork.build_prompt(s, items[i]): i for i in order}
    A = {}
    with patched(ShapeServer("lab", lap, finish, by_prompt)):
        for i in order:
            A[i] = ref.run_item(s, items[i], a_ref)
    sides = {}
    for shape in ("inline", "split"):
        rows, full = {}, {}
        fork._ENCODE_OK.clear()
        with patched(ShapeServer(shape, tur, finish, by_prompt, parser)):
            for i in order:
                rows[i], full[i] = fork.run_item_full(s, items[i], a_fork)
        sides[shape] = (rows, full)
    rs, fs = sides["split"]
    hard("split sidecar: shape recorded, inline text rebuilt exactly (sha256 of the generated text)",
         all(fs[i].get("response_shape") == "split" and fs[i].get("inline_sha256") ==
             hashlib.sha256(tur[i].encode("utf-8")).hexdigest() for i in order),
         str(sorted({fs[i].get("laptop_inline") for i in order})))
    hard("split rows really differ: the truncated item scores False on content alone, True on E47's inline path",
         rs[trunc]["correct"] is False and fs[trunc].get("inline_correct") is True and A[trunc]["correct"] is True)
    P = {}
    for shape in ("inline", "split"):
        rows, full = sides[shape]
        P[shape] = cq.pair_stats(s, "EFF-h38a_nodraft", f"TURING-shape-{shape}", A, rows, order, 0, {}, full)
    pi, ps = P["inline"], P["split"]
    xi, xs = pi["extra"], ps["extra"]
    n, k_same = len(order), len(order) - len(plus1)
    hard("raw identity of the split shape is near 0 (the defect this check guards against)",
         ps["identity_rate"] is not None and ps["identity_rate"] <= 1 / n + 1e-9, str(ps["identity_rate"]))
    hard("byte identity: inline shape (raw) = split shape (inline basis) = the true value",
         pi["identity_rate"] == xs.get("identity_rate_shape_matched") == round(k_same / n, 4),
         f"{pi['identity_rate']} / {xs.get('identity_rate_shape_matched')} / {round(k_same / n, 4)}")
    hard("matched token identity and mean tokens agree between the shapes",
         xi.get("token_identity_rate_matched") == xs.get("token_identity_rate_matched") == round(k_same / n, 4)
         and xi.get("mean_tok_A_matched") == xs.get("mean_tok_A_matched")
         and xi.get("mean_tok_B_matched") == xs.get("mean_tok_B_matched"),
         f"{xi.get('token_identity_rate_matched')} / {xs.get('token_identity_rate_matched')}")
    an_i = {k: v for k, v in xi["anatomy"].items() if k != "identity_basis"}
    an_s = {k: v for k, v in xs["anatomy"].items() if k != "identity_basis"}
    hard("anatomy agrees between the shapes", an_i == an_s, json.dumps([an_i, an_s])[:300])
    hard("anatomy token deltas on the matched basis: the five one-word variants are +1, not 0",
         an_i.get("token_delta_buckets", {}).get("1-5") == len(plus1) and an_i.get("token_delta_median") == 1
         and an_i.get("token_basis") == "encode (matched)",
         json.dumps(an_i.get("token_delta_buckets")))
    ip = xs.get("inline_path") or {}
    hard("accuracy on E47's inline path for the split shape = the inline shape's accuracy",
         ip.get("accB") == pi["accB"] and ip.get("accA") == pi["accA"] and xs.get("inline_path_differs_B") == [trunc],
         f"{ip.get('accB')} vs {pi['accB']}; differs {xs.get('inline_path_differs_B')}")
    res = {"generated": "check", "inputs": {"turing_dir": "-", "laptop_dir": "-", "datasets_dir": "-"}, "lanes": {},
           "pairs": [{"group": "custom", "suite": s, "A": "EFF-h38a_nodraft", "B": f"TURING-shape-{sh}", "limit": 0,
                      "note": sh, "stats": P[sh]} for sh in ("inline", "split")], "runs": []}
    md = cq.render_md(res)
    lines = {sh: next((l for l in md.splitlines() if l.startswith("- ") and f"TURING-shape-{sh}:" in l), "") for sh in P}
    hard("CHECK line: shapes differ (split only), with the inline-path accuracy",
         "response shapes differ" in lines["split"] and "inline path" in lines["split"]
         and "response shapes differ" not in lines["inline"], lines["split"][:200])
    hard("markdown marks the raw token fields as mixed basis",
         md.count("mixed basis") >= 2, str(md.count("mixed basis")))
    lossy = fork.laptop_inline_text({"reasoning_content": "r", "content": None}, "stop")
    hard("laptop_inline_text: whitespace-only content after the tag is marked approximate",
         lossy == ("r</think>", "tag_appended"))
    REPORT["shapes"] = {"items": n, "truncated_item": trunc, "plus_one_word": plus1,
                        "identity_raw_split": ps["identity_rate"], "identity_inline": pi["identity_rate"],
                        "identity_split_inline_basis": xs.get("identity_rate_shape_matched"),
                        "anatomy": an_i, "inline_path": ip}


def check_lane_table():
    print("6. lane table: reruns and variants do not shrink the common item set")
    sys.path.insert(0, HERE)
    cq = load_module("compare_quality", os.path.join(HERE, "compare_quality.py"))
    order = cq.dataset_order(DATASETS, "gsm8k_500")[:40]

    def mkrows(ids, wrong=()):
        return {i: {"id": i, "correct": i not in wrong, "wall_s": 1.0, "completion_tokens": 100,
                    "thinking_tokens_estimate": 50, "truncated": False} for i in ids}

    data = {"TURING-exl3-mtp": mkrows(order, order[:4]), "TURING-exl3-nodraft": mkrows(order, order[2:5]),
            "TURING-exl3-nodraft-r2": mkrows(order[:10]), "TURING-exl3-mtp-fp32": mkrows(order[:15], order[:1])}

    class MemStore:
        def rows(self, lab, suite):
            return data.get(lab)

        def full(self, lab, suite):
            return {}
    t = {x["label"]: x for x in cq.lane_table(MemStore(), "gsm8k_500", cq.lane_order(list(data)), order)}
    ok = (t["TURING-exl3-mtp"]["common_set_n"] == 40 and t["TURING-exl3-mtp"]["common_n"] == 40
          and t["TURING-exl3-mtp"]["acc_common"] == 90.0 and t["TURING-exl3-nodraft"]["acc_common"] == 92.5
          and t["TURING-exl3-nodraft-r2"]["common_n"] == 10 and t["TURING-exl3-mtp-fp32"]["common_n"] == 15
          and t["TURING-exl3-mtp-fp32"]["acc_common"] == round(100 * 14 / 15, 2))
    hard("stock lanes keep all 40 common items; the 10-item rerun and 15-item variant are scored on their overlap", ok,
         json.dumps({k: [v.get("common_set_n"), v.get("common_n"), v.get("acc_common")] for k, v in t.items()}))
    REPORT["lane_table"] = {k: [v.get("common_set_n"), v.get("common_n"), v.get("acc_common")] for k, v in t.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=None, help="scratch dir for check 2's extra-body file and check 4 "
                    "(default: a new temp dir; it is left in place)")
    ap.add_argument("--json", default=None, help="write the detailed report as JSON")
    a = ap.parse_args()
    scratch = a.scratch or tempfile.mkdtemp(prefix="turing-fidelity-")
    os.makedirs(scratch, exist_ok=True)
    ref = load_module("run_eval_ref", RUN_EVAL)
    fork = load_module("run_eval_turing", os.path.join(HERE, "run_eval_turing.py"))
    with open(os.path.join(HERE, "run_quality.sh"), encoding="utf-8") as f:
        rq_text = f.read()
    t0 = time.time()
    check_static(ref, fork, rq_text)
    check_rescoring(ref, fork)
    check_requests(ref, fork, rq_text, scratch)
    check_statistics()
    check_resume(fork, scratch)
    check_shapes(ref, fork, scratch)
    check_lane_table()
    REPORT["failures"] = FAILS
    REPORT["seconds"] = round(time.time() - t0, 1)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(REPORT, f, indent=1, ensure_ascii=False)
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)} "
          f"({REPORT['seconds']} s; scratch {scratch})")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
