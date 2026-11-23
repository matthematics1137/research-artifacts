#!/usr/bin/env python3
"""Paper 8 (Turing lab) eval client: a file-safe fork of testsuite/evals/run_eval.py.

What is identical to run_eval.py (proved by tools/turing/quality/check_fidelity.py):
  prompt construction, request body (byte for byte), scorers and the per-item row written to
  <results>/<label>/<suite>.jsonl, for the suites this fork keeps: gsm8k, mmlu, math25, needle and the
  hardening suites gsm8k_500, mmlu_500, math500. The shared functions below are verbatim copies.

What is excluded: the code-execution suites (humaneval, humaneval_plus). Their sandbox cleanup is the only
file-discarding code path in run_eval.py, and nothing in the Turing lab discards files.

What is added (none of it changes a row):
  - paths: --results-dir (default $EVAL_RESULTS_DIR, else $TURING_LAB/evals/results) and --datasets-dir
    (default $EVAL_DATASETS_DIR, else $TURING_LAB/evals/datasets); TURING_LAB defaults to
    /opt/p8;
  - every dataset file is checked against its laptop sha256 before a run (refuses on a mismatch);
  - resumable: rows are appended; an item whose latest row has no "error" is skipped on a rerun; readers take
    the latest row per id (hardening_analyze.py and compare_quality.py both do);
  - a full-response sidecar <suite>.full.jsonl (reasoning, answer, finish reason, usage, server timings,
    request sha256 and, where the server has /v1/token/encode, the re-encoded token count the laptop's
    TabbyAPI rows used) so later analyses need no rerun. It also records the response shape: E47's server
    (lab TabbyAPI 4a4f9f4, reasoning parser off) returned the reasoning inline in content, while upstream
    TabbyAPI (parser on by default since 2186cdb) and llama-server (--reasoning-format deepseek) split it
    and drop the '</think>' tag. For a split response the sidecar adds the laptop-shape text's sha256, its
    encode count and its score (inline_sha256, inline_encode_tokens, inline_correct), so a comparison with
    the laptop's rows can use one basis;
  - SIGTERM / SIGHUP are handled like Ctrl-C (the partial summary is still written);
  - the suite stops after --max-consecutive-errors failed requests in a row (the server is gone), exit 3;
  - --summary-out PATH also writes the suite summary as one JSON object (for run_quality.sh's run record).

Usage:
  run_eval_turing.py --suite gsm8k_500|mmlu_500|math500|gsm8k|mmlu|math25|needle --label <label>
                     [--base-url http://localhost:8090] [--reasoning-effort medium] [--max-tokens 4096]
                     [--temperature 1.0] [--limit N] [--concurrency 1] [--extra-body-file F]
                     [--results-dir D] [--datasets-dir D] [--no-full-text] [--no-encode-count]
                     [--max-consecutive-errors 3] [--summary-out PATH]
  run_eval_turing.py --selftest
Stdlib only.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

LAB = os.environ.get("TURING_LAB", "/opt/p8")
RESULTS = os.environ.get("EVAL_RESULTS_DIR", os.path.join(LAB, "evals", "results"))
DATASETS = os.environ.get("EVAL_DATASETS_DIR", os.path.join(LAB, "evals", "datasets"))
SUITES = ["gsm8k", "mmlu", "math25", "needle"]          # run_eval.SUITES minus the code-execution suites
HARDENING_SUITES = {"gsm8k_500": "gsm8k", "mmlu_500": "mmlu", "math500": "math25"}
DATASET_FILES = {"gsm8k": "gsm8k_100.jsonl", "mmlu": "mmlu_200.jsonl",
                 "math25": "math25.jsonl", "needle": "needle.jsonl",
                 "gsm8k_500": "gsm8k_500.jsonl", "mmlu_500": "mmlu_500.jsonl",
                 "math500": "math500.jsonl"}
# sha256 of the laptop's testsuite/evals/datasets files (2026-10-30); check_fidelity.py re-derives them.
DATASET_SHA256 = {
    "gsm8k_100.jsonl": "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37",
    "mmlu_200.jsonl": "387309894aabc7e75774ae5cc01dc741458e7e098dec194035bd263d23356cfd",
    "math25.jsonl": "c2ca0ffac2ef6a24617fd5390c1792d4168683dcbe4d08cce58a5d9a3e6c1da8",
    "needle.jsonl": "4b353f181a44933c1aa0599c9e584e0dd1f551df8b991f32eed57e38bad8054c",
    "math500.jsonl": "477e20dff29423126a4111fefa33ac9c65c5ab290c3e9069c03c7640e2b6be10",
    "gsm8k_500.jsonl": "7179bc41ad3e06c9f4b62ff1811fd272fa5ae82ddbee91dbf74d8278878ba344",
    "mmlu_500.jsonl": "0a7e6b09f8d3ec6a13b00d02d8549a8e48d255a959be6034b932372f3cdea8da",
}


def base_suite(suite):
    return HARDENING_SUITES.get(suite, suite)
REQUEST_TIMEOUT_S = 600

# ------------------------------------------------------------------ prompts (verbatim from run_eval.py)

def build_prompt(suite, item):
    suite = base_suite(suite)
    if suite == "gsm8k":
        return item["question"] + "\n\nEnd your response with: Answer: <number>"
    if suite == "mmlu":
        letters = "ABCD"
        ch = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item["choices"]))
        return (f"{item['question']}\n{ch}\n\n"
                "End your response with: Answer: <letter>")
    if suite in ("humaneval", "humaneval_plus"):
        return ("Complete the following Python function. Return the complete, "
                "runnable function (including the signature and any imports it "
                "needs) in a single ```python code block and nothing else "
                "after it.\n\n```python\n" + item["prompt"] + "\n```")
    if suite == "math25":
        return (item["problem"] +
                "\n\nEnd your response with: Answer: <final answer>")
    if suite == "needle":
        return item["context"] + "\n\n" + item["question"]
    raise ValueError(suite)

# ------------------------------------------------------------------ scoring (verbatim from run_eval.py)

NUM_RE = re.compile(r"[-+]?\$?[\d][\d,]*(?:\.\d+)?")


def _to_float(tok):
    tok = tok.strip().replace("$", "").replace(",", "")
    tok = tok.rstrip(".")
    try:
        return float(tok)
    except ValueError:
        return None


def score_gsm8k(content, item):
    expected = float(item["answer_number"])
    # last "Answer:" marker, first number after it
    matches = list(re.finditer(r"[Aa]nswer\s*:", content))
    cand = None
    if matches:
        tail = content[matches[-1].end():][:120]
        m = NUM_RE.search(tail)
        if m:
            cand = _to_float(m.group(0))
    if cand is None:
        # fallback: a bare final number on the last non-empty line
        lines = [l for l in content.strip().splitlines() if l.strip()]
        if lines:
            nums = NUM_RE.findall(lines[-1])
            if nums:
                cand = _to_float(nums[-1])
    if cand is None:
        return False
    return (abs(cand - expected) <= 1e-4
            or (expected != 0 and abs(cand - expected) / abs(expected) <= 1e-4))


def score_mmlu(content, item):
    expected = "ABCD"[item["answer_idx"]]
    matches = list(re.finditer(r"[Aa]nswer\s*:\s*\(?\**([A-Da-d])\**\)?\b",
                               content))
    if matches:
        return matches[-1].group(1).upper() == expected
    # fallback: a standalone A-D anywhere in the last 3 non-empty lines
    lines = [l for l in content.strip().splitlines() if l.strip()][-3:]
    found = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", "\n".join(lines))
    if found:
        return found[-1] == expected
    return False


def score_needle(content, item):
    return item["expected_substring"].lower() in content.lower()


# ------------------------------------------------------------- MATH scoring (verbatim from run_eval.py)

def extract_boxed(s):
    """Content of the last \\boxed{...}, brace-matched; None if absent."""
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth, out = 1, []
    while j < len(s):
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
        j += 1
    return None


def _convert_fracs(s):
    """\\frac{a}{b} -> a/b (parens around non-atomic parts); repeats for nesting."""
    def atom(x):
        return x if re.fullmatch(r"[\w.\\-]+", x) else "(" + x + ")"
    for _ in range(20):  # bounded; each pass rewrites the first remaining \frac
        i = s.find("\\frac{")
        if i < 0:
            break
        j, depth, num = i + 6, 1, []
        while j < len(s) and depth:
            c = s[j]
            depth += (c == "{") - (c == "}")
            if depth:
                num.append(c)
            j += 1
        if depth or j >= len(s) or s[j] != "{":
            break  # malformed; leave as-is
        j2, depth, den = j + 1, 1, []
        while j2 < len(s) and depth:
            c = s[j2]
            depth += (c == "{") - (c == "}")
            if depth:
                den.append(c)
            j2 += 1
        if depth:
            break
        s = s[:i] + atom("".join(num)) + "/" + atom("".join(den)) + s[j2:]
    s = re.sub(r"\\frac(\d)(\d)", r"\1/\2", s)  # \frac12 shorthand
    return s


def normalize_math(s):
    """Conservative LaTeX answer normalization (formatting only, no algebra)."""
    s = (s or "").strip()
    b = extract_boxed(s)
    if b is not None:
        s = b
    s = s.strip().rstrip(".").strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s.strip("$").strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,!;:]", "", s)                      # \, \! \; \:
    s = s.replace("\\quad", "").replace("\\qquad", "")
    s = re.sub(r"\\ ", "", s)                            # "\ " forced space
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\(?:text|mbox|mathrm)\{([^{}]*)\}", r"\1", s)
    s = _convert_fracs(s)
    s = s.replace("\\%", "%").replace("\\$", "")
    s = re.sub(r"\^\{?\\circ\}?", "", s).replace("\u00b0", "")
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")                           # thousands separators
    return s


def _parse_number(s):
    s = s.strip().lstrip("+")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def extract_math_answer(content):
    matches = list(re.finditer(r"[Aa]nswer\s*:", content))
    if matches:
        tail = content[matches[-1].end():].strip()
        if tail:
            first = tail.splitlines()[0].strip()
            if first:
                return first
    b = extract_boxed(content)
    if b is not None:
        return "\\boxed{" + b + "}"
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def score_math25(content, item):
    raw = extract_math_answer(content)
    pred = normalize_math(raw)
    gt = normalize_math(item["answer"])
    ok = pred != "" and pred == gt
    if not ok:
        a, b = _parse_number(pred), _parse_number(gt)
        if a is not None and b is not None:
            ok = abs(a - b) <= 1e-6 * max(1.0, abs(b))
    # log both sides for manual audit of near-misses
    return ok, {"expected_answer": item["answer"], "predicted_answer": raw,
                "expected_norm": gt, "predicted_norm": pred}


SCORERS = {"gsm8k": score_gsm8k, "mmlu": score_mmlu,
           "math25": score_math25, "needle": score_needle}

# ------------------------------------------------------------------ client (verbatim from run_eval.py)

def chat_completion(base_url, prompt, args, use_ctk=True):
    """One request. Returns (response_dict, ctk_dropped:bool). Raises on failure."""
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": args.max_tokens,
    }
    if use_ctk and args.reasoning_effort:
        body["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    if getattr(args, "extra_body", None):
        body.update(args.extra_body)  # e.g. logit_bias experiments; recorded in the summary
    data = json.dumps(body).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})

    def _send():
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return json.loads(r.read())

    try:
        return _send(), False
    except urllib.error.HTTPError as e:
        if use_ctk and args.reasoning_effort and e.code in (400, 422, 500):
            # server may reject chat_template_kwargs -> retry once without it
            resp, _ = chat_completion(base_url, prompt, args, use_ctk=False)
            return resp, True
        raise
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        time.sleep(2)  # one retry on connection error
        return _send(), False


def estimate_thinking_tokens(message):
    """Tokens before </think> — from reasoning_content or inline in content."""
    think_text = None
    rc = message.get("reasoning_content")
    if rc:
        think_text = rc
    else:
        content = message.get("content") or ""
        if "</think>" in content:
            think_text = content.split("</think>", 1)[0]
    if not think_text:
        return 0
    return int(len(think_text.split()) * 1.3)


def visible_content(message):
    """Answer text with any inline <think>...</think> stripped."""
    content = message.get("content") or ""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    return content.strip()

# ------------------------------------------------------------------ fork additions (no row changes)

def request_body_bytes(prompt, args, use_ctk=True):
    """The exact bytes chat_completion() sends (check_fidelity.py verifies the equality)."""
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": args.max_tokens,
    }
    if use_ctk and args.reasoning_effort:
        body["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    if getattr(args, "extra_body", None):
        body.update(args.extra_body)
    return json.dumps(body).encode("utf-8")


THINK_END = "</think>"   # Qwen3.8's end-of-reasoning tag (run_eval.py's thinking estimate splits on the same string)


def laptop_inline_text(msg, finish_reason):
    """(text, how): the text E47's server would have returned in content for this response.

    E47's server (lab TabbyAPI 4a4f9f4, reasoning parser off) returned everything inline: content =
    '<reasoning>\\n</think>\\n\\n<answer>' (the template's generation prompt opens the <think> block), reasoning_content
    empty. Upstream TabbyAPI b1c2253 (TagStreamParser) and llama-server (deepseek format) consume the end tag and
    split the text. `how`:
      'as_returned'     no reasoning_content: the server returned the text inline (exact);
      'joined'          reasoning_content + '</think>' + content (exact for TabbyAPI, which keeps the whitespace after
                        the tag in content; llama-server trims whitespace there, so approximate for it);
      'reasoning_only'  no content and finish_reason 'length': truncated inside the reasoning, no end tag (exact
                        unless the cap fell just after the tag);
      'tag_appended'    no content, not truncated: the tag is restored, whitespace after it is lost (approximate).
    """
    rc = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    if not rc:
        return content, "as_returned"
    if content:
        return rc + THINK_END + content, "joined"
    if finish_reason == "length":
        return rc, "reasoning_only"
    return rc + THINK_END, "tag_appended"


_ENCODE_OK = {}  # base_url -> False once the server shows it has no /v1/token/encode


def encode_count(base_url, text):
    """Token count of `text` from TabbyAPI's /v1/token/encode (what the laptop's E47 rows stored); None if n/a."""
    if _ENCODE_OK.get(base_url) is False:
        return None
    try:
        ereq = urllib.request.Request(
            base_url.rstrip("/") + "/v1/token/encode",
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(ereq, timeout=60) as er:
            ed = json.loads(er.read().decode("utf-8"))
        n = ed.get("length") or len(ed.get("tokens") or [])
        _ENCODE_OK[base_url] = True
        return n or None
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404, 405, 501):
            _ENCODE_OK[base_url] = False  # llama-server has no such route; stop asking
        return None
    except Exception:
        return None


def verify_dataset(suite, datasets_dir):
    """sha256 of the dataset file against the laptop's; returns (ok, got, want)."""
    name = DATASET_FILES[suite]
    h = hashlib.sha256()
    with open(os.path.join(datasets_dir, name), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got, want = h.hexdigest(), DATASET_SHA256.get(name)
    return got == want, got, want


def load_dataset(suite, limit, datasets_dir=None):
    path = os.path.join(datasets_dir or DATASETS, DATASET_FILES[suite])
    with open(path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    return items[:limit] if limit else items


def run_item_full(suite, item, args):
    """run_eval.run_item's row (same code, same order of keys) plus a full-response record for the sidecar."""
    prompt = build_prompt(suite, item)
    t0 = time.time()
    row = {"id": item["id"]}
    resp, choice, msg, ctk_dropped, encode_n = None, None, None, False, None
    try:
        resp, ctk_dropped = chat_completion(args.base_url, prompt, args)
        wall = time.time() - t0
        choice = resp["choices"][0]
        msg = choice.get("message", {})
        content = visible_content(msg)
        usage = resp.get("usage") or {}
        if usage.get("completion_tokens") is None:
            # TabbyAPI returns usage: null; count the returned text with its /v1/token/encode
            # endpoint so completion_tokens is comparable across engines (2026-09-06).
            try:
                txt = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
                ereq = urllib.request.Request(
                    args.base_url.rstrip("/") + "/v1/token/encode",
                    data=json.dumps({"text": txt}).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(ereq, timeout=60) as er:
                    ed = json.loads(er.read().decode("utf-8"))
                n_enc = ed.get("length") or len(ed.get("tokens") or [])
                if n_enc:
                    usage = dict(usage); usage["completion_tokens"] = n_enc; usage["counted_via_encode"] = True
                    encode_n = n_enc
            except Exception:
                pass
        verdict = SCORERS[base_suite(suite)](content, item)
        extras = {}
        if isinstance(verdict, tuple):
            verdict, extras = verdict
        row.update({
            "correct": bool(verdict),
            "wall_s": round(wall, 2),
            "completion_tokens": usage.get("completion_tokens"),
            "thinking_tokens_estimate": estimate_thinking_tokens(msg),
            "truncated": choice.get("finish_reason") == "length",
            "raw_answer_excerpt": content[-200:],
            # full-output fingerprint (reasoning + answer) for temperature-0 identity checks across engines/configs
            "content_sha256": hashlib.sha256(
                ((msg.get("reasoning_content") or "") + (msg.get("content") or "")).encode("utf-8")).hexdigest(),
            "content_chars": len((msg.get("reasoning_content") or "") + (msg.get("content") or "")),
        })
        row.update(extras)
        if ctk_dropped:
            row["chat_template_kwargs_dropped"] = True
    except Exception as e:
        row.update({
            "correct": False, "wall_s": round(time.time() - t0, 2),
            "completion_tokens": None, "thinking_tokens_estimate": 0,
            "truncated": False, "raw_answer_excerpt": "",
            "error": f"{type(e).__name__}: {e}"[:300],
        })
    full = {"id": item["id"], "suite": suite, "label": getattr(args, "label", None),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_s": row["wall_s"],
            "request_sha256": hashlib.sha256(request_body_bytes(prompt, args, use_ctk=not ctk_dropped)).hexdigest()}
    if "error" in row:
        full["error"] = row["error"]
        return row, full
    try:
        rusage = resp.get("usage")
        if encode_n is None and getattr(args, "encode_count", True):
            encode_n = encode_count(args.base_url, (msg.get("reasoning_content") or "") + (msg.get("content") or ""))
        full.update({
            "finish_reason": choice.get("finish_reason"),
            "reasoning_content": msg.get("reasoning_content"),
            "content": msg.get("content"),
            "usage": rusage,
            "timings": resp.get("timings"),
            "model": resp.get("model"),
            "encode_tokens": encode_n,
            "completion_tokens_source": ("usage" if isinstance(rusage, dict) and rusage.get("completion_tokens") is not None
                                         else ("encode" if row.get("completion_tokens") is not None else None)),
            "tool_calls": msg.get("tool_calls"),
        })
        # Response shape (E47's server answered inline; see laptop_inline_text). For a split response, also record
        # the laptop-shape text's fingerprint, encode count and score: row fields stay exactly run_eval.py's.
        inline, how = laptop_inline_text(msg, choice.get("finish_reason"))
        full["response_shape"] = "split" if msg.get("reasoning_content") else "inline"
        full["laptop_inline"] = how
        full["inline_sha256"] = hashlib.sha256(inline.encode("utf-8")).hexdigest()
        if full["response_shape"] == "split":
            if getattr(args, "encode_count", True):
                full["inline_encode_tokens"] = encode_count(args.base_url, inline)
            v = SCORERS[base_suite(suite)](visible_content({"content": inline}), item)
            full["inline_correct"] = bool(v[0] if isinstance(v, tuple) else v)
    except Exception as e:  # the sidecar must never change or lose the row
        full["sidecar_error"] = f"{type(e).__name__}: {e}"[:300]
    return row, full


def run_item(suite, item, args):
    """Same signature and row as run_eval.run_item."""
    return run_item_full(suite, item, args)[0]


class ServerGone(Exception):
    """Too many failed requests in a row: the lane's server is down or was stopped by the guard."""


def load_latest_rows(path):
    """Latest row per id (later lines win, as hardening_analyze.load does); torn lines are skipped."""
    latest = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and "id" in r:
                    latest[r["id"]] = r
    return latest


def _ensure_newline(path):
    """If a hard stop left a partial last line, start the next append on a fresh line."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")


def run_suite(suite, args):
    items = load_dataset(suite, args.limit, args.datasets_dir)
    outdir = os.path.join(args.results_dir, args.label)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{suite}.jsonl")
    fullpath = os.path.join(outdir, f"{suite}.full.jsonl")
    latest = load_latest_rows(outpath)
    done = {i for i, r in latest.items() if "error" not in r}
    todo = [it for it in items if it["id"] not in done]
    print(f"\n=== {suite}: {len(items)} items, {len(items) - len(todo)} already done, "
          f"{len(todo)} to run -> {outpath}", flush=True)
    wanted = [it["id"] for it in items]
    session = []
    lock = threading.Lock()
    _ensure_newline(outpath)
    outf = open(outpath, "a", encoding="utf-8")
    fullf = None
    if args.full_text:
        _ensure_newline(fullpath)
        fullf = open(fullpath, "a", encoding="utf-8")
    streak = [0]

    def record(row, full):
        with lock:
            session.append(row)
            latest[row["id"]] = row
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            outf.flush()
            if fullf is not None:
                fullf.write(json.dumps(full, ensure_ascii=False) + "\n")
                fullf.flush()
            have = [latest[i] for i in wanted if i in latest]
            n = len(have)
            acc = sum(r["correct"] for r in have) / n
            ct = row.get("completion_tokens") or 0
            tps = ct / row["wall_s"] if row["wall_s"] > 0 and ct else 0.0
            status = "ok  " if row["correct"] else "FAIL"
            extra = " [ERR]" if "error" in row else (
                " [trunc]" if row.get("truncated") else "")
            print(f"[{n}/{len(items)}] {row['id']:<28} {status} "
                  f"acc={acc:.3f} {row['wall_s']:7.1f}s "
                  f"{tps:5.1f} tok/s{extra}", flush=True)
            streak[0] = streak[0] + 1 if "error" in row else 0
            if args.max_consecutive_errors and streak[0] >= args.max_consecutive_errors:
                raise ServerGone(f"{streak[0]} failed requests in a row; last: {row.get('error')}")

    interrupted, stopped = False, None
    try:
        if args.concurrency <= 1:
            for item in todo:
                record(*run_item_full(suite, item, args))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(run_item_full, suite, it, args): it for it in todo}
                try:
                    for fut in as_completed(futs):
                        record(*fut.result())
                except (KeyboardInterrupt, ServerGone):
                    for f in futs:
                        f.cancel()
                    raise
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted: keeping {len(session)} new rows", flush=True)
    except ServerGone as e:
        stopped = str(e)
        print(f"\nstopped: {stopped}", flush=True)
    finally:
        outf.close()
        if fullf is not None:
            fullf.close()

    rows = [latest[i] for i in wanted if i in latest]
    summary = None
    if rows:
        n = len(rows)
        wall = [r["wall_s"] for r in rows]
        toks = [r["completion_tokens"] for r in rows
                if r["completion_tokens"] is not None]
        mean_wall = sum(wall) / n
        summary = {
            "label": args.label,
            "suite": suite,
            "n": n,
            "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
            "mean_wall_s": round(mean_wall, 2),
            "total_wall_s": round(sum(wall), 2),
            "mean_completion_tokens":
                round(sum(toks) / len(toks), 1) if toks else None,
            "accuracy_per_minute":
                round(sum(r["correct"] for r in rows) / n / (mean_wall / 60), 4)
                if mean_wall > 0 else None,
            "reasoning_effort": args.reasoning_effort,
            "extra_body_file": getattr(args, "extra_body_file", None),
            "partial": interrupted or stopped is not None or n < len(items),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # fork additions
            "harness": "run_eval_turing",
            "items_planned": len(items),
            "errors": sum(1 for r in rows if "error" in r),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "rows_this_session": len(session),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "base_url": args.base_url,
            "dataset_sha256": DATASET_SHA256.get(DATASET_FILES[suite]),
            "stopped": stopped,
        }
        with open(os.path.join(args.results_dir, "summary.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"--- {suite} summary: acc={summary['accuracy']} n={n} "
              f"mean_wall={summary['mean_wall_s']}s "
              f"acc/min={summary['accuracy_per_minute']} errors={summary['errors']}", flush=True)
    if getattr(args, "summary_out", None):
        with open(args.summary_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary or {"label": args.label, "suite": suite, "n": 0,
                                           "stopped": stopped, "partial": True}) + "\n")
    if interrupted:
        raise KeyboardInterrupt
    if stopped:
        raise ServerGone(stopped)
    return summary

# ------------------------------------------------------------------ selftest (run_eval's, minus code execution)

def selftest():
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
        if not ok:
            failures.append(name)

    print("selftest: gsm8k scorer")
    g = {"answer_number": 1234.0}
    check("gsm8k correct", score_gsm8k(
        "<reasoning> 600+634 </reasoning>\nSo the total.\nAnswer: $1,234", g), True)
    check("gsm8k correct bare-number fallback", score_gsm8k(
        "Working it out step by step we get\n1234", g), True)
    check("gsm8k wrong", score_gsm8k("Let me see...\nAnswer: 99", g), False)
    check("gsm8k malformed", score_gsm8k(
        "I cannot determine the answer to this.", g), False)

    print("selftest: mmlu scorer")
    m = {"answer_idx": 2}  # C
    check("mmlu correct", score_mmlu(
        "B looks tempting but no.\nAnswer: (C)", m), True)
    check("mmlu correct last-lines fallback", score_mmlu(
        "After elimination,\nthe best option is C.", m), True)
    check("mmlu wrong", score_mmlu("Answer: A", m), False)
    check("mmlu malformed", score_mmlu(
        "All of these options seem plausible to me.", m), False)

    print("selftest: needle scorer")
    n = {"expected_substring": "Ilsa Brannigan"}
    check("needle correct", score_needle(
        "The telescope was donated by ILSA BRANNIGAN in 1907.", n), True)
    check("needle wrong", score_needle(
        "It was donated by the town council.", n), False)
    check("needle malformed", score_needle("", n), False)

    print("selftest: math25 normalizer + scorer")
    mt = {"answer": "\\frac{3}{4}"}
    ok, ex = score_math25("Some work.\nAnswer: 3/4", mt)
    check("math boxed-frac vs slash", ok, True)
    ok, _ = score_math25("Thus $\\boxed{\\dfrac{3}{4}}$.\nAnswer: \\frac{3}{4}",
                         mt)
    check("math frac vs frac (boxed in-line)", ok, True)
    mi = {"answer": "42"}
    ok, _ = score_math25("Answer: 42.0", mi)
    check("math integer equality (42 vs 42.0)", ok, True)
    ok, ex = score_math25("Answer: 41", mi)
    check("math wrong number", ok, False)
    ms = {"answer": "\\text{east}"}
    ok, ex = score_math25("Answer: west", ms)
    check("math plain string mismatch", ok, False)
    check("math audit fields logged",
          sorted(ex.keys()), ["expected_answer", "expected_norm",
                              "predicted_answer", "predicted_norm"])
    ok, _ = score_math25("I could not solve this problem.", mi)
    check("math malformed (no Answer marker)", ok, False)
    check("math normalize left/right+spacing",
          normalize_math("\\left( 3, \\frac{\\pi}{2} \\right)"),
          normalize_math("(3,\\pi/2)"))

    print("selftest: thinking-token estimate + content stripping")
    msg = {"content": "<think>one two three four</think>\nAnswer: 5"}
    check("thinking estimate inline", estimate_thinking_tokens(msg), 5)
    check("visible content strips think",
          visible_content(msg), "Answer: 5")
    msg2 = {"content": "Answer: 5", "reasoning_content": "a b c"}
    check("thinking estimate reasoning_content",
          estimate_thinking_tokens(msg2), 3)

    if failures:
        print(f"\nSELFTEST FAILED: {failures}")
        return 1
    print("\nSELFTEST PASSED (all checks)")
    return 0

# --------------------------------------------------------------------- main

def _interrupt(signum, frame):
    raise KeyboardInterrupt


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES + list(HARDENING_SUITES) + ["all"])
    ap.add_argument("--label", help="config label, e.g. TURING-exl3-mtp")
    ap.add_argument("--base-url", default="http://localhost:8090")
    ap.add_argument("--reasoning-effort", default="medium",
                    help="passed via chat_template_kwargs; '' to disable")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature (default 1.0 = campaign standard; E47 used 0)")
    ap.add_argument("--limit", type=int, default=0, help="run first N items only")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--extra-body-file", default=None,
                    help="JSON file whose top-level keys are merged into every request body "
                         "(e.g. {\"logit_bias\": [...]}); label the run accordingly")
    ap.add_argument("--selftest", action="store_true",
                    help="run scorer checks (no server needed)")
    # fork additions
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--datasets-dir", default=DATASETS)
    ap.add_argument("--no-full-text", dest="full_text", action="store_false",
                    help="skip the <suite>.full.jsonl sidecar")
    ap.add_argument("--no-encode-count", dest="encode_count", action="store_false",
                    help="skip the sidecar's /v1/token/encode count when the server reports usage")
    ap.add_argument("--max-consecutive-errors", type=int, default=3,
                    help="stop the suite after this many failed requests in a row (0 = never)")
    ap.add_argument("--summary-out", default=None, help="also append the suite summary JSON to this file")
    ap.add_argument("--allow-unverified-dataset", action="store_true",
                    help="run even if a dataset's sha256 differs from the laptop's (label the run accordingly)")
    ap.add_argument("--todo", action="store_true",
                    help="print how many items still lack an error-free row (no server needed) and exit")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.extra_body = None
    if args.extra_body_file:
        with open(args.extra_body_file) as f:
            args.extra_body = json.load(f)

    if args.selftest:
        sys.exit(selftest())
    if not args.suite or not args.label:
        ap.error("--suite and --label are required (or use --selftest)")
    if "/" in args.label:
        ap.error("--label must not contain '/'")

    suites = SUITES if args.suite == "all" else [args.suite]
    for s in suites:
        ok, got, want = verify_dataset(s, args.datasets_dir)
        if not ok:
            msg = f"dataset {DATASET_FILES[s]} sha256 {got} != laptop {want}"
            if not args.allow_unverified_dataset:
                print("REFUSED: " + msg, file=sys.stderr)
                sys.exit(65)
            print("WARNING (allowed): " + msg, file=sys.stderr)
    if args.todo:
        left = 0
        for s in suites:
            latest = load_latest_rows(os.path.join(args.results_dir, args.label, f"{s}.jsonl"))
            left += sum(1 for it in load_dataset(s, args.limit, args.datasets_dir)
                        if "error" in latest.get(it["id"], {"error": "no row"}))
        print(left)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGHUP, _interrupt)
    try:
        for s in suites:
            run_suite(s, args)
    except KeyboardInterrupt:
        sys.exit(130)
    except ServerGone:
        sys.exit(3)


if __name__ == "__main__":
    main()
