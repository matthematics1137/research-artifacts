#!/usr/bin/env python3
"""Reasoning-eval harness for llama-server (OpenAI-compatible /v1/chat/completions).

Usage:
  ./run_eval.py --suite gsm8k|mmlu|humaneval|needle|all --label <config-label>
                [--base-url http://127.0.0.1:8090] [--reasoning-effort medium]
                [--max-tokens 4096] [--limit N] [--concurrency 1]
  ./run_eval.py --selftest        # scorer + sandbox checks, no server needed

Stdlib only. Per-item results stream to results/<label>/<suite>.jsonl; a
summary line is appended to results/summary.jsonl at the end of each suite.
Ctrl-C keeps partial results and still writes the summary for completed items.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "datasets")
RESULTS = os.environ.get("EVAL_RESULTS_DIR", os.path.join(HERE, "results"))
HE_RUN_DIR = os.environ.get(
    "HE_RUN_DIR", os.path.join(tempfile.gettempdir(), "qwen38_eval_runs")
)
SUITES = ["gsm8k", "mmlu", "humaneval", "humaneval_plus", "math25", "needle"]
DATASET_FILES = {"gsm8k": "gsm8k_100.jsonl", "mmlu": "mmlu_200.jsonl",
                 "humaneval": "humaneval_20.jsonl",
                 "humaneval_plus": "humaneval_plus_20.jsonl",
                 "math25": "math25.jsonl", "needle": "needle.jsonl"}
REQUEST_TIMEOUT_S = 600

# ------------------------------------------------------------------ prompts

def build_prompt(suite, item):
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

# ------------------------------------------------------------------ scoring

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


def extract_python_block(content):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", content, re.DOTALL)
    if blocks:
        return blocks[-1]
    return None


def score_humaneval(content, item, keep_dir=None, timeout=10):
    code = extract_python_block(content)
    if code is None:
        return False
    if f"def {item['entry_point']}" not in code:
        # model returned only a body/continuation: graft onto the prompt
        code = item["prompt"] + "\n" + code
    program = (code + "\n\n" + item["test"] + "\n\n"
               f"check({item['entry_point']})\n")
    os.makedirs(HE_RUN_DIR, exist_ok=True)
    tmpd = tempfile.mkdtemp(prefix=item["id"] + "_", dir=HE_RUN_DIR)
    path = os.path.join(tmpd, "prog.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(program)
    try:
        proc = subprocess.run(
            [sys.executable if os.path.exists(sys.executable) else "python3",
             path],
            cwd=tmpd, env={"PATH": "/usr/bin:/bin"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    finally:
        if keep_dir is None:
            shutil.rmtree(tmpd, ignore_errors=True)
    return ok


def score_needle(content, item):
    return item["expected_substring"].lower() in content.lower()


# ------------------------------------------------------------- MATH scoring

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
           "humaneval": score_humaneval,
           "humaneval_plus":
               lambda content, item: score_humaneval(content, item, timeout=30),
           "math25": score_math25, "needle": score_needle}

# ------------------------------------------------------------------ client

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

# ------------------------------------------------------------------ running

def load_dataset(suite, limit):
    path = os.path.join(DATASETS, DATASET_FILES[suite])
    with open(path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    return items[:limit] if limit else items


def run_item(suite, item, args):
    prompt = build_prompt(suite, item)
    t0 = time.time()
    row = {"id": item["id"]}
    try:
        resp, ctk_dropped = chat_completion(args.base_url, prompt, args)
        wall = time.time() - t0
        choice = resp["choices"][0]
        msg = choice.get("message", {})
        content = visible_content(msg)
        usage = resp.get("usage") or {}
        verdict = SCORERS[suite](content, item)
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
    return row


def run_suite(suite, args):
    items = load_dataset(suite, args.limit)
    outdir = os.path.join(RESULTS, args.label)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{suite}.jsonl")
    print(f"\n=== {suite}: {len(items)} items -> {outpath}")
    rows = []
    lock = threading.Lock()
    outf = open(outpath, "w", encoding="utf-8")

    def record(row):
        with lock:
            rows.append(row)
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            outf.flush()
            n = len(rows)
            acc = sum(r["correct"] for r in rows) / n
            ct = row.get("completion_tokens") or 0
            tps = ct / row["wall_s"] if row["wall_s"] > 0 and ct else 0.0
            status = "ok  " if row["correct"] else "FAIL"
            extra = " [ERR]" if "error" in row else (
                " [trunc]" if row.get("truncated") else "")
            print(f"[{n}/{len(items)}] {row['id']:<28} {status} "
                  f"acc={acc:.3f} {row['wall_s']:7.1f}s "
                  f"{tps:5.1f} tok/s{extra}", flush=True)

    interrupted = False
    try:
        if args.concurrency <= 1:
            for item in items:
                record(run_item(suite, item, args))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {ex.submit(run_item, suite, it, args): it for it in items}
                try:
                    for fut in as_completed(futs):
                        record(fut.result())
                except KeyboardInterrupt:
                    for f in futs:
                        f.cancel()
                    raise
    except KeyboardInterrupt:
        interrupted = True
        print(f"\ninterrupted: keeping {len(rows)} partial results", flush=True)
    finally:
        outf.close()

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
            "partial": interrupted,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(RESULTS, "summary.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"--- {suite} summary: acc={summary['accuracy']} n={n} "
              f"mean_wall={summary['mean_wall_s']}s "
              f"acc/min={summary['accuracy_per_minute']}")
    if interrupted:
        raise KeyboardInterrupt

# ------------------------------------------------------------------ selftest

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

    print("selftest: humaneval sandbox (known-good / known-bad / malformed)")
    he = {
        "id": "selftest_add",
        "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        "entry_point": "add",
        "test": ("def check(candidate):\n"
                 "    assert candidate(1, 2) == 3\n"
                 "    assert candidate(-4, 4) == 0\n"),
    }
    good = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    malformed = "I would just use the + operator, no code needed."
    check("humaneval good", score_humaneval(good, he), True)
    check("humaneval bad", score_humaneval(bad, he), False)
    check("humaneval malformed (no code block)",
          score_humaneval(malformed, he), False)
    body_only = "```python\n    return a + b\n```"
    check("humaneval body-only graft", score_humaneval(body_only, he), True)
    hang = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    t0 = time.time()
    check("humaneval infinite-loop timeout", score_humaneval(hang, he), False)
    print(f"        (timeout path took {time.time()-t0:.1f}s, limit 10s)")

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

    print("selftest: humaneval_plus synthesized battery (pass/fail pair)")
    hp = {
        "id": "selftest_hp_add",
        "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        "entry_point": "add",
        "test": ("import copy as _copy\n"
                 "_inputs = [[i, i * 7] for i in range(250)]\n"
                 "_expected = [i * 8 for i in range(250)]\n"
                 "def _norm(x):\n"
                 "    if isinstance(x, (list, tuple)):\n"
                 "        return [_norm(e) for e in x]\n"
                 "    return x\n"
                 "def check(candidate):\n"
                 "    for _args, _exp in zip(_inputs, _expected):\n"
                 "        _out = candidate(*_copy.deepcopy(_args))\n"
                 "        assert _norm(_out) == _norm(_exp)\n"),
    }
    hp_scorer = SCORERS["humaneval_plus"]
    check("humaneval_plus good (250 cases)",
          hp_scorer("```python\ndef add(a, b):\n    return a + b\n```", hp),
          True)
    check("humaneval_plus bad (off-by-one impl)",
          hp_scorer("```python\ndef add(a, b):\n    return a + b + 1\n```", hp),
          False)

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

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=SUITES + ["all"])
    ap.add_argument("--label", help="config label, e.g. UD-IQ2_S-ngl99-mtp-effmedium")
    ap.add_argument("--base-url", default="http://127.0.0.1:8090")
    ap.add_argument("--reasoning-effort", default="medium",
                    help="passed via chat_template_kwargs; '' to disable")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature (default 1.0 = campaign standard)")
    ap.add_argument("--limit", type=int, default=0, help="run first N items only")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--selftest", action="store_true",
                    help="run scorer/sandbox checks (no server needed)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.suite or not args.label:
        ap.error("--suite and --label are required (or use --selftest)")
    if "/" in args.label:
        ap.error("--label must not contain '/'")

    suites = SUITES if args.suite == "all" else [args.suite]
    try:
        for s in suites:
            run_suite(s, args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
