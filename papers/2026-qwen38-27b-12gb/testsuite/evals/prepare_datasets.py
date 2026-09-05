#!/usr/bin/env python3
"""Build the eval datasets under datasets/ (stdlib only, deterministic, seed 42).

Sources:
  gsm8k_100.jsonl         <- github.com/openai/grade-school-math (test.jsonl, 1319 rows)
  mmlu_200.jsonl          <- HF datasets-server API, cais/mmlu config=all split=test (14042 rows)
  humaneval_20.jsonl      <- github.com/openai/human-eval (HumanEval.jsonl.gz, 164 rows)
  humaneval_plus_20.jsonl <- github.com/evalplus/humanevalplus_release v0.1.10 (same 20
                             problems as humaneval_20; runnable extended tests synthesized
                             from base_input+plus_input via the efficient reference
                             canonical, validated in the sandbox)
  math25.jsonl            <- HF datasets-server API, nlile/hendrycks-MATH-benchmark
                             test split (500 rows), levels 4-5, stratified over subjects
  needle.jsonl            <- generated locally (6 items: {8k,16k,32k} tokens x {25%,75%} position)

Downloads are cached in CACHE_DIR so re-runs are cheap. Sampling is
random.Random(42) over the full source -> stable across re-runs.
"""
import gzip
import json
import os
import random
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "datasets")
CACHE_DIR = os.environ.get(
    "EVAL_PREP_CACHE",
    os.path.join("/tmp", "qwen38_eval_prep_cache"),
)
SEED = 42
UA = {"User-Agent": "evals-prep/1.0 (stdlib urllib)"}


def fetch(url, dest, binary=False):
    """Download url to dest (cached). Returns dest path."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            return dest
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt+1} for {url}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------- GSM8K
def build_gsm8k():
    src = fetch(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/test.jsonl",
        os.path.join(CACHE_DIR, "gsm8k_test.jsonl"),
    )
    with open(src, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert len(rows) == 1319, f"expected 1319 GSM8K test rows, got {len(rows)}"
    idxs = sorted(random.Random(SEED).sample(range(len(rows)), 100))
    out = []
    for i in idxs:
        r = rows[i]
        m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", r["answer"])
        assert m, f"no '#### N' marker in GSM8K row {i}"
        num = float(m.group(1).replace(",", ""))
        out.append({"id": f"gsm8k_{i}", "question": r["question"].strip(),
                    "answer_number": num})
    write_jsonl(os.path.join(OUT, "gsm8k_100.jsonl"), out)


# ----------------------------------------------------------------- MMLU
def fetch_mmlu_all():
    """Fetch all 14042 rows of cais/mmlu all/test via datasets-server.

    Resumable: pages are appended to the cache file as they arrive, so a
    rate-limit abort continues where it left off. Requests are paced and 429s
    honor Retry-After.
    """
    cache = os.path.join(CACHE_DIR, "mmlu_all_test.jsonl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    rows = []
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if len(rows) >= 14042:
            return rows
        print(f"  mmlu fetch: resuming from {len(rows)} cached rows")
    base = ("https://datasets-server.huggingface.co/rows?"
            "dataset=cais%2Fmmlu&config=all&split=test")
    total = 14042
    with open(cache, "a", encoding="utf-8") as cf:
        while len(rows) < total:
            url = f"{base}&offset={len(rows)}&length=100"
            page = None
            for attempt in range(8):
                try:
                    req = urllib.request.Request(url, headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as r:
                        page = json.loads(r.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        wait = e.headers.get("Retry-After")
                        wait = int(wait) if wait and wait.isdigit() else 25
                        print(f"  429 at offset={len(rows)}, sleeping {wait}s "
                              f"(attempt {attempt+1}/8)", file=sys.stderr)
                        time.sleep(wait)
                    else:
                        raise
                except Exception as e:
                    print(f"  retry offset={len(rows)}: {e}", file=sys.stderr)
                    time.sleep(3 * (attempt + 1))
            if page is None:
                raise RuntimeError(f"gave up at offset={len(rows)}; "
                                   "re-run to resume from cache")
            total = page["num_rows_total"]
            for rr in page["rows"]:
                row = {"row_idx": rr["row_idx"], **rr["row"]}
                rows.append(row)
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")
            cf.flush()
            if len(rows) % 2000 < 100:
                print(f"  mmlu fetch: {len(rows)}/{total}")
            time.sleep(0.8)  # pace: stay under the rate limit
    assert len(rows) == total, f"mmlu fetch incomplete: {len(rows)}/{total}"
    return rows


def build_mmlu():
    rows = fetch_mmlu_all()
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)
    rng = random.Random(SEED)
    subjects = sorted(by_subj)  # 57 subjects
    # Stratified: 3 per subject (57*3=171), then 29 more from the remainder.
    picked = []
    for s in subjects:
        picked.extend(rng.sample(by_subj[s], min(3, len(by_subj[s]))))
    chosen_ids = {r["row_idx"] for r in picked}
    remainder = [r for r in rows if r["row_idx"] not in chosen_ids]
    picked.extend(rng.sample(remainder, 200 - len(picked)))
    picked.sort(key=lambda r: r["row_idx"])
    out = []
    for r in picked:
        assert len(r["choices"]) == 4 and r["answer"] in (0, 1, 2, 3)
        out.append({"id": f"mmlu_{r['row_idx']}", "subject": r["subject"],
                    "question": r["question"].strip(), "choices": r["choices"],
                    "answer_idx": r["answer"]})
    write_jsonl(os.path.join(OUT, "mmlu_200.jsonl"), out)
    print(f"  subjects covered: {len({o['subject'] for o in out})}")


# ------------------------------------------------------------- HumanEval
def build_humaneval():
    src = fetch(
        "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        os.path.join(CACHE_DIR, "HumanEval.jsonl.gz"),
    )
    with gzip.open(src, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert len(rows) == 164, f"expected 164 HumanEval rows, got {len(rows)}"
    idxs = sorted(random.Random(SEED).sample(range(len(rows)), 20))
    out = []
    for i in idxs:
        r = rows[i]
        assert r["prompt"] and r["entry_point"] and r["test"]
        out.append({"id": r["task_id"].replace("/", "_"), "prompt": r["prompt"],
                    "entry_point": r["entry_point"], "test": r["test"]})
    write_jsonl(os.path.join(OUT, "humaneval_20.jsonl"), out)


# --------------------------------------------------------- HumanEval-Plus
EXPECTED_RUNNER = r"""
import copy, json, sys
sys.set_int_max_str_digits(0)  # some outputs exceed the 4300-digit str limit
spec = json.load(sys.stdin)
ns = {}
exec(spec["prompt"] + spec["canonical_solution"], ns)
fn = ns[spec["entry_point"]]
out = [fn(*copy.deepcopy(args)) for args in spec["inputs"]]
json.dump(out, sys.stdout)
"""

# Huge ints (special_factorial answers reach 263k digits) are embedded as hex
# string literals: decimal literals >4300 digits are rejected at compile time
# by CPython's int/str conversion limit, hex is exempt. The assert avoids
# repr() of values for the same reason.
HP_TEST_TEMPLATE = """import copy as _copy

_inputs = {inputs!r}
_expected_raw = {expected!r}

def _dec(v):
    if isinstance(v, dict) and "__bighex__" in v:
        return int(v["__bighex__"], 16)
    if isinstance(v, list):
        return [_dec(e) for e in v]
    return v

_expected = [_dec(v) for v in _expected_raw]

def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(e) for e in x]
    return x

def check(candidate):
    assert len(_inputs) == len(_expected)
    for _i, (_args, _exp) in enumerate(zip(_inputs, _expected)):
        _out = candidate(*_copy.deepcopy(_args))
        if _norm(_out) != _norm(_exp):
            raise AssertionError("case %d failed" % _i)
"""


def _enc_big(v):
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) > 10 ** 3000:
        return {"__bighex__": hex(v)}
    if isinstance(v, list):
        return [_enc_big(e) for e in v]
    return v


def build_humaneval_plus():
    import subprocess
    sys.set_int_max_str_digits(0)  # parse huge decimal ints from runner JSON
    src = fetch(
        "https://github.com/evalplus/humanevalplus_release/releases/download/"
        "v0.1.10/HumanEvalPlus.jsonl.gz",
        os.path.join(CACHE_DIR, "HumanEvalPlus.jsonl.gz"),
    )
    with gzip.open(src, "rt", encoding="utf-8") as f:
        plus = {r["task_id"].replace("/", "_"): r
                for r in (json.loads(l) for l in f if l.strip())}
    assert len(plus) == 164, f"expected 164 HumanEvalPlus rows, got {len(plus)}"
    # SAME 20 problems as humaneval_20.jsonl, matched by task_id
    with open(os.path.join(OUT, "humaneval_20.jsonl"), encoding="utf-8") as f:
        base20 = [json.loads(l) for l in f if l.strip()]
    sys.path.insert(0, HERE)
    from run_eval import score_humaneval  # sandbox validation
    out = []
    for b in base20:
        r = plus[b["id"]]
        assert r["entry_point"] == b["entry_point"]
        assert r.get("atol", 0) == 0, f"{b['id']}: atol != 0 needs float compare"
        inputs = r["base_input"] + r["plus_input"]
        spec = json.dumps({"prompt": r["prompt"],
                           "canonical_solution": r["canonical_solution"],
                           "entry_point": r["entry_point"], "inputs": inputs})
        proc = subprocess.run([sys.executable, "-c", EXPECTED_RUNNER],
                              input=spec, capture_output=True, text=True,
                              timeout=120)
        assert proc.returncode == 0, f"{b['id']} expected-runner failed:\n" \
                                     f"{proc.stderr[-800:]}"
        expected = _enc_big(json.loads(proc.stdout))
        test = HP_TEST_TEMPLATE.format(inputs=inputs, expected=expected)
        row = {"id": b["id"], "prompt": r["prompt"],
               "entry_point": r["entry_point"], "test": test}
        # validate: reference solution must pass the synthesized battery,
        # a broken candidate must fail it
        t0 = time.time()
        good = "```python\n" + r["prompt"] + r["canonical_solution"] + "\n```"
        assert score_humaneval(good, row, timeout=30) is True, \
            f"{b['id']}: canonical does not pass synthesized test"
        dt = time.time() - t0
        bad = ("```python\n" + r["prompt"] +
               "    raise ValueError('broken')\n```")
        assert score_humaneval(bad, row, timeout=30) is False, \
            f"{b['id']}: broken candidate passes synthesized test"
        print(f"  {b['id']:<16} {len(inputs):>5} cases, canonical ok "
              f"in {dt:4.1f}s")
        out.append(row)
    write_jsonl(os.path.join(OUT, "humaneval_plus_20.jsonl"), out)


# ------------------------------------------------------------------- MATH
def fetch_math500():
    cache = os.path.join(CACHE_DIR, "math500_test.jsonl")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if len(rows) == 500:
            return rows
    rows = []
    base = ("https://datasets-server.huggingface.co/rows?"
            "dataset=nlile%2Fhendrycks-MATH-benchmark&config=default&split=test")
    offset, total = 0, 500
    while offset < total:
        url = f"{base}&offset={offset}&length=100"
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    page = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 5:
                    time.sleep(25)
                else:
                    raise
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(3 * (attempt + 1))
        total = page["num_rows_total"]
        for rr in page["rows"]:
            rows.append({"row_idx": rr["row_idx"], **rr["row"]})
        offset += len(page["rows"])
        time.sleep(1.0)
    assert len(rows) == total == 500, f"math fetch incomplete: {len(rows)}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


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


def build_math25():
    rows = fetch_math500()
    hard = [r for r in rows if r["level"] in (4, 5)]
    by_subj = {}
    for r in hard:
        by_subj.setdefault(r["subject"], []).append(r)
    rng = random.Random(SEED)
    picked = []
    for s in sorted(by_subj):
        picked.extend(rng.sample(by_subj[s], min(3, len(by_subj[s]))))
    chosen = {r["row_idx"] for r in picked}
    remainder = [r for r in hard if r["row_idx"] not in chosen]
    picked.extend(rng.sample(remainder, 25 - len(picked)))
    picked.sort(key=lambda r: r["row_idx"])
    out = []
    for r in picked:
        boxed = extract_boxed(r["solution"])
        assert boxed is not None and boxed.strip(), \
            f"no \\boxed answer in {r['unique_id']}"
        num = os.path.splitext(os.path.basename(r["unique_id"]))[0]
        if boxed.strip() != (r.get("answer") or "").strip():
            print(f"  note: boxed != answer field for {r['unique_id']}: "
                  f"{boxed!r} vs {r.get('answer')!r} (using boxed)")
        out.append({"id": f"math_{r['subject'].lower().replace(' ', '_')}_{num}",
                    "subject": r["subject"], "level": r["level"],
                    "problem": r["problem"].strip(), "answer": boxed.strip()})
    write_jsonl(os.path.join(OUT, "math25.jsonl"), out)
    from collections import Counter
    print(f"  levels: {dict(Counter(o['level'] for o in out))}  "
          f"subjects: {len({o['subject'] for o in out})}")


# ---------------------------------------------------------------- Needle
SUBJECTS = ["the harbormaster", "a visiting surveyor", "the town council",
            "the orchard keeper", "the ferry crew", "a retired schoolteacher",
            "the granary clerk", "the innkeeper", "the road inspector",
            "a travelling bookbinder", "the mill foreman", "the postmistress"]
VERBS = ["reported", "noted", "recorded", "observed", "confirmed", "announced",
         "estimated", "remarked", "logged", "mentioned", "documented", "argued"]
OBJECTS = ["that the morning fog lifted later than usual",
           "that the grain shipment arrived two days behind schedule",
           "that repairs to the north bridge would continue through the week",
           "that the river gauge read slightly above its seasonal average",
           "that the market stalls sold out of root vegetables before noon",
           "that the coastal path remained closed after the recent storms",
           "that the apple harvest looked stronger than the previous year",
           "that the evening bell rang twice by mistake",
           "that the new streetlamps used less oil than expected",
           "that the schoolhouse roof needed fresh shingles before winter",
           "that the fishing boats returned early with a modest catch",
           "that the stagecoach schedule would change at the end of the month",
           "that the library received a crate of atlases from the capital",
           "that the well on the east square was measured at full depth",
           "that the woolen mill would hire extra hands for the season"]
TAILS = ["according to the weekly bulletin", "as noted in the ledger",
         "during the town assembly", "in a letter to the gazette",
         "despite earlier doubts", "to the surprise of very few",
         "after some deliberation", "as was customary that season",
         "before the matter was tabled", "which pleased the merchants"]

NEEDLES = [
    {"fact": "For the record, the brass telescope at the Verdello Observatory "
             "was donated by the cartographer Ilsa Brannigan in 1907.",
     "question": "According to the document, who donated the brass telescope "
                 "at the Verdello Observatory?",
     "expected": "Ilsa Brannigan"},
    {"fact": "It is worth noting that the rare fruit cultivated in the "
             "Quillside greenhouse is called the tamarink plum.",
     "question": "According to the document, what is the name of the rare "
                 "fruit cultivated in the Quillside greenhouse?",
     "expected": "tamarink plum"},
    {"fact": "As a point of record, the ferry across Lake Osmerel is "
             "suspended each spring for the Lanternwick Festival.",
     "question": "According to the document, for which festival is the ferry "
                 "across Lake Osmerel suspended each spring?",
     "expected": "Lanternwick"},
    {"fact": "For safekeeping it was written down that the passphrase to the "
             "Halvorsen seed vault is 'cobalt-heron-42'.",
     "question": "According to the document, what is the passphrase to the "
                 "Halvorsen seed vault?",
     "expected": "cobalt-heron-42"},
    {"fact": "The record further states that the Dunmarrow bell tower's "
             "escapement wheel was fitted by the clockmaker Petra Osgood.",
     "question": "According to the document, which clockmaker fitted the "
                 "escapement wheel of the Dunmarrow bell tower?",
     "expected": "Petra Osgood"},
    {"fact": "It was also recorded that the mineral quarried at Brindlecap "
             "Ridge is known locally as ferrovine.",
     "question": "According to the document, what is the local name of the "
                 "mineral quarried at Brindlecap Ridge?",
     "expected": "ferrovine"},
]


def gen_paragraph(rng):
    n = rng.randint(4, 7)
    sents = []
    for _ in range(n):
        s = (f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} "
             f"{rng.choice(OBJECTS)}, {rng.choice(TAILS)}.")
        sents.append(s[0].upper() + s[1:])
    return " ".join(sents)


def build_needle():
    # token target ~= words * 1.3  ->  words = tokens / 1.3
    combos = [(8000, 0.25), (8000, 0.75), (16000, 0.25), (16000, 0.75),
              (32000, 0.25), (32000, 0.75)]
    out = []
    for k, (tokens, pos) in enumerate(combos):
        rng = random.Random(SEED * 1000 + k)
        target_words = int(tokens / 1.3)
        paras, words = [], 0
        while words < target_words:
            p = gen_paragraph(rng)
            paras.append(p)
            words += len(p.split())
        needle = NEEDLES[k]
        insert_at = max(1, min(len(paras) - 1, int(len(paras) * pos)))
        paras.insert(insert_at, needle["fact"])
        ctx = "\n\n".join(paras)
        out.append({
            "id": f"needle_{tokens//1000}k_{int(pos*100)}pct",
            "context": ctx,
            "question": needle["question"] +
                        " Answer with the exact name or phrase from the document.",
            "expected_substring": needle["expected"],
        })
        # sanity: needle words never appear in filler
        body = ctx.replace(needle["fact"], "")
        assert needle["expected"].lower() not in body.lower()
    write_jsonl(os.path.join(OUT, "needle.jsonl"), out)
    for o in out:
        w = len(o["context"].split())
        print(f"  {o['id']}: {w} words (~{int(w*1.3)} tokens)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "gsm8k"):
        build_gsm8k()
    if only in ("all", "mmlu"):
        build_mmlu()
    if only in ("all", "humaneval"):
        build_humaneval()
    if only in ("all", "humaneval_plus"):
        build_humaneval_plus()
    if only in ("all", "math25"):
        build_math25()
    if only in ("all", "needle"):
        build_needle()
