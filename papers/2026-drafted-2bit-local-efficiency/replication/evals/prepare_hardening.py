#!/usr/bin/env python3
"""Build the n>=500 hardening datasets for the efficiency campaign's lossless claims (H38, 2026-09-09).

  gsm8k_500.jsonl   superset of gsm8k_100 (same ids) + 400 more rows from the 1,319-row GSM8K test set
  mmlu_500.jsonl    superset of mmlu_200 + 2 more per subject (114) + 186 random remainder (cais/mmlu all/test)
  math500.jsonl     all 500 rows of MATH-500 (nlile/hendrycks-MATH-benchmark test split), math25 schema

Same row schemas as the small sets, so run_eval.py's gsm8k / mmlu / math25 prompt builders and scorers apply
unchanged (suites gsm8k_500 / mmlu_500 / math500). Sources come from prepare_datasets.py's cache (no download
when the cache is populated; otherwise the same public fetches). Stdlib only. Usage: prepare_hardening.py [all|gsm8k|mmlu|math]
"""
import json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_datasets as pdz  # noqa: E402

OUT, SEED, CACHE_DIR = pdz.OUT, pdz.SEED, pdz.CACHE_DIR


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_gsm8k_500():
    src = pdz.fetch("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
                    os.path.join(CACHE_DIR, "gsm8k_test.jsonl"))
    rows = read_jsonl(src)
    assert len(rows) == 1319, len(rows)
    base = sorted(random.Random(SEED).sample(range(len(rows)), 100))          # exactly prepare_datasets.build_gsm8k
    small = {r["id"] for r in read_jsonl(os.path.join(OUT, "gsm8k_100.jsonl"))}
    assert small == {f"gsm8k_{i}" for i in base}, "gsm8k_100 ids do not match the SEED sample; refusing to build a non-superset"
    rest = [i for i in range(len(rows)) if i not in set(base)]
    idxs = sorted(base + random.Random(SEED + 1).sample(rest, 400))
    out = []
    for i in idxs:
        m = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", rows[i]["answer"])
        assert m, f"no '#### N' marker in GSM8K row {i}"
        out.append({"id": f"gsm8k_{i}", "question": rows[i]["question"].strip(), "answer_number": float(m.group(1).replace(",", ""))})
    pdz.write_jsonl(os.path.join(OUT, "gsm8k_500.jsonl"), out)


def build_mmlu_500():
    rows = pdz.fetch_mmlu_all()
    small = read_jsonl(os.path.join(OUT, "mmlu_200.jsonl"))
    small_ids = {r["id"] for r in small}
    by_subj = {}
    for r in rows:
        if f"mmlu_{r['row_idx']}" not in small_ids:
            by_subj.setdefault(r["subject"], []).append(r)
    rng = random.Random(SEED + 1)
    picked = []
    for s in sorted(by_subj):                                                  # +2 per subject -> 114
        picked.extend(rng.sample(by_subj[s], min(2, len(by_subj[s]))))
    chosen = small_ids | {f"mmlu_{r['row_idx']}" for r in picked}
    remainder = [r for r in rows if f"mmlu_{r['row_idx']}" not in chosen]
    picked.extend(rng.sample(remainder, 500 - len(small) - len(picked)))
    out = list(small)
    for r in picked:
        assert len(r["choices"]) == 4 and r["answer"] in (0, 1, 2, 3)
        out.append({"id": f"mmlu_{r['row_idx']}", "subject": r["subject"], "question": r["question"].strip(),
                    "choices": r["choices"], "answer_idx": r["answer"]})
    out.sort(key=lambda o: int(o["id"].split("_")[1]))
    assert len(out) == 500 and len({o["id"] for o in out}) == 500
    pdz.write_jsonl(os.path.join(OUT, "mmlu_500.jsonl"), out)
    print(f"  subjects covered: {len({o['subject'] for o in out})}")


def build_math500():
    rows = pdz.fetch_math500()
    small = {r["id"]: r for r in read_jsonl(os.path.join(OUT, "math25.jsonl"))}
    out = []
    for r in sorted(rows, key=lambda r: r["row_idx"]):
        boxed = pdz.extract_boxed(r["solution"])
        assert boxed is not None and boxed.strip(), f"no \\boxed answer in {r['unique_id']}"
        num = os.path.splitext(os.path.basename(r["unique_id"]))[0]
        rid = f"math_{r['subject'].lower().replace(' ', '_')}_{num}"
        row = {"id": rid, "subject": r["subject"], "level": r["level"], "problem": r["problem"].strip(), "answer": boxed.strip()}
        if rid in small:
            assert small[rid]["answer"] == row["answer"] and small[rid]["problem"] == row["problem"], f"math25 row {rid} differs"
        out.append(row)
    assert len(out) == 500 and len({o["id"] for o in out}) == 500
    pdz.write_jsonl(os.path.join(OUT, "math500.jsonl"), out)
    from collections import Counter
    print(f"  levels: {dict(sorted(Counter(o['level'] for o in out).items()))}  math25 rows contained: {sum(1 for o in out if o['id'] in small)}/25")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "gsm8k"): build_gsm8k_500()
    if only in ("all", "mmlu"): build_mmlu_500()
    if only in ("all", "math"): build_math500()
