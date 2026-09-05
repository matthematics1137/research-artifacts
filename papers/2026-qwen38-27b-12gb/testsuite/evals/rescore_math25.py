#!/usr/bin/env python3
"""Lenient re-score of math25 results from stored audit fields (no re-runs).
Raw scorer was conservative by design; this credits value-correct answers that
differ only in formatting. Reports raw vs lenient side by side + artifact counts.
Also recovers rows whose expected_answer field was lost (kit bug) from the dataset."""
import json, re, glob, os, sys

DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "math25.jsonl")
TRUTH = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(DS)}

def lenient_norm(s):
    if s is None: return ""
    s = str(s).strip()
    s = re.sub(r"</?final[ _]answer>", "", s)                  # tag leakage
    s = s.replace("$", "").strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()                  # \( ... \)
    for u, l in [("√", r"\\sqrt"), ("π", r"\\pi"), ("∪", r"\\cup"), ("−", "-"),
                 ("≤", r"\\le"), ("≥", r"\\ge"), ("×", r"\\times")]:
        s = s.replace(u, l.replace("\\\\", "\\"))
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\displaystyle", "").strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\sqrt(\d|[a-zA-Z])", r"\\sqrt{\1}", s)       # \sqrt2 -> \sqrt{2}
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)             # \frac9{19}
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)  # frac -> slash
    s = re.sub(r"\^\{?\\circ\}?|°", "", s)                     # degrees
    s = re.sub(r"_\{?\d+\}?$", "", s)                          # base subscript 2516_8
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)                     # x=5 -> 5
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"[\s,]+", "", s)                               # spaces + thousand sep
    s = re.sub(r"^\((\d+(?:\.\d+)?)\)/\((\d+(?:\.\d+)?)\)$", r"\1/\2", s)
    return s

def num(s):
    try: return float(eval(s, {"__builtins__": {}}, {}))       # handles 1/4 etc.
    except Exception: return None

def eq(p, e):
    P, E = lenient_norm(p), lenient_norm(e)
    if P == E and P != "": return True
    np_, ne = num(P), num(E)
    return np_ is not None and ne is not None and abs(np_ - ne) < 1e-6

print(f"{'config':38s} raw  lenient  format-artifacts  truncated")
for f in sorted(glob.glob(os.path.join(os.path.dirname(DS), "..", "results", "*", "math25.jsonl"))):
    label = os.path.basename(os.path.dirname(f))
    rows = [json.loads(l) for l in open(f)]
    raw = sum(r["correct"] for r in rows)
    lenient = arti = trunc = 0
    for r in rows:
        exp = r.get("expected_answer")
        if exp in (None, "None"): exp = TRUTH.get(r["id"])     # recover kit-bug rows
        ok = r["correct"] or eq(r.get("predicted_answer"), exp)
        if not r["correct"] and r.get("truncated"): trunc += 1
        if ok and not r["correct"]: arti += 1
        lenient += ok
    print(f"{label:38s} {raw:2d}/25  {lenient:2d}/25      {arti:2d}              {trunc}")
