#!/usr/bin/env python3
"""H38c: where and why do two temperature-0 runs of the same model diverge? Collects greedy traces with top-k
log-probabilities per token from an OpenAI-compatible server, then compares two trace files pairwise: first divergent
token, the reference run's top-1/top-2 gap at that position (a near-tie if small), whether the other run's token was
the reference's runner-up, and how far into the trace it happened.
  collect: divergence_probe.py collect --label L --port P [--suite gsm8k_500] [--limit 25] [--top 2] [--max-tokens 4096]
           -> <out-dir>/traces_<L>.jsonl   (out-dir: --out, default testsuite/results/efficiency/rows)
  compare: divergence_probe.py compare <reference.jsonl> <other.jsonl> [--tie-nat 0.5] [--json out.json]"""
import argparse, hashlib, json, os, statistics, sys, urllib.request

KIT = "/path/to/lab-repo/testsuite/evals"
ROWS = "/path/to/lab-repo/testsuite/results/efficiency/rows"
sys.path.insert(0, KIT)


def load_items(suite, limit):
    import run_eval  # noqa: E402  (prompt builders + dataset map)
    path = os.path.join(KIT, "datasets", run_eval.DATASET_FILES[suite])
    items = [json.loads(l) for l in open(path) if l.strip()]
    items = items[:limit] if limit else items
    return [(it["id"], run_eval.build_prompt(suite, it)) for it in items]


def chat(port, prompt, max_tokens, top):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0,
            "top_k": 1, "stream": False, "logprobs": True, "top_logprobs": top,
            "chat_template_kwargs": {"reasoning_effort": "medium"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    ch = d["choices"][0]; msg = ch["message"]
    lp = ch.get("logprobs") or {}
    content = lp.get("content") or []
    toks, chosen, alts = [], [], []
    for c in content:
        toks.append(c.get("token")); chosen.append(c.get("logprob"))
        tl = sorted((t for t in (c.get("top_logprobs") or []) if t.get("logprob") is not None), key=lambda t: -t["logprob"])
        alts.append([(t.get("token"), t.get("logprob")) for t in tl[:top]])
    return msg.get("reasoning_content") or "", msg.get("content") or "", toks, chosen, alts, ch.get("finish_reason")


def collect(a):
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"traces_{a.label}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    for iid, prompt in load_items(a.suite, a.limit):
        if iid in done: continue
        try:
            reasoning, content, toks, chosen, alts, fin = chat(a.port, prompt, a.max_tokens, a.top)
        except Exception as e:
            print(json.dumps({"id": iid, "error": repr(e)}), flush=True); continue
        full = reasoning + content
        row = {"id": iid, "label": a.label, "suite": a.suite, "finish": fin, "n_tokens": len(toks), "sha256": hashlib.sha256(full.encode()).hexdigest(),
               "reasoning": reasoning, "content": content, "tokens": toks, "logprob": chosen, "top": alts}
        with open(out, "a") as f: f.write(json.dumps(row) + "\n")
        print(json.dumps({"id": iid, "n_tokens": len(toks), "finish": fin, "sha8": row["sha256"][:8]}), flush=True)
    print("COLLECT_DONE", out)


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return None if len(a) == len(b) else n


def compare(a):
    for p in (a.ref, a.other):
        if not os.path.exists(p): print(f"missing trace file: {p}"); return
    A = {r["id"]: r for r in (json.loads(l) for l in open(a.ref) if l.strip()) if "tokens" in r}
    B = {r["id"]: r for r in (json.loads(l) for l in open(a.other) if l.strip()) if "tokens" in r}
    ids = [i for i in A if i in B]
    same = [i for i in ids if A[i]["sha256"] == B[i]["sha256"]]
    per = []
    for i in ids:
        if i in same: continue
        d = first_div(A[i]["tokens"], B[i]["tokens"])
        if d is None or d >= len(A[i]["tokens"]):
            per.append({"id": i, "div": d, "kind": "length-only"}); continue
        top = A[i]["top"][d] if d < len(A[i]["top"]) else []
        gap = (top[0][1] - top[1][1]) if len(top) >= 2 and top[0][1] is not None and top[1][1] is not None else None
        other_tok = B[i]["tokens"][d] if d < len(B[i]["tokens"]) else None
        runner_up = len(top) >= 2 and other_tok == top[1][0]
        per.append({"id": i, "div": d, "frac": round(d / max(1, len(A[i]["tokens"])), 3), "gap_nat": round(gap, 4) if gap is not None else None,
                    "ref_tok": A[i]["tokens"][d], "other_tok": other_tok, "other_is_runner_up": runner_up,
                    "ref_len": len(A[i]["tokens"]), "other_len": len(B[i]["tokens"])})
    gaps = [p["gap_nat"] for p in per if p.get("gap_nat") is not None]
    divs = [p["div"] for p in per if p.get("div") is not None]
    ru = [p["other_is_runner_up"] for p in per if "other_is_runner_up" in p]
    out = {"ref": a.ref, "other": a.other, "n_paired": len(ids), "identical": len(same), "identity_rate": round(len(same) / max(1, len(ids)), 3),
           "n_divergent": len(per), "median_first_div_tokens": statistics.median(divs) if divs else None,
           "near_tie_share": round(sum(1 for g in gaps if g < a.tie_nat) / len(gaps), 3) if gaps else None,
           "gap_median_nat": round(statistics.median(gaps), 4) if gaps else None, "gap_max_nat": round(max(gaps), 4) if gaps else None,
           "runner_up_share": round(sum(ru) / len(ru), 3) if ru else None,
           "same_final_answer": sum(1 for i in ids if A[i]["content"].strip()[-60:] == B[i]["content"].strip()[-60:]), "items": per}
    print(f"paired {len(ids)}  identical {len(same)} ({out['identity_rate']})  divergent {len(per)}")
    if per:
        print(f"  first divergence: median {out['median_first_div_tokens']} tokens; gap at divergence median {out['gap_median_nat']} nat, max {out['gap_max_nat']};"
              f" near-tie (<{a.tie_nat} nat) share {out['near_tie_share']}; other run's token = reference runner-up in {out['runner_up_share']}")
        for p in per[:12]: print("   ", {k: p[k] for k in p if k not in ("ref_len", "other_len")})
    if a.json:
        with open(a.json, "w") as f: json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--label", required=True); c.add_argument("--port", type=int, default=18120)
    c.add_argument("--suite", default="gsm8k_500"); c.add_argument("--limit", type=int, default=25); c.add_argument("--top", type=int, default=2)
    c.add_argument("--max-tokens", type=int, default=4096); c.add_argument("--out", default=ROWS)
    k = sub.add_parser("compare"); k.add_argument("ref"); k.add_argument("other"); k.add_argument("--tie-nat", type=float, default=0.5); k.add_argument("--json", default=None)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else compare(a)
