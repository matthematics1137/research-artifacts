#!/usr/bin/env python3
"""A2/B1 (H26): why does the 2-bit model reason longer? Temperature-0 traces with per-token logprobs on the lab's
math25 + HumanEval+ 20 items from any OpenAI-compatible server, then a paired comparison between two label files.
  collect: traces_compare.py collect --label L --port P [--suites math25,humaneval_plus] [--max-tokens 4096]
  compare: traces_compare.py compare <rows_A.jsonl> <rows_B.jsonl>
Per item: reasoning/content text, token count, logprob list (when the engine returns it), marker events
("Wait"-class transitions) with the logprob of the marker's first token and the mean logprob of the 20 tokens before
it, self-correction phrases, answer. Compare: first divergence position (tokens), marker counts, tokens per trace,
low-confidence share, and accuracy-free length ratios per item."""
import argparse, json, os, re, statistics, sys, urllib.request

KIT = "/path/to/lab-repo/testsuite/evals"
sys.path.insert(0, KIT)
MARK = re.compile(r"(\bWait\b|\bHmm+\b|\bAlternatively\b|\bHold on\b|[Ll]et me (?:double[- ]check|verify|re-?check|reconsider)|\bActually,)")
CORR = re.compile(r"(I made (?:a|an) (?:mistake|error)|that's (?:wrong|incorrect)|let me redo|let me recompute|I mis(?:read|calculated|counted))", re.I)


def load_items(suite):
    from run_eval import build_prompt, DATASET_FILES as SUITE_FILES
    rows = [json.loads(l) for l in open(os.path.join(KIT, "datasets", SUITE_FILES[suite])) if l.strip()]
    return [(r["id"], build_prompt(suite, r)) for r in rows]


def chat_logprobs(port, prompt, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1,
            "stream": False, "logprobs": True, "top_logprobs": 1, "chat_template_kwargs": {"reasoning_effort": "medium"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r: d = json.loads(r.read())
    ch = (d.get("choices") or [{}])[0]; msg = ch.get("message") or {}
    lp = ch.get("logprobs") or {}
    toks, lps = [], []
    content_entries = lp.get("content") or []
    for e in content_entries:
        toks.append(e.get("token")); lps.append(e.get("logprob"))
    return (msg.get("reasoning_content") or ""), (msg.get("content") or ""), toks, lps, ch.get("finish_reason"), (d.get("usage") or {}).get("completion_tokens")


def split_think(reasoning, content):
    """TabbyAPI returns the thinking inside content up to '</think>' (reasoning_content empty); llama.cpp separates it."""
    if "</think>" in content:
        pre, _, post = content.partition("</think>")
        return (reasoning + pre).strip(), post.strip()
    return reasoning.strip(), content.strip()


def analyze_trace(reasoning, content, toks, lps):
    reason, answer = split_think(reasoning, content)
    full_tok_text = "".join(t or "" for t in toks) if toks else ""
    markers = [m.start() for m in MARK.finditer(reason)]
    ev = []; low = None
    # map marker char offsets to token indices only when the token stream reproduces the analysed text
    stream_text = None
    if toks and lps and any(l is not None for l in lps):
        if full_tok_text.startswith(reason[:200]) and reason: stream_text = full_tok_text      # tabby: tokens cover content(=reasoning+answer)
        elif reasoning and full_tok_text.startswith(content[:200]): stream_text = None           # llama.cpp: tokens cover the answer only
        vals = [l for l in lps if l is not None]
        low = sum(1 for l in vals if l < -0.7) / max(1, len(vals))
    if stream_text is not None:
        pos, idx_at = 0, []
        for i, t in enumerate(toks):
            idx_at.append(pos); pos += len(t or "")
        for m in markers:
            ti = min(range(len(idx_at)), key=lambda i: abs(idx_at[i] - m))
            before = [l for l in lps[max(0, ti - 20):ti] if l is not None]
            ev.append({"char": m, "tok": ti, "lp_marker": lps[ti], "lp_before_mean": round(statistics.mean(before), 3) if before else None})
    else:
        ev = [{"char": m} for m in markers]
    return {"n_markers": len(markers), "markers": ev[:60], "n_corrections": len(CORR.findall(reason)), "low_conf_share": round(low, 4) if low is not None else None,
            "reason_chars": len(reason), "content_chars": len(answer), "reason_tokens_est": int(len(reason) / max(1, len(reason) + len(answer)) * len(toks)) if toks else None}


def collect(args):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"traces_{args.label}.jsonl")
    for suite in args.suites.split(","):
        for iid, prompt in load_items(suite):
            try:
                reasoning, content, toks, lps, fin, n = chat_logprobs(args.port, prompt, args.max_tokens)
            except Exception as e:
                print(json.dumps({"id": iid, "error": repr(e)}), flush=True); continue
            row = {"label": args.label, "suite": suite, "id": iid, "finish": fin, "completion_tokens": n or len(toks), "reasoning": reasoning, "content": content,
                   "tokens": toks, "logprobs": lps, **analyze_trace(reasoning, content, toks, lps)}
            with open(out, "a") as f: f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("id", "completion_tokens", "n_markers", "n_corrections", "low_conf_share", "finish")}), flush=True)
    print("COLLECT_DONE", out, flush=True)


def first_divergence(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


def compare(args):
    def load_rows(path):
        out = {}
        for l in open(path):
            if not l.strip(): continue
            r = json.loads(l); r.update(analyze_trace(r.get("reasoning", ""), r.get("content", ""), r.get("tokens") or [], r.get("logprobs") or [])); out[r["id"]] = r
        return out
    A, B = load_rows(args.a), load_rows(args.b)
    common = [i for i in A if i in B]
    print(f"A={os.path.basename(args.a)} B={os.path.basename(args.b)} paired={len(common)}")
    for suite in sorted({A[i]["suite"] for i in common}):
        ids = [i for i in common if A[i]["suite"] == suite]
        ta = [A[i]["completion_tokens"] for i in ids]; tb = [B[i]["completion_tokens"] for i in ids]
        ma = [A[i]["n_markers"] for i in ids]; mb = [B[i]["n_markers"] for i in ids]
        ca = [A[i]["n_corrections"] for i in ids]; cb = [B[i]["n_corrections"] for i in ids]
        div = [first_divergence(A[i]["tokens"], B[i]["tokens"]) for i in ids if A[i]["tokens"] and B[i]["tokens"]]
        la = [A[i]["low_conf_share"] for i in ids if A[i]["low_conf_share"] is not None]; lb = [B[i]["low_conf_share"] for i in ids if B[i]["low_conf_share"] is not None]
        ratio = [B[i]["completion_tokens"] / max(1, A[i]["completion_tokens"]) for i in ids]
        print(f"  {suite}: n={len(ids)} tokens A {statistics.mean(ta):.0f} B {statistics.mean(tb):.0f} (B/A median {statistics.median(ratio):.2f}); markers/trace A {statistics.mean(ma):.2f} B {statistics.mean(mb):.2f}; "
              f"corrections A {sum(ca)} B {sum(cb)}; first divergence median {statistics.median(div) if div else None} tokens; uncertain-token share (p<0.5) A {round(statistics.mean(la),3) if la else None} B {round(statistics.mean(lb),3) if lb else None}; reasoning tokens A {statistics.mean(A[i]['reason_tokens_est'] or 0 for i in ids):.0f} B {statistics.mean(B[i]['reason_tokens_est'] or 0 for i in ids):.0f}")
        mpa = [e["lp_marker"] for i in ids for e in A[i]["markers"] if e.get("lp_marker") is not None]; mpb = [e["lp_marker"] for i in ids for e in B[i]["markers"] if e.get("lp_marker") is not None]
        bpa = [e["lp_before_mean"] for i in ids for e in A[i]["markers"] if e.get("lp_before_mean") is not None]; bpb = [e["lp_before_mean"] for i in ids for e in B[i]["markers"] if e.get("lp_before_mean") is not None]
        if mpa and mpb:
            print(f"    marker-token logprob median A {statistics.median(mpa):.2f} B {statistics.median(mpb):.2f}; mean logprob of the 20 tokens before a marker A {statistics.median(bpa):.2f} B {statistics.median(bpb):.2f}")
        longest = sorted(ids, key=lambda i: -(B[i]["completion_tokens"] - A[i]["completion_tokens"]))[:3]
        print("    largest B-A token gaps:", [(i, A[i]["completion_tokens"], B[i]["completion_tokens"], A[i]["n_markers"], B[i]["n_markers"]) for i in longest])


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("collect"); c.add_argument("--label", required=True); c.add_argument("--port", type=int, default=18120); c.add_argument("--suites", default="math25,humaneval_plus"); c.add_argument("--max-tokens", type=int, default=4096)
    d = sub.add_parser("compare"); d.add_argument("a"); d.add_argument("b")
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else compare(args)


if __name__ == "__main__":
    main()
