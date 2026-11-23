#!/usr/bin/env python3
"""Paper 5: build every budget-matched prompt before any generation (amendment items 2-3).

Arms: T (message-level drop-oldest, fill to B), C (LLMLingua-2 over the full history,
library-exact rule at the largest attainable rate that fits; drop-oldest-chunk fallback),
Cw (LLMLingua-2 over the most recent ~4B tokens), M (dense retrieval over rounds,
MiniLM), Mbm25 (BM25 over rounds, B=4096 only), O (oracle evidence sessions, only if it
fits 4096), E (no history), S (session summaries; needs summaries.jsonl).

Every prompt is the final raw-completion string (chat template applied); n_prompt is its
exact served-tokenizer count and is <= B. Run in engines/lingua-env:
  engines/lingua-env/bin/python testsuite/paper5/build_prompts.py --arms T,C,Cw,M,Mbm25,O,E
  engines/lingua-env/bin/python testsuite/paper5/build_prompts.py --arms S
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5common as C  # noqa: E402

OUT = os.path.join(C.WORK, "prompts")
MINILM = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
                            "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
CW_WINDOW_FACTOR = 4
WORD = re.compile(r"\w+")
MODEL = "qwen38-exl3"      # set by --model; every count and wrapper uses this model's tokenizer
SUFFIX = ""                # set by --suffix; keeps cross-model prompt files separate


def count(text: str) -> int:
    return C.count_tokens(text, C.model_dir(MODEL))


# ------------------------------------------------------------------ helpers
def messages(item):
    """Chronological list of (session_index, turn_index) over the whole haystack."""
    out = []
    for si in C.chronological(item):
        for ti in range(len(item["haystack_sessions"][si])):
            out.append((si, ti))
    return out


def render_selection(item, selected):
    """selected: set of (si, ti). Sessions chronological, turns in original order."""
    blocks = []
    for si in C.chronological(item):
        turns = [t for ti, t in enumerate(item["haystack_sessions"][si]) if (si, ti) in selected]
        if turns:
            blocks.append((item["haystack_dates"][si], "".join(C.turn_text(t) for t in turns)))
    return C.render_sessions(blocks)


def final_prompt(item, history):
    return C.wrap(C.reader_prompt(history, item["question_date"], item["question"]),
                  thinking=True, model=MODEL)


def largest_fitting(n_max, count_for):
    """Largest k in [0, n_max] with count_for(k) <= budget-check (count_for returns bool fit).
    Assumes monotone: fits(k) implies fits(k-1)."""
    lo, hi = 0, n_max
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_for(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def evidence_fraction(item, selected):
    ev = C.evidence_turns(item)
    if not ev:
        return None
    return sum(1 for e in ev if e in selected) / len(ev)


def record(item, arm, budget, prompt, diag):
    n = count(prompt)
    if budget is not None and n > budget:
        raise SystemExit(f"budget violation {item['question_id']} {arm} {budget}: {n}")
    return {"job_id": f"{item['question_id']}|{arm}|{budget}", "question_id": item["question_id"],
            "question_type": item["question_type"], "arm": arm, "budget": budget,
            "prompt": prompt, "n_prompt": n, "prompt_sha256": C.sha256_text(prompt), "diag": diag}


# ------------------------------------------------------------------ T
def build_T(item, B):
    t0 = time.perf_counter()
    msgs = messages(item)

    def fits(k):
        sel = set(msgs[len(msgs) - k:]) if k else set()
        return count(final_prompt(item, render_selection(item, sel))) <= B
    k = largest_fitting(len(msgs), fits)
    sel = set(msgs[len(msgs) - k:]) if k else set()
    return record(item, "T", B, final_prompt(item, render_selection(item, sel)),
                  {"messages_kept": k, "messages_total": len(msgs),
                   "evidence_turns_kept": evidence_fraction(item, sel),
                   "policy_seconds": round(time.perf_counter() - t0, 4)})


# ------------------------------------------------------------------ rounds / retrieval
def rounds(item):
    """Rounds = a user turn plus the immediately following assistant turn (upstream round
    granularity); an assistant turn with no preceding user turn is its own round."""
    out = []
    for si in C.chronological(item):
        s = item["haystack_sessions"][si]
        ti = 0
        while ti < len(s):
            if s[ti]["role"] == "user" and ti + 1 < len(s) and s[ti + 1]["role"] == "assistant":
                out.append(((si, ti), (si, ti + 1)))
                ti += 2
            else:
                out.append(((si, ti),))
                ti += 1
    return out


def round_text(item, rd):
    return "".join(C.turn_text(item["haystack_sessions"][si][ti]) for si, ti in rd).strip()


class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(16)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MINILM)
        self.model = AutoModel.from_pretrained(MINILM).eval()
        self.cache = {}

    def embed(self, texts):
        need = [t for t in dict.fromkeys(texts) if C.sha256_text(t) not in self.cache]
        torch = self.torch
        for i in range(0, len(need), 128):
            batch = need[i:i + 128]
            enc = self.tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            for t, e in zip(batch, emb):
                self.cache[C.sha256_text(t)] = e
        return torch.stack([self.cache[C.sha256_text(t)] for t in texts])


def rank_dense(emb, item, rds):
    texts = [round_text(item, r) for r in rds]
    doc = emb.embed(texts)
    q = emb.embed([item["question"]])[0]
    scores = (doc @ q).tolist()
    # ties: later round first (recency)
    return sorted(range(len(rds)), key=lambda i: (-scores[i], -i)), scores


def rank_bm25(item, rds, k1=1.2, b=0.75):
    docs = [WORD.findall(round_text(item, r).lower()) for r in rds]
    N = len(docs)
    avg = sum(len(d) for d in docs) / max(1, N)
    df = Counter(w for d in docs for w in set(d))
    q = WORD.findall(item["question"].lower())
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1)
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return sorted(range(N), key=lambda i: (-scores[i], -i)), scores


def build_M(item, B, arm, order, rds, rank_seconds=0.0):
    t0 = time.perf_counter()

    def sel_for(k):
        s = set()
        for i in order[:k]:
            s.update(rds[i])
        return s

    def fits(k):
        return count(final_prompt(item, render_selection(item, sel_for(k)))) <= B
    k = largest_fitting(len(order), fits)
    sel = sel_for(k)
    return record(item, arm, B, final_prompt(item, render_selection(item, sel)),
                  {"rounds_kept": k, "rounds_total": len(rds),
                   "evidence_turns_kept": evidence_fraction(item, sel),
                   "policy_seconds": round(rank_seconds + time.perf_counter() - t0, 4)})


# ------------------------------------------------------------------ C / Cw
class Lingua:
    def __init__(self):
        from transformers import AutoTokenizer
        import lingua_probs as L
        self.L = L
        self.tok = AutoTokenizer.from_pretrained(L.MODEL_PATH)

    def chunks(self, qid):
        with gzip.open(os.path.join(C.WORK, "lingua", qid + ".json.gz"), "rt") as f:
            rec = json.load(f)
        return rec["chunks"], rec["history_sha256"]


def evidence_word_recall(item, text):
    ev = C.evidence_turns(item)
    if not ev:
        return None
    have = set(WORD.findall(text.lower()))
    words = set()
    for si, ti in ev:
        words.update(w for w in WORD.findall(item["haystack_sessions"][si][ti]["content"].lower()) if len(w) >= 4)
    return (sum(1 for w in words if w in have) / len(words)) if words else None


def build_C(item, B, arm, lingua, chunks):
    t0 = time.perf_counter()
    L = lingua.L
    comp = {}

    def text_at(p, drop=0):
        if p not in comp:
            comp[p] = [L.select_chunk(lingua.tok, ch, p) for ch in chunks]
        return "".join(comp[p][drop:])

    def fits_p(p):
        return count(final_prompt(item, text_at(p))) <= B
    # smallest percentile (most kept) that fits; library range 1..99
    lo, hi = 1, 99
    if not fits_p(99):
        p, fallback = 99, True
        n = len(chunks)
        drop = n - largest_fitting(n, lambda k: count(final_prompt(item, text_at(99, n - k))) <= B)
    else:
        fallback, drop = False, 0
        while lo < hi:
            mid = (lo + hi) // 2
            if fits_p(mid):
                hi = mid
            else:
                lo = mid + 1
        p = lo
    hist = text_at(p, drop)
    return record(item, arm, B, final_prompt(item, hist),
                  {"percentile": p, "fallback_chunks_dropped": drop, "fallback": fallback,
                   "chunks": len(chunks), "history_chars_kept": len(hist),
                   "evidence_word_recall": evidence_word_recall(item, hist),
                   "selection_seconds": round(time.perf_counter() - t0, 4)})


def window_chunks(chunks, B):
    target = CW_WINDOW_FACTOR * B
    tot = 0
    k = 0
    for ch in reversed(chunks):
        tot += count(ch["text"])
        k += 1
        if tot >= target:
            break
    return chunks[len(chunks) - k:], tot


# ------------------------------------------------------------------ O / E
def build_O(item_oracle):
    hist = C.full_history(item_oracle)
    prompt = final_prompt(item_oracle, hist)
    n = C.count_tokens(prompt)
    if n > 4096:
        return None
    return record(item_oracle, "O", 4096, prompt, {"sessions": len(item_oracle["haystack_sessions"])})


def build_E(item):
    return record(item, "E", None,
                  C.wrap(C.empty_prompt(item["question_date"], item["question"]), model=MODEL), {})


# ------------------------------------------------------------------ S
def load_summaries():
    path = os.path.join(C.WORK, "journals", "summaries.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("terminal") and r.get("ok"):
                out[r["job_id"]] = r["text"]
    return out


def build_S(item, B, summaries):
    import summarize as SM
    have = {}
    for idx in C.chronological(item):
        key = SM.summary_key(item["haystack_session_ids"][idx], SM.session_text(item["haystack_sessions"][idx]))
        if key in summaries:
            have[idx] = summaries[key]
    chosen = []
    for idx in reversed(C.chronological(item)):
        if idx not in have:
            raise SystemExit(f"missing summary for {item['question_id']} session {idx}")
        if count(SM.s_prompt(item, chosen + [idx], have, model=MODEL)) > B:
            break
        chosen.append(idx)
    prompt = SM.s_prompt(item, chosen, have, model=MODEL)
    ev = set(item["answer_session_ids"])
    ids = [item["haystack_session_ids"][i] for i in chosen]
    return record(item, "S", B, prompt,
                  {"sessions_kept": len(chosen), "sessions_total": len(item["haystack_sessions"]),
                   "evidence_sessions_summarized": (sum(1 for s in ids if s in ev) / len(ev)) if ev else None})


# ------------------------------------------------------------------ main
def write(arm, B, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{arm}_{B}{SUFFIX}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(rows)} prompts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="qwen38-exl3", choices=sorted(C.GEN_MODELS))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    global MODEL, SUFFIX
    MODEL, SUFFIX = a.model, a.suffix
    arms = a.arms.split(",")
    items = C.load_json_verified(C.LME_S)
    if a.limit:
        items = items[: a.limit]
    t0 = time.time()
    if "T" in arms:
        for B in C.BUDGETS:
            write("T", B, [build_T(it, B) for it in items])
    if "E" in arms:
        write("E", "none", [build_E(it) for it in items])
    if "O" in arms:
        oracle = {it["question_id"]: it for it in C.load_json_verified(C.LME_ORACLE)}
        rows = [r for r in (build_O(oracle[it["question_id"]]) for it in items) if r]
        write("O", 4096, rows)
    if "M" in arms or "Mbm25" in arms:
        emb = Embedder() if "M" in arms else None
        out = {("M", 1024): [], ("M", 4096): [], ("Mbm25", 4096): []}
        for n, it in enumerate(items):
            rds = rounds(it)
            if emb:
                t1 = time.perf_counter()
                order, _ = rank_dense(emb, it, rds)
                rs = time.perf_counter() - t1
                for B in C.BUDGETS:
                    out[("M", B)].append(build_M(it, B, "M", order, rds, rs))
            if "Mbm25" in arms:
                t1 = time.perf_counter()
                order, _ = rank_bm25(it, rds)
                out[("Mbm25", 4096)].append(build_M(it, 4096, "Mbm25", order, rds, time.perf_counter() - t1))
            if n % 50 == 0:
                print(f"M {n+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        for (arm, B), rows in out.items():
            if rows:
                write(arm, B, rows)
    if "C" in arms or "Cw" in arms:
        lingua = Lingua()
        out = {}
        for n, it in enumerate(items):
            chunks, hsha = lingua.chunks(it["question_id"])
            if hsha != C.sha256_text(C.full_history(it)):
                raise SystemExit(f"history hash mismatch for {it['question_id']}")
            for B in C.BUDGETS:
                if "C" in arms:
                    out.setdefault(("C", B), []).append(build_C(it, B, "C", lingua, chunks))
                if "Cw" in arms:
                    win, tot = window_chunks(chunks, B)
                    r = build_C(it, B, "Cw", lingua, win)
                    r["diag"]["window_tokens"] = tot
                    out.setdefault(("Cw", B), []).append(r)
            if n % 50 == 0:
                print(f"C {n+1}/{len(items)} {time.time()-t0:.0f}s", flush=True)
        for (arm, B), rows in out.items():
            write(arm, B, rows)
    if "S" in arms:
        summaries = load_summaries()
        for B in C.BUDGETS:
            write("S", B, [build_S(it, B, summaries) for it in items])
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
