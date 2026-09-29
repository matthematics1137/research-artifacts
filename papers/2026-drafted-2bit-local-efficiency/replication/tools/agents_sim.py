#!/usr/bin/env python3
"""B4 (H27): three synthetic agents on one server. Each agent has its own ~3K-token system prompt and runs 6
append-only turns (question -> ~600-token answer appended -> next question). Measures per-turn TTFT and decode tok/s
per agent, aggregate tok/s, and GPU power (nvidia-smi sampled every 2 s). usage: agents_sim.py --label L --port P
[--agents 3] [--turns 6] [--max-tokens 700]"""
import argparse, json, os, random, subprocess, threading, time, urllib.request

WORDS = ("policy retention schedule invoice ledger audit vendor contract renewal clause liability indemnity escrow milestone "
         "deliverable acceptance warranty termination notice jurisdiction arbitration confidentiality quorum amendment").split()
QUESTIONS = ["Summarize the termination and renewal rules in about 400 words.", "Draft a 400-word memo on the liability and indemnity terms.",
             "Explain the escrow and milestone acceptance process in about 400 words.", "List the vendor's obligations with a short rationale each, about 400 words.",
             "Write a 400-word risk assessment of the arbitration clause.", "Propose three amendments with justification, about 400 words."]


def filler(n_tokens, seed):
    rnd = random.Random(seed); words = max(50, int(n_tokens / 1.3))
    return "\n".join("Section %d: " % k + " ".join(rnd.choice(WORDS) for _ in range(14)) for k in range(words // 14))


def stream_chat(port, messages, max_tokens):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "stream": True, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            try: d = json.loads(line[5:])
            except Exception: continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
            if piece:
                n += 1; text += piece
                if first is None: first = time.time() - t0
    wall = time.time() - t0
    try:  # drafted streaming delivers several tokens per chunk: count real tokens with the server's tokenizer
        ereq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/token/encode", data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(ereq, timeout=60) as er: n = json.loads(er.read()).get("length") or n
    except Exception:
        pass
    return first, wall, n, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True); ap.add_argument("--port", type=int, default=18120)
    ap.add_argument("--agents", type=int, default=3); ap.add_argument("--turns", type=int, default=6); ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents_rows.jsonl"))
    args = ap.parse_args()
    power = []; stop = False

    def sampler():
        while not stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                power.append((time.time(), float(out[0]), int(out[1])))
            except Exception: pass
            time.sleep(2)
    threading.Thread(target=sampler, daemon=True).start()
    rows = []; lock = threading.Lock(); t_start = time.time()

    def agent(k):
        msgs = [{"role": "system", "content": f"You are agent {k}, a contracts assistant. Reference text:\n\n" + filler(3000, 100 + k)}]
        for t in range(args.turns):
            msgs.append({"role": "user", "content": QUESTIONS[t % len(QUESTIONS)]})
            try:
                first, wall, n, text = stream_chat(args.port, msgs, args.max_tokens)
            except Exception as e:
                with lock: rows.append({"agent": k, "turn": t, "error": repr(e)}); print(json.dumps(rows[-1]), flush=True)
                break
            msgs.append({"role": "assistant", "content": text})
            row = {"label": args.label, "agent": k, "turn": t, "ttft_s": round(first, 2) if first else None, "wall_s": round(wall, 2), "tokens": n,
                   "tok_s": round(n / max(0.01, wall - (first or 0)), 1), "elapsed": round(time.time() - t_start, 1)}
            with lock: rows.append(row); print(json.dumps(row), flush=True)

    threads = [threading.Thread(target=agent, args=(k,)) for k in range(args.agents)]
    for th in threads: th.start(); time.sleep(3)
    for th in threads: th.join()
    stop = True; dur = time.time() - t_start
    ok = [r for r in rows if "ttft_s" in r]
    pw = [p for _, p, _ in power if p > 30]
    summ = {"label": args.label, "phase": "summary", "agents": args.agents, "turns": args.turns, "wall_s": round(dur, 1), "turns_done": len(ok),
            "ttft_first_turn_mean": round(sum(r["ttft_s"] for r in ok if r["turn"] == 0 and r["ttft_s"]) / max(1, sum(1 for r in ok if r["turn"] == 0 and r["ttft_s"])), 2),
            "ttft_later_turns_mean": round(sum(r["ttft_s"] for r in ok if r["turn"] > 0 and r["ttft_s"]) / max(1, sum(1 for r in ok if r["turn"] > 0 and r["ttft_s"])), 2),
            "per_agent_tok_s_mean": round(sum(r["tok_s"] for r in ok) / max(1, len(ok)), 1), "aggregate_tok_s": round(sum(r["tokens"] for r in ok) / dur, 1),
            "mean_w": round(sum(pw) / max(1, len(pw)), 1), "max_mem_mib": max((m for _, _, m in power), default=None)}
    print(json.dumps(summ), flush=True)
    with open(args.out, "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
        f.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
