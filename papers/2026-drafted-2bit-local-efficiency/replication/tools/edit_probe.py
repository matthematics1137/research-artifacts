#!/usr/bin/env python3
"""B2 (H29): cost of editing a prompt at position p on the recurrent model. A ~6K-token document prompt is sent cold,
then re-sent with one word changed at char positions mapping to ~p tokens for p in --positions; TTFT of each edited
prompt shows whether replay starts at the last recurrent checkpoint before the edit (step function at 2048-token
boundaries) or at the edit itself. usage: edit_probe.py --label L --port P [--tokens 6000] [--positions 300,1500,2300,3000,4300,5000]"""
import argparse, json, os, random, time, urllib.request

WORDS = ("policy retention schedule invoice ledger audit vendor contract renewal clause liability indemnity escrow "
         "milestone deliverable acceptance warranty termination notice jurisdiction arbitration confidentiality").split()


def ttft(port, messages, max_tokens=24):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            try: d = json.loads(line[5:])
            except Exception: continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")) and first is None:
                first = time.time() - t0; break
    return round(first, 3) if first is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True); ap.add_argument("--port", type=int, default=18120)
    ap.add_argument("--tokens", type=int, default=6000); ap.add_argument("--positions", default="300,1500,2300,3000,4300,5000")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "edit_rows.jsonl"))
    args = ap.parse_args()
    rnd = random.Random(5); words = [rnd.choice(WORDS) for _ in range(int(args.tokens / 1.3))]
    for rep in range(args.repeat):
        words[0] = f"rev{rep}"  # fresh document each repeat (cold)
        doc = " ".join(words)
        base = [{"role": "system", "content": "Reference document follows.\n\n" + doc}, {"role": "user", "content": "Which word appears most often? One word."}]
        cold = ttft(args.port, base); warm = ttft(args.port, base)
        rows = [{"rep": rep, "edit_tokens": None, "kind": "cold", "ttft_s": cold}, {"rep": rep, "edit_tokens": 0, "kind": "repeat", "ttft_s": warm}]
        print(json.dumps(rows[0])); print(json.dumps(rows[1]))
        for p in [int(x) for x in args.positions.split(",")]:
            w = list(words); wi = min(len(w) - 1, int(p / 1.3)); w[wi] = "EDITED"
            msgs = [{"role": "system", "content": "Reference document follows.\n\n" + " ".join(w)}, base[1]]
            t = ttft(args.port, msgs)
            rows.append({"rep": rep, "edit_tokens": p, "kind": "edited", "ttft_s": t}); print(json.dumps(rows[-1]), flush=True)
            ttft(args.port, base)  # restore the original prefix as the most recent stash
        with open(args.out, "a") as f:
            for r in rows: f.write(json.dumps({"label": args.label, **r}) + "\n")


if __name__ == "__main__":
    main()
