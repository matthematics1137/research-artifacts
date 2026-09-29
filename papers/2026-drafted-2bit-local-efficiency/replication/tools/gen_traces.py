#!/usr/bin/env python3
"""E32a: calibration traces from the model itself (no eval items). Two phases against a running OpenAI-compatible
server: (1) ask the model to write fresh problems in several domains (math word problems, algebra/number theory,
Python tasks, short-answer science/QA, logic puzzles); (2) solve each with thinking on (effort medium), keeping the
full reasoning + answer. Output: cal_traces.jsonl (prompt, reasoning, answer, tokens) and cal_traces.txt (the
rendered chat text of each trace, blank-line separated) for the converter. usage: gen_traces.py --port P --n 96"""
import argparse, json, os, random, re, time, urllib.request

DOMAINS = [
    ("math word problems like GSM8K but original (money, rates, geometry, percentages), each solvable with a single number answer", "math"),
    ("competition-style algebra, number theory and counting problems (AMC/AIME flavour, original), each with a short exact answer", "comp"),
    ("small Python programming tasks (a function with a clear spec and 2 example calls), no external libraries", "code"),
    ("short-answer science and history questions that need two or three reasoning steps, with one-line answers", "qa"),
    ("logic and deduction puzzles (schedules, seating, truth-tellers) with a single unambiguous answer", "logic"),
]


def chat(port, messages, max_tokens, effort, temperature=1.0, timeout=900):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95, "top_k": 20,
            "stream": False, "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.loads(r.read())
    msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
    return msg.get("reasoning_content") or "", msg.get("content") or ""


def count_tokens(port, text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/token/encode", data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read()).get("length")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18120); ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--out-dir", default=os.path.expanduser("~/models/_exl3_work_cot"))
    ap.add_argument("--hard", action="store_true", help="ask for hard multi-step problems (longer traces)")
    ap.add_argument("--append", action="store_true", help="append to existing cal_traces files instead of overwriting")
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    hard = " Make them genuinely hard: several dependent steps, no single-formula solutions, the kind that needs careful checking." if args.hard else ""
    per = max(1, args.n // len(DOMAINS)); problems = []
    rnd = random.Random(11 if args.hard else 7)
    for desc, tag in DOMAINS:
        got = []
        for attempt in range(4):
            if len(got) >= per: break
            _, content = chat(args.port, [{"role": "user", "content": f"Write {per + 2} {desc}.{hard} Number them 1-{per + 2}, one problem per numbered line, no solutions, no preamble. Seed word for variety: {rnd.choice(['harbor', 'orchard', 'ledger', 'comet', 'lattice', 'quarry', 'meadow', 'turbine', 'glacier', 'anvil'])}."}], 1800, "low", temperature=1.0)
            for line in content.splitlines():
                m = re.match(r"\s*\d+[.)]\s+(.*\S)", line)
                if m and len(m.group(1)) > 25 and m.group(1) not in got: got.append(m.group(1))
        problems += [(tag, p) for p in got[:per]]
        print(f"domain {tag}: {min(len(got), per)} problems", flush=True)
    rnd.shuffle(problems)
    mode = "a" if args.append else "w"
    jl = open(os.path.join(args.out_dir, "cal_traces.jsonl"), mode); txt = open(os.path.join(args.out_dir, "cal_traces.txt"), mode)
    tot = 0
    for k, (tag, prob) in enumerate(problems[:args.n]):
        suffix = {"math": "\n\nEnd your response with: Answer: <number>", "comp": "\n\nEnd your response with: Answer: <final answer>", "code": "\n\nReturn the complete function in one Python code block.", "qa": "\n\nEnd your response with: Answer: <short answer>", "logic": "\n\nEnd your response with: Answer: <final answer>"}[tag]
        t0 = time.time()
        try:
            reasoning, content = chat(args.port, [{"role": "user", "content": prob + suffix}], args.max_tokens, "medium")
        except Exception as e:
            print(f"[{k}] error {e!r}", flush=True); continue
        rendered = f"<|im_start|>user\n{prob + suffix}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}<|im_end|>\n"
        n = count_tokens(args.port, rendered); tot += n or 0
        jl.write(json.dumps({"i": k, "domain": tag, "prompt": prob, "reasoning": reasoning, "answer": content, "tokens": n}) + "\n"); jl.flush()
        txt.write(rendered + "\n"); txt.flush()
        print(json.dumps({"i": k, "domain": tag, "tokens": n, "wall_s": round(time.time() - t0, 1)}), flush=True)
    print(json.dumps({"phase": "done", "traces": min(len(problems), args.n), "total_tokens": tot}), flush=True)


if __name__ == "__main__":
    main()
