#!/usr/bin/env python3
"""DEER-style dynamic early exit for reasoning (arXiv:2504.15895), as a proxy in front of an OpenAI-compatible
engine (llama-server or TabbyAPI). Client -> proxy /v1/chat/completions -> upstream /v1/completions (raw prompt,
so both engines reuse the prefix cache).

Loop: stream reasoning until a transition marker ("Wait", "Hmm", "Alternatively", "let me check", ...) or until
</think>. At a marker (first after --min-reason tokens, then at most every --probe-gap tokens) probe a trial answer:
append "\n</think>\n\n" + --inducer (e.g. "Answer:") and decode --probe-tokens greedy tokens with logprobs. If the
mean token probability is >= --threshold, accept: keep the trial and finish the answer from it. Otherwise discard the
trial and continue the reasoning from the marker. Every generated token (incl. discarded probes) counts against the
request's max_tokens. Returns a normal chat completion (content + reasoning_content + usage) plus a "deer" block.
usage: deer_proxy.py --upstream http://127.0.0.1:18120 --port 18130 --template <chat_template.jinja> [--threshold 0.95]
       [--probe-tokens 16] [--probe-gap 128] [--min-reason 128] [--inducer "Answer:"] [--markers default|wait]
       [--engine tabby|llama] [--log rows.jsonl]
--threshold 2.0 = probes run and are logged but never exit (overhead-only control).
--threshold -1  = no markers, no probes (pass-through through the same code path)."""
import argparse, json, math, re, sys, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2

ARGS = None
TEMPLATE = None
MARKER_SETS = {
    "default": r"(?:\bWait\b|\bHmm+\b|\bAlternatively\b|\bHold on\b|[Ll]et me (?:double[- ]check|verify|re-?check|reconsider|make sure)|\bActually,)",
    "wait": r"(?:\bWait\b)",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class Template:
    def __init__(self, path):
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        env.globals["strftime_now"] = lambda f: time.strftime(f)
        self.t = env.from_string(open(path).read())

    def render(self, messages, **kw):
        return self.t.render(messages=messages, add_generation_prompt=True, **kw)


def post(path, body, stream=False, timeout=900):
    req = urllib.request.Request(ARGS.upstream.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def count_tokens(text):
    if not text: return 0
    for path, key in (("/v1/token/encode", "length"), ("/tokenize", "tokens")):
        try:
            with post(path, {"text": text, "content": text, "add_bos_token": False, "add_special": False}, timeout=60) as r:
                d = json.loads(r.read()); v = d.get(key)
                return v if isinstance(v, int) else len(v)
        except Exception:
            continue
    return max(1, len(text) // 4)


def stream_segment(prompt, max_tokens, sp, marker_re, skip_chars=0):
    """Stream a completion of at most max_tokens; if marker_re is given, stop (close the socket) at the first match
    that starts after skip_chars characters of new text. Returns (text, hit_marker_or_None, finish_reason_or_None)."""
    body = dict(prompt=prompt, max_tokens=max_tokens, stream=True, **sp)
    text, hit, finish = "", None, None
    with post("/v1/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: d = json.loads(payload)
            except Exception: continue
            ch = (d.get("choices") or [{}])[0]
            piece = ch.get("text") or ""
            if piece:
                text += piece
                if marker_re is not None and len(text) > skip_chars:
                    m = marker_re.search(text, max(skip_chars, len(text) - len(piece) - 24))
                    if m:
                        text = text[:m.end()]; hit = m.group(0); break
            if ch.get("finish_reason"): finish = ch["finish_reason"]
    return text, hit, finish


def probe(prefix, probe_tokens):
    """Greedy trial answer with logprobs after an injected </think> (+ inducer). Returns (text, mean_p, min_p, done, n)."""
    body = {"prompt": prefix, "max_tokens": probe_tokens, "temperature": 0.0, "top_k": 1, "logprobs": 1, "stream": False}
    if ARGS.engine == "llama": body["n_probs"] = 1
    with post("/v1/completions", body) as r: d = json.loads(r.read())
    ch = (d.get("choices") or [{}])[0]; text = ch.get("text") or ""
    lp = ch.get("logprobs") or {}
    vals = []
    if isinstance(lp, dict):
        if lp.get("token_logprobs"): vals = [v for v in lp["token_logprobs"] if v is not None]
        elif lp.get("content"): vals = [e.get("logprob") for e in lp["content"] if e.get("logprob") is not None]
    if not vals and d.get("completion_probabilities"):
        vals = [math.log(max(1e-9, (e.get("probs") or [{}])[0].get("prob", 0))) for e in d["completion_probabilities"]]
    probs = [math.exp(v) for v in vals]
    n = len(probs) or count_tokens(text)
    return text, (sum(probs) / len(probs) if probs else 0.0), (min(probs) if probs else 0.0), ch.get("finish_reason") == "stop", n


def complete(prompt, max_tokens, sp):
    body = dict(prompt=prompt, max_tokens=max_tokens, stream=False, **sp)
    with post("/v1/completions", body) as r: d = json.loads(r.read())
    ch = (d.get("choices") or [{}])[0]; text = ch.get("text") or ""
    return text, ch.get("finish_reason") or "stop", count_tokens(text)


def handle_chat(req):
    t0 = time.time()
    msgs = req["messages"]; kw = req.get("chat_template_kwargs") or {}
    prompt = TEMPLATE.render(msgs, **kw)
    budget = int(req.get("max_tokens") or 4096)
    sp = {"temperature": req.get("temperature", 1.0), "top_p": req.get("top_p", 0.95), "top_k": req.get("top_k", 20)}
    if req.get("min_p") is not None: sp["min_p"] = req["min_p"]
    marker_re = re.compile(MARKER_SETS[ARGS.markers])
    reasoning, used, probes, exit_reason, content, finish = "", 0, [], None, "", "stop"
    last_probe_at = -10**9
    thinking = ARGS.threshold >= 0 and not prompt.rstrip().endswith("</think>")  # template ends with "<think>\n"
    force_point = int(budget * ARGS.force_frac) if ARGS.force_frac > 0 else None
    prev_trial = None
    while used < budget:
        next_allowed = max(ARGS.min_reason, last_probe_at + ARGS.probe_gap)
        skip_chars = 0 if used >= next_allowed else int((next_allowed - used) * 3.5)  # ~3.5 chars/token
        cap = budget - used
        if force_point and thinking and used < force_point: cap = max(1, min(cap, force_point - used))
        text, hit, fin = stream_segment(prompt + reasoning, cap, sp, marker_re if thinking else None, skip_chars)
        used += count_tokens(text) if text else 0
        reasoning += text
        if "</think>" in reasoning: thinking = False
        if force_point and thinking and used >= force_point:
            # forced exit: the budget would otherwise truncate the answer (scored wrong anyway)
            prefix = prompt + reasoning + "\n</think>\n\n" + ARGS.inducer
            ptext, pmean, pmin, pdone, pn = probe(prefix, ARGS.probe_tokens); used += pn
            probes.append({"at": used, "marker": "FORCED", "mean_p": round(pmean, 4), "min_p": round(pmin, 4), "done": pdone, "trial": ptext[:80]})
            exit_reason = "forced_exit"; content = ARGS.inducer + ptext
            if not pdone and used < budget:
                more, fr, mn = complete(prefix + ptext, budget - used, sp); used += mn; content += more; finish = fr
            break
        if hit and thinking:
            last_probe_at = used
            prefix = prompt + reasoning + "\n</think>\n\n" + ARGS.inducer
            ptext, pmean, pmin, pdone, pn = probe(prefix, ARGS.probe_tokens)
            used += pn
            consistent = prev_trial is not None and ptext.strip() == prev_trial.strip()
            accept = pmean >= ARGS.threshold if ARGS.rule == "mean" else (pmean >= ARGS.hi or (pmean >= ARGS.threshold and consistent))
            probes.append({"at": used, "marker": hit, "mean_p": round(pmean, 4), "min_p": round(pmin, 4), "done": pdone, "trial": ptext[:80], "consistent": consistent, "accept": accept})
            prev_trial = ptext
            if accept:
                exit_reason = "early_exit"; content = ARGS.inducer + ptext
                if not pdone and used < budget:
                    more, fr, mn = complete(prefix + ptext, budget - used, sp); used += mn; content += more; finish = fr
                break
            continue
        if fin == "stop" or not text:
            break
        # fin == "length": either our own cap (continue, probing now allowed) or the budget (loop condition ends it)
    if exit_reason is None:
        if "</think>" in reasoning:
            exit_reason = "natural"; reasoning, _, content = reasoning.partition("</think>"); content = content.strip()
        else:
            exit_reason = "budget" if used >= budget else "eos"
        if used >= budget: finish = "length"
    reasoning = reasoning.strip()
    wall = time.time() - t0
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "exit": exit_reason, "used": used, "n_probes": len(probes), "probes": probes,
           "wall_s": round(wall, 2), "finish": finish, "reason_chars": len(reasoning), "content_head": content[:60]}
    if ARGS.log:
        with open(ARGS.log, "a") as f: f.write(json.dumps(row) + "\n")
    log(f"exit={exit_reason} used={used} probes={len(probes)} wall={wall:.1f}s " + " ".join(f"{p['mean_p']:.2f}" for p in probes))
    return {"id": "deer-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(t0), "model": req.get("model", "x"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}, "finish_reason": finish}],
            "usage": {"prompt_tokens": count_tokens(prompt), "completion_tokens": used, "total_tokens": None},
            "deer": row, "timings": {"wall_s": round(wall, 2)}}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a): pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        try:
            with urllib.request.urlopen(ARGS.upstream.rstrip("/") + self.path, timeout=30) as r: self._send(200, json.loads(r.read() or b"{}"))
        except Exception as e:
            self._send(502, {"error": repr(e)})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path.rstrip("/") == "/v1/chat/completions":
                self._send(200, handle_chat(body))
            else:
                with post(self.path, body, timeout=600) as r: self._send(200, json.loads(r.read()))
        except Exception as e:
            log("ERROR", repr(e)); self._send(500, {"error": repr(e)})


def main():
    global ARGS, TEMPLATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:18120"); ap.add_argument("--port", type=int, default=18130)
    ap.add_argument("--template", required=True); ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--probe-tokens", type=int, default=16); ap.add_argument("--probe-gap", type=int, default=128)
    ap.add_argument("--min-reason", type=int, default=128); ap.add_argument("--inducer", default="")
    ap.add_argument("--markers", default="default", choices=list(MARKER_SETS))
    ap.add_argument("--log", default=None); ap.add_argument("--engine", default="tabby", choices=["tabby", "llama"])
    ap.add_argument("--force-frac", type=float, default=0.0, help="force an exit once this fraction of max_tokens is used (0 = off)")
    ap.add_argument("--rule", default="mean", choices=["mean", "consistent"], help="mean: exit at mean_p>=threshold; consistent: exit at mean_p>=hi, or mean_p>=threshold when the trial equals the previous probe's trial")
    ap.add_argument("--hi", type=float, default=0.99)
    ARGS = ap.parse_args(); TEMPLATE = Template(ARGS.template)
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    log(f"deer proxy on {ARGS.port} -> {ARGS.upstream} thr={ARGS.threshold} probe={ARGS.probe_tokens} gap={ARGS.probe_gap} "
        f"min={ARGS.min_reason} inducer={ARGS.inducer!r} markers={ARGS.markers} engine={ARGS.engine} force_frac={ARGS.force_frac} rule={ARGS.rule} hi={ARGS.hi}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
