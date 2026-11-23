#!/usr/bin/env python3
"""Turing lab speed client: one fixed prompt set, streamed against any OpenAI-compatible serving lane.

Runs on p8-host with the system python3 (stdlib only) against a lane started by tools/lanes.sh:
    python3 $L/tools/speed_bench.py --lane exl3-mtp --base-url http://localhost:8090 [--repeats 3] [--limit 4]
    python3 $L/tools/speed_bench.py --list            # the prompt set, with sizes; no server needed

Every lane gets byte-identical requests: greedy (temperature 0, with the eval kit's top_p 0.95 / top_k 20, which
greedy ignores), max_tokens 512, the same chat_template_kwargs (default thinking off, as in tools/smoke.py), streaming
with usage. Per request it records:
  ttft_s              send -> first streamed text (content or reasoning_content)
  stream_decode_tps   tokens after the first text chunk / (last text chunk - first text chunk), from client-side
                      timestamps (the first chunk's token share is estimated by characters, since TabbyAPI sends
                      several tokens per chunk when drafts are accepted; on llama-server it is exactly n - 1)
  e2e_s, e2e_tps      send -> end of stream, and completion_tokens over that
  server_*            the server's own timings: llama-server's `timings` and upstream TabbyAPI's llama-compatible
                      `timings` (prompt/predicted per second, draft_n, draft_n_accepted) plus its usage block
  draft_accept        draft_n_accepted / draft_n when the server reports drafting
  gpu_during/after    nvidia-smi samples at 1 Hz while the request streams, and one snapshot after it
                      (power, clocks, temperature, memory, clock-event reasons); desktop only
  energy_j_est        mean sampled board power x e2e_s (whole request), and j_per_token_est = that / tokens
  output_sha256       sha256 of reasoning + content text (the eval kit's content_sha256 definition), so greedy
                      outputs can be compared across lanes
  stop_kind           why generation ended: "max_tokens" (completion_tokens reached max_tokens), "context" (the
                      server stopped with finish_reason "length" before max_tokens: its context ran out, so the row's
                      rates and energy are not comparable), "stop", "error", or the server's other finish_reason;
                      hit_max_tokens and hit_ctx_limit are the first two as booleans
Rows are appended to $TURING_LAB/runs/speed.jsonl (one JSON object per request).

Context check: a prompt is skipped, not sent, when its calibrated token count (PROMPT_TOKENS) plus max_tokens and a
small margin exceeds the server's context (llama-server /props n_ctx, TabbyAPI max_seq_len). A skipped prompt still
writes a row, with "skipped" set to the reason and no measurements; --no-ctx-check sends it anyway.
The run header and every row's "server" field carry lane_start: the newest $TURING_LAB/runs/lanes.jsonl row for the
server's port (lane, context, GDN recurrent-state precision, overrides), when the client runs on the desktop.

Prefix caching: by default every request starts with "Run <repeat>." so a repeat never reuses the previous
repeat's prompt cache (both engines cache prefixes); the tag depends only on the repeat index, so all lanes still
see identical prompts. --no-cache-bust sends the bare prompts.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "speed_bench/1"
LAB = os.environ.get("TURING_LAB", "/opt/p8")  # the desktop's ssd2 drive, never the main drive
DESKTOP_HOST = "p8-host"

# The three smoke prompts, verbatim from tools/turing/smoke.py, so these rows line up with runs/smoke.jsonl.
SMOKE = {
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "math": "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed? Show the steps.",
    "prose": "Describe, in two paragraphs, how a lighthouse keeper's day might have looked in 1890.",
}

REPORT = """\
Incident report, Kelda Valley water treatment works, night shift of 14 March.

At 01:12 the operator on duty, R. Anand, saw the outflow pressure on raw-water pump P2 fall from 4.1 bar to 2.6 bar
over roughly ninety seconds. The pump kept running and drew normal current, so the control system raised only a
low-priority warning. Anand checked the trend screen, saw that the reservoir level was still inside its band, and
logged the event for the day shift instead of switching to the standby pump, as the night procedure allows.

At 02:40 the pressure fell again, this time to 1.9 bar, and the downstream filter bank started to lose head. The
automatic sequence opened the bypass valve to protect the filters, which sent partly treated water into the contact
tank for about eleven minutes. Chlorine dosing followed the flow signal, so the residual at the tank outlet stayed
within limits, but turbidity peaked at 1.8 NTU against an internal target of 1.0 NTU and a regulatory limit of 4.0 NTU.

Anand started standby pump P3 at 02:47 and isolated P2. Pressure recovered within four minutes and the bypass closed
at 02:51. The duty engineer, M. Okafor, arrived at 03:30, opened the P2 inspection cover and found a plastic sheet
wrapped around the impeller. The sheet most likely came through a torn section of the intake screen, which the last
inspection, six weeks earlier, had marked as worn but serviceable.

No customer complaints were received and the distribution network samples taken at 06:00 met every standard. The
works lost about 3 percent of its night production. Okafor noted three contributing factors: the screen had been left
in service after it was marked as worn, the first pressure drop raised only a low-priority warning, and the night
procedure lets the operator defer a pump change when the reservoir level is within its band. Day shift replaced the
torn screen section by 11:00 and returned P2 to service at 14:20 after a test run. Okafor recommended raising the
priority of pump pressure warnings and reviewing the screen inspection interval before the spring rains.
"""

PROMPTS = {
    **SMOKE,
    "code2": ("Write a Python class TokenBucket that implements a token-bucket rate limiter with methods "
              "allow(cost=1) -> bool and refill(), using time.monotonic(). Include type hints, a docstring, "
              "and three short usage examples."),
    "math2": ("Prove that the sum of the first n odd numbers equals n squared, first by induction and then with a "
              "short geometric argument."),
    "prose2": ("Explain to a new software engineer how a hash map handles collisions, comparing separate chaining "
               "with open addressing, in about 300 words."),
    "summarize": REPORT + "\nSummarize the incident report above in five bullet points, then list two follow-up "
                          "actions the plant should take.",
}

# Long-context prompt: a synthetic shipment ledger from a fixed linear congruential generator (identical on every
# Python version and machine), then a question that needs the whole ledger. LONG_ROWS was calibrated with the
# Qwen3.8 tokenizer and chat template (tokenizer.json and chat_template.jinja of qwen38-exl3-2.0, 2026-10-30) so that
# the prompt plus the default 512 generated tokens fits the smallest lane context, 8,192 (the DFlash2 lanes):
# 175 rows = 7,400 tokens of user text, 7,417 with the template and the "Run 1." tag. 188 rows (7,957) left the
# DFlash2 lanes only 235 tokens of generation. Servers report the exact prompt_tokens in every row.
LONG_ROWS = 175
WAREHOUSES = ["Halden", "Brightwater", "Kestrel Point", "Marrow Bay", "Oakhurst", "Saltmarsh", "Tamsin Cross",
              "Vellore Yard"]
ITEMS = ["copper wire 2.5 mm", "pine planks 4 m", "steel bolts M10", "LED panels 60x60", "cement 25 kg",
         "glass wool rolls", "PVC pipe 110 mm", "ceramic tiles 30x30", "roof felt", "oak veneer sheets"]
CARRIERS = ["Nordline", "Bluewake", "Harrow and Sons", "QuickHaul", "Fennick Freight"]


def ledger_prompt(rows=LONG_ROWS, seed=20261030):
    state = seed

    def rnd(n):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 33) % n

    lines = ["Shipment ledger, first half of 2026 (id | date | warehouse | item | shipped | returned | carrier)"]
    for i in range(1, rows + 1):
        month, day = 1 + rnd(6), 1 + rnd(28)
        shipped = 20 + rnd(480)
        returned = rnd(shipped // 8 + 1)
        lines.append(f"{i:04d} | 2026-{month:02d}-{day:02d} | {WAREHOUSES[rnd(len(WAREHOUSES))]} | "
                     f"{ITEMS[rnd(len(ITEMS))]} | shipped {shipped} | returned {returned} | "
                     f"{CARRIERS[rnd(len(CARRIERS))]}")
    question = ("\n\nUsing only the ledger above: for the item 'steel bolts M10', compute each warehouse's net "
                "quantity (shipped minus returned, summed over its rows). List the per-warehouse totals, then say "
                "which warehouse has the largest net quantity.")
    return "\n".join(lines) + question


PROMPTS["long8k"] = ledger_prompt()

# Prompt tokens per prompt id, measured with the same tokenizer and template (2026-10-30, LONG_ROWS 175): thinking off
# (enable_thinking false) with the "Run 1." tag. EFFORT_EXTRA is what --thinking adds or removes (the template puts a
# reasoning-effort instruction in for low and xhigh). "Run 10." and later tags add 1 token; --no-cache-bust removes 5.
# Used only for the context check; rows record the server's exact prompt_tokens. Recalibrate if a prompt changes.
PROMPT_TOKENS = {"code": 36, "math": 55, "prose": 40, "code2": 63, "math2": 43, "prose2": 46, "summarize": 542,
                 "long8k": 7417}
PROMPT_TOKENS_ROWS = 175   # the LONG_ROWS that PROMPT_TOKENS["long8k"] was measured at
EFFORT_EXTRA = {"off": 0, "low": 28, "medium": -2, "xhigh": 40}
# Tag variation, small template differences between engines (the GGUF-embedded vs the model folder's template),
# llama-server's stop rule (prompt + generated + 1 >= n_ctx) and its draft clamp (n_ctx - n_tokens - 2).
CTX_MARGIN = 16

WARMUP_PROMPT = "Reply with one word: ready."
SMI_DURING = ["power.draw", "clocks.sm", "clocks.mem", "temperature.gpu", "utilization.gpu", "memory.used",
              "clocks_event_reasons.active"]
SMI_AFTER = SMI_DURING + ["memory.total", "pstate", "power.limit"]


# ---------------------------------------------------------------- GPU telemetry (desktop only)

def smi_query(fields):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    row = {}
    for key, val in zip(fields, [v.strip() for v in out[0].split(",")]):
        try:
            row[key] = float(val) if key != "clocks_event_reasons.active" else val
        except ValueError:
            row[key] = val
    return row


class GpuPoller(threading.Thread):
    """Samples nvidia-smi about once a second while one request streams."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval, self.samples, self.halt = interval, [], threading.Event()

    def run(self):
        while not self.halt.is_set():
            s = smi_query(SMI_DURING)
            if s:
                self.samples.append(s)
            self.halt.wait(self.interval)

    def summary(self):
        def nums(key):
            return [s[key] for s in self.samples if isinstance(s.get(key), float)]

        def mean(xs):
            return round(sum(xs) / len(xs), 1) if xs else None

        reasons = 0
        for s in self.samples:
            try:
                reasons |= int(str(s.get("clocks_event_reasons.active", "0")), 16)
            except ValueError:
                pass
        power, sm = nums("power.draw"), nums("clocks.sm")
        return {"n": len(self.samples), "power_w_mean": mean(power), "power_w_max": max(power, default=None),
                "sm_mhz_mean": mean(sm), "sm_mhz_min": min(sm, default=None), "mem_mhz_mean": mean(nums("clocks.mem")),
                "temp_c_max": max(nums("temperature.gpu"), default=None), "util_mean": mean(nums("utilization.gpu")),
                "mem_used_mib_max": max(nums("memory.used"), default=None), "event_reasons_or": hex(reasons)}


# ---------------------------------------------------------------- HTTP

def get_json(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def lane_start_record(base):
    """The newest $TURING_LAB/runs/lanes.jsonl row for this server's port, or None.

    TabbyAPI's /v1/model card does not report the GDN recurrent-state precision; lanes.sh reads it from the server
    log at start-up and stores it in that row, so this links each speed row to the start that served it. Only for
    a loopback base URL on the machine that holds the lab (the desktop)."""
    try:
        parts = urllib.parse.urlsplit(base)
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None
    if parts.hostname not in ("localhost", "localhost", "::1"):
        return None
    last = None
    try:
        with open(os.path.join(LAB, "runs/lanes.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("port") == port:
                    last = row
    except OSError:
        return None
    if last is None:
        return None
    return {k: last.get(k) for k in ("time", "lane", "status", "ctx", "recurrent_state", "lane_env", "lcpp_extra",
                                     "config", "command")}


def server_info(base):
    info = {}
    models = get_json(base + "/v1/models")
    if models and models.get("data"):
        info["model"] = models["data"][0].get("id")
    props = get_json(base + "/props")  # llama-server only
    if props:
        dgs = props.get("default_generation_settings") or {}
        info["build_info"] = props.get("build_info")
        info["n_ctx"] = dgs.get("n_ctx")
        info["min_p_default"] = (dgs.get("params") or {}).get("min_p")  # what a request without min_p gets
    card = get_json(base + "/v1/model")  # TabbyAPI only
    if card and card.get("id"):
        # TabbyAPI's /v1/models lists every folder in model_dir (alphabetical: qwen38-bf16 first); /v1/model is the
        # loaded one
        info["model"] = card["id"]
    if card and isinstance(card.get("parameters"), dict):
        p = card["parameters"]
        info["tabby"] = {k: p.get(k) for k in ("max_seq_len", "cache_size", "cache_mode", "chunk_size",
                                                "max_batch_size", "draft") if k in p}
        if info.get("n_ctx") is None and isinstance(p.get("max_seq_len"), int):
            info["n_ctx"] = p["max_seq_len"]
    info["lane_start"] = lane_start_record(base)
    return info


def ctx_check(pid, args, n_ctx):
    """None if prompt pid plus max_tokens fits n_ctx (or nothing is known), else the reason to skip it."""
    est = PROMPT_TOKENS.get(pid)
    if pid == "long8k" and LONG_ROWS != PROMPT_TOKENS_ROWS:
        est = None   # recalibrate PROMPT_TOKENS["long8k"] for the new LONG_ROWS
    if est is None or not isinstance(n_ctx, int) or n_ctx <= 0:
        return None
    need = est + EFFORT_EXTRA.get(args.thinking, 0) + args.max_tokens + CTX_MARGIN
    if need <= n_ctx:
        return None
    return (f"~{est + EFFORT_EXTRA.get(args.thinking, 0)} prompt tokens + max_tokens {args.max_tokens} + margin "
            f"{CTX_MARGIN} = {need} > server context {n_ctx}")


def build_body(text, args, max_tokens):
    body = {"model": "local", "messages": [{"role": "user", "content": text}], "temperature": 0.0,
            "top_p": 0.95, "top_k": 20, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if args.thinking == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["chat_template_kwargs"] = {"reasoning_effort": args.thinking}
    return body


def stream_once(base, body, timeout):
    """One streamed chat completion. Returns a dict of client timings, text, usage and timings."""
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    res = {"chunks": 0, "text_chunks": 0, "content": [], "reasoning": [], "usage": None, "timings": None,
           "finish_reason": None, "t_first": None, "first_chars": 0, "t_last": None, "max_gap_s": 0.0, "error": None}
    t0 = time.perf_counter()
    prev = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue  # blank separators and ": ping" keep-alives
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                res["chunks"] += 1
                if obj.get("error"):
                    res["error"] = str(obj["error"])[:300]
                if obj.get("usage"):
                    res["usage"] = obj["usage"]
                if obj.get("timings"):
                    res["timings"] = obj["timings"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    c, rc = delta.get("content") or "", delta.get("reasoning_content") or ""
                    if c or rc:
                        res["content"].append(c)
                        res["reasoning"].append(rc)
                        res["text_chunks"] += 1
                        if res["t_first"] is None:
                            res["t_first"], res["first_chars"] = now, len(c) + len(rc)
                        if prev is not None:
                            res["max_gap_s"] = max(res["max_gap_s"], now - prev)
                        prev = res["t_last"] = now
                    if ch.get("finish_reason"):
                        res["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        res["error"] = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        res["error"] = repr(e)[:300]
    res["t_done"] = time.perf_counter()
    res["t0"] = t0
    return res


# ---------------------------------------------------------------- one measured request

def measure(base, pid, text, rep, args, poll):
    body = build_body(text, args, args.max_tokens)
    poller = GpuPoller() if poll else None
    if poller:
        poller.start()
    res = stream_once(base, body, args.timeout)
    if poller:
        poller.halt.set()
        poller.join(timeout=15)
    usage, tm = res["usage"] or {}, res["timings"] or {}
    reasoning, content = "".join(res["reasoning"]), "".join(res["content"])
    if usage.get("completion_tokens") is not None:
        n, n_src = usage["completion_tokens"], "usage"
    elif tm.get("predicted_n") is not None:
        n, n_src = tm["predicted_n"], "timings"
    else:
        n, n_src = res["text_chunks"], "chunks"
    t0, t1, tl, td = res["t0"], res["t_first"], res["t_last"], res["t_done"]
    # Tokens that arrived after the first text chunk. TabbyAPI streams several tokens per chunk when drafts are
    # accepted, so the first chunk's share is estimated from its share of the characters (1 token per chunk on
    # llama-server, where this reduces to n - 1).
    total_chars = len(reasoning) + len(content)
    after_first = n - max(1.0, n * res["first_chars"] / total_chars) if total_chars else 0
    # finish_reason "length" covers both max_tokens and a full context (llama-server STOP_TYPE_LIMIT); only the token
    # count tells them apart. A chunk count can undercount TabbyAPI's multi-token chunks, so it never decides "context".
    fr = res["finish_reason"]
    if res["error"]:
        stop_kind = "error"
    elif fr == "length":
        stop_kind = "max_tokens" if n >= args.max_tokens else ("context" if n_src != "chunks" else "length")
    else:
        stop_kind = fr
    row = {
        "lane": args.lane, "prompt_id": pid, "repeat": rep, "run_id": args.run_id, "tag": args.tag,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bench": VERSION, "host": socket.gethostname(),
        "base_url": base, "thinking": args.thinking, "max_tokens": args.max_tokens, "cache_bust": args.cache_bust,
        "prompt_chars": len(text), "prompt_tokens": usage.get("prompt_tokens", tm.get("prompt_n")),
        "cached_tokens": tm.get("cache_n", (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
        "completion_tokens": n, "tokens_source": n_src, "finish_reason": fr, "stop_kind": stop_kind,
        "hit_max_tokens": stop_kind == "max_tokens", "hit_ctx_limit": stop_kind == "context",
        "chunks": res["chunks"], "text_chunks": res["text_chunks"],
        "tokens_per_chunk": round(n / res["text_chunks"], 2) if res["text_chunks"] else None,
        "ttft_s": round(t1 - t0, 4) if t1 else None,
        "decode_window_s": round(tl - t1, 4) if t1 and tl else None,
        "stream_decode_tps": round(after_first / (tl - t1), 2) if t1 and tl and tl > t1 and after_first > 0 else None,
        "e2e_s": round(td - t0, 4), "e2e_tps": round(n / (td - t0), 2) if n and td > t0 else None,
        "max_gap_s": round(res["max_gap_s"], 4),
        "server_prefill_tps": tm.get("prompt_per_second", usage.get("prompt_tokens_per_sec")),
        "server_decode_tps": tm.get("predicted_per_second", usage.get("completion_tokens_per_sec")),
        "server_prompt_ms": tm.get("prompt_ms"), "server_predicted_ms": tm.get("predicted_ms"),
        "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
        "draft_accept": (round(tm["draft_n_accepted"] / tm["draft_n"], 4)
                         if tm.get("draft_n") and tm.get("draft_n_accepted") is not None else None),
        "reasoning_chars": len(reasoning), "content_chars": len(content),
        "output_sha256": hashlib.sha256((reasoning + content).encode("utf-8")).hexdigest(),
        "output_head": (reasoning + content)[:200],
        "server_timings": res["timings"], "usage": res["usage"], "error": res["error"],
    }
    if poller:
        row["gpu_during"] = poller.summary()
        if row["gpu_during"]["power_w_mean"] and row["decode_window_s"]:
            joules = row["gpu_during"]["power_w_mean"] * row["e2e_s"]
            row["energy_j_est"] = round(joules, 1)
            row["j_per_token_est"] = round(joules / n, 3) if n else None
        row["gpu_after"] = smi_query(SMI_AFTER)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8090")
    ap.add_argument("--lane", help="lane name recorded in every row (required unless --list)")
    ap.add_argument("--repeats", type=int, default=1, help="passes over the prompt set (default 1)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (0 = all)")
    ap.add_argument("--prompts", default="", help="comma-separated prompt ids (default: all, in fixed order)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", default="off", choices=["off", "low", "medium", "xhigh"],
                    help="off sends enable_thinking=false (as tools/smoke.py); otherwise reasoning_effort, as the "
                         "eval kit sends it (default off)")
    ap.add_argument("--out", default=os.path.join(LAB, "runs/speed.jsonl"))
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request socket timeout, seconds")
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up request")
    ap.add_argument("--no-cache-bust", dest="cache_bust", action="store_false")
    ap.add_argument("--smi", default="auto", choices=["auto", "on", "off"],
                    help="nvidia-smi telemetry; auto = on only on the desktop, so a client elsewhere never "
                         "records another machine's GPU")
    ap.add_argument("--tag", default="", help="free-form note stored in every row (e.g. a lane override)")
    ap.add_argument("--no-ctx-check", dest="ctx_check", action="store_false",
                    help="send prompts even when prompt + max_tokens would not fit the server's context")
    ap.add_argument("--list", action="store_true", help="print the prompt set and exit")
    args = ap.parse_args()

    order = list(PROMPTS)
    if args.prompts:
        wanted = [p.strip() for p in args.prompts.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in PROMPTS]
        if unknown:
            ap.error(f"unknown prompt ids {unknown}; known: {order}")
        order = [p for p in order if p in wanted]
    if args.limit:
        order = order[: args.limit]
    if args.list:
        for pid in order:
            print(f"{pid:10s} {len(PROMPTS[pid]):6d} chars  ~{PROMPT_TOKENS.get(pid, '?')!s:>5} tokens  "
                  f"{PROMPTS[pid][:60]!r}")
        return 0
    if not args.lane:
        ap.error("--lane is required")
    base = args.base_url.rstrip("/")
    poll = args.smi == "on" or (args.smi == "auto" and socket.gethostname() == DESKTOP_HOST
                                and shutil.which("nvidia-smi") is not None)
    args.run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    info = server_info(base)
    print(json.dumps({"run_id": args.run_id, "lane": args.lane, "base_url": base, "server": info,
                      "prompts": order, "repeats": args.repeats, "thinking": args.thinking, "smi": poll}), flush=True)
    ls = info.get("lane_start")
    if ls and (ls.get("lane") != args.lane or ls.get("status") != "ready"):
        print(json.dumps({"warning": f"--lane {args.lane}, but the newest lanes.jsonl row for this port is lane "
                                     f"{ls.get('lane')} with status {ls.get('status')} ({ls.get('time')})"}),
              file=sys.stderr, flush=True)
    skip = {pid: ctx_check(pid, args, info.get("n_ctx")) for pid in order} if args.ctx_check else {}
    for pid, why in skip.items():
        if why:
            print(json.dumps({"prompt_id": pid, "skipped": why}), flush=True)
    if not args.no_warmup:
        w = stream_once(base, build_body(("Warm-up.\n\n" if args.cache_bust else "") + WARMUP_PROMPT, args, 16),
                        args.timeout)
        if w["error"]:
            print(json.dumps({"warmup_error": w["error"]}), flush=True)
            return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    failures = 0
    with open(args.out, "a") as out:
        for rep in range(args.repeats):
            for pid in order:
                text = (f"Run {rep + 1}.\n\n" if args.cache_bust else "") + PROMPTS[pid]
                if skip.get(pid):
                    # a row without measurements, so the gap in this lane's table explains itself
                    row = {"lane": args.lane, "prompt_id": pid, "repeat": rep, "run_id": args.run_id, "tag": args.tag,
                           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bench": VERSION,
                           "host": socket.gethostname(), "base_url": base, "thinking": args.thinking,
                           "max_tokens": args.max_tokens, "cache_bust": args.cache_bust, "prompt_chars": len(text),
                           "skipped": skip[pid], "error": None, "server": info}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    continue
                row = measure(base, pid, text, rep, args, poll)
                row["server"] = info
                out.write(json.dumps(row) + "\n")
                out.flush()
                failures += bool(row["error"]) or row["stop_kind"] == "context"
                print(json.dumps({k: row.get(k) for k in (
                    "prompt_id", "repeat", "prompt_tokens", "completion_tokens", "stop_kind", "ttft_s",
                    "stream_decode_tps", "server_decode_tps", "server_prefill_tps", "draft_accept", "error")}),
                      flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
