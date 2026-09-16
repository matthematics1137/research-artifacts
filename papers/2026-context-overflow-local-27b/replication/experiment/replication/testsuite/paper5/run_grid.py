#!/usr/bin/env python3
"""Paper 5: send the frozen prompts to the generator (amendment items 1-2, 9).

  run_grid.py --pilot                       # 20 LoCoMo cat-3 items, T@4096 (operational)
  run_grid.py --arms T_4096,S_4096,...      # frozen prompt files from work/prompts/
  run_grid.py --locomo S                    # LoCoMo leg: <winner> vs T at 4096

Order is item-major with the arm order rotated per item, so slow drift (thermal
state, host load) touches every arm alike. Decode: temperature 0.2, top-p 0.95,
thinking on (reasoning_effort medium: no injected system message), cap from
work/cap.json (pilot rule), timeout 900 s. The per-request seed is recorded only
(TabbyAPI has no seed parameter; clarification 2026-09-19 14:05).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5common as C  # noqa: E402
from p5run import LlamaServer, Runner, TabbyServer, log, split_thinking  # noqa: E402

PROMPTS = os.path.join(C.WORK, "prompts")
JOURNALS = os.path.join(C.WORK, "journals")
CAP_FILE = os.path.join(C.WORK, "cap.json")
DEFAULT_CAP = 8192


def cap() -> int:
    if os.path.exists(CAP_FILE):
        with open(CAP_FILE) as f:
            return int(json.load(f)["completion_cap"])
    return DEFAULT_CAP


# Raw completions: no BOS (Qwen has none; the parity gate would catch one) and an explicit
# stop at the chat end-of-turn tokens (TabbyAPI stops on the tokenizer EOS only).
RAW = {"add_bos_token": False, "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]}


def params(max_tokens: int) -> dict:
    return {"max_tokens": max_tokens, "temperature": 0.2, "top_p": 0.95, **RAW}


def load_prompts(name: str) -> list[dict]:
    with open(os.path.join(PROMPTS, name + ".jsonl")) as f:
        return [json.loads(line) for line in f]


# Cross-model check (amendment item 6): Qwen3-8B UD-Q4_K_XL on the pinned llama.cpp build.
QWEN3_8B = os.path.expanduser("~/models/qwen3-8b/Qwen3-8B-UD-Q4_K_XL.gguf")


def make_server(model: str):
    if model == "qwen38-exl3":
        return TabbyServer()
    if model == "qwen3-8b":
        return LlamaServer("p5-qwen3-8b", 18127, QWEN3_8B,
                           ["-c", "16384", "--fit", "on", "-fa", "on", "-np", "1", "--jinja"])
    raise SystemExit(f"unknown model {model}")


def to_job(p: dict, max_tokens: int) -> dict:
    seed = C.request_seed(p["question_id"], p["arm"], p["budget"])
    return {"job_id": p["job_id"], "prompt": p["prompt"], "n_prompt": p["n_prompt"],
            "budget": p["budget"], "params": params(max_tokens), "timeout": 900,
            "verify_tokens": True,
            "meta": {"question_id": p["question_id"], "arm": p["arm"], "budget_label": p["budget"],
                     "seed_recorded_not_applied": seed, "prompt_file_sha256": p["prompt_sha256"]}}


def interleave(groups: list[list[dict]]) -> list[dict]:
    by_item = {}
    order = []
    for g in groups:
        for p in g:
            if p["question_id"] not in by_item:
                by_item[p["question_id"]] = []
                order.append(p["question_id"])
            by_item[p["question_id"]].append(p)
    out = []
    for i, qid in enumerate(order):
        ps = by_item[qid]
        r = i % len(ps)
        out.extend(ps[r:] + ps[:r])
    return out


def build_pilot() -> list[dict]:
    import build_prompts as BP
    items = C.locomo_pilot(C.locomo_items())
    rows = []
    for it in items:
        r = BP.build_T(it, 4096)
        r["job_id"] = "pilot|" + r["job_id"]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--journal", default="grid.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="qwen38-exl3")
    a = ap.parse_args()
    if a.pilot:
        prompts = build_pilot()
        journal = os.path.join(JOURNALS, "pilot.jsonl")
        max_tokens = 16384          # measure lengths without truncation
    else:
        prompts = interleave([load_prompts(n) for n in a.arms.split(",") if n])
        journal = os.path.join(JOURNALS, a.journal)
        max_tokens = cap()
    if a.limit:
        prompts = prompts[: a.limit]
    server = make_server(a.model)
    runner = Runner(server, journal)
    log(f"{len(prompts)} prompts -> {journal}; cap {max_tokens}; {len(runner.done)} already done")
    lengths = []
    try:
        for n, p in enumerate(prompts):
            row = runner.run(to_job(p, max_tokens))
            if row.get("ok"):
                comp = row.get("completion_tokens_harness")
                lengths.append(comp or 0)
                if n % 25 == 0 or a.pilot:
                    th, ans, closed = split_thinking(row["text"])
                    log(f"{n+1}/{len(prompts)} {p['job_id']} {row['seconds']:.1f}s completion {comp} "
                        f"closed={closed} answer={ans[:80]!r}")
    finally:
        server.stop()
    if a.pilot and lengths:
        mx = max(lengths)
        decision = {"pilot_max_completion": mx, "pilot_lengths": sorted(lengths),
                    "completion_cap": 16384 if mx >= 6144 else DEFAULT_CAP,
                    "rule": "amendment item 9: any pilot completion >= 6,144 tokens -> cap 16,384; else 8,192"}
        with open(CAP_FILE, "w") as f:
            json.dump(decision, f, indent=1)
        log(f"pilot cap decision: {decision['completion_cap']} (max completion {mx})")
    log("ALL DONE")


if __name__ == "__main__":
    main()
