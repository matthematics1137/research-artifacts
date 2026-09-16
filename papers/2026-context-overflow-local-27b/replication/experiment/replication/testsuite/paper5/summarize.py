#!/usr/bin/env python3
"""Paper 5, arm S: summarize LongMemEval-S sessions once each, lazily and newest first,
until every item's S prompt at B=4,096 is full (amendment item 3).

Summaries are question-agnostic and cached by (session id, session-text hash).
The generator itself summarizes: thinking off, greedy, at most 200 tokens.

  engines/exl3-env/bin/python testsuite/paper5/summarize.py            # all items
  engines/exl3-env/bin/python testsuite/paper5/summarize.py --limit 3  # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5common as C  # noqa: E402
from p5run import Runner, TabbyServer, log  # noqa: E402

JOURNAL = os.path.join(C.WORK, "journals", "summaries.jsonl")
S_BUDGET_FILL = 4096

# Frozen summarization prompt (amendment item 3).
SUMMARY_PROMPT = (
    "Below is a conversation session between a user and an AI assistant, dated {date}.\n\n"
    "Write a summary of this session in at most 120 words. Keep every fact the user states "
    "about themselves (names, numbers, places, dates, events, preferences, plans, possessions) "
    "and every specific piece of information, recommendation, or answer the assistant gives. "
    "Keep all dates and time expressions exactly as written. Write plain text without a heading."
    "\n\nSession ({date}):{session}\n\nSummary:"
)
SUMMARY_PARAMS = {"max_tokens": 200, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                  "add_bos_token": False, "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]}


def session_text(session: list) -> str:
    return "".join(C.turn_text(t) for t in session)


def summary_key(sid: str, text: str) -> str:
    return f"sum|{sid}|{C.sha256_text(text)[:12]}"


def summary_body(summary: str) -> str:
    return "\n\nSession summary: " + summary.strip()


def s_prompt(item: dict, chosen: list[int], summaries: dict, model: str = "qwen38-exl3") -> str:
    """chosen: session indices (any order) whose summaries are shown, chronologically."""
    order = [i for i in C.chronological(item) if i in set(chosen)]
    hist = C.render_sessions([(item["haystack_dates"][i], summary_body(summaries[i])) for i in order])
    return C.wrap(C.reader_prompt(hist, item["question_date"], item["question"]),
                  thinking=True, model=model)


def summary_job(item: dict, idx: int) -> dict:
    sess = item["haystack_sessions"][idx]
    text = session_text(sess)
    date = item["haystack_dates"][idx]
    prompt = C.wrap(SUMMARY_PROMPT.format(date=date, session=text), thinking=False)
    return {"job_id": summary_key(item["haystack_session_ids"][idx], text), "prompt": prompt,
            "n_prompt": C.count_tokens(prompt), "budget": None, "params": SUMMARY_PARAMS,
            "timeout": 900, "meta": {"session_id": item["haystack_session_ids"][idx], "date": date}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = C.load_json_verified(C.LME_S)
    if a.limit:
        items = items[: a.limit]
    server = TabbyServer()
    runner = Runner(server, JOURNAL)
    log(f"summary resume: {len(runner.done)} cached terminal jobs; "
        f"minimum request idle {runner.request_idle_seconds:g}s; cached jobs are not regenerated")
    t0 = time.time()
    n_new = 0
    try:
        for n, item in enumerate(items):
            summaries = {}
            chosen = []
            for idx in reversed(C.chronological(item)):
                job = summary_job(item, idx)
                fresh = job["job_id"] not in runner.done
                row = runner.run(job)
                n_new += fresh
                if not row.get("ok"):
                    log(f"summary failed for {job['job_id']}; item {item['question_id']} stops here")
                    break
                summaries[idx] = row["text"]
                if C.count_tokens(s_prompt(item, chosen + [idx], summaries)) > S_BUDGET_FILL:
                    break
                chosen.append(idx)
            log(f"item {n+1}/{len(items)} {item['question_id']}: {len(chosen)} sessions fill S@4096; "
                f"{n_new} new summaries, {time.time()-t0:.0f}s")
    finally:
        server.stop()
    log("ALL DONE")


if __name__ == "__main__":
    main()
