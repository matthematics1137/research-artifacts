#!/usr/bin/env python3
"""Qwen3.8-27B EXL3 2.0 bpw on the 2080 Ti: does it load, how much VRAM, how fast. One config per run.

Uses exllamav3's own loader arguments (model_init), so every engine flag is available, plus:
  --label NAME      row label
  --long N          also prefill an N-token prompt (repeated public-domain text) and record peak VRAM
Appends one JSON row per measurement to <lab>/runs/smoke.jsonl. Sampled: no sampler is passed, so exllamav3's DefaultSampler applies (min-p 0.08, temperature 0.8, unseeded; this said "Greedy" until 2026-10-31), thinking off.
"""
import argparse
import json
import os
import subprocess
import time

import torch
from exllamav3 import Generator, Job, model_init

PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "math": "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed? Show the steps.",
    "prose": "Describe, in two paragraphs, how a lighthouse keeper's day might have looked in 1890.",
}
FILLER = ("It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
          "foolishness, it was the epoch of belief, it was the epoch of incredulity. ")
LAB = os.environ.get("TURING_LAB", "/opt/p8")  # the desktop's ssd2 drive, never the main drive
OUT = os.path.join(LAB, "runs/smoke.jsonl")


def gpu_state():
    q = "memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    keys = ["mem_used_mib", "mem_total_mib", "temp_c", "power_w", "sm_mhz", "throttle"]
    return dict(zip(keys, out))


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def run(generator, tokenizer, prompt, max_new, stop):
    ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=max_new, stop_conditions=stop, identifier=0)
    generator.enqueue(job)
    text, last = "", None
    torch.cuda.reset_peak_memory_stats()
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            text += r.get("text", "")
            if r.get("eos"):
                last = r
    fresh = last["prompt_tokens"] - last.get("cached_tokens", 0)  # the repeat run reuses the prompt cache
    row = {"prompt_tokens": last["prompt_tokens"], "cached_tokens": last.get("cached_tokens", 0),
           "new_tokens": last["new_tokens"],
           "prefill_tps": round(fresh / max(last["time_prefill"], 1e-9), 1) if fresh > 0 else None,
           "decode_tps": round(last["new_tokens"] / max(last["time_generate"], 1e-9), 2),
           "eos_reason": last.get("eos_reason"), "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
    if "accepted_draft_tokens" in last:
        acc, rej = last["accepted_draft_tokens"], last["rejected_draft_tokens"]
        row["draft_accept"] = round(acc / max(acc + rej, 1), 3)
    return row, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_init.add_args(parser, default_cache_size=32768, add_draft_model_args=True)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--long", type=int, default=0)
    parser.add_argument("--max_new", type=int, default=384)
    args = parser.parse_args()

    t0 = time.time()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    load_s = time.time() - t0
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                          draft_cache=draft_cache, num_draft_tokens=args.num_draft_tokens,
                          dynamic_draft_tokens=args.dynamic_draft)
    stop = [tokenizer.single_id("<|im_end|>")] + (config.eos_token_id_list or [])
    base = {"label": args.label, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load_s": round(load_s, 1),
            "cache_size": args.cache_size, "cache_quant": args.cache_quant, "mtp": bool(args.mtp),
            "num_draft_tokens": args.num_draft_tokens, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0), "after_load": gpu_state()}
    with open(OUT, "a") as out:
        for name, text in PROMPTS.items():
            for attempt in ("warm", "measured"):
                row, reply = run(generator, tokenizer, chat(text), args.max_new, stop)
                if attempt == "measured":
                    row.update(base, task=name, gpu=gpu_state(), reply_head=reply[:160])
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(json.dumps({k: row[k] for k in ("task", "prefill_tps", "decode_tps", "new_tokens")} |
                                     {"draft_accept": row.get("draft_accept"), "vram": row["gpu"]["mem_used_mib"]}))
        if args.long:
            words = FILLER * (args.long // 30 + 1)
            ids = tokenizer.encode(words, add_bos=False)[:, : args.long - 64]
            prompt = chat(tokenizer.decode(ids[0]) + "\n\nIn one sentence: what is this text?")
            row, reply = run(generator, tokenizer, prompt, 64, stop)
            row.update(base, task=f"long{args.long}", gpu=gpu_state(), reply_head=reply[:160])
            out.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("task", "prompt_tokens", "prefill_tps", "decode_tps")} |
                             {"vram": row["gpu"]["mem_used_mib"], "torch_peak_mib": row["torch_peak_mib"]}))


if __name__ == "__main__":
    main()
