#!/usr/bin/env python3
"""Paper 8 W2: which kernels a decode token runs on the 2080 Ti, and for how long (run under nsys).

Loads the model with exllamav3's own loader arguments (model_init, as smoke.py does), warms up on every prompt,
then generates once more per prompt with the CUDA profiler API switched on only from the first decoded token to
the last, so an nsys capture started with --capture-range=cudaProfilerApi holds decode and nothing else. Prefill,
loading and warm-up stay outside the capture. Writes one JSON row per prompt (tokens, decode seconds) to --out, so
census_report.py can divide kernel time by decoded tokens. Sampled: no sampler is passed, so exllamav3's DefaultSampler applies (min-p 0.08, temperature 0.8, unseeded; this said "Greedy" until 2026-10-31), thinking off, the smoke prompts.
--fill N puts about N tokens of a fixed synthetic document (numbered records, the same for every run) ahead of each
prompt, so decode runs at that context depth; the row records the prompt's token count.

  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o OUT python census.py -m MODEL ...
"""
import argparse
import json
import os
import random
import time

import exllamav3
import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.version import __version__ as exl3_version

from smoke import PROMPTS, chat, gpu_state


def decode(generator, tokenizer, prompt, max_new, stop, profile):
    ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
    generator.enqueue(Job(input_ids=ids, max_new_tokens=max_new, stop_conditions=stop, identifier=0))
    started, t0, last = False, None, None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if not started and r.get("text"):  # the first decoded token: prefill is over
                torch.cuda.synchronize()
                if profile:
                    torch.cuda.cudart().cudaProfilerStart()
                started, t0 = True, time.perf_counter()
            if r.get("eos"):
                last = r
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    row = {"new_tokens": last["new_tokens"], "decode_wall_s": round(wall, 4), "eos_reason": last.get("eos_reason"),
           "decode_tps_engine": round(last["new_tokens"] / max(last["time_generate"], 1e-9), 2)}
    row["prompt_tokens"] = int(ids.shape[-1])  # encode returns (1, n); rows before 2026-10-30 19:15 EDT recorded 1
    if "accepted_draft_tokens" in last:
        acc, rej = last["accepted_draft_tokens"], last["rejected_draft_tokens"]
        row.update(accepted=acc, rejected=rej, draft_accept=round(acc / max(acc + rej, 1), 3))
    return row


WORDS = ("amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
         "kelp", "lumen", "marble", "nectar", "onyx", "prairie", "quartz", "russet", "sierra", "tundra")


def filler(tokenizer, n):
    """About n tokens of a fixed synthetic document (seeded, so every run and arm sees the same text)."""
    if n <= 0:
        return ""
    rng = random.Random(1030)
    lines, ids = [], []
    while len(ids) < n:
        base = len(lines)
        lines.extend(f"Record {base + i:05d}: {rng.choice(WORDS)} {rng.choice(WORDS)} {rng.randint(0, 99999)} "
                     f"{rng.choice(WORDS)} batch {rng.randint(1, 64)}." for i in range(200))
        ids = tokenizer.encode("\n".join(lines), add_bos=False)[0].tolist()
    text = tokenizer.decode(torch.tensor(ids[:n]))
    return "Here is a log for reference; the question after it does not depend on it.\n\n" + text + "\n\n"


def sparse_state(generator):
    """The sparse draft head's state after a decode, or None when the generator drafts through the full head."""
    sh = getattr(generator, "sparse_head", None)
    if sh is None:
        return None
    keys = ("enabled", "mode_sparse", "capacity", "n_active", "rebuilds", "mode_switches", "lookups", "misses",
            "miss_rate", "observed", "evidence", "probes", "probe_disagree_inside", "probe_disagree_outside")
    return {k: (v.item() if hasattr(v, "item") else v) for k in keys if (v := getattr(sh, k, None)) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model_init.add_args(parser, default_cache_size=32768, add_draft_model_args=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True, help="JSONL of decode rows (appended)")
    parser.add_argument("--max_new", type=int, default=256)
    parser.add_argument("--tasks", default="code,math,prose")
    parser.add_argument("--fill", type=int, default=0, help="context depth: about N filler tokens before each prompt")
    args = parser.parse_args()

    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    # -sdh reaches the generator only when set, so builds without the option (rafatxf's 1.5.3) keep their signature
    sdh = getattr(args, "sparse_draft_head", 0) or 0
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                          draft_cache=draft_cache, num_draft_tokens=args.num_draft_tokens,
                          dynamic_draft_tokens=args.dynamic_draft, **({"sparse_draft_head": sdh} if sdh else {}))
    stop = [tokenizer.single_id("<|im_end|>")] + (config.eos_token_id_list or [])
    tasks = args.tasks.split(",")
    pre = filler(tokenizer, args.fill)
    for name in tasks:  # warm-up: kernels compiled and autotuned before anything is captured
        decode(generator, tokenizer, chat(pre + PROMPTS[name]), args.max_new, stop, profile=False)
    with open(args.out, "a") as out:
        for name in tasks:
            # A fresh prompt per task (no prompt-cache reuse changes the decode path); one capture range each.
            row = decode(generator, tokenizer, chat(pre + PROMPTS[name]), args.max_new, stop, profile=True)
            row.update(tag=args.tag, task=name, fill=args.fill, mtp=bool(args.mtp), num_draft_tokens=args.num_draft_tokens,
                       cache_quant=args.cache_quant, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       torch=torch.__version__, device=torch.cuda.get_device_name(0), gpu=gpu_state(),
                       # which engine build ran (an overlay build is 1.6.5+TAG) and every switch it read
                       exllamav3=exl3_version, exllamav3_path=os.path.dirname(exllamav3.__file__),
                       env={k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
                       # proof the sparse draft head was built and what state it was in (None: full head)
                       sparse_draft_head=sdh, sparse_head=sparse_state(generator))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: row.get(k) for k in ("task", "new_tokens", "decode_wall_s", "draft_accept")}))


if __name__ == "__main__":
    main()
