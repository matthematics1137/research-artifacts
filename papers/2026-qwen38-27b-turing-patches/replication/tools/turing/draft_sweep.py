#!/usr/bin/env python3
"""Paper 8 W8: MTP draft settings on the 2080 Ti, one model load, decode speed per setting (no profiler).

Loads Qwen3.8-27B with its MTP layer once (model_init arguments, as smoke.py; pass -mtp), then for each setting in
the grid builds a fresh Generator on the same model and cache, warms it up on the three smoke prompts and measures
them again: decoded tokens, decode tok/s from the engine's own timer, draft acceptance, and a sha256 of each reply
(note, 2026-10-31: no sampler is passed, so exllamav3's DefaultSampler samples at temperature 0.8; output hashes differ between runs for that reason alone, so they cannot show whether a setting changes the output). Settings:
  nd        no drafting (same session, the ratio's denominator)
  base      the lane's setting: 6 draft tokens, dynamic, confidence 0.4, full draft head
  sdhN      base + sparse draft head of N vocabulary columns (verification keeps the full head: lossless by design)
  dcX       base with draft confidence X
  ndtN[s]   N draft tokens, dynamic (s: static)
`base` is repeated through the grid to track drift (heat, clocks). The sparse head's token profile goes to a fresh
directory per setting (no carry-over between settings or from earlier sessions). One JSON row per prompt per setting.
"""
import argparse
import gc
import hashlib
import json
import os
import time

import torch
from exllamav3 import Generator, Job, model_init
from exllamav3.modules import sparse_head
from exllamav3.version import __version__ as exl3_version

from smoke import PROMPTS, chat, gpu_state

GRID = ["nd", "base", "sdh8192", "sdh16384", "sdh32768", "base", "dc0.2", "dc0.3", "dc0.5", "dc0.6", "base",
        "ndt3", "ndt4", "ndt8", "ndt4s", "ndt3s", "base", "nd"]


def setting(name):
    s = {"num_draft_tokens": 6, "dynamic_draft_tokens": True, "draft_confidence": 0.4, "sparse_draft_head": 0,
         "draft": True}
    if name == "nd":
        s["draft"] = False
    elif name.startswith("sdh"):
        s["sparse_draft_head"] = int(name[3:])
    elif name.startswith("dc"):
        s["draft_confidence"] = float(name[2:])
    elif name.startswith("ndt"):
        static = name.endswith("s")
        s["num_draft_tokens"] = int(name[3:].rstrip("s"))
        s["dynamic_draft_tokens"] = not static
    elif name != "base":
        raise ValueError(name)
    return s


def generate(gen, tokenizer, prompt, max_new, stop):
    ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=max_new, stop_conditions=stop, identifier=0))
    text, last = "", None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            text += r.get("text", "")
            if r.get("eos"):
                last = r
    row = {"new_tokens": last["new_tokens"],
           "decode_tps": round(last["new_tokens"] / max(last["time_generate"], 1e-9), 2),
           "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
    if "accepted_draft_tokens" in last:
        acc, rej = last["accepted_draft_tokens"], last["rejected_draft_tokens"]
        row.update(accepted=acc, rejected=rej, draft_accept=round(acc / max(acc + rej, 1), 3))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model_init.add_args(ap, default_cache_size=16384, add_draft_model_args=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--grid", default=",".join(GRID))
    ap.add_argument("--profiles", required=True, help="directory for the per-setting sparse-head token profiles")
    args = ap.parse_args()
    assert args.mtp, "pass -mtp: the sweep needs the MTP layer loaded"
    # On a recurrent model the cache keeps max_history recurrent snapshots, sized at load from -ndt; every setting's
    # draft length must fit in it (2026-10-30: the first run stopped at ndt 6 against a history of 4)
    longest = max(setting(n)["num_draft_tokens"] for n in args.grid.split(",") if setting(n)["draft"])
    assert (args.num_draft_tokens or 0) >= longest, f"pass -ndt {longest} (or more) so the cache history covers it"

    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    stop = [tokenizer.single_id("<|im_end|>")] + (config.eos_token_id_list or [])
    with open(args.out, "a") as out:
        for i, name in enumerate(args.grid.split(",")):
            s = setting(name)
            sparse_head.TOKEN_STATS_DIR = os.path.join(args.profiles, f"{args.tag}-{i:02d}-{name}")
            os.makedirs(sparse_head.TOKEN_STATS_DIR, exist_ok=True)
            gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                            draft_model=draft_model if s["draft"] else None,
                            draft_cache=draft_cache if s["draft"] else None,
                            num_draft_tokens=s["num_draft_tokens"] if s["draft"] else None,
                            dynamic_draft_tokens=s["dynamic_draft_tokens"] and s["draft"],
                            draft_confidence=s["draft_confidence"],
                            sparse_draft_head=s["sparse_draft_head"] if s["draft"] else 0)
            for task in PROMPTS:  # warm-up: graphs, autotune, the confidence calibrator, the sparse-head counts
                generate(gen, tokenizer, chat(PROMPTS[task]), args.max_new, stop)
            for task in PROMPTS:
                row = generate(gen, tokenizer, chat(PROMPTS[task]), args.max_new, stop)
                row.update(tag=args.tag, seq=i, setting=name, task=task, **{k: v for k, v in s.items()},
                           time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), exllamav3=exl3_version,
                           env={k: v for k, v in os.environ.items() if k.startswith("EXL3_")}, gpu=gpu_state())
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps({k: row.get(k) for k in ("seq", "setting", "task", "decode_tps", "draft_accept",
                                                          "sha256")}))
            gen = None  # release this setting's generator (its sparse head, its job state) before the next one
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
