#!/usr/bin/env python3
"""Paper 8 W11 scouting: time exllamav3's Triton split-decode attention alone on the 2080 Ti, over launch settings.

One full-attention layer of qwen38-exl3-2.0 (24 q heads, 4 kv heads, head_dim 256) over a Q8 paged cache (8-bit K
and V, the lanes' -cq 8,8), with the block table 128 pages wide as in the 32K-cache runs. The cache holds random
packed words and small scales: the kernel's work does not depend on the values. For each context length and query
length (1 = undrafted, 7 = an MTP verify step) it times paged_attn_triton_decode over the kv tile (block_n), warps,
stages and split counts, with CUDA events, and checks each output against the default launch's (max abs diff).
The default row is what the engine runs: block_n from attn_decode_config (fitted to shared memory), 4 warps,
2 stages, splits = 2 x SMs / programs. Per-token cost is 16 layers x the call. The bandwidth floor counts the
packed words and scales the call has to read.
  python3 attn_bench.py --out ROWS.jsonl [--ctx 512,8192,28050] [--qlen 1,7] [--iters 200]
"""
import argparse
import itertools
import json
import time

import torch

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode

N_Q, N_KV, HD, PAGE, PAGES, LAYERS = 24, 4, 256, 256, 128, 16


def make_cache(device, gen):
    words = N_KV * HD // 32 * 8                       # 8-bit planes: 8 words per 32-value group
    qk = torch.randint(-2**31, 2**31 - 1, (PAGES, PAGE, words), dtype=torch.int32, device=device, generator=gen)
    qv = torch.randint(-2**31, 2**31 - 1, (PAGES, PAGE, words), dtype=torch.int32, device=device, generator=gen)
    sk = torch.full((PAGES, PAGE, N_KV * HD // 32), 0.05, dtype=torch.half, device=device)
    sv = torch.full((PAGES, PAGE, N_KV * HD // 32), 0.05, dtype=torch.half, device=device)
    return qk, qv, sk, sv


def run(q, cache, seqlens, table, qlen, **kw):
    qk, qv, sk, sv = cache
    return paged_attn_triton_decode(q=q, k=None, v=None, k_cache=qk, v_cache=qv, block_table=table,
                                    cache_seqlens=seqlens, causal=True, qc=(sk, sv, 8, 8),
                                    pre_appended_len=qlen, n_kv_heads_override=N_KV, **kw)


def time_call(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters      # microseconds per call


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", default="512,8192,28050")
    ap.add_argument("--qlen", default="1,7")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--block_n", default="16,32,64")
    ap.add_argument("--warps", default="2,4,8")
    ap.add_argument("--stages", default="1,2")
    ap.add_argument("--splits", default="0,68,136,272", help="0 = the engine's rule")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1030)
    cache = make_cache(dev, gen)
    table = torch.arange(PAGES, dtype=torch.int32, device=dev).unsqueeze(0)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ints = lambda s: [int(x) for x in s.split(",")]
    with open(a.out, "a") as out:
        for ctx, qlen in itertools.product(ints(a.ctx), ints(a.qlen)):
            q = (torch.randn((1, qlen, N_Q, HD), device=dev, generator=gen) * 0.5).half()
            seqlens = torch.tensor([ctx - qlen], dtype=torch.int32, device=dev)
            ref = run(q, cache, seqlens, table, qlen).clone()
            floor_bytes = ctx * (2 * N_KV * HD + 2 * N_KV * HD // 32 * 2)   # packed K and V bytes + fp16 scales
            base = {"ctx": ctx, "qlen": qlen, "sms": sms, "device": torch.cuda.get_device_name(dev),
                    "floor_us_at_550gbs": round(floor_bytes / 550e3, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            us = time_call(lambda: run(q, cache, seqlens, table, qlen), a.iters)
            out.write(json.dumps({**base, "config": "default", "us": round(us, 1), "per_token_ms": round(us * LAYERS / 1000, 3),
                                  "max_abs_diff": 0.0}) + "\n")
            out.flush()
            print(f"ctx {ctx} qlen {qlen} default: {us:.1f} us/call, {us * LAYERS / 1000:.2f} ms/token "
                  f"(floor {floor_bytes / 550e3:.0f} us)", flush=True)
            for bn, w, st, sp in itertools.product(ints(a.block_n), ints(a.warps), ints(a.stages), ints(a.splits)):
                kw = dict(block_n=bn, num_warps=w, num_stages=st, num_splits=sp or None)
                row = {**base, "config": {"block_n": bn, "warps": w, "stages": st, "splits": sp or "rule"}}
                try:
                    o = run(q, cache, seqlens, table, qlen, **kw)
                    torch.cuda.synchronize()
                    diff = (o.float() - ref.float()).abs().max().item()
                    us = time_call(lambda: run(q, cache, seqlens, table, qlen, **kw), a.iters)
                    row.update(us=round(us, 1), per_token_ms=round(us * LAYERS / 1000, 3), max_abs_diff=diff)
                except Exception as e:                      # a launch that does not fit (shared memory, registers)
                    torch.cuda.synchronize()
                    row.update(error=f"{type(e).__name__}: {str(e)[:200]}")
                out.write(json.dumps(row) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
