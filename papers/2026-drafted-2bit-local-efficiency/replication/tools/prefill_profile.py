#!/usr/bin/env python3
"""B8 (H34): where does EXL3 prefill time go? Bare exllamav3 API, EXL3 2.00, Q8 cache.
 0. environment check (which exllamav3, triton/fa2/fla availability)
 1. compute floor: fp16 matmul (fp32 acc) and int8 _int_mm at the model's GEMM shapes, M=1024; one fused
    reconstruct kernel timing scaled to the whole body
 2. one profiled ~4K-token prefill (torch.profiler), kernels bucketed: GEMM / reconstruct / GDN / attention / copies
 3. chunk-size A/B (512 / 1024 / 2048) on fresh 4K prompts
usage: prefill_profile.py [--out rows.jsonl]"""
import argparse, json, os, random, sys, time, torch
EXL = "/path/to/scratch/eff/exl3_147"
sys.path.insert(0, EXL)
import exllamav3
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.cache import CacheLayer_quant

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefill_profile.jsonl")); args = ap.parse_args()
OUT = open(args.out, "a")
def emit(**kw):
    kw["ts"] = time.strftime("%H:%M:%S"); print(json.dumps(kw), flush=True); OUT.write(json.dumps(kw) + "\n"); OUT.flush()

# 0. environment
env = {"exllamav3": getattr(exllamav3, "__version__", "?"), "file": exllamav3.__file__, "torch": torch.__version__}
try:
    from exllamav3.modules.attention_fn.triton_paged import has_triton; env["has_triton"] = bool(has_triton)
except Exception as e: env["has_triton"] = f"err {e!r}"[:60]
try:
    from exllamav3.modules.attention_fn.flash_attn_2 import has_fa2; env["has_fa2"] = bool(has_fa2)
except Exception as e: env["has_fa2"] = f"err {e!r}"[:60]
try:
    from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import chunk_gated_delta_rule; env["fla_chunk"] = chunk_gated_delta_rule is not None
except Exception as e: env["fla_chunk"] = f"err {e!r}"[:60]
env["sms"] = torch.cuda.get_device_properties(0).multi_processor_count
emit(step="env", **env)

# load the model first, on a clean GPU (Q8 cache, 8K, 1 slot); chunk 2048 if the split allows, else 1024
d = os.path.expanduser("~/models/qwen38-exl3-2.0")
cfg = Config.from_directory(d); model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens=8192, layer_type=CacheLayer_quant, k_bits=8, v_bits=8, max_batch_size=1, max_history=0)
LOADED_CHUNK = 2048
try:
    model.load(progressbar=False, max_chunk_size=2048, max_batch_size=1)
except RuntimeError as e:
    emit(step="load", note=f"chunk 2048 split failed ({str(e)[:60]}), retrying at 1024"); torch.cuda.empty_cache()
    model = Model.from_config(cfg); cache = Cache(model, max_num_tokens=8192, layer_type=CacheLayer_quant, k_bits=8, v_bits=8, max_batch_size=1, max_history=0)
    model.load(progressbar=False, max_chunk_size=1024, max_batch_size=1); LOADED_CHUNK = 1024
tok = Tokenizer.from_config(cfg)
emit(step="loaded", chunk=LOADED_CHUNK, reserved_mib=torch.cuda.memory_reserved() >> 20)

# 1. compute floor
def bench(fn, iters=20):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters
shapes = {"mlp_up": (5120, 17408), "mlp_down": (17408, 5120), "gdn_qkv": (5120, 10240), "gdn_z": (5120, 6144), "gdn_out": (6144, 5120), "attn_q": (5120, 12288), "attn_kv": (5120, 1024)}
M = 1024; tot16 = tot8 = 0.0
for name, (k, n) in shapes.items():
    a = torch.randn(M, k, device="cuda", dtype=torch.half); w = torch.randn(k, n, device="cuda", dtype=torch.half)
    t16 = bench(lambda: a @ w); fl = 2 * M * k * n
    a8 = torch.randint(-127, 127, (M, k), device="cuda", dtype=torch.int8); w8 = torch.randint(-127, 127, (k, n), device="cuda", dtype=torch.int8)
    try: t8 = bench(lambda: torch._int_mm(a8, w8))
    except Exception as e: t8 = None
    emit(step="gemm", shape=name, M=M, fp16_ms=round(t16 * 1e3, 3), fp16_tflops=round(fl / t16 / 1e12, 1), int8_ms=round(t8 * 1e3, 3) if t8 else None, int8_tops=round(fl / t8 / 1e12, 1) if t8 else None)
    del a, w, a8, w8
torch.cuda.empty_cache()


# reconstruct kernel timing on one MLP tensor
try:
    from exllamav3.ext import exllamav3_ext as ext
    l = model.find_module("model.language_model.layers.0.mlp.up_proj").inner
    w = torch.empty((l.in_features, l.out_features), dtype=torch.half, device="cuda")
    t = bench(lambda: ext.reconstruct_had_slice(w, l.trellis, l.suh, l.svh, l.K, l.mcg, l.mul1, 0), iters=10)
    emit(step="reconstruct", tensor="layers.0.mlp.up_proj", bytes=w.numel() * 2, ms=round(t * 1e3, 2), body_estimate_s=round(t * (48.6e9 / (w.numel() * 2)), 3))
    del w
except Exception as e:
    emit(step="reconstruct", error=repr(e)[:160])

WORDS = "policy retention schedule invoice ledger audit vendor contract renewal clause liability indemnity escrow milestone deliverable acceptance".split()
def prompt(n_tokens, seed):
    rnd = random.Random(seed); return f"seed{seed} " + " ".join(rnd.choice(WORDS) for _ in range(int(n_tokens / 1.3))) + "\n\nSummarize in one word:"

def prefill_time(gen, seed, n_tokens=4000):
    p = prompt(n_tokens, seed); ids = tok.encode(p); n = ids.shape[-1]
    torch.cuda.synchronize(); t0 = time.time()
    gen.generate(prompt=p, max_new_tokens=1, add_bos=False)
    torch.cuda.synchronize(); dt = time.time() - t0
    return n, dt

# 2. profiled 4K prefill at chunk 1024 (after a warm-up)
gen = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=1, max_chunk_size=1024)
n, dt = prefill_time(gen, 1, 1500); emit(step="warmup", tokens=n, s=round(dt, 2))
from torch.profiler import profile, ProfilerActivity
stats0 = torch.cuda.memory_stats()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    n, dt = prefill_time(gen, 2, 4000)
stats1 = torch.cuda.memory_stats()
buckets = {"gemm": 0.0, "reconstruct": 0.0, "gdn": 0.0, "attention": 0.0, "copies": 0.0, "norm_misc": 0.0, "other": 0.0}
top = []
for ev in prof.key_averages():
    name = ev.key.lower(); cuda = getattr(ev, "self_device_time_total", None)
    if cuda is None: cuda = getattr(ev, "self_cuda_time_total", 0.0)
    if not cuda: continue
    if any(s in name for s in ("gemm", "xmma", "cutlass", "hgemm", "sm89", "sm80")): b = "gemm"
    elif any(s in name for s in ("reconstruct", "had_r", "hadamard")): b = "reconstruct"
    elif any(s in name for s in ("chunk_", "l2norm", "solve_tril", "recompute_w_u", "conv1d", "gated_rms", "delta", "gdn", "cumsum")): b = "gdn"
    elif any(s in name for s in ("paged_attn", "dequant_cache", "attn", "flash", "sdpa", "softmax")): b = "attention"
    elif any(s in name for s in ("memcpy", "memset", "copy")): b = "copies"
    elif any(s in name for s in ("norm", "silu", "elementwise", "cat", "index", "embedding", "gather")): b = "norm_misc"
    else: b = "other"
    buckets[b] += cuda; top.append((cuda, ev.key[:70]))
tot = sum(buckets.values()) or 1.0
emit(step="profile", tokens=n, wall_s=round(dt, 2), tok_s=round(n / dt, 1), cuda_total_s=round(tot / 1e6, 2), gap_s=round(dt - tot / 1e6, 2),
     buckets={k: round(v / tot, 3) for k, v in buckets.items()}, alloc_retries=stats1.get("num_alloc_retries", 0) - stats0.get("num_alloc_retries", 0),
     device_allocs=stats1.get("num_device_alloc", 0) - stats0.get("num_device_alloc", 0), top=[(round(c / 1e3, 1), k) for c, k in sorted(top, reverse=True)[:12]])

# 3. chunk-size A/B on fresh prompts
for cs in [c for c in (512, 1024, 2048) if c <= LOADED_CHUNK]:
    g = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=1, max_chunk_size=cs)
    n1, d1 = prefill_time(g, 10 + cs, 4000); n2, d2 = prefill_time(g, 20 + cs, 4000)
    emit(step="chunk", chunk=cs, tokens=n1, tok_s=round(n1 / d1, 1), tok_s_rep2=round(n2 / d2, 1))
emit(step="done")
