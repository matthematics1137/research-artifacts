#!/usr/bin/env python3
"""Paper 8 W11: correctness and speed of the fd75d decode split kernel against exllamav3's Triton split kernel.

Builds tools/turing/fd75d/fd75d.cu alone as a torch extension (-DFD75D_STANDALONE). One layer of the model's
attention geometry (24 q heads, 4 kv heads, head_dim 256) over an 8-bit packed paged cache with random codes and
random per-group scales, the block table a random page permutation 128 pages wide. For each case it compares:
- out_tri: paged_attn_triton_decode (the engine's split + combine), qc = (sk, sv, 8, 8);
- out_fd: fd75d phase 1, then the same Triton combine kernel on its partials;
- out_ref: an fp64 reference (ref_attn; fp32 in this line until 2026-10-31) that dequantizes the cache and does exact softmax attention (rotated domain, then
  the 32-wide Hadamard back).
Then both kernels' GPU time inside CUDA graphs (split + combine each), and the per-token figure for 16 layers.
  python3 fd75d_test.py --out ROWS.jsonl [--quick]
"""
import argparse
import json
import math
import os
import time

import torch
import triton
from torch.utils.cpp_extension import load

from exllamav3.modules.attention_fn.triton_paged import (
    paged_attn_triton_decode, _paged_attn_decode_combine_kernel, combine_subtiles, decode_row_layout, _get_h32)

N_Q, N_KV, HD, PAGE, PAGES = 24, 4, 256, 256, 128
HERE = os.path.dirname(os.path.abspath(__file__))


def build():
    return load(name="fd75d_test", sources=[os.path.join(HERE, "fd75d", "fd75d.cu")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-DFD75D_STANDALONE", "-gencode=arch=compute_75,code=sm_75"],
                verbose=False)


def hadamard32(device):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 32:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(32)).to(device)


def ref_attn(q, qk, qv, sk, sv, table, seqlens, q_len, scale):
    """fp64 reference: exact softmax attention over the dequantized cache, in the rotated domain"""
    bsz = q.shape[0]
    H = hadamard32(q.device)
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for b in range(bsz):
        L = int(seqlens[b]) + q_len
        keys = torch.arange(L, device=q.device)
        rows = table[b, keys // PAGE].long() * PAGE + keys % PAGE
        kb = qk.view(-1, qk.shape[-1])[rows].view(torch.uint8).double()      # (L, n_kv * 256) codes
        vb = qv.view(-1, qv.shape[-1])[rows].view(torch.uint8).double()
        ks = sk.view(-1, sk.shape[-1])[rows].double().repeat_interleave(32, dim=1)
        vs = sv.view(-1, sv.shape[-1])[rows].double().repeat_interleave(32, dim=1)
        K = ((kb - 127.5) * ks / 128).view(L, N_KV, HD)
        V = ((vb - 127.5) * vs / 128).view(L, N_KV, HD)
        qr = (q[b].double().view(q_len, N_Q, HD // 32, 32) @ H).view(q_len, N_Q, HD)
        for hq in range(N_Q):
            hk = hq // (N_Q // N_KV)
            s = (qr[:, hq] @ K[:, hk].T) * scale                              # (q_len, L)
            qabs = L - q_len + torch.arange(q_len, device=q.device)
            s = s.masked_fill(keys[None, :] > qabs[:, None], -float("inf"))
            o = torch.softmax(s, dim=-1) @ V[:, hk]
            out[b, :, hq] = (o.view(q_len, HD // 32, 32) @ H).view(q_len, HD)
    return out


def run_fd(ext, q, cache, table, seqlens, q_len, splits, scale, live=False):
    qk, qv, sk, sv = cache
    bsz = q.shape[0]
    block_rows, h_blocks = decode_row_layout(q_len, N_Q // N_KV, HD)
    programs = bsz * N_KV * h_blocks
    max_k = table.shape[1] * PAGE + q_len
    split_len = triton.cdiv(triton.cdiv(max_k, splits), 16) * 16
    po = torch.empty(programs * splits * block_rows * HD, dtype=torch.float32, device=q.device)
    pml = torch.empty(programs * splits * block_rows * 2, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    h32 = _get_h32(q.device)
    rows_sub, d_sub = combine_subtiles(block_rows, HD)

    def go():
        ext.split(q, qk, qv, sk, sv, table, seqlens, po, pml, split_len, splits, splits, q_len, scale, True, N_KV, live)
        _paged_attn_decode_combine_kernel[(programs, (block_rows // rows_sub) * (HD // d_sub))](
            po, pml, out, h32, splits, q, QCV=8, HAS_SINKS=False, q_len=q_len, n_q_heads=N_Q, n_kv_heads=N_KV,
            head_dim=HD, HD_PAD=HD, V_DIM=HD, BLOCK_ROWS=block_rows, ROWS_SUB=rows_sub, D_SUB=d_sub,
            num_warps=4, num_stages=1)
        return out
    return go


def run_tri(q, cache, table, seqlens, q_len, scale, splits=None):
    qk, qv, sk, sv = cache
    out = torch.empty_like(q)

    def go():
        return paged_attn_triton_decode(q=q, k=None, v=None, k_cache=qk, v_cache=qv, block_table=table,
                                        cache_seqlens=seqlens, causal=True, softmax_scale=scale,
                                        qc=(sk, sv, 8, 8), pre_appended_len=q_len, n_kv_heads_override=N_KV,
                                        out=out, num_splits=splits)
    return go


def graph_time(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    ext = build()
    print(f"built in {time.time() - t0:.0f} s", flush=True)
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1030)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    scale = 1.0 / 16.0
    cases = [(1, 1, 37), (1, 1, 300), (1, 1, 2048), (1, 1, 8242), (1, 1, 28050),
             (1, 2, 8242), (1, 5, 8242), (1, 7, 300), (1, 7, 8242), (1, 7, 28050), (2, 1, 8242), (2, 7, 2048)]
    if a.quick:
        cases = [(1, 1, 300), (1, 7, 2048), (2, 1, 2048)]
    with open(a.out, "a") as out:
        for bsz, q_len, ctx in cases:
            pages = bsz * PAGES
            words = N_KV * HD // 32 * 8
            qk = torch.randint(-2**31, 2**31 - 1, (pages, PAGE, words), dtype=torch.int32, device=dev, generator=gen)
            qv = torch.randint(-2**31, 2**31 - 1, (pages, PAGE, words), dtype=torch.int32, device=dev, generator=gen)
            sk = (torch.rand((pages, PAGE, N_KV * HD // 32), device=dev, generator=gen) * 0.18 + 0.02).half()
            sv = (torch.rand((pages, PAGE, N_KV * HD // 32), device=dev, generator=gen) * 0.18 + 0.02).half()
            cache = (qk, qv, sk, sv)
            table = torch.randperm(pages, device=dev, generator=gen).int().view(bsz, PAGES)
            lens = [ctx - q_len] + [max(q_len, ctx // 2) - q_len for _ in range(bsz - 1)]
            seqlens = torch.tensor(lens, dtype=torch.int32, device=dev)
            q = torch.randn((bsz, q_len, N_Q, HD), device=dev, generator=gen).half()

            block_rows, h_blocks = decode_row_layout(q_len, N_Q // N_KV, HD)
            programs = bsz * N_KV * h_blocks
            rule = max(1, min(2 * sms // programs, triton.cdiv(PAGES * PAGE + q_len, 16), 128))
            ref = ref_attn(q, qk, qv, sk, sv, table, seqlens, q_len, scale)
            out_tri = run_tri(q, cache, table, seqlens, q_len, scale)().double().clone()
            row = {"bsz": bsz, "q_len": q_len, "ctx": ctx, "programs": programs, "splits_rule": rule,
                   "device": torch.cuda.get_device_name(dev), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            scale_ref = ref.abs().mean().item()
            row["ref_mean_abs"] = scale_ref
            row["tri_vs_ref_max"] = (out_tri - ref).abs().max().item()
            row["tri_vs_ref_mean"] = (out_tri - ref).abs().mean().item()
            for splits in sorted({rule, 2, 34}):
                o = run_fd(ext, q, cache, table, seqlens, q_len, splits, scale)().double()
                row[f"fd_s{splits}_vs_ref_max"] = (o - ref).abs().max().item()
                row[f"fd_s{splits}_vs_ref_mean"] = (o - ref).abs().mean().item()
                row[f"fd_s{splits}_vs_tri_max"] = (o - out_tri).abs().max().item()
                row[f"fd_s{splits}_nan"] = bool(torch.isnan(o).any().item())
            # W11b: splits over the live context (same data, the engine's split count)
            o = run_fd(ext, q, cache, table, seqlens, q_len, rule, scale, live=True)().double()
            row["fdlive_vs_ref_max"] = (o - ref).abs().max().item()
            row["fdlive_vs_ref_mean"] = (o - ref).abs().mean().item()
            row["fdlive_nan"] = bool(torch.isnan(o).any().item())
            t_tri = graph_time(run_tri(q, cache, table, seqlens, q_len, scale), a.iters)
            t_fd = graph_time(run_fd(ext, q, cache, table, seqlens, q_len, rule, scale), a.iters)
            t_live = graph_time(run_fd(ext, q, cache, table, seqlens, q_len, rule, scale, live=True), a.iters)
            row.update(tri_us=round(t_tri, 1), fd_us=round(t_fd, 1), fdlive_us=round(t_live, 1),
                       speedup=round(t_tri / t_fd, 2), live_vs_bound=round(t_fd / t_live, 2),
                       tri_ms_16layers=round(t_tri * 16 / 1000, 3), fd_ms_16layers=round(t_fd * 16 / 1000, 3),
                       fdlive_ms_16layers=round(t_live * 16 / 1000, 3))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"bsz {bsz} q_len {q_len} ctx {ctx}: tri {t_tri:.1f} us, fd {t_fd:.1f} us ({t_tri / t_fd:.2f}x), "
                  f"fd live {t_live:.1f} us ({t_fd / t_live:.2f}x over bound); "
                  f"err vs ref: tri {row['tri_vs_ref_max']:.2e}, fd {row[f'fd_s{rule}_vs_ref_max']:.2e} "
                  f"(ref mean |o| {scale_ref:.3f})", flush=True)


if __name__ == "__main__":
    main()
