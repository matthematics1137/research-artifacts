#!/usr/bin/env python3
"""Paper 8 W9: simulate the int8 sq GEMV work decomposition on the 2080 Ti (stdlib only).

Mirrors exl3_gemv_int8_sq in exllamav3 1.6.5 + the W7 and W9 patches: the kernel's own rule (upstream) and the
W9 host rule, for every (k, n, K) shape of the qwen38-exl3-2.0 model at M = 1. The occupancy model is sm_75's:
at most 1024 threads (4 blocks of 256) per SM, 64 registers per thread (59 rounded up, 4 blocks), and 64 KB of
shared memory per SM in 256-byte granules, with about 144 B of static shared memory per block. The real choices
come from the build's EXL3_INT8_SQ_FIT_LOG=1 output; this is the prediction, written before that ran.
W10b: --m 2 3 4 adds, per shape and row count, whether W7's fit check takes the instance (it sizes the slice for
the 80 KB cap), the W10b cap on the slice height (what 64 KB holds), and the W9 host choice under that cap and
W10's workspace bound.
  python3 sq_fit_sim.py [--sms 68] [--m 2 3 4]
"""
import argparse

SQ_KSPLIT_CAP, SQ_MINROWS, SQ_ROWS_MAX, SQ_FIT_SAT = 64, 16, 512, 4
STATIC_SMEM = 144
SMEM_OPTIN = 65536                                   # sm_75 opt-in shared memory per block
WS_INTS, SQ_WS_RESERVED = 16 * 1024 * 1024 // 4, 4096 + 4 * SQ_KSPLIT_CAP * 4


def cdiv(a, b):
    return (a + b - 1) // b


def rows_max(M, residual):
    cap = (80 * 1024) // (32 + 64 * M * (2 if residual else 1))
    cap &= ~7
    return min(cap, SQ_ROWS_MAX)


def stage_bytes(K, half=False):
    staged = half or K in (3, 5, 7)
    return 8 * 4 * ((8 * (2 * K + 1)) if half else 16 * K) * 4 if staged else 0


def smem_for(rp, K, M=1, residual=False, half=False):
    return rp * 16 * 2 + rp * 16 * 4 * M * (2 if residual else 1) + stage_bytes(K, half) + 2 * M * 128 * 4


def occupancy(smem):
    per = cdiv(smem + STATIC_SMEM, 256) * 256
    return max(0, min(4, 65536 // per))


def kernel_rule(k, n, K, sms, M=1, residual=False):
    rows_total, nb256 = k // 16, n // 256
    rmax = rows_max(M, residual)

    def decomp(grid):
        r = cdiv(rows_total * nb256, grid)
        rp = (max(r, min(2 * r, 32)) + 7) & ~7
        rp = max(rp, SQ_MINROWS)
        rp = min(rp, rmax)
        rp = min(rp, (rows_total + 7) & ~7)
        return rp, cdiv(rows_total, rp)

    rp, _ = decomp(6 * sms)
    grid = min(max(occupancy(smem_for(rp, K, M, residual)), 1) * sms, 1024)
    rp, ks = decomp(grid)
    return rp, ks, grid


def rows_cap(K, M=1, residual=False, half=False):
    """W10b: the tallest slice (multiple of 8) whose shared layout fits the opt-in limit, at most rows_max"""
    room = SMEM_OPTIN - STATIC_SMEM - smem_for(0, K, M, residual, half)
    return min(rows_max(M, residual), (room // (32 + 64 * M * (2 if residual else 1))) & ~7) if room > 0 else 0


def w7_fits(K, M=1, residual=False, half=False):
    """W7's check: the instance's largest request (the 80 KB cap's slice) fits the opt-in limit"""
    return smem_for(rows_max(M, residual), K, M, residual, half) + STATIC_SMEM <= SMEM_OPTIN


def host_rule(k, n, K, sms, M=1, residual=False, half=False, cap=None):
    rows_total, nb256 = k // 16, n // 256
    hi = min(rows_max(M, residual) if cap is None else cap, (rows_total + 7) & ~7)
    row_cost = 4 * (2 * K + (1 if half else 0))
    part_cost = 16 * M * (2 if residual else 1)
    best = None
    for ks in range(cdiv(rows_total, hi), min(rows_total, SQ_KSPLIT_CAP) + 1):
        rp = max((cdiv(rows_total, ks) + 7) & ~7, SQ_MINROWS)
        if rp > hi or cdiv(rows_total, rp) != ks:
            continue
        if SQ_WS_RESERVED + ks * M * n * (2 if residual else 1) > WS_INTS:   # W10: partials fit the workspace
            continue
        g = min(max(occupancy(smem_for(rp, K, M, residual, half)), 1) * sms, 1024)
        waves = cdiv(ks * nb256, g)
        cost = (waves * rp * row_cost + ks * part_cost) * max(cdiv(g, sms), SQ_FIT_SAT)
        if best is None or cost < best[0]:
            best = (cost, rp, ks, g)
    return best[1:] if best else (0, 0, 0)


SHAPES = [  # (name, k, n, K, launches per token undrafted: layers using it)
    ("GDN qkv", 5120, 10240, 2, 47), ("GDN qkv", 5120, 10240, 1, 1),
    ("GDN z", 5120, 6144, 2, 48),
    ("GDN out_proj", 6144, 5120, 2, 39), ("GDN out_proj", 6144, 5120, 3, 9),
    ("MLP gate/up", 5120, 17408, 2, 2 * 59), ("MLP gate/up", 5120, 17408, 1, 2 * 4), ("MLP gate/up", 5120, 17408, 3, 2),
    ("MLP down", 17408, 5120, 2, 61), ("MLP down", 17408, 5120, 3, 3),
    ("attn q", 5120, 12288, 2, 14), ("attn q", 5120, 12288, 1, 2),
    ("attn k/v", 5120, 1024, 3, 2 * 8), ("attn k/v", 5120, 1024, 4, 2 * 7),
    ("attn o", 6144, 5120, 2, 10), ("attn o", 6144, 5120, 3, 6),
    ("lm_head", 5120, 248320, 3, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sms", type=int, default=68)
    ap.add_argument("--m", type=int, nargs="*", default=[], help="W10b: also simulate these row counts (2-4)")
    a = ap.parse_args()
    print("| matrix | k x n | K | per token | kernel rule rows/ksplit/grid (units, waves) | W9 host rows/ksplit/grid (units, waves) | changed |")
    print("|---|---|---|---|---|---|---|")
    for name, k, n, K, cnt in SHAPES:
        nb = n // 256
        rp0, ks0, g0 = kernel_rule(k, n, K, a.sms)
        rp1, ks1, g1 = host_rule(k, n, K, a.sms)
        u0, u1 = ks0 * nb, ks1 * nb
        ch = "yes" if (rp0, g0) != (rp1, g1) else ""
        print(f"| {name} | {k} x {n} | {K} | {cnt} | {rp0}/{ks0}/{g0} ({u0}, {u0 / g0:.2f}) | {rp1}/{ks1}/{g1} ({u1}, {u1 / g1:.2f}) | {ch} |")
    if not a.m:
        return
    print()
    print("| matrix | k x n | K | M | W7 check (80 KB slice) | W10b cap rows | W10b host rows/ksplit/grid (units, waves) | smem KB | blocks/SM |")
    print("|---|---|---|---|---|---|---|---|---|")
    for M in a.m:
        for name, k, n, K, cnt in SHAPES:
            cap = rows_cap(K, M)
            rp, ks, g = host_rule(k, n, K, a.sms, M=M, cap=cap) if cap >= SQ_MINROWS else (0, 0, 0)
            u = ks * (n // 256)
            sm = smem_for(rp, K, M) if rp else 0
            occ = occupancy(sm) if rp else 0
            fit = "takes" if w7_fits(K, M) else "declines"
            pick = f"{rp}/{ks}/{g} ({u}, {u / g:.2f})" if rp else "declined"
            print(f"| {name} | {k} x {n} | {K} | {M} | {fit} | {cap} | {pick} | {sm / 1024:.1f} | {occ} |")


if __name__ == "__main__":
    main()
