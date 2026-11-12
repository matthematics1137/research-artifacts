"""P9 phase 1 (research/prereg/2026-11-05-p9-phase1.md): shapes, synthetic trellis tensors and the cache-rotated graph
timing that M1 (p9_bench.py) and M2 (p9_phases.py) share.

Each case allocates N copies of its weight tensor, N = max(20, 3 L2 / bytes) capped at 400, so that the bytes one graph
replay reads exceed three L2 sizes where the cap allows (the 1 November bench replayed one tensor and read small
matrices from L2: research-ledger-reference). One CUDA graph holds one call per copy; replays repeat until a timing covers
at least 600 calls. The per-call time is the elapsed time over the calls, so it includes the in-graph gap a decode step
also pays. Each timing follows WARM_S seconds of the same graph's replays (Amendment 1: the smoke test showed the SM
clock still rising during the first minute).
"""
import math
import time

import torch

MODEL = [("attn_kv", 5120, 1024), ("gdn_z", 5120, 6144), ("gdn_out", 6144, 5120), ("gdn_qkv", 5120, 10240),
         ("attn_q", 5120, 12288), ("mlp_gate", 5120, 17408), ("mlp_down", 17408, 5120)]
TINY = [("tiny", 128, 256)]
GRID = [(f"g{k}x{n}", k, n) for k in (2048, 5120, 10240, 17408) for n in (1024, 4096, 10240, 17408)]
FUSED = [("fused_gate_up", 5120, 34816), ("fused_qkv_z", 5120, 16384), ("fused_q_k_v", 5120, 14336)]
SETS = {"model": (MODEL, (1, 2, 3, 4)), "tiny": (TINY, (1, 2)), "grid": (GRID, (1, 2)), "fused": (FUSED, (1, 2))}
MIN_COPIES, MAX_COPIES, MIN_CALLS, WARM_S, WARM_FIRST_S = 20, 400, 600, 0.15, 2.0
HOT_C, RESUME_C = 80.0, 72.0   # Amendment 2: pause above HOT_C until below RESUME_C (the 87/84 C stop stays)
_warmed = False   # the first timing of a process warms for WARM_FIRST_S: the GPU starts from its idle clocks


def cases(names):
    """(set, shape, k, n, K) for the named sets, in a fixed order (each round rotates it)."""
    out = []
    for s in names:
        shapes, ks = SETS[s]
        out += [(s, name, k, n, K) for K in ks for name, k, n in shapes]
    return out


def rotated(items, r, step=5):
    if not items:
        return items
    o = (r * step) % len(items)
    return items[o:] + items[:o]


def copies(k, n, K, l2_bytes):
    return min(MAX_COPIES, max(MIN_COPIES, math.ceil(3 * l2_bytes / (k * n * K / 8))))


def trellis(k, n, K, dev, gen):
    return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=dev, generator=gen)


def flips(size, dev, gen):
    return (torch.randint(0, 2, (size,), device=dev, generator=gen) * 2 - 1).half()


def time_calls(calls, warm_s=WARM_S):
    """Per-call microseconds over one CUDA graph holding the calls, replayed until MIN_CALLS calls; wall start, end.
    The graph first replays for warm_s seconds, so the SM clock has risen from any idle gap before the timing."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:3]:
            c()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    global _warmed
    reps = math.ceil(MIN_CALLS / len(calls))
    w0 = time.time()
    if not _warmed:
        warm_s, _warmed = max(warm_s, WARM_FIRST_S), True
    while time.time() - w0 < warm_s:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.time()
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    t1 = time.time()
    del g
    return a.elapsed_time(b) * 1000 / (reps * len(calls)), reps * len(calls), t0, t1


def last_temp(clocks_csv):
    """The clock sampler's last temperature reading in C, or None."""
    try:
        with open(clocks_csv, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 400))
            line = f.read().decode(errors="replace").strip().splitlines()[-1]
        return float(line.split(",")[4])
    except (OSError, IndexError, ValueError):
        return None


def cool(clocks_csv, hot_c=HOT_C, resume_c=RESUME_C, max_wait_s=600):
    """Amendment 2: above hot_c, wait until the GPU is below resume_c (at most max_wait_s), then make the next timing
    warm for WARM_FIRST_S again. Returns the seconds waited."""
    global _warmed
    t = last_temp(clocks_csv) if clocks_csv else None
    if t is None or t <= hot_c:
        return 0.0
    w0 = time.time()
    while time.time() - w0 < max_wait_s:
        time.sleep(2)
        t = last_temp(clocks_csv)
        if t is not None and t < resume_c:
            break
    _warmed = False
    return time.time() - w0


def too_hot(clocks_csv, limit_c):
    """True when the clock sampler's last temperature reading is above limit_c (no file or no reading: False)."""
    if not clocks_csv or not limit_c:
        return False
    try:
        with open(clocks_csv, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 400))
            line = f.read().decode(errors="replace").strip().splitlines()[-1]
        return float(line.split(",")[4]) > limit_c
    except (OSError, IndexError, ValueError):
        return False


def device_stamp(dev):
    p = torch.cuda.get_device_properties(dev)
    return {"device": p.name, "sms": p.multi_processor_count, "l2_bytes": p.L2_cache_size, "cc": f"{p.major}.{p.minor}",
            "torch": torch.__version__}
