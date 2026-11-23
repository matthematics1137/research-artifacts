#!/usr/bin/env python3
"""P9 phase 1, M4's target (research/prereg/2026-11-05-p9-phase1.md): launches the phase kernels Nsight Compute profiles.
For mlp_gate and gdn_qkv at K = 1 and 2, one launch each of the full kernel (P = 7) and the loop alone (P = 2), at the
engine's own decomposition (kernel rule; this target runs on the laptop, where W9's host rule is off), after a warm-up.
Run only under tools/turing/p9_ncu.sh.
  python3 p9_ncu_target.py --src SRC/exllamav3/exllamav3_ext --tag NAME
"""
import argparse

import torch

import p9_common as pc
import p9_phases as pp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    minb = pp.build(a.src, a.tag).info(1, 7, 16)[2]
    ph = pp.build(a.src, a.tag, minb)
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(1105)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for name, k, n in [s for s in pc.MODEL if s[0] in ("mlp_gate", "gdn_qkv")]:
        for K in (1, 2):
            rp, grid = pp.kernel_rule(ph, K, k, n, sms, ph.rows_max())
            A = (torch.randn((1, k), device=dev, generator=gen) * 0.1).half()
            B = pc.trellis(k, n, K, dev, gen)
            suh, svh = pc.flips(k, dev, gen), pc.flips(n, dev, gen)
            C = torch.empty((1, n), dtype=torch.half, device=dev)
            locks = torch.zeros(ph.ws_reserved() + (-(-(k // 16) // rp)) * n, dtype=torch.int32, device=dev)
            for v in (7, 2):
                ph.info(K, v, rp)
                ph.launch(K, v, A, B, C, locks, suh, svh, k, n, rp, grid)
            torch.cuda.synchronize()
            print(f"{name} K={K} rows_per {rp} grid {grid}", flush=True)


if __name__ == "__main__":
    main()
