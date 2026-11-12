#!/usr/bin/env python3
"""P8 fd75d logit check: judge three numerics.py run files against research/prereg/2026-11-03-p8-fd75d-check.md.

F (fd75d), T1 and T2 (Triton) teacher-forced the same streams. Per length (28K, 48K, 64K), pooled, and per item:
exact per-position KL(T1 || F) and KL(T1 || T2) from the full logits (numerics' full_kl), and top-1 agreement of F and of
T2 with T1. fd75d diverges at a length if mean KL(T1||F) > 2 x mean KL(T1||T2) + 1e-4, or its top-1 agreement with T1
is more than 1 point below T2's. Also KL by answer position (the first value's 7 digits, later values).

Memory-light (the registration's implementation note, 2026-11-03): run files are opened lazily, each run's full logits
are read once, and KL is computed per item in chunks of positions, so float64 temporaries stay small.

  python fd75d_analyze.py F.npz T1.npz T2.npz [--out results.json]
"""
import argparse
import json
import os
import sys

import numpy as np

KL_FACTOR, KL_ABS, TOP1_POINTS = 2.0, 1e-4, 1.0
CHUNK = 16   # positions per KL chunk


def load(path):
    """A lazy view of a numerics.py run file: arrays are read only when used."""
    return np.load(path, allow_pickle=False)


def full_kl(xa, xb):
    a, b = xa.astype(np.float64), xb.astype(np.float64)
    la = a - (a.max(1, keepdims=True) + np.log(np.exp(a - a.max(1, keepdims=True)).sum(1, keepdims=True)))
    lb = b - (b.max(1, keepdims=True) + np.log(np.exp(b - b.max(1, keepdims=True)).sum(1, keepdims=True)))
    return (np.exp(la) * (la - lb)).sum(1)


class Run:
    """One run file: its full logits (read once), top-1 and targets, addressed by item id."""

    def __init__(self, D, label="default"):
        self.full = D[f"full__{label}"]
        self.top1 = D[f"{label}__top1"]
        self.targets = D["targets"]
        self.offsets = D["item_offsets"]
        self.ids = [str(x) for x in D["item_ids"]]
        self.full_row = {int(r): k for k, r in enumerate(D["full_index"])}

    def item(self, iid):
        i = self.ids.index(iid)
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        return [self.full_row[r] for r in range(s, e)], self.top1[s:e], self.targets[s:e]


def kl_rows(A, ra, B, rb):
    """Exact KL(A || B) at each paired row, CHUNK positions at a time."""
    return np.concatenate([full_kl(A.full[ra[j:j + CHUNK]], B.full[rb[j:j + CHUNK]])
                           for j in range(0, len(ra), CHUNK)])


def stats(v):
    v = np.asarray(v, np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def judge(F, T1, T2):
    f, t1, t2 = Run(F), Run(T1), Run(T2)
    ids = [i for i in t1.ids if i in f.ids and i in t2.ids]
    res = {"items": {}, "lengths": {}, "rule": {"kl_factor": KL_FACTOR, "kl_abs": KL_ABS, "top1_points": TOP1_POINTS}}
    groups = {}
    for i in ids:
        (rf, tf_, gf), (r1, t1_, g1), (r2, t2_, g2) = f.item(i), t1.item(i), t2.item(i)
        if not (np.array_equal(gf, g1) and np.array_equal(g2, g1)):   # the same forced tokens at the same positions
            raise SystemExit(f"stream mismatch on {i}")
        kf, k0 = kl_rows(t1, r1, f, rf), kl_rows(t1, r1, t2, r2)
        af, a0 = (tf_ == t1_), (t2_ == t1_)
        res["items"][i] = {"positions": int(kf.size), "kl_f": stats(kf), "kl_floor": stats(k0),
                           "top1_f": float(af.mean()), "top1_floor": float(a0.mean()),
                           "kl_f_first_value": float(kf[:7].mean()), "kl_f_later_values": float(kf[8:].mean())}
        groups.setdefault(i.split("-", 1)[0], []).append((kf, k0, af, a0))
    groups["pooled"] = [x for g in list(groups.values()) for x in g]
    for name, g in groups.items():
        kf = np.concatenate([x[0] for x in g])
        k0 = np.concatenate([x[1] for x in g])
        af = np.concatenate([x[2] for x in g])
        a0 = np.concatenate([x[3] for x in g])
        mf, m0 = float(kf.mean()), float(k0.mean())
        tf, t0 = float(af.mean()), float(a0.mean())
        res["lengths"][name] = {
            "positions": int(kf.size), "kl_f": stats(kf), "kl_floor": stats(k0), "top1_f": tf, "top1_floor": t0,
            "kl_threshold": KL_FACTOR * m0 + KL_ABS,
            "diverges": bool(mf > KL_FACTOR * m0 + KL_ABS or 100 * (t0 - tf) > TOP1_POINTS)}
    deep = [res["lengths"][k]["diverges"] for k in ("48k", "64k") if k in res["lengths"]]
    res["decision"] = {"diverges_28k": res["lengths"].get("28k", {}).get("diverges"),
                       "diverges_48k_or_64k": any(deep) if deep else None,
                       "next": ("the accuracy arm (research/prereg/2026-11-03-p8-fd75d-accuracy-arm.md), after the "
                                "ambassador's gate record" if any(deep) else
                                "stop: the kernel is ruled out as the H2s gap's source at the logit level")}
    res["predictions"] = {"F1_no_divergence_28k": None if res["decision"]["diverges_28k"] is None else
                          ("holds" if not res["decision"]["diverges_28k"] else "refuted"),
                          "F2_no_divergence_48k_64k": None if not deep else ("holds" if not any(deep) else "refuted")}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("f")
    ap.add_argument("t1")
    ap.add_argument("t2")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = judge(load(args.f), load(args.t1), load(args.t2))
    text = json.dumps(res, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(json.dumps({"lengths": {k: {x: v[x] for x in ("top1_f", "top1_floor", "kl_threshold", "diverges")}
                                  | {"kl_f_mean": v["kl_f"]["mean"], "kl_floor_mean": v["kl_floor"]["mean"]}
                                  for k, v in res["lengths"].items()}, "decision": res["decision"]}, indent=1))


if __name__ == "__main__":
    main()
