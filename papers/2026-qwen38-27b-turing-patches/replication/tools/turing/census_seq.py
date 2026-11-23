#!/usr/bin/env python3
"""Paper 8 W2 helper: the kernel sequence of one decoded token from a census SQLite export (stdlib only).

Tokens are split at a kernel family that runs exactly once per token (default: the sampler's
`unrolled_elementwise_kernel`). Prints the chosen token's kernels in order, consecutive repeats of the same
(family, grid, block) collapsed, with launches, total and mean microseconds, so the per-layer pattern and the
matrix behind each launch shape can be read off.
  python3 census_seq.py REPORT.sqlite [--token N] [--split FAMILY]
"""
import argparse
import re
import sqlite3


def family(name):
    p = re.sub(r"^void ", "", name or "?")
    p = re.sub(r"<.*", "", p)
    p = re.sub(r"\(.*", "", p)
    return p.split("::")[-1] or "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--token", type=int, default=100)
    ap.add_argument("--split", default="unrolled_elementwise_kernel")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = con.execute("SELECT start, end, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ "
                       "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    cuts = [i for i, r in enumerate(rows) if family(names.get(r[2])) == a.split]
    lo, hi = cuts[a.token], cuts[a.token + 1]
    seq = []
    for r in rows[lo + 1: hi + 1]:
        key = (family(names.get(r[2])), f"{r[3]}x{r[4]}x{r[5]}", f"{r[6]}x{r[7]}x{r[8]}")
        us = (r[1] - r[0]) / 1e3
        if seq and seq[-1][0] == key:
            seq[-1][1] += 1
            seq[-1][2] += us
        else:
            seq.append([key, 1, us])
    total = sum(s[2] for s in seq)
    span = (rows[hi][1] - rows[lo][1]) / 1e3
    print(f"token {a.token}: {hi - lo} kernels, kernel time {total:.0f} us, span {span:.0f} us")
    for (fam, grid, block), n, us in seq:
        print(f"{n:4d} x {fam:<45} grid {grid:<10} block {block:<10} {us:9.1f} us  ({us / n:7.1f} each)")


if __name__ == "__main__":
    main()
