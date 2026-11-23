#!/usr/bin/env python3
"""Paper 8 W2: per-kernel decode census from census.sh's nsys captures (stdlib only, CPU only).

For each config (nodraft, mtp) census.sh leaves <cfg>.done naming its attempt stamp, the capture reports
<cfg>-<stamp>*.nsys-rep (one per task, from --capture-range-end=repeat) and decode rows tagged <tag>-<cfg>-<stamp> in
decode.jsonl. This script exports each report to SQLite with `nsys export` (skipped when the .sqlite exists) and
writes report.md and report.json beside them:
  - per task: decoded tokens, capture span, kernel time, GPU busy time (union of kernel intervals), busy fraction,
    and the per-token span, busy and idle time (idle = the GPU waiting on the host: launches, Python, sync);
  - per kernel family (demangled name without template arguments), summed over the tasks: launches and
    microseconds per decoded token, share of kernel time, and the most frequent launch shape (grid, block,
    registers, static + dynamic shared memory);
  - memcpy/memset time and the CUDA runtime calls per token (graph launches vs single kernel launches).
"""
import argparse
import collections
import glob
import json
import os
import re
import sqlite3
import subprocess


def family(name):
    p = re.sub(r"^void ", "", name or "?")
    p = re.sub(r"<.*", "", p)
    p = re.sub(r"\(.*", "", p)
    return p.split("::")[-1] or "?"


def export(rep, nsys):
    db = rep[: -len(".nsys-rep")] + ".sqlite"
    if not os.path.exists(db):
        subprocess.run([nsys, "export", "--type", "sqlite", "--output", db, rep], check=True,
                       stdout=subprocess.DEVNULL)
    return db


def table_rows(con, sql):
    try:
        return con.execute(sql).fetchall()
    except sqlite3.OperationalError:  # the table is absent when the capture had no such events
        return []


def union_ns(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0)


def read_capture(db):
    con = sqlite3.connect(db)
    names = dict(table_rows(con, "SELECT id, value FROM StringIds"))
    kern = table_rows(con, "SELECT start, end, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, "
                           "registersPerThread, staticSharedMemory, dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL")
    mem = table_rows(con, "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY") + \
        table_rows(con, "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET")
    api = table_rows(con, "SELECT nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    con.close()
    return names, kern, mem, collections.Counter(re.sub(r"_v\d+$", "", names.get(a[0], "?")) for a in api)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--nsys", default="/usr/local/cuda-13.4/bin/nsys")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(os.path.join(a.dir, "decode.jsonl"))]
    report, md = {}, [f"# Decode kernel census ({os.path.basename(a.dir.rstrip('/'))})", ""]
    for cfg in ("nodraft", "mtp"):
        marker = os.path.join(a.dir, f"{cfg}.done")
        if not os.path.exists(marker):
            continue
        stamp = open(marker).read().split()[0]
        stem = f"{cfg}-{stamp}"  # repeat captures add a range number between the stem and the extension
        reps = sorted(glob.glob(os.path.join(a.dir, f"{stem}*.nsys-rep")),
                      key=lambda p: int("0" + re.sub(r"\D", "", os.path.basename(p)[len(stem):-len(".nsys-rep")])))
        decs = [r for r in rows if r["tag"].endswith(f"-{cfg}-{stamp}")]
        fam = collections.defaultdict(lambda: {"n": 0, "ns": 0, "shapes": collections.Counter()})
        by_shape = collections.defaultdict(lambda: [0, 0])  # (family, grid, block) -> [launches, ns]
        tasks, apis, tok_total, kern_total = [], collections.Counter(), 0, 0
        for rep, dec in zip(reps, decs):
            names, kern, mem, api = read_capture(export(rep, a.nsys))
            ivs = [(k[0], k[1]) for k in kern]
            span = (max(e for _, e in ivs + mem) - min(s for s, _ in ivs + mem)) if ivs else 0
            busy, ksum = union_ns(ivs + mem), sum(e - s for s, e in ivs)
            tok = dec["new_tokens"]
            tok_total += tok
            kern_total += ksum
            apis.update(api)
            for k in kern:
                f = fam[family(names.get(k[2]))]
                f["n"] += 1
                f["ns"] += k[1] - k[0]
                f["shapes"][(k[3], k[4], k[5], k[6], k[7], k[8], k[9], k[10] + k[11])] += 1
                s = by_shape[(family(names.get(k[2])), f"{k[3]}x{k[4]}x{k[5]}", f"{k[6]}x{k[7]}x{k[8]}")]
                s[0] += 1
                s[1] += k[1] - k[0]
            tasks.append({"task": dec["task"], "tokens": tok, "decode_wall_s": dec["decode_wall_s"],
                          "draft_accept": dec.get("draft_accept"), "kernels": len(kern),
                          "span_ms": round(span / 1e6, 2), "kernel_ms": round(ksum / 1e6, 2),
                          "busy_ms": round(busy / 1e6, 2), "busy_frac": round(busy / max(span, 1), 4),
                          "ms_per_token": round(span / 1e6 / tok, 3), "busy_ms_per_token": round(busy / 1e6 / tok, 3),
                          "idle_ms_per_token": round((span - busy) / 1e6 / tok, 3),
                          "kernels_per_token": round(len(kern) / tok, 1), "memops_ms": round(union_ns(mem) / 1e6, 2),
                          "report": os.path.basename(rep)})
        fams = sorted(({"family": n, "launches_per_token": round(v["n"] / tok_total, 2),
                        "us_per_token": round(v["ns"] / 1e3 / tok_total, 1),
                        "share": round(v["ns"] / max(kern_total, 1), 4),
                        "mean_us": round(v["ns"] / 1e3 / v["n"], 2),
                        "top_shape": dict(zip(("gx", "gy", "gz", "bx", "by", "bz", "regs", "smem"),
                                              v["shapes"].most_common(1)[0][0])),
                        "shapes": len(v["shapes"])} for n, v in fam.items()), key=lambda f: -f["us_per_token"])
        shapes = sorted(({"family": f, "grid": g, "block": b, "launches_per_token": round(n / tok_total, 2),
                          "us_per_token": round(ns / 1e3 / tok_total, 1), "share": round(ns / max(kern_total, 1), 4),
                          "mean_us": round(ns / 1e3 / n, 2)} for (f, g, b), (n, ns) in by_shape.items()),
                        key=lambda s: -s["us_per_token"])
        report[cfg] = {"stamp": stamp, "tasks": tasks, "families": fams, "shapes": shapes,
                       "runtime_calls_per_token": {k: round(v / tok_total, 2) for k, v in apis.most_common(12)}}
        md += [f"## {cfg} (attempt {stamp})", "",
               "| task | tokens | accept | ms/token (span) | GPU busy ms/token | idle ms/token | busy | kernels/token |",
               "|---|---|---|---|---|---|---|---|"]
        md += [f"| {t['task']} | {t['tokens']} | {t['draft_accept'] if t['draft_accept'] is not None else '-'} | "
               f"{t['ms_per_token']} | {t['busy_ms_per_token']} | {t['idle_ms_per_token']} | {t['busy_frac']:.1%} | "
               f"{t['kernels_per_token']} |" for t in tasks]
        md += ["", f"Kernel families, summed over the tasks (top {a.top} by time per decoded token):", "",
               "| family | launches/token | us/token | share | mean us | grid | block | regs | smem B | shapes |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for f in fams[: a.top]:
            s = f["top_shape"]
            md.append(f"| `{f['family']}` | {f['launches_per_token']} | {f['us_per_token']} | {f['share']:.1%} | "
                      f"{f['mean_us']} | {s['gx']}x{s['gy']}x{s['gz']} | {s['bx']}x{s['by']}x{s['bz']} | {s['regs']} | "
                      f"{s['smem']} | {f['shapes']} |")
        md += ["", "Launch shapes (one kernel family can serve several matrices; top 12 by time per decoded token):", "",
               "| family | grid | block | launches/token | us/token | share | mean us |", "|---|---|---|---|---|---|---|"]
        md += [f"| `{s['family']}` | {s['grid']} | {s['block']} | {s['launches_per_token']} | {s['us_per_token']} | "
               f"{s['share']:.1%} | {s['mean_us']} |" for s in shapes[:12]]
        md += ["", "Runtime calls per decoded token: " +
               ", ".join(f"{k} {v}" for k, v in report[cfg]["runtime_calls_per_token"].items()), ""]
    with open(os.path.join(a.dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
