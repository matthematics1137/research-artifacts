#!/usr/bin/env python3
"""Traffic shape from TabbyAPI server logs — lengths only, never content. Reads the per-request Metrics lines
TabbyAPI prints by default ("Process: N cached tokens and M new tokens at X T/s", "Generate: Y T/s", context
length, generated tokens) and reports what the serving config depends on: prompt lengths, cached-vs-new token
split (prefix reuse rate), generation lengths, and how often a turn looks like an early edit (cached share low on
a long prompt). usage: traffic_stats.py <log file or dir> [more ...]   (default: engines/tabbyAPI/logs/)"""
import glob, os, re, statistics, sys

# One line per request, e.g. (TabbyAPI 0.0.x, engines/tabbyAPI/logs/2026-08-26_*.log):
#   Metrics (ID: ...): 61 tokens generated in 2.5 seconds (Queue: 0.0 s, Process: 0 cached tokens and 25 new tokens
#   at 40.98 T/s, Generate: 32.33 T/s, Context: 25 tokens)
RE_PROC = re.compile(r"Process:\s*(\d[\d,]*)\s*cached tokens? and (\d[\d,]*)\s*new tokens? at\s*([\d.]+)\s*T/s", re.I)
RE_GEN = re.compile(r"Generate:\s*([\d.]+)\s*T/s", re.I)
RE_CTX = re.compile(r"Context:\s*(\d[\d,]*)\s*tokens", re.I)
RE_GENTOK = re.compile(r"(\d[\d,]*)\s*tokens generated in\s*([\d.]+)\s*seconds", re.I)
RE_QUEUE = re.compile(r"Queue:\s*([\d.]+)\s*s", re.I)


def parse(path):
    reqs = []
    for line in open(path, errors="replace"):
        if "Metrics" not in line: continue
        m = RE_PROC.search(line)
        if not m: continue
        r = {"cached": int(m.group(1).replace(",", "")), "new": int(m.group(2).replace(",", "")), "pp_tps": float(m.group(3))}
        g = RE_GEN.search(line); c = RE_CTX.search(line); t = RE_GENTOK.search(line); qu = RE_QUEUE.search(line)
        if g: r["gen_tps"] = float(g.group(1))
        if c: r["ctx"] = int(c.group(1).replace(",", ""))
        if t: r["gen_tokens"] = int(t.group(1).replace(",", "")); r["total_s"] = float(t.group(2))
        if qu: r["queue_s"] = float(qu.group(1))
        reqs.append(r)
    return reqs


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    paths = sys.argv[1:] or ["/path/to/lab-repo/engines/tabbyAPI/logs"]
    files = []
    for p in paths: files += sorted(glob.glob(os.path.join(p, "*.log"))) if os.path.isdir(p) else [p]
    reqs = [r for f in files for r in parse(f)]
    if not reqs: print("no Metrics lines found in", files); return
    prompt = [r["cached"] + r["new"] for r in reqs]; new = [r["new"] for r in reqs]; cached = [r["cached"] for r in reqs]
    share = [r["cached"] / max(1, r["cached"] + r["new"]) for r in reqs]
    edits = sum(1 for r in reqs if r["cached"] + r["new"] >= 2048 and r["cached"] < 0.5 * (r["cached"] + r["new"]))
    cold = sum(1 for r in reqs if r["cached"] == 0)
    print(f"files {len(files)}  requests {len(reqs)}")
    print(f"prompt tokens: median {q(prompt, .5)}  p90 {q(prompt, .9)}  max {max(prompt)}")
    print(f"new (prefilled) tokens per request: median {q(new, .5)}  p90 {q(new, .9)}   cached share median {statistics.median(share):.2f}")
    print(f"cold requests (0 cached): {cold} ({100 * cold / len(reqs):.0f}%)   long prompts with <50% reuse (edit-like): {edits} ({100 * edits / len(reqs):.0f}%)")
    gt = [r["gen_tokens"] for r in reqs if "gen_tokens" in r]; gs = [r["gen_tps"] for r in reqs if "gen_tps" in r]; ps = [r["pp_tps"] for r in reqs]
    qs = [r["queue_s"] for r in reqs if "queue_s" in r]
    if gt: print(f"generated tokens: median {q(gt, .5)}  p90 {q(gt, .9)}  max {max(gt)}   (decode-dominated: {100 * sum(1 for r in reqs if r.get('gen_tokens', 0) > r['new']) / len(reqs):.0f}% of requests generate more than they prefill)")
    print(f"prefill T/s median {q(ps, .5)}   decode T/s median {q(gs, .5) if gs else None}   queue wait p90 {q(qs, .9) if qs else None} s")
    print("serving implications: stash needs ~300 MB host RAM per live prefix; set sysmem_recurrent_cache to cover the number of distinct")
    print("  prompts in flight; a high edit-like share means EXL3_CKPT_PP (serving/launch.sh --ckpt-pp) is worth its 2-3% prefill cost.")


if __name__ == "__main__":
    main()
