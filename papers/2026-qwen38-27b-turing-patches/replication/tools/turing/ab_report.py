#!/usr/bin/env python3
"""Paper 8: summary of interleaved A/B speed repeats from ab_speed.sh (stdlib only).

Rows are tagged TAG-NAME-rROUND. For each arm and round: decode tok/s per prompt (new tokens over the decode wall
time, no profiler) and their mean over the prompts, plus the engine build and EXL3_* switches the rows recorded
(an arm whose rows disagree on those is flagged). Then, for each pair named with --pair A:B, B's change against A
in every round, and how many rounds B was faster.
Since 2026-10-31 it also reports pooled speed: total new tokens over total decode wall time, per round and over all
rounds, with pooled ms/token. The mean of per-prompt rates overstates speed when prompts run at very different rates
(MTP: prose drafts worst), by about 6% on the 2080 Ti's MTP arms; undrafted arms agree within about 0.3%. Pair
changes are given both ways.
  python3 ab_report.py ROWS.jsonl --tag ab1 [--pair stock:w7g1 ...] [--md OUT.md]
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pair", action="append", default=[], help="A:B, B compared with A in each round")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    pat = re.compile(rf"^{re.escape(a.tag)}-(.+)-r(\d+)$")
    tps = defaultdict(dict)            # (arm, round) -> {task: tok/s}
    tok = defaultdict(lambda: [0, 0.0])  # (arm, round) -> [new tokens, decode wall s], pooled over the prompts
    builds = defaultdict(set)          # arm -> {(version, switches)}
    for line in open(a.rows):
        r = json.loads(line)
        m = pat.match(r.get("tag", ""))
        if not m:
            continue
        arm, rnd = m.group(1), int(m.group(2))
        tps[(arm, rnd)][r["task"]] = r["new_tokens"] / r["decode_wall_s"]
        tok[(arm, rnd)][0] += r["new_tokens"]
        tok[(arm, rnd)][1] += r["decode_wall_s"]
        builds[arm].add((r.get("exllamav3"), json.dumps(r.get("env", {}), sort_keys=True)))
    arms = list(dict.fromkeys(k[0] for k in sorted(tps, key=lambda k: k[1])))  # round 1's run order
    rounds = sorted({k[1] for k in tps})
    tasks = sorted({t for v in tps.values() for t in v})
    out = [f"# Interleaved speed repeats: {a.tag}", "",
           "Decode tok/s, mean over the prompts (" + ", ".join(tasks) + "), per round; no profiler.", "",
           "| arm | build | " + " | ".join(f"round {r}" for r in rounds) + " | mean | ms/token | pooled tok/s | pooled ms/token |",
           "|---|---|" + "---|" * len(rounds) + "---|---|---|---|"]
    mean_of = {}

    def pooled(arm, rnd=None):
        keys = [k for k in tok if k[0] == arm and (rnd is None or k[1] == rnd)]
        t, w = sum(tok[k][0] for k in keys), sum(tok[k][1] for k in keys)
        return t / w if w else float("nan")

    for arm in arms:
        vals = []
        for r in rounds:
            d = tps.get((arm, r), {})
            vals.append(st.mean(d.values()) if len(d) == len(tasks) else None)
            mean_of[(arm, r)] = vals[-1]
        ok = [v for v in vals if v is not None]
        b = builds[arm]
        build = "; ".join(f"{v} {s}" for v, s in sorted(b)) + (" **(rows disagree)**" if len(b) > 1 else "")
        out.append(f"| {arm} | {build} | " + " | ".join(f"{v:.2f}" if v else "-" for v in vals)
                   + f" | {st.mean(ok):.2f} | {1000 / st.mean(ok):.2f} | {pooled(arm):.2f} | {1000 / pooled(arm):.2f} |")
    for p in a.pair:
        x, y = p.split(":")
        out += ["", f"**{y} against {x}** (tok/s change per round):", ""]
        ch = []
        for r in rounds:
            vx, vy = mean_of.get((x, r)), mean_of.get((y, r))
            if vx and vy:
                ch.append((r, (vy / vx - 1) * 100))
        out.append(", ".join(f"round {r}: {c:+.1f}%" for r, c in ch)
                   + f"; faster in {sum(c > 0 for _, c in ch)} of {len(ch)} rounds; mean {st.mean(c for _, c in ch):+.1f}%")
        pc = [(r, (pooled(y, r) / pooled(x, r) - 1) * 100) for r in rounds if (x, r) in tok and (y, r) in tok]
        if pc:
            out.append("pooled: " + ", ".join(f"round {r}: {c:+.1f}%" for r, c in pc)
                       + f"; faster in {sum(c > 0 for _, c in pc)} of {len(pc)} rounds; all rounds {(pooled(y) / pooled(x) - 1) * 100:+.1f}%")
    text = "\n".join(out) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    main()
