#!/usr/bin/env python3
"""Per-prompt decode rates of a P8 confirmation session: an exploratory breakdown, not registered.

The registered analysis (analyze_confirm.py) pools each condition over its three prompts and its rounds. This script
splits the same rows by prompt (code, math, prose): tokens over decode time per condition and prompt, pooled over the
rounds and per round, and, where the session's NInferno logs carry them, DFlash2's accepted draft tokens per prompt.

  python3 per_prompt.py SESSION_DIR > per-prompt.md
Writes per-prompt.json in SESSION_DIR. Stdlib only.
"""
import collections
import json
import os
import re
import sys

TASKS = ("code", "math", "prose")


def rates(rows_path):
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for line in open(rows_path):
        if not line.strip():
            continue
        r = json.loads(line)
        m = re.fullmatch(r"confirm-(.+)-r(\d)", r.get("tag", ""))
        if not m:
            continue
        for key in ((m.group(1), r["task"], "all"), (m.group(1), r["task"], m.group(2))):
            acc[key[:2]][key[2]][0] += r["new_tokens"]
            acc[key[:2]][key[2]][1] += r["decode_wall_s"]
    return {f"{c}|{t}": {k: v[0] / v[1] for k, v in sorted(d.items())} for (c, t), d in acc.items()}


def acceptance(log_path):
    """Accepted and drafted tokens per request, in the order the log records them (warm-up first, then the prompts)."""
    if not os.path.exists(log_path):
        return []
    return [(int(a), int(b)) for a, b in re.findall(r"dflash2 accepted (\d+)/(\d+)", open(log_path).read())]


def main():
    d = sys.argv[1]
    out = {"rates": rates(os.path.join(d, "rows.jsonl")), "acceptance": {}}
    for cond in ("nif-xxs-df2-0", "nif-xxs-df2-28k"):
        req = acceptance(os.path.join(d, "logs", f"{cond}.log"))
        # each start: one 32-token warm-up, then code, math and prose; the log appends every round's start
        per = collections.defaultdict(lambda: [0, 0])
        for i, (a, b) in enumerate(req):
            k = i % 4
            if k:
                per[TASKS[k - 1]][0] += a
                per[TASKS[k - 1]][1] += b
        if per:
            out["acceptance"][cond] = {t: {"accepted": v[0], "drafted": v[1], "rate": v[0] / v[1]} for t, v in per.items()}
    with open(os.path.join(d, "per-prompt.json"), "w") as f:
        json.dump(out, f, indent=1)
    conds = sorted({k.split("|")[0] for k in out["rates"]})
    print("# Per-prompt decode tokens/s (exploratory; pooled over rounds, then each round)\n")
    print("| condition | " + " | ".join(TASKS) + " |\n|---|" + "---|" * len(TASKS))
    for c in conds:
        cells = []
        for t in TASKS:
            v = out["rates"].get(f"{c}|{t}")
            cells.append("--" if not v else f"{v['all']:.2f} ({', '.join(f'{v[k]:.1f}' for k in v if k != 'all')})")
        print(f"| {c} | " + " | ".join(cells) + " |")
    for c, per in out["acceptance"].items():
        print(f"\n{c} DFlash2 acceptance: " + ", ".join(f"{t} {v['accepted']}/{v['drafted']} ({100 * v['rate']:.1f}%)"
                                                         for t, v in per.items()))


if __name__ == "__main__":
    main()
