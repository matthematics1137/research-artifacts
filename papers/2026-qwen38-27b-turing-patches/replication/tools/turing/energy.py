#!/usr/bin/env python3
"""Paper 8 L1: energy per token for ab_speed.sh rows, from the desktop's 1 Hz nvidia-smi log (stdlib only).

The telemetry is turing-telemetry.service's CSV: timestamp, temperature, fan, power.draw, clocks.sm, clocks.mem,
utilization, memory.used, clocks_throttle_reasons.active. Each row's decode window is [row time - decode_wall_s,
row time]. Joules are the window's mean power times decode_wall_s; tokens per joule is new_tokens over that. Per arm
(TAG-NAME-rROUND, rounds pooled), pooled tok/s and tokens per joule (since 2026-10-31; pairs use them), the medians over its rows, the share of samples with the software power-cap flag
(0x4) set, and the adopted/stock ratios per condition for each --pair.
  python3 energy.py ROWS.jsonl TELEMETRY.csv --tag final1 --pair stock-0:adopt-0 ... [--md OUT.md] [--json OUT.json]
"""
import argparse
import bisect
import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta

POWER_CAP = 0x4


def telemetry(path):
    rows = []
    for line in open(path):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9 or not f[3].endswith("W"):
            continue
        t = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f")
        rows.append((t, float(f[3].rstrip(" W")), int(f[4].split()[0]), int(f[5].split()[0]), int(f[8], 16)))
    rows.sort()
    return rows


def window(tel, times, end, seconds):
    lo = bisect.bisect_left(times, end - timedelta(seconds=seconds))
    hi = bisect.bisect_right(times, end)
    return tel[lo:hi]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows")
    ap.add_argument("telemetry")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pair", action="append", default=[], help="A:B, B compared with A")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    tel = telemetry(a.telemetry)
    times = [r[0] for r in tel]
    pat = re.compile(rf"^{re.escape(a.tag)}-(.+)-r(\d+)$")
    per = defaultdict(list)            # arm -> [row dict]
    for line in open(a.rows):
        r = json.loads(line)
        m = pat.match(r.get("tag", ""))
        if not m:
            continue
        end = datetime.fromisoformat(r["time"]).replace(tzinfo=None)   # local time, as the telemetry
        w = window(tel, times, end, r["decode_wall_s"])
        watts = st.mean(s[1] for s in w) if w else float("nan")
        joules = watts * r["decode_wall_s"]
        per[m.group(1)].append({"task": r["task"], "round": int(m.group(2)), "samples": len(w),
                                "tokens": r["new_tokens"], "wall": r["decode_wall_s"],
                                "tps": r["new_tokens"] / r["decode_wall_s"], "watts": watts,
                                "tok_per_j": r["new_tokens"] / joules if w else float("nan"),
                                "sm_mhz": st.median(s[2] for s in w) if w else None,
                                "mem_mhz": st.median(s[3] for s in w) if w else None,
                                "cap_share": sum(bool(s[4] & POWER_CAP) for s in w) / len(w) if w else None})
    arms = {}
    for arm, rs in per.items():
        ok = [x for x in rs if x["samples"] >= 3]
        arms[arm] = {"rows": len(rs), "short_windows": len(rs) - len(ok),
                     "tps": st.median(x["tps"] for x in rs),
                     "watts": st.mean(x["watts"] for x in rs if x["samples"]),
                     "tok_per_j": st.median(x["tok_per_j"] for x in rs if x["samples"]),
                     "sm_mhz": st.median(x["sm_mhz"] for x in rs if x["samples"]),
                     "mem_mhz": st.median(x["mem_mhz"] for x in rs if x["samples"]),
                     "cap_share": st.mean(x["cap_share"] for x in rs if x["samples"])}
        # pooled (2026-10-31): total tokens over total time and total joules; a median of per-row ratios overstates
        # MTP arms, whose prompts run at very different rates. Pairs use these.
        s_ = [x for x in rs if x["samples"]]
        arms[arm]["tps_pooled"] = sum(x["tokens"] for x in rs) / sum(x["wall"] for x in rs)
        arms[arm]["tok_per_j_pooled"] = sum(x["tokens"] for x in s_) / sum(x["watts"] * x["wall"] for x in s_)
    pairs = []
    for p in a.pair:
        A, B = p.split(":")
        sr = arms[B]["tps_pooled"] / arms[A]["tps_pooled"]
        er = arms[B]["tok_per_j_pooled"] / arms[A]["tok_per_j_pooled"]
        pairs.append({"a": A, "b": B, "speed_ratio": sr, "tok_per_j_ratio": er, "energy_vs_speed": er / sr - 1})
    lines = [f"# Energy per token, `{a.tag}`", "",
             "Pooled: total tokens over total time or joules. Median: energy.py's figure before 2026-10-31, kept for"
             " comparison.", "",
             "| arm | rows | short windows | pooled tok/s | mean W | pooled tokens/J | median tokens/J | SM MHz | mem MHz"
             " | power-cap share |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(arms):
        x = arms[arm]
        lines.append(f"| {arm} | {x['rows']} | {x['short_windows']} | {x['tps_pooled']:.2f} | {x['watts']:.1f} | "
                     f"{x['tok_per_j_pooled']:.4f} | {x['tok_per_j']:.4f} | {x['sm_mhz']:.0f} | {x['mem_mhz']:.0f} | "
                     f"{x['cap_share']:.0%} |")
    if pairs:
        lines += ["", "| pair | speed ratio | tokens/J ratio | tokens/J ratio vs speed ratio |", "|---|---|---|---|"]
        for p in pairs:
            lines.append(f"| {p['a']} -> {p['b']} | {p['speed_ratio']:.3f} | {p['tok_per_j_ratio']:.3f} | "
                         f"{p['energy_vs_speed']:+.1%} |")
    out = "\n".join(lines) + "\n"
    print(out)
    if a.md:
        open(a.md, "w").write(out)
    if a.json:
        json.dump({"tag": a.tag, "arms": arms, "pairs": pairs, "rows": per}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
