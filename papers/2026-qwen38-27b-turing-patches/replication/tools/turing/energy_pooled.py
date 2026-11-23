#!/usr/bin/env python3
"""Paper 8: pooled energy per token from energy.py's per-row JSON and the ab_speed.sh rows (stdlib only).

energy.py reports each arm's median over rows of tokens per joule (and of tok/s). Under MTP the rows run at very
different rates (prose drafts worst), so a median of per-row ratios overstates the arm, the same flaw the 10:56 EDT
2026-10-31 correction fixed in ab_report.py. This script pools instead: total new tokens over total joules (each
row's window mean power times its decode_wall_s, as energy.py computes them), and total tokens over total decode
time. Pairs are compared on the pooled figures. Written 2026-10-31 after Volume I.5's chapter 15 draft found the flaw.
  python3 energy_pooled.py ROWS.jsonl ENERGY.json --tag TAG [--pair A:B ...] [--md OUT.md]
"""
import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("energy")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pair", action="append", default=[], help="A:B, B compared with A")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    pat = re.compile(rf"^{re.escape(a.tag)}-(.+)-r(\d+)$")
    rows = {}
    for line in open(a.rows):
        r = json.loads(line)
        m = pat.match(r.get("tag", ""))
        if m:
            rows[(m.group(1), int(m.group(2)), r["task"])] = r
    E = json.load(open(a.energy))
    arms = {}
    for arm, rs in E["rows"].items():
        tok = wall = joules = 0.0
        med = E.get("arms", {}).get(arm, {}).get("tok_per_j")
        for x in rs:
            if not x.get("samples"):
                continue                                   # energy.py's short windows carry no power reading
            r = rows[(arm, x["round"], x["task"])]
            tok += r["new_tokens"]
            wall += r["decode_wall_s"]
            joules += x["watts"] * r["decode_wall_s"]
        arms[arm] = {"tps": tok / wall, "watts": joules / wall, "tok_per_j": tok / joules, "j_per_tok": joules / tok,
                     "median_tok_per_j": med}
    lines = [f"# Pooled energy per token, `{a.tag}`", "",
             "Total tokens over total joules (rows with power samples). The median column is energy.py's figure.", "",
             "| arm | pooled tok/s | time-weighted W | pooled tokens/J | J/token | energy.py median tokens/J |",
             "|---|---|---|---|---|---|"]
    for arm in sorted(arms):
        x = arms[arm]
        med = f"{x['median_tok_per_j']:.4f}" if x["median_tok_per_j"] is not None else "-"
        lines.append(f"| {arm} | {x['tps']:.2f} | {x['watts']:.1f} | {x['tok_per_j']:.4f} | {x['j_per_tok']:.2f} | {med} |")
    if a.pair:
        lines += ["", "| pair | speed ratio | tokens/J ratio | energy vs speed |", "|---|---|---|---|"]
        for p in a.pair:
            A, B = p.split(":")
            sr, er = arms[B]["tps"] / arms[A]["tps"], arms[B]["tok_per_j"] / arms[A]["tok_per_j"]
            lines.append(f"| {A} -> {B} | {sr:.3f} | {er:.3f} | {er / sr - 1:+.1%} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if a.md:
        open(a.md, "w").write(text)


if __name__ == "__main__":
    main()
