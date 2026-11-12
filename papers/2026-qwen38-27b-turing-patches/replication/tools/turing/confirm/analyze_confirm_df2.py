#!/usr/bin/env python3
"""Addendum analysis for the DFlash2 follow-up session (Amendment 2 of
research/prereg/2026-11-04-p8-confirmation-speed.md), written before any of its rows exist.

analyze_confirm.py runs unchanged on the session first. It writes results.json: F2's anchors, w13 undrafted against
NInferno undrafted at 0, and the "fastest" wording, whose drafted rivals now include nif-xxs-df2 (its names hold
"df"). This script imports its load() and versus() unchanged, after checking the registered SHA-256, and adds the
pairs that the registered script has no names for:
- over_nif_df2_0 and over_nif_df2_28k: w13 with MTP against NInferno with DFlash2;
- nif_df2_gain_0: NInferno with DFlash2 against NInferno undrafted;
- w13_mtp_gain_0: w13 with MTP against w13 undrafted.

  python3 analyze_confirm_df2.py ROWS.jsonl > report-df2.md
Writes results-df2.json beside ROWS.jsonl. Stdlib only.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_confirm as AC  # noqa: E402

REGISTERED = "d713742661b4b82eb5332135ff6f7c2c82fe7886a09057b44abb0f0919ac4221"  # analyze_confirm.py
PAIRS = {  # name: (a, b)
    "over_nif_df2_0": ("w13-165-mtp-0", "nif-xxs-df2-0"),
    "over_nif_df2_28k": ("w13-165-mtp-28k", "nif-xxs-df2-28k"),
    "nif_df2_gain_0": ("nif-xxs-df2-0", "nif-xxs-0"),
    "w13_mtp_gain_0": ("w13-165-mtp-0", "w13-165-nd-0"),
}


def analyze(rows_path):
    s = AC.load(rows_path)
    res = {"pooled": s, "comparisons": {name: AC.versus(s, a, b) for name, (a, b) in PAIRS.items()}}
    with open(os.path.join(os.path.dirname(os.path.abspath(rows_path)), "results-df2.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows")
    args = ap.parse_args()
    with open(os.path.join(HERE, "analyze_confirm.py"), "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != REGISTERED:
            raise SystemExit("refusing: analyze_confirm.py differs from the registration")
    res = analyze(args.rows)
    print("# P8 confirmation, DFlash2 follow-up: the addendum pairs\n")
    for name, v in res["comparisons"].items():
        if v is None:  # either arm has no rows: dropped or cut
            print(f"- {name}: not computed ({' or '.join(PAIRS[name])} has no rows)")
        else:
            print(f"- {name}: {v['a']} / {v['b']} = {v['ratio']:.3f} ({v['pooled_pct']:+.1f}%), "
                  f"{v['wins']} of {v['rounds']} rounds faster; per round "
                  f"{', '.join(f'{x:+.1f}%' for x in v['per_round_pct'])}")


if __name__ == "__main__":
    main()
