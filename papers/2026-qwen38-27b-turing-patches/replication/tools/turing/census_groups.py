#!/usr/bin/env python3
"""Kernel time by group for the decode censuses, stock wheel against w13 (P8, claims-checks GAP 10).

Reads the census reports census_report.py wrote (testsuite/results/turing/census/<tag>/report.json) and groups their
kernel families the way the four-resources priced graph does (tools/four_resources/test2_priced_graph.py: matrix
products, attention, GDN recurrent state, small kernels), with the two families the w13 build adds placed in the
roles they replace: exl3_gemv_sk_kernel with the matrix products, fd75d_split_kernel with attention. Read-only;
writes the table as JSON and Markdown.

  python3 census_groups.py OUT_PREFIX     (writes OUT_PREFIX.json and OUT_PREFIX.md)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "testsuite", "results", "turing", "census")
CAPTURES = [  # label, directory, section: the same census.py prompts and census_report.py, cache 8,8
    ("stock 1.6.5, undrafted, fill 0", "w2", "nodraft"),
    ("stock 1.6.5, MTP, fill 0", "w2", "mtp"),
    ("w13 mode 2, MTP, fill 0", "d0-w13-mtp-sk2", "mtp"),
    ("w13 mode 2, MTP, 28K", "d28-w13-mtp-sk2", "mtp"),
]
ATTN = {"_paged_attn_decode_split_kernel", "_paged_attn_decode_combine_kernel", "fd75d_split_kernel"}
GDN = {"cuda_recurrent_gated_delta_rule_kernel_128"}
GROUPS = ("matrix products", "attention", "GDN recurrent state", "small kernels")


def group(family):
    if family.startswith("exl3_"):
        return "matrix products"
    if family in ATTN:
        return "attention"
    if family in GDN:
        return "GDN recurrent state"
    return "small kernels"


def capture(directory, section):
    r = json.load(open(os.path.join(CENSUS, directory, "report.json")))[section]
    us = {g: 0.0 for g in GROUPS}
    for f in r["families"]:
        us[group(f["family"])] += f["us_per_token"]
    total = sum(us.values())
    tasks = r["tasks"]
    return {"directory": directory, "section": section,
            "ms_per_token_mean": sum(t["ms_per_token"] for t in tasks) / len(tasks),
            "ms_per_token_by_task": {t["task"]: t["ms_per_token"] for t in tasks},
            "kernel_us_per_token": total, "group_us_per_token": us, "group_share": {g: us[g] / total for g in GROUPS},
            "families": sorted({f["family"] for f in r["families"]})}


def main():
    out = {label: capture(d, s) for label, d, s in CAPTURES}
    prefix = sys.argv[1]
    with open(prefix + ".json", "w") as f:
        json.dump({"source": "tools/turing/census_groups.py", "captures": out}, f, indent=1)
    lines = ["# Decode kernel time by group: stock wheel against w13 (from the census reports)", "",
             "| capture | ms/token | kernel µs/token | " + " | ".join(GROUPS) + " |",
             "|---|---|---|" + "---|" * len(GROUPS)]
    for label, c in out.items():
        cells = [f"{c['group_us_per_token'][g]:.0f} ({100 * c['group_share'][g]:.1f}%)" for g in GROUPS]
        lines.append(f"| {label} | {c['ms_per_token_mean']:.2f} | {c['kernel_us_per_token']:.0f} | " + " | ".join(cells) + " |")
    lines += ["", "ms/token: the mean over the code, math and prose tasks of decode wall time per decoded token under "
              "Nsight Systems. Group cells: kernel µs per decoded token (share of kernel time)."]
    with open(prefix + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
