#!/usr/bin/env python3
"""P9 phase 1, M4's report (research/prereg/2026-11-05-p9-phase1.md): reads the raw metrics that p9_ncu.sh exported
(raw.csv) and prints, per profiled launch (kernel, in launch order: mlp_gate then gdn_qkv, K = 1 then 2, full then
loop only), the issue-active fraction, the warp stall reasons per issued instruction (largest first), the DRAM and
L2 throughput against peak and the achieved occupancy. Writes report.md beside raw.csv.
  python3 p9_ncu_report.py testsuite/results/ada/p9/ncu-STAMP
"""
import csv
import os
import sys

WANT = [("issue active %", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
        ("SM throughput %", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("DRAM throughput %", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("L2 throughput %", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("L1/TEX throughput %", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
        ("achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("duration (us)", "gpu__time_duration.sum")]
STALL = "smsp__average_warps_issue_stalled_"
STALL_END = "_per_issue_active.ratio"


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def main():
    d = sys.argv[1]
    rows = list(csv.reader(open(os.path.join(d, "raw.csv"))))
    head, units, data = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(head)}
    out = [f"# P9 M4: Nsight Compute counters ({d})", ""]
    for r in data:
        name = r[col.get("Kernel Name", 0)] if "Kernel Name" in col else r[4]
        out.append(f"## launch {r[col['ID']] if 'ID' in col else '?'}: {name[:90]}")
        for label, key in WANT:
            if key in col:
                v = num(r[col[key]])
                scale = 1e-3 if key == "gpu__time_duration.sum" and units[col[key]] == "nsecond" else 1
                out.append(f"- {label}: {v * scale:.2f}" if v is not None else f"- {label}: {r[col[key]]}")
        stalls = sorted(((h[len(STALL):-len(STALL_END)], num(r[i])) for h, i in col.items()
                         if h.startswith(STALL) and h.endswith(STALL_END) and num(r[i]) is not None),
                        key=lambda x: -x[1])
        if stalls:
            total = sum(v for _, v in stalls)
            out.append("- stalls per issued instruction: " + ", ".join(
                f"{k} {v:.2f} ({100 * v / total:.0f}%)" for k, v in stalls[:6] if v > 0))
        out.append("")
    text = "\n".join(out)
    open(os.path.join(d, "report.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
