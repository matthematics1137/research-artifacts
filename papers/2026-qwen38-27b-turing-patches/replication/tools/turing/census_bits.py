#!/usr/bin/env python3
"""Paper 8 W12 scouting: time per int8 GEMV launch by matrix role and bit width, from census SQLite exports (stdlib).

Roles come from each layer's kernel order (as census_shapes.py): MLP down is the int8 launch right after
`act_mul_kernel_h`, MLP gate and up the two right before it, GDN out_proj the int8 launch right after
`gated_rms_norm_kernel`. The bit width K (and HALF) comes from the kernel's template arguments,
`exl3_gemv_int8_sq_kernel<K, M, c_fp32, residual, HALF>`. For each (role, K) it prints the median duration, the
weight bytes (k x n x K / 8 for the role's shape) and the implied GB/s, and the implied weights per microsecond.
If time tracks weights rather than bytes across K, the kernel is bound by per-weight decode arithmetic.
  python3 census_bits.py CAPTURE.sqlite...
"""
import re
import sqlite3
import statistics as st
import sys
from collections import defaultdict

SHAPES = {"mlp_gate_up": (5120, 17408), "mlp_down": (17408, 5120), "gdn_out": (6144, 5120),
          "gdn_qkv": (5120, 10240), "gdn_z": (5120, 6144), "attn_q": (5120, 12288), "attn_kv": (5120, 1024),
          "attn_o": (6144, 5120), "head": (5120, 248320)}
TPL = re.compile(r"exl3_gemv_int8_sq_kernel<(?:\(int\))?(\d+), (?:\(int\))?(\d+), (?:\(bool\))?(\w+), (?:\(bool\))?(\w+), (?:\(bool\))?(\w+)>")


def main():
    times = defaultdict(list)
    for db in sys.argv[1:]:
        con = sqlite3.connect(db)
        names = dict(con.execute("SELECT id, value FROM StringIds"))
        seq = []
        for s, e, n in con.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
            nm = names.get(n, "") or ""
            m = TPL.search(nm)
            if m:
                k, mm, half = int(m.group(1)), int(m.group(2)), m.group(5) in ("true", "1")
                seq.append(("sq", (k, mm, half), (e - s) / 1000))
            else:
                parts = nm.split("(")[0].split("<")[0].split()
                seq.append((parts[-1].split("::")[-1] if parts else "?", None, (e - s) / 1000))
        for i, (fam, tpl, d) in enumerate(seq):
            if fam == "sq" and tpl[1] == 1:
                prev = seq[i - 1][0] if i > 0 else ""
                if prev == "act_mul_kernel_h":
                    times[("mlp_down", tpl)].append(d)
                elif prev == "gated_rms_norm_kernel":
                    times[("gdn_out", tpl)].append(d)
            if fam == "gdn_ba_gemv_rows_kernel" and i >= 2 and seq[i - 1][0] == "sq" and seq[i - 2][0] == "sq":
                times[("gdn_qkv", seq[i - 2][1])].append(seq[i - 2][2])     # qkv then z, the order of the two
                times[("gdn_z", seq[i - 1][1])].append(seq[i - 1][2])
            if fam == "deinterleave_qg_kernel" and i >= 1 and seq[i - 1][0] == "sq":
                times[("attn_q", seq[i - 1][1])].append(seq[i - 1][2])
                for j in (i + 1, i + 2):
                    if j < len(seq) and seq[j][0] == "sq":
                        times[("attn_kv", seq[j][1])].append(seq[j][2])
            if fam == "mul_sigmoid_kernel_h" and i + 1 < len(seq) and seq[i + 1][0] == "sq":
                times[("attn_o", seq[i + 1][1])].append(seq[i + 1][2])
            if fam == "unrolled_elementwise_kernel":          # the sampler: the head is the last int8 launch before it
                for j in range(i - 1, max(i - 8, -1), -1):
                    if seq[j][0] == "sq":
                        times[("head", seq[j][1])].append(seq[j][2])
                        break
            if fam == "act_mul_kernel_h" and i >= 2:
                for j in (i - 1, i - 2):
                    if seq[j][0] == "sq" and seq[j][1][1] == 1:
                        times[("mlp_gate_up", seq[j][1])].append(seq[j][2])
    print("| role | K | launches | median us | weight MB | GB/s | weights per us (M) |")
    print("|---|---|---|---|---|---|---|")
    for (role, (k, mm, half)), v in sorted(times.items()):
        kk, nn = SHAPES[role]
        bits = k + (0.5 if half else 0.0)
        mb = kk * nn * bits / 8 / 1e6
        med = st.median(v)
        print(f"| {role} | {bits:g} | {len(v)} | {med:.1f} | {mb:.2f} | {mb / med * 1e3:.0f} | {kk * nn / med / 1e6:.2f} |")
    fit(times)


def fit(times):
    """Least squares t = a + b * weights per K over the roles (median per role), launch-weighted roles excluded from
    nothing: one point per (role, K) with at least 50 launches"""
    by_k = defaultdict(list)
    for (role, (k, mm, half)), v in times.items():
        if len(v) >= 50 and role != "head":
            kk, nn = SHAPES[role]
            by_k[k + (0.5 if half else 0)].append((kk * nn / 1e6, st.median(v), role))
    print()
    print("| K | points | fixed us (a) | us per M weights (b) | roles |")
    print("|---|---|---|---|---|")
    for k, pts in sorted(by_k.items()):
        if len(pts) < 2:
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        a = my - b * mx
        print(f"| {k:g} | {n} | {a:.1f} | {b:.3f} | {', '.join(sorted(p[2] for p in pts))} |")


if __name__ == "__main__":
    main()
