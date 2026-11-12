#!/usr/bin/env python3
"""Paper 8 W9 helper: median int8 GEMV time of three matrix groups, from census SQLite exports (stdlib only).

Groups are read off the kernel order of each layer:
- MLP down is the int8 launch right after `act_mul_kernel_h`;
- MLP gate + up are the two int8 launches right before it;
- GDN qkv + z are the two int8 launches right before `gdn_ba_gemv_rows_kernel`.
Medians, because single launches on this desktop GPU are sometimes stretched by display time-slicing.
  python3 census_shapes.py CAPTURE.sqlite...
"""
import sqlite3
import statistics as st
import sys

for db in sys.argv[1:]:
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))

    def fam(n):
        nm = names.get(n, '') or '?'
        if 'exl3_gemv_int8_sq_kernel' in nm:
            return 'sq'
        parts = nm.split('(')[0].split('<')[0].split()
        return parts[-1].split('::')[-1] if parts else '?'
    seq = [(fam(n), (e - s) / 1000) for s, e, n in con.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    down, gdn_pair, gate_up = [], [], []
    for i, (k, d) in enumerate(seq):
        if k == 'sq' and i > 0 and seq[i - 1][0] == 'act_mul_kernel_h':
            down.append(d)
        if k == 'gdn_ba_gemv_rows_kernel' and i >= 2 and seq[i - 1][0] == 'sq' and seq[i - 2][0] == 'sq':
            gdn_pair.append(seq[i - 2][1] + seq[i - 1][1])
        if k == 'act_mul_kernel_h' and i >= 2 and seq[i - 1][0] == 'sq' and seq[i - 2][0] == 'sq':
            gate_up.append(seq[i - 2][1] + seq[i - 1][1])
    print(db.split('/')[-2][:18], db.split('.')[-2], 'down median %.1f us (n=%d) | GDN qkv+z median %.1f us | gate+up median %.1f us' % (
        st.median(down), len(down), st.median(gdn_pair), st.median(gate_up)))
