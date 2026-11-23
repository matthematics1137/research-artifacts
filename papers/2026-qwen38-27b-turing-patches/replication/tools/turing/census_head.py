#!/usr/bin/env python3
"""Paper 8 W9 helper: the output head's int8 GEMV time per decoded token, from census SQLite exports (stdlib only).

The head is the last `exl3_gemv_int8_sq_kernel` launch before each token's sampler (`unrolled_elementwise_kernel`,
once per token). Prints, per capture, the head's median and mean duration and the median of all other int8 GEMV time
per token. Medians, because single launches on this desktop GPU are sometimes stretched by display time-slicing.
  python3 census_head.py CAPTURE.sqlite...
"""
import sqlite3
import statistics as st
import sys

for db in sys.argv[1:]:
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = con.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    heads, last_sq, tok_sq, per_tok = [], None, [], []
    for s, e, n in rows:
        nm = names.get(n, '')
        if 'exl3_gemv_int8_sq_kernel' in nm:
            last_sq = (e - s) / 1000
            tok_sq.append(last_sq)
        elif 'unrolled_elementwise_kernel' in nm:   # the sampler: one per token
            if last_sq is not None:
                heads.append(last_sq)
                per_tok.append(sum(tok_sq[:-1]))
            last_sq, tok_sq = None, []
    print(db.split('/')[-2][:18], db.split('.')[-2], 'tokens %d | head median %.0f us mean %.0f | other int8 per token median %.0f us' % (
        len(heads), st.median(heads), st.mean(heads), st.median(per_tok)))
