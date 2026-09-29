#!/usr/bin/env python3
"""H32: analyze EXL3_SPEC_MEASURE rows. usage: spec_meas_analyze.py <rows.jsonl> [<label>:<start_line>:<end_line> ...]
Reports, per range: positions, actual acceptance, mean p(x*) (= current expected acceptance), mean q(x*),
mean sum(min(p,q)) (= acceptance with a sampled draft + rejection sampling), and the predicted gain."""
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ranges = [("all", 0, len(rows))] + [(a.split(":")[0], int(a.split(":")[1]), int(a.split(":")[2])) for a in sys.argv[2:]]
for name, lo, hi in ranges:
    r = rows[lo:hi]
    if not r: continue
    m = lambda k: statistics.mean(x[k] for x in r)
    by_i = {}
    for x in r: by_i.setdefault(x["i"], []).append(x)
    print(f"{name}: n={len(r)} accepted={m('acc'):.3f} p*={m('p_star'):.3f} q*={m('q_star'):.3f} sum_min={m('sum_min'):.3f} "
          f"pmax={m('p_max'):.3f} entropy={m('ent'):.2f}  predicted gain (sum_min - p*) = {m('sum_min') - m('p_star'):+.3f}  "
          f"by position: {[(i, round(statistics.mean(x['p_star'] for x in v), 2), round(statistics.mean(x['sum_min'] for x in v), 2)) for i, v in sorted(by_i.items())]}")
    hi_ent = [x for x in r if x["ent"] > 1.5]
    if hi_ent: print(f"    high-entropy positions (H>1.5 nats): n={len(hi_ent)} p*={statistics.mean(x['p_star'] for x in hi_ent):.3f} sum_min={statistics.mean(x['sum_min'] for x in hi_ent):.3f}")
