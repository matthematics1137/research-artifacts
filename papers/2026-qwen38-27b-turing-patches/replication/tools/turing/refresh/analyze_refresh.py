#!/usr/bin/env python3
"""P8 refresh comparisons: complete same-phase pairs only; no cross-session pooling."""
import argparse
import json
from pathlib import Path
import re


def pooled(rows):
    return sum(r["new_tokens"] for r in rows) / sum(r["decode_wall_s"] for r in rows)


def cells(rows, condition, tasks):
    found = {}
    for row in rows:
        tag = row.get("tag", "")
        match = re.fullmatch(r"confirm-" + re.escape(condition) + r"-r([123])", tag)
        if not match:
            continue
        key = (int(match[1]), row.get("task"))
        if key in found:
            raise ValueError("duplicate cell: " + str(key) + " " + condition)
        if (key[1] not in tasks or not isinstance(row.get("new_tokens"), int) or row["new_tokens"] <= 0 or
                row.get("decode_wall_s", 0) <= 0):
            raise ValueError("invalid measured row: " + condition)
        found[key] = row
    expected = {(rnd, task) for rnd in (1, 2, 3) for task in tasks}
    return found, expected - set(found)


def analyze(plan, phases, out):
    out = Path(out)
    rows = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines()] if (out / "rows.jsonl").exists() else []
    rungs = json.loads((out / "rungs.json").read_text()) if (out / "rungs.json").exists() else {}
    conditions = {c["id"]: c for phase in phases for c in phase.get("conditions", [])}
    results = []
    for pair in plan["comparisons"]:
        a, missing_a = cells(rows, pair["anchor"], plan["tasks"])
        b, missing_b = cells(rows, pair["competitor"], plan["tasks"])
        result = dict(pair, complete=not missing_a and not missing_b,
                      missing_anchor=sorted(missing_a), missing_competitor=sorted(missing_b))
        cfg = conditions[pair["competitor"]]["config"]
        res = rungs.get(cfg)
        result["fit"] = res
        if result["complete"]:
            by_round = []
            for rnd in (1, 2, 3):
                ar = [a[(rnd, t)] for t in plan["tasks"]]
                br = [b[(rnd, t)] for t in plan["tasks"]]
                by_round.append(dict(round=rnd, w13=pooled(ar), competitor=pooled(br), ratio=pooled(ar) / pooled(br)))
            result.update(w13=pooled(list(a.values())), competitor=pooled(list(b.values())), rounds=by_round,
                          w13_wins_every_round=all(r["ratio"] > 1 for r in by_round),
                          per_prompt={t:dict(w13=pooled([a[(n,t)] for n in (1,2,3)]),
                                             competitor=pooled([b[(n,t)] for n in (1,2,3)])) for t in plan["tasks"]})
            competitor = conditions[pair["competitor"]]
            eligible = (res or {}).get("full_offload", False) if competitor["engine"] == "llama" else True
            result["eligible_full_residency_comparison"] = eligible
            result["can_keep_current_pooled_lead"] = eligible and result["w13_wins_every_round"]
        else:
            result["can_keep_current_pooled_lead"] = False
        results.append(result)
    return dict(schema="p8-refresh-analysis-v1", comparisons=results,
                complete=all(r["complete"] for r in results),
                can_keep_all_tested_leads=all(r["can_keep_current_pooled_lead"] for r in results),
                scope="old pinned artifacts; sampled three-prompt mixture; no UD3.1 or accuracy claim")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from run_refresh import load
    plan, phases = load()
    print(json.dumps(analyze(plan, phases, args.out), indent=2))
