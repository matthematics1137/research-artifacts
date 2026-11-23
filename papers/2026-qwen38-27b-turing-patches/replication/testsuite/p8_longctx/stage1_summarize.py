#!/usr/bin/env python3
"""P8 long context, stage 1: judge the journal against the registration.

Reads stage1_run.py's journal.jsonl and writes results.json: each step's peak, pass or fail and P1 band check, each
format's reach, and the verdicts on P1-P3, K1 and B1 (research/prereg/2026-11-01-p8-longctx-stage1.md). A step is
"completed" when its prefill and decode finished; only completed steps enter P1. A format's reach is its longest
passing step; a format the stage never reached (budget, harness error, thermal stop) has no reach and no verdict.

  python stage1_summarize.py JOURNAL [--out results.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_spec as S  # noqa: E402


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def judge(rows):
    results = {r["step"]: r for r in rows if r.get("kind") == "result" and r.get("judged")}
    steps, formats = [], {}
    for fmt, ladder in S.LADDER:
        run, reach, stopped_at = [], None, None
        for length in ladder:
            r = results.get(S.step_id(fmt, length))
            if r is None:
                break
            lo, hi, predicted = S.PROJECTED[(fmt, length)]
            completed = r.get("outcome") == "completed"
            peak = r.get("peak_mib")
            band = None
            if completed and peak is not None:
                band = lo + S.P1_BAND[0] <= peak <= hi + S.P1_BAND[1]
            row = {"step": r["step"], "format": fmt, "length": length, "verdict": r["verdict"],
                   "outcome": r.get("outcome"), "reason": r.get("reason"), "peak_mib": peak,
                   "baseline_mib": r.get("baseline_mib"), "projected": [lo, hi], "predicted": predicted,
                   "prediction_right": (r["verdict"] == predicted) if r["verdict"] in ("pass", "fail") else None,
                   "p1_in_band": band, "p1_offset_mib": None if peak is None or not completed else
                   (peak - hi if peak > hi else peak - lo if peak < lo else 0),
                   "prompt_tokens": r.get("prompt_tokens"), "prompt_tokens_expected": r.get("prompt_tokens_expected"),
                   "prefill_s": r.get("prompt_time_s"), "prefill_tok_s": r.get("prompt_tok_s"),
                   "decode_tokens": r.get("completion_tokens"), "decode_tok_s": r.get("completion_tok_s"),
                   "load_s": r.get("load_s"), "wall_s": r.get("wall_s"), "peak_rss_mib": r.get("peak_rss_mib"),
                   "max_temp_c": r.get("max_temp_c")}
            steps.append(row)
            run.append(row)
            if r["verdict"] == "pass":
                reach = length
            elif r["verdict"] == "fail":
                stopped_at = length
                break
            else:
                break
        complete = stopped_at is not None or (len(run) == len(ladder) and all(x["verdict"] == "pass" for x in run))
        bands = [x["p1_in_band"] for x in run if x["p1_in_band"] is not None]
        formats[fmt] = {"steps_run": [x["step"] for x in run], "reach": reach, "first_failure": stopped_at,
                        "ladder_finished": complete,
                        "p1": None if not bands else ("holds" if all(bands) else "refuted")}
    f88, f44 = formats["8,8"], formats["4,4"]
    r88_64 = results.get(S.step_id("8,8", 64 * S.K))
    p1_parts = [f["p1"] for f in formats.values() if f["p1"] is not None]
    verdicts = {
        "P1": None if not p1_parts else ("holds" if all(p == "holds" for p in p1_parts) else
                                         "refuted for " + ", ".join(k for k, f in formats.items() if f["p1"] == "refuted")),
        "P2": None if r88_64 is None or r88_64["verdict"] not in ("pass", "fail") else
        ("holds" if r88_64["verdict"] == "pass" else "refuted"),
        # decided as soon as the 4,4 ladder passes or fails the step that settles it
        "P3": "holds" if (f44["reach"] or 0) >= 128 * S.K else
        "refuted" if f44["first_failure"] is not None and f44["first_failure"] <= 128 * S.K else None,
        "K1": "not triggered" if (f44["reach"] or 0) >= 96 * S.K else
        "triggered" if f44["first_failure"] is not None and f44["first_failure"] <= 96 * S.K else None,
        "B1": None if r88_64 is None or r88_64["verdict"] not in ("pass", "fail") else
        ("triggered" if r88_64["verdict"] == "fail" else "not triggered"),
    }
    stops = [r for r in rows if r.get("kind") in ("budget_stop", "stage_stop", "stage_done")]
    interrupted = [r for r in rows if r.get("kind") == "interrupted"]
    check = next((r for r in rows if r.get("kind") == "result" and not r.get("judged", True)), None)
    used = sum(r.get("wall_s") or 0 for r in rows if r.get("kind") == "result") + \
        sum(r.get("span_s") or 0 for r in interrupted)
    return {"steps": steps, "formats": formats, "verdicts": verdicts, "gpu_s": round(used, 1),
            "budget_s": S.BUDGET_S, "end": stops[-1] if stops else None,
            "interrupted": [{"step": r["step"], "span_s": r.get("span_s")} for r in interrupted],
            "check": None if check is None else {k: check.get(k) for k in (
                "verdict", "outcome", "reason", "peak_mib", "baseline_mib", "prompt_tokens", "prompt_tokens_expected",
                "load_s", "wall_s", "text")}}


def fmt_num(x, nd=0):
    if x is None:
        return "-"
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x)


def markdown(res):
    """The report's tables: steps, formats, verdicts."""
    out = ["| Step | Prompt tokens (server / built) | Load s | Prefill s | Prefill tok/s | Decode tok/s | "
           "Peak MiB | Projected MiB | Off band by | Verdict | Predicted |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["steps"]:
        off = s["p1_offset_mib"]
        band = "-" if s["p1_in_band"] is None else ("in band" if s["p1_in_band"] else f"{off:+,} MiB outside the range")
        if s["p1_in_band"] and off:
            band = f"in band ({off:+,})"
        verdict = s["verdict"] + ("" if s["verdict"] == "pass" or not s["reason"] else f" ({s['reason']})")
        out.append(f"| `{s['format']}` {s['length'] // S.K}K | {fmt_num(s['prompt_tokens'])} / "
                   f"{fmt_num(s['prompt_tokens_expected'])} | {fmt_num(s['load_s'], 1)} | {fmt_num(s['prefill_s'], 1)} | "
                   f"{fmt_num(s['prefill_tok_s'])} | {fmt_num(s['decode_tok_s'], 1)} | {fmt_num(s['peak_mib'])} | "
                   f"{s['projected'][0]:,}-{s['projected'][1]:,} | {band} | {verdict} | {s['predicted']} |")
    out += ["", "| Format | Steps run | Reach | First failure | P1 |", "|---|---|---|---|---|"]
    for fmt, f in res["formats"].items():
        out.append(f"| `{fmt}` | {', '.join(f['steps_run']) or 'none'} | "
                   f"{'-' if f['reach'] is None else str(f['reach'] // S.K) + 'K'} | "
                   f"{'-' if f['first_failure'] is None else str(f['first_failure'] // S.K) + 'K'} | {f['p1'] or '-'} |")
    out += ["", "| Claim | Verdict |", "|---|---|"]
    out += [f"| {k} | {v or 'not decided'} |" for k, v in res["verdicts"].items()]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal")
    ap.add_argument("--out")
    ap.add_argument("--markdown", help="also write the report tables here")
    args = ap.parse_args()
    res = judge(load(args.journal))
    text = json.dumps(res, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(markdown(res))
    print(text)


if __name__ == "__main__":
    main()
