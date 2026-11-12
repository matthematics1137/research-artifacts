#!/usr/bin/env python3
"""Stages 2-3 Amendment 2 checks, before any multivalue item is built for a GPU step:
  1. P6's scorer requires every value of a multivalue item (score_longctx: all required strings, case-insensitive).
  2. A full answer listing 48 values fits in 512 output tokens with the model's tokenizer, in the formats a model
     plausibly uses (comma list, one per line, a bulleted list, a sentence with the key).
Builds three k = 48 multivalue items with the test seed (the wrapper, as for real items) at 8K, which changes no
value: draw_slots draws the values before any length is involved. CPU only; prints a JSON verdict.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I  # noqa: E402
import stage2_stats as T  # noqa: E402


def main():
    pool = I.pool()
    out = {"scorer": [], "tokens": []}
    for idx in (0, 1, 2):
        it = I.build_one(pool, "test", "multivalue", 48, idx, 8192)
        vals, key = it["answers"], it["question"].split("for ")[1].split(" mentioned")[0]
        full = ", ".join(vals)
        out["scorer"].append({"uid": it["uid"], "n_values": len(vals),
                              "all_listed": T.score(full, it), "one_missing": T.score(", ".join(vals[:-1]), it),
                              "reordered": T.score(", ".join(reversed(vals)), it)})
        formats = {
            "comma": full,
            "lines": "\n".join(vals),
            "bullets": "\n".join("- " + v for v in vals),
            "numbered": "\n".join(f"{i + 1}. {v}" for i, v in enumerate(vals)),
            "sentence": f"The special magic numbers for {key} mentioned in the provided text are: {full}.",
            # the two list styles the 27B used on P6's multivalue items (testsuite/paper6/work/journals)
            "p6_numbered": f"Based on the text provided, the special magic numbers for {key} are:\n\n" +
                           "\n".join(f"{i + 1}.  {v}" for i, v in enumerate(vals)),
            "p6_bullets": f"Based on the text provided, the special magic numbers for {key} are:\n\n" +
                          "\n".join(f"*   {v}" for v in vals),
        }
        out["tokens"].append({"uid": it["uid"], **{name: I.G.ntok(text) + 1 for name, text in formats.items()}})
    worst = max(max(v for k, v in row.items() if k != "uid") for row in out["tokens"])
    out["verdict"] = {"scorer_requires_all": all(r["all_listed"] == 1 and r["one_missing"] == 0 and r["reordered"] == 1
                                                 for r in out["scorer"]),
                      "worst_tokens_with_end_token": worst, "fits_512": worst <= 512}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
