"""Allowlisted, unchanged presentation definitions; no private input loading."""
# Full source SHA256: babc584e551be92392b7fbeb3fa09ad2b6b138f361b0f9cdad295b05c5893822
from types import SimpleNamespace
from draft_support import integer, number, probability, pvalue, table, tex
def require(condition, message):
    if not condition:
        raise ValueError(message)

D = SimpleNamespace(require=require)

MODELS = {"cross-model": "qwen3-8b", "locomo": "qwen38-exl3"}


ARMS = ("T", "S", "C", "M")


TIES = ("S", "M", "C")


def selected_policy(primary):
    scores = {}
    for arm in TIES:
        cell = primary["cells"][arm + "@4096"]
        integer(cell["n"], "selection denominator", 500, 500)
        scores[arm] = integer(cell["correct"], "selection correct", 0, 500)
    return max(TIES, key=scores.get)


def validate(leg, a, primary):
    n = 500 if leg == "cross-model" else 200
    arms = ARMS if leg == "cross-model" else ("T", selected_policy(primary))
    D.require(a["schema"] == "paper5-followup-analysis-v1" and a["leg"] == leg and a["model"] == MODELS[leg],
              "wrong follow-up analysis identity")
    D.require(a["confirmatory"] is False and a["budget"] == 4096, "follow-up cannot become a confirmatory test")
    D.require(a["publication"] == ("aggregates-only" if leg == "locomo" else "curated-item-level"), "wrong release scope")
    D.require(set(a["cells"]) == set(arms) and set(a["contrasts_vs_T"]) == set(arms) - {"T"}, "wrong follow-up arms")
    for arm, cell in a["cells"].items():
        integer(cell["n"], "follow-up denominator", n, n)
        correct = integer(cell["correct"], "follow-up correct", 0, n)
        p = probability(cell["accuracy"])
        D.require(abs(p - correct / n) < 1e-12, "follow-up counts/accuracy differ")
        lo, hi = [probability(x) for x in cell["wilson95"]]
        D.require(lo <= p <= hi, "invalid follow-up interval")
        for key in ("failed_generations", "judge_failures", "unterminated_thinking"):
            integer(cell[key], key, 0, n)
    for arm, c in a["contrasts_vs_T"].items():
        integer(c["n"], "paired follow-up n", n, n)
        counts = {k: integer(c[k], k, 0, n) for k in ("both", "arm_only", "reference_only", "neither")}
        D.require(sum(counts.values()) == n, "invalid paired table")
        D.require(counts["both"] + counts["arm_only"] == a["cells"][arm]["correct"]
                  and counts["both"] + counts["reference_only"] == a["cells"]["T"]["correct"], "paired counts differ from cells")
        diff = number(c["difference"], "follow-up difference", -1, 1)
        D.require(abs(diff - (counts["arm_only"] - counts["reference_only"]) / n) < 1e-12, "wrong paired difference")
        lo, hi = [number(x, "paired interval", -1, 1) for x in c["newcombe95"]]
        D.require(lo <= diff <= hi, "invalid follow-up paired interval")
        probability(c["mcnemar_p_unadjusted"])
        D.require("holm_p" not in c, "follow-up must not inherit the primary Holm family")
    if leg == "locomo":
        D.require(set(a["token_f1"]) == set(arms) and set(a["token_f1_differences_vs_T"]) == {arms[1]}, "missing F1 comparison")
        records = [*a["token_f1"].values(), *a["token_f1_differences_vs_T"].values()]
        for row in records:
            integer(row["n"], "F1 denominator", 200, 200)
            integer(row["conversations"], "conversation count", 2, 200)
            D.require(row["resampling_unit"] == "conversation" and row["draws"] == 10000 and row["seed"] == 20260920,
                      "wrong F1 bootstrap rule")
            number(row["mean"], "F1 mean/difference", -1, 1)
            lo, hi = [number(x, "F1 percentile interval", -1, 1) for x in row["percentile95"]]
            D.require(lo <= hi, "invalid F1 interval")
        D.require(len({r["conversations"] for r in records}) == 1, "F1 conversation inventories differ")
        for row in a["token_f1"].values():
            probability(row["mean"], "mean token F1")
    return arms


def leaders(cells):
    best = max(cells[arm]["correct"] for arm in ARMS)
    return " = ".join(arm for arm in ARMS if cells[arm]["correct"] == best)


def sections(primary, followups):
    D.require(set(followups) == set(MODELS), "both follow-up result sets are required")
    cross, locomo = [followups[name]["analysis"] for name in ("cross-model", "locomo")]
    validate("cross-model", cross, primary)
    _, selected = validate("locomo", locomo, primary)
    names = {"T": "truncation", "S": "summaries", "C": "compression", "M": "retrieval"}

    def signed(value):
        return f"{100*value:+.1f}".replace("-", "$-$")
    text = (r"\subsection{Does the order hold for another model and another benchmark?}\label{sec:transfer}" + "\n"
            "Two descriptive checks repeat the comparison outside the primary setting, each at 4,096 tokens. The "
            "first gives the same 500 questions and session summaries to Qwen3-8B, a smaller attention-only model "
            "from the same developer, served by llama.cpp with its own tokenizer and chat template "
            r"(Table~\ref{tab:p5-crossmodel}). ")
    lead = leaders(cross["cells"])
    top = 100 * cross["cells"][lead.split(" = ")[0]]["accuracy"]
    first = max(ARMS, key=lambda arm: primary["cells"][f"{arm}@4096"]["correct"])
    text += (f"{tex(lead)} tied for the highest point estimate ({top:.1f}\\%)" if " = " in lead else
             f"{names[lead].capitalize()}{' again' if lead == first else ''} had the highest point estimate ({top:.1f}\\%)")
    counts = {arm: cross["cells"][arm]["correct"] for arm in ARMS}
    counts_primary = {arm: primary["cells"][f"{arm}@4096"]["correct"] for arm in ARMS}
    if len(set(counts.values())) == len(ARMS) == len(set(counts_primary.values())):
        order = sorted(ARMS, key=counts.get, reverse=True)
        same = order == sorted(ARMS, key=counts_primary.get, reverse=True)
        text += (", and the four point estimates " + ("kept" if same else "did not keep")
                 + f" the order of the primary 27B run ({', '.join(order)})")
    text += ". This check changes the model and the serving stack together, so it isolates neither.\n\n"
    rows = []
    for arm in ARMS:
        c = cross["cells"][arm]
        diff = cross["contrasts_vs_T"].get(arm)
        rows.append([arm, f"{100*c['accuracy']:.1f}",
                     f"[{100*c['wilson95'][0]:.1f}, {100*c['wilson95'][1]:.1f}]",
                     "--" if diff is None else signed(diff["difference"]),
                     "--" if diff is None else f"[{signed(diff['newcombe95'][0])}, {signed(diff['newcombe95'][1])}]",
                     "--" if diff is None else pvalue(diff["mcnemar_p_unadjusted"])])
    text += (f"The second check uses 200 LoCoMo questions. A pre-specified rule chose {selected} from S, M and C as "
             "the alternative with the most correct answers on the primary LongMemEval results; ties use that order, "
             "and selection does not require beating T. ")
    if selected == "S":
        text += ("This repeats the truncation-versus-summarization comparison that Kyrkewood ran on LoCoMo "
                 r"\citep{kyrkewood2026sleeping}; it does not assume that the earlier direction replicates. ")
    else:
        text += (f"Because the rule selected {names[selected]}, this check does not repeat the "
                 "truncation-versus-summarization comparison that Kyrkewood ran on LoCoMo "
                 r"\citep{kyrkewood2026sleeping}. ")
    delta, f1 = locomo["contrasts_vs_T"][selected], locomo["token_f1_differences_vs_T"][selected]
    text += (f"{names[selected].capitalize()} was judged correct on {100*locomo['cells'][selected]['accuracy']:.1f}\\% "
             f"of the questions against {100*locomo['cells']['T']['accuracy']:.1f}\\% for truncation, a difference of "
             f"{signed(delta['difference'])} points (pointwise Newcombe interval "
             f"[{signed(delta['newcombe95'][0])}, {signed(delta['newcombe95'][1])}], unadjusted McNemar "
             f"$p={pvalue(delta['mcnemar_p_unadjusted'])}$); these question-level calculations ignore that "
             f"questions share conversation histories. Token F1 compares the whole visible answer with a short gold "
             "string, so every extra word lowers it and its level is low for both arms; it changed by "
             f"{signed(f1['mean'])} points, with a conversation-resampled 95\\% interval of "
             f"[{signed(f1['percentile95'][0])}, {signed(f1['percentile95'][1])}] over only {f1['conversations']} "
             r"conversations (Table~\ref{tab:p5-locomo}). Neither result establishes general superiority across "
             "datasets." + "\n")
    text += table(["Arm", "Accuracy \\%", "95\\% interval", "$\\Delta$ vs T", "95\\% interval", "Raw $p$"], rows,
                  "Qwen3-8B, 500 items per arm. Wilson accuracy and paired Newcombe difference intervals are pointwise; "
                  "McNemar p-values are unadjusted and outside the primary Holm family.", "tab:p5-crossmodel")
    rows = []
    for arm in ("T", selected):
        c, f1 = locomo["cells"][arm], locomo["token_f1"][arm]
        rows.append([arm, c["n"], f"{100*c['accuracy']:.1f}", f"{100*f1['mean']:.1f}",
                     f"[{100*f1['percentile95'][0]:.1f}, {100*f1['percentile95'][1]:.1f}]"])
    text += table(["Arm", "$n$", "Judged \\%", "Token-F1 \\%", "Cluster 95\\% interval"], rows,
                  "LoCoMo aggregates only. F1 percentile intervals resample whole conversations, with 10,000 draws "
                  "and seed 20260920; the item-weighted mean is recomputed in every resample.", "tab:p5-locomo")
    return text

