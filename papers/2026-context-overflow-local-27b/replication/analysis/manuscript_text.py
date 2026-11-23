"""Allowlisted, unchanged presentation definitions; no private input loading."""
# Full source SHA256: 593a89e65a8a7661041dadd518eaec86d5fdb36c0b84c414ecbcb073f6d797cf
import math
from draft_support import command, integer, number, probability, pvalue, table, tex
from result_sections import (secondary_sections, subgroup_sections, diagnostic_sections, followup_summary, discussion,
    selection_section, cost_section, figure, leader, listing, ranked, share, signed, spread, apart, unresolved_note)
ARMS = ("T", "S", "C", "M")


NAMES = {"T": "truncation", "S": "summaries", "C": "compression", "M": "retrieval",
         "Cw": "recent-window compression", "Mbm25": "BM25 retrieval", "O": "oracle subset", "E": "no history"}


ACTIONS = {"T": "keeping the most recent turns", "S": "summarizing past sessions",
           "C": "compressing the history with LLMLingua-2",
           "M": "retrieving past exchanges by their similarity to the question"}


SCOPE = "not a ranking of memory policies in general or a measure of end-to-end cost"


def validate(a, evidence):
    if (a["schema"] != "paper5-claims-v1" or a["gen_journal"] != "grid.jsonl"
            or a["primary_judge"] != "gptoss" or a["secondary_judge"] != "self"):
        raise ValueError("wrong P5 analysis or judge identity")
    if set(a["cells"]) != set(evidence["cells"]):
        raise ValueError("analysis cells do not match completed evidence")
    for key, c in a["cells"].items():
        n = evidence["cells"][key]
        if c["n"] != n:
            raise ValueError("analysis denominator differs from evidence")
        integer(c["correct"], "correct", 0, n)
        probability(c["accuracy"])
        if not math.isclose(c["accuracy"], c["correct"] / n, abs_tol=1e-12):
            raise ValueError("accuracy disagrees with counts")
        lo, hi = c["wilson95"]
        if not 0 <= lo <= c["accuracy"] <= hi <= 1:
            raise ValueError("invalid accuracy interval")
        for name in ("missing_or_failed_generations", "cap_hits", "unterminated_thinking", "judge_failures"):
            integer(c[name], name, 0, n)
    primary = a["primary"]["contrasts_vs_T_at_4096"]
    if set(primary) != {"S", "C", "M"}:
        raise ValueError("incomplete confirmatory family")
    order = sorted(primary, key=lambda arm: probability(primary[arm]["mcnemar_p"]))
    previous = 0
    for rank, arm in enumerate(order):
        c = primary[arm]
        if c["n"] != 500:
            raise ValueError("incomplete paired contrast")
        adjusted = max(previous, min(1., (3-rank) * c["mcnemar_p"]))
        previous = adjusted
        if not math.isclose(c["holm_p"], adjusted, abs_tol=1e-12):
            raise ValueError("wrong Holm adjustment")
        if c["rejected_at_0.05"] is not (adjusted <= .05):
            raise ValueError("wrong confirmatory decision")
        diff = a["cells"][f"{arm}@4096"]["accuracy"] - a["cells"]["T@4096"]["accuracy"]
        if not math.isclose(c["difference"], diff, abs_tol=1e-12):
            raise ValueError("contrast disagrees with cell accuracies")
        lo, hi = c["newcombe95"]
        if not -1 <= lo <= diff <= hi <= 1:
            raise ValueError("invalid paired interval")
    if evidence["completion_cap"] not in (8192, 16384):
        raise ValueError("completion cap does not follow the pilot rule")


def answer(a):
    """The confirmatory decision in words; an unresolved contrast is 'not separated', never 'equal'."""
    family = a["primary"]["contrasts_vs_T_at_4096"]
    order = sorted(("S", "C", "M"), key=lambda arm: (-family[arm]["difference"], ("S", "C", "M").index(arm)))
    parts = []
    for i, arm in enumerate(order):
        c = family[arm]
        interval = f"{signed(c['newcombe95'][0])} to {signed(c['newcombe95'][1])}"
        parts.append(f"{signed(c['difference'])} points for {NAMES[arm]} (95\\% interval {interval})" if i == 0
                     else f"{signed(c['difference'])} for {NAMES[arm]} ({interval})")
    text = "The paired difference from truncation was " + listing(parts) + ". "
    held = [arm for arm in order if family[arm]["rejected_at_0.05"]]
    rest = [arm for arm in order if arm not in held]
    if not rest:
        return text + "All three remained significant after Holm correction across the pre-registered family."
    if not held:
        return text + "None of the three was separated from zero after Holm correction at this sample size."
    return (text + listing(NAMES[arm] for arm in held).capitalize() + " remained significant after Holm correction; "
            + listing(NAMES[arm] for arm in rest) + " could not be separated from truncation at this sample size.")


def levels(a, budget=4096):
    """Accuracy of the four policies at one budget, highest first."""
    cells = a["cells"]
    order = ranked(cells, budget)
    first = cells[f"{order[0]}@{budget}"]
    return (f"{NAMES[order[0]]} was judged correct on {share(first)}\\% of the questions, "
            + listing(f"{NAMES[arm]} on {share(cells[f'{arm}@{budget}'])}\\%" for arm in order[1:]))


def strong(a):
    """True when one policy leads at both budgets, by at least ten points at 4,096 tokens, and every
    confirmatory lead over truncation held; only then may the text say that it preserved far more answers."""
    family, cells = a["primary"]["contrasts_vs_T_at_4096"], a["cells"]
    top = leader(cells, 4096)
    if top is None or top == "T" or leader(cells, 1024) != top:
        return False
    first, second = (cells[f"{arm}@4096"]["accuracy"] for arm in ranked(cells, 4096)[:2])
    return first - second >= .1 and all(c["rejected_at_0.05"] and c["difference"] > 0 for c in family.values())


def primary_section(a):
    """The confirmatory comparison at 4,096 tokens, its tables and the accuracy figure."""
    cells, family = a["cells"], a["primary"]["contrasts_vs_T_at_4096"]
    text = (r"\section{Results}\label{sec:results}" + "\n"
            + r"\subsection{Which policy preserves the most answers at 4,096 tokens?}\label{sec:primary}" + "\n"
            "The confirmatory comparison pairs each alternative with truncation on the same 500 questions at 4,096 "
            r"tokens. Table~\ref{tab:p5-grid} lists every cell, Table~\ref{tab:p5-primary} gives the three "
            r"pre-registered contrasts, and Figure~\ref{fig:p5-arms} shows both budgets. ")
    sentence = levels(a)
    text += sentence[0].upper() + sentence[1:]
    if "E@None" in cells:
        gap = 100 * (cells["T@4096"]["accuracy"] - cells["E@None"]["accuracy"])
        text += (f"; truncation was {gap:.1f} points above the no-history control ({share(cells['E@None'])}\\%). "
                 if gap >= 0 else f"; truncation was {-gap:.1f} points below the no-history control "
                 f"({share(cells['E@None'])}\\%). ")
    else:
        text += ". "
    # The introduction states the paired differences; here the table carries them and the prose interprets.
    order = sorted(("S", "C", "M"), key=lambda arm: (-family[arm]["difference"], ("S", "C", "M").index(arm)))
    held = [arm for arm in order if family[arm]["rejected_at_0.05"]]
    if len(held) == 3 and all(family[arm]["newcombe95"][0] > 0 for arm in held):
        text += (r"All three paired intervals lie above zero, and all three contrasts remained significant after Holm "
                 r"correction (Table~\ref{tab:p5-primary}). ")
    elif not held:
        text += r"None of the three contrasts was separated from zero after Holm correction (Table~\ref{tab:p5-primary}). "
    else:
        text += (listing(NAMES[arm] for arm in held).capitalize() + " remained significant after Holm correction; "
                 + listing(NAMES[arm] for arm in order if arm not in held)
                 + r" could not be separated from truncation at this sample size (Table~\ref{tab:p5-primary}). ")
    first = family[order[0]]
    text += (f"Of the {first['arm_only'] + first['reference_only']} questions on which {NAMES[order[0]]} and truncation "
             f"disagreed, {NAMES[order[0]]} was right on {first['arm_only']}. ")
    top = leader(cells, 4096)
    if top is not None and top != "T":
        rest = [arm for arm in ranked(cells, 4096) if arm not in (top, "T")]
        leads = listing(f"{NAMES[arm]} ({100*(cells[top + '@4096']['accuracy'] - cells[arm + '@4096']['accuracy']):.1f} "
                        "points)" for arm in rest)
        text += (f"{NAMES[top].capitalize()} preserved the most answers. Only its lead over truncation was tested; its "
                 f"leads over {leads} are descriptive"
                 + (r", with non-overlapping Wilson intervals (Table~\ref{tab:p5-grid})"
                    if apart(cells, 4096, [top] + rest) else "") + ". ")
    failed = sum(c["missing_or_failed_generations"] + c["judge_failures"] for c in cells.values())
    text += ("No generation or judge verdict failed in any cell.\n" if failed == 0 else
             f"Across all cells, {failed} failed generations or verdicts were scored incorrect under the protocol.\n")
    rows = []
    for arm in ARMS + ("Cw", "Mbm25", "O", "E"):
        for budget in (1024, 4096, None):
            c = cells.get(f"{arm}@{budget}")
            if c is None:
                continue
            rows.append([tex(arm), "--" if budget is None else f"{budget:,}", c["n"], c["correct"],
                         f"{100*c['accuracy']:.1f}",
                         f"[{100*c['wilson95'][0]:.1f}, {100*c['wilson95'][1]:.1f}]",
                         c["missing_or_failed_generations"], c["judge_failures"]])
    text += table(["Arm", "Budget", "$n$", "Correct", "\\%", "95\\% interval", "Gen fail", "Judge fail"],
                  rows, "Primary-judge accuracy with pointwise Wilson intervals. T truncation, S summaries, "
                  "C compression, M retrieval (MiniLM), Cw recent-window compression, Mbm25 retrieval with BM25, "
                  "O oracle evidence sessions (fitting subset only), E no history. Failures are retained as "
                  "incorrect under the protocol.", "tab:p5-grid")
    contrasts = []
    for arm in ("S", "C", "M"):
        c = family[arm]
        contrasts.append([arm + " vs T", signed(c["difference"]),
                          f"[{signed(c['newcombe95'][0])}, {signed(c['newcombe95'][1])}]",
                          f"{c['arm_only']} / {c['reference_only']}",
                          pvalue(c["holm_p"]), "reject" if c["rejected_at_0.05"] else "not rejected"])
    text += table(["Contrast", "Points", "95\\% interval", "Discordant", "Holm $p$", "Decision"], contrasts,
                  "The three pre-registered paired comparisons at 4,096 tokens, 500 items each. Discordant pairs "
                  "count questions judged correct for the alternative only and for truncation only. Newcombe "
                  "intervals are pointwise; the Holm-adjusted exact McNemar family determines the decisions.",
                  "tab:p5-primary")
    text += figure("fig_arms", "Judged accuracy by policy at each budget, with Wilson 95\\% intervals (500 questions "
                   "per bar). The dotted line is the no-history control; the dashed oracle line covers only the "
                   "fitting subset.", "fig:p5-arms")
    return text


def abstract(a, followups):
    """The six-part abstract: problem, setting, design, results, meaning, limitation."""
    cells, family, sec = a["cells"], a["primary"]["contrasts_vs_T_at_4096"], a["secondary"]
    text = ("A local assistant whose conversation history far exceeds its prompt budget must choose what to keep. No "
            "study we found compares keeping the most recent turns with session summaries, learned compression and "
            "retrieval on full chat histories of over 100,000 tokens at one enforced prompt budget with a local model. "
            f"We ran all four on the {cells['T@4096']['n']} LongMemEval-S questions with a 2-bit Qwen3.8-27B on a "
            "12\\,GB laptop GPU, at prompt budgets of 1,024 and 4,096 tokens counted by the served tokenizer. A local "
            "judge scored every answer; pre-registered exact McNemar tests with Holm correction compared each "
            "alternative with truncation at 4,096 tokens. ")
    text += "At that budget, " + levels(a)
    text += (f", against {share(cells['E@None'])}\\% with no history. " if "E@None" in cells else ". ")
    held = [arm for arm in ("S", "C", "M") if family[arm]["rejected_at_0.05"]]
    if len(held) == 3 and all(family[arm]["difference"] > 0 for arm in held):
        text += "All three leads over truncation were significant"
    elif not held:
        text += "The test separated none of the three from truncation"
    else:
        text += "The test separated " + listing(NAMES[arm] for arm in held) + " from truncation"
    alternatives = [arm for arm in ranked(cells, 4096) if arm != "T"]
    text += ("; the order among the alternatives is descriptive"
             + (", with non-overlapping 95\\% intervals. " if apart(cells, 4096, alternatives) else ". "))
    top4, top1 = leader(cells, 4096), leader(cells, 1024)
    if top4 is not None and top1 == top4:
        others = [100 * cells[f"{arm}@1024"]["accuracy"] for arm in ARMS if arm != top1]
        text += (f"At 1,024 tokens {NAMES[top1]} again scored highest ({share(cells[top1 + '@1024'])}\\% against "
                 f"{spread(others)}\\%). ")
    else:
        text += ("At 1,024 tokens the four policies scored "
                 f"{spread([100 * cells[arm + '@1024']['accuracy'] for arm in ARMS])}\\%. ")
    o = cells.get("O@4096")
    gaps = (sec.get("oracle_gap_on_fit_subset") or {}).get("O_minus_arm") or {}
    if o:
        text += (f"Given only the evidence sessions, the model was judged correct on {share(o)}\\% of the {o['n']} "
                 "questions whose evidence fits 4,096 tokens")
        text += (", suggesting that on those questions the policies lost answers mainly by leaving the evidence out. "
                 if 100 * o["accuracy"] >= 90 and gaps and all(g["difference"] > 0 for g in gaps.values()) else ". ")
    bm25 = cells.get("Mbm25@4096")
    if bm25 and top4 == "M":
        lo, hi = sec["Mbm25_vs_M@4096"]["newcombe95"]
        text += (f"BM25 keyword ranking instead of embeddings gave {share(bm25)}\\%, "
                 + ("not separated from" if lo <= 0 <= hi else "above" if lo > 0 else "below")
                 + f" {share(cells['M@4096'])}\\%. ")
    if followups:
        cross = followups["cross-model"]["analysis"]["cells"]
        locomo = followups["locomo"]["analysis"]["cells"]
        best = max(cross[arm]["correct"] for arm in ARMS)
        lead = [arm for arm in ARMS if cross[arm]["correct"] == best]
        chosen = next(arm for arm in locomo if arm != "T")
        clusters = followups["locomo"]["analysis"]["token_f1_differences_vs_T"][chosen]["conversations"]
        where = f"{locomo['T']['n']} questions from {clusters} conversations"
        if lead == [top4] and chosen == top4 and locomo[chosen]["correct"] > locomo["T"]["correct"]:
            text += (f"In descriptive checks, {NAMES[top4]} also scored highest with Qwen3-8B, a smaller same-developer "
                     f"model ({share(cross[top4])}\\%), and, as the only alternative run on LoCoMo, it led truncation on "
                     f"{where} ({share(locomo[chosen])}\\% against {share(locomo['T'])}\\%). ")
        else:
            who = (f"{NAMES[lead[0]]} scored highest" if len(lead) == 1
                   else listing(NAMES[arm] for arm in lead) + " tied for the highest score")
            text += (f"In descriptive checks, {who} with Qwen3-8B, a smaller same-developer model "
                     f"({share(cross[lead[0]])}\\%), and on LoCoMo ({where}) the alternative selected from the primary "
                     f"results ({NAMES[chosen]}) scored {share(locomo[chosen])}\\% against {share(locomo['T'])}\\% for "
                     "truncation. ")
    else:
        text += "The descriptive transfer checks are pending. "
    if strong(a) and top4 == "M":
        text += ("For a local assistant answering questions about earlier sessions within a short prompt, the evidence "
                 "favors retrieval over the whole history rather than recency truncation. ")
    else:
        text += ("These paired measurements give operators a budget-matched comparison of the four policies for "
                 "questions about earlier sessions. ")
    return text + f"The results cover these implementations, one primary generator and local judges; they are {SCOPE}."


def answer_paragraph(a, followups):
    """The introduction's answer with its decisive evidence."""
    cells, sec = a["cells"], a["secondary"]
    sentence = levels(a)
    text = "At 4,096 tokens, " + sentence.replace("of the questions", f"of the {cells['T@4096']['n']} questions", 1)
    text += (f"; with no history the model scored {share(cells['E@None'])}\\%. " if "E@None" in cells else ". ")
    text += answer(a) + " "
    one, four = ranked(cells, 1024), ranked(cells, 4096)
    unique = all(len({cells[f"{arm}@{b}"]["correct"] for arm in ARMS}) == len(ARMS) for b in (1024, 4096))
    lower = all(cells[f"{arm}@1024"]["correct"] < cells[f"{arm}@4096"]["correct"] for arm in ARMS)
    scores = listing(f"{NAMES[arm]} {share(cells[arm + '@1024'])}\\%" for arm in one)
    if unique and one == four:
        text += (f"At 1,024 tokens the point estimates kept the same order{', at lower levels' if lower else ''} "
                 f"({scores})")
    else:
        text += f"At 1,024 tokens the four policies scored {scores}"
    text += unresolved_note(sec, 1024) + ", and the alternatives were not tested against each other. "
    gap = sec.get("oracle_gap_on_fit_subset") or {}
    if gap.get("O_minus_arm") and "O@4096" in cells:
        g = gap["O_minus_arm"]
        order = sorted(ARMS, key=lambda arm: (g[arm]["difference"], ARMS.index(arm)))
        text += (f"Shown only the evidence sessions, the generator was judged correct on {share(cells['O@4096'])}\\% "
                 f"of the {gap['n_fit']} questions whose evidence sessions fit 4,096 tokens; on those questions "
                 f"{NAMES[order[0]]} trailed this oracle by {100*g[order[0]]['difference']:.1f} points and "
                 f"{NAMES[order[-1]]} by {100*g[order[-1]]['difference']:.1f}. ")
    if followups:
        cross = followups["cross-model"]["analysis"]["cells"]
        locomo = followups["locomo"]["analysis"]["cells"]
        top = leader(cells, 4096)
        chosen = next(arm for arm in locomo if arm != "T")
        if (top is not None and leader({f"{arm}@4096": cross[arm] for arm in ARMS}, 4096) == top
                and chosen == top and locomo[top]["correct"] > locomo["T"]["correct"]):
            text += (f"In two descriptive checks, {NAMES[top]} again scored highest with Qwen3-8B and led truncation "
                     "on LoCoMo.")
        else:
            text += "Two descriptive checks, with Qwen3-8B and on LoCoMo, are reported separately."
    return text


def contribution(a):
    """The introduction's final paragraph: contribution, readers, supporting findings and scope (the map follows in main.tex)."""
    cells, sec = a["cells"], a["secondary"]
    top = leader(cells, 4096)
    if strong(a):
        others = [arm for arm in ("M", "S", "C", "T") if arm != top]
        verbs = {"T": "truncating", "S": "summarizing", "C": "compressing", "M": "retrieving from"}
        text = ("The primary contribution is this paired, budget-matched measurement of four standard policies, as "
                r"implemented here (Section~\ref{sec:policies}), for a locally served model: at the same prompt budget, "
                f"{ACTIONS[top]} preserved far more answers than "
                + listing((verbs[arm] for arm in others), "or") + " the history. ")
    else:
        text = ("The primary contribution is this paired, budget-matched measurement of four standard policies, as "
                r"implemented here (Section~\ref{sec:policies}), for a locally served model. ")
    text += ("It bears on how an operator should fill a short prompt for a local assistant and on whether recency "
             "truncation is an adequate baseline for memory evaluations of this kind; it proposes no new memory method. ")
    findings = []
    o, gap = cells.get("O@4096"), (sec.get("oracle_gap_on_fit_subset") or {}).get("O_minus_arm") or {}
    if o and gap and 100 * o["accuracy"] >= 90 and all(g["difference"] > 0 for g in gap.values()):
        findings.append("the oracle subset suggests that answers were lost in selecting the history rather than in "
                        "reading it")
    if "Mbm25_vs_M@4096" in sec:
        lo, hi = sec["Mbm25_vs_M@4096"]["newcombe95"]
        findings.append("BM25 ranking was " + ("not separated from" if lo <= 0 <= hi else "above" if lo > 0 else "below")
                        + " embedding ranking")
    if findings:
        text += ("Two findings support it: " if len(findings) == 2 else "One finding supports it: ") + ", and ".join(findings) + ". "
    return text + ("The four-policy comparison rests on LongMemEval-S, one quantized 27B generator and the local "
                   "gpt-oss-20b judge, and LoCoMo tested only retrieval against truncation; the results are "
                   f"{SCOPE}.")


def conclusion(a):
    """Answer, usable lesson and boundary; no new result."""
    cells = a["cells"]
    top = leader(cells, 4096)
    if strong(a) and top == "M":
        text = ("For Qwen3.8-27B, quantized to 2 bits per weight on a 12\\,GB laptop GPU, answering LongMemEval-S "
                "questions about chat histories of more than 100,000 tokens within the same prompt budget for every "
                "policy, retrieval over the whole history preserved far more answers than session summaries, "
                f"LLMLingua-2 compression or recency truncation: it was judged correct on {share(cells['M@4096'])}\\% "
                f"of the questions at 4,096 tokens and {share(cells['M@1024'])}\\% at 1,024, against "
                f"{share(cells['T@4096'])}\\% and {share(cells['T@1024'])}\\% for truncation. Its lead over truncation "
                "was confirmed by the pre-registered test; its lead over summaries and compression is descriptive. A "
                "local assistant that must answer questions about earlier sessions within a short prompt should start "
                "from retrieval over the full history rather than recency truncation, and evaluations on benchmarks "
                "like LongMemEval-S should not use truncation as their only baseline. ")
    else:
        text = ("At the same prompt budget, the four policies were judged correct on "
                f"{spread([100 * cells[arm + '@4096']['accuracy'] for arm in ARMS])}\\% of the LongMemEval-S "
                f"questions at 4,096 tokens and {spread([100 * cells[arm + '@1024']['accuracy'] for arm in ARMS])}\\% "
                "at 1,024. " + answer(a) + " ")
    return text + ("These results are limited to the tested implementations, one primary generator, LongMemEval-S with "
                   f"a retrieval-versus-truncation check on LoCoMo, and local judges; they are {SCOPE}.")


def failure_note(a):
    """Completions that ended inside an unterminated thinking block or at the completion cap, from the cell counts."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    arms = ARMS + ("Cw", "Mbm25", "O", "E")
    keys = sorted(a["cells"], key=lambda key: (arms.index(key.split("@")[0]) if key.split("@")[0] in arms else len(arms),
                                               key))

    def tally(name):
        hits = [(key.split("@")[0], key.split("@")[1], a["cells"][key][name]) for key in keys if a["cells"][key][name]]
        budgets = {budget for _, budget, _ in hits}
        if len(hits) == 1:
            arm, budget, _ = hits[0]
            inner = arm + ("" if budget == "None" else f" at {int(budget):,} tokens")
        else:
            inner = listing(f"{arm} {n}" + ("" if len(budgets) == 1 or budget == "None" else f" at {int(budget):,} tokens")
                            for arm, budget, n in hits) if hits else ""
            if len(budgets) == 1 and "None" not in budgets:
                inner += f", all at {int(next(iter(budgets))):,} tokens"
        return sum(n for _, _, n in hits), inner
    unclosed, where_unclosed = tally("unterminated_thinking")
    capped, where_capped = tally("cap_hits")
    plural = unclosed != 1
    text = (f"{words.get(unclosed, f'{unclosed:,}').capitalize()} answer{'s' if plural else ''} ended inside an "
            f"unterminated thinking block ({where_unclosed}) and {'were' if plural else 'was'} judged on "
            f"{'their' if plural else 'its'} empty visible answer{'s' if plural else ''}"
            if unclosed else "No answer ended inside an unterminated thinking block")
    return text + (f"; {words.get(capped, f'{capped:,}')} completion{'s' if capped != 1 else ''} ({where_capped}) "
                   "reached the completion cap." if capped else "; no completion reached the completion cap.")


def render(a, evidence, followups=None, diagnostics=None):
    validate(a, evidence)
    body = primary_section(a) + secondary_sections(a) + selection_section(a) + subgroup_sections(a)
    if followups is not None:
        from followup_results import sections
        body += sections(a, followups)
    else:
        body += ("\\subsection{Does the order hold for another model and another benchmark?}\\label{sec:transfer}\n"
                 "The cross-model and LoCoMo results are pending; this partial rendering is not final.\n")
    body += diagnostic_sections(diagnostics, followups, a)
    body += cost_section(a)
    body += discussion(a, followups)
    clusters = ("--" if not followups else
                str(next(iter(followups["locomo"]["analysis"]["token_f1_differences_vs_T"].values()))["conversations"]))
    return (command("PfiveAbstract", abstract(a, followups))
            + command("PfiveAnswerParagraph", answer_paragraph(a, followups))
            + command("PfiveContribution", contribution(a))
            + command("PfiveResultsSections", body) + command("PfiveConclusion", conclusion(a))
            + command("PfiveFollowupSummary", followup_summary(a, followups))
            + command("PfiveFailureNote", failure_note(a))
            + command("PfiveLocomoConversations", clusters)
            + command("PfiveCap", f"{evidence['completion_cap']:,}"))

