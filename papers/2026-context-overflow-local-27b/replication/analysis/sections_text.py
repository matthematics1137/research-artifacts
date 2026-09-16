"""Allowlisted, unchanged presentation definitions; no private input loading."""
# Full source SHA256: 48e2e4f63bc02a13b422960d72ad727933489ef7fe0cbd6f4adf0a2073d0de4b
from draft_support import integer, pvalue, table, tex
ARMS = ("T", "S", "C", "M")


WORDS = {"T": "truncation", "S": "summaries", "C": "compression", "M": "retrieval",
         "Cw": "recent-window compression", "Mbm25": "BM25 retrieval", "O": "oracle", "E": "no history"}


def listing(items, last="and"):
    """An English list without a serial comma, as in the rest of the paper."""
    items = list(items)
    if not items:
        raise ValueError("empty list")
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {last} " + items[-1]


def signed(value):
    """A signed percentage-point value with a typographic minus."""
    return f"{100*value:+.1f}".replace("-", "$-$")


def share(cell):
    """A cell's accuracy in percent."""
    return f"{100*cell['accuracy']:.1f}"


def spread(values):
    """'a' or 'a--b' for printed percentages, smallest first."""
    low, high = f"{min(values):.1f}", f"{max(values):.1f}"
    return low if low == high else f"{low}--{high}"


def ranked(cells, budget):
    """The four policies by correct answers at one budget, ties in the fixed arm order."""
    return sorted(ARMS, key=lambda arm: (-cells[f"{arm}@{budget}"]["correct"], ARMS.index(arm)))


def leader(cells, budget):
    """The single policy with the most correct answers, or None on a tie."""
    order = ranked(cells, budget)
    first, second = (cells[f"{arm}@{budget}"]["correct"] for arm in order[:2])
    return order[0] if first > second else None


def apart(cells, budget, arms):
    """True when the pointwise Wilson intervals of these arms at one budget do not overlap pairwise.

    This is a descriptive reading of Table 2, never a test: only the contrasts with truncation were tested."""
    spans = sorted((cells[f"{arm}@{budget}"]["wilson95"] for arm in arms), key=lambda w: w[0])
    return all(left[1] < right[0] for left, right in zip(spans, spans[1:]))


def unresolved_note(sec, budget):
    """Name every contrast with truncation at this budget whose paired interval includes zero, so that no
    sentence reads an order from it; an unresolved difference is 'not separated', never a ranking."""
    open_arms = [arm for arm in ("M", "S", "C")
                 if sec[f"{arm}_vs_T@{budget}"]["newcombe95"][0] <= 0 <= sec[f"{arm}_vs_T@{budget}"]["newcombe95"][1]]
    if not open_arms:
        return ""
    return ("; " + listing(f"{WORDS[arm]} ({span(sec[f'{arm}_vs_T@{budget}'])})" for arm in open_arms)
            + (" was" if len(open_arms) == 1 else " were") + " not separated from truncation at that budget")


def figure(filename, caption, label):
    """A full-width figure with its light and dark variants."""
    return (r"\begin{figure}[htbp]\centering" + "\n"
            + rf"\ifdarkmode\includegraphics[width=\linewidth]{{figures/dark/{filename}.pdf}}"
            + rf"\else\includegraphics[width=\linewidth]{{figures/{filename}.pdf}}\fi" + "\n"
            + rf"\caption{{{caption}}}\label{{{label}}}\end{{figure}}" + "\n")


def span(c):
    """A paired difference in points with its pointwise interval."""
    return f"{signed(c['difference'])} points [{signed(c['newcombe95'][0])}, {signed(c['newcombe95'][1])}]"


def subgroup_sections(a):
    """Show the declared question-type and overlapping abstention breakdowns."""
    names = {"single-session-user": "Single-session user", "single-session-assistant": "Single-session assistant",
             "single-session-preference": "Single-session preference", "multi-session": "Multi-session",
             "knowledge-update": "Knowledge update", "temporal-reasoning": "Temporal reasoning"}
    rows, firsts, abstention, typed = [], {}, {}, {}
    for budget in (1024, 4096):
        cells = [a["cells"][f"{arm}@{budget}"] for arm in ARMS]
        types = sorted(cells[0]["by_type"])
        if not types or any(set(c["by_type"]) != set(types) for c in cells):
            raise ValueError("question-type inventories differ across policies")
        for c in cells:
            if sum(row["n"] for row in c["by_type"].values()) != c["n"] or sum(
                    row["correct"] for row in c["by_type"].values()) != c["correct"]:
                raise ValueError("question-type counts do not sum to the cell")
        for label, group in [(names.get(t, t), [c["by_type"][t] for c in cells]) for t in types] + [
                ("Abstention (overlaps types)", [c["abstention"] for c in cells])]:
            n = integer(group[0]["n"], "subgroup denominator", 1, 500)
            if any(row["n"] != n for row in group):
                raise ValueError("subgroup denominators differ across policies")
            scores = [integer(row["correct"], "subgroup correct", 0, n) for row in group]
            rows.append([f"{budget:,}", tex(label), n, *(f"{100*k/n:.1f}" for k in scores)])
            if budget == 4096:  # point-estimate leader(s) per group, ties joined
                tops = tuple(arm for arm, k in zip(ARMS, scores) if k == max(scores))
                if label.startswith("Abstention"):
                    abstention = {"n": n, "scores": dict(zip(ARMS, scores)), "tops": tops}
                else:
                    firsts[label] = tops
                    typed[label] = {arm: 100 * k / n for arm, k in zip(ARMS, scores)}
    tally = {}
    for tops in firsts.values():
        tally[tops] = tally.get(tops, 0) + 1
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    total = words.get(len(firsts), str(len(firsts)))

    def who(tops):
        return WORDS[tops[0]] if len(tops) == 1 else listing(WORDS[arm] for arm in tops) + " (tied)"
    lead = "; ".join(f"{who(tops)} in all {total} types" if k == len(firsts)
                     else f"{who(tops)} in {words.get(k, str(k))} of the {total} types"
                     for tops, k in sorted(tally.items(), key=lambda x: (-x[1], x[0])))
    text = (r"\subsection{Does the order hold across question types?}\label{sec:types}" + "\n"
            + r"Table~\ref{tab:p5-subgroups} splits accuracy by LongMemEval's declared question types. At 4,096 tokens "
            f"the highest point estimate went to {lead}")
    only = next(iter(tally)) if len(tally) == 1 else ()
    if len(only) == 1 and only[0] != "T":
        gaps = {label: scores[only[0]] - scores["T"] for label, scores in typed.items()}
        low, high = min(gaps, key=gaps.get), max(gaps, key=gaps.get)
        text += (f", with leads over truncation from {gaps[low]:.1f} points ({low[0].lower() + low[1:]}) to "
                 f"{gaps[high]:.1f} points ({high[0].lower() + high[1:]})")
    text += ". "
    scores, n = abstention["scores"], abstention["n"]
    counts = listing(f"{WORDS[arm]} was credited on {scores[arm]} of them" if i == 0 else f"{WORDS[arm]} on {scores[arm]}"
                     for i, arm in enumerate(ARMS))
    text += f"The {n} abstention questions, whose answer is absent from the history, overlap the types"
    if abstention["tops"] == ("T",):
        text += f" and put truncation first: {counts}. "
    else:
        text += f"; {counts}. "
    text += "These groups are small, and the breakdown is descriptive; it does not test policy-by-type interactions.\n"
    return text + table(["Budget", "Question group", "$n$", "T \\%", "S \\%", "C \\%", "M \\%"], rows,
                        "Primary-judge accuracy within each declared question group. Abstention overlaps the type "
                        "rows; it is not an additional sample.", "tab:p5-subgroups")


def secondary_sections(a):
    """The other budget and the two declared variants; descriptive contrasts only."""
    sec, cells = a["secondary"], a["cells"]
    rows = []
    for name in ("S_vs_T@1024", "C_vs_T@1024", "M_vs_T@1024", "Cw_vs_T@1024",
                 "Cw_vs_T@4096", "Mbm25_vs_T@4096", "Cw_vs_C@4096", "Mbm25_vs_M@4096"):
        c = sec[name]
        rows.append([tex(name.replace("_vs_", " vs ").replace("@", ", B=")), signed(c["difference"]),
                     f"[{signed(c['newcombe95'][0])}, {signed(c['newcombe95'][1])}]", pvalue(c["mcnemar_p"])])
    one, four = ranked(cells, 1024), ranked(cells, 4096)
    body = (r"\subsection{Does the order hold at 1,024 tokens, and do the variants change it?}\label{sec:budget}" + "\n"
            + "At 1,024 tokens the four policies scored "
            + listing(f"{share(cells[arm + '@1024'])}\\% ({WORDS[arm]})" for arm in one))
    unique = all(len({cells[f"{arm}@{b}"]["correct"] for arm in ARMS}) == len(ARMS) for b in (1024, 4096))
    lower = all(cells[f"{arm}@1024"]["correct"] < cells[f"{arm}@4096"]["correct"] for arm in ARMS)
    if unique and one == four:
        body += ", the same order of point estimates as at 4,096 tokens" + (", at lower levels" if lower else "")
    body += (". Against truncation at that budget, "
             + listing((f"{WORDS[arm]} differed by {span(sec[arm + '_vs_T@1024'])}" if i == 0 else
                        f"{WORDS[arm]} by {span(sec[arm + '_vs_T@1024']).replace(' points', '', 1)}")
                       for i, arm in enumerate(("M", "S", "C")))
             + r" (Table~\ref{tab:p5-secondary})" + unresolved_note(sec, 1024)
             + ", and the alternatives were not tested against each other. ")
    lo, hi = sec["Mbm25_vs_M@4096"]["newcombe95"]
    body += ("At 4,096 tokens, "
             + (f"BM25 ranking of past exchanges was not separated from MiniLM ranking ({span(sec['Mbm25_vs_M@4096'])})"
                if lo <= 0 <= hi else
                f"ranking past exchanges with BM25 instead of MiniLM changed retrieval accuracy by {span(sec['Mbm25_vs_M@4096'])}")
             + ", and compressing only the recent window instead of the whole history changed compression accuracy by "
             f"{span(sec['Cw_vs_C@4096'])}. ")
    tokens = {arm: cells[arm + "@1024"].get("delivered_prompt_tokens_mean") for arm in ARMS}
    if tokens["M"] is not None and tokens["C"] is not None and tokens["M"] < tokens["C"]:
        body += (f"Equal budgets did not give equal prompt lengths: at 1,024 tokens retrieval delivered "
                 f"{tokens['M']:,.0f} prompt tokens on average and compression {tokens['C']:,.0f}, because retrieval "
                 r"stops at the first ranked round that does not fit (Figure~\ref{fig:p5-cost}). ")
    body += ("These comparisons are descriptive, and a contrast that is significant at one budget but not at the "
             "other does not by itself establish an interaction between policy and budget.\n")
    body += table(["Comparison", "Points", "95\\% interval", "Raw $p$"], rows,
                  "Secondary paired contrasts, 500 items each, with pointwise Newcombe intervals and unadjusted "
                  "McNemar p-values; descriptive, outside the three-test confirmatory family.", "tab:p5-secondary")
    return body


def selection_section(a):
    """Evidence retention by policy and the oracle subset: selection versus reading."""
    cells, sec = a["cells"], a["secondary"]

    def diag(key, name):
        value = (cells[key].get("diag_means") or {}).get(name)
        return None if value is None else 100 * value
    text = (r"\subsection{Do the policies lose answers in selecting the text or in reading it?}\label{sec:selection}"
            + "\n" + "Each policy has a diagnostic of how much annotated evidence its prompts kept, with its own "
            r"denominator (Section~\ref{sec:statistics}); Figure~\ref{fig:p5-evidence} plots accuracy against it. ")
    kept = {key: diag(key, "evidence_turns_kept") for key in ("T@4096", "M@4096", "Mbm25@4096")}
    covered, recalled = diag("S@4096", "evidence_sessions_summarized"), diag("C@4096", "evidence_word_recall")
    if None not in kept.values() and covered is not None and recalled is not None:
        text += (f"At 4,096 tokens the prompts built by truncation held {kept['T@4096']:.1f}\\% of the evidence turns "
                 f"and those built by retrieval {kept['M@4096']:.1f}\\% (BM25 ranking: {kept['Mbm25@4096']:.1f}\\%); "
                 f"summaries covered {covered:.1f}\\% of the evidence sessions, and compression kept "
                 f"{recalled:.1f}\\% of the evidence words.")
    text += "\n" + figure("fig_evidence", "Accuracy against each policy's evidence diagnostic at both budgets "
                          "(circles 1,024 tokens, squares 4,096). Retained evidence turns, summarized evidence "
                          "sessions and evidence-word recall have different denominators and are plotted on separate "
                          r"axes (Section~\ref{sec:statistics}).",
                          "fig:p5-evidence") + "\n"
    gap = sec.get("oracle_gap_on_fit_subset") or {}
    if gap.get("O_minus_arm") and "O@4096" in cells:
        g, o = gap["O_minus_arm"], cells["O@4096"]
        order = sorted(ARMS, key=lambda arm: (g[arm]["difference"], ARMS.index(arm)))
        text += (f"The oracle arm replaces selection with the annotated evidence: on the {gap['n_fit']} questions "
                 "whose evidence sessions fit "
                 f"4,096 tokens, the generator was judged correct on {share(o)}\\% when shown only those sessions "
                 r"(Table~\ref{tab:p5-oracle}). On the same questions "
                 + listing(f"{WORDS[arm]} trailed the oracle by {100*g[arm]['difference']:.1f} points" if i == 0
                           else f"{WORDS[arm]} by {100*g[arm]['difference']:.1f}" for i, arm in enumerate(order))
                 + ". ")
        if 100 * o["accuracy"] >= 90 and all(g[arm]["difference"] > 0 for arm in ARMS):
            text += ("On this subset the generator answered nearly every question when the evidence was all it saw, "
                     "which points to selection as the main source of lost answers. The oracle prompt also omits all "
                     "other history, so this comparison does not separate missing evidence from added distraction.\n")
        else:
            text += "\n"
        rows = [["O minus " + arm, signed(c["difference"]), f"[{signed(c['newcombe95'][0])}, {signed(c['newcombe95'][1])}]"]
                for arm in ARMS for c in (g[arm],)]
        text += table(["Same-subset comparison", "Points", "95\\% interval"], rows,
                      f"Paired gaps between the oracle and each policy on the same {gap['n_fit']} fitting items, "
                      "the only items the oracle covers.", "tab:p5-oracle")
    return text


def cost_section(a):
    """Recorded generation and selection time, with its accounting limits."""
    from reporting import first_measured
    cells = {arm: a["cells"][arm + "@4096"] for arm in ARMS}
    text = (r"\subsection{What did the policies cost in recorded time?}\label{sec:cost}" + "\n"
            + r"Figure~\ref{fig:p5-cost} shows the prompt tokens each policy delivered and the seconds recorded per "
            "question. ")
    tokens = [c.get("delivered_prompt_tokens_mean") for c in cells.values()]
    if None not in tokens:
        text += f"At 4,096 tokens the four policies delivered {min(tokens):,.0f}--{max(tokens):,.0f} prompt tokens on average. "
    seconds = {arm: c.get("generation_seconds_mean") for arm, c in cells.items()}
    if None not in seconds.values():
        order = sorted(ARMS, key=lambda arm: (seconds[arm], ARMS.index(arm)))
        text += ("Mean generation time per answer was "
                 + listing(f"{seconds[arm]:.1f}\\,s for {WORDS[arm]}" for arm in order) + ". ")
    selection = {arm: first_measured(cells[arm].get("diag_means") or {}, "policy_seconds", "selection_seconds")
                 for arm in ARMS}
    timed = sorted((arm for arm in ARMS if selection[arm] is not None), key=lambda arm: (selection[arm], ARMS.index(arm)))
    if timed:
        text += ("Recorded selection added " + listing(f"{selection[arm]:.2f}\\,s for {WORDS[arm]}" for arm in timed)
                 + (", whose timer includes computing embeddings not already cached" if "M" in timed else ""))
        untimed = [arm for arm in ARMS if selection[arm] is None]
        text += ("; " + listing(WORDS[arm] for arm in untimed) + " had no per-question selection timer. "
                 if untimed else ". ")
    text += ("These timings follow policy-specific accounting boundaries and exclude summary generation, "
             r"classifier preprocessing and the pause between requests (Section~\ref{sec:statistics}), so they do "
             "not rank the policies by end-to-end cost.\n")
    return text + figure("fig_cost", "Delivered prompt tokens (left) and recorded seconds per question at 4,096 "
                         "tokens (right): generation plus the policy's recorded selection time, with accounting "
                         "boundaries that differ by policy. Summary generation, classifier preprocessing and the pause "
                         "between requests are excluded; retrieval's selection time includes embedding cache misses.",
                         "fig:p5-cost")


def discussion(a, followups=None):
    """Measured pattern, labeled mechanism hypotheses and implications, from checked aggregates; no new tests."""
    cells, sec = a["cells"], a["secondary"]
    acc = {key: 100 * c["accuracy"] for key, c in cells.items()}

    def diag(key, name, scale=100):
        value = (cells[key].get("diag_means") or {}).get(name)
        return None if value is None else scale * value

    text = [r"\section{Discussion}\label{sec:discussion}"]
    ev_t, ev_m = diag("T@4096", "evidence_turns_kept"), diag("M@4096", "evidence_turns_kept")
    para = r"\paragraph{Measured pattern.} The policies filled similar token allowances with different material."
    if ev_t is not None and ev_m is not None:
        para += (f" At 4,096 tokens the prompts built by truncation held {ev_t:.1f}\\% of the annotated evidence turns "
                 f"and those built by retrieval {ev_m:.1f}\\%, and the two policies scored {acc['T@4096']:.1f}\\% and "
                 f"{acc['M@4096']:.1f}\\%.")
    gap = sec.get("oracle_gap_on_fit_subset") or {}
    oracle_high = "O@4096" in cells and acc["O@4096"] >= 90 and bool(gap.get("O_minus_arm"))
    if oracle_high:
        para += (f" On the {gap['n_fit']} questions whose evidence sessions fit, the generator was judged correct on "
                 f"{acc['O@4096']:.1f}\\% when shown only those sessions (Section~\\ref{{sec:selection}}). We infer "
                 "that, for this generator, the policies differed mainly in how much evidence they put in front of the "
                 "model: where two policies share a retention measure, retention and accuracy rise together, and on the "
                 "fitting questions the model read the evidence almost perfectly. The oracle prompt also leaves out the "
                 "rest of the history, so this inference does not exclude a contribution from distraction.")
    text.append(para + "\n")

    para = (r"\paragraph{Possible mechanisms.} The design does not isolate why each policy lost evidence; the "
            "following explanations are consistent with the diagnostics but untested. LongMemEval-S spreads "
            "answer-bearing sessions throughout each history, so a recency window seldom reaches them")
    kept_t, total_t = diag("T@4096", "messages_kept", 1), diag("T@4096", "messages_total", 1)
    para += (f": at 4,096 tokens truncation kept {kept_t:.1f} of {total_t:.1f} messages on average. "
             if kept_t is not None and total_t is not None else ". ")
    kept_s, total_s = diag("S@4096", "sessions_kept", 1), diag("S@4096", "sessions_total", 1)
    covered = diag("S@4096", "evidence_sessions_summarized")
    para += "Session summaries reach further back"
    para += (f" ({kept_s:.1f} of {total_s:.1f} sessions)" if kept_s is not None and total_s is not None else "")
    para += (" but were written before the question was known and are short, so a covered session can still lose "
             "the needed fact")
    para += (f"; their coverage of evidence sessions ({covered:.1f}\\%) bounds what they can supply from above. "
             if covered is not None else ". ")
    rec_c, rec_w = diag("C@4096", "evidence_word_recall"), diag("Cw@4096", "evidence_word_recall")
    para += (r"Whole-history compression runs far outside the range in which LLMLingua-2 was validated "
             r"(Section~\ref{sec:policies})")
    para += f" and kept {rec_c:.1f}\\% of the evidence words. " if rec_c is not None else ". "
    window = sec["Cw_vs_C@4096"]
    if rec_c is not None and rec_w is not None and rec_w > rec_c and window["difference"] < 0:
        para += (f"Compressing only the recent window, inside that range, changed accuracy by {span(window)} even "
                 f"though it kept more of the evidence words ({rec_w:.1f}\\%); word recall counts lexical overlap "
                 "anywhere in the window, which may still omit the answer-bearing sessions. ")
    else:
        para += f"Compressing only the recent window, inside that range, changed accuracy by {span(window)}. "
    abst = {arm: cells[f"{arm}@4096"].get("abstention") for arm in ARMS}
    if all(abst.values()) and abst["T"]["correct"] > max(abst[arm]["correct"] for arm in ("S", "C", "M")):
        others = [abst[arm]["correct"] for arm in ("S", "C", "M")]
        low, high = min(others), max(others)
        para += (f"On abstention questions, more history may give the model material to answer from when it should "
                 f"decline, which would explain why truncation was credited on {abst['T']['correct']} of "
                 f"{abst['T']['n']} and the other policies on {low if low == high else f'{low}--{high}'}.")
    text.append(para + "\n")

    para = r"\paragraph{Implications.} "
    one, four = leader(cells, 1024), leader(cells, 4096)
    rankers = sec["Mbm25_vs_M@4096"]
    if one == four == "M":
        para += ("For an operator whose local assistant must answer questions about earlier sessions within a short "
                 "prompt, the comparison favors retrieval over the whole history: it scored highest at both budgets")
        if followups:
            cross = followups["cross-model"]["analysis"]["cells"]
            locomo = followups["locomo"]["analysis"]["cells"]
            if leader({f"{arm}@4096": cross[arm] for arm in ARMS}, 4096) == "M":
                para += " and with Qwen3-8B"
            if "M" in locomo and locomo["M"]["correct"] > locomo["T"]["correct"]:
                para += ", and it led truncation on LoCoMo"
        para += ". "
    else:
        para += "No single policy scored highest at both budgets. "
    lo, hi = rankers["newcombe95"]
    para += (f"BM25 ranking, which needs no embedding model, was {'not separated from' if lo <= 0 <= hi else ('above' if lo > 0 else 'below')} "
             f"MiniLM ranking ({span(rankers)}). ")
    if "E@None" in cells:
        para += (f"On LongMemEval-S, whose evidence is spread through the history, recency truncation at the same "
                 f"budget scored {acc['T@4096'] - acc['E@None']:.1f} points above the no-history control at 4,096 "
                 "tokens, so beating truncation alone says little about a new memory method on such benchmarks. ")
    para += "Question-aware summaries or lighter compression might narrow the gap; this study did not test them.\n"
    text.append(para)
    return "\n".join(text)


def diagnostic_sections(diag, followups, a=None):
    """Scorer failures and agreement; the closing sentences answer the subsection's question from the same rates."""
    if diag is None:
        return (r"\subsection{Do the scores depend on the scorer?}\label{sec:scorers}"
                + "\nScorer diagnostics are pending; this partial rendering is not a final result paper.\n")
    cohorts = [("27B", diag, tuple(arm + "@4096" for arm in ARMS))]
    if followups:
        for leg, title in (("cross-model", "8B"), ("locomo", "LoCoMo")):
            record = followups[leg]
            arms = ARMS if leg == "cross-model" else ("T", *(a for a in ("S", "M", "C") if a in record["analysis"]["cells"]))
            cohorts.append((title, record["diagnostics"], tuple(arm + "@4096" for arm in arms)))
    failures, pairs, rates, sizes = [], [], {}, {}
    pair_names = {"gptoss_vs_self": "local / self", "gptoss_vs_containment": "local / lexical",
                  "self_vs_containment": "self / lexical"}
    warning_count = empty_count = failed = 0
    for title, diagnostics, keys in cohorts:
        for key in keys:
            c = diagnostics["cells"][key]
            j = c["judge_failures_on_generated"]
            warning_count += sum(c["noncanonical_verdicts"].values())
            empty_count += c["empty_visible_answers_credited"]
            failed += c["generation_failures"] + sum(j.values())
            failures.append([title, key.split("@")[0], c["n"], c["generation_failures"], j["gptoss"],
                             j.get("self", "--"), c["empty_visible_answers_credited"]])
            if title == "LoCoMo":
                continue
            row = [title, key.split("@")[0]]
            for pair in pair_names:
                value = c["valid_pair_agreement"][pair]
                sizes.setdefault(pair, set()).add(value["n"])
                if value["raw"] is not None:
                    rates.setdefault((title, pair), []).append(100 * value["raw"])
                    if value["kappa"] is not None:
                        rates.setdefault((title, pair, "kappa"), []).append(value["kappa"])
                row += [value["n"], "--" if value["raw"] is None else f"{100*value['raw']:.1f}",
                        "--" if value["kappa"] is None else f"{value['kappa']:.2f}"]
            pairs.append(row)
    # One row per run and policy; a pair denominator that never varies moves into the caption.
    constant = {pair: next(iter(found)) for pair, found in sizes.items() if len(found) == 1}
    short = {"gptoss_vs_self": "Local--self", "gptoss_vs_containment": "Local--lexical",
             "self_vs_containment": "Self--lexical"}
    columns, headers = [0, 1], ["Run", "Arm"]
    for i, pair in enumerate(pair_names):
        base = 2 + 3 * i
        if pair not in constant:
            columns.append(base)
            headers.append(f"{short[pair]} $n$")
        columns += [base + 1, base + 2]
        headers += [f"{short[pair]} \\%", "$\\kappa$"]
    pairs = [[row[k] for k in columns] for row in pairs]
    denominators = (" Pair denominators per policy: " + listing(f"{short[pair].lower()} {constant[pair]}"
                                                                for pair in pair_names if pair in constant) + "."
                    if constant else "")
    body = (r"\subsection{Do the scores depend on the scorer?}\label{sec:scorers}" + "\n"
            "Scorer agreement is not validation against human correctness, and identical failure labels can inflate "
            "it, so the agreement rates below use only successful generations with a nonempty verdict from each "
            "compared scorer. ")
    shown = bool(failed or warning_count or empty_count)
    if failed == 0:
        body += ("No generation or verdict failed in any 4,096-token cell of the three runs"
                 + (r" (Table~\ref{tab:p5-failures})" if shown else "")
                 + ", so the rule that scores failures as incorrect changed no accuracy. ")
    else:
        body += (f"Across the 4,096-token cells, {failed} generations or verdicts failed and were scored incorrect "
                 r"(Table~\ref{tab:p5-failures}). ")
    if rates.get(("27B", "gptoss_vs_self")) and rates.get(("27B", "gptoss_vs_containment")):
        body += (f"In the 27B run the local judge agreed with the generator acting as a second judge on "
                 f"{spread(rates[('27B', 'gptoss_vs_self')])}\\% of verdicts per policy")
        kappas = rates.get(("27B", "gptoss_vs_self", "kappa"))
        body += f" ($\\kappa$ {min(kappas):.2f}--{max(kappas):.2f})" if kappas else ""
        body += (f", and with lexical containment of the gold answer on "
                 f"{spread(rates[('27B', 'gptoss_vs_containment')])}\\% of short-answer items "
                 r"(Table~\ref{tab:p5-scorers}). ")
    if rates.get(("8B", "gptoss_vs_self")):
        body += (f"In the 8B run the second judge, Qwen3-8B judging its own answers, agreed with the local judge on "
                 f"{spread(rates[('8B', 'gptoss_vs_self')])}\\% of verdicts")
        kappas = rates.get(("8B", "gptoss_vs_self", "kappa"))
        body += f" ($\\kappa$ {min(kappas):.2f}--{max(kappas):.2f})" if kappas else ""
        body += "; it enters no reported accuracy. "
    if a is not None and failed == 0 and rates.get(("27B", "gptoss_vs_self")) and \
            constant.get("gptoss_vs_self") == a["cells"]["T@4096"]["n"]:
        # Derived, not tested: if the two judges disagree on at most d% of one policy's verdicts, each policy's
        # accuracy can move by at most d points, so no adjacent pair whose gap exceeds 2d can swap.
        order = ranked(a["cells"], 4096)
        gaps = [100 * (a["cells"][f"{x}@4096"]["accuracy"] - a["cells"][f"{y}@4096"]["accuracy"])
                for x, y in zip(order, order[1:])]
        worst = 100 - min(rates[("27B", "gptoss_vs_self")])
        if 2 * worst < min(gaps):
            body += (f"In the 27B run the two judges' largest disagreement, {worst:.1f}\\% of one policy's verdicts, is less "
                     f"than half the smallest gap between adjacent policies at 4,096 tokens ({min(gaps):.1f} points), "
                     "so scoring with the second judge could not have changed the order of the four policies")
            cont = {arm: a["cells"][f"{arm}@4096"].get("containment_vs_judge") or {} for arm in ARMS}
            if all(c.get("containment_accuracy") is not None for c in cont.values()):
                by_cont = sorted(ARMS, key=lambda arm: (-cont[arm]["containment_accuracy"], ARMS.index(arm)))
                distinct = len({cont[arm]["containment_accuracy"] for arm in ARMS}) == len(ARMS)
                counted = {cont[arm].get("n") for arm in ARMS}
                body += ("; scored by lexical containment on "
                         + (f"the {next(iter(counted))} " if len(counted) == 1 and None not in counted else "the ")
                         + "short-answer items, the order was "
                         + ("also unchanged" if distinct and by_cont == order else "different")
                         + " (" + listing(f"{WORDS[arm]} {100*cont[arm]['containment_accuracy']:.1f}\\%" for arm in by_cont)
                         + ")")
            body += ". "
    body += "No agreement rate shows which scorer is correct. "
    if warning_count == empty_count == 0:
        body += ("Every nonempty verdict in the displayed cells was a bare yes or no (allowing a final period), and "
                 "no empty visible answer was credited; ")
    else:
        body += (f"Across the displayed cells, {warning_count} nonempty verdicts were not a bare yes or no (allowing "
                 f"a final period) and {empty_count} empty visible answers were credited; ")
    body += "the upstream rule that credits any reply containing ``yes'' was kept unchanged.\n"
    if shown:
        body += table(["Run", "Arm", "$n$", "Gen fail", "Local fail", "Self fail", "Empty credited"], failures,
                      "4,096-token diagnostic counts. Judge failures are counted only among successful generations. "
                      "Empty credited counts empty visible answers that the primary judge labeled correct; they were "
                      "kept as labeled.", "tab:p5-failures")
    dash = any("--" in row for row in pairs)
    body += table(headers, pairs,
                  "Scorer agreement on valid verdicts for the two LongMemEval deployments: the local judge "
                  "(gpt-oss-20b), the generator as second judge (self) and lexical containment of the gold answer. "
                  "Every scorer pair has its own denominator; these rates exclude failed verdicts and therefore differ "
                  "from failure-coded agreement. Lexical comparisons exclude preference and abstention questions"
                  + (", and an undefined $\\kappa$ is shown as a dash" if dash else "") + "." + denominators,
                  "tab:p5-scorers", "ll" + "r" * (len(headers) - 2))
    return body


def followup_summary(primary, followups):
    if followups is None:
        return "The cross-model and LoCoMo results are pending; no transfer claim is available."
    from followup_results import leaders, selected_policy
    cross, locomo = [followups[name]["analysis"] for name in ("cross-model", "locomo")]
    selected = selected_policy(primary)
    diff = locomo["token_f1_differences_vs_T"][selected]
    lead = leaders(cross["cells"]).split(" = ")
    top = 100 * cross["cells"][lead[0]]["accuracy"]
    judged = {arm: 100 * locomo["cells"][arm]["accuracy"] for arm in ("T", selected)}
    first = (f"With Qwen3-8B, {WORDS[lead[0]]} had the highest point estimate ({top:.1f}\\%). " if len(lead) == 1 else
             f"With Qwen3-8B, {listing(WORDS[arm] for arm in lead)} tied for the highest point estimate ({top:.1f}\\%). ")
    return (first + f"On LoCoMo, {WORDS[selected]}, the alternative selected from the primary results, was judged "
            f"correct on {judged[selected]:.1f}\\% of {locomo['cells']['T']['n']} questions from {diff['conversations']} "
            f"conversations against {judged['T']:.1f}\\% for truncation; its "
            f"token-F1 difference from truncation was {signed(diff['mean'])} points (conversation-resampled 95\\% "
            f"interval [{signed(diff['percentile95'][0])}, {signed(diff['percentile95'][1])}]). Both checks are descriptive.")

