"""P5 follow-up evidence consumer and manuscript sections; read-only, no inference.

The running preparation harness is immutable. This consumer lives outside its
fingerprint and never changes prompts, judgments, selection, or execution receipts.
It replays the frozen descriptive analyzer without calling its write command.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from evidence import ROOT
import diagnostics as D
from draft_support import integer, number, probability, pvalue, read_json, sha, table, tex

MODELS = {"cross-model": "qwen3-8b", "locomo": "qwen38-exl3"}
ARMS = ("T", "S", "C", "M")
TIES = ("S", "M", "C")
SOURCE = ROOT / "testsuite/paper5_followups"
LLAMA = "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3"
TABBY = "4a4f9f44820303593844f092d424bb7506008733"
WEIGHTS = {
    "Qwen3-8B-UD-Q4_K_XL.gguf": "34a514d08f7449cb4a694a707aaa2eedccb7bb68290121bf5e5a569b2abe71c3",
    "gpt-oss-20b-MXFP4.gguf": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
    "model-00001-of-00002.safetensors": "cf8966ebc88599d904d4b1ac65d567a15b00b7f9c97ef54c8d77eb534ff8eb93",
    "model-00002-of-00002.safetensors": "2a72afda6bf041732018ee34aca772a4a268ef90905da0c710397e68a9e1a746",
}
RAW = {"add_bos_token": False, "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@contextmanager
def source_modules():
    """Isolate generic module names from other papers; imports start no serving process."""
    names = ("workspace", "legacy", "followups")
    absent = object()
    old = {k: sys.modules.get(k, absent) for k in names}
    paths = list(sys.path)
    def load(name, file, install=False):
        spec = importlib.util.spec_from_file_location(name, SOURCE / file)
        module = importlib.util.module_from_spec(spec)
        if install:
            sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    try:
        sys.path.insert(0, str(SOURCE))
        workspace = load("workspace", "workspace.py", True)
        load("legacy", "legacy.py", True)
        load("followups", "followups.py", True)
        analyzer = load("p5_followup_manuscript_analysis", "analyze.py")
        yield SimpleNamespace(W=workspace, A=analyzer)
    finally:
        for key, previous in old.items():
            if previous is absent:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous
        sys.path[:] = paths


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


def validate_execution(identity, leg, stage):
    served = "gptoss" if stage == "judge-gptoss" else MODELS[leg]
    D.require(identity["schema"] == "paper5-followup-execution-v1" and identity["leg"] == leg
              and identity["stage"] == stage, "wrong execution receipt")
    D.require(identity["machine"]["gpu"] == "NVIDIA GeForce RTX 4080 Laptop GPU", "follow-up changed study hardware")
    artifact = identity["artifact_identity"]
    engine = "tabbyAPI" if served == "qwen38-exl3" else "llama.cpp"
    D.require(artifact["served_model"] == served and artifact["engine"] == engine
              and artifact["revision"] == (TABBY if engine == "tabbyAPI" else LLAMA), "wrong served artifact/engine")
    files = artifact["files"]
    D.require(bool(files) and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in files.values()), "incomplete artifact hashes")
    required = (["gpt-oss-20b-MXFP4.gguf"] if served == "gptoss" else
                ["Qwen3-8B-UD-Q4_K_XL.gguf"] if served == "qwen3-8b" else
                ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    for name in required:
        found = [value for path, value in files.items() if Path(path).name == name]
        D.require(found == [WEIGHTS[name]], "execution did not record the pinned model weights")


def validate_generation_requests(leg, info, prompts, generations, manifest_sha):
    for prompt in prompts:
        row = generations[prompt["job_id"]]
        qid, arm = prompt["question_id"], prompt["arm"]
        seed = int(hashlib.sha256(f"{qid}|{arm}|4096".encode()).hexdigest()[:8], 16)
        meta = {"question_id": qid, "arm": arm, "budget_label": 4096,
                "prompt_file_sha256": prompt["prompt_sha256"], "leg": leg, "model": MODELS[leg],
                "prompt_manifest_sha256": manifest_sha, "prompt_sha256": prompt["prompt_sha256"]}
        params = {"max_tokens": info["completion_cap"], "temperature": .2, "top_p": .95, **RAW}
        meta["seed_applied" if leg == "cross-model" else "seed_recorded_not_applied"] = seed
        if leg == "cross-model":
            params["seed"] = seed
        D.require(canonical(row["meta"]) == canonical(meta), "generation identity/seed does not match the frozen request")
        if row["ok"]:
            D.require(canonical(row["params"]) == canonical(params), "follow-up sampling settings changed")
            D.require(row["n_prompt"] == row["server_prompt_tokens"] == prompt["n_prompt"] and row["budget"] == 4096,
                      "follow-up prompt-token parity/budget changed")
            D.require(row["completion_tokenizer_model"] == MODELS[leg], "completion count uses another model's tokenizer")
            # The server enforces the cap on the tokens it generated. The harness count re-encodes the decoded text,
            # and a reply cut at the cap can re-encode to a few more tokens (2026-10-30: three cross-model Qwen3-8B
            # replies, 8,195-8,198 against 8,192 generated). So the cap bounds the server's count where it reports
            # one, and the harness count may pass the cap only on a capped reply, by at most 1%.
            cap = info["completion_cap"]
            served = (row.get("usage") or {}).get("completion_tokens")
            if served is not None:
                integer(served, "server completion tokens", 0, cap)
            harness = integer(row["completion_tokens_harness"], "completion tokens", 0, cap + cap // 100)
            D.require(harness <= cap or (row["finish_reason"] == "length" and served in (None, cap)),
                      "completion count above the cap on a reply that was not capped")


def read_leg(leg, primary, root=ROOT):
    # Fail before expensive source/cache checks when measurements simply do not exist.
    analysis_path = Path(root) / "testsuite/paper5_followups/work" / leg / "analysis/analysis.json"
    before = sha(analysis_path)
    saved = read_json(analysis_path)
    arms = validate(leg, saved, primary)
    with source_modules() as s:
        work = s.W.Workspace(leg, root=root)
        info, items = work.load()
        prompts = work.frozen_prompts()
        D.require(tuple(work.expected_arms()) == arms, "follow-up selection no longer matches primary results")
        manifest_path = work.path("PROMPTS_MANIFEST.json")
        manifest_sha = sha(manifest_path)
        sources = {str(analysis_path): before, str(work.path("WORKSPACE.json")): sha(work.path("WORKSPACE.json")),
                   str(work.path("items.json")): sha(work.path("items.json")), str(manifest_path): manifest_sha,
                   **info["inputs"]}
        declared = ("gptoss", "self") if leg == "cross-model" else ("gptoss",)
        gen_path = work.path("journals/grid.jsonl")
        generations = s.W.terminal_rows(gen_path)
        judges = {name: s.W.terminal_rows(work.path(f"journals/judge_{name}__grid.jsonl")) for name in declared}
        recorded_inputs = {str(manifest_path): manifest_sha, str(gen_path): sha(gen_path)}
        stages = {"generate": (gen_path, generations)}
        stages.update({"judge-" + name: (work.path(f"journals/judge_{name}__grid.jsonl"), judges[name]) for name in declared})
        for stage, (journal, records) in stages.items():
            signature, finished = work.path(f"execution/{stage}.json"), work.path(f"execution/{stage}-finished.json")
            identity, receipt = read_json(signature), read_json(finished)
            fingerprint = sha(signature)
            validate_execution(identity, leg, stage)
            D.require(identity["workspace_sha256"] == sources[str(work.path("WORKSPACE.json"))]
                      and identity["prompt_manifest_sha256"] == manifest_sha, "execution used different frozen inputs")
            D.require(receipt["complete"] is True and receipt["journal_sha256"] == sha(journal)
                      and receipt["execution_fingerprint_sha256"] == fingerprint, "incomplete or stale execution finish")
            D.require(all(r["execution_fingerprint_sha256"] == fingerprint for r in records.values()), "mixed execution fingerprints")
            recorded_inputs.update({str(p): sha(p) for p in (journal, signature, finished)})
        D.require(saved["inputs"] == recorded_inputs and saved["analysis_code_sha256"] == sha(SOURCE / "analyze.py"), "stale follow-up analysis")
        validate_generation_requests(leg, info, prompts, generations, manifest_sha)
        replay = s.A.analyze_records(leg, arms, items, prompts, generations, judges, manifest_sha)
        D.require(canonical({k: v for k, v in saved.items() if k not in ("inputs", "analysis_code_sha256")}) == canonical(replay),
                  "follow-up aggregates differ from the frozen item-level analysis")
        diag = D.summarize(items, prompts, generations, judges)
        # Neither prep timings nor source licensing are silently inferred from accuracy.
        sources.update(recorded_inputs)
        if leg == "locomo":
            sources[str(work.path("SELECTION.json"))] = sha(work.path("SELECTION.json"))
            conversations = len({it["conversation_id"] for it in items})
            D.require(all(x["conversations"] == conversations for x in replay["token_f1"].values()), "F1 used different conversation clusters")
        work.frozen_prompts()
        D.require(all(sha(path) == value for path, value in sources.items()), "follow-up evidence changed while reading")
    return {"analysis": saved, "diagnostics": diag, "inputs": sources}


def read_all(primary, root=ROOT):
    return {leg: read_leg(leg, primary, root) for leg in MODELS}


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
