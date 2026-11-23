#!/usr/bin/env python3
"""Build follow-up prompts in their isolated workspace, never the primary grid.

T/S are CPU/tokenizer work. C reads cached classifier outputs; M runs the
pinned CPU embedding model. Check machine headroom before a full preparation
job. No command here starts a generation server or downloads a model.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import time

from followups import verified_primary
from legacy import algorithms
from workspace import LEGS, Workspace, digest_text, namespace_prompt, read_json, sha, terminal_rows


def build_records(a, leg, arm, items, *, summaries=None, lingua=None, embedder=None):
    """Use the primary algorithms unchanged; only identifiers are namespaced."""
    records = []
    for item in items:
        if arm == "T":
            row = a.BP.build_T(item, 4096)
        elif arm == "S":
            if summaries is None:
                raise ValueError("S requires the recorded generator's session summaries")
            row = a.BP.build_S(item, 4096, summaries)
        elif arm == "C":
            chunks, history = lingua.chunks(item["question_id"])
            if history != a.C.sha256_text(a.C.full_history(item)):
                raise ValueError("classifier cache belongs to a different history")
            row = a.BP.build_C(item, 4096, "C", lingua, chunks)
        elif arm == "M":
            rds = a.BP.rounds(item)
            started = time.perf_counter()
            order, _ = a.BP.rank_dense(embedder, item, rds)
            row = a.BP.build_M(item, 4096, "M", order, rds, time.perf_counter() - started)
        else:
            raise ValueError("unknown follow-up policy")
        records.append(namespace_prompt(leg, row))
    return records


class BoundCache:
    def __init__(self, legacy_lingua, directory):
        self.tok, self.L = legacy_lingua.tok, legacy_lingua.L
        self.directory = Path(directory)
        self.inputs = {}

    def chunks(self, qid):
        if Path(qid).name != qid:
            raise ValueError("unsafe cache question ID")
        path = self.directory / (qid + ".json.gz")
        before = sha(path)
        with gzip.open(path, "rt") as stream:
            record = json.load(stream)
        if sha(path) != before:
            raise ValueError("classifier cache changed while reading")
        if record.get("model_revision") != self.L.MODEL_REVISION or record.get("question_id") != qid:
            raise ValueError("classifier cache model/item mismatch")
        self.inputs[str(path)] = before
        return record["chunks"], record["history_sha256"]


def summary_cache(path):
    before = sha(path)
    rows = terminal_rows(path)
    values = {}
    for jid, row in rows.items():
        if row["ok"]:
            if not jid.startswith("sum|") or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("invalid or empty summary cache record")
            values[jid] = row["text"]
    if sha(path) != before:
        raise ValueError("summary journal is still changing; do not prepare S yet")
    return values, {str(path): before}


def prepare(work, arm, threads=2):
    info, items = work.load()
    allowed = ("T", "S", "C", "M") if work.leg == "cross-model" else (("T",) if arm == "T" else work.expected_arms())
    if arm not in allowed:
        raise ValueError("policy was not selected for this follow-up")
    if work.path("PROMPTS_MANIFEST.json").exists():
        raise ValueError("the follow-up prompts are already frozen")
    output = work.path(f"prompts/{arm}_4096.jsonl")
    if output.exists():
        raise ValueError("prompt file already exists; inspect its receipt instead of silently rebuilding")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("threads must be an integer from 1 to 32")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    inputs = {str(work.path("WORKSPACE.json")): sha(work.path("WORKSPACE.json"))}
    if work.leg == "locomo" and arm != "T":
        inputs[str(work.path("SELECTION.json"))] = sha(work.path("SELECTION.json"))
    started = time.monotonic()
    with algorithms(work) as a:
        summaries = lingua = embedder = None
        if arm == "S":
            if work.leg == "cross-model":
                # Primary summary generation must be over. A complete primary
                # analysis is also the protocol's prerequisite for this leg.
                _, bindings = verified_primary(work.root)
                inputs.update(bindings)
                source = work.root / "testsuite/paper5/work/journals/summaries.jsonl"
            else:
                source = work.path("journals/summaries.jsonl")
            summaries, bindings = summary_cache(source)
            inputs.update(bindings)
            supplement = work.path("journals/summaries.jsonl")
            if work.leg == "cross-model" and supplement.exists():
                # Sessions the primary never summarized but the 8B tokenizer's budget
                # reaches, generated by the same summarizer (research/P5_AMENDMENT_2026-10-30.md).
                done_path = work.path("execution/summaries-finished.json")
                done = read_json(done_path) if done_path.exists() else {}
                if done.get("complete") is not True or done.get("journal_sha256") != sha(supplement):
                    raise ValueError("the cross-model summary supplement has not completed a verified stage")
                extra, more = summary_cache(supplement)
                if set(extra) & set(summaries):
                    raise ValueError("the summary supplement repeats a primary summary")
                summaries = {**summaries, **extra}
                inputs.update(more)
                inputs[str(done_path)] = sha(done_path)
        if arm == "C":
            directory = (work.root / "testsuite/paper5/work/lingua") if work.leg == "cross-model" else work.path("lingua")
            lingua = BoundCache(a.BP.Lingua(), directory)
        if arm == "M":
            embedder = a.BP.Embedder()
            # The imported constructor defaults to 16; set the chosen CPU
            # placement's budget before doing any embeddings.
            embedder.torch.set_num_threads(threads)
            model = Path(a.BP.MINILM)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                inputs[str(model / name)] = sha(model / name)
            weights = [p for name in ("model.safetensors", "pytorch_model.bin") if (p := model / name).is_file()]
            if not weights:
                raise ValueError("pinned MiniLM weights absent")
            inputs.update({str(path): sha(path) for path in weights})
        records = build_records(a, work.leg, arm, items, summaries=summaries, lingua=lingua, embedder=embedder)
        if lingua is not None:
            inputs.update(lingua.inputs)
        work.write_prompts(arm, records, inputs)
    report = {"schema": "paper5-followup-preparation-timing-v1", "arm": arm,
              "host": platform.node(), "cpu_threads_requested": threads,
              "seconds": time.monotonic() - started, "model": info["model"],
              "boundary": "preparation only, excluding separately generated summaries/classifier caches",
              "inference_started": False, "prompts_sha256": sha(output)}
    work.write_json_once(f"preparation/{arm}-timing.json", report)
    return {"arm": arm, "prompts": len(records), "output": str(output), "inference_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", choices=tuple(LEGS))
    parser.add_argument("--arm", choices=("T", "S", "C", "M"), required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(Workspace(args.leg), args.arm, args.threads), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"P5 follow-up preparation not ready: {exc}\n")


if __name__ == "__main__":
    main()
