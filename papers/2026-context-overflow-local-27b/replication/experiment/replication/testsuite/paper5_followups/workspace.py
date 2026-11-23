"""Private, isolated P5 follow-up workspaces. No inference or network imports.

The primary experiment's prompt loader must never see secondary prompt files.
All writable paths here are confined to a named sibling workspace; immutable
inputs and prompt freezes use create-only writes, not replacement.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "publication"))
from draft_support import integer, probability, read_json, sha

LEGS = {
    "cross-model": {"dataset": "longmemeval-s-cleaned", "model": "qwen3-8b", "items": 500},
    "locomo": {"dataset": "locomo", "model": "qwen38-exl3", "items": 200},
}
TIE_ORDER = ("S", "M", "C")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_rows(path):
    """Strict JSONL, including duplicate keys and overflowed floating numbers."""
    from draft_support import _unique
    def reject(value):
        raise ValueError(f"nonfinite journal number: {value}")
    def finite(value):
        import math
        result = float(value)
        return result if math.isfinite(result) else reject(value)
    with Path(path).open() as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank row in {path}:{lineno}")
            row = json.loads(line, object_pairs_hook=_unique, parse_constant=reject, parse_float=finite)
            if not isinstance(row, dict):
                raise ValueError(f"row is not an object in {path}:{lineno}")
            yield row


def terminal_rows(path):
    result = {}
    for row in read_rows(path):
        if type(row.get("terminal")) is not bool:
            raise ValueError("journal row missing explicit terminal status")
        if not row["terminal"]:
            continue
        jid = row.get("job_id")
        if not isinstance(jid, str) or not jid or jid in result:
            raise ValueError("missing or duplicate terminal job ID")
        if type(row.get("ok")) is not bool:
            raise ValueError("terminal row missing explicit success/failure")
        result[jid] = row
    return result


class Workspace:
    def __init__(self, leg, root=ROOT):
        if leg not in LEGS:
            raise ValueError("unknown follow-up leg")
        self.leg = leg
        self.root = Path(root).resolve()
        self.base = self.root / "testsuite/paper5_followups/work" / leg
        self.path("WORKSPACE.json")  # validate existing ancestors, without writing

    def path(self, relative):
        rel = Path(relative)
        if rel.is_absolute() or not rel.parts or any(p in ("..", ".") for p in rel.parts):
            raise ValueError("workspace path must be a safe relative path")
        dest = self.base / rel
        current = self.root
        for part in dest.relative_to(self.root).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"symlink forbidden in writable workspace: {current}")
        if not dest.resolve().is_relative_to(self.base):
            raise ValueError("workspace escape")
        return dest

    def write_once(self, relative, content):
        path = self.path(relative)
        payload = content if isinstance(content, bytes) else content.encode()
        if path.exists():
            if path.is_file() and path.read_bytes() == payload:
                return path
            raise ValueError(f"refusing to replace existing workspace artifact: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path(relative)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic create-only install: a raced existing file is never overwritten.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def write_json_once(self, relative, value):
        return self.write_once(relative, canonical(value))

    def initialize(self, items, completion_cap, inputs):
        validate_items(self.leg, items)
        if type(completion_cap) is not int or completion_cap not in (8192, 16384):
            raise ValueError("completion cap must come from the recorded primary pilot")
        verify_inputs(inputs)
        item_text = canonical(items)
        info = {"schema": "paper5-followup-workspace-v1", "leg": self.leg,
                **LEGS[self.leg], "budget": 4096, "completion_cap": completion_cap,
                "items_sha256": digest_text(item_text), "inputs": inputs,
                "publication": "aggregates-only" if self.leg == "locomo" else "curated-item-level",
                "author_approved": False}
        # An interrupted initialization can be resumed only with identical inputs.
        self.write_json_once("WORKSPACE.json", info)
        self.write_once("items.json", item_text)
        return info

    def load(self):
        info = read_json(self.path("WORKSPACE.json"))
        if info.get("schema") != "paper5-followup-workspace-v1" or info.get("leg") != self.leg:
            raise ValueError("wrong workspace identity")
        for key, value in LEGS[self.leg].items():
            if info.get(key) != value:
                raise ValueError("workspace differs from declared leg")
        if info.get("budget") != 4096 or type(info.get("completion_cap")) is not int or info["completion_cap"] not in (8192, 16384):
            raise ValueError("workspace budget/cap differs from protocol")
        if sha(self.path("items.json")) != info["items_sha256"]:
            raise ValueError("frozen item file changed")
        items = read_json(self.path("items.json"))
        validate_items(self.leg, items)
        verify_inputs(info["inputs"])
        return info, items

    def expected_arms(self):
        if self.leg == "cross-model":
            return ("T", "S", "C", "M")
        selection = read_json(self.path("SELECTION.json"))
        if selection.get("schema") != "paper5-followup-selection-v1":
            raise ValueError("wrong policy-selection schema")
        if selection.get("policy") not in TIE_ORDER:
            raise ValueError("invalid selected policy")
        if select_policy(selection["primary_claims"]) != selection["policy"]:
            raise ValueError("selected policy differs from recorded primary counts")
        verify_inputs(selection["primary_inputs"])
        return ("T", selection["policy"])

    def write_prompts(self, arm, records, inputs):
        info, items = self.load()
        allowed = ("T", "S", "C", "M") if self.leg == "cross-model" else (("T",) if arm == "T" else self.expected_arms())
        if arm not in allowed:
            raise ValueError("arm not declared for this follow-up")
        if self.path("PROMPTS_MANIFEST.json").exists():
            raise ValueError("prompt inventory is already frozen")
        verify_inputs(inputs)
        validate_prompts(self.leg, arm, records, items)
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in records)
        self.write_once(f"prompts/{arm}_4096.jsonl", text)
        self.write_json_once(f"preparation/{arm}.json", {
            "schema": "paper5-followup-preparation-v1", "arm": arm,
            "model": info["model"], "prompts_sha256": digest_text(text), "inputs": inputs,
        })

    def freeze(self):
        info, items = self.load()
        arms = self.expected_arms()
        paths = {f"{arm}_4096.jsonl" for arm in arms}
        if {p.name for p in self.path("prompts").iterdir()} != paths:
            raise ValueError("unexpected or incomplete follow-up prompt inventory")
        manifest = {"schema": "paper5-followup-prompts-v1", "leg": self.leg,
                    "workspace_sha256": sha(self.path("WORKSPACE.json")), "files": {}, "preparation": {}}
        if self.leg == "locomo":
            manifest["selection_sha256"] = sha(self.path("SELECTION.json"))
        for arm in arms:
            name = f"{arm}_4096.jsonl"
            path = self.path("prompts/" + name)
            records = list(read_rows(path))
            validate_prompts(self.leg, arm, records, items)
            prep_path = self.path(f"preparation/{arm}.json")
            prep = read_json(prep_path)
            if prep.get("arm") != arm or prep.get("model") != info["model"] or prep.get("prompts_sha256") != sha(path):
                raise ValueError("stale prompt preparation receipt")
            verify_inputs(prep["inputs"])
            manifest["files"][name] = {"sha256": sha(path), "prompts": len(records)}
            manifest["preparation"][arm] = sha(prep_path)
        self.write_json_once("PROMPTS_MANIFEST.json", manifest)
        return manifest

    def frozen_prompts(self):
        # Recomputing is read-only when the create-only artifact already exists.
        if not self.path("PROMPTS_MANIFEST.json").is_file():
            raise ValueError("follow-up prompts have not been frozen")
        expected = read_json(self.path("PROMPTS_MANIFEST.json"))
        if self.freeze() != expected:
            raise ValueError("follow-up prompt freeze changed")
        return [row for name in expected["files"] for row in read_rows(self.path("prompts/" + name))]


def verify_inputs(inputs):
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("input bindings must be nonempty")
    for name, digest in inputs.items():
        if not Path(name).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("inputs require absolute local paths and full SHA-256 hashes")
        if sha(name) != digest:
            raise ValueError(f"bound input changed: {name}")


def validate_items(leg, items):
    if not isinstance(items, list) or len(items) != LEGS[leg]["items"]:
        raise ValueError("incomplete declared sample")
    seen = set()
    for item in items:
        qid = item.get("question_id")
        if not isinstance(qid, str) or not qid or "|" in qid or qid in seen:
            raise ValueError("missing or duplicate item ID")
        seen.add(qid)
        for key in ("question", "answer", "question_type", "question_date"):
            if not isinstance(item.get(key), str):
                raise ValueError(f"item requires string {key}")
        if leg == "locomo" and (not qid.startswith("locomo-") or not isinstance(item.get("conversation_id"), str) or not item["conversation_id"]):
            raise ValueError("LoCoMo needs its own IDs and conversation clusters")
        if leg == "locomo" and (type(item.get("category")) is not int or item["category"] not in (1, 2, 4)):
            raise ValueError("LoCoMo requires categories 1, 2, 4 only")
    if leg == "locomo" and Counter(item.get("category") for item in items) != {1: 70, 2: 70, 4: 60}:
        raise ValueError("LoCoMo sample must have the frozen 70/70/60 category quotas")


def job_id(leg, qid, arm):
    if leg not in LEGS or arm not in ("T", "S", "C", "M") or not isinstance(qid, str) or "|" in qid:
        raise ValueError("invalid follow-up job identity")
    return f"p5-followup|{leg}|{LEGS[leg]['model']}|{qid}|{arm}|4096"


def namespace_prompt(leg, record):
    return {**record, "job_id": job_id(leg, record["question_id"], record["arm"]),
            "leg": leg, "model": LEGS[leg]["model"]}


def validate_prompts(leg, arm, records, items):
    ids = {it["question_id"] for it in items}
    seen = set()
    for row in records:
        qid = row.get("question_id")
        if qid not in ids or qid in seen:
            raise ValueError("unexpected or duplicate prompt item")
        seen.add(qid)
        if row.get("job_id") != job_id(leg, qid, arm) or row.get("model") != LEGS[leg]["model"] or row.get("leg") != leg:
            raise ValueError("prompt lacks the correct model/leg namespace")
        if row.get("arm") != arm or row.get("budget") != 4096:
            raise ValueError("prompt arm/budget mismatch")
        integer(row.get("n_prompt"), "prompt count", 1, 4096)
        if not isinstance(row.get("prompt"), str) or digest_text(row["prompt"]) != row.get("prompt_sha256"):
            raise ValueError("prompt hash mismatch")
    if seen != ids:
        raise ValueError("prompt file does not cover the entire frozen sample")


def select_policy(claims):
    if claims.get("schema") != "paper5-claims-v1" or claims.get("primary_judge") != "gptoss":
        raise ValueError("policy selection requires the primary gptoss analysis")
    scores = {}
    for arm in TIE_ORDER:
        cell = claims["cells"][arm + "@4096"]
        integer(cell["n"], "primary denominator", 500, 500)
        scores[arm] = integer(cell["correct"], "primary correct", 0, 500)
        if abs(probability(cell["accuracy"]) - scores[arm] / 500) > 1e-12:
            raise ValueError("primary count and accuracy disagree")
    # max is stable: equal counts break S, M, C, not alphabetical order.
    return max(TIE_ORDER, key=scores.get)
