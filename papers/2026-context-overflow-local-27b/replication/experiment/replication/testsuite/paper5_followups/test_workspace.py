"""Tiny synthetic fixtures; never read real datasets or start model services."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import followups
from legacy import algorithms
from workspace import (ROOT, Workspace, canonical, digest_text, namespace_prompt,
                       select_policy, sha, terminal_rows, validate_items)


def items_for(leg):
    count = 500 if leg == "cross-model" else 200
    items = []
    for index in range(count):
        item = {"question_id": f"synthetic-{index}", "question": "Synthetic question?",
                "answer": "synthetic", "question_date": "2026/01/01", "question_type": "single-session-user",
                "haystack_sessions": [[{"role": "user", "content": "synthetic fact"}]],
                "haystack_dates": ["2025/12/01"], "haystack_session_ids": [f"s{index}"], "answer_session_ids": []}
        if leg == "locomo":
            category = 1 if index < 70 else 2 if index < 140 else 4
            item.update(question_id=f"locomo-synthetic-{index}", conversation_id=f"synthetic-conversation-{index % 10}",
                        category=category, question_type=f"locomo-cat{category}")
        items.append(item)
    return items


def prompts_for(leg, arm, items):
    rows = []
    for item in items:
        text = "Synthetic prompt " + item["question_id"] + arm
        rows.append(namespace_prompt(leg, {"question_id": item["question_id"], "arm": arm, "budget": 4096,
                                          "prompt": text, "prompt_sha256": digest_text(text), "n_prompt": 25}))
    return rows


def primary_claims(scores=(250, 250, 250)):
    return {"schema": "paper5-claims-v1", "primary_judge": "gptoss",
            "cells": {arm + "@4096": {"n": 500, "correct": k, "accuracy": k / 500}
                      for arm, k in zip(("S", "M", "C"), scores)}}


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="p5-followup-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.json"
        self.source.write_text('{"synthetic": true}\n')
        self.inputs = {str(self.source): sha(self.source)}
        self.work = Workspace("cross-model", self.root)
        self.items = items_for("cross-model")

    def init(self):
        return self.work.initialize(self.items, 8192, self.inputs)

    def test_init_does_not_touch_primary(self):
        primary = self.root / "testsuite/paper5/work/prompts"
        primary.mkdir(parents=True)
        sentinel = primary / "T_4096.jsonl"
        sentinel.write_text("primary bytes must stay unchanged\n")
        before = sha(sentinel)
        self.init()
        self.work.write_prompts("T", prompts_for("cross-model", "T", self.items), self.inputs)
        self.assertEqual(sha(sentinel), before)
        self.assertEqual([p.name for p in primary.iterdir()], ["T_4096.jsonl"])

    def test_init_idempotent_but_different_inputs_refused(self):
        info = self.init()
        self.assertEqual(self.init(), info)
        with self.assertRaisesRegex(ValueError, "replace"):
            self.work.initialize(self.items, 16384, self.inputs)
        self.assertEqual(self.work.load()[0], info)

    def test_wrong_sample_or_duplicate_rejected(self):
        for items in (self.items[:-1], [self.items[0]] * 500):
            with self.assertRaises(ValueError):
                self.work.initialize(items, 8192, self.inputs)
        self.assertFalse(self.work.base.exists())

    def test_input_and_item_drift_rejected(self):
        self.init()
        self.source.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "bound input changed"):
            self.work.load()
        self.source.write_text('{"synthetic": true}\n')
        self.work.path("items.json").write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "item file changed"):
            self.work.load()

    def test_unsafe_paths_and_legs_rejected(self):
        for name in ("../primary", "locomo/../cross-model", "primary"):
            with self.assertRaises(ValueError):
                Workspace(name, self.root)
        for name in ("../primary", "/tmp/elsewhere", "prompts/../../elsewhere"):
            with self.assertRaises(ValueError):
                self.work.path(name)
        self.work.base.mkdir(parents=True)
        (self.work.base / "prompts").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.work.path("prompts/leak.jsonl")

    def test_symlinked_base_rejected(self):
        parent = self.root / "testsuite/paper5_followups"
        parent.mkdir(parents=True)
        (parent / "work").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            Workspace("locomo", self.root)

    def test_prompt_wrong_model_missing_duplicate_or_hash_rejected(self):
        self.init()
        for mutate in (lambda x: x.pop(), lambda x: x.append(x[0]),
                       lambda x: x[0].update(model="qwen38-exl3"),
                       lambda x: x[0].update(job_id="synthetic-0|T|4096"),
                       lambda x: x[0].update(n_prompt=4097),
                       lambda x: x[0].update(prompt="tampered")):
            rows = prompts_for("cross-model", "T", self.items)
            mutate(rows)
            with self.assertRaises(ValueError):
                self.work.write_prompts("T", rows, self.inputs)
        self.assertFalse(self.work.path("prompts/T_4096.jsonl").exists())

    def test_full_freeze_and_readback_are_nonmutating(self):
        self.init()
        for arm in ("T", "S", "C", "M"):
            self.work.write_prompts(arm, prompts_for("cross-model", arm, self.items), self.inputs)
        self.work.freeze()
        before = {str(p): (sha(p), p.stat().st_mtime_ns) for p in self.work.base.rglob("*") if p.is_file()}
        self.assertEqual(len(self.work.frozen_prompts()), 2000)
        after = {str(p): (sha(p), p.stat().st_mtime_ns) for p in self.work.base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        with self.assertRaisesRegex(ValueError, "already frozen"):
            self.work.write_prompts("T", prompts_for("cross-model", "T", self.items), self.inputs)

    def test_incomplete_or_extra_prompt_inventory_rejected(self):
        self.init()
        self.work.write_prompts("T", prompts_for("cross-model", "T", self.items), self.inputs)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.work.freeze()
        for arm in ("S", "C", "M"):
            self.work.write_prompts(arm, prompts_for("cross-model", arm, self.items), self.inputs)
        self.work.path("prompts/extra.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.work.freeze()

    def test_locomo_selection_quota_and_primary_gate(self):
        work = Workspace("locomo", self.root)
        items = items_for("locomo")
        work.initialize(items, 8192, self.inputs)
        work.write_prompts("T", prompts_for("locomo", "T", items), self.inputs)
        with self.assertRaises(FileNotFoundError):
            work.expected_arms()
        with patch.object(followups, "verified_primary", side_effect=ValueError("primary incomplete")):
            with self.assertRaisesRegex(ValueError, "primary incomplete"):
                followups.select(work)
        self.assertFalse(work.path("SELECTION.json").exists())
        with patch.object(followups, "verified_primary", return_value=(primary_claims(), self.inputs)):
            self.assertEqual(followups.select(work)["policy"], "S")
        self.assertEqual(work.expected_arms(), ("T", "S"))
        work.write_prompts("S", prompts_for("locomo", "S", items), self.inputs)
        work.freeze()
        self.assertEqual(len(work.frozen_prompts()), 400)
        self.assertEqual(work.load()[0]["publication"], "aggregates-only")
        bad = deepcopy(items)
        bad[0]["category"] = 3
        with self.assertRaises(ValueError):
            validate_items("locomo", bad)

    def test_terminal_duplicates_and_nonfinite_values_rejected(self):
        path = self.root / "journal.jsonl"
        row = {"job_id": "a", "terminal": True, "ok": True}
        path.write_text(json.dumps(row) + "\n")
        self.assertEqual(set(terminal_rows(path)), {"a"})
        for text in ((json.dumps(row) + "\n") * 2, '{"terminal":true,"ok":true,"x":NaN}\n',
                     '{"terminal":true,"ok":true,"x":1e309}\n',
                     '{"terminal":true,"terminal":false}\n'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                terminal_rows(path)


class SelectionAndBridgeTests(unittest.TestCase):
    def test_primary_numeric_budget_keys_survive_json_roundtrip(self):
        computed = {"secondary": {"ranking": {1024: ["S", "T"], 4096: ["T", "S"]}}, "flag": True}
        with tempfile.TemporaryDirectory(prefix="p5-primary-roundtrip-test-") as tmp:
            root = Path(tmp)
            path = root / "paper5/derived/claims.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(computed))
            saved = json.loads(path.read_text())
            self.assertNotEqual(saved, computed)  # reproduces the former failure
            evidence = SimpleNamespace(verify=lambda root: {"inputs": {"synthetic-evidence": "a"*64}})
            analyzer = SimpleNamespace(analyze=lambda: computed)
            def load(name, source):
                return analyzer if source.name == "analyze.py" else evidence
            with patch.object(followups, "load_module", side_effect=load), patch.object(followups, "algorithms", return_value=nullcontext()):
                claims, inputs = followups.verified_primary(root)
                self.assertEqual(claims, saved)
                self.assertEqual(inputs[str(path)], sha(path))
                saved["flag"] = 1
                path.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, "stale"):
                    followups.verified_primary(root)
                path.write_text(json.dumps(computed))
                def changed_during_replay():
                    path.write_text(json.dumps({"different": "claims"}))
                    return computed
                analyzer.analyze = changed_during_replay
                with self.assertRaisesRegex(ValueError, "claims changed"):
                    followups.verified_primary(root)

    def test_frozen_tie_order(self):
        self.assertEqual(select_policy(primary_claims()), "S")
        self.assertEqual(select_policy(primary_claims((200, 250, 250))), "M")
        self.assertEqual(select_policy(primary_claims((200, 249, 250))), "C")
        for value in (float("nan"), True, "0.5"):
            claims = primary_claims()
            claims["cells"]["S@4096"]["accuracy"] = value
            with self.assertRaises(ValueError):
                select_policy(claims)

    def test_bridge_isolates_paths_and_restores_imports_without_starting_services(self):
        before_path = list(sys.path)
        sentinel = object()
        with tempfile.TemporaryDirectory(prefix="p5-bridge-test-") as tmp, patch.dict(sys.modules, {"p5common": sentinel}):
            workspace = SimpleNamespace(root=ROOT, base=Path(tmp), leg="cross-model")
            with algorithms(workspace) as a:
                self.assertEqual(a.C.WORK, tmp)
                self.assertEqual(a.BP.MODEL, "qwen3-8b")
                self.assertTrue(Path(a.grid.JOURNALS).is_relative_to(tmp))
                # Pure prompt rendering with a tiny fake tokenizer; no model load.
                a.C.count_tokens = lambda text, *args: len(text)
                row = a.BP.build_T(items_for("cross-model")[0], 4096)
                namespaced = namespace_prompt("cross-model", row)
                self.assertEqual(row["prompt"], namespaced["prompt"])
                self.assertNotEqual(row["job_id"], namespaced["job_id"])
                self.assertFalse(any(Path(tmp).iterdir()))
            self.assertIs(sys.modules["p5common"], sentinel)
        self.assertEqual(sys.path, before_path)


if __name__ == "__main__":
    unittest.main()
