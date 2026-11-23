from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jobs
from legacy import algorithms
import run as R
from test_workspace import items_for, prompts_for
from workspace import ROOT, Workspace, canonical, digest_text, sha


class JobTests(unittest.TestCase):
    def context(self, leg="cross-model"):
        temp = tempfile.TemporaryDirectory(prefix="p5-followup-job-test-")
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        marker = base / "PROMPTS_MANIFEST.json"
        marker.write_text('{"synthetic":true}\n')
        return SimpleNamespace(root=ROOT, base=base, leg=leg, path=lambda name: base / name)

    def test_cross_model_seed_applied_and_primary_seed_not_claimed_applied(self):
        for leg, model in (("cross-model", "qwen3-8b"), ("locomo", "qwen38-exl3")):
            work = self.context(leg)
            with algorithms(work) as a:
                prompt = prompts_for(leg, "T", items_for(leg)[:1])[0]
                job = jobs.generation_job(a, work, prompt, {"model": model, "completion_cap": 8192})
                self.assertEqual(job["prompt"], prompt["prompt"])
                self.assertEqual(job["meta"]["prompt_manifest_sha256"], sha(work.path("PROMPTS_MANIFEST.json")))
                if leg == "cross-model":
                    self.assertEqual(job["params"]["seed"], job["meta"]["seed_applied"])
                    self.assertNotIn("seed_recorded_not_applied", job["meta"])
                else:
                    self.assertNotIn("seed", job["params"])
                    self.assertIn("seed_recorded_not_applied", job["meta"])

    def test_self_judge_uses_actual_generator_wrapper_and_tokenizer(self):
        work = self.context()
        with algorithms(work) as a:
            calls = []
            a.C.count_tokens = lambda text, directory: calls.append(directory) or 23
            item = items_for("cross-model")[0]
            gen = {"job_id": "synthetic", "ok": True, "text": "thinking</think>synthetic"}
            info = {"model": "qwen3-8b", "completion_cap": 8192}
            job = jobs.judgment_job(a, work, "self", item, gen, info)
            self.assertEqual(calls, [a.C.model_dir("qwen3-8b")])
            self.assertEqual(job["meta"]["gen_record_sha256"], digest_text(canonical(gen)))
            self.assertTrue(job["verify_tokens"])
            self.assertIn("</think>", job["prompt"])
            primary = jobs.judgment_job(a, work, "gptoss", item, gen, info)
            self.assertEqual(primary["params"]["max_tokens"], 2048)
            self.assertNotIn("prompt", primary)

    def test_cross_model_completion_counter_and_cache_binding(self):
        work = self.context()
        with algorithms(work) as a:
            a.C.count_tokens = lambda text, directory: 7 if directory == a.C.model_dir("qwen3-8b") else 99
            Runner = R.runner_type(a, work, "qwen3-8b", "f" * 64)
            runner = Runner(SimpleNamespace(), str(work.base / "journals/test.jsonl"), idle_seconds=10)
            row = {"job_id": "x", "terminal": True, "ok": True, "text": "synthetic",
                   "completion_tokens_harness": 99, "meta": {"prompt_sha256": "p"},
                   "params": {"temperature": .2}, "prompt_sha256": "p"}
            runner._append(row)
            self.assertEqual(row["completion_tokens_harness"], 7)
            self.assertEqual(row["completion_tokenizer_model"], "qwen3-8b")
            runner.done["x"] = row
            job = {"job_id": "x", "meta": row["meta"], "params": row["params"]}
            # Cached jobs do not call a server, gate, or network.
            self.assertEqual(runner.run(job), row)
            bad = deepcopy(job)
            bad["meta"]["prompt_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "cached request"):
                runner.run(bad)
            bad = deepcopy(job)
            bad["params"]["temperature"] = .5
            with self.assertRaisesRegex(ValueError, "cached request"):
                runner.run(bad)

    def test_other_judge_does_not_claim_a_generator_tokenizer_count(self):
        work = self.context()
        with algorithms(work) as a:
            Runner = R.runner_type(a, work, "gptoss", "f" * 64)
            runner = Runner(SimpleNamespace(), str(work.base / "journals/test.jsonl"))
            row = {"job_id": "x", "terminal": True, "ok": True, "text": "yes",
                   "completion_tokens_harness": 99, "usage": {"completion_tokens": 15}}
            runner._append(row)
            self.assertIsNone(row["completion_tokens_harness"])
            self.assertEqual(row["usage"]["completion_tokens"], 15)

    def test_dry_run_cannot_start_inference_or_preflight(self):
        work = self.context()
        work.load = lambda: ({"model": "qwen3-8b", "completion_cap": 8192}, [])
        work.frozen_prompts = lambda: [1, 2, 3]
        with patch.object(R, "preflight", side_effect=AssertionError("preflight invoked")), \
             patch.object(R, "artifacts", side_effect=AssertionError("model hashing invoked")):
            result = R.execute(work, "generate")
            self.assertFalse(result["inference_started"])
            self.assertEqual(result["prompt_count"], 3)
            self.assertEqual(result["idle_seconds"], 10.)

    def test_live_run_refuses_incomplete_primary_before_gpu_checks(self):
        work = self.context()
        work.load = lambda: ({"model": "qwen3-8b", "completion_cap": 8192}, [])
        work.frozen_prompts = lambda: []
        with patch.object(R, "verified_primary", side_effect=ValueError("primary incomplete")), \
             patch.object(R, "preflight", side_effect=AssertionError("GPU checks invoked")):
            with self.assertRaisesRegex(ValueError, "primary incomplete"):
                R.execute(work, "generate", live=True)


if __name__ == "__main__":
    unittest.main()
