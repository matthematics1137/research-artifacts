"""Cross-model S summary supplement (research/P5_AMENDMENT_2026-10-30.md), with no servers.

Only sessions the primary never summarized are requested, by the primary generator,
with the primary's own summary request; selection mirrors build_S with the leg's
tokenizer; S preparation merges the supplement only after a verified finished stage.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jobs
from legacy import algorithms
import prepare as PR
import run as R
from test_integration import FakeServer
from test_workspace import items_for
from workspace import ROOT, Workspace, sha


def multi_session_items():
    items = items_for("cross-model")
    first = items[0]
    first.update(haystack_sessions=[[{"role": "user", "content": f"synthetic fact {k}"}] for k in range(3)],
                 haystack_dates=["2025/11/01", "2025/11/15", "2025/12/01"],
                 haystack_session_ids=["s0a", "s0b", "s0c"])
    return items


class SupplementTests(unittest.TestCase):
    def test_summary_job_is_the_primary_request_for_both_legs(self):
        for leg in ("cross-model", "locomo"):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                (base / "WORKSPACE.json").write_text("{}\n")
                work = SimpleNamespace(root=ROOT, base=base, leg=leg, path=lambda name, b=base: b / name)
                with algorithms(work) as a:
                    a.C.count_tokens = lambda text, *args: 25
                    item = items_for(leg)[0]
                    job = jobs.summary_job(a, work, item, 0)
                    primary = a.SM.summary_job(item, 0)
                    self.assertEqual(job["job_id"], primary["job_id"])
                    self.assertEqual(job["prompt"], primary["prompt"])
                    self.assertEqual(job["params"], primary["params"])
                    self.assertEqual(job["meta"]["model"], "qwen38-exl3")    # the summarizer, never the 8B subject

    def test_summaries_use_the_primary_generator_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = SimpleNamespace(root=ROOT, base=Path(tmp), leg="cross-model")
            with algorithms(work) as a:
                self.assertEqual(R.server_for(a, "summaries", {"model": "qwen3-8b"}).name, "tabby-p5")
                self.assertNotEqual(R.server_for(a, "generate", {"model": "qwen3-8b"}).name, "tabby-p5")

    def test_vram_floor_is_adjustable_within_bounds(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("P5_MIN_FREE_VRAM_MIB", None)
            self.assertEqual(R.min_free_vram_mib(), 11000)
            os.environ["P5_MIN_FREE_VRAM_MIB"] = "10200"
            self.assertEqual(R.min_free_vram_mib(), 10200)
            for bad in ("9000", "13000"):
                os.environ["P5_MIN_FREE_VRAM_MIB"] = bad
                with self.assertRaises(ValueError):
                    R.min_free_vram_mib()

    def test_supplement_requests_only_missing_sessions_and_s_merges_after_finish(self):
        with tempfile.TemporaryDirectory(prefix="p5-followup-summary-supplement-") as tmp:
            root = Path(tmp)
            source = root / "synthetic_source.json"
            source.write_text('{"synthetic":true}\n')
            inputs = {str(source): sha(source)}
            items = multi_session_items()
            work = Workspace("cross-model", root)
            work.initialize(items, 8192, inputs)
            requests = []

            def fake_http(url, body, timeout):
                requests.append(body)
                return {"choices": [{"text": "SUMMARY generated", "finish_reason": "stop"}], "usage": {"prompt_tokens": 25}}

            @contextmanager
            def fake_algorithms(ws):
                with algorithms(SimpleNamespace(root=ROOT, base=ws.base, leg=ws.leg)) as a:
                    a.C.count_tokens = lambda text, *args: 25 if "SUMMARY" not in text else 3000 * text.count("SUMMARY")
                    a.runner.http_json = fake_http
                    a.runner.gpu_temp = lambda: 0
                    a.runner.log = lambda text: None
                    yield a

            with fake_algorithms(work) as a:
                # The primary summarized only the newest session of every item.
                rows = []
                for it in items:
                    newest = a.C.chronological(it)[-1]
                    rows.append({"job_id": a.SM.summary_job(it, newest)["job_id"], "terminal": True, "ok": True,
                                 "text": "SUMMARY primary"})
                older = a.SM.summary_job(items[0], a.C.chronological(items[0])[-2])["job_id"]
                oldest = a.SM.summary_job(items[0], a.C.chronological(items[0])[0])["job_id"]
            journal = root / "testsuite/paper5/work/journals/summaries.jsonl"
            journal.parent.mkdir(parents=True)
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))

            def fake_artifacts(a, stage, info):
                return {"served_model": "qwen38-exl3" if stage == "summaries" else info["model"],
                        "engine": "synthetic", "revision": "synthetic", "files": inputs}

            with patch.object(R, "algorithms", fake_algorithms), \
                 patch.object(R, "verified_primary", return_value=({}, inputs)), \
                 patch.object(R, "preflight", return_value={"host": "synthetic-no-hardware"}), \
                 patch.object(R, "artifacts", fake_artifacts), \
                 patch.object(R, "server_for", side_effect=lambda *args: FakeServer()):
                result = R.execute(work, "summaries", live=True, idle_seconds=0.)
            self.assertTrue(result["complete"])
            # Item 0: newest from the primary (3000 "tokens" fits), the next older is missing and
            # requested; with two summaries (6000) the budget is exceeded, so the oldest is never asked.
            self.assertEqual(len(requests), 1)
            supplement = {r["job_id"] for r in map(json.loads, work.path("journals/summaries.jsonl").read_text().splitlines())}
            self.assertEqual(supplement, {older})
            self.assertNotIn(oldest, supplement)
            done = json.loads(work.path("execution/summaries-finished.json").read_text())
            self.assertEqual(done["journal_sha256"], sha(work.path("journals/summaries.jsonl")))

            # S preparation merges the supplement only with its verified finished receipt.
            with patch.object(PR, "algorithms", fake_algorithms), \
                 patch.object(PR, "verified_primary", return_value=({}, {})):
                receipt = work.path("execution/summaries-finished.json")
                good = receipt.read_text()
                receipt.write_text(json.dumps({**json.loads(good), "journal_sha256": "0" * 64}))
                with self.assertRaisesRegex(ValueError, "verified stage"):
                    PR.prepare(work, "S", 1)
                receipt.write_text(good)
                PR.prepare(work, "S", 1)
            first = json.loads(work.path("prompts/S_4096.jsonl").read_text().splitlines()[0])
            self.assertIn("SUMMARY primary", first["prompt"])
            self.assertNotIn("SUMMARY generated", first["prompt"])  # 3000 + 3000 > 4096: only the newest fits


if __name__ == "__main__":
    unittest.main()
