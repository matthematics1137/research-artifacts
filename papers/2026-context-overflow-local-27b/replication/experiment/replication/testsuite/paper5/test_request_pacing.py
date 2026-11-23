"""CPU-only pacing/resume regressions; no model, HTTP, or GPU is accessed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import p5run


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def read(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Server:
    base = "http://unused.invalid"

    def __init__(self):
        self.stops = 0

    def identity(self):
        return {"server": "fake"}

    def stop(self):
        self.stops += 1


def job(name):
    return {"job_id": name, "prompt": "unchanged prompt", "n_prompt": 2,
            "budget": None, "params": {"temperature": 0, "max_tokens": 200},
            "meta": {"session_id": "fake"}}


class RequestPacingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = os.path.join(self.tmp.name, "summaries.jsonl")
        self.clock = Clock()
        self.server = Server()
        self.requests = []
        self.failures = 0
        for target, kwargs in (
            ("p5run.time.monotonic", {"side_effect": self.clock.read}),
            ("p5run.time.time", {"side_effect": self.clock.read}),
            ("p5run.time.sleep", {"side_effect": self.clock.sleep}),
            ("p5run.http_json", {"side_effect": self.http}),
            ("p5run.Runner.gate", {"return_value": None}),
            ("p5run.C.count_tokens", {"return_value": 1}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def http(self, url, body, timeout):
        self.requests.append(dict(body))
        self.clock.now += 2
        if self.failures:
            self.failures -= 1
            raise OSError("simulated request failure")
        return {"choices": [{"text": "ok", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

    def runner(self, **kwargs):
        return p5run.Runner(self.server, self.journal, **kwargs)

    def rows(self):
        return [json.loads(line) for line in Path(self.journal).read_text().splitlines()]

    def test_default_does_not_pace(self):
        runner = self.runner()
        runner.run(job("one"))
        runner.run(job("two"))
        self.assertEqual(self.clock.sleeps, [])
        self.assertEqual(runner.request_idle_seconds, 0)

    def test_environment_and_explicit_override(self):
        with patch.dict(os.environ, {"P5_REQUEST_IDLE_SECONDS": "10"}):
            self.assertEqual(self.runner().request_idle_seconds, 10)
            self.assertEqual(self.runner(idle_seconds=0).request_idle_seconds, 0)

    def test_invalid_pacing_rejected_before_creating_journal(self):
        for value in (-1, float("nan"), float("inf"), float("-inf"), "bad", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.runner(idle_seconds=value)
        self.assertFalse(Path(self.journal).exists())

    def test_idle_gap_excluded_from_request_seconds(self):
        runner = self.runner(idle_seconds=10)
        first = runner.run(job("one"))
        second = runner.run(job("two"))
        self.assertEqual(self.clock.sleeps, [10])
        self.assertEqual(first["seconds"], 2)
        self.assertEqual(second["seconds"], 2)
        self.assertEqual(first["idle_wait_seconds"], 0)
        self.assertEqual(second["idle_wait_seconds"], 10)
        self.assertEqual(second["request_idle_seconds"], 10)
        self.assertEqual(self.clock.now, 114)

    def test_cached_jobs_never_wait_or_send_again(self):
        runner = self.runner(idle_seconds=10)
        first = runner.run(job("one"))
        before = Path(self.journal).read_bytes()
        self.assertEqual(runner.run(job("one")), first)
        self.assertEqual(Path(self.journal).read_bytes(), before)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.clock.sleeps, [])

    def test_restart_preserves_legacy_rows_byte_for_byte(self):
        legacy = {"job_id": "old", "terminal": True, "ok": True,
                  "text": "already saved", "harness": {"old": "hash"}}
        original = (json.dumps(legacy) + "\n").encode()
        Path(self.journal).write_bytes(original)
        runner = self.runner(idle_seconds=10)
        self.assertEqual(runner.run(job("old")), legacy)
        self.assertEqual(self.requests, [])
        runner.run(job("new"))
        self.assertTrue(Path(self.journal).read_bytes().startswith(original))
        self.assertEqual([row["job_id"] for row in self.rows()], ["old", "new"])

    def test_time_spent_between_jobs_counts_toward_gap(self):
        runner = self.runner(idle_seconds=10)
        runner.run(job("one"))
        self.clock.now += 7
        runner.run(job("two"))
        self.assertEqual(self.clock.sleeps, [3])

    def test_no_extra_wait_after_long_operator_gap(self):
        runner = self.runner(idle_seconds=10)
        runner.run(job("one"))
        self.clock.now += 30
        runner.run(job("two"))
        self.assertEqual(self.clock.sleeps, [])

    def test_retries_are_paced_and_journaled(self):
        runner = self.runner(idle_seconds=10)
        self.failures = 1
        result = runner.run(job("one"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempt"], 1)
        self.assertEqual(self.clock.sleeps, [10])
        failed, complete = self.rows()
        self.assertFalse(failed["terminal"])
        self.assertEqual(failed["request_idle_seconds"], 10)
        self.assertEqual(complete["idle_wait_seconds"], 10)
        self.assertEqual(complete["seconds"], 2)

    def test_exhausted_retries_keep_terminal_failure(self):
        runner = self.runner(idle_seconds=10)
        self.failures = 3
        result = runner.run(job("one"))
        self.assertFalse(result["ok"])
        self.assertTrue(result["terminal"])
        self.assertEqual(self.clock.sleeps, [10, 10])
        self.assertEqual(len(self.rows()), 4)
        runner.run(job("one"))
        self.assertEqual(len(self.requests), 3)

    def test_pacing_does_not_change_request_or_prompt(self):
        original = job("one")
        runner = self.runner(idle_seconds=10)
        row = runner.run(original)
        self.assertEqual(self.requests[0], {"prompt": "unchanged prompt", "stream": False,
                                            "temperature": 0, "max_tokens": 200})
        self.assertEqual(row["params"], original["params"])
        self.assertEqual(row["meta"], original["meta"])
        self.assertEqual(row["prompt_sha256"], p5run.C.sha256_text(original["prompt"]))


if __name__ == "__main__":
    unittest.main()
