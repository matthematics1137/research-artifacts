from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from legacy import algorithms
from prepare import build_records
from test_workspace import items_for
from workspace import ROOT


class PreparationTests(unittest.TestCase):
    def test_truncation_body_unchanged_for_both_models(self):
        for leg in ("locomo", "cross-model"):
            with tempfile.TemporaryDirectory() as tmp:
                work = SimpleNamespace(root=ROOT, base=Path(tmp), leg=leg)
                with algorithms(work) as a:
                    a.C.count_tokens = lambda text, *args: len(text)
                    items = items_for(leg)[:2]
                    expected = [a.BP.build_T(it, 4096) for it in items]
                    actual = build_records(a, leg, "T", items)
                    for old, new in zip(expected, actual):
                        self.assertEqual(old["prompt"], new["prompt"])
                        self.assertEqual(old["n_prompt"], new["n_prompt"])
                        self.assertNotEqual(old["job_id"], new["job_id"])
                    self.assertFalse(any(Path(tmp).iterdir()))

    def test_summary_requires_real_matching_cache_and_keeps_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = SimpleNamespace(root=ROOT, base=Path(tmp), leg="cross-model")
            with algorithms(work) as a:
                a.C.count_tokens = lambda text, *args: len(text)
                item = items_for("cross-model")[0]
                with self.assertRaisesRegex(ValueError, "requires"):
                    build_records(a, "cross-model", "S", [item])
                key = a.SM.summary_key(item["haystack_session_ids"][0], a.SM.session_text(item["haystack_sessions"][0]))
                cache = {key: "Synthetic summary."}
                result = build_records(a, "cross-model", "S", [item], summaries=cache)
                self.assertIn("Synthetic summary.", result[0]["prompt"])
                self.assertEqual(cache, {key: "Synthetic summary."})
                with self.assertRaises(SystemExit):
                    build_records(a, "cross-model", "S", [item], summaries={})


if __name__ == "__main__":
    unittest.main()
