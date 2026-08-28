"""Small mutation tests for the corrected release's evidence checks."""
import copy
import unittest
from unittest.mock import patch

import check_claims
import correction_metrics as metrics


class CorrectionTests(unittest.TestCase):
    def altered_rows(self, target, mutation):
        original = metrics.rows

        def read(path):
            result = copy.deepcopy(original(path))
            if str(path).endswith(target):
                mutation(result)
            return result
        return patch.object(metrics, "rows", side_effect=read)

    def test_frozen_evidence_passes(self):
        metrics.verify()

    def test_mtp_alignment_rejects_changed_token_count(self):
        with self.altered_rows(f"{metrics.LABEL}/math25.jsonl", lambda r: r[0].update(completion_tokens=1)):
            with self.assertRaisesRegex(ValueError, "sequence differs"):
                metrics.verify()

    def test_q4_timeout_is_not_truncation(self):
        with self.altered_rows("UD-Q4_K_XL-ngl33-effmed/math25.jsonl", lambda r: r[14].update(truncated=True)):
            with self.assertRaisesRegex(ValueError, "classification differs"):
                metrics.verify()

    def test_retry_time_stays_in_denominator(self):
        with self.altered_rows("UD-Q4_K_XL-ngl33-effmed/math25.jsonl", lambda r: r[14].update(wall_s=600)):
            with self.assertRaisesRegex(ValueError, "retry denominator differs"):
                metrics.verify()

    def test_gsm_selection_is_exact(self):
        with self.altered_rows(f"{metrics.LABEL}/gsm8k.jsonl", lambda r: r.reverse()):
            with self.assertRaisesRegex(ValueError, "selection differs"):
                metrics.verify()

    def test_server_caption_bound(self):
        def alter(result):
            next(r for r in result if r.get("stage") == "servertest" and r.get("mtp") == "off")["tg_tok_s"] = 100
        with self.altered_rows("results/phase1.jsonl", alter):
            with self.assertRaisesRegex(ValueError, "caption bound differs"):
                metrics.verify()

    def test_duplicate_math_id_rejected(self):
        original = check_claims.load_jsonl
        def read(path):
            result = copy.deepcopy(original(path))
            if path.parent.name == metrics.LABEL:
                result[0]["id"] = result[1]["id"]
            return result
        with patch.object(check_claims, "load_jsonl", side_effect=read):
            with self.assertRaisesRegex(AssertionError, "item IDs differ"):
                check_claims.math_scores()

    def test_string_boolean_rejected(self):
        original = check_claims.load_jsonl
        def read(path):
            result = copy.deepcopy(original(path))
            if path.parent.name == metrics.LABEL:
                result[0]["correct"] = "false"
            return result
        with patch.object(check_claims, "load_jsonl", side_effect=read):
            with self.assertRaisesRegex(AssertionError, "booleans are not typed"):
                check_claims.math_scores()


if __name__ == "__main__":
    unittest.main()
