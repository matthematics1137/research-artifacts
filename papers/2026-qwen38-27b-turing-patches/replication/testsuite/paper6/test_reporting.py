"""CPU-only tests of conclusion wording; no real journals are opened."""
import json
import math
import unittest

from analyze import equivalence_conclusion, tost


class EquivalenceReportingTests(unittest.TestCase):
    def test_zero_variance_is_valid_json_with_unchanged_decision(self):
        result = tost([0] * 200)
        self.assertIsNone(result["t_lower"])
        self.assertIsNone(result["t_upper"])
        self.assertTrue(result["zero_standard_error"])
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["p_tost"], 0)
        json.dumps(result, allow_nan=False)

    def test_raw_tost_pass_is_not_a_family_pass(self):
        result = {"p_holm": .08, "reject_at_0.05": False,
                  "tost": {"p_tost": .04, "equivalent": True}}
        self.assertEqual(equivalence_conclusion(result), "equivalence not established (Holm)")

    def test_adjusted_pass(self):
        self.assertEqual(equivalence_conclusion({"p_holm": .02, "reject_at_0.05": True}),
                         "equivalent within the specified margin (Holm)")

    def test_no_measurements(self):
        self.assertEqual(equivalence_conclusion({"p_holm": None}), "not evaluable")

    def test_boundary_matches_analysis(self):
        self.assertEqual(equivalence_conclusion({"p_holm": .05, "reject_at_0.05": False}),
                         "equivalence not established (Holm)")

    def test_inconsistent_decision_rejected(self):
        with self.assertRaises(ValueError):
            equivalence_conclusion({"p_holm": .08, "reject_at_0.05": True})

    def test_bad_p_rejected(self):
        for p in (math.nan, math.inf, -.1, 1.1, True, "0.01"):
            with self.subTest(p=p), self.assertRaises(ValueError):
                equivalence_conclusion({"p_holm": p})


if __name__ == "__main__":
    unittest.main()
