"""Hand-computable synthetic follow-up results; no real measurements or files."""
from copy import deepcopy
import math
import unittest

import analyze as A
from workspace import LEGS, canonical, digest_text, namespace_prompt


def fixture(leg="locomo"):
    arms = ("T", "S") if leg == "locomo" else ("T", "S", "C", "M")
    items = [{"question_id": f"synthetic-{i}", "answer": "red fox", "conversation_id": f"conv-{i // 2}"} for i in range(4)]
    prompts, generations = [], {}
    judges = {label: {} for label in (("gptoss",) if leg == "locomo" else ("gptoss", "self"))}
    for arm in arms:
        for index, item in enumerate(items):
            text = "synthetic prompt " + arm + item["question_id"]
            p = namespace_prompt(leg, {"question_id": item["question_id"], "arm": arm, "budget": 4096,
                                      "n_prompt": 20, "prompt": text, "prompt_sha256": digest_text(text)})
            prompts.append(p)
            # S/C/M are always correct; T is correct on the first two items.
            correct = arm != "T" or index < 2
            g = {"job_id": p["job_id"], "terminal": True, "ok": True,
                 "text": "reasoning</think>" + ("red fox" if correct else "blue bird"),
                 "prompt_sha256": p["prompt_sha256"], "server_prompt_tokens": 20,
                 "meta": {"leg": leg, "model": LEGS[leg]["model"], "prompt_manifest_sha256": "m" * 64,
                          "prompt_sha256": p["prompt_sha256"]}}
            generations[p["job_id"]] = g
            for label in judges:
                jid = p["job_id"] + "|judge=" + label
                judges[label][jid] = {"job_id": jid, "terminal": True, "ok": True, "text": "yes" if correct else "no",
                                      "meta": {"gen_job_id": p["job_id"], "judge": label,
                                               "gen_record_sha256": digest_text(canonical(g))}}
    return leg, arms, items, prompts, generations, judges, "m" * 64


class AnalysisTests(unittest.TestCase):
    def analyze(self, f):
        return A.analyze_records(*f, bootstrap_draws=100)

    def test_hand_computed_locomo(self):
        result = self.analyze(fixture())
        self.assertEqual(result["cells"]["T"]["correct"], 2)
        self.assertEqual(result["cells"]["S"]["correct"], 4)
        contrast = result["contrasts_vs_T"]["S"]
        self.assertEqual(contrast["difference"], .5)
        self.assertEqual(contrast["arm_only"], 2)
        self.assertEqual(contrast["mcnemar_p_unadjusted"], .5)
        self.assertEqual(result["token_f1"]["T"]["mean"], .5)
        self.assertEqual(result["token_f1_differences_vs_T"]["S"]["mean"], .5)
        self.assertEqual(result["token_f1"]["S"]["percentile95"], [1., 1.])
        self.assertEqual(result["token_f1"]["T"]["conversations"], 2)
        self.assertEqual(result["publication"], "aggregates-only")
        self.assertFalse(result["confirmatory"])

    def test_cross_model_two_judges_and_all_arms(self):
        result = self.analyze(fixture("cross-model"))
        self.assertEqual(set(result["contrasts_vs_T"]), {"S", "C", "M"})
        self.assertEqual(result["cells"]["T"]["judge_agreement"], {"n": 4, "raw": 1.0})
        self.assertNotIn("token_f1", result)

    def test_unrun_or_extra_or_duplicate_never_becomes_incorrect(self):
        for mutate in (lambda f: f[4].pop(next(iter(f[4]))),
                       lambda f: f[3].append(f[3][0]),
                       lambda f: f[4].update(extra=next(iter(f[4].values()))),
                       lambda f: f[5]["gptoss"].clear()):
            f = fixture()
            mutate(f)
            with self.assertRaises(ValueError):
                self.analyze(f)

    def test_real_terminal_failure_is_incorrect_not_unrun(self):
        f = fixture()
        gid = next(iter(f[4]))
        f[4][gid]["ok"] = False
        del f[5]["gptoss"][gid + "|judge=gptoss"]
        result = self.analyze(f)
        self.assertEqual(result["cells"]["T"]["failed_generations"], 1)
        self.assertEqual(result["cells"]["T"]["correct"], 1)

    def test_wrong_prompt_model_parity_or_manifest_rejected(self):
        for mutate in (lambda r: r.update(server_prompt_tokens=21),
                       lambda r: r.update(prompt_sha256="x" * 64),
                       lambda r: r["meta"].update(model="wrong"),
                       lambda r: r["meta"].update(prompt_manifest_sha256="old")):
            f = fixture()
            mutate(next(iter(f[4].values())))
            with self.assertRaises(ValueError):
                self.analyze(f)

    def test_stale_judge_response_rejected(self):
        f = fixture()
        next(iter(f[4].values()))["text"] = "changed after judging</think>different"
        with self.assertRaisesRegex(ValueError, "different generation"):
            self.analyze(f)

    def test_empty_judge_retry_required_then_honored(self):
        f = fixture()
        base = next(iter(f[5]["gptoss"].values()))
        base["text"] = ""
        with self.assertRaisesRegex(ValueError, "declared retry"):
            self.analyze(f)
        retry = deepcopy(base)
        retry.update(job_id=base["job_id"] + "|retry", text="yes")
        f[5]["gptoss"][retry["job_id"]] = retry
        self.assertEqual(self.analyze(f)["cells"]["T"]["correct"], 2)
        retry["text"] = ""
        self.assertEqual(self.analyze(f)["cells"]["T"]["judge_failures"], 1)

    def test_extra_retry_cannot_override_valid_judgment(self):
        f = fixture()
        base = next(iter(f[5]["gptoss"].values()))
        retry = deepcopy(base)
        retry.update(job_id=base["job_id"] + "|retry", text="no")
        f[5]["gptoss"][retry["job_id"]] = retry
        with self.assertRaisesRegex(ValueError, "retry does not follow"):
            self.analyze(f)

    def test_normalizer_visible_answer_and_f1(self):
        self.assertEqual(A.token_f1("The red fox!", "red fox"), 1.0)
        self.assertAlmostEqual(A.token_f1("red", "red fox"), 2 / 3)
        self.assertEqual(A.visible_answer("unfinished thinking"), ("", False))
        self.assertEqual(A.visible_answer("think</think> answer "), ("answer", True))
        self.assertEqual(A.token_f1("", "red fox"), 0.)

    def test_bootstrap_reproducible_and_cluster_weighted(self):
        values = {"a": 0., "b": 1., "c": 1.}
        clusters = {"a": "one", "b": "two", "c": "two"}
        got = A.clustered_f1(values, clusters, draws=100)
        self.assertEqual(got, A.clustered_f1(values, clusters, draws=100))
        self.assertAlmostEqual(got["mean"], 2 / 3)
        self.assertEqual(got["conversations"], 2)
        for bad in ({"a": math.nan, "b": 1., "c": 1.}, {"a": 0., "b": True, "c": 1.}):
            with self.assertRaises(ValueError):
                A.clustered_f1(bad, clusters, draws=100)


if __name__ == "__main__":
    unittest.main()
