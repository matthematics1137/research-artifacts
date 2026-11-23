"""CPU tests for P8 long-context stage 1: configs, judging, budget estimate, failure text, interruptions.

  engines/analysis-env/bin/python -B -m unittest discover -s testsuite/p8_longctx -p test_stage1.py -v

Temporary files go to tempfile.mkdtemp() and are left in place (nothing here deletes files).
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_config as C  # noqa: E402
import stage1_run as R  # noqa: E402
import stage1_spec as S  # noqa: E402
import stage1_summarize as J  # noqa: E402

K = S.K


def result(fmt, length, verdict, peak, outcome=None, **kw):
    outcome = outcome or ("completed" if verdict == "pass" else "oom")
    row = {"kind": "result", "step": S.step_id(fmt, length), "format": fmt, "length": length, "judged": True,
           "verdict": verdict, "outcome": outcome, "peak_mib": peak, "wall_s": 300.0, "t_start": 0.0}
    row.update(kw)
    return row


def as_predicted():
    rows = [{"kind": "result", "step": "check-88-16k", "judged": False, "verdict": "check_ok",
             "outcome": "completed", "wall_s": 60.0, "t_start": 0.0}]
    for (fmt, length), (lo, hi, pred) in S.PROJECTED.items():
        mid = (lo + hi) // 2
        if pred == "pass":
            rows.append(result(fmt, length, "pass", mid))
        else:
            rows.append(result(fmt, length, "fail", hi + 100, outcome="oom"))
    order = {f: i for i, (f, _) in enumerate(S.LADDER)}
    return sorted(rows, key=lambda r: (order.get(r.get("format"), -1), r.get("length", 0)))


class TestConfig(unittest.TestCase):
    def test_render_and_check_every_step(self):
        for fmt, ladder in S.LADDER:
            for length in ladder:
                text = C.render(fmt, length, 8097, "/x/models")
                self.assertEqual(C.check(text, fmt, length, 8097), [], (fmt, length))
                self.assertIn(f'  cache_mode: "{S.CACHE_MODE[fmt]}"', text)
                self.assertIn(f'  draft_cache_mode: "{S.CACHE_MODE[fmt]}"', text)

    def test_check_finds_a_missing_line(self):
        text = C.render("4,4", 128 * K, 8097, "/x").replace("  sysmem_recurrent_cache: 2048", "")
        self.assertEqual(C.check(text, "4,4", 128 * K, 8097), ["  sysmem_recurrent_cache: 2048"])

    def test_render_refuses_bad_input(self):
        with self.assertRaises(ValueError):
            C.render("6,6", 64 * K, 8097, "/x")
        with self.assertRaises(ValueError):
            C.render("8,8", 64 * K + 100, 8097, "/x")

    def test_spec_matches_the_registration(self):
        self.assertEqual(S.PASS_LIMIT_MIB, 10752)
        self.assertEqual([f for f, _ in S.LADDER], ["8,8", "4,4", "8,4", "fp16"])
        self.assertEqual(len(S.PROJECTED), 13)
        self.assertEqual(S.lengths(), [x * K for x in (32, 36, 40, 64, 72, 80, 88, 96, 128, 136)])


class TestJudge(unittest.TestCase):
    def test_everything_as_predicted(self):
        res = J.judge(as_predicted())
        self.assertEqual(res["formats"]["8,8"]["reach"], 72 * K)
        self.assertEqual(res["formats"]["4,4"]["reach"], 128 * K)
        self.assertEqual(res["formats"]["8,4"]["reach"], 88 * K)
        self.assertEqual(res["formats"]["fp16"]["reach"], 36 * K)
        self.assertEqual(res["verdicts"], {"P1": "holds", "P2": "holds", "P3": "holds", "K1": "not triggered",
                                           "B1": "not triggered"})
        self.assertTrue(all(s["prediction_right"] for s in res["steps"]))

    def test_premise_stands_and_4bit_short(self):
        rows = [result("8,8", 64 * K, "fail", 11200),
                result("4,4", 64 * K, "pass", 9300), result("4,4", 96 * K, "fail", 11100)]
        v = J.judge(rows)["verdicts"]
        self.assertEqual((v["P2"], v["B1"], v["P3"], v["K1"]), ("refuted", "triggered", "refuted", "triggered"))
        self.assertEqual(v["P1"], "holds")  # only the 4,4 64K step completed, and it is in band
        rows[1]["peak_mib"] = 9000  # 9,237 - 150 = 9,087: out of band
        self.assertEqual(J.judge(rows)["verdicts"]["P1"], "refuted for 4,4")

    def test_p1_counts_only_completed_steps(self):
        rows = [result("8,8", 64 * K, "pass", 10400), result("8,8", 72 * K, "fail", 9000, outcome="oom")]
        res = J.judge(rows)
        self.assertEqual(res["formats"]["8,8"]["p1"], "holds")
        self.assertIsNone(res["steps"][1]["p1_in_band"])

    def test_budget_stop_leaves_later_formats_open(self):
        rows = [result("8,8", 64 * K, "pass", 10400), result("8,8", 72 * K, "pass", 10700),
                result("8,8", 80 * K, "fail", 11000, outcome="completed"),
                result("4,4", 64 * K, "pass", 9300), result("4,4", 96 * K, "pass", 9900),
                result("4,4", 128 * K, "pass", 10500),
                {"kind": "budget_stop", "next_step": "44-136k", "used_s": 5000, "estimate_s": 600}]
        res = J.judge(rows)
        self.assertEqual(res["verdicts"]["P3"], "holds")
        self.assertEqual(res["verdicts"]["K1"], "not triggered")
        self.assertFalse(res["formats"]["4,4"]["ladder_finished"])
        self.assertEqual(res["formats"]["8,4"]["steps_run"], [])
        self.assertEqual(res["end"]["kind"], "budget_stop")

    def test_markdown_tables_render(self):
        md = J.markdown(J.judge(as_predicted()))
        self.assertIn("| `4,4` 128K |", md)
        self.assertIn("| P2 | holds |", md)
        self.assertEqual(md.count("\n| `"), 13 + 4)

    def test_peak_over_limit_fails_even_when_completed(self):
        rows = [result("fp16", 32 * K, "fail", 10800, outcome="completed")]
        res = J.judge(rows)
        self.assertEqual(res["formats"]["fp16"]["first_failure"], 32 * K)
        self.assertFalse(res["steps"][0]["prediction_right"])


class TestRates(unittest.TestCase):
    def test_rows_follow_the_results(self):
        import stage1_rates as M
        rows = M.rows_from(J.judge(as_predicted()), "2026-11-01")
        ids = {r["id"] for r in rows}
        self.assertIn("p8lc-s1-reach-88", ids)
        self.assertIn("p8lc-s1-slope-44", ids)
        self.assertIn("p8lc-s1-peak-44-128k", ids)
        self.assertFalse(any(r["status"] == "refuted" for r in rows))
        reach = next(r for r in rows if r["id"] == "p8lc-s1-reach-44")
        self.assertEqual((reach["value"], reach["unit"], reach["gets"]), (128 * K, "tokens", "capability"))
        for r in rows:
            self.assertEqual(set(r) >= {"id", "conversion", "gives", "gets", "quantity", "value", "unit", "ci95",
                                        "n", "operating_point", "status", "evidence", "prereg", "rife", "date",
                                        "supersedes", "note"}, True)

    def test_refuted_prediction_gets_a_row(self):
        import stage1_rates as M
        rows = M.rows_from(J.judge([result("8,8", 64 * K, "fail", 11200)]), "2026-11-01")
        p2 = next(r for r in rows if r["id"] == "p8lc-s1-p2-refuted")
        self.assertIsNone(p2["value"])
        self.assertEqual(p2["status"], "refuted")


class TestRunnerPieces(unittest.TestCase):
    def test_oom_text(self):
        self.assertEqual(R.oom_in("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"), "out of memory")
        self.assertEqual(R.oom_in("RuntimeError: CUDA_ERROR_OUT_OF_MEMORY"), "cuda_error_out_of_memory")
        self.assertIsNone(R.oom_in("ValueError: bad cache_mode"))

    def test_estimate_scales_the_last_prefill(self):
        d = tempfile.mkdtemp(prefix="p8lc-test-")
        j = R.Journal(os.path.join(d, "journal.jsonl"))
        self.assertAlmostEqual(R.estimate_s(j, 30000), 60 + 30000 / 624 + 120, places=6)
        j.add(result("8,8", 64 * K, "pass", 10400, prompt_time_s=100.0, prompt_tokens=65000, load_s=20.0))
        want = 20.0 + 100.0 * (98304 / 65000) ** 1.5 + 120
        self.assertAlmostEqual(R.estimate_s(j, 98304), want, places=6)

    def test_interrupted_step_is_counted_once(self):
        d = tempfile.mkdtemp(prefix="p8lc-test-")
        csv_path = os.path.join(d, "s.csv")
        with open(csv_path, "w") as f:
            f.write("t_unix,smi_time,mem_used_mib,temp_c,power_w,sm_mhz,server_rss_mib\n")
            f.write("1000.0,x,5000,50,100,1800,3000\n1042.5,x,9000,60,200,1800,3100\n")
        j = R.Journal(os.path.join(d, "journal.jsonl"))
        j.add({"kind": "start", "step": "88-64k", "t_start": 1000.0, "samples": csv_path})
        R.mark_interrupted(j)
        R.mark_interrupted(j)
        inter = [r for r in j.rows if r["kind"] == "interrupted"]
        self.assertEqual(len(inter), 1)
        self.assertEqual(inter[0]["span_s"], 42.5)
        self.assertEqual(R.elapsed_s(j), 42.5)
        reread = R.Journal(os.path.join(d, "journal.jsonl"))
        self.assertEqual(len(reread.rows), 2)

    def test_server_env_drops_allocator_overrides(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        os.environ["LD_LIBRARY_PATH"] = "/stray"
        try:
            env = R.server_env("/x/logs")
        finally:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
            os.environ.pop("LD_LIBRARY_PATH")
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
        self.assertNotIn("LD_LIBRARY_PATH", env)
        self.assertEqual(env["EXL3_GEMV_SK"], "2")
        self.assertTrue(env["PYTHONPATH"].endswith("/overlay/exl3-w13"))


if __name__ == "__main__":
    unittest.main()
