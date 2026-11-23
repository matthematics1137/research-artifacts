"""CPU tests for P8 long-context stages 2-3: item wrapper, statistics, runner pieces.

  HOME=<folder holding models/qwen38-exl3-2.0/tokenizer.json> \
  engines/exl3-env/bin/python -B -m unittest discover -s testsuite/p8_longctx -p test_stage2.py -v

The item tests need the model's tokenizer.json (the generator checks its SHA-256) and LongMemEval-S; without the
tokenizer they are skipped. Temporary files go to tempfile.mkdtemp() and are left in place.
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage2_stats as T  # noqa: E402

TOKENIZER = os.path.expanduser("~/models/qwen38-exl3-2.0/tokenizer.json")


@unittest.skipUnless(os.path.exists(TOKENIZER), "needs the model's tokenizer.json")
class TestItems(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import items as I
        cls.I = I
        cls.pool = I.pool()

    def test_overrides_are_restored(self):
        I = self.I
        state = lambda: (I.G.SEED, I.G.TARGET_TOKENS, I.G.TOL_LO, I.G.TOL_HI, I.G.N_PER_FAMILY, dict(I.G.N_NEEDLES))
        before = state()
        I.build_one(self.pool, "test", "multivalue", 24, 70, 8192, 96)
        self.assertEqual(before, state())
        self.assertEqual((before[0], before[4]), (20260919, 50))
        self.assertEqual(before[5], {"single": 1, "multikey": 4, "multivalue": 4, "vartrack": 5})

    def test_graded_multivalue_on_the_96_item_grid(self):
        I = self.I
        it = I.build_one(self.pool, "test", "multivalue", 48, 95, 8192, 96)
        self.assertEqual((it["uid"], len(it["answers"]), it["n_per_family"]), ("multivalue-k48-95", 48, 96))
        self.assertTrue(all(0 < d < 1 for d in it["depths"]))
        with self.assertRaises(ValueError):          # P6's 50-item grid cannot hold index 60
            I.build_one(self.pool, "test", "multivalue", 12, 60, 8192)
        old = I.build_one(self.pool, "calib", "multikey", 16, 0, 8192)
        self.assertNotIn("n_per_family", old)        # sets built before Amendment 4 keep their bytes

    def test_graded_items_pass_their_checks_and_repeat_exactly(self):
        I = self.I
        a = I.build_one(self.pool, "test", "multikey", 48, 5, 8192)
        b = I.build_one(self.pool, "test", "multikey", 48, 5, 8192)
        self.assertEqual(I.item_line(a), I.item_line(b))
        self.assertEqual(a["uid"], "multikey-k48-05")
        self.assertLessEqual(a["prompt_tokens"], 8192 - 512)
        self.assertEqual(len(a["answers"]), 1)
        self.assertEqual(len(a["distractors"]), 47)
        v = I.build_one(self.pool, "test", "vartrack", 3, 5, 8192)
        self.assertEqual((v["uid"], len(v["answers"])), ("vartrack-c3-05", 3))

    def test_seeds_differ_by_set(self):
        I = self.I
        t = I.build_one(self.pool, "test", "vartrack", 2, 0, 8192)
        c = I.build_one(self.pool, "calib", "vartrack", 2, 0, 8192)
        self.assertNotEqual(t["sha256"], c["sha256"])
        self.assertEqual(I.seed_hex("test"), __import__("hashlib").sha256(b"p8-longctx-2026-11").hexdigest())

    def test_cliff_uses_p6_needle_counts_and_the_calibrated_extras(self):
        I = self.I
        it = I.build_one(self.pool, "cliff", "multivalue", 4, 1, 8192)
        self.assertEqual((it["uid"], len(it["answers"])), ("multivalue-n4-01", 4))
        ex = I.build_one(self.pool, "cliff", "vartrack", 4, 9, 8192)
        self.assertEqual((ex["uid"], len(ex["answers"])), ("vartrack-c4-09", 4))
        with self.assertRaises(ValueError):
            I.build_one(self.pool, "cliff", "multikey", 16, 1, 8192)

    def test_check_item_catches_an_overlong_prompt(self):
        I = self.I
        with I.overrides(I.seed_hex("test"), I.target_tokens(8192, 16), "multikey", 16):
            rec = I.G.build_item(self.pool, "multikey", 2)
            self.assertEqual(I.check_item(rec, 8192, 16), [])
            self.assertTrue(any("L - 512" in x for x in I.check_item(rec, 8192 - 600, 16)))
            self.assertTrue(any("needle count" in x for x in I.check_item(rec, 8192, 15)))


class TestStats(unittest.TestCase):
    def test_score_uses_p6_containment(self):
        item = {"answers": ["ABCDE", "FGHIJ"]}
        self.assertEqual(T.score("The variables are abcde and FGHIJ.", item), 1)
        self.assertEqual(T.score("Only ABCDE.", item), 0)
        self.assertEqual(T.score("<think>FGHIJ ABCDE</think>none", item), 0)

    def test_select_level(self):
        self.assertEqual(T.select_level({16: 0.875, 48: 0.625, 128: 0.25}), 48)
        self.assertEqual(T.select_level({2: 0.75, 3: 0.55, 4: 0.2}), 3)       # 0.10 from 65% each: the harder
        self.assertEqual(T.select_level({16: 1.0, 48: 1.0, 128: 1.0}), 128)   # all tied: the hardest

    def test_paired_interval(self):
        a = [1] * 50 + [0] * 50
        self.assertEqual(T.paired_interval(a, a), (0.0, 0.0, 0.0))
        b = [1] * 40 + [0] * 60
        mean, lo, hi = T.paired_interval(a, b)
        self.assertAlmostEqual(mean, 10.0)
        self.assertTrue(4.0 < lo < 10.0 < hi < 16.0, (lo, hi))
        self.assertEqual(T.paired_interval(a, b), (mean, lo, hi))   # seeded: repeats exactly

    def test_verdicts(self):
        self.assertEqual(T.verdict(-4.9, 4.0), {"usable": True, "equivalent": True})
        self.assertEqual(T.verdict(-9.0, 2.0), {"usable": True, "equivalent": False})
        self.assertEqual(T.verdict(-10.0, 2.0), {"usable": False, "equivalent": False})

    def test_mcnemar_and_holm_and_wilson(self):
        self.assertEqual(T.mcnemar([1] * 5 + [0] * 5, [0] * 5 + [0] * 5), (5, 0, 0.0625))
        self.assertEqual(T.mcnemar([1, 0], [1, 0]), (0, 0, 1.0))
        self.assertEqual(T.holm({"H1": 0.01, "H3": 0.04}), {"H1": 0.02, "H3": 0.04})
        self.assertEqual(T.holm({"H1": 0.03, "H3": 0.02}), {"H3": 0.04, "H1": 0.04})
        lo, hi = T.wilson(0, 32)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1072, places=3)


def fake_items(d, name, families=("multivalue",), n=96, levels=None, length=65536):
    path = os.path.join(d, name)
    with open(path, "w") as f:
        for i in range(n):
            for fam in families:
                lv = (levels or {}).get(fam, 4)
                f.write(json.dumps({"uid": f"{fam}-x{lv}-{i:02d}", "family": fam, "level": lv, "index": i,
                                    "answers": [f"ANS{fam[:2]}{i:02d}"], "prompt_tokens": length - 600}) + "\n")
    return path


def item_rows(arm, items_path, right, status="ok"):
    rows = []
    with open(items_path) as f:
        for line in f:
            it = json.loads(line)
            ok = right(it)
            rows.append({"kind": "item", "arm": arm, "uid": it["uid"], "family": it["family"], "status": status,
                         "text": it["answers"][0] if ok else "no idea", "prompt_tokens": it["prompt_tokens"],
                         "prompt_tokens_built": it["prompt_tokens"], "prompt_time_s": 100.0,
                         "completion_tokens": 8, "completion_time_s": 0.4, "wall_s": 101.0, "finish_reason": "stop"})
    return rows


class TestAnalysis(unittest.TestCase):
    def test_calibration_rule_on_a_journal(self):
        import stage2_analyze as A
        d = tempfile.mkdtemp(prefix="p8lc-an-")
        path = os.path.join(d, "calib-64k.jsonl")
        with open(path, "w") as f:
            for j in range(8):
                for fam, levels in (("multikey", (16, 48, 128)), ("vartrack", (2, 3, 4))):
                    for lv in levels:
                        f.write(json.dumps({"uid": f"{fam}-{lv}-{j}", "family": fam, "level": lv,
                                            "answers": [f"A{fam}{lv}{j}"], "prompt_tokens": 64900}) + "\n")
        # multikey right on 8/8, 5/8, 2/8 items; vartrack on 6/8, 5/8, 1/8
        hits = {16: 8, 48: 5, 128: 2, 2: 6, 3: 5, 4: 1}
        rows = [{"kind": "arm_plan", "arm": "calib-64k-88", "items_file": "calib-64k.jsonl"}]
        rows += item_rows("calib-64k-88", path, lambda it: int(it["uid"].rsplit("-", 1)[1]) < hits[it["level"]])
        res = A.calib(rows, d)
        self.assertEqual(res["families"]["multikey"]["chosen"], 48)    # 62.5% is closest to 65%
        self.assertEqual(res["families"]["vartrack"]["chosen"], 3)     # 62.5% beats 75.0%
        self.assertFalse(res["families"]["vartrack"]["outside_30_90"])

    def test_second_calibration_on_multivalue(self):
        import stage2_analyze as A
        d = tempfile.mkdtemp(prefix="p8lc-an-")
        path = os.path.join(d, "calib2-64k.jsonl")
        with open(path, "w") as f:
            for j in range(8):
                for lv in (12, 24, 48):
                    f.write(json.dumps({"uid": f"multivalue-k{lv}-{j}", "family": "multivalue", "level": lv,
                                        "answers": [f"V{lv}{j}"], "prompt_tokens": 64900}) + "\n")
        hits = {12: 8, 24: 6, 48: 3}                                   # 100%, 75%, 37.5%
        rows = [{"kind": "arm_plan", "arm": "calib2-64k-88", "items_file": "calib2-64k.jsonl"}]
        rows += item_rows("calib2-64k-88", path, lambda it: int(it["uid"].rsplit("-", 1)[1]) < hits[it["level"]])
        res = A.calib(rows, d, "calib2-64k-88")
        self.assertEqual(list(res["families"]), ["multivalue"])
        self.assertEqual(res["families"]["multivalue"]["chosen"], 24)

    def test_stage2_comparisons_noise_and_rules(self):
        import stage2_analyze as A
        d = tempfile.mkdtemp(prefix="p8lc-an-")
        g64 = fake_items(d, "graded-64k.jsonl")
        g32 = fake_items(d, "graded-32k.jsonl", length=32768)
        rows = [{"kind": "arm_plan", "arm": a, "items_file": f} for a, f in
                (("s2-64k-88", "graded-64k.jsonl"), ("s2-64k-44", "graded-64k.jsonl"),
                 ("s2-64k-88-noise", "graded-64k.jsonl"), ("s2-32k-fp16", "graded-32k.jsonl"),
                 ("s2-32k-88", "graded-32k.jsonl"), ("s2-32k-44", "graded-32k.jsonl"))]
        base = lambda it: it["index"] % 3 != 0                                  # about 2/3 right
        rows += item_rows("s2-64k-88", g64, base)
        rows += item_rows("s2-64k-44", g64, lambda it: base(it) and it["index"] not in (1, 2))   # loses 2
        rows += item_rows("s2-64k-88-noise", g64, base)[:16]
        for arm in ("s2-32k-fp16", "s2-32k-88", "s2-32k-44"):
            rows += item_rows(arm, g32, base)
        res = A.stage2(rows, d)
        h2 = res["comparisons"]["h2"]
        self.assertEqual((h2["n"], h2["n10"], h2["n01"]), (96, 0, 2))
        self.assertAlmostEqual(h2["diff_points"], -100 * 2 / 96)
        self.assertTrue(h2["usable"])
        self.assertTrue(res["comparisons"]["h1"]["equivalent"])
        self.assertEqual(res["comparisons"]["h1"]["mcnemar_p_holm"], 1.0)
        self.assertEqual((res["noise"]["n"], res["noise"]["discordant"]), (16, 0))
        self.assertEqual(list(h2["by_family"]), ["multivalue"])
        self.assertEqual(res["rules"]["R1"]["mode"], "4,4")
        self.assertFalse(res["rules"]["R2"]["rescue_needed"])
        self.assertTrue(res["rules"]["stage3_runs"])
        self.assertIn("H2: `4,4` - `8,8`, 64K", A.markdown2(res))

    def test_stage3_usable_context(self):
        import stage2_analyze as A
        d = tempfile.mkdtemp(prefix="p8lc-an-")
        g32 = fake_items(d, "graded-32k.jsonl", length=32768)
        s96 = fake_items(d, "stage3-96k.jsonl", n=48, length=98304)
        rows2 = [{"kind": "arm_plan", "arm": "s2-32k-44", "items_file": "graded-32k.jsonl"}]
        rows2 += item_rows("s2-32k-44", g32, lambda it: True)
        rows3 = [{"kind": "arm_plan", "arm": "s3-96k", "items_file": "stage3-96k.jsonl"},
                 {"kind": "result", "step": "44-152k", "verdict": "pass", "peak_mib": 10600}]
        rows3 += item_rows("s3-96k", s96, lambda it: it["index"] % 4 != 0)      # 75%: 25 points down
        res = A.stage3(rows3, rows2, d, "4,4")
        self.assertEqual(res["usable_context_k"], 32)
        self.assertEqual(res["curve"][-1]["gap_vs_32k"]["points"], -25.0)
        self.assertEqual(res["probe"][0]["verdict"], "pass")


class TestPlans(unittest.TestCase):
    def test_every_plan_names_committed_manifests(self):
        import glob
        mans = {os.path.basename(m)[:-5] for m in glob.glob(os.path.join(HERE, "manifests", "*.json"))}
        for path in glob.glob(os.path.join(HERE, "plans", "*.json")):
            with open(path) as f:
                plan = json.load(f)
            files = [a["items"] for a in plan["arms"] if "items" in a]
            files += [v for a in plan["arms"] for v in a.get("items_by_length", {}).values()]
            for name in files:
                self.assertIn(name[:-len(".jsonl")], mans, (path, name))
            for a in plan["arms"]:
                graded = a["items"].startswith(("graded", "stage3", "calib2", "rep")) if "items" in a else True
                self.assertEqual(a.get("max_tokens", 128), 768 if graded else 128, (path, a["arm"]))

    def test_stage3_probe_ladder_follows_amendment_6(self):
        with open(os.path.join(HERE, "plans", "s3-44.json")) as f:
            plan = json.load(f)
        lengths = plan["probe"]["lengths"]
        self.assertEqual(lengths, [147456, 151552, 155648, 172032])
        self.assertEqual(sorted(int(k) for k in plan["arms"][-1]["items_by_length"]), lengths)
        self.assertEqual([a["arm"] for a in plan["arms"]], ["s3-96k", "s3-128k", "s3"])
        with open(os.path.join(HERE, "manifests", "probe-prompts-144-148.json")) as f:
            probe = json.load(f)
        self.assertEqual([p["length"] for p in probe["prompts"]], [147456, 151552])


class TestFd75dAnalysis(unittest.TestCase):
    def _run(self, logits, top1, targets, ids):
        import numpy as np
        offs = np.cumsum([0] + [len(t) for t in targets])
        return {"item_ids": np.array(ids), "item_offsets": offs, "targets": np.concatenate(targets),
                "default__top1": np.concatenate(top1), "full__default": np.concatenate(logits),
                "full_index": np.arange(offs[-1])}

    def test_rule_flags_divergence_beyond_the_floor_only(self):
        import numpy as np
        import fd75d_analyze as FA
        rng = np.random.default_rng(1)
        ids = ["28k-a", "48k-a", "64k-a"]
        base = [rng.normal(size=(20, 50)).astype(np.float32) for _ in ids]
        tg = [rng.integers(0, 50, 20) for _ in ids]
        t1 = self._run(base, [b.argmax(1) for b in base], tg, ids)
        t2 = self._run([b + 1e-4 for b in base], [b.argmax(1) for b in base], tg, ids)       # a constant shift: KL 0
        noisy = [b + (rng.normal(scale=0.5, size=b.shape) if i == "64k-a" else 0) for b, i in zip(base, ids)]
        f = self._run(noisy, [b.argmax(1) for b in noisy], tg, ids)
        res = FA.judge(f, t1, t2)
        self.assertFalse(res["lengths"]["28k"]["diverges"])
        self.assertFalse(res["lengths"]["48k"]["diverges"])
        self.assertTrue(res["lengths"]["64k"]["diverges"])
        self.assertTrue(res["decision"]["diverges_48k_or_64k"])
        self.assertEqual(res["predictions"]["F1_no_divergence_28k"], "holds")
        self.assertEqual(res["predictions"]["F2_no_divergence_48k_64k"], "refuted")

    def test_stream_mismatch_is_refused(self):
        import numpy as np
        import fd75d_analyze as FA
        lg = [np.zeros((4, 5), np.float32)]
        a = self._run(lg, [np.zeros(4, int)], [np.array([1, 2, 3, 4])], ["28k-a"])
        b = self._run(lg, [np.zeros(4, int)], [np.array([1, 2, 3, 9])], ["28k-a"])
        with self.assertRaises(SystemExit):
            FA.judge(a, a, b)


class TestArmEnv(unittest.TestCase):
    def test_arm_env_accepts_exl3_strings_only(self):
        import stage2_run as R
        self.assertEqual(R.arm_env({"arm": "a", "env": {"EXL3_FD75D": "0"}}), {"EXL3_FD75D": "0"})
        self.assertEqual(R.arm_env({"arm": "a"}), {})
        for bad in ({"PYTHONPATH": "/x"}, {"EXL3_FD75D": 0}, {"LD_LIBRARY_PATH": "/y"}):
            with self.assertRaises(R.R1.HarnessError):
                R.arm_env({"arm": "a", "env": bad})

    def test_server_env_carries_the_arm_switch_over_the_w13_environment(self):
        import types
        import stage2_run as R
        args = types.SimpleNamespace(tabby_logs="/x/logs")
        env = R.server_env_for(args, {"arm": "a", "env": {"EXL3_FD75D": "0"}})
        self.assertEqual(env["EXL3_FD75D"], "0")
        self.assertEqual((env["EXL3_GEMV_SK"], env["EXL3_INT8_SQ_MAX_M"], env["EXL3_FD75D_LIVE"]), ("2", "2", "0"))
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
        self.assertNotIn("EXL3_FD75D", R.server_env_for(args, {"arm": "b"}))   # other arms keep fd75d (the default)

    def test_fd75d_arm_plan_follows_its_registration(self):
        import stage2_run as R
        with open(os.path.join(HERE, "plans", "fd75d-arm.json")) as f:
            plan = json.load(f)
        (a,) = plan["arms"]
        self.assertEqual((plan["group"], a["format"], a["length"], a["items"], a["max_tokens"], a["env"]),
                         ("fd75d-arm", "8,8", 65536, "graded-64k.jsonl", 768, {"EXL3_FD75D": "0"}))
        self.assertEqual(R.BUDGET_S["fd75d-arm"], 12600)


class TestRunnerPieces(unittest.TestCase):
    def test_noise_selection_and_estimates(self):
        import stage2_run as R
        d = tempfile.mkdtemp(prefix="p8lc-s2-")
        path = os.path.join(d, "x.jsonl")
        with open(path, "w") as f:
            for i in range(20):
                for fam in ("multikey", "vartrack"):
                    f.write(json.dumps({"uid": f"{fam}-{i}", "family": fam, "prompt_tokens": 64000}) + "\n")
        sel = R.load_items(path, {"first_per_family": 16})
        self.assertEqual(len(sel), 32)
        self.assertEqual(sel[-1]["uid"], "vartrack-15")
        self.assertAlmostEqual(R.first_item_estimate(65536, 65024), 113.5 + 30)
        body = R.request_body({"content": "x"}, 768)
        self.assertEqual((body["max_tokens"], body["temperature"], body["chat_template_kwargs"]),
                         (768, 0, {"enable_thinking": False}))
        self.assertEqual(R.BUDGET_S["stage2"], 22 * 3600)

    def test_group_time_counts_finished_interrupted_and_probe(self):
        import stage2_run as R
        d = tempfile.mkdtemp(prefix="p8lc-s2-")
        j = R.Journal(os.path.join(d, "j.jsonl"))
        j.add({"kind": "session_start", "session": "a#1", "arm": "a", "t_start": 100.0})
        j.add({"kind": "item", "session": "a#1", "arm": "a", "uid": "u1", "t_req": 101.0, "t_end": 160.0})
        j.add({"kind": "session_end", "session": "a#1", "arm": "a", "wall_s": 70.0})
        j.add({"kind": "session_start", "session": "b#2", "arm": "b", "t_start": 200.0})
        j.add({"kind": "item", "session": "b#2", "arm": "b", "uid": "u2", "t_req": 201.0, "t_end": 230.0})
        j.add({"kind": "result", "step": "44-152k", "wall_s": 400.0})
        self.assertEqual(R.group_used_s(j), 70.0 + 30.0 + 400.0)
        self.assertEqual(R.group_used_s(j, exclude="b#2"), 470.0)
        self.assertEqual(j.items_done("a"), {"u1"})
        self.assertIsNone(j.arm_closed("a"))



class TestReplication(unittest.TestCase):
    """research/prereg/2026-11-04-p8-value-bits-replication.md: the plan, the 8,6 format, the R2 reproduction check
    and the stop rule that cuts from the end of the priority order."""

    def test_plan_follows_the_registration(self):
        import stage2_run as R
        with open(os.path.join(HERE, "plans", "replication.json")) as f:
            plan = json.load(f)
        self.assertEqual((plan["group"], plan["cut_from_end"], R.BUDGET_S["replication"]), ("replication", True, 54000))
        got = [(a["arm"], a["format"], a["length"], a["items"], a.get("env", {}), a.get("max_tokens"))
               for a in plan["arms"]]
        self.assertEqual(got, [
            ("rep-64k-88-triton", "8,8", 65536, "rep-64k.jsonl", {"EXL3_FD75D": "0"}, 768),
            ("rep-64k-84", "8,4", 65536, "rep-64k.jsonl", {}, 768),
            ("noise-64k-88-triton-qcs128", "8,8", 65536, "graded-64k.jsonl",
             {"EXL3_FD75D": "0", "EXL3_QC_STAGING_PAGES": "128"}, 768),
            ("rep-32k-84", "8,4", 32768, "graded-32k.jsonl", {}, 768),
            ("rep-32k-fp16-check", "fp16", 32768, "graded-32k.jsonl", {}, 768),
            ("rep-32k-fp16", "fp16", 32768, "graded-32k.jsonl", {}, 768),
            ("rep-64k-86", "8,6", 65536, "graded-64k.jsonl", {}, 768)])
        self.assertEqual(plan["arms"][4]["select"], {"first_per_family": 16})
        self.assertEqual(plan["arms"][5]["skip_if_reproduced"],
                         {"check_arm": "rep-32k-fp16-check", "reference_journal": "../stage2/journal.jsonl",
                          "reference_arm": "s2-32k-fp16", "n": 16})
        self.assertTrue(all("per_item_s" in a["if_budget"] for a in plan["arms"]))
        for a in plan["arms"]:
            R.arm_env(a)   # every arm's environment passes the EXL3_* rule

    def test_eight_six_renders_and_is_inside_the_bounded_reach(self):
        import stage2_run as R
        text = R.C.render("8,6", 65536, 8098, "/m")
        self.assertIn('  cache_mode: "8,6"', text)
        self.assertIn('  draft_cache_mode: "8,6"', text)
        self.assertEqual(R.C.check(text, "8,6", 65536, 8098), [])
        self.assertEqual(R.REACH["8,6"], R.REACH["8,8"])
        self.assertEqual(R.S.CACHE_MODE["fp16"], "FP16")   # stage 1's own formats unchanged

    def _journals(self, texts, ref_texts, closed="done"):
        import stage2_run as R
        d = tempfile.mkdtemp(prefix="p8lc-rep-")
        out = os.path.join(d, "replication")
        os.makedirs(os.path.join(d, "stage2"))
        with open(os.path.join(d, "stage2", "journal.jsonl"), "w") as f:
            for u, t in ref_texts.items():
                f.write(json.dumps({"kind": "item", "arm": "s2-32k-fp16", "uid": u, "status": "ok", "text": t}) + "\n")
        j = R.Journal(os.path.join(d, "j.jsonl"))
        for u, t in texts.items():
            j.add({"kind": "item", "arm": "rep-32k-fp16-check", "uid": u, "status": "ok", "text": t})
        if closed:
            j.add({"kind": "arm_end", "arm": "rep-32k-fp16-check", "status": closed})
        import types
        return types.SimpleNamespace(out=out), j

    def test_reproduction_check_needs_every_text_identical(self):
        import stage2_run as R
        rule = {"check_arm": "rep-32k-fp16-check", "reference_journal": "../stage2/journal.jsonl",
                "reference_arm": "s2-32k-fp16", "n": 3}
        ref = {"a": "1 2", "b": "3", "c": "4", "d": "5"}
        ok, info = R.reproduced(*self._journals({"a": "1 2", "b": "3", "c": "4"}, ref), rule)
        self.assertTrue(ok)
        self.assertEqual((info["n_check"], info["identical"]), (3, 3))
        ok, info = R.reproduced(*self._journals({"a": "1 2", "b": "3", "c": "4 "}, ref), rule)
        self.assertFalse(ok)
        self.assertEqual(info["identical"], 2)
        self.assertFalse(R.reproduced(*self._journals({"a": "1 2", "b": "3"}, ref), rule)[0])          # 2 of 3 ran
        self.assertFalse(R.reproduced(*self._journals({"a": "1 2", "b": "3", "c": "4"}, ref, None), rule)[0])
        self.assertFalse(R.reproduced(*self._journals({"a": "1 2", "b": "3", "c": "4"}, ref, "budget_stop"),
                                      rule)[0])

    def test_whole_arm_estimate_skips_before_any_server(self):
        import types
        import stage2_run as R
        d = tempfile.mkdtemp(prefix="p8lc-rep-")
        with open(os.path.join(d, "x.jsonl"), "w") as f:
            for i in range(96):
                f.write(json.dumps({"uid": f"u{i}", "family": "multivalue", "level": 12, "prompt_tokens": 64900})
                        + "\n")
        j = R.Journal(os.path.join(d, "j.jsonl"))
        j.add({"kind": "session_start", "session": "a#1", "arm": "a", "t_start": 0.0})
        j.add({"kind": "session_end", "session": "a#1", "arm": "a", "wall_s": 54000 - 96 * 117 - 119})
        args = types.SimpleNamespace(items=d, probe_passed={}, out=d)
        spec = {"arm": "r3", "format": "8,6", "length": 65536, "items": "x.jsonl", "if_budget": {"per_item_s": 117}}
        self.assertEqual(R.run_arm(args, j, spec, "replication", [1]), "skipped")
        row = [r for r in j.rows if r.get("kind") == "arm_skipped"][-1]
        self.assertEqual((row["arm"], row["why"], row["estimate_s"]), ("r3", "budget", 96 * 117 + 120))

    def test_main_cuts_every_arm_after_a_budget_skip(self):
        import stage2_run as R
        d = tempfile.mkdtemp(prefix="p8lc-rep-")
        plan = {"name": "p", "group": "replication", "cut_from_end": True,
                "arms": [{"arm": a, "format": "8,8", "length": 65536, "items": "x.jsonl"} for a in "abcd"]}
        with open(os.path.join(d, "plan.json"), "w") as f:
            json.dump(plan, f)
        calls = []

        def fake_run_arm(args, journal, spec, group, sessions):
            calls.append(spec["arm"])
            if spec["arm"] == "b":
                journal.add({"kind": "arm_skipped", "arm": "b", "why": "budget"})
                return "skipped"
            journal.add({"kind": "arm_end", "arm": spec["arm"], "status": "done"})
            return "done"

        saved = (R.run_arm, R.R1.smi, R.R1.LAB, sys.argv)
        try:
            R.run_arm, R.R1.smi, R.R1.LAB = fake_run_arm, (lambda q: ["260.00"]), d
            sys.argv = ["stage2_run.py", os.path.join(d, "plan.json"), "--items", d, "--out", os.path.join(d, "out")]
            self.assertEqual(R.main(), 0)
        finally:
            R.run_arm, R.R1.smi, R.R1.LAB, sys.argv = saved
        self.assertEqual(calls, ["a", "b"])
        with open(os.path.join(d, "out", "journal.jsonl")) as f:
            rows = [json.loads(x) for x in f if x.strip()]
        skipped = [(r["arm"], r["why"]) for r in rows if r.get("kind") == "arm_skipped"]
        self.assertEqual(skipped, [("b", "budget"), ("c", "budget: an earlier arm was cut"),
                                   ("d", "budget: an earlier arm was cut")])


class TestReplicationAnalysis(unittest.TestCase):
    def test_outcome_rule(self):
        import replication_analyze as RA
        cases = {(0.5, 9.0): "replicated", (5.2, 18.8): "replicated, interval above +5",
                 (-4.0, 4.9): "equivalent", (-4.0, 6.0): "not replicated", (-12.0, -1.0): "not replicated, reversed",
                 (0.0, 3.0): "equivalent"}
        for (lo, hi), want in cases.items():
            self.assertEqual(RA.outcome({"ci90": [lo, hi]}), want, (lo, hi))

    def test_judge_on_synthetic_journals(self):
        import replication_analyze as RA
        d = tempfile.mkdtemp(prefix="p8lc-rep-an-")
        g64 = fake_items(d, "graded-64k.jsonl")
        g32 = fake_items(d, "graded-32k.jsonl", length=32768)
        rep = fake_items(d, "rep-64k.jsonl")
        base = lambda it: it["index"] % 3 != 0
        plans = [("s2-64k-88-triton", "graded-64k.jsonl"), ("s2-64k-84", "graded-64k.jsonl"),
                 ("s2-32k-fp16", "graded-32k.jsonl"), ("s2-32k-88", "graded-32k.jsonl")]
        stage2 = [{"kind": "arm_plan", "arm": a, "items_file": f} for a, f in plans[1:]]
        stage2 += item_rows("s2-64k-84", g64, lambda it: base(it) or it["index"] in (0, 3, 6))
        stage2 += item_rows("s2-32k-fp16", g32, base) + item_rows("s2-32k-88", g32, base)
        arm = [{"kind": "arm_plan", "arm": "s2-64k-88-triton", "items_file": "graded-64k.jsonl"}]
        arm += item_rows("s2-64k-88-triton", g64, base)
        rp = [{"kind": "arm_plan", "arm": a, "items_file": f} for a, f in
              (("rep-64k-88-triton", "rep-64k.jsonl"), ("rep-64k-84", "rep-64k.jsonl"),
               ("noise-64k-88-triton-qcs128", "graded-64k.jsonl"), ("rep-32k-84", "graded-32k.jsonl"),
               ("rep-32k-fp16-check", "graded-32k.jsonl"), ("rep-64k-86", "graded-64k.jsonl"))]
        rp += item_rows("rep-64k-88-triton", rep, base)
        rp += item_rows("rep-64k-84", rep, base)                       # identical: D1 = 0, equivalent
        rp += item_rows("noise-64k-88-triton-qcs128", g64, lambda it: base(it) != (it["index"] == 1))  # 1 flip
        rp += item_rows("rep-32k-84", g32, base)
        rp += item_rows("rep-32k-fp16-check", g32, base)[:16]
        rp += [{"kind": "arm_end", "arm": a, "status": "done"} for a in
               ("rep-64k-88-triton", "rep-64k-84", "noise-64k-88-triton-qcs128", "rep-32k-84", "rep-32k-fp16-check")]
        rp += [{"kind": "arm_skipped", "arm": "rep-32k-fp16", "why": "reproduced"}]
        rp += item_rows("rep-64k-86", g64, base) + [{"kind": "arm_end", "arm": "rep-64k-86", "status": "done"}]
        res = RA.judge(rp, stage2, arm, d)
        self.assertEqual((res["R1"]["outcome"], res["R1"]["diff_points"]), ("equivalent", 0.0))
        self.assertEqual(res["N"]["discordant"], 1)
        self.assertTrue(res["N"]["K2_beyond_harmless_spread"])           # Wilson upper of 1/96 < 17/96
        self.assertEqual(res["N"]["first16"]["N"], {"n": 16, "discordant": 1})
        self.assertEqual((res["R2"]["fp16_reference"], res["R2"]["reproduced"]), ("s2-32k-fp16", True))
        self.assertTrue(res["R3"]["monotone"])                          # 6-bit = 8-bit; 4-bit above 6-bit
        self.assertEqual(res["status"]["rep-32k-fp16"], "skipped (reproduced)")

if __name__ == "__main__":
    unittest.main()
