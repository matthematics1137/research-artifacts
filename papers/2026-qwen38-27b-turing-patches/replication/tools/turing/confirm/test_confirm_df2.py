#!/usr/bin/env python3
"""CPU tests for run_confirm_df2.py, plan-df2.json and analyze_confirm_df2.py (Amendment 2): the frozen files, the
conditions, the drafter check, the ports and the addendum pairs.

  /usr/local/bin/zarael-sandbox -C /opt/p8/repository -- bash -c \
      'cd tools/turing/confirm && ../../../engines/analysis-env/bin/python -m unittest -v test_confirm_df2'
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_confirm_df2 as A  # noqa: E402
import run_confirm as RC  # noqa: E402
import run_confirm_df2 as D  # noqa: E402
import run_confirm_fit as F  # noqa: E402

DF2 = json.load(open(os.path.join(HERE, "plan-df2.json")))
PLAN = RC.load_plan(os.path.join(HERE, "plan.json"), "/lab")


class TestFrozen(unittest.TestCase):
    def test_registered_files_unchanged(self):
        self.assertEqual(F.frozen_ok(DF2), {"plan.json": True, "run_confirm.py": True, "analyze_confirm.py": True,
                                            "run_confirm_fit.py": True})

    def test_analysis_pin_is_the_registered_hash(self):
        self.assertEqual(A.REGISTERED, DF2["frozen_sha256"]["analyze_confirm.py"])


class TestConditions(unittest.TestCase):
    def setUp(self):
        self.conds, self.configs = D.df2_conditions(PLAN, DF2)

    def test_order_ids_and_priorities(self):
        self.assertEqual([c["id"] for c in self.conds],
                         ["w13-165-mtp-0", "nif-xxs-df2-0", "w13-165-nd-0", "nif-xxs-0", "w13-165-mtp-28k",
                          "nif-xxs-df2-28k"])
        self.assertEqual({c["id"]: c["priority"] for c in self.conds if c["priority"] == 2},
                         {"w13-165-nd-0": 2, "nif-xxs-0": 2})

    def test_dflash2_arm_is_the_registered_conditional(self):
        d = PLAN["conditional"]["nif-dflash2"]
        self.assertEqual(self.configs["nif-xxs-df2"], {"drafted": True, "args": d["config_args"]})
        self.assertEqual(d["config_args"], ["/lab/models/gguf/Qwen3.8-27B-UD-IQ2_XXS.gguf", "--spec", "dflash2"])
        df = {c["id"]: c for c in self.conds if c["config"] == "nif-xxs-df2"}
        self.assertEqual({k: (c["engine"], c["fill"]) for k, c in df.items()},
                         {"nif-xxs-df2-0": ("ninferno", 0), "nif-xxs-df2-28k": ("ninferno", 28000)})

    def test_registered_conditions_keep_their_settings(self):
        reg = {c["id"]: c for c in PLAN["conditions"]}
        for c in self.conds:
            if c["id"] in reg:
                self.assertEqual(dict(c, priority=None), dict(reg[c["id"]], priority=None))

    def test_unregistered_id_refused(self):
        with self.assertRaises(ValueError):
            D.df2_conditions(PLAN, dict(DF2, conditions=[{"id": "nif-xxs-df2-64k", "priority": 1}]))

    def test_registered_plan_is_not_mutated(self):
        self.assertNotIn("nif-xxs-df2", PLAN["configs"])


class TestRun(unittest.TestCase):
    def make(self, out):
        return D.DF2Run(PLAN, DF2, out, True)

    def test_server_commands_use_the_step_ports_and_the_registered_arguments(self):
        with tempfile.TemporaryDirectory() as out:
            run = self.make(out)
            c = next(x for x in run.plan["conditions"] if x["id"] == "nif-xxs-df2-0")
            cmd1, env = run.server_cmd(c)
            cmd2, _ = run.server_cmd(c)
            self.assertEqual(cmd1[1:4], ["/lab/models/gguf/Qwen3.8-27B-UD-IQ2_XXS.gguf", "--spec", "dflash2"])
            self.assertIn("--max-context", cmd1)
            p1, p2 = int(cmd1[cmd1.index("--port") + 1]), int(cmd2[cmd2.index("--port") + 1])
            self.assertEqual((p1, p2), (18341, 18342))
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["HF_HOME"], "/lab/cache/huggingface")

    def test_drafter_check_accepts_the_registered_file_and_drops_otherwise(self):
        with tempfile.TemporaryDirectory() as lab, tempfile.TemporaryDirectory() as out:
            blob = b"drafter bytes"
            d = dict(DF2["drafter"], bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            run = D.DF2Run(PLAN, dict(DF2, drafter=d), out, True)
            path = D.drafter_path(d, lab)
            self.assertTrue(path.endswith("models--z-lab--Qwen3.8-27B-DFlash2/snapshots/"
                                          "50307d4c4cde6860d4eee73e2547cd786fe8e8a4/model.safetensors"))
            self.assertFalse(run.check_drafter(lab))
            self.assertEqual(set(run.dropped), {"nif-xxs-df2-0", "nif-xxs-df2-28k"})
            os.makedirs(os.path.dirname(path))
            with open(path, "wb") as f:
                f.write(blob)
            run2 = D.DF2Run(PLAN, dict(DF2, drafter=d), out, True)
            self.assertTrue(run2.check_drafter(lab))
            self.assertEqual(run2.dropped, {})
            with open(path, "ab") as f:
                f.write(b"!")
            run3 = D.DF2Run(PLAN, dict(DF2, drafter=d), out, True)
            self.assertFalse(run3.check_drafter(lab))

    def test_dry_run_prints_every_condition(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, os.path.join(HERE, "run_confirm_df2.py"), "--plan",
                                os.path.join(HERE, "plan.json"), "--df2", os.path.join(HERE, "plan-df2.json"),
                                "--out", out, "--dry-run"], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            tags = [line.split(" ")[0] for line in r.stdout.splitlines()]
            self.assertEqual(tags[:6], [f"confirm-{c['id']}-r1" for c in D.df2_conditions(PLAN, DF2)[0]])
            self.assertIn("-mtp", r.stdout.splitlines()[0])
            self.assertTrue(tags[6] == "drafter")


class TestAddendum(unittest.TestCase):
    def rows(self, d, rates):
        path = os.path.join(d, "rows.jsonl")
        with open(path, "w") as f:
            for cond, per_round in rates.items():
                for rnd, rate in enumerate(per_round, 1):
                    f.write(json.dumps({"tag": f"confirm-{cond}-r{rnd}", "new_tokens": 256,
                                        "decode_wall_s": 256 / rate}) + "\n")
        return path

    def test_pairs_and_missing_arms(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.rows(d, {"w13-165-mtp-0": [72, 71, 72], "nif-xxs-df2-0": [55, 56, 54],
                                 "w13-165-nd-0": [48, 48, 47], "nif-xxs-0": [46, 46, 45]})
            with redirect_stdout(io.StringIO()):
                res = A.analyze(path)
            c = res["comparisons"]
            self.assertTrue(c["over_nif_df2_0"]["faster_every_round"])
            self.assertEqual(c["over_nif_df2_0"]["rounds"], 3)
            self.assertAlmostEqual(c["nif_df2_gain_0"]["ratio"], 165 / 137, places=2)
            self.assertIsNone(c["over_nif_df2_28k"])
            self.assertTrue(os.path.exists(os.path.join(d, "results-df2.json")))


if __name__ == "__main__":
    unittest.main()
