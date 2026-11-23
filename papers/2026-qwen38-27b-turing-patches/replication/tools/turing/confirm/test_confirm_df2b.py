#!/usr/bin/env python3
"""CPU tests for run_confirm_df2b.py and plan-df2b.json (Amendment 3): the frozen files and the one change.

  /usr/local/bin/zarael-sandbox -C /opt/p8/repository -- bash -c \
      'cd tools/turing/confirm && ../../../engines/analysis-env/bin/python -m unittest -v test_confirm_df2b'
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_confirm as RC  # noqa: E402
import run_confirm_df2 as D  # noqa: E402
import run_confirm_df2b as B  # noqa: E402
import run_confirm_fit as F  # noqa: E402

DF2B = json.load(open(os.path.join(HERE, "plan-df2b.json")))
DF2 = json.load(open(os.path.join(HERE, "plan-df2.json")))
PLAN = RC.load_plan(os.path.join(HERE, "plan.json"), "/lab")
ART = "/lab/models/gguf/ninferno-df2b/Qwen3.8-27B-UD-IQ2_XXS.qwen3_5_bf16.ninferno"


class TestFrozen(unittest.TestCase):
    def test_registered_and_amendment_2_files_unchanged(self):
        self.assertTrue(all(F.frozen_ok(DF2B).values()), F.frozen_ok(DF2B))
        self.assertEqual(set(F.frozen_ok(DF2B)), {"plan.json", "run_confirm.py", "analyze_confirm.py",
                                                   "run_confirm_fit.py", "run_confirm_df2.py", "plan-df2.json",
                                                   "analyze_confirm_df2.py"})

    def test_only_the_artifact_ports_and_labels_differ_from_amendment_2(self):
        strip = lambda p: {k: v for k, v in p.items() if k not in ("amendment", "frozen_sha256", "dflash2_artifact",
                                                                    "dflash2_max_context", "port_range", "conditions")}
        self.assertEqual(strip(DF2B), strip(DF2))
        self.assertEqual([(c["id"], c["priority"]) for c in DF2B["conditions"]],
                         [(c["id"], c["priority"]) for c in DF2["conditions"]])
        a = DF2B["dflash2_artifact"]
        self.assertTrue(isinstance(a["bytes"], int) and a["bytes"] > 7e9 and len(a["sha256"]) == 64)


class TestArgs(unittest.TestCase):
    def test_dflash2_args_swap_only_the_gguf_for_the_artifact(self):
        reg = PLAN["conditional"]["nif-dflash2"]["config_args"]
        self.assertEqual(B.dflash2_args(PLAN, DF2B, "/lab"), [ART] + reg[1:])
        self.assertEqual(reg[1:], ["--spec", "dflash2"])

    def test_run_uses_the_artifact_and_leaves_the_rest(self):
        with tempfile.TemporaryDirectory() as out:
            run = B.DF2bRun(PLAN, DF2B, out, True)
            base = D.DF2Run(PLAN, DF2, out, True)
            self.assertEqual(run.plan["conditions"], base.plan["conditions"])
            self.assertEqual(run.plan["configs"]["nif-xxs"], base.plan["configs"]["nif-xxs"])
            c = next(x for x in run.plan["conditions"] if x["id"] == "nif-xxs-df2-28k")
            cmd, _ = run.server_cmd(c)
            self.assertEqual(cmd[2:4], ["--spec", "dflash2"])
            self.assertTrue(cmd[1].endswith("ninferno-df2b/Qwen3.8-27B-UD-IQ2_XXS.qwen3_5_bf16.ninferno"))
            self.assertEqual(int(cmd[cmd.index("--port") + 1]), 18361)
            self.assertNotIn("nif-xxs-df2", PLAN["configs"])

    def test_dflash2_context_is_automatic_empty_and_29696_at_28k(self):
        with tempfile.TemporaryDirectory() as out:
            run = B.DF2bRun(PLAN, DF2B, out, True)
            by = {x["id"]: x for x in run.plan["conditions"]}
            empty, _ = run.server_cmd(by["nif-xxs-df2-0"])
            deep, _ = run.server_cmd(by["nif-xxs-df2-28k"])
            undrafted, _ = run.server_cmd(by["nif-xxs-0"])
            self.assertNotIn("--max-context", empty)
            self.assertEqual(deep[deep.index("--max-context") + 1], "29696")
            self.assertEqual(undrafted[undrafted.index("--max-context") + 1], "32768")
            self.assertEqual(empty.count("--max-concurrency") + deep.count("--max-concurrency"), 2)

    def test_artifact_check_accepts_the_recorded_file_and_drops_otherwise(self):
        with tempfile.TemporaryDirectory() as lab, tempfile.TemporaryDirectory() as out:
            blob = b"converted artifact bytes"
            art = dict(DF2B["dflash2_artifact"], bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            df2 = dict(DF2B, dflash2_artifact=art)
            run = B.DF2bRun(PLAN, df2, out, True)
            self.assertFalse(run.check_artifact(lab))
            self.assertEqual(set(run.dropped), {"nif-xxs-df2-0", "nif-xxs-df2-28k"})
            path = B.artifact_path(df2, lab)
            os.makedirs(os.path.dirname(path))
            with open(path, "wb") as f:
                f.write(blob)
            run2 = B.DF2bRun(PLAN, df2, out, True)
            self.assertTrue(run2.check_artifact(lab))
            self.assertEqual(run2.dropped, {})

    def test_session_stops_when_the_dflash2_arm_cannot_start(self):
        with tempfile.TemporaryDirectory() as out:
            run = B.DF2bRun(PLAN, DF2B, out, True)
            run.dropped.update({"nif-xxs-df2-0": "prepare: server exited 1", "nif-xxs-df2-28k": "prepare: exited 1"})
            run.prepare()   # dry: prints the prepare commands, starts nothing
            self.assertEqual(set(run.dropped), {c["id"] for c in run.plan["conditions"]})
            run2 = B.DF2bRun(PLAN, DF2B, out, True)
            run2.prepare()
            self.assertEqual(run2.dropped, {})

    def test_dry_run(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, os.path.join(HERE, "run_confirm_df2b.py"), "--plan",
                                os.path.join(HERE, "plan.json"), "--df2", os.path.join(HERE, "plan-df2b.json"),
                                "--out", out, "--dry-run"], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            lines = r.stdout.splitlines()
            self.assertEqual(len(lines), 8)
            self.assertIn(".ninferno --spec dflash2", lines[1])
            self.assertTrue(lines[7].startswith("artifact "))


if __name__ == "__main__":
    unittest.main()
