#!/usr/bin/env python3
"""CPU tests for run_confirm_fit.py and plan-fit.json (Amendment 1): the frozen files, the port wait, the ladder.

  /usr/local/bin/zarael-sandbox -C /opt/p8/repository -- bash -c \
      'cd tools/turing/confirm && ../../../engines/analysis-env/bin/python -m unittest -v test_confirm_fit'
"""
import json
import os
import socket
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_confirm as RC  # noqa: E402
import run_confirm_fit as F  # noqa: E402

FIT = json.load(open(os.path.join(HERE, "plan-fit.json")))


def free_port():
    s = socket.socket()
    s.bind(("localhost", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class TestFrozen(unittest.TestCase):
    def test_registered_files_unchanged(self):
        self.assertEqual(F.frozen_ok(FIT), {"plan.json": True, "run_confirm.py": True, "analyze_confirm.py": True})


class TestPorts(unittest.TestCase):
    def test_skips_a_held_port_and_waits_when_all_are_held(self):
        lo = free_port()
        held = []
        try:
            s = socket.socket()
            s.bind(("localhost", lo))
            s.listen(1)
            held.append(s)
            got = F.choose_port(lo, lo + 2, lo, wait_s=0)
            self.assertNotEqual(got, lo)
            self.assertIn(got, (lo + 1, lo + 2))
            for p in (lo + 1, lo + 2):
                t = socket.socket()
                try:
                    t.bind(("localhost", p))
                    t.listen(1)
                    held.append(t)
                except OSError:
                    pass
            t0 = time.time()
            self.assertIsNone(F.choose_port(lo, lo + 2, lo, wait_s=0.5, poll_s=0.1))
            self.assertGreaterEqual(time.time() - t0, 0.5)
        finally:
            for s in held:
                s.close()


class TestLadder(unittest.TestCase):
    def setUp(self):
        self.plan = RC.load_plan(os.path.join(HERE, "plan.json"), "/lab")
        self.auto = FIT["ladder"]["R3"]["engines_with_auto_fit"]

    def args(self, config, engine):
        return RC.subst(self.plan["configs"][config]["args"] + self.plan["engines"][engine]["common"], {"port": 1})

    def test_context(self):
        self.assertEqual(F.context_for(28050), 29696)
        self.assertEqual(F.context_for(28416), 29696)
        self.assertEqual(F.context_for(28417), 30720)

    def test_rungs_are_cumulative_and_in_order(self):
        self.assertEqual(FIT["ladder"]["order"], ["R0", "R1", "R2", "R3"])
        base = self.args("lcpp-xxs-df", "lcpp9")
        r0 = F.rung_args(base, "R0", "lcpp9", 29696, self.auto)
        self.assertEqual(r0, base)
        r1 = F.rung_args(base, "R1", "lcpp9", 29696, self.auto)
        self.assertEqual(r1.count("-ub"), 1)
        self.assertEqual(r1[r1.index("-ub") + 1], "128")
        self.assertEqual(r1[r1.index("-c") + 1], "32768")
        r2 = F.rung_args(base, "R2", "lcpp9", 29696, self.auto)
        self.assertEqual((r2[r2.index("-ub") + 1], r2[r2.index("-c") + 1]), ("128", "29696"))
        self.assertIn("-ngl", r2)
        r3 = F.rung_args(base, "R3", "lcpp9", 29696, self.auto)
        self.assertNotIn("-ngl", r3)
        self.assertIn("-ngld", r3)              # the drafter's own offload stays as registered
        self.assertEqual((r3[r3.index("-ub") + 1], r3[r3.index("-c") + 1]), ("128", "29696"))
        self.assertIsNone(F.rung_args(self.args("ik-s-mtp", "ik"), "R3", "ik", 29696, self.auto))
        self.assertIsNone(F.rung_args(base, "R2", "lcpp9", None, self.auto))

    def test_offload_parse(self):
        self.assertEqual(F.offloaded("load_tensors: offloaded 49/65 layers to GPU\n"), (49, 65))
        self.assertIsNone(F.offloaded("nothing here"))


class TestPlanFit(unittest.TestCase):
    def test_conditions(self):
        plan = RC.load_plan(os.path.join(HERE, "plan.json"), "/lab")
        ids = {c["id"] for c in plan["conditions"]}
        fit_ids = [c["id"] for c in FIT["conditions"]]
        self.assertTrue(set(fit_ids) <= ids)
        dropped = {"lcpp9-xxs-df-0", "lcpp9-xxs-df-28k", "lcpp9-s-mtp-0", "lcpp9-s-mtp-28k", "ik-s-mtp-0",
                   "buun-nd-0", "buun-mtp-0"}
        anchors = {"w13-165-nd-0", "w13-165-mtp-0", "w13-165-nd-28k", "w13-165-mtp-28k"}
        self.assertEqual(set(fit_ids), dropped | anchors)
        self.assertEqual((FIT["rounds"], FIT["cap_hours"]), (3, 1.0))
        self.assertTrue(all(c["priority"] == 1 for c in FIT["conditions"] if c["id"] in anchors))

    def test_dry_run_commands(self):
        plan = RC.load_plan(os.path.join(HERE, "plan.json"), "/lab")
        run = F.FitRun(plan, FIT, tempfile.mkdtemp(), dry=True)
        run.ctx = F.context_for(28050)
        lo, hi = FIT["port_range"]
        ports = []
        for c in run.plan["conditions"]:
            if run.plan["engines"][c["engine"]]["kind"] == "server":
                ok, why = run.try_rung(c, "R2")
                self.assertTrue(ok)
                cmd, _ = run.server_cmd(c)
                ports.append(int(cmd[cmd.index("--port") + 1]))
                self.assertEqual(cmd[cmd.index("-c") + 1], "29696")
        self.assertTrue(all(lo <= p <= hi for p in ports))
        self.assertEqual(len(set(ports[:2])), 2)
        self.assertEqual([c["id"] for c in run.plan["conditions"]][:4],
                         ["w13-165-nd-0", "w13-165-mtp-0", "w13-165-nd-28k", "w13-165-mtp-28k"])


if __name__ == "__main__":
    unittest.main()
