"""CPU-only safety/protocol checks. Synthetic rows are fixtures, never lab evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import run_refresh as R
from analyze_refresh import analyze


class RefreshTests(unittest.TestCase):
    def folder(self):
        # No explicit deletion; disposable sandbox owns fixture retention/cleanup.
        return Path(tempfile.mkdtemp(prefix="p8-refresh-fixture-"))

    def test_dimensions_and_pair_coverage(self):
        plan, phases = R.load()
        self.assertEqual([len(x.get("conditions", [])) for x in phases], [8, 4, 0, 6])
        self.assertEqual(len(plan["comparisons"]), 10)
        self.assertEqual(plan["cap_seconds"], 9000)
        for phase in phases:
            for rnd, ordered in R.rotations(phase.get("conditions", [])):
                self.assertEqual(len(ordered), len(phase.get("conditions", [])))
                self.assertEqual(len({c["id"] for c in ordered}), len(ordered))

    def test_dry_render_executes_nothing(self):
        plan, phases = R.load()
        with patch.object(R.subprocess, "Popen", side_effect=AssertionError("no subprocess")), \
                patch.object(R.os, "makedirs", side_effect=AssertionError("no directory")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            R.render(plan, phases)
        self.assertIn('"combined_cap_seconds": 9000', output.getvalue())

    def test_full_dry_cli_executes_nothing(self):
        args = ["run_refresh", "--freeze-sha", R.digest(R.HERE / "freeze.json"), "--dry-run"]
        with patch.object(R.sys, "argv", args), \
                patch.object(R.subprocess, "Popen", side_effect=AssertionError("no subprocess")), \
                patch.object(R.os, "makedirs", side_effect=AssertionError("no directory")), \
                patch.object(R.RC, "gpu_state", side_effect=AssertionError("no GPU query")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(R.main(), 0)

    def test_output_stays_in_lab_and_cannot_repeat(self):
        lab = self.folder();(lab / "runs").mkdir()
        with self.assertRaisesRegex(ValueError, "under the lab"):
            R.output_path(str(lab), str(lab / "elsewhere"))
        path = R.output_path(str(lab), str(lab / "runs/attempt"));path.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            R.output_path(str(lab), str(path))

    def test_duplicate_json_keys_refuse(self):
        root = self.folder();p = root / "duplicate.json"
        p.write_text('{"cap":9000,"cap":18000}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            R.read_json(p)

    def test_protocol_matches_registered_sampler(self):
        p, _ = R.load()
        for api in ("chat", "completion"):
            body = R.RC.request_body(api, "fixture", 256, p["sampler"])
            self.assertTrue(body["ignore_eos"])
            self.assertEqual(body["temperature"], .8)
            self.assertEqual(body["min_p"], .08)
            self.assertTrue(body["stream"])

    def test_changed_or_missing_artifact_refuses(self):
        root = self.folder(); path = root / "model.fixture"; path.write_bytes(b"old")
        spec = [dict(key="fixture", path=str(path), bytes=3, sha256=R.digest(path))]
        R.verify_files(spec, str(root))
        path.write_bytes(b"new")
        with self.assertRaisesRegex(ValueError, "digest"):
            R.verify_files(spec, str(root))
        spec[0]["path"] = str(root / "missing")
        with self.assertRaisesRegex(ValueError, "missing"):
            R.verify_files(spec, str(root))

    def test_overlay_change_refuses(self):
        root = self.folder(); path = root / "version.py"; path.write_text("fixture")
        spec = dict(root=str(root), suffixes=[".py"], files={"version.py": R.digest(path)})
        self.assertEqual(R.verify_overlay(spec, str(root)), 1)
        (root / "extra.py").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "overlay"):
            R.verify_overlay(spec, str(root))

    def test_freeze_cannot_be_rewritten_or_inputs_changed(self):
        root = self.folder(); here = root / "refresh"; here.mkdir()
        (here / "fixture.py").write_text("a")
        f = here / "freeze.json"
        f.write_text(json.dumps(dict(files={"fixture.py": R.digest(here / "fixture.py")})))
        sha = R.digest(f)
        R.check_freeze(here, sha)
        (here / "fixture.py").write_text("b")
        with self.assertRaisesRegex(ValueError, "frozen file"):
            R.check_freeze(here, sha)
        f.write_text("{}")
        with self.assertRaisesRegex(ValueError, "queued digest"):
            R.check_freeze(here, sha)

    def test_fit_full_requires_target_and_drafter_logs(self):
        self.assertFalse(R.residency("offloaded 65/65 layers to GPU")["verified_full"])
        self.assertTrue(R.residency("offloaded 65/65 layers to GPU\noffloaded 2/2 layers to GPU")["verified_full"])
        self.assertFalse(R.residency("offloaded 45/65 layers to GPU\noffloaded 2/2 layers to GPU")["verified_full"])
        self.assertTrue(R.residency("offloaded 65/65 layers to GPU", 1)["verified_full"])

    def test_ladder_is_bounded_and_partial_is_explicit(self):
        args = ["-ngl", "99", "-ngld", "99", "-ub", "256", "-c", "32768"]
        ctx = R.FIT.context_for(28050)
        self.assertEqual(ctx, 29696)
        self.assertEqual(R.FIT.rung_args(args,"R0","llama",ctx,["llama"]), args)
        r2 = R.FIT.rung_args(args,"R2","llama",ctx,["llama"])
        self.assertEqual(r2[r2.index("-ub")+1], "128")
        self.assertIn("29696", r2)
        self.assertNotIn("-ngl", R.FIT.rung_args(args,"R3","llama",ctx,["llama"]))

    def fake_run(self):
        plan, phases = R.load()
        obj = object.__new__(R.Refresh)
        obj.plan=plan;obj.out=str(self.folder());obj.rows=str(Path(obj.out)/"rows.jsonl")
        obj.started=time.monotonic();obj.end=obj.started+1000;obj.phase_end=obj.end
        obj.fit_records={};obj.dropped={};obj.log=lambda *a,**k:None
        obj.arm_timer=lambda *a,**k:None
        return obj, phases

    def test_guard_signal_is_not_swallowed_as_phase_timeout(self):
        obj, phases = self.fake_run()
        obj.run_condition=lambda *a: (_ for _ in ()).throw(R.CampaignStopped("guard"))
        with self.assertRaises(R.CampaignStopped):
            obj.phase_run(phases[0])

    def test_phase_timeout_is_terminal_for_that_phase(self):
        obj, phases = self.fake_run(); calls=[]
        def condition(*args):
            calls.append(args)
            raise R.BudgetExceeded("fixture deadline")
        obj.run_condition=condition
        obj.phase_run(phases[0])
        self.assertEqual(len(calls),1)

    def test_expired_global_budget_starts_nothing(self):
        obj, _ = self.fake_run();obj.end=time.monotonic()-1
        with patch.object(R.signal,"setitimer",side_effect=AssertionError("must refuse first")):
            with self.assertRaises(R.BudgetExceeded):
                R.Refresh.arm_timer(obj)

    def fixture_rows(self, plan):
        rows=[]
        for pair in plan["comparisons"]:
            for cid,seconds in [(pair["anchor"],1),(pair["competitor"],2)]:
                for rnd in (1,2,3):
                    for task in plan["tasks"]:
                        row=dict(tag=f"confirm-{cid}-r{rnd}",task=task,new_tokens=10,decode_wall_s=seconds)
                        if row not in rows:rows.append(row)
        return rows

    def test_analysis_completeness_and_residency_guards(self):
        plan,phases=R.load();root=self.folder();rows=self.fixture_rows(plan)
        (root/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        (root/"rungs.json").write_text(json.dumps({k:dict(rung="R0",full_offload=True) for k in ['lcpp-nd','lcpp-df2','lcpp-mtp']}))
        result=analyze(plan,phases,root)
        self.assertTrue(result['complete']);self.assertTrue(result['can_keep_all_tested_leads'])
        (root/"rungs.json").write_text(json.dumps({'lcpp-nd':dict(full_offload=False)}))
        self.assertFalse(analyze(plan,phases,root)['can_keep_all_tested_leads'])
        (root/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows[1:]))
        self.assertFalse(analyze(plan,phases,root)['complete'])

    def test_duplicate_measurements_refuse_pooling(self):
        plan,phases=R.load();root=self.folder();rows=self.fixture_rows(plan);rows.append(rows[0])
        (root/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            analyze(plan,phases,root)


if __name__ == "__main__":
    unittest.main()
