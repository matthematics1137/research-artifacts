"""CPU-only synthetic journals in temporary directories. Never launch a server or GPU."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze as A
import p6common as P
import p6evidence as E
import run_cells as R


def dump(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def dump_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


class Campaign:
    """Small fake text with the full registered 24 x (200 + 100) outcome shape."""
    def __enter__(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="p6-evidence-test-")))
        for name in ("journals", "logs", "token_counts"):
            (self.root / name).mkdir()
        longctx = []
        for i in range(50):
            for family in ("single", "multikey", "multivalue", "vartrack"):
                content = f"SYNTHETIC TEST ONLY {family} {i}: a fictional fact."
                longctx.append({"id": f"{family}-{i:02d}", "family": family, "content": content,
                    "sha256": P.sha256_text(content), "haystack_tokens": 16360, "content_tokens": 16400,
                    "prompt_tokens_qwen38_nothink": 16410, "answers": ["7654321"], "distractors": ["1111111"],
                    "asked_depth": (i + .5) / 50, "answer_depths": {"7654321": (i + .5) / 50}})
        gsm = [{"id": f"gsm-{i}", "question": f"SYNTHETIC TEST {i}: 1+1?", "answer_number": 2.0} for i in range(100)]
        self.longctx, self.gsm = longctx, gsm
        lcfile, gsmfile, manifest = self.root / "items.jsonl", self.root / "gsm.jsonl", self.root / "manifest.json"
        dump_rows(lcfile, longctx)
        dump_rows(gsmfile, gsm)
        dump(manifest, {"seed": 20260919, "output": {"sha256": P.sha256_file(lcfile)}})
        for name, value in {"WORK": str(self.root), "JOURNALS": str(self.root / "journals"), "LOGS": str(self.root / "logs"),
                            "LONGCTX_ITEMS": str(lcfile), "LONGCTX_MANIFEST": str(manifest),
                            "GSM8K_ITEMS": str(gsmfile), "GSM8K_SHA256": P.sha256_file(gsmfile)}.items():
            self.stack.enter_context(patch.object(P, name, value))
        self.stack.enter_context(patch.object(P, "load_gsm8k", side_effect=lambda **kw: deepcopy(gsm)))
        self.stack.enter_context(patch.object(P, "receipt_for", side_effect=self.receipt))
        self.hashes, self.inputs = E.harness_hashes(), E.input_identity()
        self.rows = {}
        for index, cell in enumerate(P.all_cells()):
            start = self.start(cell, index)
            rows = [start]
            jobs = R.build_jobs(cell[0], R.TASKS, longctx, gsm, R.DEFAULT_TIMEOUT)
            for i, job in enumerate(jobs):
                # Paired flips, negative contrasts and nonzero variance; no real responses.
                cutoff = 8 - P.TIERS.index(cell[1]) - P.KV_ORDER.index(cell[2]) - (cell[0] == P.GQA)
                correct = i % 10 < cutoff
                answer = ("7654321" if correct else "unknown") if job["task"] == "longctx" else ("Answer: 2" if correct else "Answer: 3")
                row = {"job_id": job["job_id"], "task": job["task"], "item_id": job["item"]["id"],
                       "terminal": True, "ok": True, "attempt": 0, "content": answer, "reasoning_content": None,
                       "finish_reason": "stop", "usage": {"prompt_tokens": 16410 if job["task"] == "longctx" else 50,
                                                           "completion_tokens": 3},
                       "seconds": .25, "request": job["stored_request"], "meta": deepcopy(job["meta"]),
                       "request_sha256": P.sha256_text(json.dumps(job["body"], sort_keys=True)),
                       "harness": self.hashes, "server": {k: start[k] for k in ("pid", "start_seq", "instance_id")}}
                if row["meta"].get("prompt_tokens_expected") is not None:
                    row["meta"]["prompt_tokens_delta"] = 0
                row["score"] = P.score_row(job["task"], row, job["item"])
                rows.append(row)
            name = P.cell_name(*cell)
            self.rows[name] = rows
            dump_rows(self.root / "journals" / f"{name}.jsonl", rows)
            tc = self.root / "token_counts" / f"{cell[0]}.json"
            if not tc.exists():
                dump(tc, {"schema": "p6-token-counts-v1", "model": cell[0], "cell": name,
                          "items_sha256": self.inputs["longctx_sha256"], "harness": self.hashes,
                          "server": {k: start[k] for k in ("pid", "start_seq", "instance_id")}, "props": start["props"],
                          "items": {it["id"]: {"haystack": 16360, "content": 16400} for it in longctx}})
        return self

    @staticmethod
    def receipt(path):
        model = P.HYBRID if "Qwen3.8" in str(path) else P.GQA
        pin = P.PINNED_SHA256.get(Path(path).name)
        return {"repo": P.MODELS[model]["repo"], "revision": P.MODELS[model]["revision"],
                "sha256": pin[0] if pin else P.sha256_text(Path(path).name), "bytes": 100, "source": "SYNTHETIC FIXTURE"}

    def start(self, cell, index):
        model, tier, kv = cell
        hybrid = model == P.HYBRID
        count, cached = (65, 16) if hybrid else (37, 36)
        ctk, ctv = P.KV_TYPES[kv]
        text = (f"SYNTHETIC TEST ONLY\nbuild 1 (035e227)\n"
            "common_params_fit_impl: no changes needed\ncommon_fit_params: successfully fit params\n"
            f"load_tensors: offloaded {count}/{count} layers to GPU\n"
            "load_tensors: CUDA0 model buffer size = 7000.00 MiB\n"
            "llama_context: n_ctx = 20480\nllama_context: flash_attn = enabled\n"
            "llama_kv_cache: CUDA0 KV buffer size = 1280.00 MiB\n"
            f"llama_kv_cache: size = 1280.00 MiB (20480 cells, {cached} layers, 1 seqs), K ({ctk}): 640.00 MiB, V ({ctv}): 640.00 MiB\n"
            "sched_reserve: CUDA0 compute buffer size = 100.00 MiB\n")
        if hybrid:
            text += ("llama_memory_recurrent: CUDA0 RS buffer size = 150.00 MiB\n"
                     "llama_memory_recurrent: size = 150.00 MiB (1 cells, 64 layers, 1 seqs), R (f32): 6.00 MiB, S (f32): 144.00 MiB\n")
        log_path = self.root / "logs" / f"{P.cell_name(*cell)}.log"
        log_path.write_text(text, encoding="utf-8")
        gid = self.receipt(P.gguf_path(model, tier))
        return {"kind": "server_start", "instance_id": f"{index + 1:032x}", "pid": index + 100, "start_seq": 1,
                "cell": {"model": model, "tier": tier, "kv": kv, "ctk": ctk, "ctv": ctv, "arch": P.MODELS[model]["arch"]},
                "gguf": {"path": P.gguf_path(model, tier), "receipt": gid, "sha256": gid["sha256"], "bytes": 100, "verified": True},
                "engine": {"commit": P.ENGINE_COMMIT, "commit_is_pinned": True, "tracked_changes": 0,
                           "cmake": {"GGML_CUDA_FA_ALL_QUANTS": "ON", "GGML_CUDA": "ON", "BUILD_SHARED_LIBS": "ON",
                                     "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "89"},
                           "sha256": {key: "f" * 64 for key in E.BINARIES}},
                "argv": R.server_argv(*cell), "cmd": ["nice", "-n", "5"] + R.server_argv(*cell),
                "inputs": self.inputs, "harness": self.hashes, "placement_changed": False,
                "machine": {"gpu_name": "NVIDIA GeForce RTX 4080 Laptop GPU", "gpu_uuid": "GPU-SYNTHETIC-ONLY",
                            "gpu_total_mib": 12282, "driver_version": "SYNTHETIC", "gpu_pci_bus": "SYNTHETIC"},
                "load": R.parse_server_log(text), "log_path": str(log_path), "load_log_bytes": len(text.encode()),
                "load_log_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "decoding": {"temperature": 0.0, "cache_prompt": False, "longctx_max_tokens": 128,
                             "gsm8k_max_tokens": 8192, "timeouts_s": R.DEFAULT_TIMEOUT},
                "gpu_loaded": {"process_mib": 8800},
                "props": {"build_info": "b1-035e227", "model_path": P.gguf_path(model, tier), "n_ctx": 20480, "total_slots": 1,
                          "chat_template_sha256": "e" * 64, "chat_template_chars": 100}}

    def __exit__(self, *args):
        self.stack.__exit__(*args)


class EvidenceTests(unittest.TestCase):
    def test_child_cuda_path_and_real_normal_verbosity_build_boundary(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/synthetic-parent-path"}):
            _, env, _ = R.P6Server(P.all_cells()[0]).cmd_env_cwd()
            self.assertEqual(env["LD_LIBRARY_PATH"], str(Path.home() / "cuda-12.6/lib64"))
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/synthetic-parent-path")
            self.assertEqual(env["LLAMA_ARG_LOG_VERBOSITY"], "4")      # placement lines are level-4 here
            self.assertNotIn("LLAMA_ARG_LOG_VERBOSITY", os.environ)
        with Campaign() as c:
            cell = P.all_cells()[0]
            row = deepcopy(c.rows[P.cell_name(*cell)][0])
            log = Path(row["log_path"])
            # Normal verbosity has no trace-only common_params_print_info line.
            text = log.read_text().replace("build 1 (035e227)\n", "")
            log.write_text(text)
            row.update(load=R.parse_server_log(text), load_log_bytes=len(text.encode()),
                       load_log_sha256=hashlib.sha256(text.encode()).hexdigest())
            self.assertIsNone(row["load"]["build"])
            E.validate_identity(row, cell, hashes=c.hashes, inputs=c.inputs)
            for build in (None, "", "b1-deadbee", "b1-035e227-dirty"):
                row["props"]["build_info"] = build
                with self.subTest(build=build), self.assertRaisesRegex(ValueError, "engine build in props"):
                    E.validate_identity(row, cell, hashes=c.hashes, inputs=c.inputs)

    def test_strict_json(self):
        for text in ('{"a": 1, "a": 2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                E.loads(text)

    def test_full_7200_outcome_replay_and_independent_statistics(self):
        with Campaign() as c:
            analysis = A.analyse(P.JOURNALS, P.LONGCTX_ITEMS, 100000, 20260919, 10)
            self.assertEqual(analysis["flags"], [])
            result = E.verify_analysis(analysis)
            self.assertEqual(result["successful_outcomes"], 7200)
            self.assertFalse(result["human_approval"])
            broken = deepcopy(analysis)
            broken["estimands"]["longctx"]["D"][P.HYBRID]["Q2_K_XL"]["hi_pts"] += .1
            with self.assertRaisesRegex(ValueError, "numeric mismatch"):
                E.verify_analysis(broken)
            audits = {name: {"outcomes": {row["job_id"]: row for row in rows if "job_id" in row}}
                      for name, rows in c.rows.items()}
            with self.assertRaisesRegex(ValueError, "independent paired hi_pts"):
                E.independent_statistics(broken, audits)
            journal = next((c.root / "journals").glob("*.jsonl"))
            with journal.open("a") as stream:
                stream.write('{"kind":"pass_end"}\n')
            with self.assertRaisesRegex(ValueError, "evidence has changed"):
                E.verify_analysis(analysis)

    def test_journal_corruptions_and_resume(self):
        with Campaign() as c:
            cell = P.all_cells()[0]
            base = c.rows[P.cell_name(*cell)][:2]
            path = c.root / "journal-unit.jsonl"
            def audit(rows, complete=False):
                dump_rows(path, rows)
                return E.audit_journal(path, cell, c.longctx, c.gsm, complete=complete,
                                       hashes=c.hashes, inputs=c.inputs)
            self.assertEqual(len(audit(base)["outcomes"]), 1)
            with self.assertRaisesRegex(ValueError, "missing"):
                audit(base, complete=True)
            failure = {k: deepcopy(base[1][k]) for k in ("job_id", "task", "item_id", "request", "harness", "meta")}
            failure.update(terminal=True, ok=False, error="test failed request", attempt=0)
            self.assertEqual(len(audit([base[0], failure, base[1]])["outcomes"]), 1)
            cases = [
                lambda rows: rows.append(deepcopy(rows[1])),
                lambda rows: rows[1].update(ok="true"),
                lambda rows: rows[1]["score"].update(correct=1),
                lambda rows: rows[1]["request"].update(temperature=1),
                lambda rows: rows[1]["request"].update(cache_prompt=0),
                lambda rows: rows[1].update(request_sha256="0"*64),
                lambda rows: rows[1]["server"].update(instance_id="b"*32),
                lambda rows: rows[1]["meta"].update(prompt_tokens_delta=9),
                lambda rows: rows[1]["usage"].update(completion_tokens=8192),
                lambda rows: rows[0]["harness"].update(unknown="a"*64),
                lambda rows: rows[0]["engine"].update(tracked_changes=1),
                lambda rows: rows[0]["engine"]["cmake"].update(CMAKE_BUILD_TYPE="Debug"),
                lambda rows: rows[0]["load"].update(fully_resident=False),
                lambda rows: rows[0].update(load_log_sha256="a"*64),
                lambda rows: rows[0]["props"].update(n_ctx=4096),
                lambda rows: rows[0]["props"].update(build_info="b1-deadbee"),
                lambda rows: rows[0]["props"].update(build_info="b2-035e227"),
                lambda rows: rows[0]["machine"].update(gpu_name="some other GPU"),
                lambda rows: rows[1].update(content="wrong answer"),
                lambda rows: rows.insert(1, {"kind": "pause"}),
                lambda rows: rows.append({"kind": "invented"}),
                lambda rows: rows[1].update(job_id="longctx|not-in-study"),
            ]
            for i, corrupt in enumerate(cases):
                with self.subTest(case=i):
                    rows = deepcopy(base)
                    corrupt(rows)
                    with self.assertRaises((ValueError, KeyError)):
                        audit(rows)

    def test_wrong_model_receipt_rejected(self):
        with Campaign() as c:
            first = c.rows[P.cell_name(*P.all_cells()[0])][0]
            rec = deepcopy(first["gguf"]["receipt"])
            rec["revision"] = "a" * 40
            with patch.object(P, "receipt_for", return_value=rec), self.assertRaisesRegex(ValueError, "receipt revision"):
                E.validate_identity(first, P.all_cells()[0], hashes=c.hashes, inputs=c.inputs)

    def test_cached_token_counts_identity(self):
        with Campaign() as c:
            data = E.read_json(c.root / "token_counts" / f"{P.HYBRID}.json")
            E.validate_token_counts(data, P.HYBRID, c.longctx)
            for key, value in (("items_sha256", "a"*64), ("model", P.GQA), ("harness", {}), ("items", {})):
                bad = deepcopy(data)
                bad[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    E.validate_token_counts(bad, P.HYBRID, c.longctx)

    def test_first_invocation_placement_signature_covers_more_than_layers(self):
        base = {"offloaded_layers": 65, "total_layers": 65, "fully_resident": True,
                "buffers_mib": {"model": {"CUDA0": 7000}, "kv": {"CUDA0": 1280}, "rs": {"CUDA0": 150}},
                "fit": {"overflowing": 0}}
        other = deepcopy(base)
        other["buffers_mib"]["kv"] = {"CUDA0": 1200, "CPU": 80}
        self.assertNotEqual(R.placement_signature(base), R.placement_signature(other))

    def test_changed_raw_log_invalidates_evidence_snapshot(self):
        with Campaign() as c:
            before = E.evidence_snapshot(P.JOURNALS)
            log = next((c.root / "logs").glob("*.log"))
            log.write_bytes(log.read_bytes().replace(b"7000.00", b"7100.00"))
            self.assertNotEqual(before, E.evidence_snapshot(P.JOURNALS))

    def test_gentle_pacing_does_not_delay_cached_results(self):
        with tempfile.TemporaryDirectory(prefix="p6-pacing-test-") as td:
            from types import SimpleNamespace
            class Fake:
                name, base, fresh = "fake", "http://never-opened", False
                proc = SimpleNamespace(poll=lambda: None)
                def identity(self):
                    return {"pid": 42, "start_seq": 1, "instance_id": "a"*32}
            response = {"choices": [{"message": {"content": "7654321"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            item = {"id": "fake", "family": "single", "content": "test", "sha256": P.sha256_text("test"),
                    "haystack_tokens": 1, "prompt_tokens_qwen38_nothink": 1, "answers": ["7654321"]}
            job = R.build_jobs(P.HYBRID, ["longctx"], [item], [], R.DEFAULT_TIMEOUT)[0]
            waits = []
            runner = R.P6Runner(Fake(), str(Path(td) / "journal.jsonl"), post=lambda *a, **kw: response,
                               temp_fn=lambda: 40, sleep=waits.append, pause_file=str(Path(td) / "PAUSE"))
            runner.run(job)
            self.assertEqual(waits, [10])
            runner.run(job)
            self.assertEqual(waits, [10])

    def test_ports_include_secondary_generators_and_judges(self):
        for busy in (18125, 18126, 18127, 18130):
            with self.subTest(port=busy), patch.object(R.socket, "socket") as socket:
                socket.return_value.__enter__.return_value.connect_ex.side_effect = lambda addr: 0 if addr[1] == busy else 1
                with self.assertRaisesRegex(SystemExit, str(busy)):
                    R.assert_study_ports_idle()

    def test_active_primary_unit_blocks_even_if_ports_are_idle(self):
        with patch.object(R.socket, "socket") as socket, patch.object(R.subprocess, "run") as run:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            run.return_value.stdout = "active\n"
            with self.assertRaisesRegex(SystemExit, "p5-phase1"):
                R.assert_study_ports_idle()

    def test_token_counts_are_create_only_and_isolated(self):
        with Campaign() as c:
            model = P.HYBRID
            canonical = c.root / "token_counts" / f"{model}.json"
            original = canonical.read_bytes()
            first = c.rows[P.cell_name(*P.all_cells()[0])][0]
            class Fake:
                cell, base = P.all_cells()[0], "http://never-opened"
                def identity(self):
                    return {k: first[k] for k in ("pid", "start_seq", "instance_id")}
                def props(self):
                    return first["props"]
            it = deepcopy(c.longctx[0])
            it["haystack_span"] = [0, len(it["content"])]
            destination = c.root / "separate-smoke" / "token_counts"
            path = R.tokenize_pass(Fake(), model, [it], post=lambda *a, **kw: {"tokens": [1, 2, 3]}, counts_dir=destination)
            self.assertEqual(Path(path).parent, destination)
            self.assertEqual(canonical.read_bytes(), original)
            bad = E.read_json(path)
            bad["items_sha256"] = "a" * 64
            dump(path, bad)
            with self.assertRaisesRegex(ValueError, "different items"):
                R.tokenize_pass(Fake(), model, [it], post=lambda *a, **kw: self.fail("must not tokenize"), counts_dir=destination)
            self.assertEqual(E.read_json(path), bad)


if __name__ == "__main__":
    unittest.main()
