"""Full synthetic execution/analysis in a disposable workspace, with no servers.

Every HTTP response, server start/stop, temperature, primary prerequisite and
model fingerprint is a fixture. The real runner, append-only journals,
request hashes, token gates, completion receipts and analyzer are exercised.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze as A
from legacy import algorithms
import run as R
from test_workspace import items_for, prompts_for
from workspace import ROOT, Workspace, sha


class FakeServer:
    name, base = "synthetic-no-network-server", "not-an-http-endpoint"
    def __init__(self):
        self.proc = None
    def start(self):
        self.proc = SimpleNamespace(poll=lambda: None)
    def stop(self):
        self.proc = None
    def count_tokens_server(self, prompt):
        return 25
    def identity(self):
        return {"server": self.name, "synthetic": True}


class IntegrationTests(unittest.TestCase):
    def test_complete_cross_model_resume_and_analysis_without_real_network(self):
        with tempfile.TemporaryDirectory(prefix="p5-followup-synthetic-integration-") as tmp:
            root = Path(tmp)
            source = root / "synthetic_source.json"
            source.write_text('{"synthetic":true}\n')
            inputs = {str(source): sha(source)}
            work = Workspace("cross-model", root)
            items = items_for("cross-model")
            work.initialize(items, 8192, inputs)
            for arm in ("T", "S", "C", "M"):
                work.write_prompts(arm, prompts_for("cross-model", arm, items), inputs)
            work.freeze()
            requests = []

            def fake_http(url, body, timeout):
                requests.append(body)
                if "messages" in body:
                    return {"choices": [{"message": {"content": "yes"}, "finish_reason": "stop"}],
                            "usage": {"completion_tokens": 1}}
                text = "thinking</think>synthetic" if body["prompt"].startswith("Synthetic prompt") else "yes"
                return {"choices": [{"text": text, "finish_reason": "stop"}], "usage": {"prompt_tokens": 25}}

            @contextmanager
            def fake_algorithms(ws):
                with algorithms(SimpleNamespace(root=ROOT, base=ws.base, leg=ws.leg)) as a:
                    a.C.count_tokens = lambda text, *args: 25
                    a.runner.http_json = fake_http
                    a.runner.gpu_temp = lambda: 0
                    a.runner.log = lambda text: None
                    yield a

            def fake_artifacts(a, stage, info):
                return {"served_model": "gptoss" if stage == "judge-gptoss" else info["model"],
                        "engine": "synthetic", "revision": "synthetic", "files": inputs}

            with patch.object(R, "algorithms", fake_algorithms), \
                 patch.object(R, "verified_primary", return_value=({}, inputs)), \
                 patch.object(R, "preflight", return_value={"host": "synthetic-no-hardware"}), \
                 patch.object(R, "artifacts", fake_artifacts), \
                 patch.object(R, "server_for", side_effect=lambda *args: FakeServer()):
                generated = R.execute(work, "generate", live=True, idle_seconds=0.)
                self.assertTrue(generated["complete"])
                self.assertEqual(len(requests), 2000)
                original = sha(work.path("journals/grid.jsonl"))
                resumed = R.execute(work, "generate", live=True, idle_seconds=0.)
                self.assertEqual(resumed, generated)
                self.assertEqual(len(requests), 2000)
                self.assertEqual(sha(work.path("journals/grid.jsonl")), original)
                R.execute(work, "judge-gptoss", live=True, idle_seconds=0.)
                R.execute(work, "judge-self", live=True, idle_seconds=0.)
            self.assertEqual(len(requests), 6000)
            result = A.analyze(work)
            self.assertEqual(result["cells"]["T"]["correct"], 500)
            self.assertEqual(result["contrasts_vs_T"]["S"]["difference"], 0.)
            self.assertEqual(result["contrasts_vs_T"]["S"]["mcnemar_p_unadjusted"], 1.)
            # Corrupting a finished-stage marker must not silently replace a
            # previously good analysis file or pass verification.
            analysis_path = work.path("analysis/analysis.json")
            before = sha(analysis_path)
            receipt = work.path("execution/judge-self-finished.json")
            record = json.loads(receipt.read_text())
            record["journal_sha256"] = "0" * 64
            receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "completion receipt"):
                A.analyze(work)
            self.assertEqual(sha(analysis_path), before)


if __name__ == "__main__":
    unittest.main()
