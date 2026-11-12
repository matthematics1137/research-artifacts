#!/usr/bin/env python3
"""CPU tests for run_confirm.py: the plan, the commands, the stream client against fake servers, the budget rule.

  /usr/local/bin/zarael-sandbox -C /opt/p8/repository -- bash -c \
      'cd tools/turing/confirm && ../../../engines/analysis-env/bin/python -m unittest -v test_confirm'
"""
import http.server
import json
import os
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_confirm as R  # noqa: E402

PLAN = os.path.join(HERE, "plan.json")


def serve(handler_cls):
    srv = http.server.ThreadingHTTPServer(("localhost", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://localhost:{srv.server_address[1]}"


class FakeLlama(http.server.BaseHTTPRequestHandler):
    """llama-server's /completion stream: content pieces, then a final event with timings."""
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlama.seen.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        n = body["n_predict"]
        for i in range(4):
            self.wfile.write(f"data: {json.dumps({'content': 'tok' * (n // 4), 'stop': False})}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        final = {"content": "", "stop": True, "tokens_predicted": n,
                 "timings": {"prompt_n": 31, "prompt_ms": 20.0, "predicted_n": n, "predicted_ms": 150.0}}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())


class FakeChat(http.server.BaseHTTPRequestHandler):
    """NInferno's chat stream: deltas, a finish chunk, a usage chunk with timings, then [DONE]."""
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "Qwen3.8-27B-UD-IQ2_XXS"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeChat.seen.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        n = body["max_tokens"]
        for i in range(3):
            ev = {"choices": [{"delta": {"content": "x" * 5}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]})}\n\n".encode())
        usage = {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": n},
                 "timings": {"cache_n": 0, "prompt_n": 40, "predicted_n": n, "predicted_ms": 99.0}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())


class TestClient(unittest.TestCase):
    def test_completion_stream(self):
        srv, base = serve(FakeLlama)
        try:
            body = R.request_body("completion", "hello", 256, {"temperature": 0.8, "min_p": 0.08, "top_k": 0,
                                                               "top_p": 1.0})
            self.assertTrue(body["prompt"].startswith("<|im_start|>user\nhello<|im_end|>"))
            self.assertTrue(body["prompt"].endswith("<think>\n\n</think>\n\n"))
            self.assertTrue(body["ignore_eos"] and body["stream"] and body["cache_prompt"])
            row = R.stream_answer(base, "completion", body)
            self.assertEqual(row["new_tokens"], 256)
            self.assertEqual(row["chunks"], 4)
            self.assertGreater(row["decode_wall_s"], 0.1)    # three gaps of 50 ms between first and last piece
            self.assertLess(row["decode_wall_s"], 1.0)
            self.assertEqual(row["server_timings"]["predicted_ms"], 150.0)
            self.assertEqual(row["prompt_tokens"], 31)
            self.assertIsNone(R.wait_ready(base, type("P", (), {"poll": lambda self: None})(), 5))
        finally:
            srv.shutdown()

    def test_chat_stream(self):
        srv, base = serve(FakeChat)
        try:
            mid = R.model_id(base)
            self.assertEqual(mid, "Qwen3.8-27B-UD-IQ2_XXS")
            body = R.request_body("chat", "hello", 256, {"temperature": 0.8, "min_p": 0.08, "top_k": 0,
                                                         "top_p": 1.0}, mid)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
            row = R.stream_answer(base, "chat", body)
            self.assertEqual(row["new_tokens"], 256)
            self.assertEqual(row["finish"], "length")
            self.assertEqual(row["prompt_tokens"], 40)
            self.assertEqual(FakeChat.seen[-1][0], "/v1/chat/completions")
        finally:
            srv.shutdown()


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.lab = tempfile.mkdtemp()
        self.plan = R.load_plan(PLAN, self.lab)

    def test_conditions_and_commands(self):
        p = self.plan
        ids = [c["id"] for c in p["conditions"]]
        self.assertEqual(len(ids), 29)
        self.assertEqual(sorted({c["priority"] for c in p["conditions"]}), [1, 2, 3, 4])
        keep = {c["id"] for c in p["conditions"] if c["priority"] == 1}
        for must in ("w13-165-mtp-0", "stock166-nd-0", "w13-166-mtp-0", "nif-xxs-28k", "lcpp9-s-mtp-28k"):
            self.assertIn(must, keep)
        out = tempfile.mkdtemp()
        run = R.Run(p, out, dry=True)
        c = next(x for x in p["conditions"] if x["id"] == "w13-165-mtp-28k")
        cmd, env = run.census_cmd(c, "t")
        self.assertIn("--fill", cmd)
        self.assertEqual(cmd[cmd.index("-cq") + 1], "8,8")
        self.assertEqual(cmd[-3:], ["-mtp", "-ndt", "6", "-dds"][-3:])
        self.assertEqual(env["EXL3_GEMV_SK"], "2")
        self.assertEqual(env["EXL3_INT8_SQ_MAX_M"], "2")
        self.assertTrue(env["PYTHONPATH"].endswith("overlay/exl3-w13"))
        c = next(x for x in p["conditions"] if x["id"] == "w13-165-nd-0")
        cmd, env = run.census_cmd(c, "t")
        self.assertNotIn("--fill", cmd)
        self.assertNotIn("-mtp", cmd)
        self.assertEqual((env["EXL3_GEMV_SK"], env["EXL3_INT8_SQ_MAX_M"]), ("1", "1"))
        c = next(x for x in p["conditions"] if x["id"] == "raf44-mtp-0")
        cmd, env = run.census_cmd(c, "t")
        self.assertEqual(cmd[cmd.index("-cq") + 1], "4,4")
        self.assertTrue(env["CUDA_HOME"].startswith(self.lab))
        c = next(x for x in p["conditions"] if x["id"] == "lcpp9-s-mtp-0")
        cmd1, env = run.server_cmd(c)
        cmd2, _ = run.server_cmd(c)
        self.assertNotEqual(cmd1[cmd1.index("--port") + 1], cmd2[cmd2.index("--port") + 1])
        self.assertIn(os.path.join(self.lab, "models/gguf/mtp-Qwen3.8-27B-Q4_0.gguf"), cmd1)
        self.assertNotIn("{", " ".join(cmd1))
        c = next(x for x in p["conditions"] if x["id"] == "nif-s-28k")
        cmd, env = run.server_cmd(c)
        self.assertEqual(cmd[1], os.path.join(self.lab, "models/gguf/Qwen3.8-27B-UD-IQ2_S.gguf"))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertTrue(env["HOME"].startswith(self.lab))

    def test_budget_cut_order(self):
        out = tempfile.mkdtemp()
        run = R.Run(self.plan, out, dry=True)
        kept = [c["id"] for c in self.plan["conditions"]]
        run.durations = {k: 400.0 for k in kept}   # 29 x 400 s x 2 rounds: far over the cap
        kept = run.cut_to_cap(2, list(kept), 0)
        prios = {c["id"]: c["priority"] for c in self.plan["conditions"]}
        self.assertTrue(all(prios[k] == 1 for k in kept))
        cut = [json.loads(x) for x in open(os.path.join(out, "journal.jsonl")) if '"cut"' in x]
        first = [r["condition"] for r in cut][:3]
        self.assertTrue(all(prios[c] == 4 for c in first[:2]))
        run2 = R.Run(self.plan, tempfile.mkdtemp(), dry=True)
        run2.durations = {k: 30.0 for k in prios}
        self.assertEqual(len(run2.cut_to_cap(2, list(prios), 0)), 29)

    def test_rotation(self):
        ordered = list("ABCD")
        n = len(ordered)
        self.assertEqual([ordered[(i + 2 - 1) % n] for i in range(n)], list("BCDA"))


class TestAnalysis(unittest.TestCase):
    def test_pooled_and_rounds(self):
        import subprocess
        d = tempfile.mkdtemp()
        rows = os.path.join(d, "rows.jsonl")
        with open(rows, "w") as f:
            for cond, rate in (("w13-165-nd-0", 48.0), ("stock165-nd-0", 25.0), ("nif-xxs-0", 45.0),
                               ("w13-165-mtp-0", 70.0), ("lcpp9-xxs-df-0", 55.0)):
                for rnd in (1, 2, 3):
                    for task in ("code", "math", "prose"):
                        r = rate * (1 + 0.01 * rnd)
                        f.write(json.dumps({"tag": f"confirm-{cond}-r{rnd}", "task": task, "new_tokens": 256,
                                            "decode_wall_s": 256 / r}) + "\n")
        out = subprocess.run([sys.executable, os.path.join(HERE, "analyze_confirm.py"), rows],
                             capture_output=True, text=True, check=True).stdout
        res = json.load(open(os.path.join(d, "results.json")))
        self.assertAlmostEqual(res["pooled"]["w13-165-nd-0"]["rounds"]["1"] / 48.0, 1.01, places=6)
        c = res["comparisons"]["over_stock_nd_0"]
        self.assertTrue(c["faster_every_round"])
        self.assertAlmostEqual(c["ratio"], 48 / 25, places=6)
        self.assertTrue(res["fastest"]["nd-0"]["holds"])
        self.assertEqual(res["fastest"]["nd-0"]["rivals"], ["nif-xxs-0"])
        self.assertEqual(res["fastest"]["mtp-0"]["rivals"], ["lcpp9-xxs-df-0"])
        self.assertIn("within the floor", out)


if __name__ == "__main__":
    unittest.main()
