#!/usr/bin/env python3
"""Paper 5 scoring (amendment item 4).

Primary: the upstream LongMemEval judge prompts, verbatim (evaluate_qa.py @
d6dc8b50a2d9ac0c99485ea28fa5755c62414c34, SHA-256 ecce9c4c..., MIT, Copyright (c) 2024
Di Wu), run by gpt-oss-20b (MXFP4, ggml-org @ ef9b12f) on the pinned llama.cpp build,
temperature 0, reasoning effort low; label = 'yes' in content.lower() (upstream rule).
Secondary: the generator as judge (thinking off, greedy, 16 tokens).

The judged response is the generator's visible answer (thinking removed); an
unterminated thinking block leaves an empty answer.

  judge.py --judge gptoss --gen grid.jsonl
  judge.py --judge self   --gen grid.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5common as C  # noqa: E402
from p5run import LlamaServer, Runner, TabbyServer, log, split_thinking  # noqa: E402

JOURNALS = os.path.join(C.WORK, "journals")
GPTOSS = os.path.expanduser("~/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf")
GPTOSS_SHA256 = None  # recorded in testsuite/artifacts/downloads.jsonl at download


# ---- verbatim from upstream evaluate_qa.py (MIT) ---------------------------------
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt
# ---- end verbatim ------------------------------------------------------------------


def judge_prompt(item: dict, response: str) -> str:
    task = item["question_type"]
    if task.startswith("locomo-"):
        task = "single-session-user"      # amendment item 7: the generic template
    return get_anscheck_prompt(task, item["question"], str(item["answer"]), response,
                               abstention="_abs" in item["question_id"])


def label(content) -> bool:
    return "yes" in (content or "").lower()


def gptoss_server() -> LlamaServer:
    return LlamaServer("judge-gptoss20b", 18126, GPTOSS,
                       ["-c", "16384", "-ngl", "99", "--n-cpu-moe", "6", "--jinja", "-np", "1",
                        "--temp", "0"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["gptoss", "self"], required=True)
    ap.add_argument("--gen", required=True, help="generation journal file name in work/journals")
    a = ap.parse_args()
    items = {it["question_id"]: it for it in C.load_json_verified(C.LME_S)}
    items.update({it["question_id"]: it for it in C.locomo_items()})
    gens = []
    with open(os.path.join(JOURNALS, a.gen)) as f:
        for line in f:
            r = json.loads(line)
            if r.get("terminal"):
                gens.append(r)
    journal = os.path.join(JOURNALS, f"judge_{a.judge}__{a.gen}")
    server = gptoss_server() if a.judge == "gptoss" else TabbyServer()
    runner = Runner(server, journal)
    log(f"{len(gens)} generations to judge with {a.judge} -> {journal}")
    fails = 0
    try:
        for n, g in enumerate(gens):
            qid = g["meta"]["question_id"]
            _, answer, _ = split_thinking(g.get("text") if g.get("ok") else "")
            prompt = judge_prompt(items[qid], answer)
            base = f"{g['job_id']}|judge={a.judge}"
            if a.judge == "gptoss":
                job = {"job_id": base, "messages": [{"role": "user", "content": prompt}], "n_prompt": None,
                       "budget": None, "timeout": 600,
                       "params": {"max_tokens": 2048, "temperature": 0.0,
                                  "chat_template_kwargs": {"reasoning_effort": "low"}},
                       "meta": {"question_id": qid, "gen_job_id": g["job_id"], "judge": a.judge}}
                row = runner.run(job, endpoint="/v1/chat/completions")
                if row.get("ok") and not (row.get("text") or "").strip():
                    job2 = dict(job, job_id=base + "|retry")
                    job2["params"] = dict(job["params"], max_tokens=8192)
                    row = runner.run(job2, endpoint="/v1/chat/completions")
                    fails += not (row.get("text") or "").strip()
            else:
                p = C.wrap(prompt, thinking=False)
                job = {"job_id": base, "prompt": p, "n_prompt": C.count_tokens(p), "budget": None,
                       "timeout": 600, "params": {"max_tokens": 16, "temperature": 0.0, "top_k": 1,
                                                  "add_bos_token": False,
                                                  "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]},
                       "meta": {"question_id": qid, "gen_job_id": g["job_id"], "judge": a.judge}}
                row = runner.run(job)
            if n % 100 == 0:
                log(f"{n+1}/{len(gens)} label={label(row.get('text'))} fails={fails}")
    finally:
        server.stop()
    log(f"ALL DONE; judge failures after retry: {fails}")


if __name__ == "__main__":
    main()
