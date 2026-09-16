"""Pure request construction for the isolated follow-up runner."""
import json

from workspace import canonical, digest_text, sha

SUMMARIZER = "qwen38-exl3"


def generation_job(a, work, prompt, info):
    job = a.grid.to_job(prompt, info["completion_cap"])
    job["meta"].update(leg=work.leg, model=info["model"],
                       prompt_manifest_sha256=sha(work.path("PROMPTS_MANIFEST.json")),
                       prompt_sha256=prompt["prompt_sha256"])
    if info["model"] == "qwen3-8b":
        # Unlike TabbyAPI's pinned EXL3 sampler, llama.cpp accepts the seed
        # prescribed by the original protocol. Do not label it ignored.
        seed = job["meta"].pop("seed_recorded_not_applied")
        job["params"]["seed"] = seed
        job["meta"]["seed_applied"] = seed
    return job


def judgment_job(a, work, judge, item, generation, info):
    _, answer, _ = a.runner.split_thinking(generation.get("text") if generation["ok"] else "")
    text = a.judge.judge_prompt(item, answer)
    meta = {"question_id": item["question_id"], "gen_job_id": generation["job_id"], "judge": judge,
            "gen_record_sha256": digest_text(canonical(generation)), "leg": work.leg,
            "model": info["model"], "prompt_manifest_sha256": sha(work.path("PROMPTS_MANIFEST.json"))}
    job = {"job_id": generation["job_id"] + "|judge=" + judge,
           "budget": None, "timeout": 600, "meta": meta}
    if judge == "gptoss":
        job.update(messages=[{"role": "user", "content": text}], n_prompt=None,
                   params={"max_tokens": 2048, "temperature": 0.0,
                           "chat_template_kwargs": {"reasoning_effort": "low"}})
    elif judge == "self":
        raw = a.C.wrap(text, thinking=False, model=info["model"])
        job.update(prompt=raw, n_prompt=a.C.count_tokens(raw, a.C.model_dir(info["model"])),
                   verify_tokens=True, params={"max_tokens": 16, "temperature": 0.0, "top_k": 1, **a.grid.RAW})
    else:
        raise ValueError("unknown judge")
    meta["prompt_sha256"] = a.C.sha256_text(job.get("prompt") or json.dumps(job["messages"]))
    return job


def summary_job(a, work, item, index):
    # Summaries always come from the primary generator (the pinned 27B EXL3 server). For
    # cross-model, S reuses the primary summaries and only sessions the primary never
    # summarized are generated, with this identical request (research/P5_AMENDMENT_2026-10-30.md).
    job = a.SM.summary_job(item, index)
    job["meta"].update(leg=work.leg, model=SUMMARIZER,
                       workspace_sha256=sha(work.path("WORKSPACE.json")),
                       prompt_sha256=a.C.sha256_text(job["prompt"]))
    return job
