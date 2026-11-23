"""Paper 5 shared definitions: data, the upstream LongMemEval reader rendering,
chat-template wrapping, and served-tokenizer counting.

Design: research/design-overflow-policies.md (frozen 2026-09-01; amendment and
clarification 2026-09-19). Standard library + `tokenizers` only, so the module
imports in both the exllamav3 and the LLMLingua environments.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "testsuite", "paper5_data")
LME_S = os.path.join(DATA, "longmemeval", "longmemeval_s_cleaned.json")
LME_ORACLE = os.path.join(DATA, "longmemeval", "longmemeval_oracle.json")
LOCOMO = os.path.join(DATA, "locomo", "locomo10.json")
WORK = os.path.join(REPO, "testsuite", "paper5", "work")          # gitignored bulk outputs
MODEL_DIR = os.path.expanduser("~/models/qwen38-exl3-2.0")

DATA_SHA256 = {
    LME_S: "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    LME_ORACLE: "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
    LOCOMO: "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
}

BUDGETS = (1024, 4096)

# Upstream LongMemEval run_generation.py @ d6dc8b50a2d9ac0c99485ea28fa5755c62414c34,
# non-CoT templates, copied verbatim.
READER_TEMPLATE = (
    "I will give you several history chats between you and a user. Please answer the "
    "question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\n"
    "Current Date: {}\nQuestion: {}\nAnswer:"
)
# Arm E: upstream's no-retrieval prompt is the bare question; the amendment adds the
# question date, in the same "Current Date / Question / Answer:" block every arm uses.
EMPTY_TEMPLATE = "Current Date: {}\nQuestion: {}\nAnswer:"

# Chat-template wrapping for ONE user message, each verified against
# transformers.apply_chat_template on that model's own template.
# Qwen3.8 with reasoning_effort=medium injects no system message and trims the content;
# Qwen3-8B (the cross-model check, amendment item 6) does not trim and does not prefill <think>.
WRAP_THINK = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n")
WRAP_NOTHINK = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

GEN_MODELS = {
    "qwen38-exl3": {"dir": MODEL_DIR, "strip": True,
                    "think": WRAP_THINK, "nothink": WRAP_NOTHINK},
    "qwen3-8b": {"dir": os.path.expanduser("~/models/qwen3-8b"), "strip": False,
                 "think": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
                 "nothink": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")},
}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_json_verified(path: str):
    got = sha256_file(path)
    if got != DATA_SHA256[path]:
        raise SystemExit(f"data hash mismatch for {path}: {got}")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=4)
def tokenizer(model_dir: str = MODEL_DIR):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def count_tokens(text: str, model_dir: str = MODEL_DIR) -> int:
    return len(tokenizer(model_dir).encode(text, add_special_tokens=False).ids)


def wrap(user_content: str, thinking: bool = True, model: str = "qwen38-exl3") -> str:
    m = GEN_MODELS[model]
    pre, post = m["think"] if thinking else m["nothink"]
    return pre + (user_content.strip() if m["strip"] else user_content) + post


def model_dir(model: str = "qwen38-exl3") -> str:
    return GEN_MODELS[model]["dir"]


# ---------------------------------------------------------------- rendering
def turn_text(turn: dict) -> str:
    # upstream nl format: "\n\n{role}: {content.strip()}"
    return "\n\n{}: {}".format(turn["role"], turn["content"].strip())


def session_block(index: int, date: str, body: str) -> str:
    # upstream orig-session layout
    return "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(index, date, body)


def render_sessions(sessions: list[tuple[str, str]]) -> str:
    """sessions: chronologically ordered (date, body) pairs -> history string."""
    return "".join(session_block(i + 1, d, b) for i, (d, b) in enumerate(sessions))


def reader_prompt(history: str, question_date: str, question: str) -> str:
    return READER_TEMPLATE.format(history, question_date, question)


def empty_prompt(question_date: str, question: str) -> str:
    return EMPTY_TEMPLATE.format(question_date, question)


def chronological(item: dict) -> list[int]:
    """Session indices sorted by date string (stable), as upstream sorts retrieved chunks.
    LoCoMo pseudo-items carry an explicit order (their dates are not sortable strings)."""
    if "chrono_order" in item:
        return list(item["chrono_order"])
    dates = item["haystack_dates"]
    return sorted(range(len(dates)), key=lambda i: dates[i])


def full_history(item: dict) -> str:
    order = chronological(item)
    return render_sessions([(item["haystack_dates"][i],
                             "".join(turn_text(t) for t in item["haystack_sessions"][i]))
                            for i in order])


def request_seed(item_id: str, arm: str, budget) -> int:
    return int(hashlib.sha256(f"{item_id}|{arm}|{budget}".encode()).hexdigest()[:8], 16)


def evidence_turns(item: dict) -> list[tuple[int, int]]:
    """(session index, turn index) of every has_answer turn."""
    out = []
    for si, s in enumerate(item["haystack_sessions"]):
        for ti, t in enumerate(s):
            if t.get("has_answer"):
                out.append((si, ti))
    return out


# ---------------------------------------------------------------- LoCoMo
def locomo_items() -> list[dict]:
    """LoCoMo QA as pseudo-items in the LongMemEval shape. Speaker names are the roles;
    image turns contribute their text and blip_caption only (2026-09-01 amendment 3).
    The question date is the last session's date (LoCoMo gives none)."""
    convs = load_json_verified(LOCOMO)
    out = []
    for conv in convs:
        c = conv["conversation"]
        keys = sorted((k for k in c if k.startswith("session_") and not k.endswith("date_time")),
                      key=lambda k: int(k.split("_")[1]))
        sessions, dates = [], []
        for k in keys:
            turns = []
            for t in c[k]:
                text = t.get("text", "")
                if t.get("blip_caption"):
                    text = (text + " [image: " + t["blip_caption"] + "]").strip()
                turns.append({"role": t["speaker"], "content": text, "dia_id": t.get("dia_id")})
            sessions.append(turns)
            dates.append(c.get(k + "_date_time", ""))
        for qi, qa in enumerate(conv["qa"]):
            ev = set(qa.get("evidence") or [])
            sess = [[dict(t, has_answer=(t.get("dia_id") in ev)) for t in s] for s in sessions]
            out.append({"question_id": f"locomo-{conv['sample_id']}-{qi}", "question_type": f"locomo-cat{qa['category']}",
                        "category": qa["category"], "question": qa["question"],
                        "answer": str(qa.get("answer", qa.get("adversarial_answer", ""))),
                        "question_date": dates[-1], "haystack_sessions": sess, "haystack_dates": dates,
                        "haystack_session_ids": [f"{conv['sample_id']}-{k}" for k in keys],
                        "answer_session_ids": [], "chrono_order": list(range(len(keys)))})
    return out


def locomo_subset(items: list[dict]) -> list[dict]:
    """Frozen 200-item subset (2026-09-01 amendment 3): categories 1/2/4 at 70/70/60,
    ascending SHA-256(question_id) within category."""
    quota = {1: 70, 2: 70, 4: 60}
    out = []
    for cat, k in quota.items():
        pool = sorted((it for it in items if it["category"] == cat), key=lambda it: sha256_text(it["question_id"]))
        out.extend(pool[:k])
    return out


def locomo_pilot(items: list[dict]) -> list[dict]:
    """Pilot (2026-09-19 amendment 9): 20 category-3 questions, excluded from every analysis."""
    pool = sorted((it for it in items if it["category"] == 3), key=lambda it: sha256_text(it["question_id"]))
    return pool[:20]
