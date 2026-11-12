"""Paper 6 shared definitions: paths, the 24-cell grid, model artifacts, scoring, and
journal reading.

Design: research/design-kv-allocation.md (frozen 2026-09-01; the 2026-09-19
narrowed-question amendment is binding). Standard library only at import time;
`run_eval` (the Paper 1 GSM8K scorer) is imported lazily and never modified.
Every choice the amendment does not fix is listed in HARNESS_NOTES.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.join(HERE, "work")                       # bulk outputs (not for git)
JOURNALS = os.path.join(WORK, "journals")
LOGS = os.path.join(WORK, "logs")
PAUSE_FILE = os.path.join(HERE, "PAUSE")                # touch to pause (server is stopped)
PAPER5 = os.path.join(REPO, "testsuite", "paper5")
EVALS = os.path.join(REPO, "testsuite", "evals")

LONGCTX_ITEMS = os.path.join(WORK, "longctx_200.jsonl")
LONGCTX_MANIFEST = os.path.join(HERE, "longctx_manifest.json")
GSM8K_ITEMS = os.path.join(EVALS, "datasets", "gsm8k_100.jsonl")
GSM8K_SHA256 = "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37"
TOKENIZER_QWEN38 = os.path.expanduser("~/models/qwen38-exl3-2.0/tokenizer.json")
TOKENIZER_QWEN38_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"

# ---------------------------------------------------------------- engine (amendment §5)
ENGINE_DIR = os.path.join(REPO, "engines", "llama.cpp-p6fa")
SERVER_EXE = os.path.join(ENGINE_DIR, "build", "bin", "llama-server")
ENGINE_COMMIT = "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3"   # pinned llama.cpp commit
PORT = 18130
N_CTX = 20480            # 16,384 haystack + instruction/question/template + answer; GSM8K 8,192 cap fits
PAPER5_PORT = 18125      # tabby-p5: Paper 6 never runs beside Paper 5's GPU work

# ---------------------------------------------------------------- grid (amendment §2-4)
MODELS = {
    "qwen38-27b": {"arch": "hybrid", "dir": "~/models/qwen38", "file": "Qwen3.8-27B-UD-{tier}.gguf",
                   "repo": "unsloth/Qwen3.8-27B-GGUF", "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502"},
    "qwen3-8b": {"arch": "gqa", "dir": "~/models/qwen3-8b", "file": "Qwen3-8B-UD-{tier}.gguf",
                 "repo": "unsloth/Qwen3-8B-GGUF", "revision": "a6adef130ffb23ddaf1a62fec9dced968c9bc482"},
}
HYBRID, GQA = "qwen38-27b", "qwen3-8b"
TIERS = ("Q2_K_XL", "IQ3_XXS", "Q4_K_XL")               # low -> high weight precision
KV_TYPES = {"f16": ("f16", "f16"), "q8_0": ("q8_0", "q8_0"), "q4_0": ("q4_0", "q4_0"),
            "k8v4": ("q8_0", "q4_0")}                   # name -> (-ctk, -ctv)
KV_ORDER = ("f16", "q8_0", "q4_0", "k8v4")

# The on-disk 27B UD-Q4_K_XL predates downloads.jsonl. Its SHA-256 is named in the
# amendment (3f227079...), matches testsuite/results/efficiency/hash_audit_2026-09-06.json,
# and the HF cache metadata beside it records revision 4ca72078...
PINNED_SHA256 = {
    "Qwen3.8-27B-UD-Q4_K_XL.gguf": ("3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e",
                                    "amendment §3; hash_audit_2026-09-06.json; HF cache metadata"),
}
DOWNLOADS_JSONL = os.path.join(REPO, "testsuite", "artifacts", "downloads.jsonl")

# ---------------------------------------------------------------- decoding (amendment §6)
LONGCTX_MAX_TOKENS = 128
GSM8K_MAX_TOKENS = 8192


def all_cells() -> list[tuple[str, str, str]]:
    """Canonical order: model, then tier, then KV type (same GGUF stays hot in page cache)."""
    return [(m, t, k) for m in MODELS for t in TIERS for k in KV_ORDER]


def cell_name(model: str, tier: str, kv: str) -> str:
    return f"{model}__{tier}__{kv}"


def journal_path(model: str, tier: str, kv: str, journals: str = JOURNALS) -> str:
    return os.path.join(journals, cell_name(model, tier, kv) + ".jsonl")


def gguf_path(model: str, tier: str) -> str:
    m = MODELS[model]
    return os.path.join(os.path.expanduser(m["dir"]), m["file"].format(tier=tier))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def receipt_for(path: str, downloads: str = DOWNLOADS_JSONL) -> dict | None:
    """Pinned identity of a GGUF: the fetcher's receipt (downloads.jsonl) when present,
    else the amendment-named hash for the pre-existing 27B Q4 file, else None."""
    base = os.path.basename(path)
    if os.path.exists(downloads):
        found = None
        with open(downloads) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if os.path.expanduser(r.get("dest", "")) == path or os.path.basename(r.get("path", "")) == base:
                    found = r                               # the last receipt for the file wins
        if found:
            return {"sha256": found["sha256"], "repo": found.get("repo"), "revision": found.get("revision"),
                    "bytes": found.get("bytes"), "source": "downloads.jsonl"}
    if base in PINNED_SHA256:
        sha, src = PINNED_SHA256[base]
        return {"sha256": sha, "repo": "unsloth/Qwen3.8-27B-GGUF",
                "revision": MODELS[HYBRID]["revision"], "bytes": None, "source": src}
    return None


# ---------------------------------------------------------------- items
def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_longctx(path: str = LONGCTX_ITEMS) -> list[dict]:
    return load_jsonl(path)


def load_gsm8k(path: str = GSM8K_ITEMS, verify: bool = True) -> list[dict]:
    if verify and sha256_file(path) != GSM8K_SHA256:
        raise SystemExit(f"GSM8K-100 hash mismatch: {path}")
    return load_jsonl(path)


@lru_cache(maxsize=1)
def run_eval():
    """The Paper 1 harness module (testsuite/evals/run_eval.py), imported unmodified."""
    if EVALS not in sys.path:
        sys.path.insert(0, EVALS)
    import run_eval as RE  # noqa: E402
    return RE


def gsm8k_prompt(item: dict) -> str:
    """Exactly the prompt run_eval.py sends for its gsm8k suite."""
    return run_eval().build_prompt("gsm8k", item)


# ---------------------------------------------------------------- scoring
def visible_answer(message: dict) -> str:
    """Long-context answer text with thinking removed. Thinking is off for these
    requests; if a think block still shows up inline, text before the first </think>
    is dropped, and an unterminated <think> leaves no visible answer."""
    content = (message or {}).get("content") or ""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    if "<think>" in content:
        return ""
    return content.strip()


def score_longctx(answer: str, item: dict) -> bool:
    """Deterministic containment: every required string must appear (case-insensitive)."""
    a = (answer or "").lower()
    return all(r.lower() in a for r in item["answers"])


def score_gsm8k(message: dict, item: dict) -> bool:
    """The Paper 1 path: run_eval.visible_content, then run_eval.score_gsm8k."""
    RE = run_eval()
    return bool(RE.score_gsm8k(RE.visible_content(message or {}), item))


def score_row(task: str, row: dict, item: dict) -> dict:
    msg = {"content": row.get("content"), "reasoning_content": row.get("reasoning_content")}
    if task == "longctx":
        ans = visible_answer(msg)
        hits = [r for r in item["answers"] if r.lower() in ans.lower()]
        distract = [d for d in item.get("distractors", []) if d.lower() in ans.lower()]
        return {"correct": score_longctx(ans, item), "visible_answer": ans[:2000],
                "answers_found": hits, "distractors_found": distract}
    if task == "gsm8k":
        return {"correct": score_gsm8k(msg, item),
                "visible_answer": run_eval().visible_content(msg)[-400:]}
    raise ValueError(task)


# ---------------------------------------------------------------- journals
def read_journal(path: str) -> dict:
    """{'outcomes': {job_id: terminal row}, 'identity': [server-start rows], 'events': [...]}.
    Outcome rule: the first terminal ok row for a job wins; with none, the last terminal row
    (a failure) is kept so the item counts as missing, never as wrong."""
    out, ident, events = {}, [], []
    if not os.path.exists(path):
        return {"outcomes": out, "identity": ident, "events": events}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            kind = r.get("kind")
            if kind == "server_start":
                ident.append(r)
            elif kind:
                events.append(r)
            elif r.get("terminal"):
                prev = out.get(r["job_id"])
                if prev is None or not prev.get("ok"):
                    out[r["job_id"]] = r
    return {"outcomes": out, "identity": ident, "events": events}


def job_id(task: str, item_id: str) -> str:
    return f"{task}|{item_id}"
