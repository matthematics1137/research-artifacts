#!/usr/bin/env python3
"""Paper 6: the frozen, seeded generator of the 200 primary long-context items
(2026-09-19 amendment §6). Built once, before any measurement.

  engines/exl3-env/bin/python testsuite/paper6/make_longctx.py             # build -> work/, manifest
  engines/exl3-env/bin/python testsuite/paper6/make_longctx.py --selftest  # determinism + item checks

Four RULER-style families x 50 items:
  single      one needle, one key                  -> its value
  multikey    4 needles with 4 keys, 1 key asked   -> that key's value (the other 3 are distractors)
  multivalue  4 needles with the same key          -> all 4 values
  vartrack    a chain of 5 assignments             -> all 5 variable names

Haystack: LongMemEval-S sessions (MIT; SHA-256 verified), rendered as plain chat text
("user: ...\\n\\nassistant: ..."), sessions in a seeded order, cut to 16,384 Qwen3.8 tokens
(+-1%, needles included). Needles go at sentence ends (turn ends included) at depths spread
evenly through the haystack. All randomness is SHA-256 counter mode keyed by the seed and
the item id, so builds are byte-identical across Python versions and hash seeds. No
timestamps are written, so the manifest is a pure function of the inputs.

CPU only: the `tokenizers` library on the Qwen3.8 tokenizer.json. Choices the amendment
does not fix are listed in HARNESS_NOTES.md.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6common as P  # noqa: E402

sys.path.insert(0, P.PAPER5)
import p5common as C5  # noqa: E402  (LME_S path and load_json_verified only)

SEED = 20260919
TARGET_TOKENS = 16384
TOLERANCE = 0.01
TOL_LO, TOL_HI = int(TARGET_TOKENS * (1 - TOLERANCE)) + 1, int(TARGET_TOKENS * (1 + TOLERANCE))  # 16221..16547
FILL_SLACK = 48          # filler is accepted when it ends <= 48 tokens short of its budget
N_PER_FAMILY = 50
FAMILIES = ("single", "multikey", "multivalue", "vartrack")
N_NEEDLES = {"single": 1, "multikey": 4, "multivalue": 4, "vartrack": 5}

# ---------------------------------------------------------------- frozen text
# One instruction for every family; only the question differs.
INSTRUCTION = ("Below is a long text made of chat conversations. A few special facts are hidden in it. "
               "Read the text, then answer the question that follows it. Reply with the answer only.")
QUESTION = {
    "single": "What is the special magic number for {key} mentioned in the provided text?",
    "multikey": "What is the special magic number for {key} mentioned in the provided text?",
    "multivalue": "What are all the special magic numbers for {key} mentioned in the provided text?",
    "vartrack": ("Find all variables that are assigned the value {value}, directly or through a chain "
                 "of other variables, in the provided text."),
}
NEEDLE_NIAH = "One of the special magic numbers for {key} is: {value}."      # RULER wording
NEEDLE_VT_FIRST = "VAR {name} = {value}."                                     # RULER variable tracking
NEEDLE_VT_LINK = "VAR {name} = VAR {prev}."

# Wrapper of one user message with thinking off (the Qwen3.8 template; same strings as
# testsuite/paper5/p5common.WRAP_NOTHINK). Used only to record an expected prompt count.
WRAP_NOTHINK = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

ADJECTIVES = (
    "amber", "brisk", "calm", "dusky", "eager", "faded", "gentle", "hollow", "icy", "jolly", "keen",
    "lively", "mellow", "nimble", "olive", "polished", "quiet", "rustic", "silver", "tidy", "upbeat",
    "vivid", "woolen", "young", "zesty", "bold", "crisp", "dapper", "early", "fuzzy", "glossy", "hasty",
    "ivory", "jagged", "knotty", "lunar", "misty", "narrow", "orange", "plucky", "quaint", "rapid",
    "sandy", "tawny", "unruly", "velvet", "wintry", "yellow", "breezy", "cobalt", "dainty", "elfin",
    "frosty", "golden", "hazel", "inky", "jaunty", "leafy", "mossy", "nutty", "oaken", "pearly", "rosy",
    "smoky")
NOUNS = (
    "lantern", "harbor", "violin", "meadow", "compass", "teapot", "glacier", "pebble", "orchard",
    "saddle", "falcon", "anchor", "blanket", "canyon", "dolphin", "easel", "feather", "gazebo", "hammock",
    "igloo", "jigsaw", "kettle", "lagoon", "mitten", "nectar", "otter", "parsnip", "quarry", "ribbon",
    "sparrow", "thimble", "umbrella", "valley", "walrus", "yarn", "zeppelin", "acorn", "bramble",
    "chimney", "dune", "ember", "fjord", "garnet", "heron", "inkwell", "juniper", "kayak", "locket",
    "marble", "nutmeg", "oyster", "pylon", "quill", "rafter", "spindle", "trellis", "urchin", "vessel",
    "willow", "badger", "cobble", "dahlia", "ferry", "gondola")
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Sessions containing any of these are left out of the pool: chat special tokens would be
# parsed as control tokens, think tags could confuse the reasoning parser, and the needle
# phrasings must not occur naturally.
FORBIDDEN = (("special_token", re.compile(r"<\||\|>")), ("think_tag", re.compile(r"</?think>")),
             ("needle_phrase", re.compile(r"special magic", re.I)), ("var_statement", re.compile(r"\bVAR\b")))

SENT_END = re.compile(r"[.!?][\"')\]]?(?=\s|$)")
PREV_WORD = re.compile(r"(?<![A-Za-z.])([A-Za-z][A-Za-z.]*)$")
NEXT_CHAR = re.compile(r"\s*(\S)")
LIST_MARKER = re.compile(r"(?:^|[\s(\[])\d{1,3}$")   # "3." of a numbered list, not a sentence end
INITIALS = re.compile(r"(?:[a-z]\.)+[a-z]")            # "A.J." / "e.g." style initials
ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "jr", "sr", "no", "fig", "approx",
          "inc", "ltd", "co", "u.s", "a.m", "p.m", "vol", "ch", "sec", "min", "max", "dept", "est"}


# ---------------------------------------------------------------- deterministic randomness
class HashStream:
    """Counter-mode SHA-256 integers: independent of Python's `random` and PYTHONHASHSEED."""

    def __init__(self, *parts):
        self.prefix = "|".join(str(p) for p in parts)
        self.n = 0

    def u64(self) -> int:
        d = hashlib.sha256(f"{self.prefix}|{self.n}".encode()).digest()
        self.n += 1
        return int.from_bytes(d[:8], "big")

    def below(self, n: int) -> int:
        lim = (1 << 64) - ((1 << 64) % n)          # rejection sampling: no modulo bias
        while True:
            x = self.u64()
            if x < lim:
                return x % n


def session_order(item_id: str, sids: list[str]) -> list[int]:
    """A seeded permutation of the pool: ascending SHA-256(seed | item | session id)."""
    return sorted(range(len(sids)),
                  key=lambda j: hashlib.sha256(f"{SEED}|{item_id}|order|{sids[j]}".encode()).digest())


# ---------------------------------------------------------------- tokenizer
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        got = P.sha256_file(P.TOKENIZER_QWEN38)
        if got != P.TOKENIZER_QWEN38_SHA256:
            raise SystemExit(f"Qwen3.8 tokenizer hash mismatch: {got}")
        _TOK = Tokenizer.from_file(P.TOKENIZER_QWEN38)
    return _TOK


def encode(text: str):
    return tokenizer().encode(text, add_special_tokens=False)


def ntok(text: str) -> int:
    return len(encode(text).ids)


# ---------------------------------------------------------------- boundaries
def fence_spans(content: str) -> list[tuple[int, int]]:
    """Spans inside ``` code fences (an unclosed fence runs to the end of the turn)."""
    marks = [m.start() for m in re.finditer(r"```", content)]
    spans = []
    for a in range(0, len(marks), 2):
        spans.append((marks[a], marks[a + 1] + 3 if a + 1 < len(marks) else len(content)))
    return spans


def _inside(pos: int, spans) -> bool:
    return any(a <= pos < b for a, b in spans)


def sentence_ends(content: str) -> list[int]:
    """Positions just after a sentence terminator: followed by whitespace or the end of the
    turn, not inside a code fence, next word not lower-case, previous word not an abbreviation
    or an initial."""
    spans = fence_spans(content)
    out = []
    for m in SENT_END.finditer(content):
        b = m.end()
        if _inside(m.start(), spans):
            continue
        nxt = NEXT_CHAR.match(content, b)
        if nxt and nxt.group(1).islower():
            continue
        if LIST_MARKER.search(content, max(0, m.start() - 6), m.start()):
            continue
        pw = PREV_WORD.search(content, max(0, m.start() - 40), m.start())
        if pw:
            w = pw.group(1).lower()
            if w in ABBREV or len(w) == 1 or INITIALS.fullmatch(w):
                continue
        out.append(b)
    return out


def line_ends(content: str) -> list[int]:
    spans = fence_spans(content)
    return [m.start() for m in re.finditer(r"(?<=\S)\n", content) if not _inside(m.start(), spans)]


# ---------------------------------------------------------------- session pool
class Pool:
    """Unique LongMemEval-S sessions (by session id, sorted), rendered as plain chat text.
    Token counts and boundaries are computed lazily, for the sessions an item actually uses."""

    def __init__(self, data: list[dict]):
        raw = {}
        for it in data:
            for sid, s in zip(it["haystack_session_ids"], it["haystack_sessions"]):
                raw.setdefault(sid, s)
        self.unique = len(raw)
        self.excluded = {name: 0 for name, _ in FORBIDDEN}
        self.excluded["empty"] = 0
        self.sids, self.texts, self.spans = [], [], []
        for sid in sorted(raw):
            turns = [(t["role"], t["content"].strip()) for t in raw[sid] if t["content"].strip()]
            if not turns:
                self.excluded["empty"] += 1
                continue
            text, spans = "", []
            for role, content in turns:
                if text:
                    text += "\n\n"
                start = len(text) + len(role) + 2
                text += f"{role}: {content}"
                spans.append((start, len(text)))
            bad = next((name for name, rx in FORBIDDEN if rx.search(text)), None)
            if bad:
                self.excluded[bad] += 1
                continue
            self.sids.append(sid)
            self.texts.append(text)
            self.spans.append(spans)
        self._enc, self._bounds = {}, {}

    def bounds(self, j: int) -> tuple[list[int], list[int]]:
        """(needle insertion points, truncation cuts) of session j. Insertion points are
        sentence ends; cuts add line ends and turn ends."""
        if j not in self._bounds:
            ins, cuts = [], []
            for a, b in self.spans[j]:
                content = self.texts[j][a:b]
                se = [a + x for x in sentence_ends(content)]
                ins += se
                cuts += se + [a + x for x in line_ends(content)] + [b]
            self._bounds[j] = (ins, sorted(set(cuts)))
        return self._bounds[j]

    def tokens(self, j: int) -> tuple[int, list[int]]:
        """(token count, token end offsets) of session j, cached."""
        if j not in self._enc:
            e = encode(self.texts[j])
            self._enc[j] = (len(e.ids), [b for _, b in e.offsets])
        return self._enc[j]

    def best_cut(self, j: int, budget: int):
        """Longest prefix of session j ending at a cut that holds <= budget tokens."""
        n, ends = self.tokens(j)
        best = None
        for c in self.bounds(j)[1]:
            k = bisect.bisect_right(ends, c)       # tokens wholly inside text[:c]
            if k <= budget:
                best = (c, k)
            else:
                break
        return best


def build_filler(pool: Pool, item_id: str, budget: int):
    """Whole sessions in seeded order while they fit; the first that does not fit is cut at
    a turn/sentence/line end if that lands within FILL_SLACK of the budget, else skipped.
    Returns (filler text, needle insertion points, [(session id, cut or None)], exact tokens)."""
    order = session_order(item_id, pool.sids)
    parts, used, est = [], [], 0
    for j in order:
        n, _ = pool.tokens(j)
        join = 1 if parts else 0                    # "\n\n" between sessions is one token
        if est + join + n <= budget:
            parts.append((j, None))
            est += join + n
        else:
            cut = pool.best_cut(j, budget - est - join)
            if cut is None or budget - (est + join + cut[1]) > FILL_SLACK:
                continue
            parts.append((j, cut[0]))
            est += join + cut[1]
        if budget - est <= FILL_SLACK:
            break
    # exact recount; trim the last part at earlier cuts while over budget
    while True:
        text, inserts = assemble(pool, parts)
        n = ntok(text)
        if n <= budget:
            break
        j, cut = parts[-1]
        prev = [c for c in pool.bounds(j)[1] if c < (cut if cut is not None else len(pool.texts[j]))]
        if not prev:
            parts.pop()
        else:
            parts[-1] = (j, prev[-1])
    return text, inserts, [(pool.sids[j], cut) for j, cut in parts], n


def assemble(pool: Pool, parts):
    text, inserts = "", []
    for j, cut in parts:
        if text:
            text += "\n\n"
        off = len(text)
        body = pool.texts[j] if cut is None else pool.texts[j][:cut]
        inserts += [off + b for b in pool.bounds(j)[0] if b <= len(body)]
        text += body
    return text, inserts


# ---------------------------------------------------------------- needles
def draw_value(stream: HashStream, digits: int, bad) -> str:
    while True:
        v = str(10 ** (digits - 1) + stream.below(9 * 10 ** (digits - 1)))
        if not bad(v):
            return v


def draw_key(stream: HashStream, bad) -> str:
    while True:
        k = f"{ADJECTIVES[stream.below(len(ADJECTIVES))]}-{NOUNS[stream.below(len(NOUNS))]}"
        if not bad(k):
            return k


def draw_name(stream: HashStream, bad) -> str:
    while True:
        k = "".join(UPPER[stream.below(26)] for _ in range(5))
        if not bad(k):
            return k


def draw_slots(family: str, i: int, stream: HashStream, filler_low: str | None) -> dict:
    """Keys, values and names of one item. `filler_low` (lower-cased filler) is None on the
    provisional draw that sizes the filler; the final draw also rejects anything the filler
    contains. Both draws read the same stream from its start, so they agree unless a drawn
    string occurs in the filler."""
    used = set()

    def bad(s):
        s = s.lower()
        clash = s in used or (filler_low is not None and s in filler_low) \
            or s in INSTRUCTION.lower() or any(s in q.lower() for q in QUESTION.values())
        if not clash:
            used.add(s)
        return clash

    k = N_NEEDLES[family]
    if family == "single":
        return {"keys": [draw_key(stream, bad)], "values": [draw_value(stream, 7, bad)], "asked": 0}
    if family == "multikey":
        keys = [draw_key(stream, bad) for _ in range(k)]
        return {"keys": keys, "values": [draw_value(stream, 7, bad) for _ in range(k)], "asked": i % k}
    if family == "multivalue":
        return {"keys": [draw_key(stream, bad)], "values": [draw_value(stream, 7, bad) for _ in range(k)]}
    if family == "vartrack":
        return {"names": [draw_name(stream, bad) for _ in range(k)], "vt_value": draw_value(stream, 5, bad)}
    raise ValueError(family)


def needle_texts(family: str, s: dict) -> list[str]:
    """Needles in haystack order (depth ascending)."""
    if family in ("single", "multikey"):
        return [NEEDLE_NIAH.format(key=k, value=v) for k, v in zip(s["keys"], s["values"])]
    if family == "multivalue":
        return [NEEDLE_NIAH.format(key=s["keys"][0], value=v) for v in s["values"]]
    names = s["names"]
    return [NEEDLE_VT_FIRST.format(name=names[0], value=s["vt_value"])] + \
           [NEEDLE_VT_LINK.format(name=names[j], prev=names[j - 1]) for j in range(1, len(names))]


def question_answers(family: str, s: dict):
    if family in ("single", "multikey"):
        a = s["asked"]
        return (QUESTION[family].format(key=s["keys"][a]), [s["values"][a]],
                [v for j, v in enumerate(s["values"]) if j != a])
    if family == "multivalue":
        return QUESTION[family].format(key=s["keys"][0]), list(s["values"]), []
    return QUESTION[family].format(value=s["vt_value"]), list(s["names"]), []


def target_depths(family: str, i: int) -> list[float]:
    """Rotated even grid: the k needles of an item sit 1/k apart, and across the 50 items the
    grid shifts by 1/(50k), so every family covers (0, 1) evenly."""
    k = N_NEEDLES[family]
    u = (i + 0.5) / N_PER_FAMILY
    return [(u + j) / k for j in range(k)]


def place(filler_starts: list[int], n_filler: int, inserts: list[int], depths: list[float]) -> list[int]:
    """Nearest sentence end (by token position) to each target depth, strictly increasing."""
    tok_at = [bisect.bisect_left(filler_starts, b) for b in inserts]
    chosen, lo = [], 0
    for d in depths:
        p = d * n_filler
        k = bisect.bisect_left(tok_at, p, lo)
        cands = [c for c in (k - 1, k) if lo <= c < len(inserts)]
        if not cands:
            raise RuntimeError("no insertion point left")
        best = min(cands, key=lambda c: (abs(tok_at[c] - p), c))
        chosen.append(inserts[best])
        lo = best + 1
    return chosen


# ---------------------------------------------------------------- items
def build_item(pool: Pool, family: str, i: int) -> dict:
    item_id = f"{family}-{i:02d}"
    stream_seed = (SEED, item_id, "needles")
    prov = draw_slots(family, i, HashStream(*stream_seed), None)
    needle_budget = sum(ntok(" " + t) for t in needle_texts(family, prov))
    filler, inserts, sessions, n_filler = build_filler(pool, item_id, TARGET_TOKENS - needle_budget)
    slots = draw_slots(family, i, HashStream(*stream_seed), filler.lower())
    needles = needle_texts(family, slots)
    question, answers, distractors = question_answers(family, slots)
    depths_t = target_depths(family, i)

    fenc = encode(filler)
    positions = place([a for a, _ in fenc.offsets], n_filler, inserts, depths_t)
    hay, spans, prev = "", [], 0
    for pos, t in zip(positions, needles):
        hay += filler[prev:pos]
        spans.append(len(hay))                                   # the inserted " " + needle
        hay += " " + t
        prev = pos
    hay += filler[prev:]

    henc = encode(hay)
    n_hay = len(henc.ids)
    hstarts = [a for a, _ in henc.offsets]
    needle_rows = []
    for j, (t, sp) in enumerate(zip(needles, spans)):
        tk = bisect.bisect_left(hstarts, sp)
        role = ("asked" if j == slots.get("asked") else "distractor") if family == "multikey" else \
            ("chain" if family == "vartrack" else "value")
        needle_rows.append({"text": t, "role": role, "target_depth": round(depths_t[j], 6),
                            "depth": round(tk / n_hay, 6), "char_offset": sp + 1, "token_offset": tk})
    if family == "vartrack":
        answer_depths = {nm: needle_rows[j]["depth"] for j, nm in enumerate(slots["names"])}
    elif family == "multivalue":
        answer_depths = {v: needle_rows[j]["depth"] for j, v in enumerate(slots["values"])}
    else:
        a = slots["asked"]
        answer_depths = {slots["values"][a]: needle_rows[a]["depth"]}

    content = INSTRUCTION + "\n\n" + hay + "\n\nQuestion: " + question
    start = len(INSTRUCTION) + 2
    rec = {
        "id": item_id, "family": family, "index": i, "question": question,
        "answers": answers, "distractors": distractors,
        "keys": slots.get("keys", []), "vt_value": slots.get("vt_value"),
        "needles": needle_rows, "depths": [r["depth"] for r in needle_rows],
        "asked_depth": needle_rows[slots["asked"]]["depth"] if "asked" in slots else None,
        "answer_depths": answer_depths,
        "haystack_tokens": n_hay, "filler_tokens": n_filler,
        "content_tokens": ntok(content),
        "prompt_tokens_qwen38_nothink": ntok(WRAP_NOTHINK[0] + content + WRAP_NOTHINK[1]),
        "haystack_span": [start, start + len(hay)],
        "sessions": [{"id": sid, "cut_chars": cut} for sid, cut in sessions],
        "content": content, "sha256": P.sha256_text(content),
    }
    problems = validate_item(rec, recount=False)
    if problems:
        raise RuntimeError(f"{item_id}: {problems}")
    return rec


def validate_item(rec: dict, recount: bool = True) -> list[str]:
    """Every property the harness relies on; returns a list of problems (empty = ok)."""
    p = []
    c = rec["content"]
    a, b = rec["haystack_span"]
    hay = c[a:b]
    if P.sha256_text(c) != rec["sha256"]:
        p.append("sha256")
    if c != INSTRUCTION + "\n\n" + hay + "\n\nQuestion: " + rec["question"]:
        p.append("layout")
    n = ntok(hay) if recount else rec["haystack_tokens"]
    if n != rec["haystack_tokens"] or not TOL_LO <= n <= TOL_HI:
        p.append(f"length {n}")
    stripped = hay
    for nd in rec["needles"]:
        if c.count(nd["text"]) != 1:
            p.append(f"needle count {nd['text']!r}")
        stripped = stripped.replace(" " + nd["text"], "", 1)
    low = stripped.lower()
    probe = list(rec["answers"]) + list(rec["distractors"]) + list(rec["keys"]) + \
        ([rec["vt_value"]] if rec["vt_value"] else [])
    for s in probe:
        if s.lower() in low:
            p.append(f"{s!r} occurs outside its needles")
    for s in rec["answers"]:
        if s.lower() in INSTRUCTION.lower() or s.lower() in rec["question"].lower():
            p.append(f"answer {s!r} in instruction/question")
        if not any(s in nd["text"] for nd in rec["needles"]):
            p.append(f"answer {s!r} in no needle")
    ds = rec["depths"]
    if any(not 0.0 <= d <= 1.0 for d in ds) or ds != sorted(ds):
        p.append("depths")
    if len(rec["needles"]) != N_NEEDLES[rec["family"]]:
        p.append("needle number")
    return p


def build_all(per_family: int = N_PER_FAMILY, log=print) -> tuple[bytes, dict]:
    t0 = time.time()
    data = C5.load_json_verified(C5.LME_S)
    pool = Pool(data)
    del data
    log(f"pool: {len(pool.sids)} sessions ({pool.unique} unique; excluded {pool.excluded}) "
        f"in {time.time() - t0:.1f}s")
    items = []
    for i in range(per_family):                  # interleaved families: a fixed, mixed run order
        for fam in FAMILIES:
            items.append(build_item(pool, fam, i))
        if (i + 1) % 10 == 0:
            log(f"  {len(items)} items, {time.time() - t0:.0f}s")
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items).encode("utf-8")
    info = {"pool_sessions": len(pool.sids), "unique_sessions": pool.unique, "excluded": pool.excluded}
    return blob, {"items": items, "pool": info}


def manifest(blob: bytes, built: dict) -> dict:
    items = built["items"]
    toks = [r["haystack_tokens"] for r in items]
    fam_counts = {f: sum(r["family"] == f for r in items) for f in FAMILIES}
    return {
        "what": "Paper 6 primary long-context items (2026-09-19 amendment §6); frozen before measurement",
        "seed": SEED, "target_haystack_tokens": TARGET_TOKENS, "tolerance": TOLERANCE,
        "accepted_range": [TOL_LO, TOL_HI], "families": list(FAMILIES), "counts": fam_counts,
        "n_items": len(items),
        "generator": {"file": "testsuite/paper6/make_longctx.py", "sha256": P.sha256_file(os.path.abspath(__file__)),
                      "p6common_sha256": P.sha256_file(os.path.join(HERE, "p6common.py"))},
        "inputs": {"longmemeval_s": {"path": os.path.relpath(C5.LME_S, P.REPO), "sha256": C5.DATA_SHA256[C5.LME_S]},
                   "tokenizer_qwen38": {"path": P.TOKENIZER_QWEN38, "sha256": P.TOKENIZER_QWEN38_SHA256}},
        "templates": {"instruction": INSTRUCTION, "questions": QUESTION, "needle_niah": NEEDLE_NIAH,
                      "needle_vt_first": NEEDLE_VT_FIRST, "needle_vt_link": NEEDLE_VT_LINK,
                      "content_layout": "INSTRUCTION + '\\n\\n' + haystack + '\\n\\nQuestion: ' + question"},
        "pool": built["pool"],
        "output": {"path": os.path.relpath(P.LONGCTX_ITEMS, P.REPO), "sha256": hashlib.sha256(blob).hexdigest(),
                   "bytes": len(blob)},
        "haystack_tokens": {"min": min(toks), "max": max(toks), "mean": round(sum(toks) / len(toks), 1)},
        "items": [{"id": r["id"], "family": r["family"], "sha256": r["sha256"],
                   "haystack_tokens": r["haystack_tokens"]} for r in items],
    }


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{(': ' + str(detail)) if detail else ''}", flush=True)
        if not ok:
            fails.append(name)

    print("selftest: unit checks")
    s1, s2 = HashStream(1, "x"), HashStream(1, "x")
    check("hash stream deterministic", [s1.below(1000) for _ in range(5)] == [s2.below(1000) for _ in range(5)])
    check("sentence ends skip abbreviations/initials/lower-case continuations",
          sentence_ends("Hi Dr. Smith. A.J. Doe wrote it. e.g. this is fine. End") == [13, 51],
          sentence_ends("Hi Dr. Smith. A.J. Doe wrote it. e.g. this is fine. End"))
    check("no boundary inside a code fence",
          sentence_ends("Run it. ```x = 1. Y = 2.``` Done.") == [7, 33], sentence_ends("Run it. ```x = 1. Y = 2.``` Done."))
    check("no boundary after a numbered-list marker",
          sentence_ends("Steps below.\n1. Chop the onion. Then wait.") == [12, 31, 42],
          sentence_ends("Steps below.\n1. Chop the onion. Then wait."))
    check("target depths rotate evenly",
          [round(d, 6) for d in target_depths("multikey", 0)] == [0.0025, 0.2525, 0.5025, 0.7525])

    print("selftest: build A (in process)")
    t0 = time.time()
    blob_a, built = build_all(log=lambda m: print("   ", m, flush=True))
    print(f"    built in {time.time() - t0:.0f}s, {len(blob_a)} bytes, sha256 {hashlib.sha256(blob_a).hexdigest()}")

    print("selftest: build B (subprocess, different PYTHONHASHSEED)")
    with tempfile.TemporaryDirectory(dir=P.WORK) as td:
        out_b = os.path.join(td, "b.jsonl")
        env = dict(os.environ, PYTHONHASHSEED="12345")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--out", out_b, "--no-manifest", "--quiet"],
                       check=True, env=env)
        with open(out_b, "rb") as f:
            blob_b = f.read()
    check("two builds byte-identical", blob_a == blob_b,
          f"{hashlib.sha256(blob_a).hexdigest()[:16]} vs {hashlib.sha256(blob_b).hexdigest()[:16]}")

    print("selftest: item checks (full recount with the Qwen3.8 tokenizer)")
    items = built["items"]
    check("200 items, 50 per family", len(items) == 200 and
          all(sum(r["family"] == f for r in items) == 50 for f in FAMILIES))
    check("ids unique", len({r["id"] for r in items}) == len(items))
    probs = {r["id"]: validate_item(r, recount=True) for r in items}
    bad = {k: v for k, v in probs.items() if v}
    check("lengths within 16,384 +-1%, needles present exactly once, answers absent elsewhere, layout",
          not bad, bad if bad else "")
    toks = [r["haystack_tokens"] for r in items]
    check("haystack token range", TOL_LO <= min(toks) and max(toks) <= TOL_HI, f"{min(toks)}..{max(toks)}")
    inst = {r["content"][:len(INSTRUCTION)] for r in items}
    check("instruction identical across items", inst == {INSTRUCTION})
    for fam in FAMILIES:
        ds = sorted(d for r in items if r["family"] == fam for d in r["depths"])
        k = N_NEEDLES[fam] * N_PER_FAMILY
        # evenly spread: every fifth of (0,1) holds about a fifth of the needles
        bins = [sum(lo <= d < lo + 0.2 for d in ds) for lo in (0, .2, .4, .6, .8)]
        check(f"{fam}: depths spread evenly", all(abs(b - k / 5) <= max(3, 0.1 * k) for b in bins), bins)
    asked = sorted(r["asked_depth"] for r in items if r["family"] == "multikey")
    check("multikey: asked depths cover all quarters",
          all(any(q <= d < q + 0.25 for d in asked) for q in (0, .25, .5, .75)))
    mv = [r for r in items if r["family"] == "multivalue"]
    check("multivalue: 4 distinct required values", all(len(set(r["answers"])) == 4 for r in mv))
    vt = [r for r in items if r["family"] == "vartrack"]
    check("vartrack: 5 distinct names, chain in text order",
          all(len(set(r["answers"])) == 5 and
              all(r["content"].index(r["needles"][j]["text"]) < r["content"].index(r["needles"][j + 1]["text"])
                  for j in range(4)) for r in vt))
    if os.path.exists(P.LONGCTX_ITEMS):
        with open(P.LONGCTX_ITEMS, "rb") as f:
            check("existing work/longctx_200.jsonl equals this build", f.read() == blob_a)
    if os.path.exists(P.LONGCTX_MANIFEST):
        with open(P.LONGCTX_MANIFEST) as f:
            m = json.load(f)
        check("manifest output sha256 matches this build",
              m["output"]["sha256"] == hashlib.sha256(blob_a).hexdigest())
        check("manifest generator sha256 matches this file",
              m["generator"]["sha256"] == P.sha256_file(os.path.abspath(__file__)))
    if fails:
        print(f"\nSELFTEST FAILED: {fails}")
        return 1
    print("\nSELFTEST PASSED (all checks)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=P.LONGCTX_ITEMS)
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite a different existing item file")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    log = (lambda m: None) if a.quiet else (lambda m: print(m, flush=True))
    blob, built = build_all(log=log)
    if os.path.exists(a.out):
        with open(a.out, "rb") as f:
            old = f.read()
        if old != blob and not a.force:
            raise SystemExit(f"{a.out} exists and differs from this build (frozen items); use --force deliberately")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(blob)
    log(f"wrote {a.out}: {len(built['items'])} items, {len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()}")
    if not a.no_manifest:
        with open(P.LONGCTX_MANIFEST, "w") as f:
            json.dump(manifest(blob, built), f, indent=1, ensure_ascii=False)
            f.write("\n")
        log(f"wrote {P.LONGCTX_MANIFEST}")


if __name__ == "__main__":
    main()
