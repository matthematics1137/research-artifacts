"""P8 long context, stages 2-3: items from P6's frozen generator, with only the registered module constants overridden.

Registration: research/prereg/2026-11-01-p8-longctx-stages2-3.md ("Items"; Amendments 1-3 for the choices it leaves
open). testsuite/paper6/make_longctx.py is imported unchanged. While one item is built, this module sets only:
  - SEED (the SHA-256 hex digest of the set's seed text);
  - TARGET_TOKENS, with TOL_LO and TOL_HI recomputed from P6's 1% tolerance;
  - N_NEEDLES[family] for a level other than P6's (multivalue k; the cliff set's multikey k 128 and vartrack c 4);
  - N_PER_FAMILY for the 96-item sets, so the generator's rotated depth grid stays inside (0, 1) (Amendment 3).
Everything is restored afterwards. Each item then passes the generator's own validate_item with a full recount, plus
this module's checks (check_item): needle count, distinct keys, values and names, chain order, gold answers, and the
no-think prompt at most L - 512 tokens.
"""
import contextlib
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "testsuite", "paper6"))
import make_longctx as G  # noqa: E402  (P6's frozen generator, unchanged)

PROMPT_MARGIN = 512
LEVELS = {"multivalue": (12, 24, 48)}                          # the graded family (Amendment 2); last is hardest
CALIB1_LEVELS = {"multikey": (16, 48, 128), "vartrack": (2, 3, 4)}  # the first calibration, as registered
CLIFF_EXTRA = {"multikey": 128, "vartrack": 4}                   # calibrated levels joining the cliff set (Amendment 2)
LETTER = {"multikey": "k", "multivalue": "k", "vartrack": "c", "single": "n"}
SEED_TEXT = {"calib": "p8-longctx-calib-2026-11", "test": "p8-longctx-2026-11",
             "cliff": "p8-longctx-cliff-2026-11"}               # cliff: Amendment 1, item 3
P6_N_NEEDLES = dict(G.N_NEEDLES)
P6_SETTINGS = {"SEED": G.SEED, "TARGET_TOKENS": G.TARGET_TOKENS, "TOL_LO": G.TOL_LO, "TOL_HI": G.TOL_HI,
               "N_NEEDLES": dict(G.N_NEEDLES), "N_PER_FAMILY": G.N_PER_FAMILY}


def seed_hex(kind):
    return hashlib.sha256(SEED_TEXT[kind].encode()).hexdigest()


def target_tokens(length, n_needles):
    """Haystack target that keeps the no-think prompt at most L - 512 tokens (Amendment 1, item 4): 128 tokens for the
    instruction, question and chat wrap, plus 2 per needle for tokenization at the insertion points."""
    return length - PROMPT_MARGIN - 128 - 2 * n_needles


@contextlib.contextmanager
def overrides(seed, target, family=None, level=None, n_per_family=None):
    saved = (G.SEED, G.TARGET_TOKENS, G.TOL_LO, G.TOL_HI, G.N_PER_FAMILY, dict(G.N_NEEDLES))
    try:
        G.SEED = seed
        G.TARGET_TOKENS = target
        G.TOL_LO, G.TOL_HI = int(target * (1 - G.TOLERANCE)) + 1, int(target * (1 + G.TOLERANCE))
        if family is not None and level is not None:
            G.N_NEEDLES[family] = level
        if n_per_family is not None:
            G.N_PER_FAMILY = n_per_family
        yield
    finally:
        G.SEED, G.TARGET_TOKENS, G.TOL_LO, G.TOL_HI, G.N_PER_FAMILY = saved[:5]
        G.N_NEEDLES.clear()
        G.N_NEEDLES.update(saved[5])


def prompt_tokens(content):
    return G.ntok(G.WRAP_NOTHINK[0] + content + G.WRAP_NOTHINK[1])


def check_item(rec, length, level):
    """The wrapper's checks for the overridden settings; returns a list of problems (empty = ok)."""
    p = []
    fam, c = rec["family"], rec["content"]
    needles = [n["text"] for n in rec["needles"]]
    if len(needles) != level:
        p.append(f"needle count {len(needles)} != {level}")
    if any(c.count(t) != 1 for t in needles):
        p.append("a needle does not occur exactly once")
    if fam in ("multikey", "single"):
        keys, gold = rec["keys"], rec["answers"]
        values = gold + rec["distractors"]
        if len(set(keys)) != len(keys) or len(set(values)) != len(values) or len(values) != level:
            p.append("keys or values not distinct")
        asked = [n for n in rec["needles"] if n["role"] == "asked"] if fam == "multikey" else rec["needles"]
        if len(gold) != 1 or len(asked) != 1 or gold[0] not in asked[0]["text"]:
            p.append("gold answer is not the asked needle's value")
        key_in_q = [k for k in keys if k in rec["question"]]
        if len(key_in_q) != 1 or key_in_q[0] not in asked[0]["text"]:
            p.append("question does not name exactly the asked key")
    elif fam == "vartrack":
        names = rec["answers"]
        if len(set(names)) != len(names) or len(names) != level:
            p.append("names not distinct or wrong count")
        pos = [c.index(t) for t in needles]
        if pos != sorted(pos):
            p.append("chain not in text order")
        if not all(names[j] in needles[j] for j in range(len(names))):
            p.append("gold names do not follow the chain")
        a, b = rec["haystack_span"]
        if c[a:b].count(rec["vt_value"]) != 1 or rec["vt_value"] not in needles[0]:  # the question names it too
            p.append("chain value not only in the first assignment")
    elif fam == "multivalue":
        if len(set(rec["answers"])) != level:
            p.append("values not distinct")
    n = prompt_tokens(c)
    if n != rec["prompt_tokens_qwen38_nothink"]:
        p.append(f"prompt recount {n} != recorded {rec['prompt_tokens_qwen38_nothink']}")
    if n > length - PROMPT_MARGIN:
        p.append(f"prompt {n} tokens > L - 512 = {length - PROMPT_MARGIN}")
    p += [f"generator: {x}" for x in G.validate_item(rec, recount=True)]
    return p


def build_one(pool, kind, family, level, index, length, n_per_family=None):
    """One checked item. `level` is the needle count; `n_per_family` overrides the depth grid's item count."""
    target = target_tokens(length, level)
    if index >= (n_per_family or G.N_PER_FAMILY):
        raise ValueError(f"index {index} is outside the depth grid of {n_per_family or G.N_PER_FAMILY} items")
    if kind == "cliff" and level not in (P6_N_NEEDLES[family], CLIFF_EXTRA.get(family)):
        raise ValueError(f"cliff items use P6's needle counts or the calibrated extras ({family}: {level})")
    own = level != P6_N_NEEDLES[family]
    with overrides(seed_hex(kind), target, family if own else None, level if own else None, n_per_family):
        rec = G.build_item(pool, family, index)
        problems = check_item(rec, length, level)
    if problems:
        raise RuntimeError(f"{kind} {family} level {level} index {index} at {length}: {problems}")
    letter = "n" if kind == "cliff" and not own else LETTER[family]
    uid = f"{family}-{letter}{level}-{index:02d}"
    it = {"uid": uid, "set_kind": kind, "family": family, "level": level, "index": index, "length": length,
          "target_tokens": target, "seed_text": SEED_TEXT[kind], "content": rec["content"],
          "answers": rec["answers"], "distractors": rec["distractors"], "question": rec["question"],
          "depths": rec["depths"], "asked_depth": rec["asked_depth"], "haystack_tokens": rec["haystack_tokens"],
          "prompt_tokens": rec["prompt_tokens_qwen38_nothink"], "n_sessions": len(rec["sessions"]),
          "sha256": rec["sha256"]}
    if n_per_family is not None:   # only when overridden, so sets built before Amendment 3 rebuild byte for byte
        it["n_per_family"] = n_per_family
    return it


def item_line(it):
    return json.dumps(it, sort_keys=True, ensure_ascii=False) + "\n"


def pool():
    return G.Pool(G.C5.load_json_verified(G.C5.LME_S))
