#!/usr/bin/env python3
"""P8 long context, stage 1: build the filled prompts, one per ladder length, with P6's frozen generator.

testsuite/paper6/make_longctx.py is imported unchanged. This wrapper only sets three of its module values before
calling it (Amendment 1, item 4): SEED to "p8-longctx-stage1", TARGET_TOKENS to the haystack size that makes the
no-think prompt at most L - 512 tokens, and the tolerance bounds the way P6 derives them from TARGET_TOKENS. Each
prompt is one "single" item (one needle, one key). Content does not affect memory; one prompt per length is shared
by every cache format. CPU only, on the laptop: the generator checks the Qwen3.8 tokenizer.json's SHA-256 and
LongMemEval-S's before use.

  engines/exl3-env/bin/python testsuite/p8_longctx/build_stage1_prompts.py [--out DIR]

Writes DIR/<L>.json per length (content plus its counts; not committed, rebuildable) and
testsuite/p8_longctx/stage1_prompts_manifest.json (hashes and counts; committed). Nothing is deleted.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "testsuite", "paper6"))
import make_longctx as G  # noqa: E402  (P6's frozen generator, unchanged)
import stage1_spec as S  # noqa: E402

SEED = "p8-longctx-stage1"
FAMILY, INDEX = "single", 0
OVERHEAD_GUESS = 80        # instruction + question + chat wrap, in tokens; corrected by the first build
WINDOW = 64                # accept a prompt within 64 tokens under its target (the filler stops up to 48 short)
CHECK_TOLERANCE = 0.06


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def set_target(t, tol):
    G.TARGET_TOKENS = t
    G.TOL_LO, G.TOL_HI = int(t * (1 - tol)) + 1, int(t * (1 + tol))


def build(pool, want, tol):
    """The item whose no-think prompt is the longest at or under `want` tokens, within WINDOW of it."""
    t = want - OVERHEAD_GUESS
    best = None
    for _ in range(8):
        set_target(t, tol)
        rec = G.build_item(pool, FAMILY, INDEX)
        n = rec["prompt_tokens_qwen38_nothink"]
        if n <= want and (best is None or n > best[1]["prompt_tokens_qwen38_nothink"]):
            best = (t, rec)
        if n <= want and want - n <= WINDOW:
            return t, rec
        t += want - n if n > want else max(1, (want - n) // 2)
    if best is None:
        raise SystemExit(f"no prompt at or under {want} tokens")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "work", "stage1_prompts"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "stage1_prompts_manifest.json"))
    ap.add_argument("--lengths", default="", help="comma-separated ladder lengths instead of stage 1's (no check "
                    "prompt); stage 3's reach probe uses 155648,172032 (stages 2-3 Amendment 1, item 13)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    G.SEED = SEED
    pool = G.Pool(G.C5.load_json_verified(G.C5.LME_S))
    if args.lengths:
        wants = [(int(x) - S.PROMPT_MARGIN, "ladder") for x in args.lengths.split(",")]
    else:
        wants = [(S.CHECK_PROMPT_TOKENS, "check")] + [(length - S.PROMPT_MARGIN, "ladder") for length in S.lengths()]
    rows = []
    for want, role in wants:
        # P6's 1% tolerance is narrower than the filler's 48-token slack at the 1K check prompt, so the check alone
        # allows 6% (Amendment 1, item 4); every ladder prompt keeps P6's 1%
        tol = CHECK_TOLERANCE if role == "check" else G.TOLERANCE
        t, rec = build(pool, want, tol)
        length = want + S.PROMPT_MARGIN if role == "ladder" else S.CHECK_STEP[1]
        body = {"length": length, "role": role, "want_prompt_tokens": want, "target_tokens": t, "seed": SEED,
                "item": rec["id"], "prompt_tokens_qwen38_nothink": rec["prompt_tokens_qwen38_nothink"],
                "haystack_tokens": rec["haystack_tokens"], "content_tokens": rec["content_tokens"],
                "answers": rec["answers"], "depths": rec["depths"], "n_sessions": len(rec["sessions"]),
                "sha256": rec["sha256"], "content": rec["content"]}
        name = f"{role}-{length}.json"
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False)
        rows.append({k: v for k, v in body.items() if k != "content"} | {"file": name})
        print(f"{role} L={length}: prompt {rec['prompt_tokens_qwen38_nothink']} (want <= {want}), "
              f"haystack {rec['haystack_tokens']}, target {t}", flush=True)
    paper6 = os.path.join(REPO, "testsuite", "paper6")
    manifest = {
        "schema": "p8-longctx-stage1-prompts-v1",
        "registration": "research/prereg/2026-11-01-p8-longctx-stage1.md (Amendment 1, item 4)",
        "generator": {"make_longctx.py": sha256_file(os.path.join(paper6, "make_longctx.py")),
                      "p6common.py": sha256_file(os.path.join(paper6, "p6common.py")),
                      "p5common.py": sha256_file(os.path.join(REPO, "testsuite", "paper5", "p5common.py"))},
        "tokenizer_sha256": G.P.TOKENIZER_QWEN38_SHA256,
        "family": FAMILY, "index": INDEX, "seed": SEED, "prompts": rows,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
