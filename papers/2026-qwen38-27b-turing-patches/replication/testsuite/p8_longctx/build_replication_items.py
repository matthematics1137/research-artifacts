#!/usr/bin/env python3
"""P8 value-bits replication: build R1's fresh item set and its manifest
(research/prereg/2026-11-04-p8-value-bits-replication.md).

R1 needs items the split cache's 64K lead was never seen on: the graded design unchanged (multivalue at k = 12, 96
items, indices 0-95 on the 96-item depth grid, 64K), from a new seed text, "p8-value-bits-replication-2026-11". The
item ids carry a "rep-" prefix so no analysis can pair them with the original items, whose ids are otherwise the same.

items.py and build_stage2_items.py stay byte-identical (their hashes are in the earlier manifests): this script adds
the new seed text to items.SEED_TEXT at run time, under its own set kind "rep", and builds through items.build_one,
which runs P6's frozen generator unchanged with the same overrides and checks as every earlier set.

Writes OUT/items/rep-64k.jsonl (prompt text; not committed, rebuildable byte for byte) and OUT/manifests/rep-64k.json
(hashes, counts, gold answers; committed before any GPU step uses the set). HOME must hold the model's tokenizer.json
(the generator checks its SHA-256), as for build_stage2_items.py.

  python build_replication_items.py --out DIR
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I  # noqa: E402
from build_stage2_items import GRID96, sha256_file  # noqa: E402

K = 1024
KIND = "rep"
SEED_TEXT = "p8-value-bits-replication-2026-11"
LENGTH = 64 * K
LEVEL = 12


def plan():
    """(family, level, index, n_per_family) in manifest order: the graded design, new seed."""
    return [("multivalue", LEVEL, i, GRID96) for i in range(96)]


def build(pool, length=LENGTH, indices=None):
    I.SEED_TEXT.setdefault(KIND, SEED_TEXT)
    if I.SEED_TEXT[KIND] != SEED_TEXT:
        raise SystemExit(f"items.SEED_TEXT[{KIND!r}] is already {I.SEED_TEXT[KIND]!r}")
    out = []
    for fam, level, index, grid in plan():
        if indices is not None and index not in indices:
            continue
        it = I.build_one(pool, KIND, fam, level, index, length, grid)
        it["uid"] = "rep-" + it["uid"]
        out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    name = f"rep-{LENGTH // K}k"
    os.makedirs(os.path.join(args.out, "items"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "manifests"), exist_ok=True)
    built = []
    for it in build(I.pool()):
        built.append(it)
        print(f"{it['uid']}: prompt {it['prompt_tokens']} (limit {LENGTH - I.PROMPT_MARGIN}), "
              f"haystack {it['haystack_tokens']}", flush=True)
    if len({it["uid"] for it in built}) != len(built):
        raise SystemExit("duplicate item ids")
    blob = "".join(I.item_line(it) for it in built).encode("utf-8")
    with open(os.path.join(args.out, "items", name + ".jsonl"), "wb") as f:
        f.write(blob)
    paper6 = os.path.join(I.REPO, "testsuite", "paper6")
    manifest = {
        "schema": "p8-longctx-items-v1",
        "registration": "research/prereg/2026-11-04-p8-value-bits-replication.md",
        "set": name, "kind": KIND, "length": LENGTH, "prompt_limit": LENGTH - I.PROMPT_MARGIN,
        "seed_text": SEED_TEXT, "seed_sha256": hashlib.sha256(SEED_TEXT.encode()).hexdigest(),
        "levels": {"multivalue": LEVEL}, "n_items": len(built),
        "file": {"name": name + ".jsonl", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)},
        "code": {"make_longctx.py": sha256_file(os.path.join(paper6, "make_longctx.py")),
                 "p6common.py": sha256_file(os.path.join(paper6, "p6common.py")),
                 "items.py": sha256_file(os.path.join(HERE, "items.py")),
                 "build_stage2_items.py": sha256_file(os.path.join(HERE, "build_stage2_items.py")),
                 "build_replication_items.py": sha256_file(os.path.abspath(__file__))},
        "tokenizer_sha256": I.G.P.TOKENIZER_QWEN38_SHA256,
        "items": [{k: it[k] for k in ("uid", "family", "level", "index", "prompt_tokens", "haystack_tokens",
                                      "target_tokens", "answers", "asked_depth", "sha256")} for it in built],
    }
    with open(os.path.join(args.out, "manifests", name + ".json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    toks = [it["prompt_tokens"] for it in built]
    print(f"{name}: {len(built)} items, prompts {min(toks)}-{max(toks)} tokens, sha256 {manifest['file']['sha256']}")


if __name__ == "__main__":
    main()
