#!/usr/bin/env python3
"""P8 long context, stages 2-3: build one item set and its manifest (research/prereg/2026-11-01-p8-longctx-stages2-3.md,
Amendment 1).

Sets (Amendments 1-3):
  calib     the first calibration, as registered: 48 items at 64K, the calibration seed, 8 per level per family,
            index 6j + 2l for item j of level l, in the order j, family, level (multikey k 16/48/128, vartrack c 2/3/4)
  calib2    the second calibration (Amendment 2): 24 multivalue items at 64K, the calibration seed, 8 per level
            (k 12/24/48), index 6j + 2l, in the order j, level
  graded    96 multivalue items at the chosen level, indices 0-95 on a 96-item depth grid (--length, --levels)
  stage3    48 multivalue items at the chosen level, the even indices 0-94 on the same grid (--length, --levels)
  cliff     the cliff seed: P6's four families at P6's needle counts, indices 0-7 (order index, family), then the
            calibrated multikey k 128 and vartrack c 4, indices 8-15 (order index, family): 48 items (--length)
  fdcheck   the fd75d logit check (research/prereg/2026-11-03-p8-fd75d-check.md): multivalue k 12, the test seed,
            indices 0-2 on the 96-item grid, the same items as graded's first three (--length)

Writes OUT/items/<set>-<L/1024>k.jsonl (prompt text; not committed, rebuildable byte for byte) and
OUT/manifests/<set>-<L/1024>k.json (hashes, counts, gold answers; committed before any GPU step uses the set).
Runs in zarael-sandbox with HOME pointing at a folder that holds the model's tokenizer.json, whose SHA-256 the
generator checks.

  python build_stage2_items.py calib --out DIR
  python build_stage2_items.py graded --length 65536 --levels multivalue=24 --out DIR
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I  # noqa: E402

K = 1024


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


GRID96 = 96   # depth-grid items for the 96-item sets and their stage 3 subsets (Amendment 3)


def plan(kind, levels):
    """(family, level, index, n_per_family) in manifest order."""
    if kind == "calib":
        return [(fam, I.CALIB1_LEVELS[fam][lv], 6 * j + 2 * lv, None) for j in range(8)
                for fam in ("multikey", "vartrack") for lv in range(3)]
    if kind == "calib2":
        return [("multivalue", I.LEVELS["multivalue"][lv], 6 * j + 2 * lv, None) for j in range(8) for lv in range(3)]
    if kind in ("graded", "stage3"):
        idx = range(96) if kind == "graded" else range(0, 96, 2)
        return [("multivalue", levels["multivalue"], i, GRID96) for i in idx]
    if kind == "fdcheck":
        return [("multivalue", 12, i, GRID96) for i in range(3)]
    if kind == "cliff":
        p6 = [(fam, I.P6_N_NEEDLES[fam], i, None) for i in range(8)
              for fam in ("single", "multikey", "multivalue", "vartrack")]
        return p6 + [(fam, I.CLIFF_EXTRA[fam], i, None) for i in range(8, 16) for fam in ("multikey", "vartrack")]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", choices=("calib", "calib2", "graded", "stage3", "cliff", "fdcheck"))
    ap.add_argument("--length", type=int, default=64 * K)
    ap.add_argument("--levels", default="", help="multivalue=K (graded and stage3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    levels = dict((k, int(v)) for k, v in (x.split("=") for x in args.levels.split(",") if x))
    if args.set in ("graded", "stage3"):
        if set(levels) != set(I.LEVELS) or any(levels[f] not in I.LEVELS[f] for f in levels):
            raise SystemExit(f"--levels must pick one registered level per graded family: {I.LEVELS}")
    if args.set in ("calib", "calib2") and args.length != 64 * K:
        raise SystemExit("calibration is registered at 64K")
    kind = {"calib": "calib", "calib2": "calib", "graded": "test", "stage3": "test", "cliff": "cliff",
            "fdcheck": "test"}[args.set]
    name = f"{args.set}-{args.length // K}k"
    os.makedirs(os.path.join(args.out, "items"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "manifests"), exist_ok=True)
    pool = I.pool()
    built = []
    for fam, level, index, grid in plan(args.set, levels):
        it = I.build_one(pool, kind, fam, level, index, args.length, grid)
        built.append(it)
        print(f"{it['uid']}: prompt {it['prompt_tokens']} (limit {args.length - I.PROMPT_MARGIN}), "
              f"haystack {it['haystack_tokens']}", flush=True)
    if len({it["uid"] for it in built}) != len(built):
        raise SystemExit("duplicate item ids")
    blob = "".join(I.item_line(it) for it in built).encode("utf-8")
    items_path = os.path.join(args.out, "items", name + ".jsonl")
    with open(items_path, "wb") as f:
        f.write(blob)
    paper6 = os.path.join(I.REPO, "testsuite", "paper6")
    manifest = {
        "schema": "p8-longctx-items-v1",
        "registration": "research/prereg/2026-11-01-p8-longctx-stages2-3.md (Amendment 1)",
        "set": name, "kind": args.set, "length": args.length, "prompt_limit": args.length - I.PROMPT_MARGIN,
        "seed_text": I.SEED_TEXT[kind], "seed_sha256": I.seed_hex(kind),
        "levels": levels if levels else {"calib": {f: list(v) for f, v in I.CALIB1_LEVELS.items()},
                                         "calib2": {f: list(v) for f, v in I.LEVELS.items()},
                                         "cliff": {"p6": I.P6_N_NEEDLES, "calibrated": I.CLIFF_EXTRA},
                                         "fdcheck": {"multivalue": 12}}[args.set],
        "n_items": len(built),
        "file": {"name": name + ".jsonl", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)},
        "code": {"make_longctx.py": sha256_file(os.path.join(paper6, "make_longctx.py")),
                 "p6common.py": sha256_file(os.path.join(paper6, "p6common.py")),
                 "items.py": sha256_file(os.path.join(HERE, "items.py")),
                 "build_stage2_items.py": sha256_file(os.path.abspath(__file__))},
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
