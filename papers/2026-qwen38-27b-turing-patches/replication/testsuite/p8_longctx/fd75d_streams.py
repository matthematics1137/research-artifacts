#!/usr/bin/env python3
"""P8 fd75d logit check (research/prereg/2026-11-03-p8-fd75d-check.md): the teacher-forced token streams for
tools/turing/numerics.py --streams. Runs ON the desktop, CPU only, with the model's own tokenizer.

Each item's prompt is rendered as TabbyAPI rendered it in stage 2 (P6's no-think chat wrap); its response is the item's
12 gold values, one per line, then <|im_end|>. Rows carry the fields numerics.py reads: id, prompt_ids, response_ids.

  python fd75d_streams.py --items DIR --out streams.jsonl
"""
import argparse
import json
import os

MODEL = os.path.join(os.environ.get("TURING_LAB", "/opt/p8"), "models", "qwen38-exl3-2.0")
SETS = (("fdcheck-28k.jsonl", 28 * 1024), ("fdcheck-48k.jsonl", 48 * 1024), ("fdcheck-64k.jsonl", 64 * 1024))
WRAP = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")   # P6 WRAP_NOTHINK


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from exllamav3 import Config, Tokenizer
    tok = Tokenizer.from_config(Config.from_directory(MODEL))
    end = tok.single_id("<|im_end|>")
    rows = []
    for name, length in SETS:
        with open(os.path.join(args.items, name), encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                prompt = tok.encode(WRAP[0] + it["content"] + WRAP[1], add_bos=False,
                                    encode_special_tokens=True)[0].tolist()
                if len(prompt) != it["prompt_tokens"] or len(prompt) > length - 512:
                    raise SystemExit(f"{name} {it['uid']}: {len(prompt)} prompt tokens, built {it['prompt_tokens']}")
                response = tok.encode("\n".join(it["answers"]), add_bos=False)[0].tolist() + [end]
                rows.append({"id": f"{length // 1024}k-{it['uid']}", "suite": "fdcheck", "length": length,
                             "item_sha256": it["sha256"], "prompt_ids": prompt, "response_ids": response})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} streams -> {args.out}; prompts "
          f"{[len(r['prompt_ids']) for r in rows]}, responses {[len(r['response_ids']) for r in rows]}")


if __name__ == "__main__":
    main()
