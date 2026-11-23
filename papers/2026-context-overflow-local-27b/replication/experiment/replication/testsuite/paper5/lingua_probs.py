#!/usr/bin/env python3
"""Paper 5, arm C: compute LLMLingua-2 keep-probabilities ONCE per LongMemEval-S item.

The library (llmlingua 0.2.2, `compress_prompt` with default arguments, as in P2R1)
chunks the text, classifies every chunk, and then keeps, per chunk, the words whose
probability exceeds the int(100*(1-rate)+1)-th percentile of the token-weighted word
probabilities. That selection is the only rate-dependent step, so this script stores
each chunk's words, word probabilities, and token weights, and `select()` re-applies
the library's rule for any percentile. `--selftest` checks byte identity against the
library's own compress_prompt on real items before anything is written.

Run in engines/lingua-env (GPU when free):
  engines/lingua-env/bin/python testsuite/paper5/lingua_probs.py --selftest
  engines/lingua-env/bin/python testsuite/paper5/lingua_probs.py --all
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p5common as C  # noqa: E402

MODEL_REVISION = "ebaba9b0e874dadd3003ffcff828e4397e568089"
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/"
    + MODEL_REVISION)
OUT = os.path.join(C.WORK, "lingua")


def load_compressor(device):
    from llmlingua import PromptCompressor
    return PromptCompressor(model_name=MODEL_PATH, use_llmlingua2=True, device_map=device)


def chunk_probs(comp, text: str):
    """Replicates PromptCompressor.__compress up to the threshold (rate-independent part)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from llmlingua.utils import TokenClfDataset

    chunks = comp._PromptCompressor__chunk_context(text, chunk_end_tokens={".", "\n"})
    dataset = TokenClfDataset(chunks, tokenizer=comp.tokenizer, max_len=comp.max_seq_len)
    loader = DataLoader(dataset, batch_size=comp.max_batch_size, shuffle=False, drop_last=False)
    out = []
    k = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(comp.device, dtype=torch.long)
            mask = batch["mask"].to(comp.device, dtype=torch.long) == 1
            probs = F.softmax(comp.model(input_ids=ids, attention_mask=mask).logits, dim=-1)
            for j in range(ids.shape[0]):
                active_probs = torch.masked_select(probs[j, :, 1], mask[j])
                active_ids = torch.masked_select(ids[j], mask[j])
                tokens = comp.tokenizer.convert_ids_to_tokens(active_ids.squeeze().tolist())
                token_probs = [p for p in active_probs.cpu().numpy()]
                words, valid_token_probs, _ = comp._PromptCompressor__merge_token_to_word(
                    tokens=tokens, token_probs=token_probs, force_tokens=[], token_map={},
                    force_reserve_digit=False)
                word_probs = comp._PromptCompressor__token_prob_to_word_prob(
                    valid_token_probs, convert_mode="mean")
                n_oai = [len(comp.oai_tokenizer.encode(w)) for w in words]
                out.append({"text": chunks[k], "words": words,
                            "p": [float(x) for x in word_probs], "n": n_oai})
                k += 1
    assert k == len(chunks)
    return out


def select_chunk(comp_tokenizer, ch: dict, percentile: int) -> str:
    """The library's per-chunk keep rule (drop_consecutive=False, no force tokens)."""
    probs = ch["p"]
    new_token_probs = []
    for p, n in zip(probs, ch["n"]):
        new_token_probs.extend([p] * n)
    threshold = np.percentile(new_token_probs, percentile)
    keep = [w for w, p in zip(ch["words"], probs)
            if p > threshold or (threshold == 1.0 and p == threshold)]
    return comp_tokenizer.convert_tokens_to_string(keep)


def select(comp_tokenizer, chunks: list, percentile: int) -> str:
    return "".join(select_chunk(comp_tokenizer, ch, percentile) for ch in chunks)


def percentile_for_rate(rate: float) -> int:
    return int(100 * max(0, 1 - rate) + 1)


def selftest(comp, items):
    for it in items[:3]:
        text = C.full_history(it)
        chunks = chunk_probs(comp, text)
        for rate in (0.5, 0.2, 0.05, 0.02):
            lib = comp.compress_prompt(text, rate=rate)["compressed_prompt"]
            mine = select(comp.tokenizer, chunks, percentile_for_rate(rate))
            status = "PASS" if lib == mine else "FAIL"
            print(f"{status} {it['question_id']} rate={rate} lib={len(lib)} mine={len(mine)}", flush=True)
            if lib != mine:
                raise SystemExit("selftest failed: cached selection differs from compress_prompt")
    print("SELFTEST PASSED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    items = C.load_json_verified(C.LME_S)
    comp = load_compressor(a.device)
    if a.selftest:
        selftest(comp, items)
    if a.all:
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        for n, it in enumerate(items):
            path = os.path.join(OUT, it["question_id"] + ".json.gz")
            if os.path.exists(path):
                continue
            text = C.full_history(it)
            chunks = chunk_probs(comp, text)
            rec = {"question_id": it["question_id"], "history_sha256": C.sha256_text(text),
                   "model_revision": MODEL_REVISION, "device": a.device, "chunks": chunks}
            tmp = path + ".tmp"
            with gzip.open(tmp, "wt") as f:
                json.dump(rec, f)
            os.replace(tmp, path)
            if n % 20 == 0:
                print(f"{n+1}/500 {time.time()-t0:.0f}s", flush=True)
        print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
