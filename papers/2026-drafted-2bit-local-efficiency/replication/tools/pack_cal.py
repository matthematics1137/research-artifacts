#!/usr/bin/env python3
"""Pack the model's own reasoning traces (cal_traces.jsonl) into the converter's -cd format: input_ids [250, 2048].
Own rows first (shuffled traces concatenated and chopped), then turboderp's self-calibration rows fill to 250
(TASA-style mix: task-focused on-policy traces + the diverse general trace). usage: pack_cal.py <jsonl> <out.safetensors>"""
import json, random, sys, torch
from tokenizers import Tokenizer
from safetensors.torch import save_file, load_file
SNAP = "/path/to/user-home/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
src, out = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(f"{SNAP}/tokenizer.json")
traces = [json.loads(l) for l in open(src) if l.strip()]
random.Random(0).shuffle(traces)
ids = []
for t in traces:
    txt = f"<|im_start|>user\n{t['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n{t['reasoning'].strip()}\n</think>\n\n{t['answer'].strip()}<|im_end|>\n"
    ids += tok.encode(txt, add_special_tokens=False).ids
cols = 2048; n = len(ids) // cols
own = torch.tensor(ids[: n * cols], dtype=torch.long).view(n, cols)
sc = load_file("/path/to/models/qwen38-exl3-2.0/cal_trace.safetensors")["input_ids"]
packed = torch.cat([own, sc[: max(0, 250 - n)]], 0)[:250]
assert int(packed.max()) < 248077 and packed.shape == (250, 2048), packed.shape
save_file({"input_ids": packed.contiguous()}, out)
print(json.dumps({"traces": len(traces), "own_tokens": len(ids), "own_rows": n, "filled_rows": int(packed.shape[0]) - n, "out": out}))
