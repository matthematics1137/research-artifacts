#!/usr/bin/env python3
"""H31 control: an OFF-POLICY calibration set with no reasoning traces — half generic prose (/usr/share/doc text),
half Python code (the 3.12 stdlib, tests excluded) — packed to the converter's [250, 2048] input_ids format.
usage: pack_offpolicy.py <out.safetensors>"""
import glob, os, random, sys, torch
from tokenizers import Tokenizer
from safetensors.torch import save_file
SNAP = "/path/to/user-home/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
tok = Tokenizer.from_file(f"{SNAP}/tokenizer.json"); rnd = random.Random(3)

def read_ok(p, lo=2000, hi=400000):
    try:
        b = open(p, "rb").read(hi)
        if len(b) < lo or b"\x00" in b: return None
        return b.decode("utf-8")
    except Exception:
        return None

prose = [t for p in glob.glob("/usr/share/doc/**/*", recursive=True) if os.path.isfile(p) and (p.endswith((".txt", ".md", ".markdown")) or "README" in os.path.basename(p)) for t in [read_ok(p)] if t]
code = [t for p in glob.glob("/usr/lib/python3.12/**/*.py", recursive=True) if "/test" not in p and "site-packages" not in p for t in [read_ok(p)] if t]
rnd.shuffle(prose); rnd.shuffle(code)
def pack(texts, rows, cols=2048):
    ids = []
    for t in texts:
        ids += tok.encode(t, add_special_tokens=False).ids + [tok.token_to_id("<|endoftext|>") or 0]
        if len(ids) >= rows * cols: break
    assert len(ids) >= rows * cols, (len(ids), rows * cols)
    return torch.tensor(ids[: rows * cols], dtype=torch.long).view(rows, cols)
packed = torch.cat([pack(prose, 125), pack(code, 125)], 0)
perm = torch.randperm(250, generator=torch.Generator().manual_seed(3)); packed = packed[perm].contiguous()
assert int(packed.max()) < 248077
save_file({"input_ids": packed}, sys.argv[1])
print({"prose_files": len(prose), "code_files": len(code), "shape": list(packed.shape), "out": sys.argv[1]})
