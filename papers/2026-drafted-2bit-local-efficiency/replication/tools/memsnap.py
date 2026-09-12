#!/usr/bin/env python3
"""E28 instrumented load: where does the MTP draft path allocate its ~2.4 GiB? Loads the main model (+Q8 cache),
then the MTP component and the generator exactly like TabbyAPI does, printing torch allocated/reserved after each
step and the largest live allocations with their exllamav3 stack frames. usage: memsnap.py [mtp|none]"""
import os, sys, time, torch
EXL = "/path/to/scratch/eff/exl3_147"
sys.path.insert(0, EXL)
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.cache import CacheLayer_quant

mode = sys.argv[1] if len(sys.argv) > 1 else "mtp"
MBS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
HIST = int(sys.argv[3]) if len(sys.argv) > 3 else 4
print(f"mode={mode} max_batch_size={MBS} max_history={HIST}", flush=True)
d = os.path.expanduser("~/models/qwen38-exl3-2.0")
torch.cuda.memory._record_memory_history(max_entries=300000)


def mem(tag):
    torch.cuda.synchronize()
    print(f"[{tag}] alloc {torch.cuda.memory_allocated() >> 20} MiB  reserved {torch.cuda.memory_reserved() >> 20} MiB", flush=True)


def big_blocks(tag, min_mib=150):
    snap = torch.cuda.memory._snapshot()
    rows = []
    for seg in snap["segments"]:
        for b in seg.get("blocks", []):
            if b.get("state") == "active_allocated" and b["size"] >= min_mib << 20:
                fr = [f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in b.get("frames", []) if "exllamav3" in f["filename"] or "tabby" in f["filename"].lower()]
                rows.append((b["size"] >> 20, fr[:7]))
    print(f"--- {tag}: live blocks >= {min_mib} MiB", flush=True)
    for s, fr in sorted(rows, key=lambda r: -r[0])[:15]:
        print(f"  {s:6d} MiB  {' <- '.join(fr) if fr else '(no exllamav3 frames)'}", flush=True)


cfg = Config.from_directory(d)
model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens=12288, layer_type=CacheLayer_quant, k_bits=8, v_bits=8, max_batch_size=MBS, max_history=HIST)
model.load(progressbar=False, max_chunk_size=1024, max_batch_size=MBS)
mem("main model + Q8 cache loaded")
tok = Tokenizer.from_config(cfg)
draft = dcache = None
if mode == "mtp":
    draft = Model.from_config(cfg, component="mtp")
    dcache = Cache(draft, max_num_tokens=12288, layer_type=CacheLayer_quant, k_bits=8, v_bits=8, max_batch_size=MBS, max_history=HIST)
    draft.load(progressbar=False)
    mem("draft (mtp) loaded")
    big_blocks("after draft load")
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache, max_batch_size=MBS, max_chunk_size=1024,
                num_draft_tokens=4, dynamic_draft_tokens=True)
mem("generator created")
big_blocks("after generator")
prompt = "<|im_start|>user\nSay hi in five words.<|im_end|>\n<|im_start|>assistant\n<think>\n"
t0 = time.time()
out = gen.generate(prompt=prompt, max_new_tokens=48, add_bos=False)
mem(f"after generate ({time.time() - t0:.1f}s)")
big_blocks("after generate")
print("OUT:", repr(out[-120:]), flush=True)
