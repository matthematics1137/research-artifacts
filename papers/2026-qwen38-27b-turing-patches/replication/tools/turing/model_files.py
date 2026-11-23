#!/usr/bin/env python3
"""Paper 8's model files, measured from their own headers: parameters, tensor bytes and average bits per weight, with
each file's size and SHA-256 (from the 2026-09-06 hash audit where the Hub has a digest; hashed here otherwise).
GGUF files are read with llama.cpp's gguf-py (engines/llama.cpp/gguf-py); the EXL3 shards' trellis tensors count k x n
weights each (their shape is k/16 x n/16 x 16K words), and their small sign-flip vectors count no weights. Writes
testsuite/results/turing/model_files.json, which paper8/scripts/build_claims.py turns into a table, so the claim
build does not need the model files.
  python3 tools/turing/model_files.py
"""
import hashlib
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "engines", "llama.cpp", "gguf-py"))
import numpy as np            # noqa: E402
from gguf import GGUFReader   # noqa: E402

M = os.path.expanduser("~/models")
FILES = [  # (key, paths, format, source, used by)
    ("exl3", ["qwen38-exl3-2.0/model-00001-of-00002.safetensors", "qwen38-exl3-2.0/model-00002-of-00002.safetensors"],
     "EXL3, 2.0 bpw", "turboderp, revision SC_2.00bpw_H3", "every exllamav3 arm; rafatxf's fork"),
    ("iq2xxs", ["qwen38/Qwen3.8-27B-UD-IQ2_XXS.gguf"], "GGUF UD-IQ2_XXS", "unsloth", "llama.cpp, NInferno"),
    ("iq2s", ["qwen38/Qwen3.8-27B-UD-IQ2_S.gguf"], "GGUF UD-IQ2_S", "unsloth", "llama.cpp with MTP; quality lanes"),
    ("mtp_full", ["qwen38/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"], "GGUF Q4_0, MTP layer with embedding and output tables",
     "unsloth", "llama.cpp with MTP"),
    ("mtp_shared", ["qwen38/MTP-shared/mtp-Qwen3.8-27B-Q4_0-shared.gguf"], "GGUF Q4_0, MTP layer only",
     "ours, from unsloth's with the tables stripped", "the quality lanes' MTP arm"),
    ("dflash2", ["qwen38-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"], "GGUF Q4_K_M, block-diffusion drafter", "z-lab",
     "llama.cpp and NInferno with DFlash2"),
]


def gguf_counts(path):
    r = GGUFReader(path)
    return sum(int(np.prod(t.shape)) for t in r.tensors), sum(int(t.n_bytes) for t in r.tensors)


def exl3_counts(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    params = tbytes = qp = qb = 0
    for name, t in header.items():
        if name == "__metadata__":
            continue
        a, b = t["data_offsets"]
        tbytes += b - a
        if name.endswith(".trellis"):
            k16, n16, _ = t["shape"]
            params += k16 * 16 * n16 * 16
            qp, qb = qp + k16 * 16 * n16 * 16, qb + b - a
        elif not name.endswith((".suh", ".svh")):
            params += int(np.prod(t["shape"]))
    return params, tbytes, qp, qb


def main():
    audit = {os.path.expanduser(x["local"]): x for x in json.load(open(os.path.join(
        ROOT, "testsuite", "results", "efficiency", "hash_audit_2026-09-06.json")))}
    out = []
    for key, rel, fmt, source, used in FILES:
        params = tbytes = size = qp = qb = 0
        shas = []
        for r in rel:
            p = os.path.join(M, r)
            c = exl3_counts(p) if p.endswith(".safetensors") else gguf_counts(p)
            params, tbytes = params + c[0], tbytes + c[1]
            if len(c) == 4:   # EXL3: the trellis-coded part on its own
                qp, qb = qp + c[2], qb + c[3]
            size += os.path.getsize(p)
            if p in audit:
                shas.append(audit[p]["sha256"])
            else:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 24), b""):
                        h.update(chunk)
                shas.append(h.hexdigest())
        out.append({"key": key, "files": rel, "format": fmt, "source": source, "used_by": used, "parameters": params,
                    "tensor_bytes": tbytes, "file_bytes": size, "bits_per_weight": tbytes * 8 / params, "sha256": shas,
                    "trellis_parameters": qp or None, "trellis_bits_per_weight": (qb * 8 / qp) if qp else None,
                    "hub_digest_checked": all(os.path.join(M, r) in audit for r in rel)})
        print(f"{key:10s} {params / 1e9:7.3f} B params {size / 2**30:6.2f} GiB {tbytes * 8 / params:6.3f} bpw")
    with open(os.path.join(ROOT, "testsuite", "results", "turing", "model_files.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
