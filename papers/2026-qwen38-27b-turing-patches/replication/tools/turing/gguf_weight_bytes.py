#!/usr/bin/env python3
"""Per-token weight bytes of a GGUF: every tensor but the embedding table, which decoding reads one row per token.
Sizes come from the file's tensor table (consecutive data offsets), so no quantization table is needed. Stdlib only.

  python3 gguf_weight_bytes.py FILE.gguf ... > testsuite/results/turing/census/weight_bytes_gguf.json
"""
import json
import os
import struct
import sys

SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def text(f):
    (n,) = read(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def skip(f, kind):
    if kind == 8:
        text(f)
    elif kind == 9:
        (inner,), (n,) = read(f, "I"), read(f, "Q")
        if inner == 8:
            for _ in range(n):
                text(f)
        else:
            f.seek(SIZE[inner] * n, 1)
    else:
        f.seek(SIZE[kind], 1)


def tensor_bytes(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        read(f, "I")
        n_tensors, n_kv = read(f, "QQ")
        align = 32
        for _ in range(n_kv):
            key = text(f)
            (kind,) = read(f, "I")
            if key == "general.alignment":
                (align,) = read(f, "I")
            else:
                skip(f, kind)
        infos = []
        for _ in range(n_tensors):
            name = text(f)
            (nd,) = read(f, "I")
            read(f, "Q" * nd)
            read(f, "I")
            (offset,) = read(f, "Q")
            infos.append((offset, name))
        start = f.tell() + (-f.tell()) % align
    size = os.path.getsize(path)
    infos.sort()
    return {name: (infos[i + 1][0] if i + 1 < len(infos) else size - start) - off for i, (off, name) in enumerate(infos)}


def main():
    out = {}
    for path in sys.argv[1:]:
        t = tensor_bytes(path)
        emb = sum(v for k, v in t.items() if k.startswith("token_embd"))
        out[os.path.basename(path)] = {"file_bytes": os.path.getsize(path), "tensor_bytes": sum(t.values()),
                                       "token_embd_bytes": emb, "output_bytes": sum(v for k, v in t.items()
                                                                                    if k.startswith("output.")),
                                       "per_token_bytes": sum(t.values()) - emb, "tensors": len(t)}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
