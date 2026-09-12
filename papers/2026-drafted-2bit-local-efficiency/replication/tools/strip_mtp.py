#!/usr/bin/env python3
"""Strip token_embd / output / output_norm from an MTP sidecar GGUF (they duplicate the target's).
usage: strip_mtp.py in.gguf out.gguf
Uses gguf-py from the v0.4.0 tree. Copies every KV pair and every other tensor byte-for-byte.
"""
import sys
sys.path.insert(0, "/path/to/lab-repo/engines/llama.cpp-v0.4.0/gguf-py")
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

DROP = {"token_embd.weight", "output.weight", "output_norm.weight"}

def main(src, dst):
    r = GGUFReader(src)
    arch = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[0]]).decode("utf-8")
    w = GGUFWriter(dst, arch)
    skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for f in r.fields.values():
        if f.name in skip: continue
        vt = f.types[0]
        if vt == GGUFValueType.ARRAY:
            inner = f.types[1]
            vals = []
            for idx in f.data:
                p = f.parts[idx]
                if inner == GGUFValueType.STRING: vals.append(bytes(p).decode("utf-8"))
                else: vals.append(p.tolist()[0] if hasattr(p, "tolist") else p)
            w.add_array(f.name, vals)
        elif vt == GGUFValueType.STRING:
            w.add_string(f.name, bytes(f.parts[f.data[0]]).decode("utf-8"))
        else:
            v = f.parts[f.data[0]]
            v = v.tolist()[0] if hasattr(v, "tolist") else v
            {GGUFValueType.UINT8: w.add_uint8, GGUFValueType.INT8: w.add_int8, GGUFValueType.UINT16: w.add_uint16,
             GGUFValueType.INT16: w.add_int16, GGUFValueType.UINT32: w.add_uint32, GGUFValueType.INT32: w.add_int32,
             GGUFValueType.FLOAT32: w.add_float32, GGUFValueType.BOOL: w.add_bool, GGUFValueType.UINT64: w.add_uint64,
             GGUFValueType.INT64: w.add_int64, GGUFValueType.FLOAT64: w.add_float64}[vt](f.name, v)
    kept = 0; dropped = 0
    for t in r.tensors:
        if t.name in DROP:
            dropped += 1; print("drop", t.name, t.shape, t.tensor_type.name, f"{t.n_bytes/2**20:.1f} MiB"); continue
        shape = [int(x) for x in reversed(t.shape.tolist())]
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        kept += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"kept {kept} tensors, dropped {dropped} -> {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
