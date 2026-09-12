#!/usr/bin/env python3
"""Ledger item 22: byte-pin every benchmarked Qwen3.8-27B artifact against the Hub (LFS sha256).
Lists each candidate repo tree (expand=True -> lfs.sha256), hashes the local files (sha256, streaming),
and reports MATCH / MISMATCH / NOT_ON_HUB per file. Output: hash_audit.json next to this script."""
import hashlib, json, os, sys, time
from huggingface_hub import HfApi

LOCAL = [
    "~/models/qwen38/Qwen3.8-27B-UD-IQ2_S.gguf", "~/models/qwen38/Qwen3.8-27B-UD-IQ2_XXS.gguf",
    "~/models/qwen38/Qwen3.8-27B-UD-Q4_K_XL.gguf", "~/models/qwen38/mmproj-F16.gguf",
    "~/models/qwen38/MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "~/models/qwen38-bartowski/Qwen3.8-27B-IQ2_S.gguf",
    "~/models/qwen38-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    "~/models/qwen38-exl3-2.0/model-00001-of-00002.safetensors", "~/models/qwen38-exl3-2.0/model-00002-of-00002.safetensors",
    "~/models/qwen38-exl3-2.2v3/model-00001-of-00002.safetensors", "~/models/qwen38-exl3-2.2v3/model-00002-of-00002.safetensors",
]
REPOS = [("unsloth/Qwen3.8-27B-GGUF", None), ("bartowski/Qwen3.8-27B-GGUF", None), ("z-lab/Qwen3.8-27B-DFlash2", None),
         ("z-lab/Qwen3.8-27B-DFlash2-GGUF", None), ("turboderp/Qwen3.8-27B-exl3", "SC_2.00bpw_H3"),
         ("turboderp/Qwen3.8-27B-exl3", "SC_2.00bpw_H3_V3"), ("turboderp/Qwen3.8-27B-exl3", "SC_2.20bpw_H3_V3"),
         ("turboderp/Qwen3.8-27B-exl3", "SC_2.20bpw_H3")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(64 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()


def main():
    api = HfApi(); remote = {}  # sha256 -> [(repo, rev, path, size)]
    by_name = {}
    for repo, rev in REPOS:
        try:
            for e in api.list_repo_tree(repo, revision=rev, recursive=True, expand=True):
                lfs = getattr(e, "lfs", None)
                if lfs and lfs.get("sha256"):
                    remote.setdefault(lfs["sha256"], []).append((repo, rev, e.path, lfs.get("size")))
                    by_name.setdefault(os.path.basename(e.path), []).append((repo, rev, e.path, lfs["sha256"], lfs.get("size")))
            print(f"listed {repo}@{rev}", flush=True)
        except Exception as ex:
            print(f"SKIP {repo}@{rev}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
    out = []
    for lp in LOCAL:
        p = os.path.expanduser(lp)
        if not os.path.exists(p):
            out.append({"local": lp, "status": "MISSING"}); print("MISSING", lp, flush=True); continue
        t0 = time.time(); h = sha256(p); size = os.path.getsize(p)
        hits = remote.get(h, [])
        status = "MATCH" if hits else ("MISMATCH" if os.path.basename(p) in by_name else "NOT_ON_HUB")
        row = {"local": lp, "bytes": size, "sha256": h, "status": status, "hub": hits,
               "hub_same_name": [x[:3] + (x[3][:12], x[4]) for x in by_name.get(os.path.basename(p), [])] if not hits else [],
               "hash_s": round(time.time() - t0, 1)}
        out.append(row); print(status, lp, size, h[:16], hits[:2] if hits else row["hub_same_name"][:2], flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hash_audit.json"), "w"), indent=1)
    print("HASH_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
