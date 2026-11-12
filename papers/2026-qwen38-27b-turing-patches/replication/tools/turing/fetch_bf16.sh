#!/bin/bash
# Fetch the original Qwen3.8-27B BF16 checkpoint into the lab (desktop ssd2), pinned to the revision the lab
# downloaded on 2026-08-25 and backed up to Drive on 2026-09-12 (repo unchanged since 2026-08-14), then check
# every file's SHA-256 against the Hub's LFS record. Approved by Matt 2026-10-30. Nothing goes outside the lab;
# nothing is deleted. Run inside a capped unit, for example:
#   systemd-run --user --unit=turing-fetch-bf16 --collect -p Nice=15 -p IOSchedulingClass=idle -p CPUQuota=200% \
#     -p MemoryMax=4G bash /opt/p8/tools/fetch_bf16.sh
set -euo pipefail
L=${TURING_LAB:-/opt/p8}
REPO=Qwen/Qwen3.8-27B
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
DEST=$L/models/qwen38-bf16
export HF_HOME=$L/cache/huggingface HF_XET_CACHE=$L/cache/huggingface/xet XDG_CACHE_HOME=$L/cache
exec >> "$L/logs/fetch-bf16.log" 2>&1
echo "== $(date -Is) fetch $REPO@$REV -> $DEST"
avail=$(df -BG --output=avail "$L" | tail -1 | tr -dc 0-9)
[ "$avail" -ge 120 ] || { echo "STOP: under 120 GB free on the lab drive"; exit 1; }
"$L/venv/bin/hf" download "$REPO" --revision "$REV" --local-dir "$DEST" --max-workers 4
curl -sS --max-time 60 "https://huggingface.co/api/models/$REPO/tree/$REV" > "$L/logs/qwen38-bf16-tree.json"
"$L/venv/bin/python" - "$DEST" "$L/logs/qwen38-bf16-tree.json" <<'EOF'
import hashlib, json, os, sys
dest, tree = sys.argv[1], json.load(open(sys.argv[2]))
bad = 0
for entry in tree:
    path = os.path.join(dest, entry["path"])
    if not os.path.isfile(path) or os.path.getsize(path) != entry["size"]:
        print("MISSING OR WRONG SIZE", entry["path"]); bad += 1; continue
    if "lfs" in entry:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != entry["lfs"]["oid"]:
            print("SHA256 MISMATCH", entry["path"]); bad += 1
print(f"checked {len(tree)} files, {bad} problems")
sys.exit(1 if bad else 0)
EOF
echo "BF16 VERIFIED $(du -sb "$DEST" | cut -f1) bytes"
echo "== $(date -Is) done"
