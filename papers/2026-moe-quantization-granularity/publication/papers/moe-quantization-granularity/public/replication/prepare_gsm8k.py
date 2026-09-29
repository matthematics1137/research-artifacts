#!/usr/bin/env python3
"""Rebuild the study's 100-item GSM8K subset from a pinned upstream file.

The release does not redistribute benchmark questions or answers. This script
downloads the upstream MIT-licensed test split at a fixed Git revision, checks
its bytes, repeats the original seed-42 selection, and verifies the resulting
item IDs against the public measurement rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
UPSTREAM_REVISION = "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{UPSTREAM_REVISION}/grade_school_math/data/test.jsonl"
)
UPSTREAM_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
SUBSET_SHA256 = "184d25be0b9e2e53ba6611fda207ebef7987d69cfc1695e003be2328deca5d37"
SEED = 42
SOURCE_ROWS = 1319
SUBSET_ROWS = 100


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def expected_item_ids() -> list[str]:
    return [
        f"gsm8k_{index}"
        for index in sorted(random.Random(SEED).sample(range(SOURCE_ROWS), SUBSET_ROWS))
    ]


def obtain_source(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.stat().st_mode & 0o077:
        raise SystemExit(f"benchmark source directory must be owner-only (0700): {path.parent}")
    if path.is_file():
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise SystemExit(f"benchmark source must be a private regular file: {path}")
        payload = path.read_bytes()
    else:
        if path.exists() or path.is_symlink():
            raise SystemExit(f"unsafe benchmark source path: {path}")
        request = urllib.request.Request(
            UPSTREAM_URL,
            headers={"User-Agent": "paper4-gsm8k-replication/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        with path.open("xb") as handle:
            handle.write(payload)
        os.chmod(path, 0o600)
    digest = sha256_bytes(payload)
    if digest != UPSTREAM_SHA256:
        raise SystemExit(f"refusing upstream GSM8K bytes with SHA-256 {digest}")
    return payload


def build(payload: bytes) -> list[dict[str, object]]:
    source = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(source) != SOURCE_ROWS:
        raise SystemExit(f"upstream test split has {len(source)} rows, expected {SOURCE_ROWS}")
    indices = sorted(random.Random(SEED).sample(range(len(source)), SUBSET_ROWS))
    rows: list[dict[str, object]] = []
    for index in indices:
        row = source[index]
        match = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", row["answer"])
        if not match:
            raise SystemExit(f"no numeric answer marker in upstream row {index}")
        answer = float(match.group(1).replace(",", ""))
        rows.append(
            {
                "id": f"gsm8k_{index}",
                "question": row["question"].strip(),
                "answer_number": answer,
            }
        )
    if [row["id"] for row in rows] != expected_item_ids():
        raise SystemExit("seed-42 item IDs do not match the public primary evidence")
    return rows


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=HERE / "work" / f"gsm8k-test-{UPSTREAM_REVISION}.jsonl",
        help="cached pinned upstream test.jsonl (downloaded if absent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "work" / "gsm8k_100.jsonl",
        help="local benchmark subset to create",
    )
    args = parser.parse_args()

    rows = build(obtain_source(args.source))
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )
    digest = sha256_bytes(payload)
    if digest != SUBSET_SHA256:
        raise SystemExit(f"rebuilt subset has unexpected SHA-256 {digest}")
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.output.parent.stat().st_mode & 0o077:
        raise SystemExit(f"benchmark work directory must be owner-only (0700): {args.output.parent}")
    if args.output.exists() or args.output.is_symlink():
        if args.output.is_file() and sha256_bytes(args.output.read_bytes()) == SUBSET_SHA256:
            print(f"verified existing {len(rows)}-row subset at {args.output}")
            print(f"subset SHA-256: {digest}")
            return 0
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    with args.output.open("xb") as handle:
        handle.write(payload)
    os.chmod(args.output, 0o600)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"subset SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
