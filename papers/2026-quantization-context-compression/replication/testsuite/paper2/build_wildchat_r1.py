#!/usr/bin/env python3
"""Build P2R1 WildChat windows from an immutable, hash-checked Parquet shard.

The old datasets-server endpoint ignored a requested revision. This builder
downloads the exact Parquet object at the frozen WildChat commit, checks its
size and SHA-256, and records row/segment provenance in a private manifest.
Raw text and upstream conversation hashes must never enter the public release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ladder  # noqa: E402
from identity_guard import sha256_file, utc_now  # noqa: E402
from r1_safety import RunSafetyError  # noqa: E402


REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
REPOSITORY = "allenai/WildChat-1M"
SHARDS = {
    "data/train-00000-of-00014.parquet": (230786095, "abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c"),
    "data/train-00001-of-00014.parquet": (215399142, "f10fa35bb52703baad5e3177d3382d63b855fce9e9d1b9ab1fe2f0e762464f2a"),
    "data/train-00002-of-00014.parquet": (205701692, "9a4912ec14c1012970f57b6710518dda17dbf6cdfb9b6e694880f5456b0b031f"),
    "data/train-00003-of-00014.parquet": (216891660, "3f2d50ee227e32848f6c4fdb0f8ea543b379b2219bec72be9446e232d3d2a670"),
    "data/train-00004-of-00014.parquet": (208292393, "08c7e0c31edc120606d67cfa82486471d82945a7a414761501ea44b1ab0cd804"),
    "data/train-00005-of-00014.parquet": (200855598, "1caab40d8d4f01d76d8e1fa03e80f31b2f4d8b8b4c47cb3aa0280f4467e04555"),
    "data/train-00006-of-00014.parquet": (189507683, "0bd2fff31c74feb6b44b0397b4e7ab6285dd69ccbe4cc8b5f19c42cdbf691303"),
    "data/train-00007-of-00014.parquet": (188410403, "f498ea6771be5bf00d259ec76bd9fba51ceaa99c76147a0bb95fd7fa1769b984"),
    "data/train-00008-of-00014.parquet": (180994027, "a69578183dde5de1aa94134da2d7c2d13f00d88a577ef91140a492fbbbf10792"),
    "data/train-00009-of-00014.parquet": (268963848, "d2eb8868323274c60c9fd5437eaf2159a08ac278598b1197bc20ac04c633648a"),
    "data/train-00010-of-00014.parquet": (336477988, "328b0d702d91ffd47e51aff2de56d74d0f2b874a31f2530f1948474a582f0467"),
    "data/train-00011-of-00014.parquet": (299818606, "a458b915aeab7a4a81bd18770f33c3ebb908c91e67d916db5525819c88dbe778"),
    "data/train-00012-of-00014.parquet": (282697282, "5353f1a829ac86de165c2fb537a30d7d756259dbe37a8a0fa6ec791321a37d6e"),
    "data/train-00013-of-00014.parquet": (336039603, "440be579bf012a6e98d179b8dd0bedd73634e5527f7000c067d19c88b5cefa6c"),
}
PRIMARY_SHARD = "data/train-00000-of-00014.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=40)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=HERE / "r1" / "private_inputs" / "wildchat" / "upstream",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "r1" / "private_inputs" / "wildchat" / "windows_base.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "r1" / "private_inputs" / "wildchat" / "source_manifest.json",
    )
    return parser.parse_args()


def download_pinned_shard(source_dir: Path) -> Path:
    rel = PRIMARY_SHARD
    expected_size, expected_hash = SHARDS[rel]
    target = source_dir / Path(rel).name
    source_dir.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{rel}"
        temporary = target.with_name(target.name + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "p2r1-wildchat/1"})
        temp_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with urllib.request.urlopen(request, timeout=120) as response, os.fdopen(temp_fd, "wb") as out:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
    target.chmod(0o600)
    if target.stat().st_size != expected_size:
        raise RunSafetyError(f"WildChat shard size mismatch: {target}")
    actual_hash = sha256_file(target)
    if actual_hash != expected_hash:
        raise RunSafetyError(
            f"WildChat shard hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return target


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.exists():
        raise RunSafetyError(f"refusing to overwrite frozen R1 input: {path}")
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.exists():
        raise RunSafetyError(f"refusing to overwrite frozen R1 manifest: {path}")
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    args = parse_args()
    private_root = (HERE / "r1" / "private_inputs").resolve()
    # The manifest contains upstream conversation hashes and the window file
    # contains raw conversation text. The builder refuses a misleading
    # public-named destination even when explicitly requested.
    for path in (args.source_dir, args.out, args.manifest):
        try:
            path.resolve().relative_to(private_root)
        except ValueError as exc:
            raise RunSafetyError(
                f"raw WildChat R1 material must stay below {private_root}: {path}"
            ) from exc
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RunSafetyError(
            "pyarrow is required; install the version pinned in requirements-r1.txt"
        ) from exc
    if importlib.metadata.version("pyarrow") != "21.0.0":
        raise RunSafetyError("pinned WildChat build requires pyarrow==21.0.0")

    shard = download_pinned_shard(args.source_dir)
    parquet_file = parquet.ParquetFile(shard)
    records: list[tuple[int, dict]] = []
    row_index = 0
    columns = ["language", "conversation", "conversation_hash"]
    for batch in parquet_file.iter_batches(batch_size=256, columns=columns):
        for record in batch.to_pylist():
            if row_index >= args.rows:
                break
            records.append((row_index, record))
            row_index += 1
        if row_index >= args.rows:
            break
    if len(records) != args.rows:
        raise RunSafetyError(f"requested {args.rows} rows; pinned shard yielded {len(records)}")

    targets = {"S": 4000, "L": 12000}
    candidates = []
    for source_row, record in records:
        if record.get("language") != "English":
            continue
        conversation = record.get("conversation") or []
        texts = [turn.get("content") or "" for turn in conversation]
        if len(texts) < 3:
            continue
        for stratum, budget in targets.items():
            start = 0
            while start < len(texts):
                end, chars = start, 0
                while end < len(texts) and chars < budget:
                    chars += len(texts[end])
                    end += 1
                segment = texts[start:end]
                if len(segment) >= 3 and chars >= budget * 0.6:
                    original = "\n\n".join(segment).strip()
                    rung_e = ladder.rung_E(original)
                    if original and len(rung_e) >= 200:
                        candidates.append({
                            "_source_row": source_row,
                            "_segment_start": start,
                            "_segment_end": end,
                            "_conversation_hash": record.get("conversation_hash"),
                            "n_msgs": len(segment),
                            "stratum": stratum,
                            "has_code": int("```" in original),
                            "text_O": original,
                            "text_E": rung_e,
                            "chars_O": len(original),
                            "chars_E": len(rung_e),
                        })
                start = end

    random.Random(args.seed).shuffle(candidates)
    selected = []
    for stratum in targets:
        for has_code in (0, 1):
            cell = [
                row for row in candidates
                if row["stratum"] == stratum and row["has_code"] == has_code
            ]
            if len(cell) < args.per_cell:
                raise RunSafetyError(
                    f"insufficient pinned WildChat windows for {stratum}/code={has_code}: "
                    f"{len(cell)} < {args.per_cell}"
                )
            selected.extend(cell[: args.per_cell])

    cluster_sources = sorted({
        str(row["_conversation_hash"] or f"row:{row['_source_row']}")
        for row in selected
    })
    cluster_ids = {
        source: f"cwc{index:05d}" for index, source in enumerate(cluster_sources)
    }
    output_rows = []
    provenance = []
    for index, row in enumerate(selected):
        window_id = f"wc{index:04d}"
        cluster_source = str(row["_conversation_hash"] or f"row:{row['_source_row']}")
        source_cluster_id = cluster_ids[cluster_source]
        output_rows.append({
            "window_id": window_id,
            "source_cluster_id": source_cluster_id,
            "n_msgs": row["n_msgs"],
            "stratum": row["stratum"],
            "has_code": row["has_code"],
            "text_O": row["text_O"],
            "text_E": row["text_E"],
            "chars_O": row["chars_O"],
            "chars_E": row["chars_E"],
        })
        provenance.append({
            "window_id": window_id,
            "source_cluster_id": source_cluster_id,
            "shard": PRIMARY_SHARD,
            "source_row_index": row["_source_row"],
            "segment_start_turn": row["_segment_start"],
            "segment_end_turn_exclusive": row["_segment_end"],
            "upstream_conversation_hash": row["_conversation_hash"],
            "text_O_sha256": hashlib.sha256(row["text_O"].encode()).hexdigest(),
        })

    ids = [row["window_id"] for row in output_rows]
    if len(ids) != len(set(ids)):
        raise RunSafetyError("generated WildChat window IDs are not unique")

    atomic_jsonl(args.out, output_rows)
    manifest = {
        "schema": "paper2-r1-wildchat-source-v1",
        "corpus": "WC",
        "created_at": utc_now(),
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "ODC-By-1.0 database license; independent content/privacy rights not asserted",
        "shard": {
            "path": PRIMARY_SHARD,
            "bytes": SHARDS[PRIMARY_SHARD][0],
            "sha256": SHARDS[PRIMARY_SHARD][1],
        },
        "all_snapshot_shards": [
            {"path": path, "bytes": values[0], "sha256": values[1]}
            for path, values in sorted(SHARDS.items())
        ],
        "rows_scanned": args.rows,
        "selection_seed": args.seed,
        "per_cell": args.per_cell,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "ladder_source_sha256": sha256_file(HERE / "ladder.py"),
        "pyarrow_version": "21.0.0",
        "requirements_sha256": sha256_file(HERE / "requirements-r1.txt"),
        "windows": len(output_rows),
        "output_sha256": sha256_file(args.out),
        "private_window_provenance": provenance,
        "release_boundary": (
            "Do not publish raw text, questions, golds, upstream conversation hashes, "
            "this private provenance mapping, deterministic local IDs, or per-item "
            "rows. Public derivatives are aggregate-only counts, rates, timing "
            "summaries, and test statistics with WildChat attribution."
        ),
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps({
        "windows": len(output_rows),
        "revision": REVISION,
        "shard_sha256": SHARDS[PRIMARY_SHARD][1],
        "output": str(args.out),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 WILDCHAT BUILD ABORTED: {exc}") from exc
