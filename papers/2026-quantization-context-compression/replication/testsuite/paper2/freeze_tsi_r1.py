#!/usr/bin/env python3
"""Freeze TSI source windows and recompute E with the pinned R1 ladder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from identity_guard import sha256_file, utc_now  # noqa: E402
from r1_safety import RunSafetyError  # noqa: E402
import ladder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "private" / "windows.jsonl")
    parser.add_argument(
        "--out", type=Path, default=HERE / "r1" / "private_inputs" / "tsi" / "windows_base.jsonl"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "r1" / "private_inputs" / "tsi" / "source_manifest.json",
    )
    args = parser.parse_args()
    private_root = (HERE / "r1" / "private_inputs" / "tsi").resolve()
    for path in (args.out, args.manifest):
        try:
            path.resolve().relative_to(private_root)
        except ValueError as exc:
            raise RunSafetyError(f"TSI R1 inputs must stay below {private_root}") from exc
    if args.out.exists() or args.manifest.exists():
        raise RunSafetyError("refusing to overwrite a frozen TSI R1 input")

    with args.source.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 126:
        raise RunSafetyError(f"expected 126 frozen TSI windows; found {len(rows)}")
    ids = [row["window_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RunSafetyError("TSI window IDs are not unique")
    for row in rows:
        if not row.get("text_O"):
            raise RunSafetyError(f"{row['window_id']} lacks original text")
        recomputed_e = ladder.rung_E(row["text_O"])
        if not recomputed_e or len(recomputed_e) < 200:
            raise RunSafetyError(
                f"{row['window_id']} pinned ladder produced an ineligible E condition"
            )

    cluster_sources = sorted({str(row["conversation_id"]) for row in rows})
    cluster_ids = {
        source: f"ctsi{index:05d}" for index, source in enumerate(cluster_sources)
    }
    frozen_rows = []
    legacy_e_exact_matches = 0
    for row in rows:
        recomputed_e = ladder.rung_E(row["text_O"])
        legacy_e_exact_matches += int(row.get("text_E") == recomputed_e)
        frozen = {
            key: value for key, value in row.items()
            if key not in {
                "conversation_id",
                "text_A", "text_L2", "text_E",
                "chars_A", "chars_L2", "chars_E",
            }
        }
        frozen["text_E"] = recomputed_e
        frozen["chars_E"] = len(frozen["text_E"])
        frozen["source_cluster_id"] = cluster_ids[str(row["conversation_id"])]
        frozen_rows.append(frozen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.chmod(0o700)
    out_temp = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    out_fd = os.open(out_temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(out_fd, "w") as target:
        for row in frozen_rows:
            target.write(json.dumps(row, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(out_temp, args.out)
    strata = Counter(
        f"{row['stratum']}-{'code' if row['has_code'] else 'nocode'}" for row in rows
    )
    manifest = {
        "schema": "paper2-r1-tsi-source-v1",
        "corpus": "TSI",
        "created_at": utc_now(),
        "source_sha256": sha256_file(args.source),
        "builder_source_sha256": sha256_file(Path(__file__)),
        "output_sha256": sha256_file(args.out),
        "windows": len(rows),
        "opaque_source_clusters": len(cluster_sources),
        "strata_counts": dict(sorted(strata.items())),
        "condition_keys": ["O", "E"],
        "ladder_source_sha256": sha256_file(HERE / "ladder.py"),
        "e_recomputed_from_text_o": True,
        "legacy_derived_condition_fields_discarded": ["A", "L2", "E"],
        "legacy_e_exact_match_count": legacy_e_exact_matches,
        "legacy_e_mismatch_count": len(rows) - legacy_e_exact_matches,
        "condition_note": "E is recomputed from frozen text_O by the pinned ladder; A and L2 are recomputed in the next stage. Legacy derived text is not treated as R1 evidence.",
        "release_boundary": "All TSI text, questions, golds, responses, and per-item rows remain private. Release only non-content hashes/counts/schema and aggregates."
    }
    manifest_temp = args.manifest.with_name(args.manifest.name + f".tmp.{os.getpid()}")
    manifest_fd = os.open(
        manifest_temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(manifest_fd, "w") as target:
        target.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(manifest_temp, args.manifest)
    print(json.dumps({"windows": len(rows), "output_sha256": manifest["output_sha256"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunSafetyError as exc:
        raise SystemExit(f"P2R1 TSI SOURCE FREEZE ABORTED: {exc}") from exc
