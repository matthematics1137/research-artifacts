#!/usr/bin/env python3
"""Recreate P5's EXL3 overlay bytes in a NEW directory; never install or run it.

Check live machine availability first: the wheel expands to about 1.3 GB.
The withdrawn quantizer workaround is retained for historical byte fidelity,
not recommended as a bug fix or credited as an inference optimization.
Dependencies, loader behavior, hardware and model execution are separate gates.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

import verify_p5_runtime_wheel as W

PATCHES = (
    {"patch": "exl3_all_lab_patches.patch", "patch_sha256": "ba359262a550cfb937e3d590030ebaa0816066535f3b61844d820b171e746a00",
     "path": "exllamav3/generator/generator.py", "sections": 1,
     "before": "d9679f40791c091382e00e5278d815a19fa7856b9788c376b2ae7093d1cf8231",
     "after": "f0d7bee9e713b5b755e199babdc8fe8fa4e6c0ad116d072cef679ef8c353148e",
     "scope": "combined generator changes; P5 keeps experimental measurement/sampled-draft switches off"},
    {"patch": "exl3_spec_sampled.patch", "patch_sha256": "c2308050f5a6fd2778a10d5886d5d8478eddc522c6841ba24514067d4382f58c",
     "path": "exllamav3/architecture/qwen3_5_mtp.py", "sections": 2,
     "before": "91a71f29dd7c80336527a7cac6aa566e7b7b97ce9e38f0afe3469eeeff51d829",
     "after": "8f619e54c0423c9174ed5c75869db49d2334c181e2e0362a789f64b4fa6000a2",
     "scope": "architecture section only; generator section is already in the combined patch"},
    {"patch": "withdrawn/exl3_cholesky_cpu_fallback.patch", "patch_sha256": "30aef35c2ce5a77dff74bc2fd04852253292797e77ecc0f4b7105632bcab4ca9",
     "path": "exllamav3/modules/quant/exl3_lib/quantize.py", "sections": 1,
     "before": "b79f0e5ffae1e9d2316c440aa58666fbc7a359bd5ec7afe2b79945c3a7bcd2c4",
     "after": "791787889f755b1d7108b133d35c0accf3bf8eca0792f245296a9998014403c4",
     "scope": "inherited withdrawn conversion workaround; historical fidelity, not an inference optimization"},
)
MANIFEST = "OVERLAY_MANIFEST.json"
SCHEMA = "paper5-exl3-overlay-byte-reconstruction-v1"


def apply_section(original, patch_bytes, spec):
    """Apply one hash-pinned unified section, with exact offsets and context.

    Patch headers are labels only, never filesystem destinations. There is no
    shell, patch subprocess, offset search, fuzz, reversal or inference import.
    """
    W.require(W.sha(patch_bytes) == spec["patch_sha256"], "patch input hash differs")
    W.require(W.sha(original) == spec["before"], "unpatched Python input differs")
    lines = patch_bytes.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(b"--- ")]
    W.require(len(starts) == spec["sections"] and starts[0] == 0, "patch section inventory differs")
    selected = []
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        W.require(lines[start + 1].startswith(b"+++ "), "missing new-file patch header")
        label = lines[start + 1][4:].split(b"\t", 1)[0].decode().rstrip("\n")
        if label.endswith("/" + spec["path"]):
            selected.append(lines[start + 2:end])
    W.require(len(selected) == 1, "patch target is missing or ambiguous")
    source, output, cursor = original.splitlines(keepends=True), [], 0
    hunk_lines, index, hunks = selected[0], 0, 0
    while index < len(hunk_lines):
        if hunk_lines[index] == b"\n":
            index += 1  # separator between file sections, not a context line
            continue
        header = re.fullmatch(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", hunk_lines[index])
        W.require(header is not None, "unexpected unified-diff content")
        old_start, old_count, new_start, new_count = (int(header[1]), int(header[2] or b"1"),
                                                     int(header[3]), int(header[4] or b"1"))
        at = old_start - (old_count != 0)
        W.require(cursor <= at <= len(source), "patch hunk offset differs")
        output.extend(source[cursor:at])
        cursor, old_seen, new_seen = at, 0, 0
        W.require(len(output) == new_start - (new_count != 0), "patch output offset differs")
        index, hunks = index + 1, hunks + 1
        while old_seen < old_count or new_seen < new_count:
            W.require(index < len(hunk_lines), "truncated patch hunk")
            line = hunk_lines[index]
            W.require(line[:1] in (b" ", b"+", b"-"), "unsupported patch hunk line")
            if line[:1] in (b" ", b"-"):
                W.require(cursor < len(source) and source[cursor] == line[1:], "patch context differs")
                cursor, old_seen = cursor + 1, old_seen + 1
            if line[:1] in (b" ", b"+"):
                output.append(line[1:])
                new_seen += 1
            W.require(old_seen <= old_count and new_seen <= new_count, "patch hunk counts differ")
            index += 1
    W.require(hunks > 0, "patch contains no hunks")
    payload = b"".join(output + source[cursor:])
    W.require(W.sha(payload) == spec["after"], "patched Python output differs from the P5 observation")
    ast.parse(payload, filename=spec["path"])
    return payload


def checked_record(record):
    core = dict(record)
    core["receipt_sha256"] = W.sha(W.canonical(core).encode())
    return core


def prepare(wheel, patches, output):
    wheel, patches, output = map(W.safe_path, (wheel, patches, output))
    W.require(not output.exists() and output.parent.is_dir(), "overlay needs a new directory in an existing parent")
    W.require(patches.is_dir() and not output.is_relative_to(patches) and not patches.is_relative_to(output)
              and not wheel.is_relative_to(output), "overlay/input paths overlap")
    upstream = W.verify(wheel)  # hashes every archive member; never imports it
    patched = {}
    with zipfile.ZipFile(wheel) as archive:
        for spec in PATCHES:
            path = W.safe_path(patches / spec["patch"])
            W.require(path.is_file() and path.stat().st_size <= 128 * 1024, "missing/oversize patch")
            W.require(archive.getinfo(spec["path"]).file_size <= 1024 * 1024, "oversize Python patch target")
            patched[spec["path"]] = apply_section(archive.read(spec["path"]), path.read_bytes(), spec)
        output.mkdir(mode=0o700)  # exclusive; failed attempts are retained, never silently reused
        records = []
        for row in upstream["members"]:
            path = output / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            h, count = hashlib.sha256(), 0
            with path.open("xb") as destination:
                if row["path"] in patched:
                    data = patched[row["path"]]
                    destination.write(data)
                    h.update(data)
                    count = len(data)
                else:
                    with archive.open(row["path"]) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            count += len(block)
                            W.require(count <= row["bytes"], "copy exceeded verified member size")
                            destination.write(block)
                            h.update(block)
                    W.require(count == row["bytes"] and h.hexdigest() == row["sha256"], "copied wheel member changed")
            records.append({"path": row["path"], "bytes": count, "sha256": h.hexdigest()})
    record = checked_record({"schema": SCHEMA, "status": "reconstructed-bytes-not-runtime-execution",
                             "checked_at": datetime.now(timezone.utc).isoformat(), "author_approval": False,
                             "wheel": upstream["wheel"], "upstream_member_inventory_sha256": upstream["member_inventory_sha256"],
                             "tools": {"prepare_p5_runtime_overlay.py": W.sha(Path(__file__).read_bytes()),
                                       "verify_p5_runtime_wheel.py": W.sha(Path(W.__file__).read_bytes())},
                             "patches": list(PATCHES), "files": records,
                             "package_installed": False, "native_loaded": False,
                             "not_established": ["dependency closure or native import/model execution",
                                                 "continuous identity of the historical runtime or prior model conversion",
                                                 "new inference benefit from inherited/disabled/withdrawn patches"],
                             "wheel_record_note": "Original upstream RECORD is preserved as historical input; it does not describe the three patched files. Use this manifest for overlay hashes."})
    with (output / MANIFEST).open("x") as stream:
        stream.write(W.canonical(record))
    return record


def verify_overlay(output):
    output = W.safe_path(output)
    path = W.safe_path(output / MANIFEST)
    W.require(path.is_file() and path.stat().st_size <= 1024 * 1024, "missing/oversize overlay manifest")
    record = json.loads(path.read_text())
    claim = record.pop("receipt_sha256")
    W.require(W.sha(W.canonical(record).encode()) == claim, "overlay manifest checksum differs")
    W.require(record["schema"] == SCHEMA and record["status"] == "reconstructed-bytes-not-runtime-execution"
              and record["author_approval"] is False and record["package_installed"] is False
              and record["native_loaded"] is False, "overlay receipt boundary differs")
    W.require(record["wheel"] == {"name": W.NAME, "bytes": W.SPEC["bytes"], "sha256": W.SPEC["sha256"], "download_url": W.URL}
              and record["patches"] == list(PATCHES), "overlay input pins differ")
    W.require(record["tools"] == {"prepare_p5_runtime_overlay.py": W.sha(Path(__file__).read_bytes()),
                                   "verify_p5_runtime_wheel.py": W.sha(Path(W.__file__).read_bytes())}, "overlay verifier source differs")
    rows = record["files"]
    W.require(0 < len(rows) <= W.MAX_FILES and sum(r["bytes"] for r in rows) <= W.MAX_EXPANDED, "overlay exceeds resource bounds")
    names = [r["path"] for r in rows]
    W.require(len(names) == len(set(names)), "duplicate overlay member")
    actual = []
    for entry in output.rglob("*"):
        W.require(not entry.is_symlink(), "overlay contains symlink")
        if not entry.is_dir():
            W.require(entry.is_file(), "overlay contains special file")
            actual.append(entry.relative_to(output).as_posix())
    W.require(set(actual) == set(names) | {MANIFEST}, "overlay file inventory differs")
    lookup = {r["path"]: r for r in rows}
    for spec in PATCHES:
        W.require(lookup[spec["path"]]["sha256"] == spec["after"], "patched overlay identity differs")
    for row in rows:
        path = W.safe_path(output / row["path"])
        W.require(path.is_relative_to(output) and path.stat().st_size == row["bytes"], "overlay path/size differs")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        W.require(h.hexdigest() == row["sha256"], "overlay file hash differs: " + row["path"])
    record["receipt_sha256"] = claim
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--patches", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="rehash a previously created, unused overlay; check live placement first")
    args = parser.parse_args()
    try:
        if args.verify:
            W.require(not any((args.wheel, args.patches, args.output)), "--verify is separate from creation")
            result = verify_overlay(args.verify)
        else:
            W.require(all((args.wheel, args.patches, args.output)), "creation requires --wheel, --patches and --output")
            result = prepare(args.wheel, args.patches, args.output)
        print(W.canonical({"status": result["status"], "files": len(result["files"]),
                           "bytes": sum(r["bytes"] for r in result["files"]), "patches": len(result["patches"]),
                           "receipt_sha256": result["receipt_sha256"], "author_approval": False}), end="")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, SyntaxError) as exc:
        parser.exit(2, "Overlay reconstruction refused (any partial new output is retained): " + str(exc) + "\n")


if __name__ == "__main__":
    main()
