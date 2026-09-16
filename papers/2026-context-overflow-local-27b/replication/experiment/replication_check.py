#!/usr/bin/env python3
"""Inspect or hash explicitly located replication assets; never download or run models.

The default is a size/existence PLAN, not content verification. Full --verify
may read tens of GB: check live candidate-machine availability before using it.
This verifies asset bytes, not experiment execution, licenses or author approval.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

SCHEMA = "paper-replication-assets-v1"
ROOTS = ("repo", "models", "hf_cache")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
            "invalid asset path")
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and str(path) == value and ".." not in path.parts,
            "asset path is not a normalized relative path")
    return path


def digest(value):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "invalid SHA-256")
    return value


def validate(catalog):
    require(catalog["schema"] == SCHEMA and catalog["paper"] in ("paper5", "paper6") and
            catalog["status"] == "prospective-asset-inventory-not-a-release" and
            catalog["author_approval"] is False, "wrong asset catalog identity")
    require(catalog["assets"] and isinstance(catalog["assets"], list), "empty asset inventory")
    ids, locations = set(), set()
    for row in catalog["assets"]:
        require(re.fullmatch("[a-z0-9][a-z0-9_-]*", row["id"]), "invalid asset identifier")
        require(row["id"] not in ids, "duplicate asset identifier")
        ids.add(row["id"])
        require(row["root"] in ROOTS, "unknown asset root")
        relative(row["path"])
        location = (row["root"], row["path"])
        require(location not in locations, "duplicate asset location")
        locations.add(location)
        require(type(row["bytes"]) is int and row["bytes"] > 0, "invalid asset size")
        digest(row["sha256"])
        require(row["verification"] == "identity-inventory-not-an-asset-check",
                "an identity inventory cannot attest a separate asset check")
        require(isinstance(row["evidence"], str) and row["evidence"], "missing identity evidence")
        upstream = row.get("upstream")
        if upstream is not None:
            require(upstream["service"] in ("huggingface", "github"), "unknown upstream service")
            require(upstream["kind"] in ("model", "dataset", "repository"), "unknown upstream kind")
            require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", upstream["repo"]),
                    "invalid upstream repository")
            require(re.fullmatch("[0-9a-f]{40}", upstream["revision"]), "upstream revision is not immutable")
            relative(upstream["file"])
    return catalog


def locate(row, roots):
    require(row["root"] in roots, "missing root: " + row["root"])
    root = Path(roots[row["root"]]).resolve(strict=True)
    require(root.is_dir(), "asset root is not a directory")
    path = root / relative(row["path"])
    # HF snapshots intentionally link to blobs in the same explicitly chosen
    # cache. Never follow a link outside that root, including an intermediate one.
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root), "asset symlink escapes its selected root")
    require(stat.S_ISREG(resolved.stat().st_mode), "asset is not a regular file")
    return path


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def inspect(catalog, roots, *, verify=False, asset_ids=()):
    validate(catalog)
    require(set(asset_ids) <= {r["id"] for r in catalog["assets"]}, "unknown requested asset")
    rows = [r for r in catalog["assets"] if not asset_ids or r["id"] in asset_ids]
    results = []
    for row in rows:
        result = {"id": row["id"], "bytes_expected": row["bytes"], "content_verified": False}
        try:
            path = locate(row, roots)
            target = path.resolve(strict=True)
            before = path.stat()
            require(before.st_size == row["bytes"], "asset size mismatch")
            result["size_matches"] = True
            if verify:
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    require(identity(os.fstat(stream.fileno())) == identity(before),
                            "asset changed before opening")
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(block)
                require(identity(path.stat()) == identity(before) and
                        locate(row, roots).resolve(strict=True) == target,
                        "asset changed during verification")
                require(h.hexdigest() == row["sha256"], "asset SHA-256 mismatch")
                result["content_verified"] = True
            result["ok"] = True
        except (OSError, RuntimeError, ValueError) as exc:
            # Do not project home paths from exception messages into a shareable report.
            result.update(ok=False, error=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        results.append(result)
    return {"schema": "replication-asset-check-v1", "paper": catalog["paper"],
            "mode": "content-verification" if verify else "existence-and-size-plan-only",
            "assets": results, "bytes_selected": sum(r["bytes"] for r in rows),
            "ok": all(r["ok"] for r in results), "author_approval": False}


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("non-finite JSON value")

    def floating(value):
        number = float(value)
        return number if math.isfinite(number) else invalid(value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid, parse_float=floating)


def read_json(path):
    return parse_json(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    for root in ROOTS:
        parser.add_argument("--" + root.replace("_", "-") + "-root", dest=root, type=Path)
    parser.add_argument("--asset", action="append", default=[], help="restrict to this asset ID (repeatable)")
    parser.add_argument("--verify", action="store_true", help="hash asset contents; check machine availability first")
    args = parser.parse_args()
    try:
        catalog = read_json(args.catalog)
        report = inspect(catalog, {r: getattr(args, r) for r in ROOTS if getattr(args, r)},
                         verify=args.verify, asset_ids=args.asset)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["ok"] else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "Asset check refused: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
