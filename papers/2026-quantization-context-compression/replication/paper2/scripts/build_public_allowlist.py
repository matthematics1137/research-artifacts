#!/usr/bin/env python3
"""Build Paper 2's promotion-bound, exact public-file inventory.

This is a packaging step, not publication. It reads only already-promoted
aggregate claims, canonical public sources/figures, and explicit staged
``publication/papers/.../public`` roots. It never copies or publishes files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
OUTPUT = PAPER_DIR / "derived" / "public_release_allowlist.json"
PROMOTION = ROOT / "testsuite" / "paper2" / "r1" / "private" / "promotion.json"
FIGURES = (
    "figures/fig_grid.pdf", "figures/fig_ratio.pdf",
    "figures/fig_accmin.pdf", "figures/fig_survival.pdf",
    "figures/dark/fig_grid.pdf", "figures/dark/fig_ratio.pdf",
    "figures/dark/fig_accmin.pdf", "figures/dark/fig_survival.pdf",
)
CANONICAL_INPUTS = (
    "derived/claims.json", "derived/claims.tex", "main.tex", "explainer.tex",
    "references.bib", "VERIFICATION.md", "BIBLIOGRAPHY_AUDIT.md", *FIGURES,
    "review/light/main.pdf", "review/light/explainer.pdf",
    "review/dark/main_dark_review.pdf",
    "review/dark/explainer_dark_review.pdf",
)
ALLOWED_PUBLIC_SUFFIXES = {
    ".bib", ".cff", ".cls", ".csv", ".json", ".lock", ".md", ".pdf",
    ".py", ".sh", ".sty", ".svg", ".tex", ".toml", ".txt", ".yaml",
    ".yml",
}
ALLOWED_EXTENSIONLESS_BASENAMES = {"LICENSE", "Makefile", "paper-Makefile"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_release_root(repo_root: Path, relative: str, *, require_exists: bool) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("release root must be a repository-relative path")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to((repo_root / "publication" / "papers").resolve())
    except ValueError as exc:
        raise ValueError("release root must be beneath publication/papers") from exc
    if resolved.name != "public":
        raise ValueError("release root must be a dedicated directory named public")
    if require_exists and not resolved.is_dir():
        raise ValueError(f"release root is missing: {resolved}")
    return resolved


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def build(release_root_args: list[str]) -> dict:
    gate_spec = importlib.util.spec_from_file_location(
        "paper2_release_gate", PAPER_DIR / "scripts" / "check_release_state.py"
    )
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    # Validates the exact promotion, raw-evidence retention, claims binding, and
    # aggregate-only schema without requiring an allowlist that does not exist yet.
    gate.canonical_gate(pre_figure=True)

    release_roots = sorted(set(release_root_args))
    if len(release_roots) != len(release_root_args) or not release_roots:
        raise SystemExit("STOP: provide one or more distinct --release-root values")
    resolved_roots = [
        resolve_release_root(ROOT, relative, require_exists=True)
        for relative in release_roots
    ]

    files = {(PAPER_DIR / relative).resolve() for relative in CANONICAL_INPUTS}
    for path in files:
        if not path.is_file():
            raise SystemExit(f"STOP: canonical public input is missing: {path}")
    for release_root in resolved_roots:
        for path in release_root.rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"STOP: symlink forbidden in public tree: {path}")
            if path.is_file():
                if (
                    path.suffix.casefold() not in ALLOWED_PUBLIC_SUFFIXES
                    and path.name not in ALLOWED_EXTENSIONLESS_BASENAMES
                ):
                    raise SystemExit(
                        f"STOP: unaudited file type in public tree: {path}"
                    )
                files.add(path.resolve())

    entries = [
        {
            "path": str(path.relative_to(ROOT.resolve())),
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    return {
        "schema": "paper2-r1-public-release-allowlist-v2",
        "promotion_sha256": sha256_file(PROMOTION),
        "release_roots": release_roots,
        "files": entries,
    }


def self_test() -> None:
    fake = Path("/tmp/p2r1-public-builder-test")
    assert resolve_release_root(
        fake, "publication/papers/paper2-test/public", require_exists=False
    ) == fake / "publication/papers/paper2-test/public"
    for invalid in ("paper2/public", "publication/papers/paper2-test/private", "/tmp/public"):
        try:
            resolve_release_root(fake, invalid, require_exists=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe release root accepted: {invalid}")
    assert {"LICENSE", "Makefile", "paper-Makefile"} <= ALLOWED_EXTENSIONLESS_BASENAMES
    assert "weights" not in ALLOWED_EXTENSIONLESS_BASENAMES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", action="append", default=[])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("Paper 2 public-allowlist builder self-test passed")
        return 0
    content = json.dumps(build(args.release_root), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != content:
            raise SystemExit("STOP: public release allowlist is missing or stale")
        print("Paper 2 public release allowlist is current")
        return 0
    atomic_write(OUTPUT, content)
    print(f"wrote {OUTPUT}; no files were copied or published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
