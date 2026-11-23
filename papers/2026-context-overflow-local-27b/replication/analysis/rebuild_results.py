#!/usr/bin/env python3
"""Rebuild P5 results TeX and light/dark figures from its checked data bundle.

Copied as rebuild_results.py. No model, network, or original repo is required.
This does not compile or approve the full manuscript. Outputs are create-only.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType


def load_checker(path):
    module = ModuleType("p5_portable_checker")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def rebuild(bundle, output):
    bundle, output = Path(bundle), Path(output)
    if any(p.is_symlink() for p in (bundle, *bundle.parents, output, *output.parents)):
        raise ValueError("bundle/output path contains a symlink")
    bundle, output = bundle.resolve(), output.resolve()
    if output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("render output must be separate from the checked bundle")
    checker = load_checker(bundle / "check_results.py")
    manifest_sha = checker.sha(bundle / "MANIFEST.json")
    verification = checker.check(bundle)
    manifest = checker.read_json(bundle / "MANIFEST.json")
    data = checker.read_json(bundle / "observations.json")
    results = checker.read_json(bundle / "results.json")
    sources = {file: (bundle / file).read_text() for file in checker.TEXT_MODULES.values()}
    claims = checker.claims_text(data, results, sources).encode()
    checker.require(claims == (bundle / "claims.tex").read_bytes(), "manuscript macros changed")
    files = {"derived/claims.tex": claims}
    previous_cache = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="p5-portable-render-") as tmp:
        stage = Path(tmp)
        os.environ["MPLCONFIGDIR"] = str(stage / "font-cache")
        try:
            import matplotlib
            import numpy
            # Restore the caller's plotting defaults and module namespace after
            # drawing. Each build starts with the same fresh-process base style.
            with matplotlib.rc_context(rc=matplotlib.rcParamsDefault), checker.text_modules(sources), redirect_stdout(io.StringIO()):
                figures = checker.load_source(bundle / "figure_core.py")
                for dark in (False, True):
                    directory = stage / ("dark" if dark else "light")
                    figures.render_figures(results["primary"]["analysis"], directory, dark)
                    for name in ("fig_arms.pdf", "fig_evidence.pdf", "fig_cost.pdf"):
                        relative = ("figures/dark/" if dark else "figures/") + name
                        files[relative] = (directory / name).read_bytes()
            versions = {"python": sys.version.split()[0], "numpy": numpy.__version__, "matplotlib": matplotlib.__version__}
        finally:
            if previous_cache is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_cache
    checker.require(manifest_sha == checker.sha(bundle / "MANIFEST.json") and
                    all(checker.sha(bundle / name) == value for name, value in manifest["files"].items()),
                    "bundle changed during rendering")
    receipt = {"schema": "p5-analysis-render-v1", "bundle_manifest_sha256": manifest_sha, "verification": verification,
               "dependencies": versions, "outputs": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
               "author_approval": False, "full_manuscript_pdf": False}
    files["BUILD.json"] = (checker.canonical(receipt) + "\n").encode()
    if output.exists():
        found = {p.relative_to(output).as_posix() for p in output.rglob("*")}
        checker.require(found == set(files) | {"derived", "figures", "figures/dark"} and
                        all(not p.is_symlink() for p in output.rglob("*")) and
                        all((output / name).is_file() and (output / name).read_bytes() == value for name, value in files.items()),
                        "existing render differs or contains unrelated files; choose a new output directory")
        return receipt
    output.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(rebuild(args.bundle, args.output), indent=2))
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(2, "P5 results rendering refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
