#!/usr/bin/env python3
"""CPU build/validation entry for Zarael sandbox. Only /out is retained."""
import contextlib
import io
import json
from pathlib import Path
import py_compile
import unittest

import seal
import run_refresh as R
import build_cpu


def main():
    out = Path("/out")
    out.mkdir(exist_ok=True)
    for path in R.HERE.glob("*.py"):
        py_compile.compile(str(path), cfile=str(out / (path.stem + ".pyc")), doraise=True)
    with contextlib.redirect_stdout(io.StringIO()):
        seal.seal()
    sha = R.digest(R.HERE / "freeze.json")
    R.check_freeze(R.HERE, sha)
    (out / "freeze.json").write_bytes((R.HERE / "freeze.json").read_bytes())
    plan, phases = R.load()
    with (out / "dry-run.jsonl").open("w") as f, contextlib.redirect_stdout(f):
        R.render(plan, phases)
    (out / "build-recipes.json").write_text(json.dumps(build_cpu.commands(plan,R.RC.L),indent=2)+"\n")
    with (out / "unittest.txt").open("w") as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(R.HERE),pattern="test_refresh.py"))
    report = dict(cpu_only=True, tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  freeze_sha256=sha, engine_binaries_built=False, gpu_touched=False,
                  note="Synthetic fixtures verify controls; no model/server/GPU execution or speed result.")
    (out / "validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
