#!/usr/bin/env python3
"""Seal the CPU candidate's runtime inputs; print the digest for queue-step text.

Sealing is reproducible packaging, not approval of a registration or GPU launch.
"""
import json
from pathlib import Path
from run_refresh import digest

HERE = Path(__file__).resolve().parent


def seal():
    names = [p.name for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".json") and p.name != "freeze.json"]
    names += ["../confirm/" + p for p in ("run_confirm.py", "run_confirm_fit.py", "run_confirm_df2.py",
              "run_confirm_df2b.py", "plan.json", "plan-fit.json", "plan-df2.json", "plan-df2b.json")]
    names += ["../census.py", "../smoke.py"]
    names += ["../patches/" + p.name for p in (HERE.parent / "patches").glob("*.patch") if "w12c" not in p.name]
    out = HERE / "freeze.json"
    data = dict(schema="p8-refresh-freeze-v1", files={p: digest(HERE / p) for p in sorted(names)})
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(digest(out))


if __name__ == "__main__":
    seal()
