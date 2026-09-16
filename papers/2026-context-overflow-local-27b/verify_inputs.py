#!/usr/bin/env python3
"""Verify assembled paper/results/source inputs, optionally rerendering results.

Copied as verify_inputs.py. No original repository, model or network is needed.
This checks the prepared inputs, not author approval or experimental execution.
Run untrusted artifacts only through the release controller's isolated verifier.
"""
import argparse
import hashlib
import json
import math
from os import environ
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile

SCHEMA = "paper-release-input-component-v1"
PROFILES = {
    "paper5": {"data_schema": "p5-analysis-data-candidate-v1", "render_schema": "p5-analysis-render-v1",
               "figures": ("fig_arms.pdf", "fig_evidence.pdf", "fig_cost.pdf")},
    "paper6": {"data_schema": "p6-analysis-data-candidate-v1", "render_schema": "p6-analysis-render-v1",
               "figures": ("fig_grid.pdf", "fig_damage.pdf", "fig_estimands.pdf")},
}
ANALYSIS = "replication/analysis"
EXPERIMENT = "replication/experiment"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON number")
    def floating(value):
        number = float(value)
        return number if math.isfinite(number) else reject(value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=reject, parse_float=floating)


def relative(name):
    require(type(name) is str and "\\" not in name and "\x00" not in name, "invalid input path")
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and ".." not in path.parts and str(path) == name, "input path escapes its root")
    return name


def tree(root):
    root = Path(root)
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), "invalid input directory")
    result = {}
    for path in root.rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "nonregular input entry")
        if path.is_file():
            require(path.stat().st_size <= 50 * 1024 * 1024, "oversize input file")
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def verify(root, *, staged=False):
    root = Path(root)
    found = tree(root)
    record = read_json(root / "INPUTS.json")
    require(set(record) == {"schema", "paper", "status", "author_approval", "sources", "files"}, "unexpected input manifest fields")
    require(record["schema"] == SCHEMA and record["paper"] in PROFILES and
            record["status"] == "verified-input-preparation-not-a-release" and record["author_approval"] is False,
            "wrong input identity/authority")
    require(set(record["sources"]) == {"reading_bundle", "reading_build", "data", "experiment"}, "incomplete component bindings")
    for digest in record["sources"].values():
        require(type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest), "invalid component binding")
    files = record["files"]
    require(type(files) is dict and files, "empty input inventory")
    for name, digest in files.items():
        relative(name)
        require(type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest), "invalid input digest")
        require(found.get(name) == digest, "prepared input missing or changed: " + name)
        require(name == "verify_inputs.py" or name.startswith(("paper/", ANALYSIS + "/", EXPERIMENT + "/")), "unexpected owned input")
    extras = set(found) - set(files) - {"INPUTS.json"}
    # A final release may add curated metadata/cards around this component.
    # The release controller owns their checksums. Never allow an extra file
    # inside the owned scientific source/data/paper trees or a private reading tree.
    allowed_metadata = {"README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md", "FULL_RERUN.md",
                        "CLAIM_EVIDENCE_MAP.md", "requirements-analysis.txt", "check_claims.py", "release-files.json", "SHA256SUMS"}
    require(not extras or (staged and all(name in allowed_metadata or name.startswith(("huggingface/", "environment/", "LICENSES/"))
                                         for name in extras)), "extra or private files in prepared inputs")
    require(files.get("verify_inputs.py") == sha(Path(__file__)), "input verifier differs from component")
    profile = PROFILES[record["paper"]]
    expected_paper = {"paper/main.pdf", "paper/main.tex", "paper/references.bib", "paper/derived/claims.tex"}
    expected_paper |= {"paper/figures/" + name for name in profile["figures"]}
    require({name for name in files if name.startswith("paper/")} == expected_paper, "unexpected public paper files")
    data = read_json(root / ANALYSIS / "MANIFEST.json")
    source = read_json(root / EXPERIMENT / "SOURCE_MANIFEST.json")
    require(data["schema"] == profile["data_schema"] and data["author_approval"] is False and
            source["paper"] == record["paper"] and source["author_approval"] is False, "components belong to different papers")
    for prefix, manifest, filename, binding in ((ANALYSIS, data, "MANIFEST.json", "data"),
                                               (EXPERIMENT, source, "SOURCE_MANIFEST.json", "experiment")):
        require(files[prefix + "/" + filename] == record["sources"][binding], "component manifest binding differs")
        require({name.removeprefix(prefix + "/"): digest for name, digest in files.items() if name.startswith(prefix + "/")} ==
                {**manifest["files"], filename: record["sources"][binding]}, "component inventory differs")
    require(files["paper/derived/claims.tex"] == files[ANALYSIS + "/claims.tex"], "paper and checked results macros differ")
    return record, found


def run(command, cwd, timeout=180):
    env = dict(environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    result = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    require(result.returncode == 0, "component check failed:\n" + (result.stderr + result.stdout)[-4000:])
    return result.stdout


def check(root, *, staged=False, render=False):
    root = Path(root)
    record, before = verify(root, staged=staged)
    root = root.resolve()
    run([sys.executable, "-B", str(root / ANALYSIS / "check_results.py")], root / ANALYSIS)
    run([sys.executable, "-B", str(root / EXPERIMENT / "replication_source.py")], root / EXPERIMENT)
    if render:
        with tempfile.TemporaryDirectory(prefix="release-input-render-") as tmp:
            output = Path(tmp) / "render"
            run([sys.executable, "-B", str(root / ANALYSIS / "rebuild_results.py"), "--output", str(output)], Path(tmp))
            report = read_json(output / "BUILD.json")
            require(report["schema"] == PROFILES[record["paper"]]["render_schema"] and report["author_approval"] is False and
                    report["bundle_manifest_sha256"] == record["sources"]["data"], "wrong result-render identity")
            expected = {"derived/claims.tex"} | {"figures/" + f for f in PROFILES[record["paper"]]["figures"]}
            expected |= {"figures/dark/" + f for f in PROFILES[record["paper"]]["figures"]}
            require(set(report["outputs"]) == expected and tree(output) == {**report["outputs"], "BUILD.json": sha(output / "BUILD.json")},
                    "incomplete or altered result render")
            for name in expected:
                if not name.startswith("figures/dark/"):
                    require(report["outputs"][name] == record["files"]["paper/" + name], "rerender differs from paper input: " + name)
    require(tree(root) == before, "verification changed its inputs")
    return {"paper": record["paper"], "files": len(record["files"]), "input_unchanged": True,
            "analysis_checked": True, "experiment_source_checked": True, "render_checked": render,
            "experimental_inference_reexecuted": False, "author_approval": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--staged", action="store_true", help="allow surrounding curated release metadata checked by the controller")
    parser.add_argument("--render-check", action="store_true", help="also regenerate and compare light figures/macros; no inference or PDF build")
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.directory, staged=args.staged, render=args.render_check), indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(2, "Release input check refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
