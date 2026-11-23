#!/usr/bin/env python3
"""Prepare a NEW private P5 source layout with explicit local input/runtime paths.

This performs no download, model read, package install, native import or
inference. It does not turn the original study's execution receipts into a
new campaign baseline. Check live machine availability before heavy work.
"""
import argparse
import ast
import json
from pathlib import Path

import replication_check as A
import replication_source as S

SCHEMA = "paper5-rerun-setup-v1"
PRIMARY = "testsuite/paper5/"
COMMON = PRIMARY + "p5common.py"
CONFIG = PRIMARY + "tabby-p5.yml"
MINILM = "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
LINGUA = "models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ebaba9b0e874dadd3003ffcff828e4397e568089"
PATH_KEYS = ("models", "repo_assets", "hf_cache", "engines", "llama_library")
CHANGED = {COMMON, CONFIG, PRIMARY + "p5run.py", PRIMARY + "build_prompts.py", PRIMARY + "lingua_probs.py",
           PRIMARY + "run_grid.py", PRIMARY + "judge.py", "testsuite/paper5_followups/run.py",
           "paper5/scripts/align_oracle_dates.py", "paper5/scripts/primary_audit.py"}
ORIGINAL_WRAPPERS = ("paper5/scripts/primary_execution.py", "paper5/scripts/primary_receipts.py",
                     "paper5/scripts/primary_server.py", "paper5/scripts/execution_snapshot.py")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def absolute(raw):
    path = Path(raw)
    A.require(path.is_absolute() and ".." not in path.parts and str(path) == str(raw) and
              not any(ord(c) < 32 for c in str(path)) and not any(p.is_symlink() for p in (path, *path.parents)),
              "use an absolute, normalized, non-symlink local path")
    return path


def rewrite_expressions(name, data, replacements):
    """Exact AST-selected byte edits; every other expression remains identical."""
    tree = ast.parse(data, filename=name)
    selected = []
    for before, after, reason in replacements:
        expected = ast.dump(ast.parse(before, mode="eval").body)
        matches = [node for node in ast.walk(tree) if ast.dump(node) == expected]
        A.require(len(matches) == 1, "rerun expression missing/ambiguous in " + name + ": " + before)
        selected.append((matches[0], after.encode(), ast.parse(after, mode="eval").body, reason))
    offsets, offset = [], 0
    for line in data.splitlines(keepends=True):
        offsets.append(offset)
        offset += len(line)
    spans, substitutions = [], {}
    for node, payload, replacement, reason in selected:
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        spans.append((start, end, payload, reason))
        substitutions[id(node)] = replacement
    spans.sort()
    A.require(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])), "overlapping rerun substitutions")
    result = data
    for start, end, payload, _ in reversed(spans):
        result = result[:start] + payload + result[end:]
    class Expected(ast.NodeTransformer):
        def visit(self, node):
            return substitutions[id(node)] if id(node) in substitutions else super().visit(node)
    A.require(ast.dump(ast.parse(result)) == ast.dump(Expected().visit(tree)), "undeclared rerun Python change")
    return result, [{"reason": reason, "original_expression": data[start:end].decode(), "new_expression": payload.decode()}
                    for start, end, payload, reason in spans]


def adapt_python(name, data, paths):
    literal = lambda value: json.dumps(str(value), ensure_ascii=False)
    engines = paths["engines"]
    replacements = []
    def replace(before, after, reason="explicit local path"):
        replacements.append((before, after, reason))
    if name == COMMON:
        replace('os.path.join(REPO, "testsuite", "paper5_data")', literal(paths["repo_assets"] / "testsuite/paper5_data"))
        for relative in ("qwen38-exl3-2.0", "qwen3-8b"):
            replace(literal("~/models/" + relative), literal(paths["models"] / relative))
    elif name == PRIMARY + "p5run.py":
        for before, relative in (("os.path.join(repo, 'engines', 'exl3-1.4.7-lab')", "exl3-1.4.7-lab"),
                                 ("os.path.join(repo, 'engines', 'exl3-env', 'bin', 'python')", "exl3-env/bin/python"),
                                 ("os.path.join(repo, 'engines', 'tabbyAPI')", "tabbyAPI"),
                                 ("os.path.join(C.REPO, 'engines', 'llama.cpp', 'build', 'bin', 'llama-server')", "llama.cpp/build/bin/llama-server")):
            replace(before, literal(engines / relative))
        replace("dict(os.environ)", "dict(os.environ, LD_LIBRARY_PATH=" + literal(paths["llama_library"]) + ")",
                "explicit child-only llama.cpp loader path; no sampling/request change")
    elif name == PRIMARY + "build_prompts.py":
        replace("os.path.expanduser(" + literal("~/.cache/huggingface/hub/" + MINILM) + ")", literal(paths["hf_cache"] / MINILM))
    elif name == PRIMARY + "lingua_probs.py":
        before = "os.path.expanduser(" + literal("~/.cache/huggingface/hub/" + LINGUA.rsplit("/", 1)[0] + "/") + " + MODEL_REVISION)"
        replace(before, literal(paths["hf_cache"] / LINGUA))
    elif name in (PRIMARY + "run_grid.py", PRIMARY + "judge.py"):
        relative = "qwen3-8b/Qwen3-8B-UD-Q4_K_XL.gguf" if name.endswith("run_grid.py") else "gpt-oss-20b/gpt-oss-20b-MXFP4.gguf"
        replace(literal("~/models/" + relative), literal(paths["models"] / relative))
    elif name == "testsuite/paper5_followups/run.py":
        replace('Path(a.C.REPO) / "engines" / engine', "Path(" + literal(engines) + ") / engine")
        for relative in ("llama.cpp/build/bin", "exl3-1.4.7-lab"):
            replace('Path(a.C.REPO) / ' + literal("engines/" + relative), "Path(" + literal(engines / relative) + ")")
    elif name in ("paper5/scripts/align_oracle_dates.py", "paper5/scripts/primary_audit.py"):
        replace('root / "testsuite/paper5_data/longmemeval" / name',
                "Path(" + literal(paths["repo_assets"] / "testsuite/paper5_data/longmemeval") + ") / name")
        if name.endswith("primary_audit.py"):
            replace('Path(root) / "engines/exl3-env/bin/python"', "Path(" + literal(engines / "exl3-env/bin/python") + ")")
    if not replacements:
        return data, []
    return rewrite_expressions(name, data, replacements)


def adapt_config(data, paths):
    old = b"  model_dir: /path/to/models\n"
    new = ("  model_dir: " + json.dumps(str(paths["models"]), ensure_ascii=False) + "\n").encode()
    A.require(data.count(old) == 1, "rerun Tabby model directory field changed")
    result = data.replace(old, new)
    changes = [{"reason": "explicit local model directory; all runtime scalar values unchanged",
                "original_expression": old.decode().rstrip(), "new_expression": new.decode().rstrip()}]
    # These are comments in the copied config, not edits to the frozen study.
    for before, after in (
        ("# 1 slot, 32K context, chunk 1024, draft length 6, 16 GB recurrent stash in host RAM.",
         "# 1 slot, 32K context, chunk 1024, draft length 6, 4 GB recurrent stash in host RAM."),
        ("# Launch with serving/launch.sh daily  (sets PYTHONPATH to the lab-patched exllamav3 copy and EXL3_CKPT_PP=1024).",
         "# Use this rerun's explicit engine paths and experiment harness; see RERUN.md."),
    ):
        A.require(result.count(before.encode()) == 1, "historical Tabby comment changed")
        result = result.replace(before.encode(), after.encode())
        changes.append({"reason": "copied configuration comment only", "original_expression": before, "new_expression": after})
    return result, changes


def validate_assets(catalog):
    A.validate(catalog)
    expected = {("models", "qwen38-exl3-2.0/" + name) for name in
                ("chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                 "quantization_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")}
    expected |= {("models", "qwen3-8b/" + name) for name in
                 ("config.json", "tokenizer.json", "tokenizer_config.json", "Qwen3-8B-UD-Q4_K_XL.gguf")}
    expected.add(("models", "gpt-oss-20b/gpt-oss-20b-MXFP4.gguf"))
    expected |= {("repo", "testsuite/paper5_data/" + name) for name in
                 ("longmemeval/longmemeval_s_cleaned.json", "longmemeval/longmemeval_oracle.json", "locomo/locomo10.json")}
    for folder in (MINILM, LINGUA):
        names = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
        if folder == MINILM:
            names.append("vocab.txt")
        expected |= {("hf_cache", folder + "/" + name) for name in names}
    A.require(catalog["paper"] == "paper5" and len(catalog["assets"]) == 30 and
              {(row["root"], row["path"]) for row in catalog["assets"]} == expected, "P5 requires the exact thirty asset locations")
    return catalog


def assemble(component, paths):
    manifest, before = S.verify(component)
    A.require(manifest["paper"] == "paper5", "this rerun adapter is for P5 only")
    validate_assets(A.read_json(component / "ASSETS.json"))
    files, changes = {}, []
    for row in manifest["sources"]:
        source = row["output"]
        A.require(source.startswith("replication/"), "unexpected P5 source layout")
        name = source.removeprefix("replication/")
        data = (component / source).read_bytes()
        if name == CONFIG:
            payload, substitutions = adapt_config(data, paths)
        else:
            payload, substitutions = adapt_python(name, data, paths) if name.endswith(".py") else (data, [])
        files[name] = payload
        if substitutions:
            changes.append({"path": name, "component_source_sha256": S.sha(data), "adapted_sha256": S.sha(payload),
                            "substitutions": substitutions})
    A.require({row["path"] for row in changes} == CHANGED, "missing/unexpected P5 source adaptations")
    files["ASSETS.json"] = (component / "ASSETS.json").read_bytes()
    for name in ("SOURCE_MANIFEST.json", "ENVIRONMENT.json", "RUNTIME.md", "THIRD_PARTY_NOTICES.md"):
        files["reference/" + name] = (component / name).read_bytes()
    for name in before:
        if name.startswith("LICENSES/"):
            files[name] = (component / name).read_bytes()
    files["RERUN.md"] = GUIDE.encode()
    files["verify_setup.py"] = Path(__file__).read_bytes()
    files["freeze_inputs.py"] = (component / "freeze_p5_rerun.py").read_bytes()
    files["replication_check.py"] = Path(A.__file__).read_bytes()
    files["replication_source.py"] = Path(S.__file__).read_bytes()
    record = {"schema": SCHEMA, "paper": "paper5", "status": "new-rerun-source-layout-not-executed",
              "author_approval": False, "original_component_sha256": before["SOURCE_MANIFEST.json"],
              "paths": {name: str(path) for name, path in paths.items()}, "changes": changes,
              "assets_content_verified": False, "inference_started": False, "new_execution_baseline_created": False,
              "original_study_wrappers_preserved_not_rebound": list(ORIGINAL_WRAPPERS),
              "remaining_execution_work": "new campaign preflight/pilot and independently bound execution path; freeze_inputs.py checks inputs only, not execution",
              "files": {name: S.sha(data) for name, data in files.items()}}
    record["setup_record_sha256"] = S.sha(canonical(record).encode())
    files["RERUN_SETUP.json"] = canonical(record).encode()
    A.require(S.verify(component)[1] == before, "source component changed during adaptation")
    return files


def verify_setup(directory):
    directory = absolute(directory)
    record = A.read_json(directory / "RERUN_SETUP.json")
    A.require(record["setup_record_sha256"] == S.sha(canonical({k: v for k, v in record.items() if k != "setup_record_sha256"}).encode()),
              "P5 rerun setup checksum differs")
    A.require(record["schema"] == SCHEMA and record["paper"] == "paper5" and
              record["status"] == "new-rerun-source-layout-not-executed" and record["author_approval"] is False and
              record["assets_content_verified"] is False and record["inference_started"] is False and
              record["new_execution_baseline_created"] is False, "P5 setup identity/authority differs")
    A.require(set(record["paths"]) == set(PATH_KEYS) and {row["path"] for row in record["changes"]} == CHANGED and
              record["original_study_wrappers_preserved_not_rebound"] == list(ORIGINAL_WRAPPERS), "P5 setup path/source boundary differs")
    A.digest(record["original_component_sha256"])
    for value in record["paths"].values():
        absolute(value)
    required = CHANGED | set(ORIGINAL_WRAPPERS) | {"ASSETS.json", "RERUN.md", "verify_setup.py", "replication_source.py",
                "replication_check.py", "freeze_inputs.py", "reference/SOURCE_MANIFEST.json", "reference/ENVIRONMENT.json",
                "research/design-overflow-policies.md", "testsuite/paper5/summarize.py", "testsuite/paper5/analyze.py",
                "testsuite/paper5_followups/followups.py", "testsuite/paper5_followups/legacy.py", "testsuite/paper5_followups/jobs.py"}
    A.require(required <= record["files"].keys(), "P5 setup is missing required inputs")
    for name, digest in record["files"].items():
        A.relative(name)
        A.digest(digest)
        path = directory / name
        A.require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file() and
                  path.stat().st_size <= 4 * 1024 * 1024 and S.sha(path.read_bytes()) == digest, "P5 setup input changed: " + name)
    A.require(record["files"]["verify_setup.py"] == S.sha(Path(__file__).read_bytes()) and
              record["files"]["reference/SOURCE_MANIFEST.json"] == record["original_component_sha256"], "P5 setup/reference checker identity differs")
    validate_assets(A.read_json(directory / "ASSETS.json"))
    return {"paper": "paper5", "setup_inputs_unchanged": True, "experiment_outputs_checked": False,
            "new_execution_baseline_created": False, "author_approval": False,
            "setup_sha256": S.sha((directory / "RERUN_SETUP.json").read_bytes())}


def prepare(component, output, **paths):
    component, output = absolute(component), absolute(output)
    A.require(set(paths) == set(PATH_KEYS), "P5 setup requires all five explicit asset/runtime roots")
    paths = {name: absolute(value) for name, value in paths.items()}
    for protected in (component, *paths.values()):
        A.require(not output.is_relative_to(protected) and not protected.is_relative_to(output), "P5 rerun overlaps a source/asset/runtime tree")
    A.require(not output.exists() and output.parent.is_dir(), "P5 rerun output must be new and its parent must exist")
    files = assemble(component, paths)
    output.mkdir(mode=0o700)
    for name, payload in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(payload)
    return verify_setup(output)


GUIDE = """# A new P5 workspace, not the original study

This is a private SOURCE LAYOUT, not a completed inference setup or a results
release. RERUN_SETUP.json records every source/path adaptation. No old prompt,
pilot, summary, journal, cached classifier output, approval or original study
execution baseline was copied. All generated work belongs in this NEW tree.
Do not publish its private local paths or claim its files are original executed
bytes. reference/ holds the source manifest and descriptive documentation only.

The adapter changes nine Python files and one copied YAML config. Those changes
are limited to explicit input/runtime paths and a child-scoped llama.cpp CUDA
library path. Two misleading config comments are corrected; the actual 4,096
MiB stash and every serving/sampling/scoring setting are unchanged. The common
module's WORK and all primary/follow-up writable paths remain under this tree.
Dataset/model pins, seeds, task cohorts, policy algorithms, request construction,
retry rules, pause/thermal behavior and follow-up hardware checks are preserved.

Check setup with `python3 -B verify_setup.py --verify .`. This checks named setup
inputs only, not later generated data, runtime identity or experiment outputs.
The command has no inference, network, download or package-install option.

The five explicit roots are in RERUN_SETUP.json. engines contains the named
exl3-env, lingua-env, exl3-1.4.7-lab, tabbyAPI and llama.cpp directories; it is
not the writable experiment repository. ASSETS.json retains all 30 original
relative locations and immutable identities. Use replication_check.py with
your repo/models/hf-cache roots for a size plan, or explicitly --verify after
checking live machine availability. An identity catalog is not a backup or a
download receipt. Model/dataset licenses still apply.
Setup paths must name physical directories, not symlink aliases. In particular,
CUDA's lib64 is commonly an alias: resolve and inspect its directory target
before passing --llama-library-dir; do not alter the original CUDA installation.

Required next stages, NOT performed by this adapter:

1. Recreate the pinned native/dependency stacks. The original source component's
   wheel/overlay helpers reproduce EXL3 bytes only. CUDA/Python dependencies,
   model acquisition and native imports require separate checks. The copied
   reference/RUNTIME.md is historical component documentation; its example
   paths refer to that component, not this flattened source layout.
2. Make a new prospective campaign baseline and guarded execution path. Preserve
   the study's RTX 4080 Laptop hardware condition; inspect live candidate-machine
   CPU/RAM/GPU headroom before any heavy job. Never overlap primary/follow-up/P6
   GPU campaigns. Ports 18125/18126/18127/18130 and ten-second request pacing
   remain the operating convention. A pause marker reduces activity, not VRAM.
3. Run a NEW complete 20-item LoCoMo category-3 pilot, then its fixed cap rule.
   Recreate classifier probabilities and verify selection against LLMLingua;
   build all primary T/C/Cw/M/Mbm25/O/E prompts and session summaries, then S.
   CPU-only/classifier work can still be heavy and needs a placement check.
4. Before any primary freeze/generation, use the copied
   paper5/scripts/align_oracle_dates.py to inspect, then explicitly --apply the
   date-only oracle correction. It creates its own preserved original and
   input/source-bound alignment receipt in this workspace. The directory name
   ending 20260920 names the correction design, not a claimed run timestamp.
5. After the complete new pilot, preparation and oracle correction, run
   `python3 -B freeze_inputs.py` for a read-only input check. It rechecks pinned
   datasets/tokenizer, all 20 pilot requests, the cap rule, all 13 prompt cells,
   corrected oracle evidence and adapted source hashes. Tokenization is CPU
   work; check live headroom first. `--freeze` explicitly creates the manifest
   and private RERUN_INPUT_FREEZE.json; `--verify` rechecks that saved pair.
   No input is overwritten. An identical partial manifest can be completed,
   but a different manifest or prior primary measurement journal is refused.
   This proves input consistency, NOT actual pilot runtime/placement/ownership.
   A separate prospective pilot/execution evidence path is still required.
   Use a separate inspected smoke journal for a NEW campaign. The original
   study's special retained 24-row prefix is not a license to invent or copy
   unbound rows. Complete primary generation and both graders with separately
   bound execution evidence before claims or downstream policy selection.
6. The isolated follow-up code uses these local paths and its existing preflight,
   exact-weight/engine checks, independent workspaces and complete-primary gate.
   Initialize/freeze new workspaces only after the new pilot. Cross-model needs
   all four policies; LoCoMo T plus S/M/C selected from complete primary claims.
   Keep all actual failures/attempts, then run both independent result audits.

The four original-study execution wrappers listed in RERUN_SETUP.json remain
unchanged for reference. They deliberately require the ORIGINAL private
snapshot and must refuse this new workspace. Do not replace their baseline
hash with a new file or bypass their guards to make them pass. The separately
bound new-campaign execution path is still work to complete. The new input
checker supplies only the explicit input-freeze stage described above. The ordinary legacy
run_grid.py and summarize.py can start a server: do not use them as safety
preflights. No execution command is launched by setup or its verifier.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--component", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    for name in PATH_KEYS:
        parser.add_argument("--" + name.replace("_", "-") + ("-dir" if name == "llama_library" else "-root"), dest=name, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in PATH_KEYS}
    try:
        if args.verify:
            A.require(not args.output and not any(paths.values()), "--verify cannot create another P5 workspace")
            result = verify_setup(args.verify.absolute())
        else:
            A.require(args.output and all(paths.values()), "P5 setup needs --output and all five explicit roots")
            result = prepare(args.component, args.output, **paths)
        print(canonical(result), end="")
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        parser.exit(2, "P5 rerun setup refused (any partial new output is retained): " + str(exc) + "\n")


if __name__ == "__main__":
    main()
