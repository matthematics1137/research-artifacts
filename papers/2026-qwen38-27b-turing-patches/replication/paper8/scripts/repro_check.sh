#!/usr/bin/env bash
# Paper 8: the analysis-only reproduction check (publication/PAPER_CONTRACT.md, section 7).
#
# From a clean copy of one commit's files, with no model and no GPU, the script
#   1. extracts the analysis-only paths (PATHS below) with `git archive COMMIT PATHS | tar -x` into a new work folder;
#   2. rebuilds the git-ignored long-context item files the scorer reads from the committed manifests, with the
#      committed generators (CPU only), and checks every rebuilt file and manifest against the committed one;
#   3. runs paper8/scripts/build_claims.py in the copy and compares derived/claims.tex byte for byte with the
#      commit's, then runs paper8/scripts/make_figures.py (light and --dark) and compares the figure PDFs.
# The commit's own claims.tex and figure PDFs are moved out of the copy (to WORK/committed/) before anything runs, so
# every compared file is one the scripts wrote. The script writes only inside the work folder and leaves it in place.
# Exit status: 0 when everything matches, 1 on any mismatch or failed step, 2 on a usage or setup problem.
#
#   paper8/scripts/repro_check.sh [options] [COMMIT]          (COMMIT defaults to HEAD)
#
#   --work DIR         the work folder, new or empty (default: a new folder from mktemp -d under ${TMPDIR:-/tmp})
#   --tokenizer FILE   the Qwen3.8 tokenizer.json (default: $HOME/models/qwen38-exl3-2.0/tokenizer.json), SHA-256
#                      0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3. The generators read
#                      ~/models/qwen38-exl3-2.0/tokenizer.json, so they run with HOME set to WORK/home, which holds a
#                      link to this file.
#   --lme FILE         LongMemEval-S cleaned, git-ignored (default: <repo>/testsuite/paper5_data/longmemeval/
#                      longmemeval_s_cleaned.json), SHA-256
#                      d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442; provenance in
#                      testsuite/paper5_data/MANIFEST.md. P6's generator reads it; it is linked into the copy there.
#   --tok-python PY    a Python with the tokenizers package, for the generators (default: <repo>/engines/exl3-env/
#                      bin/python when present, else python3)
#   --python PY        the Python for the claims and the figures (default: python3, as paper8/Makefile uses). The
#                      figure PDFs are byte-identical only with matplotlib 3.10.6, the version the committed PDFs
#                      record (3.11.2 writes different content streams). The claims need only the standard library.
#   --jobs N           item sets built at once (default 4; each generator holds about 2.5 GB of memory)
#   --all-items        rebuild the set of every committed item and prompt manifest (16 item sets, 3 prompt sets),
#                      not only the sets the archived journals point to (8, the ones the claims read)
#   --no-figures       skip the figures
#   --repo DIR         the git repository (default: the one that holds this script)
#
# Needs git, tar, sha256sum, cmp, awk and Python 3; no network, no model weights, no GPU.

set -uo pipefail

# The committed paths the analysis-only path needs. tools/four_resources is never read (kept from the starting list);
# testsuite/paper5/p5common.py is imported by P6's generator (testsuite/paper6/make_longctx.py).
PATHS=(
  paper8
  paper6/derived/claims.tex
  testsuite/results/turing
  testsuite/results/four_resources
  testsuite/results/ada
  testsuite/results/ideas
  testsuite/results/efficiency/hash_audit_2026-09-06.json
  testsuite/results/p9/analysis
  testsuite/p8_longctx
  testsuite/paper6
  testsuite/paper5/p5common.py
  tools/turing
  tools/four_resources
  research/raw/turing-competition-refresh-2026-11-04
  # Dated fleet replication: only the P8 summaries, verdicts and text-excluded exports it reads.
  research/raw/2026-11-12-fleet-lanes/results-2026-11-13-7c2e/analysis/registered-summaries.json
  research/raw/2026-11-12-fleet-lanes/results-2026-11-14-c4d8/available/registered-summaries.json
  research/raw/2026-11-12-fleet-lanes/results-2026-11-14-c4d8/prediction-verdicts.json
  research/raw/2026-11-12-fleet-lanes/p8-card-labels.json
)
TOK_SHA=0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
LME_SHA=d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
TOK_REL=models/qwen38-exl3-2.0/tokenizer.json
LME_REL=testsuite/paper5_data/longmemeval/longmemeval_s_cleaned.json
# paper8/Makefile exports these for every target
BUILD_ENV=(SOURCE_DATE_EPOCH=1794506400 FORCE_SOURCE_DATE=1 TZ=UTC OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
           CUDA_VISIBLE_DEVICES=)

die() { echo "repro_check: $*" >&2; exit 2; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
now() { echo "${EPOCHREALTIME/,/.}"; }
secs() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }'; }
sha() { sha256sum < "$1" | cut -d' ' -f1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="" WORK="" COMMIT=HEAD TOKENIZER="$HOME/$TOK_REL" LME="" TOKPY="" PY=python3 JOBS=4 ALL_ITEMS=0 FIGURES=1
while [ $# -gt 0 ]; do
  case $1 in
    --work|--tokenizer|--lme|--tok-python|--python|--jobs|--repo)
      [ $# -ge 2 ] || die "$1 needs a value"
      case $1 in
        --work) WORK=$2 ;; --tokenizer) TOKENIZER=$2 ;; --lme) LME=$2 ;; --tok-python) TOKPY=$2 ;;
        --python) PY=$2 ;; --jobs) JOBS=$2 ;; --repo) REPO=$2 ;;
      esac
      shift 2 ;;
    --all-items) ALL_ITEMS=1; shift ;;
    --no-figures) FIGURES=0; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) COMMIT=$1; shift ;;
  esac
done
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || die "--jobs needs a positive whole number"

# ------------------------------------------------------------------------------------------------ preflight
for tool in git tar sha256sum cmp awk; do command -v "$tool" > /dev/null || die "$tool is not installed"; done
[ -n "$REPO" ] || REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2> /dev/null) || die "no git repository; pass --repo"
REPO=$(cd "$REPO" && pwd) || die "no folder $REPO"
SHA=$(git -C "$REPO" rev-parse --verify --quiet "$COMMIT^{commit}") || die "no such commit: $COMMIT"
# Resolve private evidence paths from the committed card-fixture map, outside public paper sources.
CARD_MAP=research/raw/2026-11-12-fleet-lanes/p8-card-labels.json
CARD_ROWS=$(git -C "$REPO" show "$SHA:$CARD_MAP" | "$PY" -c '
import json,sys
cards=json.load(sys.stdin)
assert set(cards)=={"CardA","CardB"} and len(set(cards.values()))==2
for tag in sorted(cards):
    value=cards[tag]
    assert isinstance(value,str) and value and all(c.isalnum() or c in "-_" for c in value)
    print(value)
') || die "invalid committed card-fixture map"
while IFS= read -r card; do
  for mode in nd mtp; do
    PATHS+=("research/raw/2026-11-12-fleet-lanes/results-2026-11-14-c4d8/available/quality-rows/FLEET-${card}-w13-${mode}")
  done
  PATHS+=("legal_tech/results/2026-11-07-fleet-baselines/${card}/runs/ns-20261112-p8-speed/journal.jsonl")
done <<< "$CARD_ROWS"
[ -n "$LME" ] || LME=$REPO/$LME_REL
[ -n "$TOKPY" ] || { if [ -x "$REPO/engines/exl3-env/bin/python" ]; then TOKPY=$REPO/engines/exl3-env/bin/python; else TOKPY=python3; fi; }
[ -f "$TOKENIZER" ] || die "no tokenizer at $TOKENIZER (pass --tokenizer)"
[ "$(sha "$TOKENIZER")" = "$TOK_SHA" ] || die "$TOKENIZER: SHA-256 is not $TOK_SHA"
[ -f "$LME" ] || die "no LongMemEval-S cleaned at $LME (pass --lme)"
[ "$(sha "$LME")" = "$LME_SHA" ] || die "$LME: SHA-256 is not $LME_SHA"
"$PY" -c 'import sys; assert sys.version_info >= (3, 8)' 2> /dev/null || die "$PY is not Python 3.8 or newer"
if [ "$FIGURES" = 1 ]; then
  "$PY" -c 'import matplotlib' 2> /dev/null || die "$PY has no matplotlib (pass --python, or --no-figures)"
fi
TOKPY=$(command -v "$TOKPY") || die "no Python $TOKPY"
# the generators run with a temporary HOME; PYTHONUSERBASE keeps the real user site-packages visible to them
USERBASE=$("$TOKPY" -c 'import site; print(site.getuserbase())' 2> /dev/null) || die "$TOKPY does not run"
TOK_VER=$(env PYTHONUSERBASE="$USERBASE" "$TOKPY" -c 'import tokenizers; print(tokenizers.__version__)' 2> /dev/null) \
  || die "$TOKPY cannot import tokenizers (pass --tok-python)"

if [ -z "$WORK" ]; then
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/p8-repro.XXXXXX") || die "mktemp failed"
else
  [ ! -e "$WORK" ] || [ -z "$(ls -A "$WORK")" ] || die "the work folder $WORK is not empty"
  mkdir -p "$WORK" || die "cannot make $WORK"
fi
WORK=$(cd "$WORK" && pwd)
mkdir -p "$WORK/repo" "$WORK/committed" "$WORK/logs" "$WORK/tools" "$WORK/opt/p8-user "$TOK_REL")"
LOG=$WORK/logs/repro_check.log
say() { echo "$*" | tee -a "$LOG"; }
GEN_ENV=(HOME="$WORK/home" PYTHONUSERBASE="$USERBASE" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=)
T_ALL=$(now)
PROBLEMS=()
problem() { PROBLEMS+=("$*"); say "  MISMATCH: $*"; }

{
  echo "commit: $SHA ($COMMIT)"
  echo "repository: $REPO"
  echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -srm), $(nproc 2> /dev/null || echo '?') CPUs"
  echo "python (claims, figures): $(command -v "$PY") $("$PY" -c 'import sys; print(sys.version.split()[0])')"
  [ "$FIGURES" = 1 ] && echo "matplotlib: $("$PY" -c 'import matplotlib; print(matplotlib.__version__)')"
  echo "python (generators): $TOKPY $("$TOKPY" -c 'import sys; print(sys.version.split()[0])'), tokenizers $TOK_VER"
  echo "tokenizer: $TOKENIZER ($TOK_SHA)"
  echo "longmemeval_s_cleaned: $LME ($LME_SHA)"
} > "$WORK/env.txt"
say "Paper 8 analysis-only reproduction check"
say "  commit $SHA, work folder $WORK"

# ------------------------------------------------------------------------------------------------ helper
cat > "$WORK/tools/p8_repro_helper.py" << 'PYEOF'
"""Helper for paper8/scripts/repro_check.sh (written by it into the work folder): plan the item rebuilds, check
them against the committed manifests, and compare PDFs. Standard library only.

  p8_repro_helper.py plan WORK ALL          the generator jobs (WORK/item-plan.json; tab-separated lines on stdout)
  p8_repro_helper.py verify WORK REPO       rebuilt files and manifests against the committed manifests
  p8_repro_helper.py pdfcmp REBUILT COMMITTED
"""
import glob
import hashlib
import json
import os
import re
import sys

P8 = os.path.join("repo", "testsuite", "p8_longctx")    # paths are relative to the work folder
OUT = os.path.join(P8, "work")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def journal_sets(work):
    """The item files the archived journals' arm plans name: the files the scorer opens."""
    names = set()
    pattern = os.path.join(work, "repo", "testsuite", "results", "turing", "longctx", "*", "journal*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "arm_plan" in line:
                    r = json.loads(line)
                    if r.get("kind") == "arm_plan" and r.get("items_file"):
                        names.add(r["items_file"])
    return names


def plan(work, everything):
    items, prompts = {}, []
    paths = sorted(glob.glob(os.path.join(work, P8, "manifests", "*.json")))
    paths.append(os.path.join(work, P8, "stage1_prompts_manifest.json"))
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        rel = os.path.relpath(path, work)
        if m.get("schema") == "p8-longctx-items-v1":
            items[m["file"]["name"]] = (rel, m)
        elif m.get("schema") == "p8-longctx-stage1-prompts-v1":
            prompts.append((rel, m))
    want = set(items) if everything else journal_sets(work)
    if not want:
        raise SystemExit("no item files named in the archived journals")
    missing = sorted(want - set(items))
    if missing:
        raise SystemExit(f"no committed manifest for the item files {missing}")
    jobs = []
    for name in sorted(want):
        rel, m = items[name]
        if m["kind"] == "rep":
            script, args = "build_replication_items.py", []
        else:
            script, args = "build_stage2_items.py", [m["kind"], "--length", str(m["length"])]
            if m["kind"] in ("graded", "stage3"):
                args += ["--levels", ",".join(f"{k}={v}" for k, v in sorted(m["levels"].items()))]
        jobs.append({"name": m["set"], "type": "items", "manifest": rel, "script": script,
                     "args": args + ["--out", OUT], "items": os.path.join(OUT, "items", name),
                     "regen": os.path.join(OUT, "manifests", os.path.basename(rel)),
                     "weight": m["length"] * m["n_items"]})
    if everything:
        for rel, m in prompts:
            base = os.path.basename(rel)
            regen = os.path.join(OUT, "manifests", base)
            if any(p["role"] == "check" for p in m["prompts"]):    # stage 1's set: the check prompt and the ladder
                args, out = ["--manifest", regen], os.path.join(OUT, "stage1_prompts")
            else:                                                 # the reach probes: their ladder lengths
                args = ["--lengths", ",".join(str(p["length"]) for p in m["prompts"]), "--manifest", regen]
                out = os.path.join(OUT, "items", "probe")
            jobs.append({"name": base[:-len(".json")], "type": "prompts", "manifest": rel,
                         "script": "build_stage1_prompts.py", "args": args + ["--out", out], "out": out,
                         "regen": regen, "weight": sum(p["length"] for p in m["prompts"])})
    with open(os.path.join(work, "item-plan.json"), "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=1)
    for j in sorted(jobs, key=lambda j: -j["weight"]):     # the longest builds first
        print("\t".join([j["name"], j["script"]] + j["args"]))


def verify(work, repo):
    with open(os.path.join(work, "item-plan.json"), encoding="utf-8") as f:
        jobs = json.load(f)
    bad, notes = 0, []
    for j in jobs:
        with open(os.path.join(work, j["manifest"]), encoding="utf-8") as f:
            committed = json.load(f)
        regen_path = os.path.join(work, j["regen"])
        if not os.path.exists(regen_path):
            print(f"  MISMATCH: {j['name']}: no rebuilt manifest (logs/items-{j['name']}.log)")
            bad += 1
            continue
        with open(regen_path, encoding="utf-8") as f:
            regen = json.load(f)
        code_key = "code" if j["type"] == "items" else "generator"
        problems, info = [], ""
        if j["type"] == "items":
            path = os.path.join(work, j["items"])
            if not os.path.exists(path):
                problems.append("no rebuilt item file")
            else:
                got, size, want = sha256(path), os.path.getsize(path), committed["file"]
                if got != want["sha256"] or size != want["bytes"]:
                    problems.append(f"rebuilt {got} ({size} bytes); the manifest says {want['sha256']} "
                                    f"({want['bytes']} bytes)")
                info = f"{got[:16]}  {size:>11,} bytes"
                local = os.path.join(repo, "testsuite", "p8_longctx", "work", "items", os.path.basename(path))
                if os.path.exists(local):
                    info += "; repo's work copy " + ("identical" if sha256(local) == got else "DIFFERS")
        else:
            outdir = os.path.join(work, j["out"])
            info = f"{len(committed['prompts'])} prompts"
            local_dir = os.path.join(repo, "testsuite", "p8_longctx", "work",
                                     "stage1_prompts" if j["out"].endswith("stage1_prompts") else "items/probe")
            same = [sha256(os.path.join(outdir, p["file"])) == sha256(os.path.join(local_dir, p["file"]))
                    for p in committed["prompts"] if os.path.exists(os.path.join(local_dir, p["file"]))
                    and os.path.exists(os.path.join(outdir, p["file"]))]
            if same:
                info += "; repo's work copies " + ("identical" if all(same) else "DIFFER")
        differ = sorted(k for k in set(committed) | set(regen) if k != code_key and committed.get(k) != regen.get(k))
        if differ:
            problems.append(f"the rebuilt manifest differs in {differ}")
        c0, c1 = committed.get(code_key) or {}, regen.get(code_key) or {}
        for k in sorted(set(c0) | set(c1)):
            if c0.get(k) != c1.get(k):
                notes.append(f"{j['name']}: the manifest records {k} {str(c0.get(k))[:12]}, this commit's file is "
                             f"{str(c1.get(k))[:12]}")
        if problems:
            bad += 1
            print(f"  MISMATCH: {j['name']}: " + "; ".join(problems))
        else:
            print(f"  ok  {j['name']:<24} {info}")
    if notes:
        print("  note: these sets were built by an earlier committed version of the generator; the outputs above are "
              "checked against the manifests' file hashes and items, not their code hashes:")
        for n in notes:
            print(f"    {n}")
    return 1 if bad else 0


def pdf_norm(b):
    b = re.sub(rb"/(CreationDate|ModDate)\s*\((?:\\.|[^\\)])*\)", b"", b)
    b = re.sub(rb"\nxref\n.*?\ntrailer", b"\nxref trailer", b, flags=re.S)
    return re.sub(rb"startxref\s+\d+", b"startxref", b)


def pdfcmp(a, b):
    with open(a, "rb") as f:
        x = f.read()
    with open(b, "rb") as f:
        y = f.read()
    if x == y:
        print("identical")
        return 0
    if pdf_norm(x) == pdf_norm(y):
        print("differs only in embedded dates")
        return 0

    def producer(z):
        m = re.search(rb"/Producer\s*\(([^)]*)\)", z)
        return m.group(1).decode("latin-1") if m else "?"
    print(f"DIFFERS (rebuilt: {producer(x)}; committed: {producer(y)})")
    return 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan(sys.argv[2], sys.argv[3] == "1")
    elif cmd == "verify":
        sys.exit(verify(sys.argv[2], sys.argv[3]))
    elif cmd == "pdfcmp":
        sys.exit(pdfcmp(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {cmd}")
PYEOF
HELPER=("$PY" -B "$WORK/tools/p8_repro_helper.py")

# ------------------------------------------------------------------------------------------------ 1. extract
say "1. extract the committed paths"
printf '%s\n' "${PATHS[@]}" > "$WORK/paths.txt"
echo "$SHA" > "$WORK/commit.txt"
t=$(now)
git -C "$REPO" archive --format=tar "$SHA" -- "${PATHS[@]}" 2> "$WORK/logs/archive.log" | tar -x -C "$WORK/repo"
st=("${PIPESTATUS[@]}")
[ "${st[0]}" = 0 ] && [ "${st[1]}" = 0 ] || { cat "$WORK/logs/archive.log" >&2; die "git archive | tar failed (${st[*]})"; }
T_EXTRACT=$(secs "$t")
say "  $(find "$WORK/repo" -type f | wc -l) files from ${#PATHS[@]} paths in ${T_EXTRACT} s"
# set the commit's outputs aside: everything compared below must be written by this run
OUTPUTS=(paper8/derived/claims.tex paper8/derived/f2_per_prompt.tex paper8/derived/model_files.tex
         paper8/derived/bonsai_counterpoint.tex paper8/derived/bonsai_sentence.tex
         paper8/derived/fleet_replication.tex)
if [ "$FIGURES" = 1 ]; then
  while IFS= read -r f; do OUTPUTS+=("$f"); done < <(cd "$WORK/repo" && find paper8/figures -name '*.pdf' | sort)
fi
for f in "${OUTPUTS[@]}"; do
  [ -f "$WORK/repo/$f" ] || die "the commit has no $f"
  mkdir -p "$WORK/committed/$(dirname "$f")"
  mv "$WORK/repo/$f" "$WORK/committed/$f" || die "cannot move $f aside"
done

# ------------------------------------------------------------------------------------------------ 2. items
say "2. rebuild the item files from the committed manifests ($TOKPY, tokenizers $TOK_VER, $JOBS at a time)"
ln -s "$(realpath "$TOKENIZER")" "$WORK/opt/p8-user" || die "cannot link the tokenizer"
mkdir -p "$WORK/repo/$(dirname "$LME_REL")"
ln -s "$(realpath "$LME")" "$WORK/repo/$LME_REL" || die "cannot link LongMemEval-S"
"${HELPER[@]}" plan "$WORK" "$ALL_ITEMS" > "$WORK/item-jobs.tsv" || die "could not plan the item rebuilds"
run_job() {   # name script args...
  local name=$1 script=$2 t0 rc
  shift 2
  t0=$(now)
  ( cd "$WORK" && env "${GEN_ENV[@]}" nice -n 10 "$TOKPY" -B "repo/testsuite/p8_longctx/$script" "$@" ) \
    > "$WORK/logs/items-$name.log" 2>&1 < /dev/null
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(secs "$t0")" > "$WORK/logs/items-$name.status"
}
t=$(now)
running=0
while IFS=$'\t' read -r -a job; do
  [ "${#job[@]}" -ge 2 ] || continue
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
  run_job "${job[@]}" &
  running=$((running + 1))
done < "$WORK/item-jobs.tsv"
wait
T_ITEMS=$(secs "$t")
while IFS=$'\t' read -r -a job; do
  st_file=$WORK/logs/items-${job[0]}.status
  if [ ! -f "$st_file" ]; then problem "${job[0]}: the generator did not finish"; continue; fi
  IFS=$'\t' read -r name rc took < "$st_file"
  [ "$rc" = 0 ] || problem "$name: the generator exited $rc (logs/items-$name.log)"
  echo "  $name: exit $rc in $took s" >> "$LOG"
done < "$WORK/item-jobs.tsv"
say "  $(wc -l < "$WORK/item-jobs.tsv") sets built in ${T_ITEMS} s; checking them against the committed manifests"
"${HELPER[@]}" verify "$WORK" "$REPO" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || PROBLEMS+=("rebuilt item files or manifests differ from the committed manifests")

# ------------------------------------------------------------------------------------------------ 3. claims, figures
say "3. build the claims in the copy ($PY)"
t=$(now)
( cd "$WORK/repo/paper8" && env "${BUILD_ENV[@]}" nice -n 10 "$PY" -B scripts/build_claims.py \
      && env "${BUILD_ENV[@]}" nice -n 10 "$PY" -B scripts/check_bonsai.py --write-derived ) \
  > "$WORK/logs/claims.log" 2>&1
rc=$?
T_CLAIMS=$(secs "$t")
say "  build_claims.py exit $rc in ${T_CLAIMS} s: $(tail -n 1 "$WORK/logs/claims.log")"
NEW=$WORK/repo/paper8/derived/claims.tex OLD=$WORK/committed/paper8/derived/claims.tex
if [ "$rc" != 0 ] || [ ! -f "$NEW" ]; then
  problem "build_claims.py failed (logs/claims.log)"
elif cmp -s "$NEW" "$OLD"; then
  say "  claims.tex identical to the commit's ($(grep -c '^\\newcommand' "$NEW") macros, sha256 $(sha "$NEW"))"
else
  problem "claims.tex differs from the commit's"
  diff "$OLD" "$NEW" | head -n 40 | tee -a "$LOG"
fi
for t in f2_per_prompt.tex model_files.tex bonsai_counterpoint.tex bonsai_sentence.tex fleet_replication.tex; do   # generated tables, written by the same run
  if cmp -s "$WORK/repo/paper8/derived/$t" "$WORK/committed/paper8/derived/$t"; then say "  $t identical to the commit's"
  else problem "$t differs from the commit's (or was not written)"; fi
done
if [ -f "$REPO/paper8/derived/claims.tex" ] && [ -f "$NEW" ]; then   # information only: uncommitted edits show here
  if cmp -s "$NEW" "$REPO/paper8/derived/claims.tex"; then say "  (the working tree's claims.tex is identical too)"
  else say "  (note: the working tree's claims.tex differs from this rebuild; it is not part of commit $SHA)"; fi
fi

T_FIGURES=skipped
if [ "$FIGURES" = 1 ]; then
  say "4. build the figures in the copy (matplotlib $("$PY" -c 'import matplotlib; print(matplotlib.__version__)'))"
  t=$(now)
  ( cd "$WORK/repo/paper8" && env "${BUILD_ENV[@]}" nice -n 10 "$PY" -B scripts/make_figures.py \
      && env "${BUILD_ENV[@]}" nice -n 10 "$PY" -B scripts/make_figures.py --dark ) > "$WORK/logs/figures.log" 2>&1
  rc=$?
  T_FIGURES=$(secs "$t")
  [ "$rc" = 0 ] || problem "make_figures.py failed (logs/figures.log)"
  for f in "${OUTPUTS[@]:1}"; do
    if [ ! -f "$WORK/repo/$f" ]; then problem "$f was not rebuilt"; continue; fi
    verdict=$("${HELPER[@]}" pdfcmp "$WORK/repo/$f" "$WORK/committed/$f")
    if [ $? = 0 ]; then say "  ${f#paper8/figures/}: $verdict"; else problem "$f: $verdict"; fi
  done
  while IFS= read -r f; do   # figures the commit does not have
    [ -f "$WORK/committed/$f" ] || say "  (note: $f is new in this build; the commit has no copy to compare)"
  done < <(cd "$WORK/repo" && find paper8/figures -name '*.pdf' | sort)
  say "  figures built in ${T_FIGURES} s"
fi

# ------------------------------------------------------------------------------------------------ result
T_TOTAL=$(secs "$T_ALL")
say "times (s): extract ${T_EXTRACT}, items ${T_ITEMS}, claims ${T_CLAIMS}, figures ${T_FIGURES}, total ${T_TOTAL}"
if [ "${#PROBLEMS[@]}" = 0 ]; then
  say "RESULT: PASS - Paper 8's claims$([ "$FIGURES" = 1 ] && echo " and figures") rebuild byte for byte from commit $SHA"
  say "work folder (left in place): $WORK"
  echo "PASS $SHA" > "$WORK/result.txt"
  exit 0
fi
say "RESULT: FAIL - ${#PROBLEMS[@]} problem(s):"
for p in "${PROBLEMS[@]}"; do say "  - $p"; done
say "work folder (left in place): $WORK"
{ echo "FAIL $SHA"; printf '%s\n' "${PROBLEMS[@]}"; } > "$WORK/result.txt"
exit 1
