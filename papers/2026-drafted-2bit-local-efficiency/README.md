# Self-Drafting Qwen3.8-27B at Two Bits: Speed, Energy, and Memory on a 12 GB Laptop GPU — reproducibility artifact

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This is the curated evidence bundle for “Self-Drafting Qwen3.8-27B at Two Bits: Speed, Energy, and Memory on a 12 GB Laptop GPU” (2026).

The report measures self-drafting on one 12 GB RTX 4080 Laptop GPU at 80 W.
EXL3 completion throughput rises from about 29 to 48.7–50.4 tokens/s on
code/math and 35.7 on prose (E7). Sustained GPU-board efficiency is 0.559 for
EXL3 and 0.577 tokens/J for GGUF (E25); one run per lane does not establish
an energy ranking. Reducing four allocated slots to one saves 2,176 MiB
(E28), and denser checkpoints reduce one edited-prompt latency from 10.1 to
5.0 s (E43). Drafting shows no measured accuracy decrease on 1,000 paired
greedy items (E47), not equivalence. This is a frozen deployment benchmark
with configuration changes, not a new decoding method or current optimum.

## Release identity

- Version: `1.0.0`
- GitHub tag: `drafted-2bit-local-efficiency-v1.0.0`
- Tagged artifact path: <https://github.com/matthematics1137/research-artifacts/tree/drafted-2bit-local-efficiency-v1.0.0/papers/2026-drafted-2bit-local-efficiency>
- Zenodo version DOI: [10.5281/zenodo.22847652](https://doi.org/10.5281/zenodo.22847652)
- Evidence freeze: laboratory commit `9a62b2da50ef8198a3a12dc3837358444cb4d710`

The version DOI identifies the frozen reproducibility bundle; it is not a
separate manuscript DOI. Public availability is established by the released
Zenodo record and matching GitHub tag, not by the presence of a DOI here.

## What the bundle contains

- `paper/`: canonical light PDF and sanitized TeX source.
- `data/rows/`: the campaign's measurement rows (timings, throughput, GPU power
  and clock samples, VRAM, acceptance quantities, profiles, per-comparison
  summaries), with model output text, prompts, and local paths removed.
- `data/items/EFF-<label>/<suite>.jsonl`: per-item evaluation outcomes for every
  campaign label: item ID, correctness, completion tokens, wall time, truncation,
  the SHA-256 of the reasoning-plus-answer text (`content_sha256`, used for the
  byte-identity tests), and for GSM8K/MMLU the SHA-256 of the parsed final
  answer. Benchmark questions, answers, and model responses are not included.
- `data/traces/`: numeric statistics of the E40 and H38c temperature-0 traces
  (token counts, transition counts, log-probability gaps, first-divergence
  positions) recomputed from the private traces; no token strings.
- `data/logs/vram_log_extracts.json`, `data/hash_audit_2026-09-06.json`,
  `data/lab_quant_hashes.json`, `data/dataset_item_order.json`: the VRAM log
  lines, the artifact hash audit, the shard hashes of the lab-made quantizations,
  and the item order of the evaluation sets.
- `data/EXPORT_MANIFEST.json`: SHA-256 of every exported file, of every private
  input (each equal to the file at the evidence-freeze commit), and the list of
  transformations.
- `replication/`: the serving configurations, the lab patches, the measurement
  and hardening tools, and the evaluation harness, with machine-local paths
  replaced by placeholders (`/path/to/lab-repo`, `/path/to/models`,
  `/path/to/scratch`). Two further edits change no code: the harness's scratch
  directory defaults to the system temporary directory, and two comment lines of
  the llama.cpp patch were adjusted (one context line trimmed, one comment
  reworded; the patch still applies to v0.4.0). `data/EXPORT_MANIFEST.json`
  lists the source hash of every file.
- `environment/`: machine, engine, and model-file identities.
- `check_claims.py`, `CLAIM_EVIDENCE_MAP.md`, `DESIGN_RECORD.md`, `FULL_RERUN.md`.

## Five-minute analysis-only check

Python 3.11 or newer; no model, GPU, network, or package installation.

```bash
python3 check_claims.py
python3 replication/evals/run_eval.py --selftest
```

The first command verifies every exported file against the export manifest,
recomputes the abstract, conclusion, Table 1, Table 2, and hardening-block
numbers from the rows, and must end with `ALL PAPER 7 CLAIM CHECKS PASSED`. It
also prints the documented discrepancies: places where the paper prints a
figure that the rows reproduce only to a neighbouring rounding or definition
(see `CLAIM_EVIDENCE_MAP.md`). The second command runs the evaluation harness's
offline scorer and sandbox self-test and must end with
`SELFTEST PASSED (all checks)`.

## Where the paper's repository paths are

The report names paths in the private laboratory repository. In this bundle:

| Paper path | Artifact path |
|---|---|
| `serving/*` | `replication/serving/*` |
| `testsuite/results/efficiency/tools/<tool>` | `replication/tools/<tool>`; patches in `replication/patches/` |
| `research/patches/llamacpp-v0.4.0-qwen35-mtp-shared-embd.patch` | `replication/patches/` |
| `testsuite/evals/prepare_hardening.py` | `replication/evals/prepare_hardening.py` |
| `testsuite/results/efficiency/rows/<file>` | `data/rows/<file>` (sanitized); `traces_*` as statistics in `data/traces/` |
| `testsuite/evals/results/EFF-<label>/<suite>.jsonl` | `data/items/EFF-<label>/<suite>.jsonl` (sanitized) |
| `testsuite/results/efficiency/hash_audit_2026-09-06.json` | `data/hash_audit_2026-09-06.json` |
| `paper7/VERIFICATION.md` (claim ledger) | `CLAIM_EVIDENCE_MAP.md` |

## Full inference replication

See `FULL_RERUN.md`. It pins the model files, both engines and their builds, the
lab patches, and the serving configurations, and it describes the measurement
tools and the hardening-block drivers included here (the per-block launch
scripts of E1–E45 are not included; their protocols are in the paper and the
tools' headers). Exact historical timings are not expected: the laptop was loaded,
temperature-1 runs are stochastic, and even temperature-0 decoding on
exllamav3 1.4.7 is not bit-reproducible run to run (E49, E50).

## Result boundary

- One loaded laptop, one cap, named artifacts and pinned engines.
- Small temperature-1 suites have uncertain differences; one repeated baseline
  is not a general noise threshold. Larger paired greedy tests do not prove equivalence.
- The higher-bit comparison covers 200 GSM8K items, not general quality parity.
- Most EXL3 performance rows use mixed CUDA libraries; small clean controls do
  not replace a complete performance rerun.
- GPU-board energy excludes the host; newer releases and artifacts are untested.

## Rights, citation, and security

Authored code is MIT-licensed. The paper, original documentation, and
measurement records are offered under CC BY 4.0 to the extent the author holds
the relevant rights. The patches carry context lines of exllamav3 and llama.cpp
(both MIT; notices in `LICENSES/`). Model weights, engine source trees, and
benchmark text are not redistributed; `THIRD_PARTY_NOTICES.md` lists their
terms. Use `CITATION.cff` for the report and the versioned artifact.

The evaluation harness executes model-generated code for HumanEval+. Run it only
in an isolated container or VM with no secrets and no network access.

## AI assistance

Claude Code (Anthropic) operated experiments under the author's standing rules and pre-registrations; wrote engine patches, measurement tools and analysis scripts; drafted the experiment record, manuscript, claim ledger and bibliography; and prepared the artifact exporter and claim checker. Codex (OpenAI) reviewed the evidence and prose, checked primary-source literature and upstream status, revised the manuscript and companions, and checked the reading builds and artifact instructions. Matthew Schwartz directed the research and is responsible for its methods, results, interpretation, claims, citations, rights, and released artifacts.
