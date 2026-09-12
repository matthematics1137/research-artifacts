---
pretty_name: "Self-Drafting Qwen3.8-27B at Two Bits: Speed, Energy, and Memory on a 12 GB Laptop GPU — frozen measurements"
license: other
license_name: "Mixed CC-BY-4.0, MIT, and third-party terms"
license_link: "https://huggingface.co/datasets/mv1137/p2026-007-qwen38-27b-2bit-drafting-results/blob/main/LICENSE"
task_categories:
  - text-generation
language:
  - en
tags:
  - reproducibility
  - llm-inference
  - speculative-decoding
  - quantization
  - energy-efficiency
  - consumer-hardware
configs:
  - config_name: "e47-gsm8k500-drafted"
    data_files:
      - split: train
        path: "data/e47-gsm8k500-drafted.jsonl"
  - config_name: "e47-gsm8k500-undrafted"
    data_files:
      - split: train
        path: "data/e47-gsm8k500-undrafted.jsonl"
  - config_name: "e47-mmlu500-drafted"
    data_files:
      - split: train
        path: "data/e47-mmlu500-drafted.jsonl"
  - config_name: "e47-mmlu500-undrafted"
    data_files:
      - split: train
        path: "data/e47-mmlu500-undrafted.jsonl"
  - config_name: "e48-gsm8k-q4kxl"
    data_files:
      - split: train
        path: "data/e48-gsm8k-q4kxl.jsonl"
---

# Self-Drafting Qwen3.8-27B at Two Bits: Speed, Energy, and Memory on a 12 GB Laptop GPU — result index

<!-- publication-authors:v1 -->
- Matthew Schwartz — [ORCID 0009-0009-4171-7247](https://orcid.org/0009-0009-4171-7247)
<!-- /publication-authors:v1 -->

This dataset accompanies “Self-Drafting Qwen3.8-27B at Two Bits: Speed, Energy, and Memory on a 12 GB Laptop GPU.” The immutable tagged report and full
artifact use the following versioned path:

<https://github.com/matthematics1137/research-artifacts/tree/drafted-2bit-local-efficiency-v1.0.0/papers/2026-drafted-2bit-local-efficiency>

The version-1.0.0 artifact is [10.5281/zenodo.22847652](https://doi.org/10.5281/zenodo.22847652). The Hub copy is
a discovery mirror; the versioned Zenodo archive is the artifact of record.

## Data

| Path | Description | Rights/provenance |
|---|---|---|
| `data/items/EFF-<label>/<suite>.jsonl` | per-item outcomes of every campaign label: item ID, correctness, completion tokens, wall time, truncation, SHA-256 of the output text and of the parsed final answer | original measurements, CC BY 4.0; no benchmark or response text |
| `data/rows/` | throughput, energy, GPU telemetry, VRAM, acceptance, profile, and per-comparison summary rows | original measurements, CC BY 4.0 |
| `data/traces/` | numeric statistics of the temperature-0 traces (E40, H38c) | derived measurements, CC BY 4.0 |
| `data/logs/`, `data/hash_audit_2026-09-06.json`, `data/lab_quant_hashes.json`, `data/dataset_item_order.json` | VRAM log lines, artifact hash audit, lab-quant shard hashes, evaluation item order | original records, CC BY 4.0 |
| `environment/` | machine, engines, and exact upstream model-file identities | original documentation; weights are not included |

The five viewer configurations select copies of the hardening block's
item-level files at the top of `data/` (E47: drafted and undrafted on GSM8K-500
and MMLU-500, from `data/items/EFF-h38a_*`; E48: the 4-bit GGUF on the first
200 GSM8K-500 items, from `data/items/EFF-h38b_q4kxl_mtp/`). All other files are
supporting data. Prompts,
benchmark text, model responses, private paths, model weights, and engine source
are excluded.

## Question and headline result

The report measures self-drafting on one 12 GB RTX 4080 Laptop GPU at 80 W.
EXL3 completion throughput rises from about 29 to 48.7–50.4 tokens/s on
code/math and 35.7 on prose (E7). Sustained GPU-board efficiency is 0.559 for
EXL3 and 0.577 tokens/J for GGUF (E25); one run per lane does not establish
an energy ranking. Reducing four allocated slots to one saves 2,176 MiB
(E28), and denser checkpoints reduce one edited-prompt latency from 10.1 to
5.0 s (E43). Drafting shows no measured accuracy decrease on 1,000 paired
greedy items (E47), not equivalence. This is a frozen deployment benchmark
with configuration changes, not a new decoding method or current optimum.

## Verify

From this dataset or the full artifact:

```bash
python3 check_claims.py
```

It verifies every data file against `data/EXPORT_MANIFEST.json` and recomputes
the abstract, conclusion, Table 1, Table 2, and hardening-block numbers.
`FULL_RERUN.md` in the tagged artifact has the full inference instructions.

## Upstream artifacts

- `turboderp/Qwen3.8-27B-exl3`, branch `SC_2.00bpw_H3` (revision
  `d7e02d63dddd4e047a8d397acb22aeb2b0bedcec`), Apache-2.0.
- `unsloth/Qwen3.8-27B-GGUF` (revision
  `4ca720788d1e01f1bff70c033e0d0028fd02e502`), Apache-2.0.
- `z-lab/Qwen3.8-27B-DFlash2-GGUF`, Apache-2.0 (revision `2d9571f8ce46e151f61c6499c99dee6079e1d610`).

Every filename, byte count, and SHA-256 is in `environment/model_artifacts.json`.
This dataset claims no ownership of the weights.

## Limitations

- One model, one laptop GPU at 80 W, two engines; a loaded machine.
- Temperature-1 suites of 20–200 items, single runs; larger greedy comparisons
  remain finite-sample evidence, not equivalence tests.
- 2-bit against 4-bit is claimed only on GSM8K at n = 200.
- EXL3-lane throughput was measured under a mixed CUDA library set (E7–E49).
- Energy is GPU-board energy only.

## Rights and citation

Measurement fields and documentation are CC BY 4.0; authored code is MIT.
Benchmark and model terms differ, so the dataset uses Hugging Face's `other`
label; consult `LICENSE` and `THIRD_PARTY_NOTICES.md`. Cite the report for
scientific claims and the version DOI for frozen evidence.

## AI assistance

Claude Code (Anthropic) operated experiments under the author's standing rules and pre-registrations; wrote engine patches, measurement tools and analysis scripts; drafted the experiment record, manuscript, claim ledger and bibliography; and prepared the artifact exporter and claim checker. Codex (OpenAI) reviewed the evidence and prose, checked primary-source literature and upstream status, revised the manuscript and companions, and checked the reading builds and artifact instructions. Matthew Schwartz directed the research and is responsible for its methods, results, interpretation, claims, citations, rights, and released artifacts.
