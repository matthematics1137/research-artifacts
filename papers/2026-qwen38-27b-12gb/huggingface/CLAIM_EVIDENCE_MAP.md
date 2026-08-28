# Claim-to-evidence map

| Paper result | Frozen source | Verification |
|---|---|---|
| GGUF fit boundary, tg128, pp512, VRAM, and MTP microbenchmarks | `testsuite/results/phase1.jsonl` | `check_claims.py` asserts the eight controlled tg128 points; the frozen file and Figure 1/2 generator expose the remaining records |
| Effective bpw and artifact footprint | `environment/model_artifacts.json` | `check_claims.py` asserts the record count, immutable identifiers, hashes, and EXL3 aggregate footprint; the figure generator prints all eight effective-bpw values |
| Easy-suite aggregate scores | Qwen-only `testsuite/evals/results/summary.jsonl`; retained per-item rows except the disclosed Q4 gap | frozen row/file inventory; regenerate aggregates from retained rows where available |
| Original versus alternate MATH scores and format upgrades | `testsuite/evals/results/*/math25.jsonl`, fixed MATH subset, `rescore_math25.py` | `python3 check_claims.py`; `python3 testsuite/evals/rescore_math25.py` |
| HumanEval+ scores | retained `humaneval_plus.jsonl` files | `python3 check_claims.py` |
| Matched-footprint EXL3/GGUF generation rates | sanitized 45-request timing extracts in `testsuite/results/server_EXL3-2.0.log` and `server_bart_hard.log` | `python3 check_claims.py` |
| Correct responses per minute | per-item correctness and `wall_s` fields | `python3 check_claims.py`; Figure 4 generator |
| MMLU and needle validation | retained MMLU/needle rows and Qwen-only summary | `python3 check_claims.py` |

`paper/scripts/make_figures.py` regenerates the four figures and prints effective
bpw, MATH rescoring, Wilson intervals, matched-workload rates, and bandwidth-fit
constants. `check_claims.py` is a fail-closed check for the named headline
values and paired tests; it is not a line-by-line reimplementation of every
table. The release inventory and `SHA256SUMS` freeze every other source row.

The release contains response excerpts, not full transcripts. It contains no
Q4 per-item GSM8K/HumanEval rows because those files were not retained.
