# Version 1.0.1 correction — 2026-09-04

This version corrects reporting in the original [1.0.0 artifact](https://doi.org/10.5281/zenodo.22166977).
No inference was rerun. Original measurement and result rows, task scores,
statistical tests, and the central deployment conclusions are unchanged.
The published 1.0.0 files remain intact.

1. **Request errors and retries:** the MATH table now separates two Q4 request
   time-outs from output-length truncations. Both errors were already scored
   wrong. Their 2404.36 seconds remain in the 7313.90-second goodput denominator
   (0.1887 correct/min). The client could automatically retry once; one retained
   response does not mean one attempted stochastic request. Attempt counts
   were not recorded and are not reconstructed as new observations.
2. **GSM8K selection:** the evaluated 50 items were the first half of a sorted
   seed-42 sample of 100, not a direct random sample of 50. Their source indices
   span 13–563. This is disclosed without changing the sample or its scores.
3. **MTP timing evidence:** the contemporaneous server log, already present in
   the original evidence-freeze commit, is now included with model paths
   sanitized. Its 45 request completion-token counts match the MATH-25 and
   HumanEval+-20 rows in sequence. That retrospective alignment is not the
   same as run-time benchmark-ID instrumentation. The MATH per-request decode
   range is 41.09–50.84 tok/s; aggregate decode is 45.35 tok/s. Aggregate logged
   HTTP-response rate is 44.96 tok/s, a distinct metric. The narrower old
   45–50 prose range and 48–50 figure annotation are replaced. There is still
   no identical-prompt draft-off comparison establishing a causal MTP effect.
4. **Server/tg128 caption:** the matching draft-off server measurements differ
   from tg128 by at most 0.73 tok/s, not 0.4.
5. **Effective-bpw figure labels:** labels now derive from checkpoint bytes and
   parameter count; Q4 is 5.22 and bartowski IQ2_S is 3.06 effective bpw.
6. **Needle context sizes:** 8K/16K/32K are nominal estimates based on word count,
   not actual model-tokenizer measurements.
7. **Offload fit:** the two-bandwidth fit is explicitly descriptive, assessed
   on its eight fitting configurations. It does not isolate CPU dequantization
   costs or provide held-out predictive validation.

The added `correction_metrics.py` checks log alignment, MTP rates, error IDs,
retry-time accounting, and GSM selection. `check_claims.py` also rejects missing,
duplicate, unexpected, or incorrectly typed MATH result IDs/booleans and prints
the paired McNemar result as the primary MATH comparison. The public TeX no
longer has a dormant dependency on a private dark-mode style. Hugging Face
retains explicit viewer configurations for its summary and throughput tables.

## What did not change

The 1.64× matched-footprint MATH generation-rate comparison, fully resident
versus partially offloaded deployment conclusion, paired MATH result, and MMLU
non-resolution all reproduce. Quantizer and engine remain confounded; the
study is still a one-machine, small-sample deployment characterization.

Run `python3 check_claims.py` from this artifact root. Additional evidence
provenance is in `environment/correction-provenance.json`; complete release
hashes are supplied by the generated artifact inventory.
