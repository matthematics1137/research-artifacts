# P4R1 claim-to-evidence map

Run `python3 check_claims.py` from a staged artifact, or run the same script in
this source directory after `export_public_data.py`. The checker reads only the
privacy-sanitized data and environment records.

| Manuscript result | Public evidence | Deterministic check |
|---|---|---|
| All six cells use the same 100 GSM8K item IDs | `data/primary_gsm8k.jsonl` | requires 600 rows, six complete 100-ID maps, and the seed-42 item set |
| Mixtral Q4/IQ2/IQ1 scores are 68/46/26 | item-level `correct` fields | recomputes all three totals and checks `data/claims.json` |
| Qwen Q4/IQ2/IQ1 scores are 91/97/96 | item-level `correct` fields | recomputes all three totals and checks `data/claims.json` |
| Mixtral Q4-minus-IQ2 paired table is 38 both correct, 30 Q4-only, 8 IQ2-only, 24 both wrong | paired item outcomes | reconstructs the exact table |
| Mixtral Q4-minus-IQ1 paired table is 23/45/3/29 in the same orientation | paired item outcomes | reconstructs the exact table |
| Qwen Q4-minus-IQ2 paired table is 91/0/6/3 | paired item outcomes | reconstructs the exact table |
| Qwen Q4-minus-IQ1 paired table is 91/0/5/4 | paired item outcomes | reconstructs the exact table |
| Signed Q4-minus-low differences are +22, +42, -6, and -5 percentage points | reconstructed cell totals and paired tables | checks sign and magnitude against the frozen comparison records |
| Exact McNemar p-values, Newcombe method-10 paired intervals, and four-test Holm adjustment | reconstructed paired tables | recomputes all statistics and checks `data/claims.json` |
| Mixtral's two paired differences survive Holm correction; Qwen's do not | Holm-adjusted values in `data/claims.json` | checks adjusted p-values 0.00141596, 5.250e-10, 0.0625, and 0.0625 |
| Qwen non-rejection does not establish improvement or equivalence | analysis contract and absence of an equivalence margin | enforced as an interpretation limit in the manuscript and metadata |
| Truncations are 1/0/0 for Mixtral IQ1/IQ2/Q4 and 1/1/1 for Qwen IQ1/IQ2/Q4 | public `truncated` fields | recomputes all six counts without excluding rows |
| No P4R1 API error or chat-template fallback occurred | public Boolean telemetry fields | requires zero in all 600 rows |
| Per-item inference seed is `4,200,000 + item suffix`, shared across artifacts | public `sampler_seed` and item IDs | recomputes every row's seed |
| All cells used context 4,096 | `data/claims.json` and `environment/software.json` | checks the common context contract |
| GPU-offloaded-layer counters were Mixtral 12/33, 26/33, 29/33 for Q4/IQ2/IQ1 and Qwen 49/49 for all tiers; counters include partial offload | P4R1 cell receipts summarized in claims and environment metadata | checks the exact counters, without inferring complete GPU weight residency |
| Qwen Q4 and IQ2 automatic fitting reported 25 and two overflowing layers; IQ1 required no fit adjustment; equal counters do not mean equal tensor placement | `environment/placement_readback_excerpt.json`: exact startup lines with source-log and cell-receipt hashes; pinned engine-source semantics | excerpt digest is pinned by the public checker; local preparation also verifies each line against the immutable source log and promoted receipt |
| Separate Qwen CPU-mapped/CUDA model buffers were 17447.91/9370.43, 9613.59/9464.11, and 166.92/9064.43 MiB for Q4/IQ2/IQ1 | the same source-bound startup excerpts | buffer sizes are not added or described as disjoint physical resident memory |
| Exact file bytes and effective whole-file bpw for all six artifacts | `data/claims.json` and `environment/model_artifacts.json` | recomputes bpw and cross-checks exact identities |
| IQ2/IQ1 files are 56%/62% smaller than Mixtral Q4 and 44%/48% smaller than Qwen Q4 | exact byte counts | recomputes rounded decimal reductions |
| Claims and rows come from the promoted P4R1 evidence | `data/EXPORT_MANIFEST.json` | checks the claims, rows, promotion-receipt, and public-checker hashes |

P4R1 is corrective and exploratory rather than independent: the item IDs,
historical outcomes, and motivating hypothesis were already known. The public
rows omit prompts, expected answers, and response prose because the paired
analysis requires only item identity, scored outcome, and the retained protocol
and provenance fields.
