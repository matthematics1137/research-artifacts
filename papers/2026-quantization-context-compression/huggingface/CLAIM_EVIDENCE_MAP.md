# Claim–evidence map: Paper 2 (P2R1)

Every quantitative statement in the report is a macro in
`paper/derived/claims.tex`. `python3 check_claims.py` recomputes each one from
`data/claims.json` and requires an exact match for every name and value.

"Recomputed" below means the checker derives the value from counts or
histograms. It does not merely read the value back.

| Report claim | Location in report | Evidence in `data/claims.json` | Check |
|---|---|---|---|
| 24 cells, 2,844 scored requests, 0 evaluation and 0 question-generation request errors | Abstract; §3.5 | `headline.n_cells`, `n_scored_requests`, `evaluation_request_errors`, `question_generation_request_errors`; every cell's `request_outcomes` | Totals recomputed from the cells |
| Frozen sample of 117 private and 120 WildChat items; quotas 37/37/6/37 and 30/30/30/30; 107 and 120 source-conversation clusters | Abstract; §3.3; Table 2 | `grid.<corpus>.n_items`, `item_strata_counts`, `n_source_clusters` | Strata sums; cluster counts recomputed from each histogram |
| Exact accuracy and 95% Wilson intervals for all 24 cells | §4.1; Table 4; Figure 1 | `grid.<corpus>.cells.<tier>-<cond>.{k_exact,n,acc,wilson}` | Recomputed |
| Paired cost relative to O, with Newcombe method-10 intervals and exact McNemar tests | §4.3; Table 6 | `cells.*.paired_vs_O.{O_only,cond_only,delta,ci,phi,p_mcnemar}` | Recomputed from the paired 2×2 table |
| LLMLingua-2 costs 15–21 points; the naive rungs cost 47–62 points; every LLMLingua-2 cost is below every naive cost | Abstract; §4.3; Conclusion | `headline.l2_delta_pp_*`, `naive_delta_pp_*`, `qualitative_assertions.llmlingua_cost_below_every_naive_cost` | Recomputed |
| 12 interaction estimates from −5.0 to 6.0 points; pointwise interval envelope −11.6 to 12.9, not a detection threshold or simultaneous bound | Abstract; §4.2; Table 5 | `interaction.<corpus>.<cond>-<tier>.{delta_low_minus_q4_reference,ci_cluster_95}`; `headline.interaction_*` | Recomputed from each cluster histogram: estimate, CR2 variance, Satterthwaite df, t critical value, interval |
| CR2/Satterthwaite p from 0.06 to 1.00; Holm rejects 0 of 12; branch `no-interaction-detected` | Abstract; §4.2; Table 5; Conclusion | `interaction.*.{p_cluster_cr2_satterthwaite,familywise}`; `headline.interaction_result_branch` | Recomputed: t distribution, Holm step-down over all 12 tests, and the branch |
| Every Satterthwaite reference df is at least 5 (observed at least 97) | §3.6 | `interaction.*.cluster_reference_df_satterthwaite` | Recomputed; the checker fails below 5 |
| Secondary sign-flip sensitivity: no raw p below 0.05 | §3.6; Table 5 caption | `interaction.*.p_sign_symmetry_sensitivity`; `headline.interaction_unadjusted_signflip_p_below_005` | Recomputed by exhaustive enumeration over cluster sums |
| At every deployment O > L2 > both naive rungs in accuracy; O yields the most correct answers per serving minute; every compressed condition has a longer mean request time | Abstract; §4.4; §4.6; Conclusion | `cells.*.{acc,acc_per_min,mean_wall_s}` | Checked for each of the 6 deployment–corpus pairs |
| Character retention 51–65%, matched L2/E requested rates | §3.4; Table 3 | `corpora.<corpus>.char_retention_vs_O` | Macro match |
| Answer survival: 198 deleted item–condition pairs, 594 tier evaluations, 10 exact-correct; survivor accuracy ranges | §4.5; Figure 4 | `survival.*`; `headline.destroyed_*`, `surv_acc_*_range` | Macro match |
| Artifact sizes, effective bpw, engines, placements, hardware | §3.2; Table 1 | `artifacts.*`; `hardware`; `software`; also `environment/*.json` | Macro match; environment files are exact projections |
| The 33/66 and 65/65 loader placement readbacks | §3.2; Table 1 | Private stage logs bound by the promotion record | Not publicly re-derivable. Stated in the frozen protocol and enforced by the shipped `replication/testsuite/paper2/campaign_guard.py` |
| Forensic v3/v4 byte-identity with v5 (artifact provenance, not a manuscript result) | `DESIGN_RECORD.md` | Private evidence only | Not publicly re-derivable. Retained as an operator observation in the design record |

Values in the report are rounded as the macros print them. The checker
compares unrounded values with a tolerance of 1e-9, or 1e-7 for quantities
computed by bisection or by the incomplete-beta series.
