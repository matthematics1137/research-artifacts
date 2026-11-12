# Claim–evidence map

The corrected named-arm attribution and generated timing checks are retained.
Run `python3 -B check_claims.py` from the artifact root to verify the inventory,
recompute all 1027 macros and every generated TeX fragment, check the quality
counts/pairs, replay the registered Bonsai numeric outcomes into its row and
sentence, and validate the two complete fleet speed/eight quality fixtures. Use `--figures` for exact light-figure replay. All paths below
are under `replication/`. Final independent review and author approval remain pending.

The focused manuscript leads with the stock-engine speed result. Its title's
72 tokens/s rounds the frozen 71.61 MTP rate; “nearly double” describes the
separate 1.88 undrafted ratio (47.53 tokens/s), not a twofold MTP gain.
Methods, Results and Performance analysis retain the speed, mechanism, bounded
quality, 72K memory reach and 64K tensor-core retrieval findings. Appendix A,
“Null results and limits,” retains the complete other-engine comparisons,
cache nulls, earlier 16K study, failed predictions and detailed limitations.
The original MMLU comparison used stock exllamav3; the patched-build quality
acceptance checks used GSM8K. Memory reach is distinct from tested retrieval.

| Paper evidence | Frozen inputs | Derivation and scope |
|---|---|---|
| Title, abstract, speed tables/figure | `testsuite/results/turing/ab/final2.jsonl`, `final2.md` | `paper8/scripts/build_claims.py`: `speed_final2`; pooled tokens/time per arm and round; `f2_per_prompt` gives exploratory splits; within-session only |
| Confirmation and competitor fit | `testsuite/results/turing/confirm/` session rows/results, fit rungs, DFlash2 logs | `confirmation`; each session checked separately; absent/partial-offload cells retained |
| Earlier November 12 refresh (NInferno comparison replaced) | `testsuite/results/turing/p8-refresh-76b5/`, `tools/turing/refresh/artifacts.json` | `refresh`; preserves dated numerical observations on old model hashes; reconstructed NInferno presence 1.5 is superseded by the replacement, not evidence of a causal penalty effect |
| November 14 replacement | `research/raw/2026-11-14-p8-rc-refresh-a72c/refresh14-numeric.json` | `refresh_fourteen`; 69 new macros and one table from independently reviewed pooled/round/task tables, whose hashes are retained. Nine of fourteen comparisons completed; only four NInferno pairs qualify current full-residency leads. Neutral penalties still leave top-k/order differences; code/prose favor the patched engine and math favors NInferno at both fills. This compact aggregate projection does not reproduce the private raw sampler/residency audit or re-score responses. |
| Cross-session variability | `testsuite/results/ideas/four-resources-formal/audit.json` | `audit`; sessions are not interchangeable replicates |
| Matrix time, int8/unfusing, verify and depth | `testsuite/results/turing/census/`, `attn/`, `ab/depth2.pooled.md`, `ab/final1.md`, `testsuite/results/four_resources/test2/`, `test3/` | `mechanism`, `census_groups`, `audit_numbers`; profiled attribution and unprofiled speed claims remain distinct |
| Registered one-row bench and Ada repeat | `testsuite/results/p9/replication/report.json`, `testsuite/results/ada/census/`, original registrations | `p9_bench`, `call_costs`; reported observations only; card/kernel confound retained |
| Quality and time/answer | `testsuite/results/turing/quality/compare_lcpp_fp32.json`, T4 comparisons, projected per-item `.jsonl` and `.full.jsonl` files | `quality_lanes` plus independent counts/paired discordance in `check_claims.py`; no benchmark text or responses redistributed; no adjusted significant gap is not equivalence |
| Bonsai descriptive counterpoint (Appendix A) | `testsuite/results/turing/bonsai/`, the five named archived EXL3 controls and `research/prereg/2026-11-12-p8-bonsai-lane.md` | `paper8/scripts/check_bonsai.py`, with unchanged registered arithmetic in `bonsai_stats.py`: all nine summaries, five exact paired tests/Holm and fixed-seed bootstrap intervals; regenerates row and sentence. All five adjusted accuracy differences are unresolved; no equivalence margin/test was specified. Request-only rates, normalization and capped answers retained. The separate 58.5/95.9 tok/s figures are vendor-reported October CLI runs, not a matched throughput arm or an isolated component claim. |
| Fleet deployment replication | `fixtures/fleet/card-{4070,4080-super}/runs/ns-20261112-p8-speed/journal.jsonl`, `research/raw/2026-11-12-fleet-lanes/` P8-only summaries, prediction verdicts and quality projections | `fleet_replication` checks 36 complete rotated speed requests/card, recomputes pooled tokens/server time and checks eight 500-item quality ID/count lanes against summaries and archived Turing anchor. Nineteen macros and the fleet table replay. Saved correctness is not response re-scoring; HTTP path, nominal non-forwarded seeds, separate sessions and missing actual-limit snapshots preclude a controlled cross-card claim. |
| Model sizes and hashes | `testsuite/results/turing/model_files.json`, September hash audit, November desktop rehash record | `model_files`, `weight_hashes`, `bytes_per_token`; full immutable upstream pins in `environment/model_artifacts.json` |
| Cache reach | `testsuite/results/turing/longctx/stage1/` | `stage1`; filled prompt peak at a registered ladder point, lower bound where no failure occurred |
| Cache accuracy, kernel check, 96K, non-replication | stage2/stage3/fd75d/fd75d-arm/replication journals, logs/configs/results; scoring projections in `testsuite/p8_longctx/work/items/` | `stage2`, `stage3`, `arm`, `replication_r1`, `replication_final`; registered scorer/statistics are imported unchanged except path/host normalization; 90% paired intervals, registered usability/equivalence bounds |
| Earlier 16K study | `paper6/derived/claims.tex`, `testsuite/results/paper6/analysis-2026-11-01/`, registration/design/amendment; root `p6-outcomes.json` | P8 copies the earlier frozen macros and includes per-item outcome projections; P8's default checker does not independently replay P6 analysis |
| Predictions | `research/prereg/` and the W7/W13 note | The table preserves the dated predictions and adds replacement coverage; failed predictions remain failed; registrations are historical records, not new run authority |
| External published comparisons | pinned upstream benchmark rows and bibliography sources; archived November 12 search record | Dated source observations, not matched lab measurements; no current blanket ranking inferred |

`EXPORT_MANIFEST.json` distinguishes original bytes from normalized/projected
exports and records each transformation. Re-running a normalized harness needs
a new preregistration/seal; old seals are preserved as historical evidence.
No raw lab source was edited to make this export verify. Three light figures
ship; the removed per-call figure is outside the manuscript and this release.

The compact long-context projections were produced from complete original item
files whose bytes matched the frozen manifests. Synthetic question/gold and
metadata are retained; chat haystacks are omitted. The inventory binds full-file
and projection hashes separately. No complete-prompt regeneration was rerun
during this package maintenance; the accepted RC full-item replay is separate
evidence, and the portable checker alone cannot replace it.
