# Empirical Research Loop Contract

**Version:** 1.1.0

**Effective date:** 2026-08-27

**Status:** Required default for new empirical projects

**Review cadence:** Quarterly, before a new paper series, and after any replication, review, or release failure

## Purpose

This contract turns the path used in the 12 GB quantized-inference study into a repeatable research method. It is meant to preserve what worked: begin with a practical constraint, measure the real system rather than its labels, explore broadly, freeze a defensible comparison, test likely explanations, and limit the final claim to what the evidence supports.

The contract does not authorize changes to an active experiment. It governs how future projects are designed and recorded. Existing simulations, raw results, logs, and research state remain untouched unless a separate project decision explicitly places them in scope.

The central integrity rule is simple:

> Exploration may discover the question and the pattern. Confirmation must test a frozen version of them. The record must never pretend that an explanation chosen after seeing the data was specified before the data existed.

## Scope and authority

This contract applies whenever a project will make an empirical claim, whether the output is a paper, technical report, dataset, benchmark, or platform evaluation. A venue may add requirements, but it may not weaken the evidence, provenance, or disclosure requirements here.

Publication begins only after the research gates below pass. Writing and release are governed separately by:

- [`../publication/PAPER_CONTRACT.md`](../publication/PAPER_CONTRACT.md)
- [`../publication/RELEASE_CONTRACT.md`](../publication/RELEASE_CONTRACT.md)

If those files are revised, the version used for a project must be recorded in that project's release manifest.

## Required project record

Every project keeps the following records. Their format may change, but their functions may not.

1. **Research charter:** the decision or question, why it matters, scope, resource ceiling, and stop condition.
2. **Design-space map:** candidate systems, interventions, baselines, constraints, and known confounders.
3. **Run registry:** immutable identifiers for inputs, code, environment, hardware, seeds, settings, and outputs.
4. **Decision log:** what changed, when, why, and which evidence was visible at the time.
5. **Analysis contract:** frozen endpoints, scoring rules, exclusions, comparisons, uncertainty method, and sample plan for confirmation.
6. **Evidence ledger:** every prospective paper claim linked to the exact table, run, transcript, calculation, or source that supports it.
7. **Claim ledger:** each statement classified as observation, inference, hypothesis, or external fact.
8. **Release manifest:** the paper, artifact, licenses, identifiers, public links, and gate status.

Raw outputs are append-only. Corrections create a new derived artifact with its rule, reason, and parent recorded; they do not silently replace the original.

## The research loop

### Stage 0 — Frame the decision

State the practical decision the work should improve. Define the fixed resource budget and the output metric before choosing the winning system.

For the inference study, the useful question was not “which quantization is best?” It was “which artifact–engine pair returns correct answers fastest within a 12 GB VRAM budget?” That formulation made model quality, runtime, and memory residency part of one decision.

**Gate 0 passes when:**

- the decision and intended user are explicit;
- the primary constraint is measurable;
- the primary outcome is useful rather than merely convenient;
- a negative or null result would still be interpretable; and
- the project has a bounded stop condition.

### Stage 1 — Map the actual system

Enumerate the units that are truly deployed and compared. A label is not a system. In the inference study, the unit was an artifact–engine pair, not a nominal bit-rate label. File footprint, runtime format, cache, buffers, GPU residency, and fallback behavior all affected the result.

Measure rather than infer operational properties whenever possible. Distinguish nominal labels from actual bytes, configured memory from observed residency, and theoretical bandwidth from effective bandwidth.

**Gate 1 passes when:**

- the comparison unit is defined;
- actual resource use is measured;
- baselines and boundary cases are listed;
- incompatible paths are identified rather than averaged together; and
- likely confounders have a planned control, measurement, or limitation statement.

### Stage 2 — Validate the measurement path

Before a broad sweep, test that the harness measures what the paper will claim. Use small guarded runs to verify loading, prompts, stopping rules, timing boundaries, output capture, scorer behavior, failure recovery, and provenance. A guarded harness should fail loudly on missing or malformed outputs and should never convert a crash into a zero score without recording the failure separately.

The scorer is part of the experiment. Parser success, semantic correctness, execution success, and exact-match success are different outcomes and must not be conflated.

For experiments that call a model or service through a network endpoint, service health is not service identity. Before accepting any response, the harness must establish that the process it just started is still alive and that the endpoint reports the expected engine and model artifact. Each stage records its process identifier or process group, bound address and port, launch command, engine revision, artifact identifier and checksum, and identity-readback response. A stage must refuse an occupied fixed port; automatic port changes are permitted only when the actual assigned port is captured and passed explicitly to the client. Shutdown is scoped to the recorded process or process group and must be verified before another stage begins.

If endpoint identity was not recorded and cannot be reconstructed unambiguously from independent logs, the affected observations are invalid. Preserve them as quarantined provenance evidence; do not relabel, overwrite, or use them in a comparison.

**Gate 2 passes when:**

- a smoke test completes end to end;
- the client is proven to reach the process, engine, and artifact named by the run;
- expected failures are distinguishable from wrong answers;
- timing and memory measurement boundaries are documented;
- scorer tests cover representative edge cases; and
- every output can be traced back to its run configuration.

### Stage 3 — Explore the design space

Exploration is allowed to be adaptive. Inspect outputs, change prompts, add candidate configurations, fit rough models, investigate anomalies, and discover better questions. Record each material decision and the evidence that motivated it.

Exploratory results may support a descriptive report, but they cannot be called preregistered or confirmatory. Post-hoc scoring changes must be applied to preserved outputs and reported as post hoc.

**Gate 3 passes when:**

- the important operating regimes and failure boundaries are visible;
- the likely leading configurations and useful controls are known;
- anomalies have been investigated or explicitly retained;
- no hidden manual intervention is needed to reproduce the analysis; and
- the team can state a small set of testable hypotheses.

### Stage 4 — Freeze the confirmation contract

Turn the exploratory findings into a prospective test. Freeze the primary comparison, outcome, sample unit, scoring implementation, exclusions, seeds or sampling procedure, uncertainty calculation, stopping rule, and minimum effect worth discussing.

Use fresh runs, held-out tasks, new seeds, or an independent dataset whenever feasible. If no independent confirmation is possible, the result remains an exploratory systems characterization and the paper must say so.

**Gate 4 passes when:**

- the analysis contract is timestamped and versioned;
- primary and secondary outcomes are separated;
- the sample plan can distinguish the claimed effect at a useful resolution;
- the data used to choose the hypothesis are identified; and
- any overlap between exploration and confirmation has an explicit consequence for claim strength.

### Stage 5 — Run confirmation without adaptive drift

Execute the frozen plan. Operational fixes are permitted only when logged and when their effect on comparability is assessed. Do not drop a surprising result, change the endpoint, or rerun a condition merely because the answer is inconvenient.

If the contract must change, close the current confirmation attempt, preserve it, version the contract, and begin a new attempt. The old result remains in the record.

**Gate 5 passes when:**

- all planned conditions are accounted for;
- every service-backed stage has a valid identity receipt and verified shutdown record;
- deviations and failures are visible;
- primary analysis is reproducible from immutable outputs;
- uncertainty is reported at the level permitted by the design; and
- the result is described independently of whether it is positive.

### Stage 6 — Test mechanisms and alternatives

Separate the measured result from the explanation for it. Mechanistic models—such as weight traffic divided by effective bandwidth—are valuable when they predict observations or expose a constraint. They do not become causal proof solely because they fit the same data that suggested them.

Run cheap, high-information checks first: matched-footprint comparisons, fit-boundary cases, repeated timings, scorer audits on saved outputs, ablations, and alternative normalizations. Broader hardware, models, tasks, and reference precisions are follow-up work when they would materially change the claim.

**Gate 6 passes when:**

- at least one plausible alternative explanation has been tested or bounded;
- model fit is not presented as mechanism proof;
- sensitivity to scoring and exclusions is reported;
- deployment-specific findings are not assigned to an encoding method alone; and
- unresolved alternatives appear in limitations or future work.

### Stage 7 — Build and audit the claims

Write the claim ledger before polishing the paper. Every major sentence must be one of:

- **Observation:** directly measured in this study.
- **Inference:** a reasoned interpretation of observations.
- **Hypothesis:** a proposed explanation or future test.
- **External fact:** supported by a verified citation.

Each numerical observation points to its source. Each inference names the observations and assumptions it depends on. “State of the art” is permitted only against a current, defined comparison set using matched evaluation. A best configuration within one experiment is not automatically state of the art.

**Gate 7 passes when:**

- the central claim is supported at its stated scope;
- contradictory or null evidence is included;
- every number and external fact is traceable;
- the title and abstract do not outrun the evidence; and
- an independent reader can identify what was measured, inferred, and left open.

### Stage 8 — Reproduce, package, and hand off

Create a clean artifact from the immutable research record. The public artifact must not expose private corpora, credentials, unrelated experiments, or active research state. It should contain the smallest complete path to inspect the evidence and reproduce the reported analysis; a full rerun path is included when licensing and compute permit.

**Gate 8 passes when:**

- a clean checkout builds the paper and reproduces tables and figures;
- exact revisions, checksums, environment, and hardware are recorded;
- licenses and redistribution rights are audited;
- public links resolve;
- known irreproducible components are named; and
- the publication contracts pass.

### Stage 9 — Learn after release

Record review comments, replication attempts, corrections, downstream use, and failed assumptions. A correction is versioned and visible. Feed process failures into the next quarterly contract review, not into silent changes to the historical record.

## Exploration and confirmation rules

The following distinctions are mandatory:

| Activity | Exploratory use | Confirmatory use |
|---|---|---|
| Inspecting outputs while choosing hypotheses | Allowed and logged | Not allowed for the held-out test |
| Changing prompts, parsers, exclusions, or metrics | Allowed and versioned | Requires a new frozen contract |
| Selecting the best configuration | Generates a candidate | Requires fresh or held-out comparison to estimate selection-adjusted performance |
| Fitting a mechanism after seeing the curve | Produces a hypothesis | Requires prospective prediction, an ablation, or restrained language |
| Re-scoring saved outputs | Valid sensitivity analysis | Confirmatory only if the rule was frozen before those outputs were inspected |
| Reusing the same data | Valid for description | Must be disclosed; does not supply independent confirmation |

When a fresh confirmation set is too expensive, the correct response is a narrower claim, not a fictional separation.

## Why the inference-study path was logical

The reusable sequence was:

1. Fix the real hardware constraint.
2. Treat each deployable artifact–engine pair as a system.
3. Measure actual footprint and identify the residency boundary.
4. Evaluate both scored quality and generation speed.
5. Combine them into correct-response throughput for the practical decision.
6. Use matched-footprint cases to separate size from stack effects where possible.
7. Fit a simple bandwidth model to explain the throughput shape.
8. Audit scorer sensitivity on preserved outputs.
9. Report the best observed deployment while limiting claims to the tested hardware, engines, artifacts, tasks, and sample sizes.

That path supports an empirical systems characterization. It does not, by itself, establish a new quantization algorithm, a universal ranking, or an absolute model-quality ceiling. Those would require different experiments.

## Automation boundaries

Automation may scaffold records, schedule frozen runs, capture provenance, reproduce analyses, check claim links, and flag contract violations. It may not silently alter raw data, decide that a failed gate passed, broaden a claim, suppress an inconvenient result, or publish externally.

A human project owner must approve:

- the research charter;
- the exploration-to-confirmation freeze;
- any material deviation;
- the final claim ledger;
- licenses and private-data exclusions; and
- public release.

## Gate outcomes

Every gate ends in one of three recorded states:

- **PASS:** all requirements are satisfied.
- **PASS WITH SCOPE:** the work may proceed only with the recorded narrower claim or limitation.
- **STOP:** proceeding would make the result unreliable, unsafe, or misleading.

Automation can recommend a state; only the project owner can sign it. A paper can be worth releasing after a scoped pass. It cannot be made stronger by hiding why the scope was necessary.

## Contract maintenance

Every revision receives a semantic version and changelog entry. Patch versions clarify language, minor versions add compatible requirements, and major versions change gate behavior. A project records the version it began with and any later version it adopted.

Review this contract:

- every three months;
- before starting a multi-paper campaign;
- when a platform or venue changes an integrity requirement;
- after a material scoring, provenance, replication, or review failure; and
- when automation gains a new class of authority.

### Changelog

- **1.1.0 — 2026-08-30:** Added fail-closed service-identity, port-ownership, process-lifecycle, and run-quarantine requirements after a multi-engine grid accepted a healthy orphan service on the intended port.
- **1.0.0 — 2026-08-27:** Initial contract derived from the quantized-inference study workflow.
