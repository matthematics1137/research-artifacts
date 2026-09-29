# Public design and audit record

This report is a retrospective, hardware-specific benchmark. It was not
preregistered. The purpose of this record is to preserve what was planned,
what was retained, what changed during audit, and what the final evidence can
support.

## Chronology

- The Flash campaign tested unmodified llama.cpp mmap generation, a small placement
  sweep, prefill probes, a later quiet-gated repeat, and small task samples.
- The gpt-oss campaign added a second near-memory-boundary artifact using the
  same principal placement and generation prompt, plus small task samples.
- A later evidence audit parsed the frozen GGUF headers, recovered exact model
  revisions and file identities, reconstructed row-level score provenance, and
  separated decode timing from physical-device I/O scope.

## Audit corrections retained in the paper

- Loaded-versus-quiet observations are no longer described as a causal cache
  effect. Groups were sequential, not randomized, and they retained different
  memory fields.
- `/proc/diskstats` deltas are labeled whole-invocation, physical-device-global
  reads. They are not decode-only bytes, expert misses, or bytes per token.
- Arithmetic no-reuse expert-traffic values are tensor-inventory
  counterfactuals, not measured I/O.
- Original and post-hoc lenient MATH scores remain separate.
- The Flash HumanEval+ aggregate is omitted because the saved file contains
  only 17 of 20 item rows and cannot substantiate the ledger's 17/20 score.
- The result is framed as a reproducible public datapoint for two exact
  deployments, not a new inference method, state-of-the-art result, custom
  NVMe-streaming system, or model-quality comparison.

## Final evidentiary boundary

The retained evidence supports successful generation and observed rates for
two exact artifact–engine deployments on one machine. It does not support a
page-cache hit rate, decode bandwidth, a causal host-state mechanism, quality
preservation, or generalization to other models, storage devices, kernels, or
engine revisions.
